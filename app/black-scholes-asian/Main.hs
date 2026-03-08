module Main (main) where

import qualified Data.Map.Strict as Map
import Molten (DeviceId(..), withContext)
import Molten.Examples.BlackScholesAsian
  ( BlackScholesAsianConfig(..)
  , BlackScholesAsianShadowSummary(..)
  , BlackScholesAsianStressSummary(..)
  , defaultBlackScholesAsianShadowConfig
  , defaultBlackScholesAsianStressConfig
  , runBlackScholesAsianExample
  , runBlackScholesAsianShadowCheck
  )
import Molten.Examples.Common (parseFlagMap)
import ROCm.HIP.Device (hipGetCurrentDeviceGcnArchName, hipGetCurrentDeviceName)
import System.Environment (getArgs)
import Text.Read (readMaybe)

main :: IO ()
main = do
  args <- getArgs
  case parseFlagMap args of
    Left message -> fail message
    Right flags -> do
      deviceName <- hipGetCurrentDeviceName
      archName <- hipGetCurrentDeviceGcnArchName
      let shadowConfig = applyBlackScholesFlags defaultBlackScholesAsianShadowConfig flags
          stressConfig = applyBlackScholesFlags defaultBlackScholesAsianStressConfig flags
      putStrLn "Molten example: black-scholes-asian"
      putStrLn ("Device: " <> deviceName <> " (" <> archName <> ")")
      putStrLn ("Shadow config: " <> show shadowConfig)
      putStrLn ("Stress config: " <> show stressConfig)
      withContext (DeviceId 0) $ \ctx -> do
        shadowSummary <- runBlackScholesAsianShadowCheck ctx shadowConfig
        putStrLn ("Shadow check: " <> if blackScholesAsianShadowPassed shadowSummary then "PASS" else "FAIL")
        putStrLn ("Shadow price abs err: " <> show (blackScholesAsianShadowPriceAbsError shadowSummary))
        putStrLn ("Shadow std-error abs err: " <> show (blackScholesAsianShadowStdErrorAbsError shadowSummary))
        if not (blackScholesAsianShadowPassed shadowSummary)
          then fail "black-scholes-asian shadow check failed"
          else pure ()
        stressSummary <- runBlackScholesAsianExample ctx stressConfig
        putStrLn ("Stress run: " <> if blackScholesAsianPassed stressSummary then "PASS" else "FAIL")
        putStrLn ("Price estimate: " <> show (blackScholesAsianPrice stressSummary))
        putStrLn ("Std error: " <> show (blackScholesAsianStdError stressSummary))
        putStrLn ("95% CI: [" <> show (blackScholesAsianConfidenceLow stressSummary) <> ", " <> show (blackScholesAsianConfidenceHigh stressSummary) <> "]")
        putStrLn ("Repeated abs diff: " <> show (blackScholesAsianRepeatedAbsDiff stressSummary))
        putStrLn ("GPU run time (s): " <> show (blackScholesAsianMeasuredSeconds stressSummary))
        if not (blackScholesAsianPassed stressSummary)
          then fail "black-scholes-asian stress check failed"
          else pure ()

applyBlackScholesFlags :: BlackScholesAsianConfig -> Map.Map String String -> BlackScholesAsianConfig
applyBlackScholesFlags config flags =
  config
    { blackScholesAsianPaths = readFlag "paths" (blackScholesAsianPaths config)
    , blackScholesAsianSteps = readFlag "steps" (blackScholesAsianSteps config)
    , blackScholesAsianSpot = readFlag "spot" (blackScholesAsianSpot config)
    , blackScholesAsianStrike = readFlag "strike" (blackScholesAsianStrike config)
    , blackScholesAsianRate = readFlag "rate" (blackScholesAsianRate config)
    , blackScholesAsianVolatility = readFlag "vol" (blackScholesAsianVolatility config)
    , blackScholesAsianMaturity = readFlag "maturity" (blackScholesAsianMaturity config)
    , blackScholesAsianSeed = readFlag "seed" (blackScholesAsianSeed config)
    }
  where
    readFlag :: Read a => String -> a -> a
    readFlag name defaultValue =
      case Map.lookup name flags of
        Nothing -> defaultValue
        Just value -> maybe defaultValue id (readMaybe value)
