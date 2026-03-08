module Main (main) where

import Molten (DeviceId(..), withContext)
import Molten.Examples.Common (parseFlagMap)
import Molten.Examples.MonteCarloBachelier
  ( MonteCarloConfig(..)
  , MonteCarloSummary(..)
  , defaultMonteCarloConfig
  , runMonteCarloExample
  )
import qualified Data.Map.Strict
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
      let config = applyMonteCarloFlags defaultMonteCarloConfig flags
      putStrLn "Molten example: monte-carlo-bachelier"
      putStrLn ("Device: " <> deviceName <> " (" <> archName <> ")")
      putStrLn ("Config: " <> show config)
      withContext (DeviceId 0) $ \ctx -> do
        summary <- runMonteCarloExample ctx config
        putStrLn ("Run: " <> if monteCarloPassed summary then "PASS" else "FAIL")
        putStrLn ("Price estimate: " <> show (monteCarloPrice summary))
        putStrLn ("Std error: " <> show (monteCarloStdError summary))
        putStrLn ("95% CI: [" <> show (monteCarloConfidenceLow summary) <> ", " <> show (monteCarloConfidenceHigh summary) <> "]")
        putStrLn ("Repeated abs diff: " <> show (monteCarloRepeatedAbsDiff summary))
        putStrLn ("Measured time (s): " <> show (monteCarloMeasuredSeconds summary))
        if not (monteCarloPassed summary)
          then fail "monte-carlo-bachelier self-check failed"
          else pure ()

applyMonteCarloFlags :: MonteCarloConfig -> Data.Map.Strict.Map String String -> MonteCarloConfig
applyMonteCarloFlags config flags =
  config
    { monteCarloPaths = readFlag "paths" (monteCarloPaths config)
    , monteCarloSeed = readFlag "seed" (monteCarloSeed config)
    , monteCarloSpot = readFlag "spot" (monteCarloSpot config)
    , monteCarloStrike = readFlag "strike" (monteCarloStrike config)
    , monteCarloSigma = readFlag "sigma" (monteCarloSigma config)
    , monteCarloSqrtT = readFlag "sqrtT" (monteCarloSqrtT config)
    }
  where
    readFlag :: Read a => String -> a -> a
    readFlag name defaultValue =
      case Data.Map.Strict.lookup name flags of
        Nothing -> defaultValue
        Just value -> maybe defaultValue id (readMaybe value)
