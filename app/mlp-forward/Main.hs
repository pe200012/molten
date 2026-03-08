module Main (main) where

import qualified Data.Map.Strict
import Molten (DeviceId(..), withContext)
import Molten.Examples.Common (parseFlagMap)
import Molten.Examples.MlpForward
  ( MlpConfig(..)
  , MlpShadowSummary(..)
  , MlpStressSummary(..)
  , defaultMlpShadowConfig
  , defaultMlpStressConfig
  , mkDeterministicMlpInputs
  , runMlpShadowCheck
  , runMlpStress
  )
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
      let shadowConfig = applyMlpFlags defaultMlpShadowConfig flags
          stressConfig = applyMlpFlags defaultMlpStressConfig flags
      putStrLn "Molten example: mlp-forward"
      putStrLn ("Device: " <> deviceName <> " (" <> archName <> ")")
      putStrLn ("Shadow config: " <> show shadowConfig)
      putStrLn ("Stress config: " <> show stressConfig)
      withContext (DeviceId 0) $ \ctx -> do
        let shadowInputs = mkDeterministicMlpInputs shadowConfig
        shadowSummary <- runMlpShadowCheck ctx shadowInputs 1.0e-3
        putStrLn ("Shadow check: " <> if mlpShadowPassed shadowSummary then "PASS" else "FAIL")
        putStrLn ("Shadow max abs err: " <> show (mlpShadowMaxAbsError shadowSummary))
        putStrLn ("Shadow checksum abs err: " <> show (mlpShadowChecksumAbsError shadowSummary))
        if not (mlpShadowPassed shadowSummary)
          then fail "mlp-forward shadow check failed"
          else pure ()
        stressSummary <- runMlpStress ctx stressConfig
        putStrLn ("Stress run: " <> if mlpStressPassed stressSummary then "PASS" else "FAIL")
        putStrLn ("GPU run time (s): " <> show (mlpStressSeconds stressSummary))
        putStrLn ("Output checksum: " <> show (mlpStressChecksum stressSummary))
        putStrLn ("Output shape: " <> show (mlpStressOutputSize stressSummary))
        if not (mlpStressPassed stressSummary)
          then fail "mlp-forward stress check failed"
          else pure ()

applyMlpFlags :: MlpConfig -> Data.Map.Strict.Map String String -> MlpConfig
applyMlpFlags config flags =
  config
    { mlpBatchSize = readFlag "batch" (mlpBatchSize config)
    , mlpInputWidth = readFlag "in" (mlpInputWidth config)
    , mlpHiddenWidth = readFlag "hidden" (mlpHiddenWidth config)
    , mlpOutputWidth = readFlag "out" (mlpOutputWidth config)
    }
  where
    readFlag :: Read a => String -> a -> a
    readFlag name defaultValue =
      case Data.Map.Strict.lookup name flags of
        Nothing -> defaultValue
        Just value -> maybe defaultValue id (readMaybe value)
