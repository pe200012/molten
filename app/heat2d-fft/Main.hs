module Main (main) where

import Molten (DeviceId(..), withContext)
import Molten.Examples.Common (parseFlagMap)
import Molten.Examples.Heat2dFft
  ( Heat2dConfig(..)
  , Heat2dSummary(..)
  , defaultHeat2dConfig
  , runHeat2dExample
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
      let config = applyHeatFlags defaultHeat2dConfig flags
      putStrLn "Molten example: heat2d-fft"
      putStrLn ("Device: " <> deviceName <> " (" <> archName <> ")")
      putStrLn ("Config: " <> show config)
      withContext (DeviceId 0) $ \ctx -> do
        summary <- runHeat2dExample ctx config
        putStrLn ("Run: " <> if heatRunPassed summary then "PASS" else "FAIL")
        putStrLn ("Initial energy: " <> show (heatInitialEnergy summary))
        putStrLn ("Final energy: " <> show (heatFinalEnergy summary))
        putStrLn ("Repeated max abs diff: " <> show (heatRepeatedMaxAbsDiff summary))
        putStrLn ("Measured time (s): " <> show (heatMeasuredSeconds summary))
        if not (heatRunPassed summary)
          then fail "heat2d-fft self-check failed"
          else pure ()

applyHeatFlags :: Heat2dConfig -> Data.Map.Strict.Map String String -> Heat2dConfig
applyHeatFlags config flags =
  config
    { heatNx = readFlag "nx" (heatNx config)
    , heatNy = readFlag "ny" (heatNy config)
    , heatSteps = readFlag "steps" (heatSteps config)
    , heatAlpha = readFlag "alpha" (heatAlpha config)
    , heatDt = readFlag "dt" (heatDt config)
    }
  where
    readFlag :: Read a => String -> a -> a
    readFlag name defaultValue =
      case Data.Map.Strict.lookup name flags of
        Nothing -> defaultValue
        Just value -> maybe defaultValue id (readMaybe value)
