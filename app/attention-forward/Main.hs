module Main (main) where

import qualified Data.Map.Strict as Map
import Molten (DeviceId(..), withContext)
import Molten.Examples.Attention
  ( AttentionConfig(..)
  , AttentionShadowSummary(..)
  , AttentionStressSummary(..)
  , defaultAttentionShadowConfig
  , defaultAttentionStressConfig
  , mkDeterministicAttentionInputs
  , runAttentionShadowCheck
  , runAttentionStress
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
      let shadowConfig = applyAttentionFlags defaultAttentionShadowConfig flags
          stressConfig = applyAttentionFlags defaultAttentionStressConfig flags
      putStrLn "Molten example: attention-forward"
      putStrLn ("Device: " <> deviceName <> " (" <> archName <> ")")
      putStrLn ("Shadow config: " <> show shadowConfig)
      putStrLn ("Stress config: " <> show stressConfig)
      withContext (DeviceId 0) $ \ctx -> do
        let shadowInputs = mkDeterministicAttentionInputs shadowConfig
        shadowSummary <- runAttentionShadowCheck ctx shadowInputs 1.0e-3
        putStrLn ("Shadow check: " <> if attentionShadowPassed shadowSummary then "PASS" else "FAIL")
        putStrLn ("Shadow output max abs err: " <> show (attentionShadowOutputMaxAbsError shadowSummary))
        putStrLn ("Shadow probability max abs err: " <> show (attentionShadowProbMaxAbsError shadowSummary))
        putStrLn ("Shadow row-sum max drift: " <> show (attentionShadowRowSumMaxDrift shadowSummary))
        if not (attentionShadowPassed shadowSummary)
          then fail "attention-forward shadow check failed"
          else pure ()
        stressSummary <- runAttentionStress ctx stressConfig
        putStrLn ("Stress run: " <> if attentionStressPassed stressSummary then "PASS" else "FAIL")
        putStrLn ("Stress output shape: " <> show (attentionStressOutputSize stressSummary))
        putStrLn ("Stress output checksum: " <> show (attentionStressChecksum stressSummary))
        putStrLn ("Stress row-sum max drift: " <> show (attentionStressRowSumMaxDrift stressSummary))
        putStrLn ("GPU run time (s): " <> show (attentionStressSeconds stressSummary))
        if not (attentionStressPassed stressSummary)
          then fail "attention-forward stress check failed"
          else pure ()

applyAttentionFlags :: AttentionConfig -> Map.Map String String -> AttentionConfig
applyAttentionFlags config flags =
  config
    { attentionTokens = readFlag "tokens" (attentionTokens config)
    , attentionModelWidth = readFlag "model" (attentionModelWidth config)
    , attentionValueWidth = readFlag "value" (attentionValueWidth config)
    }
  where
    readFlag :: Read a => String -> a -> a
    readFlag name defaultValue =
      case Map.lookup name flags of
        Nothing -> defaultValue
        Just value -> maybe defaultValue id (readMaybe value)
