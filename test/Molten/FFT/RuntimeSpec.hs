{-# LANGUAGE PatternSynonyms #-}

module Molten.FFT.RuntimeSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.List (isInfixOf)
import Molten.FFT.Runtime
  ( CachedFftPlan(cachedFftPlanId)
  , FftPlanKey(fftPlanKeyWorkspaceMode)
  , FftTransform(..)
  , FftWorkspaceMode(..)
  , lookupOrCreateFftPlan
  , mkFftPlanKey
  , withFftRuntime
  )
import Molten.TestSupport (withGpuContext)
import ROCm.FFI.Core.Exception (ArgumentError(argFunction, argMessage))
import ROCm.RocFFT (pattern RocfftPrecisionSingle)
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "mkFftPlanKey" $ do
    it "rejects ranks outside 1..3" $
      mkFftPlanKey FftTransformComplexForward RocfftPrecisionSingle [] 1 FftWorkspaceAuto
        `shouldThrow` isArgumentError "mkFftPlanKey" "lengths"

    it "rejects batch sizes smaller than 1" $
      mkFftPlanKey FftTransformComplexForward RocfftPrecisionSingle [8] 0 FftWorkspaceAuto
        `shouldThrow` isArgumentError "mkFftPlanKey" "batch"

    it "preserves the chosen workspace mode" $ do
      autoKey <- mkFftPlanKey FftTransformComplexForward RocfftPrecisionSingle [8] 1 FftWorkspaceAuto
      explicitKey <- mkFftPlanKey FftTransformComplexForward RocfftPrecisionSingle [8] 1 FftWorkspaceExplicit
      fftPlanKeyWorkspaceMode autoKey `shouldBe` FftWorkspaceAuto
      fftPlanKeyWorkspaceMode explicitKey `shouldBe` FftWorkspaceExplicit

  describe "lookupOrCreateFftPlan" $ do
    it "reuses cached plans for the same key" $
      withGpuContext $ \ctx ->
        withFftRuntime ctx $ \runtime -> do
          key <- mkFftPlanKey FftTransformComplexForward RocfftPrecisionSingle [8] 1 FftWorkspaceAuto
          plan1 <- lookupOrCreateFftPlan runtime key
          plan2 <- lookupOrCreateFftPlan runtime key
          cachedFftPlanId plan1 `shouldBe` cachedFftPlanId plan2

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
