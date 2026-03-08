module Molten.Examples.MlpForwardSpec (spec) where

import qualified Data.Massiv.Array as A
import Molten.Array.Program (programResultValue, valueSize)
import Molten.Examples.MlpForward
  ( MlpConfig(..)
  , MlpInputs(..)
  , MlpShadowSummary(..)
  , buildMlpProgram
  , mkDeterministicMlpInputs
  , runMlpShadowCheck
  , summarizeMlpShadow
  )
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "mkDeterministicMlpInputs" $ do
    it "builds host tensors with the requested shapes" $ do
      let config = MlpConfig {mlpBatchSize = 4, mlpInputWidth = 3, mlpHiddenWidth = 5, mlpOutputWidth = 2}
          inputs = mkDeterministicMlpInputs config
      A.size (mlpInputMatrix inputs) `shouldBe` A.Sz2 4 3
      A.size (mlpWeight1 inputs) `shouldBe` A.Sz2 3 5
      A.size (mlpBias1 inputs) `shouldBe` A.Sz2 4 5
      A.size (mlpWeight2 inputs) `shouldBe` A.Sz2 5 2

  describe "buildMlpProgram" $ do
    it "returns output and checksum values with the expected shapes" $ do
      let config = MlpConfig {mlpBatchSize = 4, mlpInputWidth = 3, mlpHiddenWidth = 5, mlpOutputWidth = 2}
          inputs = mkDeterministicMlpInputs config
      program <- buildMlpProgram inputs
      let (outputValue, checksumValue) = programResultValue program
      valueSize outputValue `shouldBe` A.Sz2 4 2
      valueSize checksumValue `shouldBe` A.Sz1 1

  describe "runMlpShadowCheck" $ do
    it "matches CPU and GPU on a small shadow case" $ do
      withGpuContext $ \ctx -> do
        let config = MlpConfig {mlpBatchSize = 8, mlpInputWidth = 4, mlpHiddenWidth = 6, mlpOutputWidth = 3}
            inputs = mkDeterministicMlpInputs config
        summary <- runMlpShadowCheck ctx inputs 1.0e-4
        mlpShadowPassed summary `shouldBe` True

  describe "summarizeMlpShadow" $ do
    it "fails when tolerance is too small for a known mismatch" $ do
      let summary = summarizeMlpShadow 1.0e-8 [0.0 :: Float, 1.0] [0.0, 1.0001] [1.0 :: Float] [1.0]
      mlpShadowPassed summary `shouldBe` False
      mlpShadowMaxAbsError summary `shouldSatisfy` (> 0)

    it "accepts checksum drift when relative error is small" $ do
      let summary = summarizeMlpShadow 1.0e-4 [1, 2, 3 :: Float] [1, 2, 3.00001] [10000 :: Float] [10000.01]
      mlpShadowPassed summary `shouldBe` True
