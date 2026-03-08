{-# LANGUAGE TypeApplications #-}

module Molten.Examples.AttentionSpec (spec) where

import qualified Data.Massiv.Array as A
import Molten.Array.Program (programResultValue, valueSize)
import Molten.Examples.Attention
  ( AttentionConfig(..)
  , AttentionInputs(..)
  , AttentionShadowSummary(..)
  , buildAttentionProgram
  , defaultAttentionShadowConfig
  , mkDeterministicAttentionInputs
  , runAttentionShadowCheck
  )
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "Attention inputs" $ do
    it "builds deterministic Q/K/V matrices with the expected shapes" $ do
      let inputs = mkDeterministicAttentionInputs defaultAttentionShadowConfig
      A.size (attentionQueryMatrix inputs) `shouldBe` A.Sz2 64 64
      A.size (attentionKeyMatrix inputs) `shouldBe` A.Sz2 64 64
      A.size (attentionValueMatrix inputs) `shouldBe` A.Sz2 64 64

  describe "buildAttentionProgram" $ do
    it "returns output and probability matrices with the expected shapes" $ do
      let inputs = mkDeterministicAttentionInputs defaultAttentionShadowConfig
      program <- buildAttentionProgram inputs
      let (outputValue, probabilityValue) = programResultValue program
      valueSize outputValue `shouldBe` A.Sz2 64 64
      valueSize probabilityValue `shouldBe` A.Sz2 64 64

  describe "runAttentionShadowCheck" $ do
    it "passes a small CPU-vs-GPU shadow case" $
      withGpuContext $ \ctx -> do
        let config = defaultAttentionShadowConfig {attentionTokens = 8, attentionModelWidth = 8, attentionValueWidth = 8}
            inputs = mkDeterministicAttentionInputs config
        summary <- runAttentionShadowCheck ctx inputs 1.0e-3
        attentionShadowPassed summary `shouldBe` True
        attentionShadowRowSumMaxDrift summary `shouldSatisfy` (<= 1.0e-3)
