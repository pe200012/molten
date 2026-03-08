{-# LANGUAGE TypeApplications #-}

module Molten.Examples.BlackScholesAsianSpec (spec) where

import qualified Data.Massiv.Array as A
import Molten.Array.Program (programResultValue, valueSize)
import Molten.Examples.BlackScholesAsian
  ( BlackScholesAsianConfig(..)
  , BlackScholesAsianShadowSummary(..)
  , BlackScholesAsianStressSummary(..)
  , buildBlackScholesAsianShadowProgram
  , buildBlackScholesAsianStressProgram
  , defaultBlackScholesAsianShadowConfig
  , defaultBlackScholesAsianStressConfig
  , mkDeterministicNormalsVector
  , runBlackScholesAsianExample
  , runBlackScholesAsianShadowCheck
  )
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "mkDeterministicNormalsVector" $ do
    it "builds a deterministic vector with the expected shape" $ do
      let normals = mkDeterministicNormalsVector defaultBlackScholesAsianShadowConfig
      A.size normals `shouldBe` A.Sz1 4096
      mkDeterministicNormalsVector defaultBlackScholesAsianShadowConfig `shouldBe` normals

  describe "buildBlackScholesAsianShadowProgram" $ do
    it "returns scalar payoff summaries" $ do
      let config = defaultBlackScholesAsianShadowConfig {blackScholesAsianPaths = 8, blackScholesAsianSteps = 4}
          shocks = mkDeterministicNormalsVector config
      program <- buildBlackScholesAsianShadowProgram config shocks
      let (sumValue, sumSqValue) = programResultValue program
      valueSize sumValue `shouldBe` A.Sz1 1
      valueSize sumSqValue `shouldBe` A.Sz1 1

  describe "runBlackScholesAsianShadowCheck" $ do
    it "passes a small CPU-vs-GPU shadow case" $
      withGpuContext $ \ctx -> do
        let config = defaultBlackScholesAsianShadowConfig {blackScholesAsianPaths = 256, blackScholesAsianSteps = 4}
        summary <- runBlackScholesAsianShadowCheck ctx config
        blackScholesAsianShadowPassed summary `shouldBe` True

  describe "buildBlackScholesAsianStressProgram" $ do
    it "returns scalar payoff summaries for the RAND path" $ do
      let config = defaultBlackScholesAsianStressConfig {blackScholesAsianPaths = 32, blackScholesAsianSteps = 8}
      program <- buildBlackScholesAsianStressProgram config
      let (sumValue, sumSqValue) = programResultValue program
      valueSize sumValue `shouldBe` A.Sz1 1
      valueSize sumSqValue `shouldBe` A.Sz1 1

  describe "runBlackScholesAsianExample" $ do
    it "produces a non-negative price, finite stderr, and stable repeated run" $
      withGpuContext $ \ctx -> do
        let config = defaultBlackScholesAsianStressConfig {blackScholesAsianPaths = 20000, blackScholesAsianSteps = 16}
        summary <- runBlackScholesAsianExample ctx config
        blackScholesAsianPassed summary `shouldBe` True
        blackScholesAsianPrice summary `shouldSatisfy` (>= 0)
        blackScholesAsianStdError summary `shouldSatisfy` (>= 0)
        blackScholesAsianRepeatedAbsDiff summary `shouldSatisfy` (<= 1.0e-5)
