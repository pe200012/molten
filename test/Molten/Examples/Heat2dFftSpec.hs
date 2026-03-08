module Molten.Examples.Heat2dFftSpec (spec) where

import qualified Data.Massiv.Array as A
import Molten.Examples.Heat2dFft
  ( Heat2dConfig(..)
  , Heat2dSummary(..)
  , buildHeatInitialField
  , buildHeatSpectralMultiplier
  , runHeat2dExample
  )
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "buildHeatSpectralMultiplier" $ do
    it "matches the configured 2D shape" $ do
      let config = Heat2dConfig {heatNx = 8, heatNy = 6, heatSteps = 2, heatAlpha = 0.05, heatDt = 1.0e-3}
      A.size (buildHeatSpectralMultiplier config) `shouldBe` A.Sz2 8 6

  describe "buildHeatInitialField" $ do
    it "matches the configured 2D shape" $ do
      let config = Heat2dConfig {heatNx = 8, heatNy = 6, heatSteps = 2, heatAlpha = 0.05, heatDt = 1.0e-3}
      A.size (buildHeatInitialField config) `shouldBe` A.Sz2 8 6

  describe "runHeat2dExample" $ do
    it "is stable across repeated runs and does not grow energy on a small case" $ do
      withGpuContext $ \ctx -> do
        let config = Heat2dConfig {heatNx = 16, heatNy = 16, heatSteps = 3, heatAlpha = 0.05, heatDt = 1.0e-3}
        summary <- runHeat2dExample ctx config
        heatRunPassed summary `shouldBe` True
        heatRepeatedMaxAbsDiff summary `shouldSatisfy` (<= 1.0e-3)
        heatFinalEnergy summary `shouldSatisfy` (<= heatInitialEnergy summary + 1.0e-3)
