module Molten.Examples.MonteCarloBachelierSpec (spec) where

import qualified Data.Massiv.Array as A
import Molten.Array.Program (programResultValue, valueSize)
import Molten.Examples.MonteCarloBachelier
  ( MonteCarloConfig(..)
  , MonteCarloSummary(..)
  , buildMonteCarloProgram
  , runMonteCarloExample
  )
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "buildMonteCarloProgram" $ do
    it "returns scalar reduction outputs" $ do
      let config = MonteCarloConfig
            { monteCarloPaths = 32
            , monteCarloSeed = 12345
            , monteCarloSpot = 100
            , monteCarloStrike = 100
            , monteCarloSigma = 20
            , monteCarloSqrtT = 1
            }
      program <- buildMonteCarloProgram config
      let (sumValue, sumSqValue) = programResultValue program
      valueSize sumValue `shouldBe` A.Sz1 1
      valueSize sumSqValue `shouldBe` A.Sz1 1

  describe "runMonteCarloExample" $ do
    it "repeats deterministically for the same seed and returns a valid confidence interval" $ do
      withGpuContext $ \ctx -> do
        let config = MonteCarloConfig
              { monteCarloPaths = 4096
              , monteCarloSeed = 12345
              , monteCarloSpot = 100
              , monteCarloStrike = 100
              , monteCarloSigma = 20
              , monteCarloSqrtT = 1
              }
        summary <- runMonteCarloExample ctx config
        monteCarloPassed summary `shouldBe` True
        monteCarloPrice summary `shouldSatisfy` (>= 0)
        monteCarloRepeatedAbsDiff summary `shouldSatisfy` (<= 1.0e-5)
        monteCarloConfidenceLow summary `shouldSatisfy` (<= monteCarloPrice summary)
        monteCarloConfidenceHigh summary `shouldSatisfy` (>= monteCarloPrice summary)
