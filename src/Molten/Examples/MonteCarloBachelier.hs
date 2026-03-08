{-# LANGUAGE PatternSynonyms #-}

module Molten.Examples.MonteCarloBachelier
  ( MonteCarloConfig(..)
  , MonteCarloSummary(..)
  , buildMonteCarloProgram
  , defaultMonteCarloConfig
  , runMonteCarloExample
  ) where

import Data.Word (Word64)
import qualified Data.Massiv.Array as A
import Molten.Array.Expr (Binary(..), Unary(..), constant, select, (.+.), (.*.), (.<.))
import Molten.Array.Program
  ( Program
  , Value
  , buildProgram
  , mapExpr
  , randNormalP
  , reduceAll
  , runProgram
  , withProgramRuntime
  )
import Molten.Array.Transfer (readDeviceArrayToHostArray)
import Molten.Core.Context (Context)
import Molten.Examples.Common (Timed(..), measureOnce)
import Molten.RAND.Runtime (RandGeneratorConfig(..))
import ROCm.RocRAND.Types (pattern RocRandRngPseudoDefault)

data MonteCarloConfig = MonteCarloConfig
  { monteCarloPaths :: !Int
  , monteCarloSeed :: !Word64
  , monteCarloSpot :: !Float
  , monteCarloStrike :: !Float
  , monteCarloSigma :: !Float
  , monteCarloSqrtT :: !Float
  }
  deriving (Eq, Show)

data MonteCarloSummary = MonteCarloSummary
  { monteCarloPassed :: !Bool
  , monteCarloPrice :: !Float
  , monteCarloStdError :: !Float
  , monteCarloConfidenceLow :: !Float
  , monteCarloConfidenceHigh :: !Float
  , monteCarloRepeatedAbsDiff :: !Float
  , monteCarloMeasuredSeconds :: !Double
  }
  deriving (Eq, Show)

defaultMonteCarloConfig :: MonteCarloConfig
defaultMonteCarloConfig =
  MonteCarloConfig
    { monteCarloPaths = 8000000
    , monteCarloSeed = 12345
    , monteCarloSpot = 100
    , monteCarloStrike = 100
    , monteCarloSigma = 20
    , monteCarloSqrtT = 1
    }

buildMonteCarloProgram :: MonteCarloConfig -> IO (Program (Value A.Ix1 Float, Value A.Ix1 Float))
buildMonteCarloProgram config =
  buildProgram $ do
    samples <- randNormalP generatorConfig 0 1 (A.Sz1 (monteCarloPaths config))
    terminal <- mapExpr terminalUnary samples
    payoff <- mapExpr payoffUnary terminal
    payoffSq <- mapExpr squareUnary payoff
    payoffSum <- reduceAll addBinary 0 payoff
    payoffSqSum <- reduceAll addBinary 0 payoffSq
    pure (payoffSum, payoffSqSum)
  where
    generatorConfig =
      RandGeneratorConfig
        { randGeneratorType = RocRandRngPseudoDefault
        , randGeneratorSeed = monteCarloSeed config
        }
    terminalUnary =
      Unary
        (\z ->
            constant (monteCarloSpot config)
              .+. (constant (monteCarloSigma config * monteCarloSqrtT config) .*. z)
        )
    payoffUnary =
      Unary
        (\terminalValue ->
            select
              (terminalValue .<. constant (monteCarloStrike config))
              (constant 0)
              (terminalValue .+. constant (negate (monteCarloStrike config)))
        )
    squareUnary = Unary (\x -> x .*. x)
    addBinary = Binary (\x y -> x .+. y)

runMonteCarloExample :: Context -> MonteCarloConfig -> IO MonteCarloSummary
runMonteCarloExample ctx config = do
  Timed firstEstimate measuredSeconds <- measureOnce (runMonteCarloEstimate ctx config)
  secondEstimate <- runMonteCarloEstimate ctx config
  let repeatedAbsDiff = abs (estimatePrice firstEstimate - estimatePrice secondEstimate)
      stdError = estimateStdError firstEstimate
      price = estimatePrice firstEstimate
      low = price - 1.96 * stdError
      high = price + 1.96 * stdError
      passed =
        price >= 0
          && stdError >= 0
          && low <= price
          && high >= price
          && repeatedAbsDiff <= 1.0e-5
  pure
    MonteCarloSummary
      { monteCarloPassed = passed
      , monteCarloPrice = price
      , monteCarloStdError = stdError
      , monteCarloConfidenceLow = low
      , monteCarloConfidenceHigh = high
      , monteCarloRepeatedAbsDiff = repeatedAbsDiff
      , monteCarloMeasuredSeconds = measuredSeconds
      }

data MonteCarloEstimate = MonteCarloEstimate
  { estimatePrice :: !Float
  , estimateStdError :: !Float
  }

runMonteCarloEstimate :: Context -> MonteCarloConfig -> IO MonteCarloEstimate
runMonteCarloEstimate ctx config = do
  program <- buildMonteCarloProgram config
  withProgramRuntime ctx $ \runtime -> do
    (sumDevice, sumSqDevice) <- runProgram runtime program
    sumArray <- readDeviceArrayToHostArray ctx sumDevice
    sumSqArray <- readDeviceArrayToHostArray ctx sumSqDevice
    let sampleCount = fromIntegral (monteCarloPaths config)
        sampleMean = scalarFromArray sumArray / sampleCount
        secondMoment = scalarFromArray sumSqArray / sampleCount
        sampleVariance = max 0 (secondMoment - sampleMean * sampleMean)
        stdError = sqrt (sampleVariance / sampleCount)
    pure MonteCarloEstimate {estimatePrice = sampleMean, estimateStdError = stdError}

scalarFromArray :: A.Array A.S A.Ix1 Float -> Float
scalarFromArray array =
  case A.toList array of
    [value] -> value
    values -> error ("expected scalar array, got length " <> show (length values))
