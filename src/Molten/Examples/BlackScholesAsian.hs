{-# LANGUAGE PatternSynonyms #-}

module Molten.Examples.BlackScholesAsian
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
  ) where

import Data.Word (Word64)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Expr (Binary(..), Unary(..), constant, expE, select, (.+.), (.*.), (.<.))
import Molten.Array.Program
  ( Program
  , ProgramBuilder
  , Value
  , buildProgram
  , fillArrayP
  , forLoopP
  , inputArray
  , mapExpr
  , randNormalP
  , reduceAll
  , runProgram
  , withProgramRuntime
  , zipWithExpr
  )
import Molten.Array.Transfer (readDeviceArrayToHostArray)
import Molten.Core.Context (Context)
import Molten.Examples.Common (Timed(..), measureOnce)
import Molten.RAND.Runtime (RandGeneratorConfig(..))
import Molten.Reference (runProgramCpu)
import ROCm.RocRAND.Types (pattern RocRandRngPseudoDefault)

data BlackScholesAsianConfig = BlackScholesAsianConfig
  { blackScholesAsianPaths :: !Int
  , blackScholesAsianSteps :: !Int
  , blackScholesAsianSpot :: !Float
  , blackScholesAsianStrike :: !Float
  , blackScholesAsianRate :: !Float
  , blackScholesAsianVolatility :: !Float
  , blackScholesAsianMaturity :: !Float
  , blackScholesAsianSeed :: !Word64
  }
  deriving (Eq, Show)

data BlackScholesAsianShadowSummary = BlackScholesAsianShadowSummary
  { blackScholesAsianShadowPassed :: !Bool
  , blackScholesAsianShadowPriceAbsError :: !Float
  , blackScholesAsianShadowStdErrorAbsError :: !Float
  }
  deriving (Eq, Show)

data BlackScholesAsianStressSummary = BlackScholesAsianStressSummary
  { blackScholesAsianPassed :: !Bool
  , blackScholesAsianPrice :: !Float
  , blackScholesAsianStdError :: !Float
  , blackScholesAsianConfidenceLow :: !Float
  , blackScholesAsianConfidenceHigh :: !Float
  , blackScholesAsianRepeatedAbsDiff :: !Float
  , blackScholesAsianMeasuredSeconds :: !Double
  , blackScholesAsianShadow :: !BlackScholesAsianShadowSummary
  }
  deriving (Eq, Show)

defaultBlackScholesAsianShadowConfig :: BlackScholesAsianConfig
defaultBlackScholesAsianShadowConfig =
  BlackScholesAsianConfig
    { blackScholesAsianPaths = 4096
    , blackScholesAsianSteps = 16
    , blackScholesAsianSpot = 100.0
    , blackScholesAsianStrike = 100.0
    , blackScholesAsianRate = 5.0e-2
    , blackScholesAsianVolatility = 2.0e-1
    , blackScholesAsianMaturity = 1.0
    , blackScholesAsianSeed = 12345
    }

defaultBlackScholesAsianStressConfig :: BlackScholesAsianConfig
defaultBlackScholesAsianStressConfig =
  BlackScholesAsianConfig
    { blackScholesAsianPaths = 1000000
    , blackScholesAsianSteps = 252
    , blackScholesAsianSpot = 100.0
    , blackScholesAsianStrike = 100.0
    , blackScholesAsianRate = 5.0e-2
    , blackScholesAsianVolatility = 2.0e-1
    , blackScholesAsianMaturity = 1.0
    , blackScholesAsianSeed = 12345
    }

mkDeterministicNormalsVector :: BlackScholesAsianConfig -> A.Array A.S A.Ix1 Float
mkDeterministicNormalsVector config =
  AMV.fromVector' A.Seq (A.Sz1 (blackScholesAsianPaths config)) $
    VS.generate (blackScholesAsianPaths config) $ \pathIndex ->
      scaleToUnit (pathIndex * 17 + 11)

buildBlackScholesAsianShadowProgram :: BlackScholesAsianConfig -> A.Array A.S A.Ix1 Float -> IO (Program (Value A.Ix1 Float, Value A.Ix1 Float))
buildBlackScholesAsianShadowProgram config shocks =
  buildProgram $ do
    current0 <- fillArrayP @A.Ix1 @Float (blackScholesAsianSpot config) (A.Sz1 (blackScholesAsianPaths config))
    acc0 <- fillArrayP @A.Ix1 @Float 0 (A.Sz1 (blackScholesAsianPaths config))
    shockValue <- inputArray shocks
    (finalCurrent, finalAcc) <-
      forLoopP (blackScholesAsianSteps config) (current0, acc0) $ \(currentValue, accValue) -> do
        nextCurrent <- evolveStep config currentValue shockValue
        nextAcc <- zipWithExpr addBinary accValue nextCurrent
        pure (nextCurrent, nextAcc)
    payoff <- buildPayoff config finalCurrent finalAcc
    payoffSq <- mapExpr squareUnary payoff
    payoffSum <- reduceAll addBinary 0 payoff
    payoffSqSum <- reduceAll addBinary 0 payoffSq
    pure (payoffSum, payoffSqSum)

buildBlackScholesAsianStressProgram :: BlackScholesAsianConfig -> IO (Program (Value A.Ix1 Float, Value A.Ix1 Float))
buildBlackScholesAsianStressProgram config =
  buildProgram $ do
    current0 <- fillArrayP @A.Ix1 @Float (blackScholesAsianSpot config) (A.Sz1 (blackScholesAsianPaths config))
    acc0 <- fillArrayP @A.Ix1 @Float 0 (A.Sz1 (blackScholesAsianPaths config))
    (finalCurrent, finalAcc) <-
      forLoopP (blackScholesAsianSteps config) (current0, acc0) $ \(currentValue, accValue) -> do
        shocks <- randNormalP generatorConfig 0 1 (A.Sz1 (blackScholesAsianPaths config))
        nextCurrent <- evolveStep config currentValue shocks
        nextAcc <- zipWithExpr addBinary accValue nextCurrent
        pure (nextCurrent, nextAcc)
    payoff <- buildPayoff config finalCurrent finalAcc
    payoffSq <- mapExpr squareUnary payoff
    payoffSum <- reduceAll addBinary 0 payoff
    payoffSqSum <- reduceAll addBinary 0 payoffSq
    pure (payoffSum, payoffSqSum)
  where
    generatorConfig =
      RandGeneratorConfig
        { randGeneratorType = RocRandRngPseudoDefault
        , randGeneratorSeed = blackScholesAsianSeed config
        }

runBlackScholesAsianShadowCheck :: Context -> BlackScholesAsianConfig -> IO BlackScholesAsianShadowSummary
runBlackScholesAsianShadowCheck ctx config = do
  let shocks = mkDeterministicNormalsVector config
  program <- buildBlackScholesAsianShadowProgram config shocks
  (cpuSum, cpuSumSq) <- runProgramCpu program
  withProgramRuntime ctx $ \runtime -> do
    (gpuSumDevice, gpuSumSqDevice) <- runProgram runtime program
    gpuSum <- readDeviceArrayToHostArray ctx gpuSumDevice
    gpuSumSq <- readDeviceArrayToHostArray ctx gpuSumSqDevice
    let (cpuPrice, cpuStdError) = summarizeEstimate config cpuSum cpuSumSq
        (gpuPrice, gpuStdError) = summarizeEstimate config gpuSum gpuSumSq
        priceError = abs (cpuPrice - gpuPrice)
        stdErrorError = abs (cpuStdError - gpuStdError)
    pure
      BlackScholesAsianShadowSummary
        { blackScholesAsianShadowPassed = priceError <= 1.0e-3 && stdErrorError <= 1.0e-3
        , blackScholesAsianShadowPriceAbsError = priceError
        , blackScholesAsianShadowStdErrorAbsError = stdErrorError
        }

runBlackScholesAsianExample :: Context -> BlackScholesAsianConfig -> IO BlackScholesAsianStressSummary
runBlackScholesAsianExample ctx config = do
  shadowSummary <- runBlackScholesAsianShadowCheck ctx defaultBlackScholesAsianShadowConfig
  Timed firstEstimate measuredSeconds <- measureOnce (runBlackScholesAsianEstimate ctx config)
  secondEstimate <- runBlackScholesAsianEstimate ctx config
  let repeatedAbsDiff = abs (estimatePrice firstEstimate - estimatePrice secondEstimate)
      price = estimatePrice firstEstimate
      stdError = estimateStdError firstEstimate
      low = price - 1.96 * stdError
      high = price + 1.96 * stdError
      passed =
        blackScholesAsianShadowPassed shadowSummary
          && price >= 0
          && stdError >= 0
          && low <= price
          && high >= price
          && repeatedAbsDiff <= 1.0e-5
  pure
    BlackScholesAsianStressSummary
      { blackScholesAsianPassed = passed
      , blackScholesAsianPrice = price
      , blackScholesAsianStdError = stdError
      , blackScholesAsianConfidenceLow = low
      , blackScholesAsianConfidenceHigh = high
      , blackScholesAsianRepeatedAbsDiff = repeatedAbsDiff
      , blackScholesAsianMeasuredSeconds = measuredSeconds
      , blackScholesAsianShadow = shadowSummary
      }

data BlackScholesAsianEstimate = BlackScholesAsianEstimate
  { estimatePrice :: !Float
  , estimateStdError :: !Float
  }

runBlackScholesAsianEstimate :: Context -> BlackScholesAsianConfig -> IO BlackScholesAsianEstimate
runBlackScholesAsianEstimate ctx config = do
  program <- buildBlackScholesAsianStressProgram config
  withProgramRuntime ctx $ \runtime -> do
    (sumDevice, sumSqDevice) <- runProgram runtime program
    sumArray <- readDeviceArrayToHostArray ctx sumDevice
    sumSqArray <- readDeviceArrayToHostArray ctx sumSqDevice
    let (price, stdError) = summarizeEstimate config sumArray sumSqArray
    pure BlackScholesAsianEstimate {estimatePrice = price, estimateStdError = stdError}

summarizeEstimate :: BlackScholesAsianConfig -> A.Array A.S A.Ix1 Float -> A.Array A.S A.Ix1 Float -> (Float, Float)
summarizeEstimate config sumArray sumSqArray =
  let sampleCount = fromIntegral (blackScholesAsianPaths config)
      discount = exp (negate (blackScholesAsianRate config * blackScholesAsianMaturity config))
      payoffMean = scalarFromArray sumArray / sampleCount
      payoffSecondMoment = scalarFromArray sumSqArray / sampleCount
      discountedMean = discount * payoffMean
      discountedSecondMoment = discount * discount * payoffSecondMoment
      sampleVariance = max 0 (discountedSecondMoment - discountedMean * discountedMean)
      stdError = sqrt (sampleVariance / sampleCount)
   in (discountedMean, stdError)

buildPayoff :: BlackScholesAsianConfig -> Value A.Ix1 Float -> Value A.Ix1 Float -> ProgramBuilder (Value A.Ix1 Float)
buildPayoff config _ finalAcc = do
  let stepScale = recip (fromIntegral (max 1 (blackScholesAsianSteps config)))
      averageUnary = Unary (\x -> constant stepScale .*. x)
      payoffUnary =
        Unary
          (\averageValue ->
              select
                (averageValue .<. constant (blackScholesAsianStrike config))
                (constant 0)
                (averageValue .+. constant (negate (blackScholesAsianStrike config)))
          )
  averageValue <- mapExpr averageUnary finalAcc
  mapExpr payoffUnary averageValue

evolveStep :: BlackScholesAsianConfig -> Value A.Ix1 Float -> Value A.Ix1 Float -> ProgramBuilder (Value A.Ix1 Float)
evolveStep config currentValue shockValue = do
  let dt = blackScholesAsianMaturity config / fromIntegral (max 1 (blackScholesAsianSteps config))
      driftDt = (blackScholesAsianRate config - 0.5 * blackScholesAsianVolatility config * blackScholesAsianVolatility config) * dt
      volSqrtDt = blackScholesAsianVolatility config * sqrt dt
      growthUnary = Unary (\z -> expE (constant driftDt .+. constant volSqrtDt .*. z))
      multiplyBinary = Binary (\spot growth -> spot .*. growth)
  growth <- mapExpr growthUnary shockValue
  zipWithExpr multiplyBinary currentValue growth

squareUnary :: Unary Float Float
squareUnary = Unary (\x -> x .*. x)

addBinary :: Binary Float Float Float
addBinary = Binary (\x y -> x .+. y)

scaleToUnit :: Int -> Float
scaleToUnit seed = fromIntegral ((seed `mod` 29) - 14) / 10.0

scalarFromArray :: A.Array A.S A.Ix1 Float -> Float
scalarFromArray array =
  case VS.toList (AM.toStorableVector array) of
    [value] -> value
    values -> error ("expected scalar array, got length " <> show (length values))
