{-# LANGUAGE TypeApplications #-}

module Molten.Examples.MlpForward
  ( MlpConfig(..)
  , MlpInputs(..)
  , MlpShadowSummary(..)
  , MlpStressSummary(..)
  , buildMlpProgram
  , defaultMlpShadowConfig
  , defaultMlpStressConfig
  , mkDeterministicMlpInputs
  , runMlpShadowCheck
  , runMlpStress
  , summarizeMlpShadow
  ) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Expr (Binary(..), Unary(..), constant, select, (.+.), (.*.), (.<.))
import Molten.Array.Program
  ( MatrixGemmValue(..)
  , Program
  , Value
  , buildProgram
  , gemmMatrixP
  , inputArray
  , mapExpr
  , reduceAll
  , runProgram
  , withProgramRuntime
  , zipWithExpr
  )
import Molten.Array.Transfer (readDeviceArrayToHostArray)
import Molten.BLAS.Types (Transpose(NoTranspose))
import Molten.Core.Context (Context)
import Molten.Examples.Common (Timed(..), measureOnce)
import Molten.Reference (runProgramCpu)

data MlpConfig = MlpConfig
  { mlpBatchSize :: !Int
  , mlpInputWidth :: !Int
  , mlpHiddenWidth :: !Int
  , mlpOutputWidth :: !Int
  }
  deriving (Eq, Show)

data MlpInputs = MlpInputs
  { mlpInputMatrix :: !(A.Array A.S A.Ix2 Float)
  , mlpWeight1 :: !(A.Array A.S A.Ix2 Float)
  , mlpBias1 :: !(A.Array A.S A.Ix2 Float)
  , mlpWeight2 :: !(A.Array A.S A.Ix2 Float)
  }
  deriving (Eq, Show)

data MlpShadowSummary = MlpShadowSummary
  { mlpShadowPassed :: !Bool
  , mlpShadowMaxAbsError :: !Float
  , mlpShadowChecksumAbsError :: !Float
  }
  deriving (Eq, Show)

data MlpStressSummary = MlpStressSummary
  { mlpStressPassed :: !Bool
  , mlpStressSeconds :: !Double
  , mlpStressOutputSize :: !(A.Sz A.Ix2)
  , mlpStressChecksum :: !Float
  , mlpStressShadow :: !MlpShadowSummary
  }
  deriving (Eq, Show)

defaultMlpShadowConfig :: MlpConfig
defaultMlpShadowConfig =
  MlpConfig
    { mlpBatchSize = 64
    , mlpInputWidth = 128
    , mlpHiddenWidth = 256
    , mlpOutputWidth = 64
    }

defaultMlpStressConfig :: MlpConfig
defaultMlpStressConfig =
  MlpConfig
    { mlpBatchSize = 4096
    , mlpInputWidth = 1024
    , mlpHiddenWidth = 2048
    , mlpOutputWidth = 512
    }

mkDeterministicMlpInputs :: MlpConfig -> MlpInputs
mkDeterministicMlpInputs config =
  MlpInputs
    { mlpInputMatrix = buildMatrix (A.Sz2 (mlpBatchSize config) (mlpInputWidth config)) inputValue
    , mlpWeight1 = buildMatrix (A.Sz2 (mlpInputWidth config) (mlpHiddenWidth config)) weight1Value
    , mlpBias1 = buildMatrix (A.Sz2 (mlpBatchSize config) (mlpHiddenWidth config)) biasValue
    , mlpWeight2 = buildMatrix (A.Sz2 (mlpHiddenWidth config) (mlpOutputWidth config)) weight2Value
    }
  where
    inputValue row col = scaleToUnit (row * 17 + col * 13 + 3)
    weight1Value row col = 0.25 * scaleToUnit (row * 11 + col * 7 + 5)
    biasValue row col = 0.10 * scaleToUnit (row * 5 + col * 19 + 1)
    weight2Value row col = 0.20 * scaleToUnit (row * 23 + col * 3 + 9)

buildMlpProgram :: MlpInputs -> IO (Program (Value A.Ix2 Float, Value A.Ix1 Float))
buildMlpProgram inputs =
  buildProgram $ do
    x <- inputArray (mlpInputMatrix inputs)
    w1 <- inputArray (mlpWeight1 inputs)
    b1 <- inputArray (mlpBias1 inputs)
    w2 <- inputArray (mlpWeight2 inputs)
    cHidden <- inputArray (zeroMatrixLike (A.Sz2 batchSize hiddenWidth))
    cOutput <- inputArray (zeroMatrixLike (A.Sz2 batchSize outputWidth))
    hiddenLinear <-
      gemmMatrixP
        MatrixGemmValue
          { matrixGemmValueTransA = NoTranspose
          , matrixGemmValueTransB = NoTranspose
          , matrixGemmValueAlpha = 1
          , matrixGemmValueA = x
          , matrixGemmValueB = w1
          , matrixGemmValueBeta = 0
          , matrixGemmValueC = cHidden
          }
    hiddenBiased <- zipWithExpr addBinary hiddenLinear b1
    hiddenActivated <- mapExpr reluUnary hiddenBiased
    output <-
      gemmMatrixP
        MatrixGemmValue
          { matrixGemmValueTransA = NoTranspose
          , matrixGemmValueTransB = NoTranspose
          , matrixGemmValueAlpha = 1
          , matrixGemmValueA = hiddenActivated
          , matrixGemmValueB = w2
          , matrixGemmValueBeta = 0
          , matrixGemmValueC = cOutput
          }
    checksumSquares <- mapExpr squareUnary hiddenActivated
    checksum <- reduceAll addBinary 0 checksumSquares
    pure (output, checksum)
  where
    A.Sz2 batchSize _ = A.size (mlpInputMatrix inputs)
    A.Sz2 _ hiddenWidth = A.size (mlpWeight1 inputs)
    A.Sz2 _ outputWidth = A.size (mlpWeight2 inputs)

runMlpShadowCheck :: Context -> MlpInputs -> Float -> IO MlpShadowSummary
runMlpShadowCheck ctx inputs tolerance = do
  program <- buildMlpProgram inputs
  (cpuOutput, cpuChecksum) <- runProgramCpu program
  withProgramRuntime ctx $ \runtime -> do
    (gpuOutputDevice, gpuChecksumDevice) <- runProgram runtime program
    gpuOutput <- readDeviceArrayToHostArray ctx gpuOutputDevice
    gpuChecksum <- readDeviceArrayToHostArray ctx gpuChecksumDevice
    pure
      ( summarizeMlpShadow
          tolerance
          (VS.toList (AM.toStorableVector cpuOutput))
          (VS.toList (AM.toStorableVector gpuOutput))
          (VS.toList (AM.toStorableVector cpuChecksum))
          (VS.toList (AM.toStorableVector gpuChecksum))
      )

runMlpStress :: Context -> MlpConfig -> IO MlpStressSummary
runMlpStress ctx config = do
  let stressInputs = mkDeterministicMlpInputs config
      shadowInputs = mkDeterministicMlpInputs defaultMlpShadowConfig
  shadowSummary <- runMlpShadowCheck ctx shadowInputs 1.0e-4
  Timed {timedValue = (outputHost, checksumHost), timedSeconds = runtimeSeconds} <-
    measureOnce $ do
      program <- buildMlpProgram stressInputs
      withProgramRuntime ctx $ \runtime -> do
        (outputDevice, checksumDevice) <- runProgram runtime program
        outputHost <- readDeviceArrayToHostArray ctx outputDevice
        checksumHost <- readDeviceArrayToHostArray ctx checksumDevice
        pure (outputHost, checksumHost)
  let checksumValue = scalarFromIx1 checksumHost
  pure
    MlpStressSummary
      { mlpStressPassed = mlpShadowPassed shadowSummary
      , mlpStressSeconds = runtimeSeconds
      , mlpStressOutputSize = A.size outputHost
      , mlpStressChecksum = checksumValue
      , mlpStressShadow = shadowSummary
      }

summarizeMlpShadow :: Float -> [Float] -> [Float] -> [Float] -> [Float] -> MlpShadowSummary
summarizeMlpShadow tolerance cpuOutput gpuOutput cpuChecksum gpuChecksum =
  MlpShadowSummary
    { mlpShadowPassed = sameLength && outputError <= tolerance && checksumError <= checksumTolerance
    , mlpShadowMaxAbsError = outputError
    , mlpShadowChecksumAbsError = checksumError
    }
  where
    sameLength = length cpuOutput == length gpuOutput && length cpuChecksum == length gpuChecksum
    outputError = maxAbsDifference cpuOutput gpuOutput
    checksumError = maxAbsDifference cpuChecksum gpuChecksum
    checksumScale = maximum (1 : map abs cpuChecksum ++ map abs gpuChecksum)
    checksumTolerance = tolerance * checksumScale

reluUnary :: Unary Float Float
reluUnary = Unary (\x -> select (x .<. constant 0) (constant 0) x)

squareUnary :: Unary Float Float
squareUnary = Unary (\x -> x .*. x)

addBinary :: Binary Float Float Float
addBinary = Binary (\x y -> x .+. y)

buildMatrix :: A.Sz A.Ix2 -> (Int -> Int -> Float) -> A.Array A.S A.Ix2 Float
buildMatrix size@(A.Sz2 rows cols) entry =
  AMV.fromVector' A.Seq size $ VS.generate (rows * cols) $ \linearIndex ->
    let (row, col) = linearIndex `quotRem` cols
     in entry row col

zeroMatrixLike :: A.Sz A.Ix2 -> A.Array A.S A.Ix2 Float
zeroMatrixLike size@(A.Sz2 rows cols) =
  AMV.fromVector' A.Seq size (VS.replicate (rows * cols) 0)

scaleToUnit :: Int -> Float
scaleToUnit seed = fromIntegral ((seed `mod` 29) - 14) / 14.0

maxAbsDifference :: [Float] -> [Float] -> Float
maxAbsDifference left right =
  maximum (0 : zipWith (\l r -> abs (l - r)) left right)

scalarFromIx1 :: A.Array A.S A.Ix1 Float -> Float
scalarFromIx1 array =
  case VS.toList (AM.toStorableVector array) of
    [value] -> value
    values -> error ("expected Ix1 length-1 array, got " <> show (length values))
