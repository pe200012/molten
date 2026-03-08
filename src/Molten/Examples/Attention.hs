{-# LANGUAGE TypeApplications #-}

module Molten.Examples.Attention
  ( AttentionConfig(..)
  , AttentionInputs(..)
  , AttentionShadowSummary(..)
  , AttentionStressSummary(..)
  , buildAttentionProgram
  , defaultAttentionShadowConfig
  , defaultAttentionStressConfig
  , mkDeterministicAttentionInputs
  , runAttentionShadowCheck
  , runAttentionStress
  ) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Expr (Binary(..), Unary(..), constant, (.*.))
import Molten.Array.Program
  ( MatrixGemmValue(..)
  , Program
  , Value
  , buildProgram
  , gemmMatrixP
  , inputArray
  , mapExpr
  , runProgram
  , softmaxRowsP
  , withProgramRuntime
  )
import Molten.Array.Transfer (readDeviceArrayToHostArray)
import Molten.BLAS.Types (Transpose(..))
import Molten.Core.Context (Context)
import Molten.Examples.Common (Timed(..), measureOnce)
import Molten.Reference (runProgramCpu)

data AttentionConfig = AttentionConfig
  { attentionTokens :: !Int
  , attentionModelWidth :: !Int
  , attentionValueWidth :: !Int
  }
  deriving (Eq, Show)

data AttentionInputs = AttentionInputs
  { attentionQueryMatrix :: !(A.Array A.S A.Ix2 Float)
  , attentionKeyMatrix :: !(A.Array A.S A.Ix2 Float)
  , attentionValueMatrix :: !(A.Array A.S A.Ix2 Float)
  }
  deriving (Eq, Show)

data AttentionShadowSummary = AttentionShadowSummary
  { attentionShadowPassed :: !Bool
  , attentionShadowOutputMaxAbsError :: !Float
  , attentionShadowProbMaxAbsError :: !Float
  , attentionShadowRowSumMaxDrift :: !Float
  }
  deriving (Eq, Show)

data AttentionStressSummary = AttentionStressSummary
  { attentionStressPassed :: !Bool
  , attentionStressSeconds :: !Double
  , attentionStressOutputSize :: !(A.Sz A.Ix2)
  , attentionStressChecksum :: !Float
  , attentionStressRowSumMaxDrift :: !Float
  , attentionStressShadow :: !AttentionShadowSummary
  }
  deriving (Eq, Show)

defaultAttentionShadowConfig :: AttentionConfig
defaultAttentionShadowConfig =
  AttentionConfig
    { attentionTokens = 64
    , attentionModelWidth = 64
    , attentionValueWidth = 64
    }

defaultAttentionStressConfig :: AttentionConfig
defaultAttentionStressConfig =
  AttentionConfig
    { attentionTokens = 2048
    , attentionModelWidth = 128
    , attentionValueWidth = 128
    }

mkDeterministicAttentionInputs :: AttentionConfig -> AttentionInputs
mkDeterministicAttentionInputs config =
  AttentionInputs
    { attentionQueryMatrix = buildMatrix (A.Sz2 tokens modelWidth) queryValue
    , attentionKeyMatrix = buildMatrix (A.Sz2 tokens modelWidth) keyValue
    , attentionValueMatrix = buildMatrix (A.Sz2 tokens valueWidth) valueValue
    }
  where
    tokens = attentionTokens config
    modelWidth = attentionModelWidth config
    valueWidth = attentionValueWidth config
    queryValue row col = 0.20 * scaleToUnit (row * 13 + col * 17 + 1)
    keyValue row col = 0.20 * scaleToUnit (row * 7 + col * 11 + 3)
    valueValue row col = 0.25 * scaleToUnit (row * 19 + col * 5 + 2)

buildAttentionProgram :: AttentionInputs -> IO (Program (Value A.Ix2 Float, Value A.Ix2 Float))
buildAttentionProgram inputs =
  buildProgram $ do
    query <- inputArray (attentionQueryMatrix inputs)
    key <- inputArray (attentionKeyMatrix inputs)
    value <- inputArray (attentionValueMatrix inputs)
    scoreInit <- inputArray (zeroMatrixLike (A.Sz2 tokens tokens))
    outputInit <- inputArray (zeroMatrixLike (A.Sz2 tokens valueWidth))
    scores <-
      gemmMatrixP
        MatrixGemmValue
          { matrixGemmValueTransA = NoTranspose
          , matrixGemmValueTransB = Transpose
          , matrixGemmValueAlpha = 1
          , matrixGemmValueA = query
          , matrixGemmValueB = key
          , matrixGemmValueBeta = 0
          , matrixGemmValueC = scoreInit
          }
    scaledScores <- mapExpr (scaleUnary scaleFactor) scores
    probabilities <- softmaxRowsP scaledScores
    output <-
      gemmMatrixP
        MatrixGemmValue
          { matrixGemmValueTransA = NoTranspose
          , matrixGemmValueTransB = NoTranspose
          , matrixGemmValueAlpha = 1
          , matrixGemmValueA = probabilities
          , matrixGemmValueB = value
          , matrixGemmValueBeta = 0
          , matrixGemmValueC = outputInit
          }
    pure (output, probabilities)
  where
    A.Sz2 tokens modelWidth = A.size (attentionQueryMatrix inputs)
    A.Sz2 _ valueWidth = A.size (attentionValueMatrix inputs)
    scaleFactor = recip (sqrt (fromIntegral modelWidth))

runAttentionShadowCheck :: Context -> AttentionInputs -> Float -> IO AttentionShadowSummary
runAttentionShadowCheck ctx inputs tolerance = do
  program <- buildAttentionProgram inputs
  (cpuOutput, cpuProbabilities) <- runProgramCpu program
  withProgramRuntime ctx $ \runtime -> do
    (gpuOutputDevice, gpuProbabilitiesDevice) <- runProgram runtime program
    gpuOutput <- readDeviceArrayToHostArray ctx gpuOutputDevice
    gpuProbabilities <- readDeviceArrayToHostArray ctx gpuProbabilitiesDevice
    let outputError = maxAbsDifference (AM.toStorableVector cpuOutput) (AM.toStorableVector gpuOutput)
        probabilityError = maxAbsDifference (AM.toStorableVector cpuProbabilities) (AM.toStorableVector gpuProbabilities)
        rowSumDrift = maxRowSumDrift gpuProbabilities
    pure
      AttentionShadowSummary
        { attentionShadowPassed = outputError <= tolerance && probabilityError <= tolerance && rowSumDrift <= tolerance * 10
        , attentionShadowOutputMaxAbsError = outputError
        , attentionShadowProbMaxAbsError = probabilityError
        , attentionShadowRowSumMaxDrift = rowSumDrift
        }

runAttentionStress :: Context -> AttentionConfig -> IO AttentionStressSummary
runAttentionStress ctx config = do
  let shadowInputs = mkDeterministicAttentionInputs defaultAttentionShadowConfig
      stressInputs = mkDeterministicAttentionInputs config
  shadowSummary <- runAttentionShadowCheck ctx shadowInputs 1.0e-3
  Timed {timedValue = (outputHost, probabilityHost), timedSeconds = runtimeSeconds} <-
    measureOnce $ do
      program <- buildAttentionProgram stressInputs
      withProgramRuntime ctx $ \runtime -> do
        (outputDevice, probabilityDevice) <- runProgram runtime program
        outputHost <- readDeviceArrayToHostArray ctx outputDevice
        probabilityHost <- readDeviceArrayToHostArray ctx probabilityDevice
        pure (outputHost, probabilityHost)
  pure
    AttentionStressSummary
      { attentionStressPassed = attentionShadowPassed shadowSummary && maxRowSumDrift probabilityHost <= 5.0e-3
      , attentionStressSeconds = runtimeSeconds
      , attentionStressOutputSize = A.size outputHost
      , attentionStressChecksum = VS.sum (AM.toStorableVector outputHost)
      , attentionStressRowSumMaxDrift = maxRowSumDrift probabilityHost
      , attentionStressShadow = shadowSummary
      }

scaleUnary :: Float -> Unary Float Float
scaleUnary factor = Unary (\x -> x .*. constant factor)

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

maxAbsDifference :: VS.Vector Float -> VS.Vector Float -> Float
maxAbsDifference left right =
  if VS.length left /= VS.length right
    then 1 / 0
    else VS.maximum (VS.cons 0 (VS.zipWith (\l r -> abs (l - r)) left right))

maxRowSumDrift :: A.Array A.S A.Ix2 Float -> Float
maxRowSumDrift probabilities =
  let A.Sz2 rows cols = A.size probabilities
      values = AM.toStorableVector probabilities
      rowDrift rowIndex =
        abs (VS.sum (VS.slice (rowIndex * cols) cols values) - 1)
   in if rows == 0
        then 0
        else maximum [rowDrift rowIndex | rowIndex <- [0 .. rows - 1]]
