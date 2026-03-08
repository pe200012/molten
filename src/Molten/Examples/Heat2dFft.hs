module Molten.Examples.Heat2dFft
  ( Heat2dConfig(..)
  , Heat2dSummary(..)
  , buildHeatInitialField
  , buildHeatSpectralMultiplier
  , defaultHeat2dConfig
  , runHeat2dExample
  ) where

import Control.Monad (foldM)
import Data.Complex (Complex((:+)), magnitude)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Expr (Binary(..), (.*.))
import Molten.Array.Runtime (withArrayRuntime, zipWithArray)
import Molten.Array.Transfer (copyHostArrayToDevice, readDeviceArrayToHostArray)
import Molten.Examples.Common (Timed(..), measureOnce)
import Molten.FFT (fftForwardC2C, fftInverseC2C)
import Molten.FFT.Runtime (withFftRuntime)
import Molten.Core.Context (Context)

data Heat2dConfig = Heat2dConfig
  { heatNx :: !Int
  , heatNy :: !Int
  , heatSteps :: !Int
  , heatAlpha :: !Float
  , heatDt :: !Float
  }
  deriving (Eq, Show)

data Heat2dSummary = Heat2dSummary
  { heatRunPassed :: !Bool
  , heatInitialEnergy :: !Double
  , heatFinalEnergy :: !Double
  , heatRepeatedMaxAbsDiff :: !Float
  , heatMeasuredSeconds :: !Double
  }
  deriving (Eq, Show)

defaultHeat2dConfig :: Heat2dConfig
defaultHeat2dConfig =
  Heat2dConfig
    { heatNx = 1024
    , heatNy = 1024
    , heatSteps = 100
    , heatAlpha = 0.05
    , heatDt = 1.0e-3
    }

buildHeatInitialField :: Heat2dConfig -> A.Array A.S A.Ix2 (Complex Float)
buildHeatInitialField config =
  matrixFromVector (A.Sz2 (heatNx config) (heatNy config)) $ \row col ->
    let x = centeredCoord (heatNx config) row
        y = centeredCoord (heatNy config) col
        sigma = 0.18 :: Float
        amplitude = exp (negate ((x * x + y * y) / (2 * sigma * sigma)))
     in amplitude :+ 0

buildHeatSpectralMultiplier :: Heat2dConfig -> A.Array A.S A.Ix2 (Complex Float)
buildHeatSpectralMultiplier config =
  matrixFromVector (A.Sz2 (heatNx config) (heatNy config)) $ \row col ->
    let kx = fromIntegral (signedFrequency (heatNx config) row) :: Float
        ky = fromIntegral (signedFrequency (heatNy config) col) :: Float
        inverseScale = 1 / fromIntegral (heatNx config * heatNy config)
        factor = inverseScale / (1 + heatDt config * heatAlpha config * (kx * kx + ky * ky))
     in factor :+ 0

runHeat2dExample :: Context -> Heat2dConfig -> IO Heat2dSummary
runHeat2dExample ctx config = do
  let initialField = buildHeatInitialField config
      initialEnergy = heatEnergy initialField
  Timed firstOutput measuredSeconds <- measureOnce (runHeatPipeline ctx config initialField)
  secondOutput <- runHeatPipeline ctx config initialField
  let finalEnergy = heatEnergy firstOutput
      repeatedDiff = maxAbsDiff firstOutput secondOutput
      tolerance = 1.0e-4 * fromIntegral (max 1 (heatSteps config))
      passed = repeatedDiff <= tolerance && finalEnergy <= initialEnergy + 1.0e-3
  pure
    Heat2dSummary
      { heatRunPassed = passed
      , heatInitialEnergy = initialEnergy
      , heatFinalEnergy = finalEnergy
      , heatRepeatedMaxAbsDiff = repeatedDiff
      , heatMeasuredSeconds = measuredSeconds
      }

runHeatPipeline :: Context -> Heat2dConfig -> A.Array A.S A.Ix2 (Complex Float) -> IO (A.Array A.S A.Ix2 (Complex Float))
runHeatPipeline ctx config initialField =
  withArrayRuntime ctx $ \arrayRuntime ->
    withFftRuntime ctx $ \fftRuntime -> do
      initialDevice <- copyHostArrayToDevice ctx initialField
      multiplierDevice <- copyHostArrayToDevice ctx (buildHeatSpectralMultiplier config)
      finalDevice <-
        foldM
          (\currentDevice _ -> do
              frequencyDevice <- fftForwardC2C fftRuntime currentDevice
              dampedDevice <- zipWithArray arrayRuntime (Binary (\x y -> x .*. y)) frequencyDevice multiplierDevice
              fftInverseC2C fftRuntime dampedDevice
          )
          initialDevice
          [1 .. heatSteps config]
      readDeviceArrayToHostArray ctx finalDevice

matrixFromVector :: A.Sz A.Ix2 -> (Int -> Int -> Complex Float) -> A.Array A.S A.Ix2 (Complex Float)
matrixFromVector size@(A.Sz2 rows cols) elementAt =
  AMV.fromVector' A.Seq size $ VS.generate (rows * cols) $ \linearIndex ->
    let (row, col) = linearIndex `divMod` cols
     in elementAt row col

centeredCoord :: Int -> Int -> Float
centeredCoord extent ix =
  let scaled = fromIntegral ix / fromIntegral extent
   in scaled - 0.5

signedFrequency :: Int -> Int -> Int
signedFrequency extent ix
  | ix <= extent `div` 2 = ix
  | otherwise = ix - extent

heatEnergy :: A.Array A.S A.Ix2 (Complex Float) -> Double
heatEnergy array =
  VS.foldl' (\acc value -> acc + realToFrac (magnitude value) ^ (2 :: Int)) 0 (AM.toStorableVector array)

maxAbsDiff :: A.Array A.S A.Ix2 (Complex Float) -> A.Array A.S A.Ix2 (Complex Float) -> Float
maxAbsDiff left right =
  VS.maximum
    ( VS.zipWith
        (\lhs rhs -> magnitude (lhs - rhs))
        (AM.toStorableVector left)
        (AM.toStorableVector right)
    )
