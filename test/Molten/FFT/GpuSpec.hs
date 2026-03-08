{-# LANGUAGE TypeApplications #-}

module Molten.FFT.GpuSpec (spec) where

import Data.Complex (Complex((:+)))
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Transfer (copyHostArrayToDevice, readDeviceArrayToHostArray)
import Molten.FFT
  ( fftForwardC2C
  , fftForwardR2C
  , fftInverseC2C
  , fftInverseC2R
  )
import Molten.FFT.Runtime (withFftRuntime)
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldSatisfy)

spec :: Spec
spec = do
  describe "fftForwardC2C/fftInverseC2C" $ do
    it "roundtrips an Ix1 complex array up to the rocFFT scaling factor" $
      withGpuContext $ \ctx ->
        withFftRuntime ctx $ \runtime -> do
          let input = makeComplexVector [1 :+ 0, 2 :+ 1, 3 :+ 0, 4 :+ (-1)]
              expected = scaleComplexArray 4 input
          deviceInput <- copyHostArrayToDevice ctx input
          forward <- fftForwardC2C runtime deviceInput
          inverse <- fftInverseC2C runtime forward
          output <- readDeviceArrayToHostArray ctx inverse
          output `shouldSatisfy` approxComplexArray expected

  describe "fftForwardR2C/fftInverseC2R" $ do
    it "roundtrips an Ix1 real array up to the rocFFT scaling factor" $
      withGpuContext $ \ctx ->
        withFftRuntime ctx $ \runtime -> do
          let input = makeRealVector [0, 1, 2, 3, 4, 5, 6, 7]
              expected = scaleRealArray 8 input
          deviceInput <- copyHostArrayToDevice ctx input
          forward <- fftForwardR2C runtime deviceInput
          inverse <- fftInverseC2R runtime forward
          output <- readDeviceArrayToHostArray ctx inverse
          output `shouldSatisfy` approxRealArray expected

  describe "Ix2 complex closures" $ do
    it "roundtrips an Ix2 complex array up to the 2D scaling factor" $
      withGpuContext $ \ctx ->
        withFftRuntime ctx $ \runtime -> do
          let input =
                (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 2) ([1 :+ 0, 2 :+ 1, 3 :+ 0, 4 :+ (-1)] !!) :: A.Array A.D A.Ix2 (Complex Float))) :: A.Array A.S A.Ix2 (Complex Float)
              expected = scaleComplexArray 4 input
          deviceInput <- copyHostArrayToDevice ctx input
          forward <- fftForwardC2C runtime deviceInput
          inverse <- fftInverseC2C runtime forward
          output <- readDeviceArrayToHostArray ctx inverse
          output `shouldSatisfy` approxComplexArray expected

makeComplexVector :: [Complex Float] -> A.Array A.S A.Ix1 (Complex Float)
makeComplexVector values =
  (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 (length values)) (values !!) :: A.Array A.D A.Ix1 (Complex Float)))

makeRealVector :: [Float] -> A.Array A.S A.Ix1 Float
makeRealVector values =
  (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 (length values)) (values !!) :: A.Array A.D A.Ix1 Float))

scaleComplexArray :: A.Index ix => Float -> A.Array A.S ix (Complex Float) -> A.Array A.S ix (Complex Float)
scaleComplexArray factor arr =
  let scaled = VS.map (\value -> value * (factor :+ 0)) (AM.toStorableVector arr)
   in AMV.fromVector' A.Seq (A.size arr) scaled

scaleRealArray :: A.Index ix => Float -> A.Array A.S ix Float -> A.Array A.S ix Float
scaleRealArray factor arr =
  let scaled = VS.map (* factor) (AM.toStorableVector arr)
   in AMV.fromVector' A.Seq (A.size arr) scaled

approxComplexArray :: A.Index ix => A.Array A.S ix (Complex Float) -> A.Array A.S ix (Complex Float) -> Bool
approxComplexArray expected actual =
  let expectedVec = AM.toStorableVector expected
      actualVec = AM.toStorableVector actual
   in VS.length expectedVec == VS.length actualVec
        && VS.and (VS.zipWith approxComplex expectedVec actualVec)

approxRealArray :: A.Index ix => A.Array A.S ix Float -> A.Array A.S ix Float -> Bool
approxRealArray expected actual =
  let expectedVec = AM.toStorableVector expected
      actualVec = AM.toStorableVector actual
   in VS.length expectedVec == VS.length actualVec
        && VS.and (VS.zipWith approxFloat expectedVec actualVec)

approxComplex :: Complex Float -> Complex Float -> Bool
approxComplex (expectedReal :+ expectedImag) (actualReal :+ actualImag) =
  approxFloat expectedReal actualReal && approxFloat expectedImag actualImag

approxFloat :: Float -> Float -> Bool
approxFloat expected actual = abs (expected - actual) <= 1.0e-2
