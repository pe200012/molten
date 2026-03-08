module Molten.Array.Axis2DSpec (spec) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Data.Int (Int32)
import Molten.Array.Axis2D
  ( Axis2(..)
  , broadcastCols
  , broadcastRows
  , maxRows
  , sumCols
  , sumRows
  )
import Molten.Array.Runtime (withArrayRuntime)
import Molten.Array.Transfer (copyHostArrayToDevice, readDeviceArrayToHostArray)
import Molten.Reference (broadcastColsRef, broadcastRowsRef, maxRowsRef, sumColsRef, sumRowsRef)
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe)

spec :: Spec
spec =
  describe "Molten.Array.Axis2D" $ do
    it "loads the Axis2D module" $ do
      AxisRows `shouldBe` AxisRows

    it "computes row sums on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          let input = matrix2 2 3 [1, 2, 3, 4, 5, 6 :: Int32]
          inputDevice <- copyHostArrayToDevice ctx input
          outputDevice <- sumRows runtime inputDevice
          output <- readDeviceArrayToHostArray ctx outputDevice
          expected <- sumRowsRef input
          AM.toStorableVector output `shouldBe` AM.toStorableVector expected

    it "computes column sums on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          let input = matrix2 2 3 [1, 2, 3, 4, 5, 6 :: Int32]
          inputDevice <- copyHostArrayToDevice ctx input
          outputDevice <- sumCols runtime inputDevice
          output <- readDeviceArrayToHostArray ctx outputDevice
          expected <- sumColsRef input
          AM.toStorableVector output `shouldBe` AM.toStorableVector expected

    it "computes row maxima on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          let input = matrix2 2 3 [1, 9, 3, 4, 2, 6 :: Int32]
          inputDevice <- copyHostArrayToDevice ctx input
          outputDevice <- maxRows runtime inputDevice
          output <- readDeviceArrayToHostArray ctx outputDevice
          expected <- maxRowsRef input
          AM.toStorableVector output `shouldBe` AM.toStorableVector expected

    it "broadcasts row summaries on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          let input = vector1 [10, 20 :: Int32]
          inputDevice <- copyHostArrayToDevice ctx input
          outputDevice <- broadcastRows runtime 3 inputDevice
          output <- readDeviceArrayToHostArray ctx outputDevice
          expected <- broadcastRowsRef 3 input
          AM.toStorableVector output `shouldBe` AM.toStorableVector expected

    it "broadcasts column summaries on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          let input = vector1 [10, 20, 30 :: Int32]
          inputDevice <- copyHostArrayToDevice ctx input
          outputDevice <- broadcastCols runtime 2 inputDevice
          output <- readDeviceArrayToHostArray ctx outputDevice
          expected <- broadcastColsRef 2 input
          AM.toStorableVector output `shouldBe` AM.toStorableVector expected

vector1 :: VS.Storable a => [a] -> A.Array A.S A.Ix1 a
vector1 values =
  AMV.fromVector' A.Seq (A.Sz1 (length values)) (VS.fromList values)

matrix2 :: VS.Storable a => Int -> Int -> [a] -> A.Array A.S A.Ix2 a
matrix2 rows cols values =
  AMV.fromVector' A.Seq (A.Sz2 rows cols) (VS.fromList values)
