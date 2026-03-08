{-# LANGUAGE TypeApplications #-}

module Molten.BLAS.ArraySpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import Molten.Array.Transfer (copyHostArrayToDevice, readDeviceArrayToHostArray)
import Molten.BLAS (MatrixGemm(..), axpyVector, dotVector, gemmMatrix)
import Molten.BLAS.Types (Transpose(..))
import Molten.TestSupport (withBlasContext)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy, shouldThrow)

spec :: Spec
spec = do
  describe "axpyVector" $ do
    it "rejects vectors with different shapes" $
      withBlasContext $ \ctx -> do
        let x = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 3) (fromIntegral @Int @Float) :: A.Array A.D A.Ix1 Float)
            y = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 4) (fromIntegral @Int @Float) :: A.Array A.D A.Ix1 Float)
        devX <- copyHostArrayToDevice ctx x
        devY <- copyHostArrayToDevice ctx y
        axpyVector ctx 3 devX devY `shouldThrow` isArgumentError "axpyVector" "same length"

  describe "dotVector" $ do
    it "computes a host-visible scalar result for Ix1 device arrays" $
      withBlasContext $ \ctx -> do
        let x = (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 3) ([2, 4, 6 :: Float] !!) :: A.Array A.D A.Ix1 Float)) :: A.Array A.S A.Ix1 Float
            y = (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 3) ([4, 5, 6 :: Float] !!) :: A.Array A.D A.Ix1 Float)) :: A.Array A.S A.Ix1 Float
        devX <- copyHostArrayToDevice ctx x
        devY <- copyHostArrayToDevice ctx y
        result <- dotVector ctx devX devY
        result `shouldSatisfy` approxEq 64

  describe "gemmMatrix" $ do
    it "infers row-major gemm dimensions from Ix2 device arrays" $
      withBlasContext $ \ctx -> do
        let a = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 2) ([1, 2, 3, 4 :: Float] !!) :: A.Array A.D A.Ix2 Float)
            b = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 2) ([5, 6, 7, 8 :: Float] !!) :: A.Array A.D A.Ix2 Float)
            c0 = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 2) (const 0) :: A.Array A.D A.Ix2 Float)
            expected = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 2) ([19, 22, 43, 50 :: Float] !!) :: A.Array A.D A.Ix2 Float)
        devA <- copyHostArrayToDevice ctx a
        devB <- copyHostArrayToDevice ctx b
        devC <- copyHostArrayToDevice ctx c0
        gemmMatrix
          ctx
          MatrixGemm
            { matrixGemmTransA = NoTranspose
            , matrixGemmTransB = NoTranspose
            , matrixGemmAlpha = 1
            , matrixGemmA = devA
            , matrixGemmB = devB
            , matrixGemmBeta = 0
            , matrixGemmC = devC
            }
        out <- readDeviceArrayToHostArray ctx devC
        AM.toStorableVector out `shouldBe` AM.toStorableVector expected

approxEq :: Float -> Float -> Bool
approxEq expected actual = abs (expected - actual) <= 1.0e-4

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
