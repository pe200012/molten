{-# LANGUAGE TypeApplications #-}

module Molten.Reference.BlasSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.BLAS.Types (Transpose(..))
import Molten.Reference
  ( MatrixGemmRef(..)
  , axpyVectorRef
  , dotVectorRef
  , gemmMatrixRef
  )
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "axpyVectorRef" $ do
    it "computes y := alpha * x + y" $ do
      let x = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Float])
          y = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [10, 20, 30, 40 :: Float])
      output <- axpyVectorRef 2 x y
      AM.toStorableVector output `shouldBe` VS.fromList [12, 24, 36, 48 :: Float]

  describe "dotVectorRef" $ do
    it "computes a vector dot product" $ do
      let x = AMV.fromVector' A.Seq (A.Sz1 3) (VS.fromList [2, 4, 6 :: Float])
          y = AMV.fromVector' A.Seq (A.Sz1 3) (VS.fromList [4, 5, 6 :: Float])
      result <- dotVectorRef x y
      result `shouldBe` 64

  describe "gemmMatrixRef" $ do
    it "computes row-major GEMM with NoTranspose operands" $ do
      let a = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [1, 2, 3, 4 :: Float])
          b = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [5, 6, 7, 8 :: Float])
          c = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [0, 0, 0, 0 :: Float])
      output <-
        gemmMatrixRef
          MatrixGemmRef
            { matrixGemmRefTransA = NoTranspose
            , matrixGemmRefTransB = NoTranspose
            , matrixGemmRefAlpha = 1
            , matrixGemmRefA = a
            , matrixGemmRefB = b
            , matrixGemmRefBeta = 0
            , matrixGemmRefC = c
            }
      AM.toStorableVector output `shouldBe` VS.fromList [19, 22, 43, 50 :: Float]

    it "rejects matrix inner-dimension mismatches" $ do
      let a = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1 .. 6 :: Float])
          b = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [1 .. 4 :: Float])
          c = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [0, 0, 0, 0 :: Float])
      gemmMatrixRef
        MatrixGemmRef
          { matrixGemmRefTransA = NoTranspose
          , matrixGemmRefTransB = NoTranspose
          , matrixGemmRefAlpha = 1
          , matrixGemmRefA = a
          , matrixGemmRefB = b
          , matrixGemmRefBeta = 0
          , matrixGemmRefC = c
          }
        `shouldThrow` isArgumentError "gemmMatrixRef" "inner dimensions"

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
