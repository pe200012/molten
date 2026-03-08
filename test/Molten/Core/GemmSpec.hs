module Molten.Core.GemmSpec (spec) where

import Molten.BLAS.Types
  ( Gemm(..)
  , NativeGemm(..)
  , Transpose(..)
  , rowMajorToNativeGemm
  )
import Test.Hspec (Spec, describe, it, shouldBe)

spec :: Spec
spec = do
  describe "rowMajorToNativeGemm" $ do
    it "swaps operands without flipping transpose flags for row-major gemm" $ do
      let rowMajor =
            Gemm
              { gemmTransA = NoTranspose
              , gemmTransB = NoTranspose
              , gemmM = 2
              , gemmN = 3
              , gemmK = 4
              , gemmAlpha = 1.5 :: Float
              , gemmA = "A"
              , gemmLda = 4
              , gemmB = "B"
              , gemmLdb = 3
              , gemmBeta = 0.25
              , gemmC = "C"
              , gemmLdc = 3
              }
          native = rowMajorToNativeGemm rowMajor
      native
        `shouldBe` NativeGemm
          { nativeGemmTransA = NoTranspose
          , nativeGemmTransB = NoTranspose
          , nativeGemmM = 3
          , nativeGemmN = 2
          , nativeGemmK = 4
          , nativeGemmAlpha = 1.5
          , nativeGemmA = "B"
          , nativeGemmLda = 3
          , nativeGemmB = "A"
          , nativeGemmLdb = 4
          , nativeGemmBeta = 0.25
          , nativeGemmC = "C"
          , nativeGemmLdc = 3
          }

    it "preserves transpose semantics while swapping row-major operands" $ do
      let rowMajor =
            Gemm
              { gemmTransA = Transpose
              , gemmTransB = NoTranspose
              , gemmM = 5
              , gemmN = 7
              , gemmK = 11
              , gemmAlpha = 1.0 :: Float
              , gemmA = "left"
              , gemmLda = 5
              , gemmB = "right"
              , gemmLdb = 7
              , gemmBeta = 0.0
              , gemmC = "out"
              , gemmLdc = 7
              }
          native = rowMajorToNativeGemm rowMajor
      nativeGemmTransA native `shouldBe` NoTranspose
      nativeGemmTransB native `shouldBe` Transpose
      nativeGemmM native `shouldBe` 7
      nativeGemmN native `shouldBe` 5
      nativeGemmK native `shouldBe` 11
      nativeGemmA native `shouldBe` "right"
      nativeGemmB native `shouldBe` "left"
