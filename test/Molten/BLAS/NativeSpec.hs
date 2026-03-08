{-# LANGUAGE TypeApplications #-}

module Molten.BLAS.NativeSpec (spec) where

import qualified Data.Vector.Storable as VS
import Molten.BLAS.Native (axpyNative, dotIntoNative, dotNative, gemmNative)
import Molten.BLAS.Types (NativeGemm(..), Transpose(..))
import Molten.Core.Buffer (withDeviceBuffer, withHostBuffer)
import Molten.Core.Transfer (copyD2H, copyH2D)
import Molten.Interop.Vector (readHostBufferToVector, withHostBufferFromVector)
import Molten.TestSupport (withBlasContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "axpyNative" $ do
    it "computes y := alpha * x + y on device buffers" $
      withBlasContext $ \ctx -> do
        let x = VS.fromList [1 .. 8 :: Float]
            y = VS.fromList [100, 101 .. 107 :: Float]
            expected = VS.zipWith (\xv yv -> 2 * xv + yv) x y
            n = VS.length x
        withHostBufferFromVector x $ \hostX ->
          withHostBufferFromVector y $ \hostY ->
            withDeviceBuffer @Float ctx n $ \devX ->
              withDeviceBuffer @Float ctx n $ \devY ->
                withHostBuffer @Float n $ \hostOut -> do
                  copyH2D ctx devX hostX
                  copyH2D ctx devY hostY
                  axpyNative ctx 2 devX devY
                  copyD2H ctx hostOut devY
                  out <- readHostBufferToVector hostOut
                  out `shouldBe` expected

  describe "dotNative" $ do
    it "computes a scalar dot product on device buffers" $
      withBlasContext $ \ctx -> do
        let x = VS.fromList [2, 4, 6 :: Float]
            y = VS.fromList [4, 5, 6 :: Float]
        withHostBufferFromVector x $ \hostX ->
          withHostBufferFromVector y $ \hostY ->
            withDeviceBuffer @Float ctx 3 $ \devX ->
              withDeviceBuffer @Float ctx 3 $ \devY -> do
                copyH2D ctx devX hostX
                copyH2D ctx devY hostY
                result <- dotNative ctx devX devY
                result `shouldSatisfy` approxEq 64

    it "can write the dot result into a device buffer" $
      withBlasContext $ \ctx -> do
        let x = VS.fromList [2, 4, 6 :: Float]
            y = VS.fromList [4, 5, 6 :: Float]
        withHostBufferFromVector x $ \hostX ->
          withHostBufferFromVector y $ \hostY ->
            withDeviceBuffer @Float ctx 3 $ \devX ->
              withDeviceBuffer @Float ctx 3 $ \devY ->
                withDeviceBuffer @Float ctx 1 $ \devOut ->
                  withHostBuffer @Float 1 $ \hostOut -> do
                    copyH2D ctx devX hostX
                    copyH2D ctx devY hostY
                    dotIntoNative ctx devX devY devOut
                    copyD2H ctx hostOut devOut
                    out <- readHostBufferToVector hostOut
                    VS.length out `shouldBe` 1
                    VS.head out `shouldSatisfy` approxEq 64

  describe "gemmNative" $ do
    it "computes column-major gemm" $
      withBlasContext $ \ctx -> do
        let a = VS.fromList [1, 3, 2, 4 :: Float]
            b = VS.fromList [5, 7, 6, 8 :: Float]
            c0 = VS.replicate 4 0
            expected = VS.fromList [19, 43, 22, 50 :: Float]
        withHostBufferFromVector a $ \hostA ->
          withHostBufferFromVector b $ \hostB ->
            withHostBufferFromVector c0 $ \hostC0 ->
              withDeviceBuffer @Float ctx 4 $ \devA ->
                withDeviceBuffer @Float ctx 4 $ \devB ->
                  withDeviceBuffer @Float ctx 4 $ \devC ->
                    withHostBuffer @Float 4 $ \hostOut -> do
                      copyH2D ctx devA hostA
                      copyH2D ctx devB hostB
                      copyH2D ctx devC hostC0
                      gemmNative
                        ctx
                        NativeGemm
                          { nativeGemmTransA = NoTranspose
                          , nativeGemmTransB = NoTranspose
                          , nativeGemmM = 2
                          , nativeGemmN = 2
                          , nativeGemmK = 2
                          , nativeGemmAlpha = 1
                          , nativeGemmA = devA
                          , nativeGemmLda = 2
                          , nativeGemmB = devB
                          , nativeGemmLdb = 2
                          , nativeGemmBeta = 0
                          , nativeGemmC = devC
                          , nativeGemmLdc = 2
                          }
                      copyD2H ctx hostOut devC
                      out <- readHostBufferToVector hostOut
                      out `shouldBe` expected

approxEq :: Float -> Float -> Bool
approxEq expected actual = abs (expected - actual) <= 1.0e-4
