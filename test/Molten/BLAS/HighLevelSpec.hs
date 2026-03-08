{-# LANGUAGE TypeApplications #-}

module Molten.BLAS.HighLevelSpec (spec) where

import qualified Data.Vector.Storable as VS
import Molten.BLAS (axpy, dot, gemm)
import Molten.BLAS.Types (Gemm(..), Transpose(..))
import Molten.Core.Buffer (withDeviceBuffer, withHostBuffer)
import Molten.Core.Transfer (copyD2H, copyH2D)
import Molten.Interop.Vector (readHostBufferToVector, withHostBufferFromVector)
import Molten.TestSupport (withBlasContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "axpy" $ do
    it "computes y := alpha * x + y using the high-level API" $
      withBlasContext $ \ctx -> do
        let x = VS.fromList [1 .. 8 :: Float]
            y = VS.fromList [100, 101 .. 107 :: Float]
            expected = VS.zipWith (\xv yv -> 3 * xv + yv) x y
            n = VS.length x
        withHostBufferFromVector x $ \hostX ->
          withHostBufferFromVector y $ \hostY ->
            withDeviceBuffer @Float ctx n $ \devX ->
              withDeviceBuffer @Float ctx n $ \devY ->
                withHostBuffer @Float n $ \hostOut -> do
                  copyH2D ctx devX hostX
                  copyH2D ctx devY hostY
                  axpy ctx 3 devX devY
                  copyD2H ctx hostOut devY
                  out <- readHostBufferToVector hostOut
                  out `shouldBe` expected

  describe "dot" $ do
    it "computes a host-visible scalar result" $
      withBlasContext $ \ctx -> do
        let x = VS.fromList [2, 4, 6 :: Float]
            y = VS.fromList [4, 5, 6 :: Float]
        withHostBufferFromVector x $ \hostX ->
          withHostBufferFromVector y $ \hostY ->
            withDeviceBuffer @Float ctx 3 $ \devX ->
              withDeviceBuffer @Float ctx 3 $ \devY -> do
                copyH2D ctx devX hostX
                copyH2D ctx devY hostY
                result <- dot ctx devX devY
                result `shouldSatisfy` approxEq 64

  describe "gemm" $ do
    it "computes row-major gemm while preserving row-major storage for the result" $
      withBlasContext $ \ctx -> do
        let a = VS.fromList [1, 2, 3, 4 :: Float]
            b = VS.fromList [5, 6, 7, 8 :: Float]
            c0 = VS.replicate 4 0
            expected = VS.fromList [19, 22, 43, 50 :: Float]
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
                      gemm
                        ctx
                        Gemm
                          { gemmTransA = NoTranspose
                          , gemmTransB = NoTranspose
                          , gemmM = 2
                          , gemmN = 2
                          , gemmK = 2
                          , gemmAlpha = 1
                          , gemmA = devA
                          , gemmLda = 2
                          , gemmB = devB
                          , gemmLdb = 2
                          , gemmBeta = 0
                          , gemmC = devC
                          , gemmLdc = 2
                          }
                      copyD2H ctx hostOut devC
                      out <- readHostBufferToVector hostOut
                      out `shouldBe` expected

approxEq :: Float -> Float -> Bool
approxEq expected actual = abs (expected - actual) <= 1.0e-4
