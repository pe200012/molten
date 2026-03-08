{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Molten.Array.RuntimeSpec (spec) where

import Control.Exception (SomeException, displayException, try)
import qualified Data.ByteString
import Data.Complex (Complex((:+)))
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Foreign.Storable (Storable)
import Molten.Array.Expr
  ( Binary(..)
  , Exp
  , Unary(..)
  , cast
  , constant
  , (.+.)
  , (.*.)
  )
import Molten.Array.Runtime
  ( arrayRuntimeKernelCount
  , fillArray
  , mapArray
  , reduceAllArray
  , withArrayRuntime
  , zipWithArray
  )
import Molten.Array.Transfer (copyHostArrayToDevice, readDeviceArrayToHostArray)
import Molten.Core.Context (contextDeviceId)
import Molten.Internal.HIPRTC (compileHipKernel)
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "compileHipKernel" $ do
    it "reports the kernel name and compiler log when compilation fails" $
      withGpuContext $ \ctx -> do
        result <-
          try
            ( compileHipKernel
                (contextDeviceId ctx)
                "extern \"C\" __global__ void broken_kernel( {"
                "broken_kernel"
                []
            )
            :: IO (Either SomeException Data.ByteString.ByteString)
        case result of
          Left err -> do
            let message = displayException err
            message `shouldSatisfy` ("broken_kernel" `isInfixOf`)
            message `shouldSatisfy` (\text -> "hiprtc" `isInfixOf` text || "error:" `isInfixOf` text)
          Right _ -> error "compileHipKernel should have failed"

  describe "kernel cache" $ do
    it "reuses a compiled map kernel for the same signature and expression" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          let input = vector1 [1 .. 8 :: Float]
          inputDevice <- copyHostArrayToDevice ctx input
          _ <- mapArray runtime (Unary (\x -> x .+. constant 1.0)) inputDevice
          _ <- mapArray runtime (Unary (\x -> x .+. constant 1.0)) inputDevice
          kernelCount <- arrayRuntimeKernelCount runtime
          kernelCount `shouldBe` 1

  describe "fillArray/mapArray/zipWithArray/reduceAllArray" $ do
    it "fills a device array with a constant value" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          output <- fillArray runtime (7 :: Int32) (A.Sz1 5)
          hostArray <- readDeviceArrayToHostArray ctx output
          AM.toStorableVector hostArray `shouldBe` VS.fromList [7, 7, 7, 7, 7 :: Int32]

    it "maps a unary expression on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          inputDevice <- copyHostArrayToDevice ctx (vector1 [1 .. 8 :: Float])
          output <- mapArray runtime (Unary (\x -> x .+. constant 3.0)) inputDevice
          hostArray <- readDeviceArrayToHostArray ctx output
          AM.toStorableVector hostArray `shouldBe` VS.fromList [4 .. 11 :: Float]

    it "zips two arrays with a binary expression on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          left <- copyHostArrayToDevice ctx (vector1 [1, 2, 3, 4 :: Float])
          right <- copyHostArrayToDevice ctx (vector1 [10, 20, 30, 40 :: Float])
          output <- zipWithArray runtime (Binary (\x y -> x .+. y)) left right
          hostArray <- readDeviceArrayToHostArray ctx output
          AM.toStorableVector hostArray `shouldBe` VS.fromList [11, 22, 33, 44 :: Float]

    it "reduces an array to an Ix1 length-1 result on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          inputDevice <- copyHostArrayToDevice ctx (vector1 [1 .. 32 :: Float])
          output <- reduceAllArray runtime (Binary (\x y -> x .+. y)) 0 inputDevice
          hostArray <- readDeviceArrayToHostArray ctx output
          A.size hostArray `shouldBe` A.Sz1 1
          AM.toStorableVector hostArray `shouldBe` VS.fromList [sum [1 .. 32 :: Float]]

    it "can compose map and reduce across different scalar types" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          inputDevice <- copyHostArrayToDevice ctx (vector1 [1, 2, 3, 4 :: Int32])
          mapped <- mapArray runtime (Unary (\x -> cast x .*. constant 0.5 :: Exp Float)) inputDevice
          output <- reduceAllArray runtime (Binary (\x y -> x .+. y)) 0 mapped
          hostArray <- readDeviceArrayToHostArray ctx output
          AM.toStorableVector hostArray `shouldBe` VS.fromList [5 :: Float]

    it "zips complex arrays with pointwise multiplication on the GPU" $
      withGpuContext $ \ctx ->
        withArrayRuntime ctx $ \runtime -> do
          left <- copyHostArrayToDevice ctx (vector1 [1 :+ 2, 3 :+ 4 :: Complex Float])
          right <- copyHostArrayToDevice ctx (vector1 [5 :+ 6, 7 :+ 8 :: Complex Float])
          output <- zipWithArray runtime (Binary (\x y -> x .*. y)) left right
          hostArray <- readDeviceArrayToHostArray ctx output
          AM.toStorableVector hostArray `shouldBe` VS.fromList [(-7) :+ 16, (-11) :+ 52 :: Complex Float]

vector1 :: Storable a => [a] -> A.Array A.S A.Ix1 a
vector1 values =
  AMV.fromVector' A.Seq (A.Sz1 (length values)) (VS.fromList values)
