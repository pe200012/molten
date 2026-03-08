{-# LANGUAGE DataKinds #-}

module Molten.BLAS
  ( axpy
  , dot
  , dotInto
  , gemm
  ) where

import GHC.Stack (HasCallStack)
import Molten.BLAS.Native (axpyNative, dotIntoNative, dotNative, gemmNative)
import Molten.BLAS.Types (BlasElement, Gemm, rowMajorToNativeGemm, validateRowMajorGemmShape)
import Molten.Core.Buffer (Buffer, Location(..))
import Molten.Core.Context (Context)

axpy :: (HasCallStack, BlasElement a) => Context -> a -> Buffer 'Device a -> Buffer 'Device a -> IO ()
axpy = axpyNative

dot :: (HasCallStack, BlasElement a) => Context -> Buffer 'Device a -> Buffer 'Device a -> IO a
dot = dotNative

dotInto :: (HasCallStack, BlasElement a) => Context -> Buffer 'Device a -> Buffer 'Device a -> Buffer 'Device a -> IO ()
dotInto = dotIntoNative

gemm :: (HasCallStack, BlasElement a) => Context -> Gemm (Buffer 'Device a) a -> IO ()
gemm ctx gemmSpec = do
  validateRowMajorGemmShape "gemm" gemmSpec
  gemmNative ctx (rowMajorToNativeGemm gemmSpec)
