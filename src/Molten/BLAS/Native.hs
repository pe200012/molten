module Molten.BLAS.Native
  ( axpyNative
  , dotIntoNative
  , dotNative
  , gemmNative
  ) where

import GHC.Stack (HasCallStack)
import Molten.BLAS.Types
  ( BlasElement
  , NativeGemm(..)
  , blasAxpy
  , blasDot
  , blasDotInto
  , blasGemm
  , validateNativeGemmShape
  )
import Molten.Core.Buffer
  ( Buffer
  , Location(..)
  , bufferDeviceId
  , bufferLength
  , withDevicePtr
  )
import Molten.Core.Context (Context, contextDeviceId, withContextBlasHandle, withContextDevice)
import Molten.Internal.Validation (ensureDeviceMatch, ensureSameLength)
import ROCm.FFI.Core.Exception (throwArgumentError)

axpyNative :: (HasCallStack, BlasElement a) => Context -> a -> Buffer 'Device a -> Buffer 'Device a -> IO ()
axpyNative ctx alpha x y = do
  ensureSameLength "axpyNative" (bufferLength x) (bufferLength y)
  ensureDeviceMatch "axpyNative" (contextDeviceId ctx) (bufferDeviceId x)
  ensureDeviceMatch "axpyNative" (bufferDeviceId x) (bufferDeviceId y)
  withContextDevice ctx $
    withContextBlasHandle ctx $ \handle ->
      withDevicePtr x $ \xPtr ->
        withDevicePtr y $ \yPtr ->
          blasAxpy handle (bufferLength x) alpha xPtr 1 yPtr 1

dotNative :: (HasCallStack, BlasElement a) => Context -> Buffer 'Device a -> Buffer 'Device a -> IO a
dotNative ctx x y = do
  ensureSameLength "dotNative" (bufferLength x) (bufferLength y)
  ensureDeviceMatch "dotNative" (contextDeviceId ctx) (bufferDeviceId x)
  ensureDeviceMatch "dotNative" (bufferDeviceId x) (bufferDeviceId y)
  withContextDevice ctx $
    withContextBlasHandle ctx $ \handle ->
      withDevicePtr x $ \xPtr ->
        withDevicePtr y $ \yPtr ->
          blasDot handle (bufferLength x) xPtr 1 yPtr 1

dotIntoNative :: (HasCallStack, BlasElement a) => Context -> Buffer 'Device a -> Buffer 'Device a -> Buffer 'Device a -> IO ()
dotIntoNative ctx x y out = do
  ensureSameLength "dotIntoNative" (bufferLength x) (bufferLength y)
  ensureDeviceMatch "dotIntoNative" (contextDeviceId ctx) (bufferDeviceId x)
  ensureDeviceMatch "dotIntoNative" (bufferDeviceId x) (bufferDeviceId y)
  ensureDeviceMatch "dotIntoNative" (bufferDeviceId y) (bufferDeviceId out)
  if bufferLength out == 1
    then
      withContextDevice ctx $
        withContextBlasHandle ctx $ \handle ->
          withDevicePtr x $ \xPtr ->
            withDevicePtr y $ \yPtr ->
              withDevicePtr out $ \outPtr ->
                blasDotInto handle (bufferLength x) xPtr 1 yPtr 1 outPtr
    else throwArgumentError "dotIntoNative" "output buffer must have length 1"

gemmNative :: (HasCallStack, BlasElement a) => Context -> NativeGemm (Buffer 'Device a) a -> IO ()
gemmNative ctx nativeGemm = do
  validateNativeGemmShape "gemmNative" nativeGemm
  ensureDeviceMatch "gemmNative" (contextDeviceId ctx) (bufferDeviceId (nativeGemmA nativeGemm))
  ensureDeviceMatch "gemmNative" (bufferDeviceId (nativeGemmA nativeGemm)) (bufferDeviceId (nativeGemmB nativeGemm))
  ensureDeviceMatch "gemmNative" (bufferDeviceId (nativeGemmB nativeGemm)) (bufferDeviceId (nativeGemmC nativeGemm))
  withContextDevice ctx $
    withContextBlasHandle ctx $ \handle ->
      withDevicePtr (nativeGemmA nativeGemm) $ \aPtr ->
        withDevicePtr (nativeGemmB nativeGemm) $ \bPtr ->
          withDevicePtr (nativeGemmC nativeGemm) $ \cPtr ->
            blasGemm
              handle
              NativeGemm
                { nativeGemmTransA = nativeGemmTransA nativeGemm
                , nativeGemmTransB = nativeGemmTransB nativeGemm
                , nativeGemmM = nativeGemmM nativeGemm
                , nativeGemmN = nativeGemmN nativeGemm
                , nativeGemmK = nativeGemmK nativeGemm
                , nativeGemmAlpha = nativeGemmAlpha nativeGemm
                , nativeGemmA = aPtr
                , nativeGemmLda = nativeGemmLda nativeGemm
                , nativeGemmB = bPtr
                , nativeGemmLdb = nativeGemmLdb nativeGemm
                , nativeGemmBeta = nativeGemmBeta nativeGemm
                , nativeGemmC = cPtr
                , nativeGemmLdc = nativeGemmLdc nativeGemm
                }
