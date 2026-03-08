module Molten.Core.Transfer
  ( copyD2D
  , copyD2H
  , copyD2HAsync
  , copyH2D
  , copyH2DAsync
  ) where

import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Core.Buffer
  ( Buffer
  , Location(..)
  , bufferDeviceId
  , bufferLength
  , bufferSizeInBytes
  , withDevicePtr
  , withHostPtr
  , withPinnedHostPtr
  )
import Molten.Core.Context (Context, contextDeviceId, withContextDevice)
import Molten.Core.Future (GpuFuture, makeFutureFromStream)
import Molten.Core.Stream (Stream, streamDeviceId, withRawHipStream)
import Molten.Internal.Device (withDeviceId)
import Molten.Internal.Validation (ensureDeviceMatch, ensureSameLength)
import ROCm.HIP (hipMemcpyD2D, hipMemcpyD2H, hipMemcpyD2HAsync, hipMemcpyH2D, hipMemcpyH2DAsync)

copyH2D :: (HasCallStack, Storable a) => Context -> Buffer 'Device a -> Buffer 'Host a -> IO ()
copyH2D ctx deviceBuffer hostBuffer = do
  ensureDeviceMatch "copyH2D" (contextDeviceId ctx) (bufferDeviceId deviceBuffer)
  ensureSameLength "copyH2D" (bufferLength deviceBuffer) (bufferLength hostBuffer)
  withContextDevice ctx $
    withDevicePtr deviceBuffer $ \devicePtr ->
      withHostPtr hostBuffer $ \hostPtr ->
        hipMemcpyH2D devicePtr hostPtr (bufferSizeInBytes deviceBuffer)

copyD2H :: (HasCallStack, Storable a) => Context -> Buffer 'Host a -> Buffer 'Device a -> IO ()
copyD2H ctx hostBuffer deviceBuffer = do
  ensureDeviceMatch "copyD2H" (contextDeviceId ctx) (bufferDeviceId deviceBuffer)
  ensureSameLength "copyD2H" (bufferLength hostBuffer) (bufferLength deviceBuffer)
  withContextDevice ctx $
    withHostPtr hostBuffer $ \hostPtr ->
      withDevicePtr deviceBuffer $ \devicePtr ->
        hipMemcpyD2H hostPtr devicePtr (bufferSizeInBytes deviceBuffer)

copyD2D :: (HasCallStack, Storable a) => Context -> Buffer 'Device a -> Buffer 'Device a -> IO ()
copyD2D ctx dst src = do
  ensureDeviceMatch "copyD2D" (contextDeviceId ctx) (bufferDeviceId dst)
  ensureDeviceMatch "copyD2D" (bufferDeviceId dst) (bufferDeviceId src)
  ensureSameLength "copyD2D" (bufferLength dst) (bufferLength src)
  withContextDevice ctx $
    withDevicePtr dst $ \dstPtr ->
      withDevicePtr src $ \srcPtr ->
        hipMemcpyD2D dstPtr srcPtr (bufferSizeInBytes dst)

copyH2DAsync :: (HasCallStack, Storable a) => Stream -> Buffer 'Device a -> Buffer 'PinnedHost a -> IO (GpuFuture ())
copyH2DAsync stream deviceBuffer pinnedBuffer = do
  ensureDeviceMatch "copyH2DAsync" (streamDeviceId stream) (bufferDeviceId deviceBuffer)
  ensureSameLength "copyH2DAsync" (bufferLength deviceBuffer) (bufferLength pinnedBuffer)
  withDeviceId (streamDeviceId stream) $
    withDevicePtr deviceBuffer $ \devicePtr ->
      withPinnedHostPtr pinnedBuffer $ \pinnedPtr ->
        withRawHipStream stream $ \rawStream -> do
          hipMemcpyH2DAsync devicePtr pinnedPtr (bufferSizeInBytes deviceBuffer) rawStream
          makeFutureFromStream stream ()

copyD2HAsync :: (HasCallStack, Storable a) => Stream -> Buffer 'PinnedHost a -> Buffer 'Device a -> IO (GpuFuture ())
copyD2HAsync stream pinnedBuffer deviceBuffer = do
  ensureDeviceMatch "copyD2HAsync" (streamDeviceId stream) (bufferDeviceId deviceBuffer)
  ensureSameLength "copyD2HAsync" (bufferLength pinnedBuffer) (bufferLength deviceBuffer)
  withDeviceId (streamDeviceId stream) $
    withPinnedHostPtr pinnedBuffer $ \pinnedPtr ->
      withDevicePtr deviceBuffer $ \devicePtr ->
        withRawHipStream stream $ \rawStream -> do
          hipMemcpyD2HAsync pinnedPtr devicePtr (bufferSizeInBytes deviceBuffer) rawStream
          makeFutureFromStream stream ()
