{-# LANGUAGE ScopedTypeVariables #-}

module Molten.Core.Buffer
  ( Buffer
  , Location(..)
  , bufferDeviceId
  , bufferLength
  , bufferSizeInBytes
  , withDeviceBuffer
  , withDevicePtr
  , withHostBuffer
  , withHostPtr
  , withPinnedBuffer
  , withPinnedHostPtr
  ) where

import Control.Exception (bracket)
import Data.Proxy (Proxy(..))
import qualified Foreign.Concurrent as FC
import Foreign.C.Types (CSize)
import Foreign.ForeignPtr (finalizeForeignPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.Storable (Storable, sizeOf)
import GHC.Stack (HasCallStack)
import Molten.Core.Context (Context, contextDeviceId, withContextDevice)
import Molten.Core.Types (Buffer(..), DeviceId, Location(..))
import Molten.Internal.Validation (ensureNonNegative)
import ROCm.FFI.Core.Exception (throwArgumentError)
import ROCm.FFI.Core.Types (DevicePtr(..), HostPtr(..), PinnedHostPtr(..))
import ROCm.HIP (hipFree, hipHostFree, hipHostMallocBytes, hipMallocBytes)

withHostBuffer :: forall a r. (HasCallStack, Storable a) => Int -> (Buffer 'Host a -> IO r) -> IO r
withHostBuffer length_ = bracket (allocateHostBuffer length_) freeBuffer

withPinnedBuffer :: forall a r. (HasCallStack, Storable a) => Int -> (Buffer 'PinnedHost a -> IO r) -> IO r
withPinnedBuffer length_ = bracket (allocatePinnedBuffer length_) freeBuffer

withDeviceBuffer :: forall a r. (HasCallStack, Storable a) => Context -> Int -> (Buffer 'Device a -> IO r) -> IO r
withDeviceBuffer ctx length_ = bracket (allocateDeviceBuffer ctx length_) freeBuffer

bufferLength :: Buffer loc a -> Int
bufferLength buffer =
  case buffer of
    HostBuffer _ len -> len
    PinnedHostBuffer _ len -> len
    DeviceBuffer _ _ len -> len

bufferDeviceId :: Buffer 'Device a -> DeviceId
bufferDeviceId (DeviceBuffer deviceId _ _) = deviceId

bufferSizeInBytes :: forall loc a. Storable a => Buffer loc a -> CSize
bufferSizeInBytes = uncheckedByteCount (Proxy :: Proxy a) . bufferLength

withHostPtr :: Buffer 'Host a -> (HostPtr a -> IO r) -> IO r
withHostPtr (HostBuffer foreignPtr _) action =
  withForeignPtr foreignPtr (action . HostPtr)

withPinnedHostPtr :: Buffer 'PinnedHost a -> (PinnedHostPtr a -> IO r) -> IO r
withPinnedHostPtr (PinnedHostBuffer foreignPtr _) action =
  withForeignPtr foreignPtr (action . PinnedHostPtr)

withDevicePtr :: Buffer 'Device a -> (DevicePtr a -> IO r) -> IO r
withDevicePtr (DeviceBuffer _ foreignPtr _) action =
  withForeignPtr foreignPtr (action . DevicePtr)

allocateHostBuffer :: forall a. (HasCallStack, Storable a) => Int -> IO (Buffer 'Host a)
allocateHostBuffer length_ = do
  ensureNonNegative "withHostBuffer" "length" length_
  foreignPtr <- mallocForeignPtrArray length_
  pure (HostBuffer foreignPtr length_)

allocatePinnedBuffer :: forall a. (HasCallStack, Storable a) => Int -> IO (Buffer 'PinnedHost a)
allocatePinnedBuffer length_ = do
  bytes <- byteCountForLength (Proxy :: Proxy a) "withPinnedBuffer" length_
  rawPtr@(PinnedHostPtr ptr) <- hipHostMallocBytes (allocationByteCount bytes)
  foreignPtr <- FC.newForeignPtr ptr (hipHostFree rawPtr)
  pure (PinnedHostBuffer foreignPtr length_)

allocateDeviceBuffer :: forall a. (HasCallStack, Storable a) => Context -> Int -> IO (Buffer 'Device a)
allocateDeviceBuffer ctx length_ = do
  bytes <- byteCountForLength (Proxy :: Proxy a) "withDeviceBuffer" length_
  withContextDevice ctx $ do
    rawPtr@(DevicePtr ptr) <- hipMallocBytes (allocationByteCount bytes)
    foreignPtr <- FC.newForeignPtr ptr (hipFree rawPtr)
    pure (DeviceBuffer (contextDeviceId ctx) foreignPtr length_)

freeBuffer :: Buffer loc a -> IO ()
freeBuffer buffer =
  case buffer of
    HostBuffer foreignPtr _ -> finalizeForeignPtr foreignPtr
    PinnedHostBuffer foreignPtr _ -> finalizeForeignPtr foreignPtr
    DeviceBuffer _ foreignPtr _ -> finalizeForeignPtr foreignPtr

byteCountForLength :: forall a proxy. (HasCallStack, Storable a) => proxy a -> String -> Int -> IO CSize
byteCountForLength _ functionName length_ = do
  ensureNonNegative functionName "length" length_
  let totalBytes = toInteger length_ * toInteger (sizeOf (undefined :: a))
      maxBytes = toInteger (maxBound :: CSize)
  if totalBytes > maxBytes
    then throwArgumentError functionName "buffer size overflow"
    else pure (fromInteger totalBytes)

uncheckedByteCount :: forall a proxy. Storable a => proxy a -> Int -> CSize
uncheckedByteCount _ length_ =
  fromIntegral (length_ * sizeOf (undefined :: a))

allocationByteCount :: CSize -> CSize
allocationByteCount bytes
  | bytes == 0 = 1
  | otherwise = bytes
