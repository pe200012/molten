{-# LANGUAGE DataKinds #-}

module Molten.Array.Transfer
  ( cloneDeviceArray
  , copyHostArrayToDevice
  , copyPinnedArrayToDeviceAsync
  , readDeviceArrayToHostArray
  , readDeviceArrayToPinnedArrayAsync
  , reshapeDeviceArray
  ) where

import qualified Data.Massiv.Array as A
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Array.Device
  ( DeviceArray
  , deviceArrayBuffer
  , deviceArraySize
  , mkDeviceArray
  )
import Molten.Core.Buffer
  ( destroyBuffer
  , newDeviceBuffer
  , newDeviceBufferOn
  , newHostBuffer
  , newPinnedBuffer
  )
import Molten.Core.Context (Context)
import Molten.Core.Future (GpuFuture, makeFutureFromStreamWith)
import Molten.Core.Stream (Stream, streamDeviceId)
import Molten.Core.Transfer (copyD2D, copyD2H, copyD2HAsync, copyH2D, copyH2DAsync)
import Molten.Interop.Massiv
  ( readHostBufferToArray
  , readPinnedBufferToArray
  , withHostBufferFromArray
  , withPinnedBufferFromArray
  )
import ROCm.FFI.Core.Exception (throwArgumentError)

reshapeDeviceArray :: (HasCallStack, A.Index ix, A.Index ix') => A.Sz ix' -> DeviceArray ix a -> IO (DeviceArray ix' a)
reshapeDeviceArray targetSize deviceArray = do
  let sourceElemCount = A.totalElem (deviceArraySize deviceArray)
      targetElemCount = A.totalElem targetSize
  if sourceElemCount == targetElemCount
    then mkDeviceArray targetSize (deviceArrayBuffer deviceArray)
    else
      throwArgumentError
        "reshapeDeviceArray"
        ("source totalElem = " <> show sourceElemCount <> ", target totalElem = " <> show targetElemCount)

cloneDeviceArray :: (HasCallStack, A.Index ix, Storable a) => Context -> DeviceArray ix a -> IO (DeviceArray ix a)
cloneDeviceArray ctx deviceArray = do
  deviceBuffer <- newDeviceBuffer ctx (A.totalElem (deviceArraySize deviceArray))
  copyD2D ctx deviceBuffer (deviceArrayBuffer deviceArray)
  mkDeviceArray (deviceArraySize deviceArray) deviceBuffer

copyHostArrayToDevice :: (HasCallStack, A.Load r ix a, Storable a) => Context -> A.Array r ix a -> IO (DeviceArray ix a)
copyHostArrayToDevice ctx arr =
  withHostBufferFromArray arr $ \size hostBuffer -> do
    deviceBuffer <- newDeviceBuffer ctx (A.totalElem size)
    copyH2D ctx deviceBuffer hostBuffer
    mkDeviceArray size deviceBuffer

copyPinnedArrayToDeviceAsync :: (HasCallStack, A.Load r ix a, Storable a) => Stream -> A.Array r ix a -> IO (GpuFuture (DeviceArray ix a))
copyPinnedArrayToDeviceAsync stream arr =
  withPinnedBufferFromArray arr $ \size pinnedBuffer -> do
    deviceBuffer <- newDeviceBufferOn (streamDeviceId stream) (A.totalElem size)
    _ <- copyH2DAsync stream deviceBuffer pinnedBuffer
    makeFutureFromStreamWith stream (mkDeviceArray size deviceBuffer)

readDeviceArrayToHostArray :: (HasCallStack, A.Index ix, Storable a) => Context -> DeviceArray ix a -> IO (A.Array A.S ix a)
readDeviceArrayToHostArray ctx deviceArray = do
  hostBuffer <- newHostBuffer (A.totalElem (deviceArraySize deviceArray))
  copyD2H ctx hostBuffer (deviceArrayBuffer deviceArray)
  result <- readHostBufferToArray (deviceArraySize deviceArray) hostBuffer
  destroyBuffer hostBuffer
  pure result

readDeviceArrayToPinnedArrayAsync :: (HasCallStack, A.Index ix, Storable a) => Stream -> DeviceArray ix a -> IO (GpuFuture (A.Array A.S ix a))
readDeviceArrayToPinnedArrayAsync stream deviceArray = do
  pinnedBuffer <- newPinnedBuffer (A.totalElem (deviceArraySize deviceArray))
  _ <- copyD2HAsync stream pinnedBuffer (deviceArrayBuffer deviceArray)
  makeFutureFromStreamWith stream $ do
    result <- readPinnedBufferToArray (deviceArraySize deviceArray) pinnedBuffer
    destroyBuffer pinnedBuffer
    pure result
