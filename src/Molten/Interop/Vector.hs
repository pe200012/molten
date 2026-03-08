{-# LANGUAGE DataKinds #-}

module Molten.Interop.Vector
  ( readHostBufferToVector
  , readPinnedBufferToVector
  , withHostBufferFromVector
  , withPinnedBufferFromVector
  ) where

import qualified Data.Vector.Storable as VS
import Foreign.Marshal.Array (copyArray, peekArray)
import Foreign.Storable (Storable)
import Molten.Core.Buffer
  ( Buffer
  , Location(..)
  , bufferLength
  , withHostBuffer
  , withHostPtr
  , withPinnedBuffer
  , withPinnedHostPtr
  )
import ROCm.FFI.Core.Types (HostPtr(..), PinnedHostPtr(..))

withHostBufferFromVector :: Storable a => VS.Vector a -> (Buffer 'Host a -> IO r) -> IO r
withHostBufferFromVector vector action =
  withHostBuffer (VS.length vector) $ \buffer -> do
    VS.unsafeWith vector $ \sourcePtr ->
      withHostPtr buffer $ \(HostPtr targetPtr) ->
        copyArray targetPtr sourcePtr (VS.length vector)
    action buffer

withPinnedBufferFromVector :: Storable a => VS.Vector a -> (Buffer 'PinnedHost a -> IO r) -> IO r
withPinnedBufferFromVector vector action =
  withPinnedBuffer (VS.length vector) $ \buffer -> do
    VS.unsafeWith vector $ \sourcePtr ->
      withPinnedHostPtr buffer $ \(PinnedHostPtr targetPtr) ->
        copyArray targetPtr sourcePtr (VS.length vector)
    action buffer

readHostBufferToVector :: Storable a => Buffer 'Host a -> IO (VS.Vector a)
readHostBufferToVector buffer =
  withHostPtr buffer $ \(HostPtr sourcePtr) -> VS.fromList <$> peekArray (bufferLength buffer) sourcePtr

readPinnedBufferToVector :: Storable a => Buffer 'PinnedHost a -> IO (VS.Vector a)
readPinnedBufferToVector buffer =
  withPinnedHostPtr buffer $ \(PinnedHostPtr sourcePtr) -> VS.fromList <$> peekArray (bufferLength buffer) sourcePtr
