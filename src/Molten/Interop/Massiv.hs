{-# LANGUAGE DataKinds #-}

module Molten.Interop.Massiv
  ( readHostBufferToArray
  , readPinnedBufferToArray
  , withHostBufferFromArray
  , withPinnedBufferFromArray
  ) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import Foreign.Storable (Storable)
import Molten.Core.Buffer (Buffer, Location(..), bufferLength)
import ROCm.FFI.Core.Exception (throwArgumentError)
import Molten.Interop.Vector
  ( readHostBufferToVector
  , readPinnedBufferToVector
  , withHostBufferFromVector
  , withPinnedBufferFromVector
  )

withHostBufferFromArray :: (A.Load r ix e, Storable e) => A.Array r ix e -> (A.Sz ix -> Buffer 'Host e -> IO a) -> IO a
withHostBufferFromArray arr action =
  let manifest = A.computeAs A.S arr
      sz = A.size manifest
      vec = AM.toStorableVector manifest
   in withHostBufferFromVector vec (action sz)

withPinnedBufferFromArray :: (A.Load r ix e, Storable e) => A.Array r ix e -> (A.Sz ix -> Buffer 'PinnedHost e -> IO a) -> IO a
withPinnedBufferFromArray arr action =
  let manifest = A.computeAs A.S arr
      sz = A.size manifest
      vec = AM.toStorableVector manifest
   in withPinnedBufferFromVector vec (action sz)

readHostBufferToArray :: (A.Index ix, Storable e) => A.Sz ix -> Buffer 'Host e -> IO (A.Array A.S ix e)
readHostBufferToArray sz hostBuf = do
  ensureShapeMatchesBuffer "readHostBufferToArray" sz (bufferLength hostBuf)
  vec <- readHostBufferToVector hostBuf
  pure (AMV.fromVector' A.Seq sz vec)

readPinnedBufferToArray :: (A.Index ix, Storable e) => A.Sz ix -> Buffer 'PinnedHost e -> IO (A.Array A.S ix e)
readPinnedBufferToArray sz pinnedBuf = do
  ensureShapeMatchesBuffer "readPinnedBufferToArray" sz (bufferLength pinnedBuf)
  vec <- readPinnedBufferToVector pinnedBuf
  pure (AMV.fromVector' A.Seq sz vec)

ensureShapeMatchesBuffer :: A.Index ix => String -> A.Sz ix -> Int -> IO ()
ensureShapeMatchesBuffer functionName sz actualLength = do
  let expectedLength = A.totalElem sz
  if actualLength == expectedLength
    then pure ()
    else throwArgumentError functionName ("totalElem size = " <> show expectedLength <> ", bufferLength = " <> show actualLength)
