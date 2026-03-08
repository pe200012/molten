{-# LANGUAGE DataKinds #-}

module Molten.Array.Device
  ( DeviceArray
  , deviceArrayBuffer
  , deviceArraySize
  , mkDeviceArray
  , mkDeviceMatrix
  , mkDeviceVector
  , withDeviceArray
  , withDeviceMatrix
  , withDeviceVector
  ) where

import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import qualified Data.Massiv.Array as A
import Molten.Core.Buffer (Buffer, Location(..), bufferLength, withDeviceBuffer)
import Molten.Core.Context (Context)
import Molten.Internal.Validation (ensureNonNegative)
import ROCm.FFI.Core.Exception (throwArgumentError)

data DeviceArray ix a = DeviceArray
  { deviceArraySize :: !(A.Sz ix)
  , deviceArrayBuffer :: !(Buffer 'Device a)
  }

mkDeviceArray :: (HasCallStack, A.Index ix) => A.Sz ix -> Buffer 'Device a -> IO (DeviceArray ix a)
mkDeviceArray sz buf = do
  let expected = A.totalElem sz
      actual = bufferLength buf
  if actual == expected
    then pure DeviceArray {deviceArraySize = sz, deviceArrayBuffer = buf}
    else throwArgumentError "mkDeviceArray" ("totalElem size = " <> show expected <> ", bufferLength = " <> show actual)

mkDeviceVector :: HasCallStack => Int -> Buffer 'Device a -> IO (DeviceArray A.Ix1 a)
mkDeviceVector n = mkDeviceArray (A.Sz1 n)

mkDeviceMatrix :: HasCallStack => Int -> Int -> Buffer 'Device a -> IO (DeviceArray A.Ix2 a)
mkDeviceMatrix rows cols = mkDeviceArray (A.Sz2 rows cols)

withDeviceArray :: (HasCallStack, A.Index ix, Storable a) => Context -> A.Sz ix -> (DeviceArray ix a -> IO r) -> IO r
withDeviceArray ctx sz action =
  withDeviceBuffer ctx (A.totalElem sz) $ \buf ->
    mkDeviceArray sz buf >>= action

withDeviceVector :: (HasCallStack, Storable a) => Context -> Int -> (DeviceArray A.Ix1 a -> IO r) -> IO r
withDeviceVector ctx n action = do
  ensureNonNegative "withDeviceVector" "length" n
  withDeviceArray ctx (A.Sz1 n) action

withDeviceMatrix :: (HasCallStack, Storable a) => Context -> Int -> Int -> (DeviceArray A.Ix2 a -> IO r) -> IO r
withDeviceMatrix ctx rows cols action = do
  ensureNonNegative "withDeviceMatrix" "rows" rows
  ensureNonNegative "withDeviceMatrix" "cols" cols
  withDeviceArray ctx (A.Sz2 rows cols) action
