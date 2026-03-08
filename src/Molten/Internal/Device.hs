module Molten.Internal.Device
  ( withDeviceId
  ) where

import Control.Exception (bracket)
import Control.Monad (unless)
import Molten.Core.Types (DeviceId(..))
import ROCm.HIP (hipGetCurrentDevice, hipSetDevice)

withDeviceId :: DeviceId -> IO a -> IO a
withDeviceId (DeviceId targetId) action =
  bracket switch restore (const action)
  where
    switch :: IO Int
    switch = do
      current <- hipGetCurrentDevice
      unless (current == targetId) (hipSetDevice targetId)
      pure current

    restore :: Int -> IO ()
    restore previous =
      unless (previous == targetId) (hipSetDevice previous)
