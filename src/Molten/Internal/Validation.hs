module Molten.Internal.Validation
  ( ensureDeviceMatch
  , ensureNonNegative
  , ensureNonZero
  , ensurePositive
  , ensureSameLength
  ) where

import Control.Monad (unless, when)
import GHC.Stack (HasCallStack)
import Molten.Core.Types (DeviceId)
import ROCm.FFI.Core.Exception (throwArgumentError)

ensureNonNegative :: HasCallStack => String -> String -> Int -> IO ()
ensureNonNegative functionName fieldName value =
  when (value < 0) (throwArgumentError functionName (fieldName <> " must be >= 0"))

ensurePositive :: HasCallStack => String -> String -> Int -> IO ()
ensurePositive functionName fieldName value =
  when (value <= 0) (throwArgumentError functionName (fieldName <> " must be > 0"))

ensureNonZero :: HasCallStack => String -> String -> Int -> IO ()
ensureNonZero functionName fieldName value =
  when (value == 0) (throwArgumentError functionName (fieldName <> " must not be 0"))

ensureSameLength :: HasCallStack => String -> Int -> Int -> IO ()
ensureSameLength functionName leftLength rightLength =
  unless (leftLength == rightLength) (throwArgumentError functionName "buffers must have the same length")

ensureDeviceMatch :: HasCallStack => String -> DeviceId -> DeviceId -> IO ()
ensureDeviceMatch functionName expected actual =
  unless (expected == actual) (throwArgumentError functionName "resources must belong to the same device")
