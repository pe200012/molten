module Molten.Array.Axis2D
  ( Axis2(..)
  , broadcastCols
  , broadcastRows
  , maxCols
  , maxRows
  , sumCols
  , sumRows
  ) where

import qualified Data.Massiv.Array as A
import GHC.Stack (HasCallStack)
import Molten.Array.Device (DeviceArray)
import Molten.Array.Expr (ArrayScalar, Comparable, NumericExp)
import Molten.Array.Runtime
  ( ArrayRuntime
  , broadcastColsArray
  , broadcastRowsArray
  , maxColsArray
  , maxRowsArray
  , sumColsArray
  , sumRowsArray
  )

data Axis2
  = AxisRows
  | AxisCols
  deriving (Eq, Show)

sumRows :: (HasCallStack, NumericExp a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
sumRows = sumRowsArray

sumCols :: (HasCallStack, NumericExp a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
sumCols = sumColsArray

maxRows :: (HasCallStack, Comparable a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
maxRows = maxRowsArray

maxCols :: (HasCallStack, Comparable a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
maxCols = maxColsArray

broadcastRows :: (HasCallStack, ArrayScalar a) => ArrayRuntime -> Int -> DeviceArray A.Ix1 a -> IO (DeviceArray A.Ix2 a)
broadcastRows = broadcastRowsArray

broadcastCols :: (HasCallStack, ArrayScalar a) => ArrayRuntime -> Int -> DeviceArray A.Ix1 a -> IO (DeviceArray A.Ix2 a)
broadcastCols = broadcastColsArray
