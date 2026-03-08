module Molten.Internal.Reference.Axis2D
  ( broadcastColsRef
  , broadcastRowsRef
  , maxColsRef
  , maxRowsRef
  , sumColsRef
  , sumRowsRef
  ) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Array.Expr (Comparable, NumericExp, addScalar, lessThanScalar)
import ROCm.FFI.Core.Exception (throwArgumentError)

sumRowsRef :: (HasCallStack, NumericExp a) => A.Array A.S A.Ix2 a -> IO (A.Array A.S A.Ix1 a)
sumRowsRef input = do
  (rows, cols) <- matrixShape "sumRowsRef" input
  if cols == 0
    then throwArgumentError "sumRowsRef" "input matrix must have at least one column"
    else pure (AMV.fromVector' A.Seq (A.Sz1 rows) (VS.generate rows (sumRow cols vector)))
  where
    vector = AM.toStorableVector input

sumColsRef :: (HasCallStack, NumericExp a) => A.Array A.S A.Ix2 a -> IO (A.Array A.S A.Ix1 a)
sumColsRef input = do
  (rows, cols) <- matrixShape "sumColsRef" input
  if rows == 0
    then throwArgumentError "sumColsRef" "input matrix must have at least one row"
    else pure (AMV.fromVector' A.Seq (A.Sz1 cols) (VS.generate cols (sumCol rows cols vector)))
  where
    vector = AM.toStorableVector input

maxRowsRef :: (HasCallStack, Comparable a) => A.Array A.S A.Ix2 a -> IO (A.Array A.S A.Ix1 a)
maxRowsRef input = do
  (rows, cols) <- matrixShape "maxRowsRef" input
  if cols == 0
    then throwArgumentError "maxRowsRef" "input matrix must have at least one column"
    else pure (AMV.fromVector' A.Seq (A.Sz1 rows) (VS.generate rows (maxRow cols vector)))
  where
    vector = AM.toStorableVector input

maxColsRef :: (HasCallStack, Comparable a) => A.Array A.S A.Ix2 a -> IO (A.Array A.S A.Ix1 a)
maxColsRef input = do
  (rows, cols) <- matrixShape "maxColsRef" input
  if rows == 0
    then throwArgumentError "maxColsRef" "input matrix must have at least one row"
    else pure (AMV.fromVector' A.Seq (A.Sz1 cols) (VS.generate cols (maxCol rows cols vector)))
  where
    vector = AM.toStorableVector input

broadcastRowsRef :: (HasCallStack, Storable a) => Int -> A.Array A.S A.Ix1 a -> IO (A.Array A.S A.Ix2 a)
broadcastRowsRef cols input
  | cols < 0 = throwArgumentError "broadcastRowsRef" "column count must be non-negative"
  | otherwise =
      let A.Sz1 rows = A.size input
          vector = AM.toStorableVector input
       in pure
            ( AMV.fromVector' A.Seq (A.Sz2 rows cols)
                (VS.generate (rows * cols) (\ix -> VS.unsafeIndex vector (ix `div` cols)))
            )

broadcastColsRef :: (HasCallStack, Storable a) => Int -> A.Array A.S A.Ix1 a -> IO (A.Array A.S A.Ix2 a)
broadcastColsRef rows input
  | rows < 0 = throwArgumentError "broadcastColsRef" "row count must be non-negative"
  | otherwise =
      let A.Sz1 cols = A.size input
          vector = AM.toStorableVector input
       in pure
            ( AMV.fromVector' A.Seq (A.Sz2 rows cols)
                (VS.generate (rows * cols) (\ix -> VS.unsafeIndex vector (ix `mod` cols)))
            )

matrixShape :: HasCallStack => String -> A.Array A.S A.Ix2 a -> IO (Int, Int)
matrixShape functionName input =
  case A.size input of
    A.Sz2 rows cols
      | rows < 0 || cols < 0 -> throwArgumentError functionName "matrix dimensions must be non-negative"
      | otherwise -> pure (rows, cols)

sumRow :: NumericExp a => Int -> VS.Vector a -> Int -> a
sumRow cols vector rowIndex =
  let rowOffset = rowIndex * cols
      initialValue = VS.unsafeIndex vector rowOffset
   in foldFrom 1 initialValue (\ix acc -> addScalar acc (VS.unsafeIndex vector (rowOffset + ix))) cols

sumCol :: NumericExp a => Int -> Int -> VS.Vector a -> Int -> a
sumCol rows cols vector colIndex =
  let initialValue = VS.unsafeIndex vector colIndex
   in foldFrom 1 initialValue (\ix acc -> addScalar acc (VS.unsafeIndex vector (ix * cols + colIndex))) rows

maxRow :: Comparable a => Int -> VS.Vector a -> Int -> a
maxRow cols vector rowIndex =
  let rowOffset = rowIndex * cols
      initialValue = VS.unsafeIndex vector rowOffset
   in foldFrom 1 initialValue (\ix acc -> maxScalar acc (VS.unsafeIndex vector (rowOffset + ix))) cols

maxCol :: Comparable a => Int -> Int -> VS.Vector a -> Int -> a
maxCol rows cols vector colIndex =
  let initialValue = VS.unsafeIndex vector colIndex
   in foldFrom 1 initialValue (\ix acc -> maxScalar acc (VS.unsafeIndex vector (ix * cols + colIndex))) rows

maxScalar :: Comparable a => a -> a -> a
maxScalar left right =
  if lessThanScalar left right
    then right
    else left

foldFrom :: Int -> a -> (Int -> a -> a) -> Int -> a
foldFrom start initialValue step limit = go start initialValue
  where
    go ix acc
      | ix >= limit = acc
      | otherwise = go (ix + 1) (step ix acc)
