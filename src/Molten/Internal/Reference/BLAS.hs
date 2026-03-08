module Molten.Internal.Reference.BLAS
  ( MatrixGemmRef(..)
  , axpyVectorRef
  , dotVectorRef
  , gemmMatrixRef
  ) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.BLAS.Types (Transpose(..))
import ROCm.FFI.Core.Exception (throwArgumentError)

data MatrixGemmRef a = MatrixGemmRef
  { matrixGemmRefTransA :: !Transpose
  , matrixGemmRefTransB :: !Transpose
  , matrixGemmRefAlpha :: !a
  , matrixGemmRefA :: !(A.Array A.S A.Ix2 a)
  , matrixGemmRefB :: !(A.Array A.S A.Ix2 a)
  , matrixGemmRefBeta :: !a
  , matrixGemmRefC :: !(A.Array A.S A.Ix2 a)
  }
  deriving (Eq, Show)

axpyVectorRef :: (HasCallStack, Num a, Storable a) => a -> A.Array A.S A.Ix1 a -> A.Array A.S A.Ix1 a -> IO (A.Array A.S A.Ix1 a)
axpyVectorRef alpha x y =
  if A.size x == A.size y
    then
      pure
        ( AMV.fromVector' A.Seq (A.size y)
            (VS.zipWith (\xValue yValue -> alpha * xValue + yValue) (AM.toStorableVector x) (AM.toStorableVector y))
        )
    else throwArgumentError "axpyVectorRef" "vectors must have the same length"

dotVectorRef :: (HasCallStack, Num a, Storable a) => A.Array A.S A.Ix1 a -> A.Array A.S A.Ix1 a -> IO a
dotVectorRef x y =
  if A.size x == A.size y
    then pure (VS.sum (VS.zipWith (*) (AM.toStorableVector x) (AM.toStorableVector y)))
    else throwArgumentError "dotVectorRef" "vectors must have the same length"

gemmMatrixRef :: (HasCallStack, Num a, Storable a) => MatrixGemmRef a -> IO (A.Array A.S A.Ix2 a)
gemmMatrixRef matrixGemm = do
  let aRows = matrixRowCount (matrixGemmRefA matrixGemm)
      aCols = matrixColumnCount (matrixGemmRefA matrixGemm)
      bRows = matrixRowCount (matrixGemmRefB matrixGemm)
      bCols = matrixColumnCount (matrixGemmRefB matrixGemm)
      cRows = matrixRowCount (matrixGemmRefC matrixGemm)
      cCols = matrixColumnCount (matrixGemmRefC matrixGemm)
      m = effectiveRows (matrixGemmRefTransA matrixGemm) aRows aCols
      kLeft = effectiveCols (matrixGemmRefTransA matrixGemm) aRows aCols
      kRight = effectiveRows (matrixGemmRefTransB matrixGemm) bRows bCols
      n = effectiveCols (matrixGemmRefTransB matrixGemm) bRows bCols
  if kLeft /= kRight
    then throwArgumentError "gemmMatrixRef" "matrix inner dimensions must agree"
    else
      if cRows /= m || cCols /= n
        then throwArgumentError "gemmMatrixRef" "output matrix shape must match the GEMM result shape"
        else do
          let aValues = AM.toStorableVector (matrixGemmRefA matrixGemm)
              bValues = AM.toStorableVector (matrixGemmRefB matrixGemm)
              cValues = AM.toStorableVector (matrixGemmRefC matrixGemm)
              resultValues =
                VS.generate (cRows * cCols) $ \index ->
                  let (row, col) = index `divMod` cCols
                      total =
                        foldl
                          (\acc k -> acc + matrixElement (matrixGemmRefTransA matrixGemm) aRows aCols aValues row k * matrixElement (matrixGemmRefTransB matrixGemm) bRows bCols bValues k col)
                          0
                          [0 .. kLeft - 1]
                      cValue = cValues VS.! index
                   in matrixGemmRefAlpha matrixGemm * total + matrixGemmRefBeta matrixGemm * cValue
          pure (AMV.fromVector' A.Seq (A.Sz2 cRows cCols) resultValues)

matrixRowCount :: A.Array A.S A.Ix2 a -> Int
matrixRowCount array =
  case A.size array of
    A.Sz2 rows _ -> rows

matrixColumnCount :: A.Array A.S A.Ix2 a -> Int
matrixColumnCount array =
  case A.size array of
    A.Sz2 _ cols -> cols

effectiveRows :: Transpose -> Int -> Int -> Int
effectiveRows trans rows cols =
  case trans of
    NoTranspose -> rows
    Transpose -> cols

effectiveCols :: Transpose -> Int -> Int -> Int
effectiveCols trans rows cols =
  case trans of
    NoTranspose -> cols
    Transpose -> rows

matrixElement :: Storable a => Transpose -> Int -> Int -> VS.Vector a -> Int -> Int -> a
matrixElement trans _rows cols values row col =
  case trans of
    NoTranspose -> values VS.! (row * cols + col)
    Transpose -> values VS.! (col * cols + row)
