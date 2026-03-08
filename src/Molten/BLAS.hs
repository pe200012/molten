{-# LANGUAGE DataKinds #-}

module Molten.BLAS
  ( MatrixGemm(..)
  , axpy
  , axpyVector
  , dot
  , dotInto
  , dotVector
  , gemm
  , gemmMatrix
  ) where

import GHC.Stack (HasCallStack)
import qualified Data.Massiv.Array as A
import Molten.Array.Device (DeviceArray, deviceArrayBuffer, deviceArraySize)
import Molten.BLAS.Native (axpyNative, dotIntoNative, dotNative, gemmNative)
import Molten.BLAS.Types (BlasElement, Gemm(..), Transpose(..), rowMajorToNativeGemm, validateRowMajorGemmShape)
import Molten.Core.Buffer (Buffer, Location(..))
import Molten.Core.Context (Context)
import ROCm.FFI.Core.Exception (throwArgumentError)

axpy :: (HasCallStack, BlasElement a) => Context -> a -> Buffer 'Device a -> Buffer 'Device a -> IO ()
axpy = axpyNative

axpyVector :: (HasCallStack, BlasElement a) => Context -> a -> DeviceArray A.Ix1 a -> DeviceArray A.Ix1 a -> IO ()
axpyVector ctx alpha x y = do
  let xLength = vectorLength x
      yLength = vectorLength y
  if xLength == yLength
    then axpy ctx alpha (deviceArrayBuffer x) (deviceArrayBuffer y)
    else throwArgumentError "axpyVector" "vectors must have the same length"

dot :: (HasCallStack, BlasElement a) => Context -> Buffer 'Device a -> Buffer 'Device a -> IO a
dot = dotNative

dotVector :: (HasCallStack, BlasElement a) => Context -> DeviceArray A.Ix1 a -> DeviceArray A.Ix1 a -> IO a
dotVector ctx x y = do
  let xLength = vectorLength x
      yLength = vectorLength y
  if xLength == yLength
    then dot ctx (deviceArrayBuffer x) (deviceArrayBuffer y)
    else throwArgumentError "dotVector" "vectors must have the same length"

dotInto :: (HasCallStack, BlasElement a) => Context -> Buffer 'Device a -> Buffer 'Device a -> Buffer 'Device a -> IO ()
dotInto = dotIntoNative

gemm :: (HasCallStack, BlasElement a) => Context -> Gemm (Buffer 'Device a) a -> IO ()
gemm ctx gemmSpec = do
  validateRowMajorGemmShape "gemm" gemmSpec
  gemmNative ctx (rowMajorToNativeGemm gemmSpec)

data MatrixGemm a = MatrixGemm
  { matrixGemmTransA :: !Transpose
  , matrixGemmTransB :: !Transpose
  , matrixGemmAlpha :: !a
  , matrixGemmA :: !(DeviceArray A.Ix2 a)
  , matrixGemmB :: !(DeviceArray A.Ix2 a)
  , matrixGemmBeta :: !a
  , matrixGemmC :: !(DeviceArray A.Ix2 a)
  }

gemmMatrix :: (HasCallStack, BlasElement a) => Context -> MatrixGemm a -> IO ()
gemmMatrix ctx matrixGemm = do
  let aRows = matrixRowCount (matrixGemmA matrixGemm)
      aCols = matrixColumnCount (matrixGemmA matrixGemm)
      bRows = matrixRowCount (matrixGemmB matrixGemm)
      bCols = matrixColumnCount (matrixGemmB matrixGemm)
      cRows = matrixRowCount (matrixGemmC matrixGemm)
      cCols = matrixColumnCount (matrixGemmC matrixGemm)
      m = effectiveRows (matrixGemmTransA matrixGemm) aRows aCols
      kLeft = effectiveCols (matrixGemmTransA matrixGemm) aRows aCols
      kRight = effectiveRows (matrixGemmTransB matrixGemm) bRows bCols
      n = effectiveCols (matrixGemmTransB matrixGemm) bRows bCols
  if kLeft /= kRight
    then throwArgumentError "gemmMatrix" "matrix inner dimensions must agree"
    else
      if cRows /= m || cCols /= n
        then throwArgumentError "gemmMatrix" "output matrix shape must match the GEMM result shape"
        else
          gemm
            ctx
            Gemm
              { gemmTransA = matrixGemmTransA matrixGemm
              , gemmTransB = matrixGemmTransB matrixGemm
              , gemmM = m
              , gemmN = n
              , gemmK = kLeft
              , gemmAlpha = matrixGemmAlpha matrixGemm
              , gemmA = deviceArrayBuffer (matrixGemmA matrixGemm)
              , gemmLda = aCols
              , gemmB = deviceArrayBuffer (matrixGemmB matrixGemm)
              , gemmLdb = bCols
              , gemmBeta = matrixGemmBeta matrixGemm
              , gemmC = deviceArrayBuffer (matrixGemmC matrixGemm)
              , gemmLdc = cCols
              }

vectorLength :: DeviceArray A.Ix1 a -> Int
vectorLength deviceArray =
  case deviceArraySize deviceArray of
    A.Sz1 n -> n

matrixRowCount :: DeviceArray A.Ix2 a -> Int
matrixRowCount deviceArray =
  case deviceArraySize deviceArray of
    A.Sz2 rows _ -> rows

matrixColumnCount :: DeviceArray A.Ix2 a -> Int
matrixColumnCount deviceArray =
  case deviceArraySize deviceArray of
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
