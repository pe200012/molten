module Molten.BLAS.Types
  ( BlasElement(..)
  , Gemm(..)
  , NativeGemm(..)
  , Transpose(..)
  , rowMajorToNativeGemm
  , toRocblasOperation
  , validateNativeGemmShape
  , validateRowMajorGemmShape
  ) where

import Control.Exception (bracket_)
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Internal.Validation (ensureNonNegative, ensureNonZero, ensurePositive)
import ROCm.FFI.Core.Exception (throwArgumentError)
import ROCm.FFI.Core.Types (DevicePtr(..), RocblasHandle(..))
import ROCm.RocBLAS
  ( rocblasDaxpy
  , rocblasDdot
  , rocblasDgemm
  , rocblasSaxpy
  , rocblasSdot
  , rocblasSgemm
  , rocblasSetPointerMode
  )
import ROCm.RocBLAS.Error (checkRocblas)
import ROCm.RocBLAS.Raw (c_rocblas_ddot, c_rocblas_sdot)
import qualified ROCm.RocBLAS.Types as RocBLAS

data Transpose
  = NoTranspose
  | Transpose
  deriving (Eq, Show)

data Gemm mat a = Gemm
  { gemmTransA :: !Transpose
  , gemmTransB :: !Transpose
  , gemmM :: !Int
  , gemmN :: !Int
  , gemmK :: !Int
  , gemmAlpha :: !a
  , gemmA :: !mat
  , gemmLda :: !Int
  , gemmB :: !mat
  , gemmLdb :: !Int
  , gemmBeta :: !a
  , gemmC :: !mat
  , gemmLdc :: !Int
  }
  deriving (Eq, Show)

data NativeGemm mat a = NativeGemm
  { nativeGemmTransA :: !Transpose
  , nativeGemmTransB :: !Transpose
  , nativeGemmM :: !Int
  , nativeGemmN :: !Int
  , nativeGemmK :: !Int
  , nativeGemmAlpha :: !a
  , nativeGemmA :: !mat
  , nativeGemmLda :: !Int
  , nativeGemmB :: !mat
  , nativeGemmLdb :: !Int
  , nativeGemmBeta :: !a
  , nativeGemmC :: !mat
  , nativeGemmLdc :: !Int
  }
  deriving (Eq, Show)

class Storable a => BlasElement a where
  blasAxpy :: HasCallStack => RocblasHandle -> Int -> a -> DevicePtr a -> Int -> DevicePtr a -> Int -> IO ()
  blasDot :: HasCallStack => RocblasHandle -> Int -> DevicePtr a -> Int -> DevicePtr a -> Int -> IO a
  blasDotInto :: HasCallStack => RocblasHandle -> Int -> DevicePtr a -> Int -> DevicePtr a -> Int -> DevicePtr a -> IO ()
  blasGemm :: HasCallStack => RocblasHandle -> NativeGemm (DevicePtr a) a -> IO ()

instance BlasElement Float where
  blasAxpy handle n alpha x incx y incy =
    rocblasSaxpy handle (fromIntegral n) alpha (castDevicePtr x) (fromIntegral incx) (castDevicePtr y) (fromIntegral incy)

  blasDot handle n x incx y incy =
    rocblasSdot handle (fromIntegral n) (castDevicePtr x) (fromIntegral incx) (castDevicePtr y) (fromIntegral incy)

  blasDotInto handle n x incx y incy out =
    bracket_
      (rocblasSetPointerMode handle RocBLAS.RocblasPointerModeDevice)
      (rocblasSetPointerMode handle RocBLAS.RocblasPointerModeHost)
      (case handle of
        RocblasHandle rawHandle ->
          checkRocblas "rocblas_sdot" =<<
            c_rocblas_sdot
              rawHandle
              (fromIntegral n)
              (rawDevicePtr (castDevicePtr x))
              (fromIntegral incx)
              (rawDevicePtr (castDevicePtr y))
              (fromIntegral incy)
              (rawDevicePtr (castDevicePtr out)))
  blasGemm handle nativeGemm =
    rocblasSgemm
      handle
      (toRocblasOperation (nativeGemmTransA nativeGemm))
      (toRocblasOperation (nativeGemmTransB nativeGemm))
      (fromIntegral (nativeGemmM nativeGemm))
      (fromIntegral (nativeGemmN nativeGemm))
      (fromIntegral (nativeGemmK nativeGemm))
      (nativeGemmAlpha nativeGemm)
      (castDevicePtr (nativeGemmA nativeGemm))
      (fromIntegral (nativeGemmLda nativeGemm))
      (castDevicePtr (nativeGemmB nativeGemm))
      (fromIntegral (nativeGemmLdb nativeGemm))
      (nativeGemmBeta nativeGemm)
      (castDevicePtr (nativeGemmC nativeGemm))
      (fromIntegral (nativeGemmLdc nativeGemm))

instance BlasElement Double where
  blasAxpy handle n alpha x incx y incy =
    rocblasDaxpy handle (fromIntegral n) alpha (castDevicePtr x) (fromIntegral incx) (castDevicePtr y) (fromIntegral incy)

  blasDot handle n x incx y incy =
    rocblasDdot handle (fromIntegral n) (castDevicePtr x) (fromIntegral incx) (castDevicePtr y) (fromIntegral incy)

  blasDotInto handle n x incx y incy out =
    bracket_
      (rocblasSetPointerMode handle RocBLAS.RocblasPointerModeDevice)
      (rocblasSetPointerMode handle RocBLAS.RocblasPointerModeHost)
      (case handle of
        RocblasHandle rawHandle ->
          checkRocblas "rocblas_ddot" =<<
            c_rocblas_ddot
              rawHandle
              (fromIntegral n)
              (rawDevicePtr (castDevicePtr x))
              (fromIntegral incx)
              (rawDevicePtr (castDevicePtr y))
              (fromIntegral incy)
              (rawDevicePtr (castDevicePtr out)))
  blasGemm handle nativeGemm =
    rocblasDgemm
      handle
      (toRocblasOperation (nativeGemmTransA nativeGemm))
      (toRocblasOperation (nativeGemmTransB nativeGemm))
      (fromIntegral (nativeGemmM nativeGemm))
      (fromIntegral (nativeGemmN nativeGemm))
      (fromIntegral (nativeGemmK nativeGemm))
      (nativeGemmAlpha nativeGemm)
      (castDevicePtr (nativeGemmA nativeGemm))
      (fromIntegral (nativeGemmLda nativeGemm))
      (castDevicePtr (nativeGemmB nativeGemm))
      (fromIntegral (nativeGemmLdb nativeGemm))
      (nativeGemmBeta nativeGemm)
      (castDevicePtr (nativeGemmC nativeGemm))
      (fromIntegral (nativeGemmLdc nativeGemm))

rowMajorToNativeGemm :: Gemm mat a -> NativeGemm mat a
rowMajorToNativeGemm gemmSpec =
  NativeGemm
    { nativeGemmTransA = gemmTransB gemmSpec
    , nativeGemmTransB = gemmTransA gemmSpec
    , nativeGemmM = gemmN gemmSpec
    , nativeGemmN = gemmM gemmSpec
    , nativeGemmK = gemmK gemmSpec
    , nativeGemmAlpha = gemmAlpha gemmSpec
    , nativeGemmA = gemmB gemmSpec
    , nativeGemmLda = gemmLdb gemmSpec
    , nativeGemmB = gemmA gemmSpec
    , nativeGemmLdb = gemmLda gemmSpec
    , nativeGemmBeta = gemmBeta gemmSpec
    , nativeGemmC = gemmC gemmSpec
    , nativeGemmLdc = gemmLdc gemmSpec
    }

validateRowMajorGemmShape :: HasCallStack => String -> Gemm mat a -> IO ()
validateRowMajorGemmShape functionName gemmSpec = do
  ensureNonNegative functionName "m" (gemmM gemmSpec)
  ensureNonNegative functionName "n" (gemmN gemmSpec)
  ensureNonNegative functionName "k" (gemmK gemmSpec)
  ensurePositive functionName "lda" (gemmLda gemmSpec)
  ensurePositive functionName "ldb" (gemmLdb gemmSpec)
  ensurePositive functionName "ldc" (gemmLdc gemmSpec)
  ensureLeadingDimension functionName "lda" (requiredRowMajorLda gemmSpec) (gemmLda gemmSpec)
  ensureLeadingDimension functionName "ldb" (requiredRowMajorLdb gemmSpec) (gemmLdb gemmSpec)
  ensureLeadingDimension functionName "ldc" (requiredRowMajorLdc gemmSpec) (gemmLdc gemmSpec)

validateNativeGemmShape :: HasCallStack => String -> NativeGemm mat a -> IO ()
validateNativeGemmShape functionName nativeGemm = do
  ensureNonNegative functionName "m" (nativeGemmM nativeGemm)
  ensureNonNegative functionName "n" (nativeGemmN nativeGemm)
  ensureNonNegative functionName "k" (nativeGemmK nativeGemm)
  ensurePositive functionName "lda" (nativeGemmLda nativeGemm)
  ensurePositive functionName "ldb" (nativeGemmLdb nativeGemm)
  ensurePositive functionName "ldc" (nativeGemmLdc nativeGemm)
  ensureLeadingDimension functionName "lda" (requiredNativeLda nativeGemm) (nativeGemmLda nativeGemm)
  ensureLeadingDimension functionName "ldb" (requiredNativeLdb nativeGemm) (nativeGemmLdb nativeGemm)
  ensureLeadingDimension functionName "ldc" (requiredNativeLdc nativeGemm) (nativeGemmLdc nativeGemm)

toRocblasOperation :: Transpose -> RocBLAS.RocblasOperation
toRocblasOperation trans =
  case trans of
    NoTranspose -> RocBLAS.RocblasOperationNone
    Transpose -> RocBLAS.RocblasOperationTranspose

requiredRowMajorLda :: Gemm mat a -> Int
requiredRowMajorLda gemmSpec =
  case gemmTransA gemmSpec of
    NoTranspose -> max 1 (gemmK gemmSpec)
    Transpose -> max 1 (gemmM gemmSpec)

requiredRowMajorLdb :: Gemm mat a -> Int
requiredRowMajorLdb gemmSpec =
  case gemmTransB gemmSpec of
    NoTranspose -> max 1 (gemmN gemmSpec)
    Transpose -> max 1 (gemmK gemmSpec)

requiredRowMajorLdc :: Gemm mat a -> Int
requiredRowMajorLdc gemmSpec = max 1 (gemmN gemmSpec)

requiredNativeLda :: NativeGemm mat a -> Int
requiredNativeLda nativeGemm =
  case nativeGemmTransA nativeGemm of
    NoTranspose -> max 1 (nativeGemmM nativeGemm)
    Transpose -> max 1 (nativeGemmK nativeGemm)

requiredNativeLdb :: NativeGemm mat a -> Int
requiredNativeLdb nativeGemm =
  case nativeGemmTransB nativeGemm of
    NoTranspose -> max 1 (nativeGemmK nativeGemm)
    Transpose -> max 1 (nativeGemmN nativeGemm)

requiredNativeLdc :: NativeGemm mat a -> Int
requiredNativeLdc nativeGemm = max 1 (nativeGemmM nativeGemm)

ensureLeadingDimension :: HasCallStack => String -> String -> Int -> Int -> IO ()
ensureLeadingDimension functionName fieldName minimumValue actualValue = do
  ensureNonZero functionName fieldName actualValue
  if actualValue < minimumValue
    then throwArgumentError functionName (fieldName <> " must be >= " <> show minimumValue)
    else pure ()

castDevicePtr :: DevicePtr a -> DevicePtr b
castDevicePtr (DevicePtr ptr) = DevicePtr (castPtr ptr)

rawDevicePtr :: DevicePtr a -> Ptr a
rawDevicePtr (DevicePtr ptr) = ptr
