{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeSynonymInstances #-}

module Molten.FFT
  ( FftComplex(..)
  , FftReal(..)
  , FftShape(..)
  , fftForwardC2C
  , fftForwardR2C
  , fftInverseC2C
  , fftInverseC2R
  ) where

import qualified Data.Massiv.Array as A
import Data.Complex (Complex)
import Data.Proxy (Proxy(..))
import Foreign.Ptr (Ptr, castPtr)
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Array.Device
  ( DeviceArray
  , deviceArrayBuffer
  , deviceArraySize
  , mkDeviceArray
  )
import Molten.Core.Buffer (newDeviceBuffer, withDevicePtr)
import Molten.FFT.Runtime
  ( FftRuntime(fftRuntimeContext)
  , FftTransform(..)
  , FftWorkspaceMode(..)
  , lookupOrCreateFftPlan
  , mkFftPlanKey
  , runCachedFftPlan
  )
import ROCm.FFI.Core.Exception (throwArgumentError)
import ROCm.FFI.Core.Types (DevicePtr(..))
import ROCm.RocFFT
  ( RocfftPrecision
  , pattern RocfftPrecisionDouble
  , pattern RocfftPrecisionSingle
  )

class Storable a => FftComplex a where
  fftComplexPrecision :: proxy a -> RocfftPrecision

class Storable a => FftReal a where
  fftRealPrecision :: proxy a -> RocfftPrecision

class FftShape ix where
  fftLengths :: A.Sz ix -> [Int]
  fftR2cOutputSize :: HasCallStack => A.Sz ix -> IO (A.Sz ix)
  fftC2rOutputSize :: HasCallStack => A.Sz ix -> IO (A.Sz ix)

instance FftComplex (Complex Float) where
  fftComplexPrecision _ = RocfftPrecisionSingle

instance FftComplex (Complex Double) where
  fftComplexPrecision _ = RocfftPrecisionDouble

instance FftReal Float where
  fftRealPrecision _ = RocfftPrecisionSingle

instance FftReal Double where
  fftRealPrecision _ = RocfftPrecisionDouble

instance FftShape A.Ix1 where
  fftLengths (A.Sz1 n) = [n]

  fftR2cOutputSize (A.Sz1 n) = pure (A.Sz1 (n `div` 2 + 1))

  fftC2rOutputSize (A.Sz1 frequencyLength)
    | frequencyLength >= 2 = pure (A.Sz1 (2 * (frequencyLength - 1)))
    | otherwise = throwArgumentError "fftInverseC2R" "input frequency length must be >= 2"

instance FftShape A.Ix2 where
  fftLengths (A.Sz2 rows cols) = [rows, cols]

  fftR2cOutputSize (A.Sz2 rows cols) = pure (A.Sz2 rows (cols `div` 2 + 1))

  fftC2rOutputSize (A.Sz2 rows frequencyCols)
    | frequencyCols >= 2 = pure (A.Sz2 rows (2 * (frequencyCols - 1)))
    | otherwise = throwArgumentError "fftInverseC2R" "input frequency width must be >= 2"

instance FftShape A.Ix3 where
  fftLengths (A.Sz3 dim0 dim1 dim2) = [dim0, dim1, dim2]

  fftR2cOutputSize (A.Sz3 dim0 dim1 dim2) = pure (A.Sz3 dim0 dim1 (dim2 `div` 2 + 1))

  fftC2rOutputSize (A.Sz3 dim0 dim1 frequencyDim2)
    | frequencyDim2 >= 2 = pure (A.Sz3 dim0 dim1 (2 * (frequencyDim2 - 1)))
    | otherwise = throwArgumentError "fftInverseC2R" "input frequency depth must be >= 2"

fftForwardC2C :: forall ix a. (HasCallStack, A.Index ix, FftShape ix, FftComplex a) => FftRuntime -> DeviceArray ix a -> IO (DeviceArray ix a)
fftForwardC2C runtime input =
  fftC2C FftTransformComplexForward runtime input

fftInverseC2C :: forall ix a. (HasCallStack, A.Index ix, FftShape ix, FftComplex a) => FftRuntime -> DeviceArray ix a -> IO (DeviceArray ix a)
fftInverseC2C runtime input =
  fftC2C FftTransformComplexInverse runtime input

fftForwardR2C :: forall ix a. (HasCallStack, A.Index ix, FftShape ix, FftReal a) => FftRuntime -> DeviceArray ix a -> IO (DeviceArray ix (Complex a))
fftForwardR2C runtime input = do
  outputSize <- fftR2cOutputSize (deviceArraySize input)
  planKey <-
    mkFftPlanKey
      FftTransformRealForward
      (fftRealPrecision (Proxy :: Proxy a))
      (fftLengths (deviceArraySize input))
      1
      FftWorkspaceAuto
  cachedPlan <- lookupOrCreateFftPlan runtime planKey
  outputBuffer <- newDeviceBuffer (fftRuntimeContext runtime) (A.totalElem outputSize)
  withDevicePtr (deviceArrayBuffer input) $ \inputPtr ->
    withDevicePtr outputBuffer $ \outputPtr ->
      runCachedFftPlan runtime cachedPlan [castPtrDevice inputPtr] [castPtrDevice outputPtr]
  mkDeviceArray outputSize outputBuffer

fftInverseC2R :: forall ix a. (HasCallStack, A.Index ix, FftShape ix, FftReal a) => FftRuntime -> DeviceArray ix (Complex a) -> IO (DeviceArray ix a)
fftInverseC2R runtime input = do
  outputSize <- fftC2rOutputSize (deviceArraySize input)
  planKey <-
    mkFftPlanKey
      FftTransformRealInverse
      (fftRealPrecision (Proxy :: Proxy a))
      (fftLengths outputSize)
      1
      FftWorkspaceAuto
  cachedPlan <- lookupOrCreateFftPlan runtime planKey
  outputBuffer <- newDeviceBuffer (fftRuntimeContext runtime) (A.totalElem outputSize)
  withDevicePtr (deviceArrayBuffer input) $ \inputPtr ->
    withDevicePtr outputBuffer $ \outputPtr ->
      runCachedFftPlan runtime cachedPlan [castPtrDevice inputPtr] [castPtrDevice outputPtr]
  mkDeviceArray outputSize outputBuffer

fftC2C :: forall ix a. (HasCallStack, A.Index ix, FftShape ix, FftComplex a) => FftTransform -> FftRuntime -> DeviceArray ix a -> IO (DeviceArray ix a)
fftC2C transform runtime input = do
  planKey <-
    mkFftPlanKey
      transform
      (fftComplexPrecision (Proxy :: Proxy a))
      (fftLengths (deviceArraySize input))
      1
      FftWorkspaceAuto
  cachedPlan <- lookupOrCreateFftPlan runtime planKey
  outputBuffer <- newDeviceBuffer (fftRuntimeContext runtime) (A.totalElem (deviceArraySize input))
  withDevicePtr (deviceArrayBuffer input) $ \inputPtr ->
    withDevicePtr outputBuffer $ \outputPtr ->
      runCachedFftPlan runtime cachedPlan [castPtrDevice inputPtr] [castPtrDevice outputPtr]
  mkDeviceArray (deviceArraySize input) outputBuffer

castPtrDevice :: DevicePtr a -> Ptr ()
castPtrDevice (DevicePtr ptr) = castPtr ptr
