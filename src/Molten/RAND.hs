{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Molten.RAND
  ( RandNormal(..)
  , RandUniform(..)
  , randNormal
  , randNormalOn
  , randUniform
  , randUniformOn
  ) where

import qualified Data.Massiv.Array as A
import Foreign.C.Types (CDouble, CFloat)
import Foreign.Ptr (Ptr, castPtr)
import GHC.Stack (HasCallStack)
import Molten.Array.Device (DeviceArray, deviceArrayBuffer, deviceArraySize)
import Molten.Core.Buffer (withDevicePtr)
import Molten.Core.Context (contextDefaultStream)
import Molten.Core.Stream (Stream)
import Molten.RAND.Runtime
  ( CachedGenerator
  , RandGeneratorConfig
  , RandRuntime(randRuntimeContext)
  , lookupOrCreateGenerator
  , withGeneratorOnStream
  )
import ROCm.FFI.Core.Types (DevicePtr(..), RocRandGenerator)
import ROCm.RocRAND
  ( rocrandGenerateNormal
  , rocrandGenerateNormalDouble
  , rocrandGenerateUniform
  , rocrandGenerateUniformDouble
  )

class RandUniform a where
  randUniformWithGenerator :: HasCallStack => RocRandGenerator -> DevicePtr a -> Int -> IO ()

class RandNormal a where
  randNormalWithGenerator :: HasCallStack => RocRandGenerator -> DevicePtr a -> Int -> a -> a -> IO ()

instance RandUniform Float where
  randUniformWithGenerator generator (DevicePtr outPtr) n =
    rocrandGenerateUniform generator (DevicePtr (castPtr outPtr :: CFloatPtr)) (fromIntegral n)

instance RandUniform Double where
  randUniformWithGenerator generator (DevicePtr outPtr) n =
    rocrandGenerateUniformDouble generator (DevicePtr (castPtr outPtr :: CDoublePtr)) (fromIntegral n)

instance RandNormal Float where
  randNormalWithGenerator generator (DevicePtr outPtr) n meanValue stddevValue =
    rocrandGenerateNormal generator (DevicePtr (castPtr outPtr :: CFloatPtr)) (fromIntegral n) meanValue stddevValue

instance RandNormal Double where
  randNormalWithGenerator generator (DevicePtr outPtr) n meanValue stddevValue =
    rocrandGenerateNormalDouble generator (DevicePtr (castPtr outPtr :: CDoublePtr)) (fromIntegral n) meanValue stddevValue

randUniform :: (HasCallStack, RandUniform a, A.Index ix) => RandRuntime -> RandGeneratorConfig -> DeviceArray ix a -> IO ()
randUniform runtime config deviceArray =
  randUniformOn (contextDefaultStream (randRuntimeContext runtime)) runtime config deviceArray

randUniformOn :: (HasCallStack, RandUniform a, A.Index ix) => Stream -> RandRuntime -> RandGeneratorConfig -> DeviceArray ix a -> IO ()
randUniformOn stream runtime config deviceArray = do
  cachedGenerator <- lookupOrCreateGenerator runtime config
  fillUniform cachedGenerator stream deviceArray

randNormal :: (HasCallStack, RandNormal a, A.Index ix) => RandRuntime -> RandGeneratorConfig -> a -> a -> DeviceArray ix a -> IO ()
randNormal runtime config meanValue stddevValue deviceArray =
  randNormalOn (contextDefaultStream (randRuntimeContext runtime)) runtime config meanValue stddevValue deviceArray

randNormalOn :: (HasCallStack, RandNormal a, A.Index ix) => Stream -> RandRuntime -> RandGeneratorConfig -> a -> a -> DeviceArray ix a -> IO ()
randNormalOn stream runtime config meanValue stddevValue deviceArray = do
  cachedGenerator <- lookupOrCreateGenerator runtime config
  fillNormal cachedGenerator stream meanValue stddevValue deviceArray

fillUniform :: (HasCallStack, RandUniform a, A.Index ix) => CachedGenerator -> Stream -> DeviceArray ix a -> IO ()
fillUniform cachedGenerator stream deviceArray =
  withGeneratorOnStream cachedGenerator stream $ \generator ->
    withDevicePtr (deviceArrayBuffer deviceArray) $ \devicePtr ->
      randUniformWithGenerator generator devicePtr (A.totalElem (deviceArraySize deviceArray))

fillNormal :: (HasCallStack, RandNormal a, A.Index ix) => CachedGenerator -> Stream -> a -> a -> DeviceArray ix a -> IO ()
fillNormal cachedGenerator stream meanValue stddevValue deviceArray =
  withGeneratorOnStream cachedGenerator stream $ \generator ->
    withDevicePtr (deviceArrayBuffer deviceArray) $ \devicePtr ->
      randNormalWithGenerator generator devicePtr (A.totalElem (deviceArraySize deviceArray)) meanValue stddevValue

type CFloatPtr = Ptr CFloat

type CDoublePtr = Ptr CDouble
