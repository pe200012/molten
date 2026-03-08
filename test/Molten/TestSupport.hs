{-# LANGUAGE ScopedTypeVariables #-}

module Molten.TestSupport
  ( TestRuntime(..)
  , detectTestRuntime
  , withGpuContext
  , withBlasContext
  ) where

import Control.Exception (SomeException, catch)
import Molten.Core.Context (Context, DeviceId(..), withContext)
import ROCm.HIP.Device (hipGetCurrentDeviceGcnArchName, hipGetCurrentDeviceName, hipGetDeviceCount)
import Test.Hspec (pendingWith)

data TestRuntime = TestRuntime
  { runtimeHasGpu :: !Bool
  , runtimeDeviceName :: !String
  , runtimeArchName :: !String
  , runtimeSkipReason :: !(Maybe String)
  }
  deriving (Eq, Show)

detectTestRuntime :: IO TestRuntime
detectTestRuntime = do
  deviceCount <- catch hipGetDeviceCount noGpu
  if deviceCount <= 0
    then pure missingGpuRuntime
    else do
      deviceName <- catch hipGetCurrentDeviceName (\(_ :: SomeException) -> pure "unknown-device")
      archName <- catch hipGetCurrentDeviceGcnArchName (\(_ :: SomeException) -> pure "unknown-arch")
      pure
        TestRuntime
          { runtimeHasGpu = True
          , runtimeDeviceName = deviceName
          , runtimeArchName = archName
          , runtimeSkipReason = Nothing
          }
  where
    noGpu :: SomeException -> IO Int
    noGpu _ = pure 0

    missingGpuRuntime :: TestRuntime
    missingGpuRuntime =
      TestRuntime
        { runtimeHasGpu = False
        , runtimeDeviceName = "no-device"
        , runtimeArchName = "no-arch"
        , runtimeSkipReason = Just "ROCm GPU not available"
        }

withGpuContext :: (Context -> IO ()) -> IO ()
withGpuContext action = do
  runtime <- detectTestRuntime
  if runtimeHasGpu runtime
    then withContext (DeviceId 0) action
    else pendingWith (maybe "ROCm GPU not available" id (runtimeSkipReason runtime))

withBlasContext :: (Context -> IO ()) -> IO ()
withBlasContext = withGpuContext
