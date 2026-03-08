module Molten.Internal.HIPRTC
  ( LoadedHipKernel(..)
  , compileHipKernel
  , loadHipKernel
  , unloadHipKernel
  ) where

import Control.Exception (catch)
import Data.ByteString (ByteString)
import GHC.Stack (HasCallStack)
import Molten.Core.Types (DeviceId)
import Molten.Internal.Device (withDeviceId)
import ROCm.FFI.Core.Exception (FFIError(..), throwFFIError)
import ROCm.FFI.Core.Types (HipFunction, HipModule)
import ROCm.HIP (hipGetCurrentDeviceGcnArchName, hipModuleGetFunction, hipModuleLoadData, hipModuleUnload)
import ROCm.HIP.RTC (hiprtcCompileProgram, hiprtcGetCode, withHiprtcProgram)

data LoadedHipKernel = LoadedHipKernel
  { loadedHipKernelModule :: !HipModule
  , loadedHipKernelFunction :: !HipFunction
  }

compileHipKernel :: HasCallStack => DeviceId -> String -> String -> [String] -> IO ByteString
compileHipKernel deviceId source kernelName extraOptions =
  withDeviceId deviceId $ do
    archName <- hipGetCurrentDeviceGcnArchName
    let options = "-std=c++17" : ("--offload-arch=" <> archName) : extraOptions
    withHiprtcProgram source kernelName $ \program ->
      hiprtcCompileProgram program options `catch` rethrow archName
      >> hiprtcGetCode program
  where
    rethrow :: String -> FFIError -> IO a
    rethrow archName err =
      throwFFIError
        (ffiLibrary err)
        ("compileHipKernel:" <> kernelName)
        (ffiStatus err)
        ("kernel=" <> kernelName <> ", arch=" <> archName <> "\n" <> ffiMessage err)

loadHipKernel :: HasCallStack => ByteString -> String -> IO LoadedHipKernel
loadHipKernel code kernelName = do
  hipModule <- hipModuleLoadData code
  hipFunction <- hipModuleGetFunction hipModule kernelName
  pure
    LoadedHipKernel
      { loadedHipKernelModule = hipModule
      , loadedHipKernelFunction = hipFunction
      }

unloadHipKernel :: HasCallStack => LoadedHipKernel -> IO ()
unloadHipKernel = hipModuleUnload . loadedHipKernelModule
