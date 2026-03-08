module Molten.Core.Context
  ( Context
  , DeviceId(..)
  , contextDefaultStream
  , contextDeviceId
  , withContext
  , withContextBlasHandle
  , withContextDevice
  , withEvent
  , withStream
  ) where

import Control.Exception (bracket, onException)
import qualified Foreign.Concurrent as FC
import Foreign.ForeignPtr (finalizeForeignPtr, withForeignPtr)
import GHC.Stack (HasCallStack)
import Molten.Core.Event (Event, createEvent, destroyEvent)
import Molten.Core.Stream (Stream, createStream, destroyStream, withRawHipStream)
import Molten.Core.Types (BlasHandle(..), Context(..), DeviceId(..))
import Molten.Internal.Device (withDeviceId)
import ROCm.FFI.Core.Types (RocblasHandle(..))
import ROCm.RocBLAS (rocblasCreateHandle, rocblasDestroyHandle, rocblasInitialize, rocblasSetStream)

withContext :: HasCallStack => DeviceId -> (Context -> IO a) -> IO a
withContext deviceId = bracket (createContext deviceId) destroyContext

withStream :: HasCallStack => Context -> (Stream -> IO a) -> IO a
withStream ctx = bracket (createStream (contextDeviceId ctx)) destroyStream

withEvent :: HasCallStack => Context -> (Event -> IO a) -> IO a
withEvent ctx = bracket (createEvent (contextDeviceId ctx)) destroyEvent

withContextDevice :: Context -> IO a -> IO a
withContextDevice ctx = withDeviceId (contextDeviceId ctx)

withContextBlasHandle :: Context -> (RocblasHandle -> IO a) -> IO a
withContextBlasHandle ctx action =
  withContextDevice ctx $
    withForeignPtr (blasHandleForeignPtr (contextBlasHandle ctx)) (action . RocblasHandle)

createContext :: HasCallStack => DeviceId -> IO Context
createContext deviceId =
  withDeviceId deviceId $ do
    defaultStream <- createStream deviceId
    blasHandle <- createBlasHandle deviceId defaultStream `onException` destroyStream defaultStream
    pure
      Context
        { contextDeviceId = deviceId
        , contextDefaultStream = defaultStream
        , contextBlasHandle = blasHandle
        }

destroyContext :: Context -> IO ()
destroyContext ctx = do
  finalizeForeignPtr (blasHandleForeignPtr (contextBlasHandle ctx))
  destroyStream (contextDefaultStream ctx)

createBlasHandle :: HasCallStack => DeviceId -> Stream -> IO BlasHandle
createBlasHandle deviceId stream = do
  rocblasInitialize
  rawHandle@(RocblasHandle rawPtr) <- rocblasCreateHandle
  withRawHipStream stream (rocblasSetStream rawHandle) `onException` rocblasDestroyHandle rawHandle
  handlePtr <- FC.newForeignPtr rawPtr (rocblasDestroyHandle rawHandle)
  pure BlasHandle {blasHandleDeviceId = deviceId, blasHandleForeignPtr = handlePtr}
