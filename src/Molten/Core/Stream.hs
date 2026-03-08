module Molten.Core.Stream
  ( Stream
  , createStream
  , destroyStream
  , streamDeviceId
  , synchronizeStream
  , withRawHipStream
  ) where

import qualified Foreign.Concurrent as FC
import Foreign.ForeignPtr (finalizeForeignPtr, withForeignPtr)
import GHC.Stack (HasCallStack)
import Molten.Core.Types (DeviceId, Stream(..))
import Molten.Internal.Device (withDeviceId)
import ROCm.FFI.Core.Types (HipStream(..))
import ROCm.HIP (hipStreamCreate, hipStreamDestroy, hipStreamSynchronize)

createStream :: HasCallStack => DeviceId -> IO Stream
createStream deviceId =
  withDeviceId deviceId $ do
    rawStream@(HipStream rawPtr) <- hipStreamCreate
    streamPtr <- FC.newForeignPtr rawPtr (hipStreamDestroy rawStream)
    pure Stream {streamDeviceId = deviceId, streamForeignPtr = streamPtr}

destroyStream :: Stream -> IO ()
destroyStream = finalizeForeignPtr . streamForeignPtr

withRawHipStream :: Stream -> (HipStream -> IO a) -> IO a
withRawHipStream stream action =
  withForeignPtr (streamForeignPtr stream) (action . HipStream)

synchronizeStream :: HasCallStack => Stream -> IO ()
synchronizeStream stream =
  withDeviceId (streamDeviceId stream) $ withRawHipStream stream hipStreamSynchronize
