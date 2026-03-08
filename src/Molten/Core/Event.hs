module Molten.Core.Event
  ( Event
  , createEvent
  , destroyEvent
  , eventDeviceId
  , eventReady
  , recordEvent
  , synchronizeEvent
  , withRawHipEvent
  ) where

import qualified Foreign.Concurrent as FC
import Foreign.ForeignPtr (finalizeForeignPtr, withForeignPtr)
import GHC.Stack (HasCallStack)
import Molten.Core.Stream (Stream, streamDeviceId, withRawHipStream)
import Molten.Core.Types (DeviceId, Event(..))
import Molten.Internal.Device (withDeviceId)
import Molten.Internal.Validation (ensureDeviceMatch)
import ROCm.FFI.Core.Types (HipEvent(..))
import ROCm.HIP (hipEventCreate, hipEventDestroy, hipEventQuery, hipEventRecord, hipEventSynchronize)

createEvent :: HasCallStack => DeviceId -> IO Event
createEvent deviceId =
  withDeviceId deviceId $ do
    rawEvent@(HipEvent rawPtr) <- hipEventCreate
    eventPtr <- FC.newForeignPtr rawPtr (hipEventDestroy rawEvent)
    pure Event {eventDeviceId = deviceId, eventForeignPtr = eventPtr}

destroyEvent :: Event -> IO ()
destroyEvent = finalizeForeignPtr . eventForeignPtr

withRawHipEvent :: Event -> (HipEvent -> IO a) -> IO a
withRawHipEvent event action =
  withForeignPtr (eventForeignPtr event) (action . HipEvent)

recordEvent :: HasCallStack => Event -> Stream -> IO ()
recordEvent event stream = do
  ensureDeviceMatch "recordEvent" (eventDeviceId event) (streamDeviceId stream)
  withDeviceId (eventDeviceId event) $
    withRawHipEvent event $ \rawEvent ->
      withRawHipStream stream $ \rawStream ->
        hipEventRecord rawEvent rawStream

synchronizeEvent :: HasCallStack => Event -> IO ()
synchronizeEvent event =
  withDeviceId (eventDeviceId event) $ withRawHipEvent event hipEventSynchronize

eventReady :: HasCallStack => Event -> IO Bool
eventReady event =
  withDeviceId (eventDeviceId event) $ withRawHipEvent event hipEventQuery
