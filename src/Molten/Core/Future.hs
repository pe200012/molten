module Molten.Core.Future
  ( GpuFuture
  , await
  , makeFutureFromStream
  ) where

import Control.Concurrent.MVar (MVar, modifyMVar, newMVar)
import Control.Exception (onException)
import GHC.Stack (HasCallStack)
import Molten.Core.Event (Event, createEvent, destroyEvent, recordEvent, synchronizeEvent)
import Molten.Core.Stream (Stream, streamDeviceId)

data GpuFuture a = GpuFuture
  { futureStream :: !Stream
  , futureEventState :: !(MVar (Maybe Event))
  , futureValue :: a
  }

makeFutureFromStream :: HasCallStack => Stream -> a -> IO (GpuFuture a)
makeFutureFromStream stream value = do
  event <- createEvent (streamDeviceId stream)
  recordEvent event stream `onException` destroyEvent event
  state <- newMVar (Just event)
  pure
    GpuFuture
      { futureStream = stream
      , futureEventState = state
      , futureValue = value
      }

await :: HasCallStack => GpuFuture a -> IO a
await future = do
  _ <-
    modifyMVar (futureEventState future) $ \maybeEvent ->
      case maybeEvent of
        Nothing -> pure (Nothing, ())
        Just event -> do
          synchronizeEvent event
          destroyEvent event
          pure (Nothing, ())
  pure (futureValue future)
