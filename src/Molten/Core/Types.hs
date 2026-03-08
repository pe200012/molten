module Molten.Core.Types
  ( BlasHandle(..)
  , Buffer(..)
  , Context(..)
  , DeviceId(..)
  , Event(..)
  , Location(..)
  , Stream(..)
  ) where

import Foreign.ForeignPtr (ForeignPtr)
import ROCm.FFI.Core.Types (HipEventTag, HipStreamTag, RocblasHandleTag)

newtype DeviceId = DeviceId Int
  deriving (Eq, Ord, Show)

data Stream = Stream
  { streamDeviceId :: !DeviceId
  , streamForeignPtr :: !(ForeignPtr HipStreamTag)
  }

data Event = Event
  { eventDeviceId :: !DeviceId
  , eventForeignPtr :: !(ForeignPtr HipEventTag)
  }

data BlasHandle = BlasHandle
  { blasHandleDeviceId :: !DeviceId
  , blasHandleForeignPtr :: !(ForeignPtr RocblasHandleTag)
  }

data Context = Context
  { contextDeviceId :: !DeviceId
  , contextDefaultStream :: !Stream
  , contextBlasHandle :: !BlasHandle
  }

data Location = Host | PinnedHost | Device


data Buffer (loc :: Location) a where
  HostBuffer ::
    { hostBufferForeignPtr :: !(ForeignPtr a)
    , hostBufferLength :: !Int
    } -> Buffer 'Host a
  PinnedHostBuffer ::
    { pinnedHostBufferForeignPtr :: !(ForeignPtr a)
    , pinnedHostBufferLength :: !Int
    } -> Buffer 'PinnedHost a
  DeviceBuffer ::
    { deviceBufferDeviceId :: !DeviceId
    , deviceBufferForeignPtr :: !(ForeignPtr a)
    , deviceBufferLength :: !Int
    } -> Buffer 'Device a
