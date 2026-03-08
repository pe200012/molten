module Molten.Core.ContextSpec (spec) where

import Molten.Core.Context (DeviceId(..), withContext)
import ROCm.HIP.Device (hipGetDeviceCount)
import Test.Hspec (Spec, describe, it, pendingWith)

spec :: Spec
spec = do
  describe "withContext" $ do
    it "initializes the default BLAS handle during context creation" $ do
      deviceCount <- hipGetDeviceCount
      if deviceCount <= 0
        then pendingWith "ROCm GPU not available"
        else withContext (DeviceId 0) (const (pure ()))
