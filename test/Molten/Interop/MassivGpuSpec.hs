{-# LANGUAGE TypeApplications #-}

module Molten.Interop.MassivGpuSpec (spec) where

import Data.Int (Int32)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import Molten.Array.Device (deviceArrayBuffer, withDeviceArray)
import Molten.Core.Context (contextDefaultStream)
import Molten.Core.Future (await)
import Molten.Core.Transfer (copyD2H, copyD2HAsync, copyH2D, copyH2DAsync)
import Molten.Interop.Massiv
  ( readHostBufferToArray
  , readPinnedBufferToArray
  , withHostBufferFromArray
  , withPinnedBufferFromArray
  )
import Molten.TestSupport (withGpuContext)
import Test.Hspec (Spec, describe, it, shouldBe)

spec :: Spec
spec = do
  describe "Host -> DeviceArray -> Host" $ do
    it "roundtrips an Ix2 array through device memory" $
      withGpuContext $ \ctx -> do
        let arr =
              (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 3) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix2 Int32))
                :: A.Array A.S A.Ix2 Int32
        withHostBufferFromArray arr $ \sz hostIn ->
          withDeviceArray ctx sz $ \dev ->
            withHostBufferFromArray arr $ \_ hostOut -> do
              copyH2D ctx (deviceArrayBuffer dev) hostIn
              copyD2H ctx hostOut (deviceArrayBuffer dev)
              out <- readHostBufferToArray sz hostOut
              AM.toStorableVector out `shouldBe` AM.toStorableVector arr

  describe "PinnedHost -> DeviceArray -> PinnedHost" $ do
    it "roundtrips an Ix1 array asynchronously through device memory" $
      withGpuContext $ \ctx -> do
        let arr =
              (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 8) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix1 Int32))
                :: A.Array A.S A.Ix1 Int32
            stream = contextDefaultStream ctx
        withPinnedBufferFromArray arr $ \sz pinnedIn ->
          withDeviceArray ctx sz $ \dev ->
            withPinnedBufferFromArray arr $ \_ pinnedOut -> do
              upload <- copyH2DAsync stream (deviceArrayBuffer dev) pinnedIn
              () <- await upload
              download <- copyD2HAsync stream pinnedOut (deviceArrayBuffer dev)
              () <- await download
              out <- readPinnedBufferToArray sz pinnedOut
              AM.toStorableVector out `shouldBe` AM.toStorableVector arr
