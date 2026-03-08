{-# LANGUAGE TypeApplications #-}

module Molten.Array.TransferSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import Molten.Array.Device (deviceArraySize, withDeviceArray)
import Molten.Array.Transfer
  ( cloneDeviceArray
  , copyHostArrayToDevice
  , copyPinnedArrayToDeviceAsync
  , readDeviceArrayToHostArray
  , readDeviceArrayToPinnedArrayAsync
  , reshapeDeviceArray
  )
import qualified Molten.Core.Context as Context
import Molten.Core.Future (await)
import Molten.TestSupport (withGpuContext)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "reshapeDeviceArray" $ do
    it "reshapes when the total element count is preserved" $
      withGpuContext $ \ctx ->
        withDeviceArray @A.Ix1 @Int32 ctx (A.Sz1 6) $ \dev -> do
          reshaped <- reshapeDeviceArray (A.Sz2 2 3) dev
          deviceArraySize reshaped `shouldBe` A.Sz2 2 3

    it "rejects reshape when the total element count changes" $
      withGpuContext $ \ctx ->
        withDeviceArray @A.Ix1 @Int32 ctx (A.Sz1 6) $ \dev ->
          reshapeDeviceArray (A.Sz2 4 2) dev `shouldThrow` isArgumentError "reshapeDeviceArray" "totalElem"

  describe "cloneDeviceArray" $ do
    it "copies device contents into a new device array" $
      withGpuContext $ \ctx -> do
        let arr = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 6) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix1 Int32)
        dev <- copyHostArrayToDevice ctx arr
        cloned <- cloneDeviceArray ctx dev
        out <- readDeviceArrayToHostArray ctx cloned
        AM.toStorableVector out `shouldBe` AM.toStorableVector arr

  describe "copyHostArrayToDevice/readDeviceArrayToHostArray" $ do
    it "roundtrips a host array through device memory" $
      withGpuContext $ \ctx -> do
        let arr = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 3) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix2 Int32)
        dev <- copyHostArrayToDevice ctx arr
        out <- readDeviceArrayToHostArray ctx dev
        AM.toStorableVector out `shouldBe` AM.toStorableVector arr

  describe "copyPinnedArrayToDeviceAsync/readDeviceArrayToPinnedArrayAsync" $ do
    it "roundtrips a pinned array asynchronously through device memory" $
      withGpuContext $ \ctx -> do
        let arr = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 8) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix1 Int32)
            stream = Context.contextDefaultStream ctx
        upload <- copyPinnedArrayToDeviceAsync stream arr
        dev <- await upload
        download <- readDeviceArrayToPinnedArrayAsync stream dev
        out <- await download
        AM.toStorableVector out `shouldBe` AM.toStorableVector arr

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
