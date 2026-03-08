{-# LANGUAGE TypeApplications #-}

module Molten.Array.DeviceSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import Molten.Array.Device
import Molten.Core.Buffer (bufferLength, withDeviceBuffer)
import Molten.TestSupport (withGpuContext)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "mkDeviceVector" $ do
    it "wraps a device buffer with Ix1 shape" $
      withGpuContext $ \ctx ->
        withDeviceBuffer @Int32 ctx 6 $ \buf -> do
          dev <- mkDeviceVector 6 buf
          deviceArraySize dev `shouldBe` A.Sz1 6
          bufferLength (deviceArrayBuffer dev) `shouldBe` 6

  describe "mkDeviceMatrix" $ do
    it "wraps a device buffer with Ix2 shape" $
      withGpuContext $ \ctx ->
        withDeviceBuffer @Int32 ctx 6 $ \buf -> do
          dev <- mkDeviceMatrix 2 3 buf
          deviceArraySize dev `shouldBe` A.Sz2 2 3
          bufferLength (deviceArrayBuffer dev) `shouldBe` 6

    it "rejects mismatched size and device buffer length" $
      withGpuContext $ \ctx ->
        withDeviceBuffer @Int32 ctx 5 $ \buf ->
          mkDeviceMatrix 2 3 buf `shouldThrow` isArgumentError "mkDeviceArray" "totalElem"

  describe "withDeviceVector/withDeviceMatrix" $ do
    it "rejects negative vector length" $
      withGpuContext $ \ctx ->
        withDeviceVector @Int32 ctx (-1) pure `shouldThrow` isArgumentError "withDeviceVector" "length"

    it "rejects negative matrix dimensions" $
      withGpuContext $ \ctx ->
        withDeviceMatrix @Int32 ctx (-1) 3 pure `shouldThrow` isArgumentError "withDeviceMatrix" "rows"

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
