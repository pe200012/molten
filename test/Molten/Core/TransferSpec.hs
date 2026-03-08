{-# LANGUAGE TypeApplications #-}

module Molten.Core.TransferSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Vector.Storable as VS
import Molten.Core.Buffer (withDeviceBuffer, withHostBuffer, withPinnedBuffer)
import Molten.Core.Context (contextDefaultStream)
import Molten.Core.Future (await)
import Molten.Core.Transfer (copyD2H, copyD2HAsync, copyH2D, copyH2DAsync)
import Molten.Interop.Vector
  ( readHostBufferToVector
  , readPinnedBufferToVector
  , withHostBufferFromVector
  , withPinnedBufferFromVector
  )
import Molten.TestSupport (withGpuContext)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "copyH2D/copyD2H" $ do
    it "roundtrips host buffers through device memory" $
      withGpuContext $ \ctx -> do
        let input = VS.fromList [0 .. 15 :: Int32]
            n = VS.length input
        withHostBufferFromVector input $ \hostIn ->
          withDeviceBuffer @Int32 ctx n $ \deviceBuf ->
            withHostBuffer @Int32 n $ \hostOut -> do
              copyH2D ctx deviceBuf hostIn
              copyD2H ctx hostOut deviceBuf
              output <- readHostBufferToVector hostOut
              output `shouldBe` input

    it "rejects mismatched synchronous copy lengths" $
      withGpuContext $ \ctx ->
        withHostBufferFromVector (VS.fromList [1, 2, 3 :: Int32]) $ \hostIn ->
          withDeviceBuffer @Int32 ctx 2 $ \deviceBuf ->
            copyH2D ctx deviceBuf hostIn
              `shouldThrow` isArgumentError "copyH2D" "same length"

  describe "copyH2DAsync/copyD2HAsync" $ do
    it "roundtrips pinned buffers through device memory" $
      withGpuContext $ \ctx -> do
        let input = VS.fromList [0 .. 15 :: Int32]
            n = VS.length input
            stream = contextDefaultStream ctx
        withPinnedBufferFromVector input $ \pinnedIn ->
          withDeviceBuffer @Int32 ctx n $ \deviceBuf ->
            withPinnedBuffer @Int32 n $ \pinnedOut -> do
              upload <- copyH2DAsync stream deviceBuf pinnedIn
              () <- await upload
              download <- copyD2HAsync stream pinnedOut deviceBuf
              () <- await download
              output <- readPinnedBufferToVector pinnedOut
              output `shouldBe` input

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
