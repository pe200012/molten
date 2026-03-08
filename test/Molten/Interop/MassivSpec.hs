{-# LANGUAGE TypeApplications #-}

module Molten.Interop.MassivSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import Molten.Interop.Massiv
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "withHostBufferFromArray/readHostBufferToArray" $ do
    it "roundtrips an Ix2 array through a Host buffer" $ do
      let arr =
            (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 3) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix2 Int32))
              :: A.Array A.S A.Ix2 Int32
      withHostBufferFromArray arr $ \sz hostBuf -> do
        sz `shouldBe` A.Sz2 2 3
        out <- readHostBufferToArray sz hostBuf
        AM.toStorableVector out `shouldBe` AM.toStorableVector arr

    it "materializes delayed input before copying into a Host buffer" $ do
      let arr =
            A.makeArrayLinear A.Seq (A.Sz2 2 3) (fromIntegral @Int @Int32)
              :: A.Array A.D A.Ix2 Int32
          expected = A.computeAs A.S arr
      withHostBufferFromArray arr $ \sz hostBuf -> do
        out <- readHostBufferToArray sz hostBuf
        AM.toStorableVector out `shouldBe` AM.toStorableVector expected

  describe "withPinnedBufferFromArray/readPinnedBufferToArray" $ do
    it "roundtrips an Ix1 array through a PinnedHost buffer" $ do
      let arr =
            (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 5) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix1 Int32))
              :: A.Array A.S A.Ix1 Int32
      withPinnedBufferFromArray arr $ \sz pinnedBuf -> do
        sz `shouldBe` A.Sz1 5
        out <- readPinnedBufferToArray sz pinnedBuf
        AM.toStorableVector out `shouldBe` AM.toStorableVector arr

  describe "shape validation on rebuild" $ do
    it "rejects mismatched size when rebuilding from a Host buffer" $ do
      let arr =
            (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 3) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix1 Int32))
              :: A.Array A.S A.Ix1 Int32
      withHostBufferFromArray arr $ \_ hostBuf ->
        readHostBufferToArray (A.Sz2 2 2) hostBuf `shouldThrow` isArgumentError "readHostBufferToArray" "totalElem"

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
