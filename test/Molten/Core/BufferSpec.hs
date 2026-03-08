{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeApplications #-}

module Molten.Core.BufferSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.List (isInfixOf)
import Molten.Core.Buffer (bufferLength, withHostBuffer)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldReturn, shouldThrow)

spec :: Spec
spec = do
  describe "withHostBuffer" $ do
    it "rejects negative lengths" $
      withHostBuffer @Int (-1) (const (pure ()))
        `shouldThrow` isArgumentError "withHostBuffer" "length must be >= 0"

    it "creates a buffer with the requested length" $
      withHostBuffer @Int 4 (pure . bufferLength)
        `shouldReturn` 4

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
