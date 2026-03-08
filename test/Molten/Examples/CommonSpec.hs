module Molten.Examples.CommonSpec (spec) where

import qualified Data.Map.Strict as Map
import Molten.Examples.Common
  ( Timed(..)
  , approxEqAbsolute
  , approxEqRelative
  , measureOnce
  , measureRepeated
  , parseFlagMap
  )
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "measureOnce" $ do
    it "returns a non-negative duration" $ do
      timed <- measureOnce (pure (42 :: Int))
      timedValue timed `shouldBe` 42
      timedSeconds timed `shouldSatisfy` (>= 0)

  describe "measureRepeated" $ do
    it "collects the requested number of samples" $ do
      samples <- measureRepeated 3 (pure ())
      length samples `shouldBe` 3
      map timedSeconds samples `shouldSatisfy` all (>= 0)

  describe "approx helpers" $ do
    it "accepts close values by absolute tolerance" $ do
      approxEqAbsolute 1.0e-3 (1.0 :: Double) 1.0005 `shouldBe` True

    it "rejects distant values by relative tolerance" $ do
      approxEqRelative 1.0e-4 (1.0 :: Double) 1.1 `shouldBe` False

  describe "parseFlagMap" $ do
    it "parses --name value pairs" $ do
      parseFlagMap ["--batch", "64", "--hidden", "256"]
        `shouldBe` Right (Map.fromList [("batch", "64"), ("hidden", "256")])
