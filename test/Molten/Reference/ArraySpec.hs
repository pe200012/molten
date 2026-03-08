{-# LANGUAGE TypeApplications #-}

module Molten.Reference.ArraySpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Expr (Binary(..), Unary(..), constant, (.+.))
import Molten.Reference (mapArrayRef, reduceAllArrayRef, reshapeArrayRef, zipWithArrayRef)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "mapArrayRef" $ do
    it "maps over an Ix1 array with the Exp interpreter" $ do
      let input = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
      output <- mapArrayRef (Unary (\x -> x .+. constant 1)) input
      AM.toStorableVector output `shouldBe` VS.fromList [2, 3, 4, 5 :: Int32]

  describe "zipWithArrayRef" $ do
    it "zips two arrays with the Exp interpreter" $ do
      let left = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
          right = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [5, 6, 7, 8 :: Int32])
      output <- zipWithArrayRef (Binary (\x y -> x .+. y)) left right
      AM.toStorableVector output `shouldBe` VS.fromList [6, 8, 10, 12 :: Int32]

    it "rejects shape mismatches" $ do
      let left = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
          right = AMV.fromVector' A.Seq (A.Sz1 5) (VS.fromList [1, 2, 3, 4, 5 :: Int32])
      zipWithArrayRef (Binary (\x y -> x .+. y)) left right `shouldThrow` isArgumentError "zipWithArrayRef" "same shape"

  describe "reduceAllArrayRef" $ do
    it "reduces in row-major linear order" $ do
      let input = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 2, 3, 4, 5, 6 :: Int32])
      output <- reduceAllArrayRef (Binary (\x y -> x .+. y)) 0 input
      AM.toStorableVector output `shouldBe` VS.fromList [21 :: Int32]

  describe "reshapeArrayRef" $ do
    it "reshapes without changing linear order" $ do
      let input = AMV.fromVector' A.Seq (A.Sz1 6) (VS.fromList [1, 2, 3, 4, 5, 6 :: Int32])
      output <- reshapeArrayRef (A.Sz2 2 3) input
      AM.toStorableVector output `shouldBe` VS.fromList [1, 2, 3, 4, 5, 6 :: Int32]

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
