{-# LANGUAGE TypeApplications #-}

module Molten.Reference.Axis2DSpec (spec) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Data.Int (Int32)
import Molten.Reference
  ( broadcastColsRef
  , broadcastRowsRef
  , maxColsRef
  , maxRowsRef
  , sumColsRef
  , sumRowsRef
  )
import Test.Hspec (Spec, describe, it, shouldBe)

spec :: Spec
spec = do
  describe "sumRowsRef" $ do
    it "reduces each row of an Ix2 array" $ do
      let input = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 2, 3, 4, 5, 6 :: Int32])
      output <- sumRowsRef input
      AM.toStorableVector output `shouldBe` VS.fromList [6, 15 :: Int32]

  describe "sumColsRef" $ do
    it "reduces each column of an Ix2 array" $ do
      let input = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 2, 3, 4, 5, 6 :: Int32])
      output <- sumColsRef input
      AM.toStorableVector output `shouldBe` VS.fromList [5, 7, 9 :: Int32]

  describe "maxRowsRef" $ do
    it "takes the maximum of each row" $ do
      let input = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 9, 3, 4, 2, 6 :: Int32])
      output <- maxRowsRef input
      AM.toStorableVector output `shouldBe` VS.fromList [9, 6 :: Int32]

  describe "maxColsRef" $ do
    it "takes the maximum of each column" $ do
      let input = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 9, 3, 4, 2, 6 :: Int32])
      output <- maxColsRef input
      AM.toStorableVector output `shouldBe` VS.fromList [4, 9, 6 :: Int32]

  describe "broadcastRowsRef" $ do
    it "broadcasts an Ix1 row summary across columns" $ do
      let input = AMV.fromVector' A.Seq (A.Sz1 2) (VS.fromList [10, 20 :: Int32])
      output <- broadcastRowsRef 3 input
      AM.toStorableVector output `shouldBe` VS.fromList [10, 10, 10, 20, 20, 20 :: Int32]

  describe "broadcastColsRef" $ do
    it "broadcasts an Ix1 column summary across rows" $ do
      let input = AMV.fromVector' A.Seq (A.Sz1 3) (VS.fromList [10, 20, 30 :: Int32])
      output <- broadcastColsRef 2 input
      AM.toStorableVector output `shouldBe` VS.fromList [10, 20, 30, 10, 20, 30 :: Int32]
