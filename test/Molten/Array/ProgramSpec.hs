{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TypeApplications #-}

module Molten.Array.ProgramSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import Molten.Array.Expr (Binary(..), Unary(..), constant, (.+.))
import Molten.Array.Program
  ( broadcastRowsP
  , buildProgram
  , forLoopP
  , fillArrayP
  , inputArray
  , mapExpr
  , programNodeDependencies
  , programNodeIds
  , programResultValue
  , randUniformP
  , reduceAll
  , reshapeValue
  , scheduleProgram
  , softmaxRowsP
  , sumRowsP
  , valueSize
  , withProgramRuntime
  , zipWithExpr
  )
import Molten.RAND.Runtime (RandGeneratorConfig(..))
import Molten.TestSupport (withGpuContext)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import ROCm.RocRAND (pattern RocRandRngPseudoDefault)
import Molten.Internal.Scheduler (ScheduledNode(scheduledNodeStreamSlot))
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy, shouldThrow)

spec :: Spec
spec = do
  describe "buildProgram" $ do
    it "produces a stable SSA node ordering" $ do
      program <-
        buildProgram $ do
          input0 <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 4)
          input1 <- fillArrayP @A.Ix1 @Int32 5 (A.Sz1 4)
          mapped <- mapExpr (Unary (\x -> x .+. constant 1)) input0
          zipped <- zipWithExpr (Binary (\x y -> x .+. y)) mapped input1
          reduceAll (Binary (\x y -> x .+. y)) 0 zipped
      programNodeIds program `shouldBe` [0, 1, 2, 3, 4]
      programNodeDependencies program `shouldBe` [(0, []), (1, []), (2, [0]), (3, [2, 1]), (4, [3])]

    it "rejects shape-mismatched zipWithExpr inputs" $
      buildProgram
        (do
          left <- fillArrayP @A.Ix1 @Int32 1 (A.Sz1 4)
          right <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 5)
          zipWithExpr (Binary (\x y -> x .+. y)) left right)
        `shouldThrow` isArgumentError "zipWithExpr" "same shape"

    it "gives reduceAll a fixed Ix1 length-1 result" $ do
      program <-
        buildProgram $ do
          input0 <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 4)
          reduceAll (Binary (\x y -> x .+. y)) 0 input0
      valueSize (programResultValue program) `shouldBe` A.Sz1 1

    it "accepts host array inputs in Program" $ do
      let arr =
            (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 4) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix1 Int32))
              :: A.Array A.S A.Ix1 Int32
      program <-
        buildProgram $ do
          input0 <- inputArray arr
          mapExpr (Unary (\x -> x .+. constant 1)) input0
      programNodeIds program `shouldBe` [0, 1]
      programNodeDependencies program `shouldBe` [(0, []), (1, [0])]

    it "gives sumRowsP and broadcastRowsP the expected shapes" $ do
      let input = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 3) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix2 Int32)
      program <-
        buildProgram $ do
          input0 <- inputArray input
          rowSums <- sumRowsP input0
          broadcastRowsP 3 rowSums
      valueSize (programResultValue program) `shouldBe` A.Sz2 2 3

    it "gives softmaxRowsP the same Ix2 shape as its input" $ do
      let input = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 3) (fromIntegral :: Int -> Float) :: A.Array A.D A.Ix2 Float)
      program <-
        buildProgram $ do
          input0 <- inputArray input
          softmaxRowsP input0
      valueSize (programResultValue program) `shouldBe` A.Sz2 2 3

    it "rejects negative broadcast dimensions" $
      buildProgram
        (do
          input0 <- fillArrayP @A.Ix1 @Int32 1 (A.Sz1 4)
          broadcastRowsP (-1) input0)
        `shouldThrow` isArgumentError "broadcastRowsP" "non-negative"

    it "rejects nested forLoopP builders" $
      buildProgram
        (do
          input0 <- fillArrayP @A.Ix1 @Int32 1 (A.Sz1 2)
          forLoopP 1 input0 (\loopValue -> forLoopP 1 loopValue pure))
        `shouldThrow` isArgumentError "forLoopP" "nested loops"

    it "rejects reshapeValue when totalElem changes" $
      buildProgram
        (do
          input0 <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 4)
          reshapeValue (A.Sz2 3 2) input0)
        `shouldThrow` isArgumentError "reshapeValue" "preserve totalElem"

  describe "scheduleProgram" $ do
    it "assigns independent JIT branches to different stream slots before they merge" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          program <-
            buildProgram $ do
              left <- fillArrayP @A.Ix1 @Int32 1 (A.Sz1 8)
              right <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 8)
              mappedLeft <- mapExpr (Unary (\x -> x .+. constant 1)) left
              mappedRight <- mapExpr (Unary (\x -> x .+. constant 2)) right
              zipWithExpr (Binary (\x y -> x .+. y)) mappedLeft mappedRight
          scheduled <- scheduleProgram runtime program
          let slots = fmap scheduledNodeStreamSlot scheduled
          slots `shouldSatisfy` (\xs -> length xs >= 4 && xs !! 0 /= xs !! 1)

    it "serializes RAND nodes onto the conservative shared-resource stream" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let config = RandGeneratorConfig RocRandRngPseudoDefault 20260308
          program <-
            buildProgram $ do
              _ <- randUniformP @A.Ix1 @Float config (A.Sz1 16)
              randUniformP @A.Ix1 @Float config (A.Sz1 16)
          scheduled <- scheduleProgram runtime program
          fmap scheduledNodeStreamSlot scheduled `shouldBe` [0, 0]

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
