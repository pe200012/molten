{-# LANGUAGE TypeApplications #-}

module Molten.Reference.ProgramSpec (spec) where

import Control.Exception (SomeException, fromException)
import Data.Int (Int32)
import Data.List (isInfixOf)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Expr (Binary(..), Unary(..), constant, (.+.), (.*.))
import Molten.Array.Program
  ( MatrixGemmValue(..)
  , broadcastRowsP
  , buildProgram
  , forLoopP
  , fillArrayP
  , gemmMatrixP
  , inputArray
  , inputDeviceArray
  , mapExpr
  , reduceAll
  , softmaxRowsP
  , sumRowsP
  , zipWithExpr
  )
import Molten.Array.Transfer (copyHostArrayToDevice)
import Molten.BLAS.Types (Transpose(..))
import Molten.Reference (runProgramCpu)
import Molten.TestSupport (withGpuContext)
import ROCm.FFI.Core.Exception (ArgumentError(..))
import Test.Hspec (Spec, describe, it, shouldBe, shouldThrow)

spec :: Spec
spec = do
  describe "runProgramCpu" $ do
    it "runs fill -> map -> zipWith -> reduceAll on CPU" $ do
      program <-
        buildProgram $ do
          base <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 4)
          mapped <- mapExpr (Unary (\x -> x .+. constant 1)) base
          zipped <- mapExpr (Unary (\x -> x .*. constant 2)) mapped
          reduceAll (Binary (\x y -> x .+. y)) 0 zipped
      resultArray <- runProgramCpu program
      AM.toStorableVector resultArray `shouldBe` VS.fromList [24 :: Int32]

    it "runs inputArray -> mapExpr on CPU" $ do
      let input = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
      program <-
        buildProgram $ do
          value0 <- inputArray input
          mapExpr (Unary (\x -> x .+. constant 1)) value0
      resultArray <- runProgramCpu program
      AM.toStorableVector resultArray `shouldBe` VS.fromList [2, 3, 4, 5 :: Int32]

    it "runs gemmMatrixP on CPU" $ do
      let a = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [1, 2, 3, 4 :: Float])
          b = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [5, 6, 7, 8 :: Float])
          c = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [0, 0, 0, 0 :: Float])
      program <-
        buildProgram $ do
          aValue <- inputArray a
          bValue <- inputArray b
          cValue <- inputArray c
          gemmMatrixP
            MatrixGemmValue
              { matrixGemmValueTransA = NoTranspose
              , matrixGemmValueTransB = NoTranspose
              , matrixGemmValueAlpha = 1
              , matrixGemmValueA = aValue
              , matrixGemmValueB = bValue
              , matrixGemmValueBeta = 0
              , matrixGemmValueC = cValue
              }
      resultArray <- runProgramCpu program
      AM.toStorableVector resultArray `shouldBe` VS.fromList [19, 22, 43, 50 :: Float]

    it "runs sumRowsP -> broadcastRowsP on CPU" $ do
      let input = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 2, 3, 4, 5, 6 :: Int32])
      program <-
        buildProgram $ do
          value0 <- inputArray input
          rowSums <- sumRowsP value0
          broadcastRowsP 3 rowSums
      resultArray <- runProgramCpu program
      AM.toStorableVector resultArray `shouldBe` VS.fromList [6, 6, 6, 15, 15, 15 :: Int32]

    it "runs softmaxRowsP on CPU" $ do
      let input = AMV.fromVector' A.Seq (A.Sz2 1 3) (VS.fromList [0, 0, 0 :: Float])
      program <-
        buildProgram $ do
          value0 <- inputArray input
          softmaxRowsP value0
      resultArray <- runProgramCpu program
      AM.toStorableVector resultArray `shouldBe` VS.fromList [1 / 3, 1 / 3, 1 / 3 :: Float]

    it "runs forLoopP with zero iterations on CPU" $ do
      let input = AMV.fromVector' A.Seq (A.Sz1 3) (VS.fromList [1, 2, 3 :: Int32])
      program <-
        buildProgram $ do
          value0 <- inputArray input
          forLoopP 0 value0 (mapExpr (Unary (\x -> x .+. constant 1)))
      resultArray <- runProgramCpu program
      AM.toStorableVector resultArray `shouldBe` VS.fromList [1, 2, 3 :: Int32]

    it "runs forLoopP with tuple state on CPU" $ do
      let left0 = AMV.fromVector' A.Seq (A.Sz1 2) (VS.fromList [1, 1 :: Int32])
          right0 = AMV.fromVector' A.Seq (A.Sz1 2) (VS.fromList [0, 0 :: Int32])
      program <-
        buildProgram $ do
          leftValue <- inputArray left0
          rightValue <- inputArray right0
          forLoopP 2 (leftValue, rightValue) $ \(leftState, rightState) -> do
            leftNext <- mapExpr (Unary (\x -> x .+. constant 1)) leftState
            rightNext <- zipWithExpr (Binary (\x y -> x .+. y)) leftState rightState
            pure (leftNext, rightNext)
      (leftResult, rightResult) <- runProgramCpu program
      AM.toStorableVector leftResult `shouldBe` VS.fromList [3, 3 :: Int32]
      AM.toStorableVector rightResult `shouldBe` VS.fromList [3, 3 :: Int32]

    it "rejects device-only inputs" $
      withGpuContext $ \ctx -> do
        let input = (AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])) :: A.Array A.S A.Ix1 Int32
        deviceInput <- copyHostArrayToDevice ctx input
        program <- buildProgram (inputDeviceArray deviceInput)
        runProgramCpu program `shouldThrow` isArgumentError "runProgramCpu" "device-only inputs"

isArgumentError :: String -> String -> SomeException -> Bool
isArgumentError functionName messageFragment err =
  case fromException err of
    Just argErr ->
      argFunction argErr == functionName
        && messageFragment `isInfixOf` argMessage argErr
    Nothing -> False
