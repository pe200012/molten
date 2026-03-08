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
  , buildProgram
  , fillArrayP
  , gemmMatrixP
  , inputArray
  , inputDeviceArray
  , mapExpr
  , reduceAll
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
