{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TypeApplications #-}

module Molten.Array.ProgramGpuSpec (spec) where

import Data.Complex (Complex((:+)))
import Data.Int (Int32)
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Molten.Array.Expr (Binary(..), Unary(..), constant, (.+.), (.*.))
import Molten.Array.Program
  ( MatrixGemmValue(..)
  , buildProgram
  , fftForwardP
  , fftInverseP
  , fillArrayP
  , forLoopP
  , gemmMatrixP
  , inputArray
  , inputDeviceArray
  , mapExpr
  , randUniformP
  , reduceAll
  , runProgram
  , softmaxRowsP
  , withProgramRuntime
  , zipWithExpr
  )
import Molten.Array.Transfer (copyHostArrayToDevice, readDeviceArrayToHostArray)
import Molten.BLAS.Types (Transpose(NoTranspose))
import Molten.RAND.Runtime (RandGeneratorConfig(..))
import Molten.Reference (runProgramCpu)
import Molten.TestSupport (withGpuContext)
import ROCm.RocRAND (pattern RocRandRngPseudoDefault)
import Test.Hspec (Spec, describe, it, shouldBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "JIT nodes" $ do
    it "executes fill -> map -> zipWith -> reduceAll" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          program <-
            buildProgram $ do
              base <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 4)
              mapped <- mapExpr (Unary (\x -> x .+. constant 1)) base
              zipped <- zipWithExpr (Binary (\x y -> x .*. y)) mapped base
              reduceAll (Binary (\x y -> x .+. y)) 0 zipped
          resultArray <- runProgram runtime program
          hostArray <- readDeviceArrayToHostArray ctx resultArray
          AM.toStorableVector hostArray `shouldBe` VS.fromList [24 :: Int32]

    it "waits across independent map branches before merging" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          program <-
            buildProgram $ do
              left0 <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 4)
              right0 <- fillArrayP @A.Ix1 @Int32 5 (A.Sz1 4)
              left1 <- mapExpr (Unary (\x -> x .+. constant 1)) left0
              right1 <- mapExpr (Unary (\x -> x .+. constant 2)) right0
              zipWithExpr (Binary (\x y -> x .+. y)) left1 right1
          resultArray <- runProgram runtime program
          hostArray <- readDeviceArrayToHostArray ctx resultArray
          AM.toStorableVector hostArray `shouldBe` VS.fromList [10, 10, 10, 10 :: Int32]

    it "uploads inputArray inputs before running on GPU" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let arr = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
          program <-
            buildProgram $ do
              input0 <- inputArray arr
              mapExpr (Unary (\x -> x .+. constant 1)) input0
          resultArray <- runProgram runtime program
          hostArray <- readDeviceArrayToHostArray ctx resultArray
          AM.toStorableVector hostArray `shouldBe` VS.fromList [2, 3, 4, 5 :: Int32]

    it "matches CPU reference for inputArray -> mapExpr -> reduceAll" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let input = (AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])) :: A.Array A.S A.Ix1 Int32
          program <-
            buildProgram $ do
              value0 <- inputArray input
              value1 <- mapExpr (Unary (\x -> x .+. constant 1)) value0
              reduceAll (Binary (\x y -> x .+. y)) 0 value1
          cpuResult <- runProgramCpu program
          gpuResult <- runProgram runtime program
          gpuHost <- readDeviceArrayToHostArray ctx gpuResult
          AM.toStorableVector gpuHost `shouldBe` AM.toStorableVector cpuResult

    it "matches CPU reference for inputArray -> gemmMatrixP" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let a = (AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [1, 2, 3, 4 :: Float])) :: A.Array A.S A.Ix2 Float
              b = (AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [5, 6, 7, 8 :: Float])) :: A.Array A.S A.Ix2 Float
              c0 = (AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [0, 0, 0, 0 :: Float])) :: A.Array A.S A.Ix2 Float
          program <-
            buildProgram $ do
              a0 <- inputArray a
              b0 <- inputArray b
              cInit <- inputArray c0
              gemmMatrixP
                MatrixGemmValue
                  { matrixGemmValueTransA = NoTranspose
                  , matrixGemmValueTransB = NoTranspose
                  , matrixGemmValueAlpha = 1
                  , matrixGemmValueA = a0
                  , matrixGemmValueB = b0
                  , matrixGemmValueBeta = 0
                  , matrixGemmValueC = cInit
                  }
          cpuResult <- runProgramCpu program
          gpuResult <- runProgram runtime program
          gpuHost <- readDeviceArrayToHostArray ctx gpuResult
          AM.toStorableVector gpuHost `shouldBe` AM.toStorableVector cpuResult

    it "matches CPU reference for inputArray -> softmaxRowsP" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let input = (AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 2, 3, 0, -1, 1 :: Float])) :: A.Array A.S A.Ix2 Float
          program <-
            buildProgram $ do
              value0 <- inputArray input
              softmaxRowsP value0
          cpuResult <- runProgramCpu program
          gpuResult <- runProgram runtime program
          gpuHost <- readDeviceArrayToHostArray ctx gpuResult
          approxFloatArray cpuResult gpuHost `shouldBe` True

    it "matches CPU reference for inputArray -> forLoopP map recurrence" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let input = (AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])) :: A.Array A.S A.Ix1 Int32
          program <-
            buildProgram $ do
              value0 <- inputArray input
              forLoopP 3 value0 (mapExpr (Unary (\x -> x .+. constant 1)))
          cpuResult <- runProgramCpu program
          gpuResult <- runProgram runtime program
          gpuHost <- readDeviceArrayToHostArray ctx gpuResult
          AM.toStorableVector gpuHost `shouldBe` AM.toStorableVector cpuResult

    it "matches CPU reference for forLoopP with outer invariant values" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let input = (AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])) :: A.Array A.S A.Ix1 Int32
              bias = (AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [10, 10, 10, 10 :: Int32])) :: A.Array A.S A.Ix1 Int32
          program <-
            buildProgram $ do
              state0 <- inputArray input
              bias0 <- inputArray bias
              forLoopP 2 state0 (\stateValue -> zipWithExpr (Binary (\x y -> x .+. y)) stateValue bias0)
          cpuResult <- runProgramCpu program
          gpuResult <- runProgram runtime program
          gpuHost <- readDeviceArrayToHostArray ctx gpuResult
          AM.toStorableVector gpuHost `shouldBe` AM.toStorableVector cpuResult

  describe "BLAS / FFT / RAND nodes" $ do
    it "runs randUniformP -> mapExpr -> gemmMatrixP in one program" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let config = RandGeneratorConfig RocRandRngPseudoDefault 20260308
          program <-
            buildProgram $ do
              a <- randUniformP @A.Ix2 @Float config (A.Sz2 2 2)
              b <- randUniformP @A.Ix2 @Float config (A.Sz2 2 2)
              a' <- mapExpr (Unary (\x -> x .+. constant 1.0)) a
              c0 <- fillArrayP @A.Ix2 @Float 0 (A.Sz2 2 2)
              gemmMatrixP
                MatrixGemmValue
                  { matrixGemmValueTransA = NoTranspose
                  , matrixGemmValueTransB = NoTranspose
                  , matrixGemmValueAlpha = 1
                  , matrixGemmValueA = a'
                  , matrixGemmValueB = b
                  , matrixGemmValueBeta = 0
                  , matrixGemmValueC = c0
                  }
          resultArray <- runProgram runtime program
          hostArray <- readDeviceArrayToHostArray ctx resultArray
          let values = AM.toStorableVector hostArray
          VS.length values `shouldBe` 4
          VS.all (\x -> x > 0 && x <= 8) values `shouldSatisfy` id

    it "runs fftForwardP -> fftInverseP in one program" $
      withGpuContext $ \ctx ->
        withProgramRuntime ctx $ \runtime -> do
          let input = complexVector [1 :+ 0, 2 :+ 1, 3 :+ 0, 4 :+ 2]
          deviceInput <- copyHostArrayToDevice ctx input
          program <-
            buildProgram $ do
              value0 <- inputDeviceArray deviceInput
              spectrum <- fftForwardP value0
              fftInverseP spectrum
          resultArray <- runProgram runtime program
          hostArray <- readDeviceArrayToHostArray ctx resultArray
          hostArray `shouldSatisfy` approxComplexArray (scaleComplexArray 4 input)

complexVector :: [Complex Float] -> A.Array A.S A.Ix1 (Complex Float)
complexVector values =
  AMV.fromVector' A.Seq (A.Sz1 (length values)) (VS.fromList values)

scaleComplexArray :: Float -> A.Array A.S A.Ix1 (Complex Float) -> A.Array A.S A.Ix1 (Complex Float)
scaleComplexArray factor array =
  AMV.fromVector' A.Seq (A.size array) (VS.map (\value -> value * (factor :+ 0)) (AM.toStorableVector array))

approxComplexArray :: A.Array A.S A.Ix1 (Complex Float) -> A.Array A.S A.Ix1 (Complex Float) -> Bool
approxComplexArray expected actual =
  let expectedValues = AM.toStorableVector expected
      actualValues = AM.toStorableVector actual
   in VS.length expectedValues == VS.length actualValues
        && VS.and (VS.zipWith approxComplex expectedValues actualValues)

approxComplex :: Complex Float -> Complex Float -> Bool
approxComplex (expectedReal :+ expectedImag) (actualReal :+ actualImag) =
  abs (expectedReal - actualReal) <= 1.0e-2
    && abs (expectedImag - actualImag) <= 1.0e-2

approxFloatArray :: A.Array A.S A.Ix2 Float -> A.Array A.S A.Ix2 Float -> Bool
approxFloatArray expected actual =
  let expectedValues = AM.toStorableVector expected
      actualValues = AM.toStorableVector actual
   in VS.length expectedValues == VS.length actualValues
        && VS.and (VS.zipWith (\x y -> abs (x - y) <= 1.0e-5) expectedValues actualValues)
