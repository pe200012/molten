{-# LANGUAGE PatternSynonyms #-}

module Molten.RAND.GpuSpec (spec) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Vector.Storable as VS
import Molten.Array.Device (withDeviceArray, withDeviceMatrix)
import Molten.Array.Transfer (readDeviceArrayToHostArray)
import Molten.RAND
  ( randNormal
  , randUniform
  )
import Molten.RAND.Runtime
  ( RandGeneratorConfig(..)
  , withRandRuntime
  )
import Molten.TestSupport (withGpuContext)
import ROCm.RocRAND
  ( pattern RocRandRngPseudoDefault
  )
import Test.Hspec (Spec, describe, it, shouldNotBe, shouldSatisfy)

spec :: Spec
spec = do
  describe "randUniform" $ do
    it "fills an Ix1 Float device array with values in (0, 1]" $
      withGpuContext $ \ctx ->
        withRandRuntime ctx $ \runtime ->
          withDeviceArray ctx (A.Sz1 2048) $ \deviceArray -> do
            randUniform runtime defaultConfig deviceArray
            hostArray <- readDeviceArrayToHostArray ctx deviceArray
            let values = AM.toStorableVector hostArray
            VS.length values `shouldNotBe` 0
            VS.all (\x -> x > 0 && x <= (1 :: Float)) values `shouldSatisfy` id

  describe "randNormal" $ do
    it "fills an Ix1 Float device array with roughly standard-normal data" $
      withGpuContext $ \ctx ->
        withRandRuntime ctx $ \runtime ->
          withDeviceArray ctx (A.Sz1 4096) $ \deviceArray -> do
            randNormal runtime defaultConfig 0 1 deviceArray
            hostArray <- readDeviceArrayToHostArray ctx deviceArray
            let values = VS.toList (AM.toStorableVector hostArray)
                meanValue = mean values
                varianceValue = variance values
            meanValue `shouldSatisfy` (\x -> abs x <= 0.15)
            varianceValue `shouldSatisfy` (\x -> abs (x - 1) <= 0.25)

    it "fills an Ix2 Double device array in place" $
      withGpuContext $ \ctx ->
        withRandRuntime ctx $ \runtime ->
          withDeviceMatrix ctx 32 16 $ \deviceArray -> do
            randUniform runtime defaultConfig deviceArray
            hostArray <- readDeviceArrayToHostArray ctx deviceArray
            let values = AM.toStorableVector hostArray
            VS.length values `shouldSatisfy` (> 0)
            VS.all (\x -> x > 0 && x <= (1 :: Double)) values `shouldSatisfy` id

defaultConfig :: RandGeneratorConfig
defaultConfig =
  RandGeneratorConfig
    { randGeneratorType = RocRandRngPseudoDefault
    , randGeneratorSeed = 20260308
    }

mean :: [Float] -> Float
mean xs = sum xs / fromIntegral (length xs)

variance :: [Float] -> Float
variance xs =
  let avg = mean xs
      squaredDiffs = fmap (\x -> (x - avg) * (x - avg)) xs
   in sum squaredDiffs / fromIntegral (length xs)
