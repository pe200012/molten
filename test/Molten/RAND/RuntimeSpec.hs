{-# LANGUAGE PatternSynonyms #-}

module Molten.RAND.RuntimeSpec (spec) where

import Data.Word (Word64)
import Molten.RAND.Runtime
  ( CachedGenerator(cachedGeneratorId)
  , RandGeneratorConfig(..)
  , lookupOrCreateGenerator
  , withRandRuntime
  )
import Molten.TestSupport (withGpuContext)
import ROCm.RocRAND
  ( RocRandRngType
  , pattern RocRandRngPseudoDefault
  , pattern RocRandRngPseudoPhilox4x32_10
  )
import Test.Hspec (Spec, describe, it, shouldBe, shouldNotBe)

spec :: Spec
spec = do
  describe "lookupOrCreateGenerator" $ do
    it "reuses the cached generator for the same config" $
      withGpuContext $ \ctx ->
        withRandRuntime ctx $ \runtime -> do
          let config = mkConfig RocRandRngPseudoDefault 20260308
          gen1 <- lookupOrCreateGenerator runtime config
          gen2 <- lookupOrCreateGenerator runtime config
          cachedGeneratorId gen1 `shouldBe` cachedGeneratorId gen2

    it "uses a different cache key when the seed changes" $
      withGpuContext $ \ctx ->
        withRandRuntime ctx $ \runtime -> do
          gen1 <- lookupOrCreateGenerator runtime (mkConfig RocRandRngPseudoDefault 1)
          gen2 <- lookupOrCreateGenerator runtime (mkConfig RocRandRngPseudoDefault 2)
          cachedGeneratorId gen1 `shouldNotBe` cachedGeneratorId gen2

    it "uses a different cache key when the RNG type changes" $
      withGpuContext $ \ctx ->
        withRandRuntime ctx $ \runtime -> do
          gen1 <- lookupOrCreateGenerator runtime (mkConfig RocRandRngPseudoDefault 1)
          gen2 <- lookupOrCreateGenerator runtime (mkConfig RocRandRngPseudoPhilox4x32_10 1)
          cachedGeneratorId gen1 `shouldNotBe` cachedGeneratorId gen2

mkConfig :: RocRandRngType -> Word64 -> RandGeneratorConfig
mkConfig rngType seed =
  RandGeneratorConfig
    { randGeneratorType = rngType
    , randGeneratorSeed = seed
    }
