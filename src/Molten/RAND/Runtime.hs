module Molten.RAND.Runtime
  ( CachedGenerator
      ( cachedGeneratorConfig
      , cachedGeneratorHandle
      , cachedGeneratorId
      )
  , RandGeneratorConfig(..)
  , RandRuntime
      ( randRuntimeContext
      )
  , lookupOrCreateGenerator
  , withGeneratorOnStream
  , withRandRuntime
  ) where

import Control.Concurrent.MVar (MVar, modifyMVar, newMVar)
import Control.Exception (bracket)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Word (Word64)
import GHC.Stack (HasCallStack)
import Molten.Core.Context (Context, withContextDevice)
import Molten.Core.Stream (Stream, withRawHipStream)
import ROCm.FFI.Core.Types (RocRandGenerator)
import ROCm.RocRAND
  ( RocRandRngType
  , rocrandCreateGenerator
  , rocrandDestroyGenerator
  , rocrandSetSeed
  , rocrandSetStream
  )

data RandGeneratorConfig = RandGeneratorConfig
  { randGeneratorType :: !RocRandRngType
  , randGeneratorSeed :: !Word64
  }
  deriving (Eq, Ord, Show)

data CachedGenerator = CachedGenerator
  { cachedGeneratorId :: !Int
  , cachedGeneratorConfig :: !RandGeneratorConfig
  , cachedGeneratorHandle :: !RocRandGenerator
  }

data RandRuntime = RandRuntime
  { randRuntimeContext :: !Context
  , randRuntimeGeneratorCache :: !(MVar (Map RandGeneratorConfig CachedGenerator))
  , randRuntimeNextGeneratorId :: !(MVar Int)
  }

withRandRuntime :: HasCallStack => Context -> (RandRuntime -> IO a) -> IO a
withRandRuntime ctx = bracket (createRandRuntime ctx) destroyRandRuntime

lookupOrCreateGenerator :: HasCallStack => RandRuntime -> RandGeneratorConfig -> IO CachedGenerator
lookupOrCreateGenerator runtime config =
  modifyMVar (randRuntimeGeneratorCache runtime) $ \cache ->
    case Map.lookup config cache of
      Just cachedGenerator -> pure (cache, cachedGenerator)
      Nothing -> do
        generatorId <-
          modifyMVar (randRuntimeNextGeneratorId runtime) $ \nextGeneratorId ->
            pure (nextGeneratorId + 1, nextGeneratorId)
        cachedGenerator <- createCachedGenerator generatorId config
        pure (Map.insert config cachedGenerator cache, cachedGenerator)

withGeneratorOnStream :: HasCallStack => CachedGenerator -> Stream -> (RocRandGenerator -> IO a) -> IO a
withGeneratorOnStream cachedGenerator stream action =
  withRawHipStream stream $ \rawStream -> do
    rocrandSetStream (cachedGeneratorHandle cachedGenerator) rawStream
    action (cachedGeneratorHandle cachedGenerator)

createRandRuntime :: Context -> IO RandRuntime
createRandRuntime ctx = do
  generatorCache <- newMVar Map.empty
  nextGeneratorId <- newMVar 0
  pure
    RandRuntime
      { randRuntimeContext = ctx
      , randRuntimeGeneratorCache = generatorCache
      , randRuntimeNextGeneratorId = nextGeneratorId
      }

destroyRandRuntime :: HasCallStack => RandRuntime -> IO ()
destroyRandRuntime runtime =
  withContextDevice (randRuntimeContext runtime) $ do
    cachedGenerators <-
      modifyMVar (randRuntimeGeneratorCache runtime) $ \cache ->
        pure (Map.empty, Map.elems cache)
    mapM_ (rocrandDestroyGenerator . cachedGeneratorHandle) cachedGenerators

createCachedGenerator :: HasCallStack => Int -> RandGeneratorConfig -> IO CachedGenerator
createCachedGenerator generatorId config = do
  generatorHandle <- rocrandCreateGenerator (randGeneratorType config)
  rocrandSetSeed generatorHandle (randGeneratorSeed config)
  pure
    CachedGenerator
      { cachedGeneratorId = generatorId
      , cachedGeneratorConfig = config
      , cachedGeneratorHandle = generatorHandle
      }
