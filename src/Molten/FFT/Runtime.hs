{-# LANGUAGE DataKinds #-}
{-# LANGUAGE PatternSynonyms #-}

module Molten.FFT.Runtime
  ( CachedFftPlan
      ( cachedFftPlanHandle
      , cachedFftPlanId
      , cachedFftPlanKey
      , cachedFftPlanWorkBytes
      , cachedFftPlanWorkspace
      )
  , FftPlanKey
      ( fftPlanKeyBatch
      , fftPlanKeyLengths
      , fftPlanKeyPrecision
      , fftPlanKeyTransform
      , fftPlanKeyWorkspaceMode
      )
  , FftRuntime
      ( fftRuntimeContext
      )
  , FftTransform(..)
  , FftWorkspaceMode(..)
  , lookupOrCreateFftPlan
  , mkFftPlanKey
  , runCachedFftPlan
  , withFftRuntime
  ) where

import Control.Concurrent.MVar (MVar, modifyMVar, newMVar)
import Control.Exception (bracket)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Word (Word8)
import Foreign.C.Types (CSize)
import Foreign.Ptr (Ptr)
import GHC.Stack (HasCallStack)
import Molten.Core.Buffer (Buffer, Location(..), destroyBuffer, newDeviceBufferOn, withDevicePtr)
import Molten.Core.Context (Context, contextDefaultStream, contextDeviceId, withContextDevice)
import Molten.Core.Stream (withRawHipStream)
import Molten.Internal.Validation (ensurePositive)
import ROCm.FFI.Core.Exception (throwArgumentError)
import ROCm.FFI.Core.Types (RocfftPlan, RocfftPlanDescription)
import ROCm.RocFFT
  ( rocfftCleanup
  , rocfftExecute
  , rocfftExecutionInfoSetStream
  , rocfftExecutionInfoSetWorkBuffer
  , rocfftPlanCreate
  , rocfftPlanDescriptionSetDataLayout
  , rocfftPlanDestroy
  , rocfftPlanGetWorkBufferSize
  , rocfftSetup
  , withRocfftExecutionInfo
  , withRocfftPlanDescription
  , pattern RocfftArrayTypeHermitianInterleaved
  , pattern RocfftArrayTypeReal
  , RocfftPrecision
  , RocfftTransformType
  , pattern RocfftPlacementNotInplace
  , pattern RocfftTransformTypeComplexForward
  , pattern RocfftTransformTypeComplexInverse
  , pattern RocfftTransformTypeRealForward
  , pattern RocfftTransformTypeRealInverse
  )

data FftTransform
  = FftTransformComplexForward
  | FftTransformComplexInverse
  | FftTransformRealForward
  | FftTransformRealInverse
  deriving (Eq, Ord, Show)

data FftWorkspaceMode
  = FftWorkspaceAuto
  | FftWorkspaceExplicit
  deriving (Eq, Ord, Show)

data FftPlanKey = FftPlanKey
  { fftPlanKeyTransform :: !FftTransform
  , fftPlanKeyPrecision :: !RocfftPrecision
  , fftPlanKeyLengths :: ![Int]
  , fftPlanKeyBatch :: !Int
  , fftPlanKeyWorkspaceMode :: !FftWorkspaceMode
  }
  deriving (Eq, Ord, Show)

data CachedFftPlan = CachedFftPlan
  { cachedFftPlanId :: !Int
  , cachedFftPlanKey :: !FftPlanKey
  , cachedFftPlanHandle :: !RocfftPlan
  , cachedFftPlanWorkBytes :: !CSize
  , cachedFftPlanWorkspace :: !(Maybe (Buffer 'Device Word8))
  }

data FftRuntime = FftRuntime
  { fftRuntimeContext :: !Context
  , fftRuntimePlanCache :: !(MVar (Map FftPlanKey CachedFftPlan))
  , fftRuntimeNextPlanId :: !(MVar Int)
  }

withFftRuntime :: HasCallStack => Context -> (FftRuntime -> IO a) -> IO a
withFftRuntime ctx = bracket (createFftRuntime ctx) destroyFftRuntime

mkFftPlanKey :: HasCallStack => FftTransform -> RocfftPrecision -> [Int] -> Int -> FftWorkspaceMode -> IO FftPlanKey
mkFftPlanKey transform precision lengths batch workspaceMode = do
  if null lengths || length lengths > 3
    then throwArgumentError "mkFftPlanKey" "lengths must contain between 1 and 3 dimensions"
    else pure ()
  mapM_ (ensurePositive "mkFftPlanKey" "length") lengths
  ensurePositive "mkFftPlanKey" "batch" batch
  case transform of
    FftTransformRealForward ->
      if length lengths == 1
        then pure ()
        else throwArgumentError "mkFftPlanKey" "real transforms currently require a single length"
    FftTransformRealInverse ->
      if length lengths == 1
        then pure ()
        else throwArgumentError "mkFftPlanKey" "real transforms currently require a single length"
    FftTransformComplexForward -> pure ()
    FftTransformComplexInverse -> pure ()
  pure
    FftPlanKey
      { fftPlanKeyTransform = transform
      , fftPlanKeyPrecision = precision
      , fftPlanKeyLengths = lengths
      , fftPlanKeyBatch = batch
      , fftPlanKeyWorkspaceMode = workspaceMode
      }

lookupOrCreateFftPlan :: HasCallStack => FftRuntime -> FftPlanKey -> IO CachedFftPlan
lookupOrCreateFftPlan runtime planKey =
  modifyMVar (fftRuntimePlanCache runtime) $ \cache ->
    case Map.lookup planKey cache of
      Just cachedPlan -> pure (cache, cachedPlan)
      Nothing -> do
        planId <- modifyMVar (fftRuntimeNextPlanId runtime) $ \nextPlanId -> pure (nextPlanId + 1, nextPlanId)
        cachedPlan <- withContextDevice (fftRuntimeContext runtime) (createCachedFftPlan runtime planId planKey)
        pure (Map.insert planKey cachedPlan cache, cachedPlan)

runCachedFftPlan :: HasCallStack => FftRuntime -> CachedFftPlan -> [Ptr ()] -> [Ptr ()] -> IO ()
runCachedFftPlan runtime cachedPlan inPtrs outPtrs =
  withContextDevice (fftRuntimeContext runtime) $
    withRocfftExecutionInfo $ \execInfo ->
      withRawHipStream (contextDefaultStream (fftRuntimeContext runtime)) $ \rawStream -> do
        rocfftExecutionInfoSetStream execInfo rawStream
        withPlanWorkspace runtime cachedPlan $ \maybeWorkspace -> do
          case maybeWorkspace of
            Nothing -> pure ()
            Just workspaceBuffer ->
              withDevicePtr workspaceBuffer $ \workspacePtr ->
                rocfftExecutionInfoSetWorkBuffer execInfo workspacePtr (cachedFftPlanWorkBytes cachedPlan)
          rocfftExecute (cachedFftPlanHandle cachedPlan) inPtrs outPtrs (Just execInfo)

createFftRuntime :: HasCallStack => Context -> IO FftRuntime
createFftRuntime ctx =
  withContextDevice ctx $ do
    rocfftSetup
    planCache <- newMVar Map.empty
    nextPlanId <- newMVar 0
    pure
      FftRuntime
        { fftRuntimeContext = ctx
        , fftRuntimePlanCache = planCache
        , fftRuntimeNextPlanId = nextPlanId
        }

destroyFftRuntime :: HasCallStack => FftRuntime -> IO ()
destroyFftRuntime runtime =
  withContextDevice (fftRuntimeContext runtime) $ do
    cachedPlans <- modifyMVar (fftRuntimePlanCache runtime) $ \cache -> pure (Map.empty, Map.elems cache)
    mapM_ destroyCachedFftPlan cachedPlans
    rocfftCleanup

createCachedFftPlan :: HasCallStack => FftRuntime -> Int -> FftPlanKey -> IO CachedFftPlan
createCachedFftPlan runtime planId planKey =
  withPlanDescription planKey $ \maybeDescription -> do
    planHandle <-
      rocfftPlanCreate
        RocfftPlacementNotInplace
        (toRocfftTransformType (fftPlanKeyTransform planKey))
        (fftPlanKeyPrecision planKey)
        (fmap fromIntegral (fftPlanKeyLengths planKey))
        (fromIntegral (fftPlanKeyBatch planKey))
        maybeDescription
    workBytes <- rocfftPlanGetWorkBufferSize planHandle
    workspace <-
      case fftPlanKeyWorkspaceMode planKey of
        FftWorkspaceAuto -> pure Nothing
        FftWorkspaceExplicit -> allocatePlanWorkspace runtime workBytes
    pure
      CachedFftPlan
        { cachedFftPlanId = planId
        , cachedFftPlanKey = planKey
        , cachedFftPlanHandle = planHandle
        , cachedFftPlanWorkBytes = workBytes
        , cachedFftPlanWorkspace = workspace
        }

destroyCachedFftPlan :: HasCallStack => CachedFftPlan -> IO ()
destroyCachedFftPlan cachedPlan = do
  mapM_ destroyBuffer (cachedFftPlanWorkspace cachedPlan)
  rocfftPlanDestroy (cachedFftPlanHandle cachedPlan)

allocatePlanWorkspace :: HasCallStack => FftRuntime -> CSize -> IO (Maybe (Buffer 'Device Word8))
allocatePlanWorkspace runtime workBytes
  | workBytes == 0 = pure Nothing
  | otherwise = do
      workLength <- cSizeToInt "allocatePlanWorkspace" workBytes
      Just <$> newDeviceBufferOn (contextDeviceId (fftRuntimeContext runtime)) workLength

withPlanWorkspace :: HasCallStack => FftRuntime -> CachedFftPlan -> (Maybe (Buffer 'Device Word8) -> IO a) -> IO a
withPlanWorkspace runtime cachedPlan action
  | cachedFftPlanWorkBytes cachedPlan == 0 = action Nothing
  | otherwise =
      case cachedFftPlanWorkspace cachedPlan of
        Just workspaceBuffer -> action (Just workspaceBuffer)
        Nothing ->
          bracket
            (cSizeToInt "runCachedFftPlan" (cachedFftPlanWorkBytes cachedPlan) >>= newDeviceBufferOn (contextDeviceId (fftRuntimeContext runtime)))
            destroyBuffer
            (action . Just)

withPlanDescription :: HasCallStack => FftPlanKey -> (Maybe RocfftPlanDescription -> IO a) -> IO a
withPlanDescription planKey action =
  case fftPlanKeyTransform planKey of
    FftTransformComplexForward -> action Nothing
    FftTransformComplexInverse -> action Nothing
    FftTransformRealForward -> do
      realLength <- singleRealLength planKey
      withRocfftPlanDescription $ \description -> do
        let frequencyLength = realLength `div` 2 + 1
        rocfftPlanDescriptionSetDataLayout
          description
          RocfftArrayTypeReal
          RocfftArrayTypeHermitianInterleaved
          Nothing
          Nothing
          [1]
          (fromIntegral realLength)
          [1]
          (fromIntegral frequencyLength)
        action (Just description)
    FftTransformRealInverse -> do
      realLength <- singleRealLength planKey
      withRocfftPlanDescription $ \description -> do
        let frequencyLength = realLength `div` 2 + 1
        rocfftPlanDescriptionSetDataLayout
          description
          RocfftArrayTypeHermitianInterleaved
          RocfftArrayTypeReal
          Nothing
          Nothing
          [1]
          (fromIntegral frequencyLength)
          [1]
          (fromIntegral realLength)
        action (Just description)

singleRealLength :: HasCallStack => FftPlanKey -> IO Int
singleRealLength planKey =
  case fftPlanKeyLengths planKey of
    [length1] -> pure length1
    _ -> throwArgumentError "singleRealLength" "real FFT plans require exactly one length"

toRocfftTransformType :: FftTransform -> RocfftTransformType
toRocfftTransformType transform =
  case transform of
    FftTransformComplexForward -> RocfftTransformTypeComplexForward
    FftTransformComplexInverse -> RocfftTransformTypeComplexInverse
    FftTransformRealForward -> RocfftTransformTypeRealForward
    FftTransformRealInverse -> RocfftTransformTypeRealInverse

cSizeToInt :: HasCallStack => String -> CSize -> IO Int
cSizeToInt functionName value =
  let maxInt = fromIntegral (maxBound :: Int)
   in if value > maxInt
        then throwArgumentError functionName "workspace size overflow"
        else pure (fromIntegral value)
