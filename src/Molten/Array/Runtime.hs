{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Molten.Array.Runtime
  ( ArrayRuntime
  , arrayRuntimeKernelCount
  , broadcastColsArray
  , broadcastColsArrayOn
  , broadcastRowsArray
  , broadcastRowsArrayOn
  , fillArray
  , fillArrayOn
  , mapArray
  , mapArrayOn
  , maxColsArray
  , maxColsArrayOn
  , maxRowsArray
  , maxRowsArrayOn
  , reduceAllArray
  , reduceAllArrayOn
  , sumColsArray
  , sumColsArrayOn
  , sumRowsArray
  , sumRowsArrayOn
  , withArrayRuntime
  , zipWithArray
  , zipWithArrayOn
  ) where

import Control.Concurrent.MVar (MVar, modifyMVar, newMVar, readMVar)
import Control.Exception (bracket)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Proxy (Proxy(..))
import Data.Word (Word32, Word64)
import qualified Data.Massiv.Array as A
import Foreign.Marshal.Array (withArray)
import Foreign.Marshal.Utils (with)
import Foreign.Ptr (Ptr, castPtr, nullPtr)
import Foreign.Storable (sizeOf)
import GHC.Stack (HasCallStack)
import Molten.Array.Device (DeviceArray, deviceArrayBuffer, deviceArraySize, mkDeviceArray, mkDeviceMatrix, mkDeviceVector)
import Molten.Array.Expr
  ( ArrayScalar(arrayScalarCType)
  , Binary(..)
  , Comparable
  , NumericExp
  , Unary(..)
  , renderBinaryExpression
  , renderUnaryExpression
  )
import Molten.Core.Buffer
  ( Buffer
  , bufferLength
  , destroyBuffer
  , newDeviceBufferOn
  , withDevicePtr
  )
import Molten.Core.Context (Context, contextDefaultStream, contextDeviceId)
import Molten.Core.Stream (Stream, streamDeviceId, synchronizeStream, withRawHipStream)
import Molten.Internal.HIPRTC (LoadedHipKernel(..), compileHipKernel, loadHipKernel, unloadHipKernel)
import ROCm.FFI.Core.Exception (throwArgumentError)
import ROCm.FFI.Core.Types (DevicePtr(..))
import ROCm.HIP (HipDim3(..), hipModuleLaunchKernel)

data CachedKernel = CachedKernel
  { cachedKernelId :: !Int
  , cachedKernelHandle :: !LoadedHipKernel
  }

data ArrayRuntime = ArrayRuntime
  { arrayRuntimeContext :: !Context
  , arrayRuntimeKernelCache :: !(MVar (Map String CachedKernel))
  , arrayRuntimeNextKernelId :: !(MVar Int)
  }

withArrayRuntime :: HasCallStack => Context -> (ArrayRuntime -> IO a) -> IO a
withArrayRuntime ctx = bracket (createArrayRuntime ctx) destroyArrayRuntime

arrayRuntimeKernelCount :: ArrayRuntime -> IO Int
arrayRuntimeKernelCount runtime = Map.size <$> readMVar (arrayRuntimeKernelCache runtime)

fillArray :: (HasCallStack, ArrayScalar a, A.Index ix) => ArrayRuntime -> a -> A.Sz ix -> IO (DeviceArray ix a)
fillArray runtime value size =
  fillArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime value size

fillArrayOn :: forall a ix. (HasCallStack, ArrayScalar a, A.Index ix) => Stream -> ArrayRuntime -> a -> A.Sz ix -> IO (DeviceArray ix a)
fillArrayOn stream runtime value size = do
  outputBuffer <- newDeviceBufferOn (streamDeviceId stream) (A.totalElem size)
  cachedKernel <- lookupOrCompileKernel runtime (fillKernelKey value) fillKernelName (fillKernelSource (Proxy @a))
  withDevicePtr outputBuffer $ \outputPtr ->
    launchFillKernel cachedKernel stream outputPtr (A.totalElem size) value
  mkDeviceArray size outputBuffer

mapArray :: (HasCallStack, ArrayScalar a, ArrayScalar b, A.Index ix) => ArrayRuntime -> Unary a b -> DeviceArray ix a -> IO (DeviceArray ix b)
mapArray runtime unary input =
  mapArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime unary input

mapArrayOn :: forall a b ix. (HasCallStack, ArrayScalar a, ArrayScalar b, A.Index ix) => Stream -> ArrayRuntime -> Unary a b -> DeviceArray ix a -> IO (DeviceArray ix b)
mapArrayOn stream runtime unary input = do
  outputBuffer <- newDeviceBufferOn (streamDeviceId stream) (A.totalElem (deviceArraySize input))
  cachedKernel <- lookupOrCompileKernel runtime (mapKernelKey unary) mapKernelName (mapKernelSource (Proxy @a) (Proxy @b) unary)
  withDevicePtr (deviceArrayBuffer input) $ \inputPtr ->
    withDevicePtr outputBuffer $ \outputPtr ->
      launchMapKernel cachedKernel stream inputPtr outputPtr (A.totalElem (deviceArraySize input))
  mkDeviceArray (deviceArraySize input) outputBuffer

zipWithArray ::
  (HasCallStack, ArrayScalar a, ArrayScalar b, ArrayScalar c, A.Index ix) =>
  ArrayRuntime ->
  Binary a b c ->
  DeviceArray ix a ->
  DeviceArray ix b ->
  IO (DeviceArray ix c)
zipWithArray runtime binary left right =
  zipWithArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime binary left right

zipWithArrayOn :: forall a b c ix.
  (HasCallStack, ArrayScalar a, ArrayScalar b, ArrayScalar c, A.Index ix) =>
  Stream ->
  ArrayRuntime ->
  Binary a b c ->
  DeviceArray ix a ->
  DeviceArray ix b ->
  IO (DeviceArray ix c)
zipWithArrayOn stream runtime binary left right = do
  if deviceArraySize left /= deviceArraySize right
    then throwArgumentError "zipWithArrayOn" "input arrays must have the same shape"
    else pure ()
  let size = deviceArraySize left
  outputBuffer <- newDeviceBufferOn (streamDeviceId stream) (A.totalElem size)
  cachedKernel <- lookupOrCompileKernel runtime (zipKernelKey binary) zipKernelName (zipKernelSource (Proxy @a) (Proxy @b) (Proxy @c) binary)
  withDevicePtr (deviceArrayBuffer left) $ \leftPtr ->
    withDevicePtr (deviceArrayBuffer right) $ \rightPtr ->
      withDevicePtr outputBuffer $ \outputPtr ->
        launchZipKernel cachedKernel stream leftPtr rightPtr outputPtr (A.totalElem size)
  mkDeviceArray size outputBuffer

reduceAllArray :: (HasCallStack, ArrayScalar a, NumericExp a, A.Index ix) => ArrayRuntime -> Binary a a a -> a -> DeviceArray ix a -> IO (DeviceArray A.Ix1 a)
reduceAllArray runtime binary initialValue input =
  reduceAllArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime binary initialValue input

reduceAllArrayOn :: forall a ix. (HasCallStack, ArrayScalar a, NumericExp a, A.Index ix) => Stream -> ArrayRuntime -> Binary a a a -> a -> DeviceArray ix a -> IO (DeviceArray A.Ix1 a)
reduceAllArrayOn stream runtime binary initialValue input
  | totalLength == 0 = fillArrayOn stream runtime initialValue (A.Sz1 1)
  | otherwise = do
      reduceKernel <- lookupOrCompileKernel runtime (reduceKernelKey binary) reduceKernelName (reduceKernelSource (Proxy @a) binary)
      combineKernel <- lookupOrCompileKernel runtime (reduceCombineKernelKey binary) reduceCombineKernelName (reduceCombineKernelSource (Proxy @a) binary)
      (reducedBuffer, finalOwned, temporaryBuffers) <- reduceToSingleBuffer reduceKernel stream input
      outputBuffer <- newDeviceBufferOn (streamDeviceId stream) 1
      withDevicePtr reducedBuffer $ \reducedPtr ->
        withDevicePtr outputBuffer $ \outputPtr ->
          launchReduceCombineKernel combineKernel stream initialValue reducedPtr outputPtr
      synchronizeStream stream
      mapM_ destroyBuffer temporaryBuffers
      if finalOwned
        then destroyBuffer reducedBuffer
        else pure ()
      mkDeviceVector 1 outputBuffer
  where
    totalLength = A.totalElem (deviceArraySize input)

sumRowsArray :: (HasCallStack, ArrayScalar a, NumericExp a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
sumRowsArray runtime input =
  sumRowsArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime input

sumRowsArrayOn :: forall a. (HasCallStack, ArrayScalar a, NumericExp a) => Stream -> ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
sumRowsArrayOn stream runtime input =
  reduceAxis2ArrayOn stream runtime AxisRowsReduce AxisReduceSum input

sumColsArray :: (HasCallStack, ArrayScalar a, NumericExp a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
sumColsArray runtime input =
  sumColsArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime input

sumColsArrayOn :: forall a. (HasCallStack, ArrayScalar a, NumericExp a) => Stream -> ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
sumColsArrayOn stream runtime input =
  reduceAxis2ArrayOn stream runtime AxisColsReduce AxisReduceSum input

maxRowsArray :: (HasCallStack, ArrayScalar a, Comparable a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
maxRowsArray runtime input =
  maxRowsArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime input

maxRowsArrayOn :: forall a. (HasCallStack, ArrayScalar a, Comparable a) => Stream -> ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
maxRowsArrayOn stream runtime input =
  reduceAxis2ArrayOn stream runtime AxisRowsReduce AxisReduceMax input

maxColsArray :: (HasCallStack, ArrayScalar a, Comparable a) => ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
maxColsArray runtime input =
  maxColsArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime input

maxColsArrayOn :: forall a. (HasCallStack, ArrayScalar a, Comparable a) => Stream -> ArrayRuntime -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
maxColsArrayOn stream runtime input =
  reduceAxis2ArrayOn stream runtime AxisColsReduce AxisReduceMax input

broadcastRowsArray :: (HasCallStack, ArrayScalar a) => ArrayRuntime -> Int -> DeviceArray A.Ix1 a -> IO (DeviceArray A.Ix2 a)
broadcastRowsArray runtime cols input =
  broadcastRowsArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime cols input

broadcastRowsArrayOn :: forall a. (HasCallStack, ArrayScalar a) => Stream -> ArrayRuntime -> Int -> DeviceArray A.Ix1 a -> IO (DeviceArray A.Ix2 a)
broadcastRowsArrayOn stream runtime cols input =
  broadcastAxis2ArrayOn stream runtime AxisRowsBroadcast cols input

broadcastColsArray :: (HasCallStack, ArrayScalar a) => ArrayRuntime -> Int -> DeviceArray A.Ix1 a -> IO (DeviceArray A.Ix2 a)
broadcastColsArray runtime rows input =
  broadcastColsArrayOn (contextDefaultStream (arrayRuntimeContext runtime)) runtime rows input

broadcastColsArrayOn :: forall a. (HasCallStack, ArrayScalar a) => Stream -> ArrayRuntime -> Int -> DeviceArray A.Ix1 a -> IO (DeviceArray A.Ix2 a)
broadcastColsArrayOn stream runtime rows input =
  broadcastAxis2ArrayOn stream runtime AxisColsBroadcast rows input

createArrayRuntime :: Context -> IO ArrayRuntime
createArrayRuntime ctx = do
  kernelCache <- newMVar Map.empty
  nextKernelId <- newMVar 0
  pure
    ArrayRuntime
      { arrayRuntimeContext = ctx
      , arrayRuntimeKernelCache = kernelCache
      , arrayRuntimeNextKernelId = nextKernelId
      }

destroyArrayRuntime :: HasCallStack => ArrayRuntime -> IO ()
destroyArrayRuntime runtime = do
  cachedKernels <- Map.elems <$> readMVar (arrayRuntimeKernelCache runtime)
  mapM_ (unloadHipKernel . cachedKernelHandle) cachedKernels

reduceAxis2ArrayOn :: forall a. (HasCallStack, ArrayScalar a) => Stream -> ArrayRuntime -> AxisReduceDirection -> AxisReduceKind -> DeviceArray A.Ix2 a -> IO (DeviceArray A.Ix1 a)
reduceAxis2ArrayOn stream runtime direction reduceKind input = do
  let A.Sz2 rows cols = deviceArraySize input
      outputLength =
        case direction of
          AxisRowsReduce -> rows
          AxisColsReduce -> cols
  validateAxisReduction direction reduceKind rows cols
  outputBuffer <- newDeviceBufferOn (streamDeviceId stream) outputLength
  if outputLength == 0
    then mkDeviceVector 0 outputBuffer
    else do
      cachedKernel <-
        lookupOrCompileKernel
          runtime
          (axisReduceKernelKey (Proxy @a) direction reduceKind)
          (axisReduceKernelName direction reduceKind)
          (axisReduceKernelSource (Proxy @a) direction reduceKind)
      withDevicePtr (deviceArrayBuffer input) $ \inputPtr ->
        withDevicePtr outputBuffer $ \outputPtr ->
          launchAxisReduceKernel cachedKernel stream inputPtr outputPtr outputLength rows cols
      mkDeviceVector outputLength outputBuffer

broadcastAxis2ArrayOn :: forall a. (HasCallStack, ArrayScalar a) => Stream -> ArrayRuntime -> AxisBroadcastDirection -> Int -> DeviceArray A.Ix1 a -> IO (DeviceArray A.Ix2 a)
broadcastAxis2ArrayOn stream runtime direction repeatedExtent input = do
  if repeatedExtent < 0
    then throwArgumentError "broadcastAxis2ArrayOn" "broadcast extent must be non-negative"
    else pure ()
  let A.Sz1 baseLength = deviceArraySize input
      (rows, cols) =
        case direction of
          AxisRowsBroadcast -> (baseLength, repeatedExtent)
          AxisColsBroadcast -> (repeatedExtent, baseLength)
      outputLength = rows * cols
  outputBuffer <- newDeviceBufferOn (streamDeviceId stream) outputLength
  if outputLength == 0
    then mkDeviceMatrix rows cols outputBuffer
    else do
      cachedKernel <-
        lookupOrCompileKernel
          runtime
          (axisBroadcastKernelKey (Proxy @a) direction)
          (axisBroadcastKernelName direction)
          (axisBroadcastKernelSource (Proxy @a) direction)
      withDevicePtr (deviceArrayBuffer input) $ \inputPtr ->
        withDevicePtr outputBuffer $ \outputPtr ->
          launchAxisBroadcastKernel cachedKernel stream inputPtr outputPtr outputLength rows cols
      mkDeviceMatrix rows cols outputBuffer

validateAxisReduction :: HasCallStack => AxisReduceDirection -> AxisReduceKind -> Int -> Int -> IO ()
validateAxisReduction direction reduceKind rows cols =
  case (direction, reduceKind, rows, cols) of
    (_, _, negativeRows, _) | negativeRows < 0 ->
      throwArgumentError "reduceAxis2ArrayOn" "row count must be non-negative"
    (_, _, _, negativeCols) | negativeCols < 0 ->
      throwArgumentError "reduceAxis2ArrayOn" "column count must be non-negative"
    (AxisRowsReduce, _, positiveRows, 0) | positiveRows > 0 ->
      throwArgumentError "reduceAxis2ArrayOn" "row reduction requires at least one column"
    (AxisColsReduce, _, 0, positiveCols) | positiveCols > 0 ->
      throwArgumentError "reduceAxis2ArrayOn" "column reduction requires at least one row"
    _ -> pure ()

lookupOrCompileKernel :: HasCallStack => ArrayRuntime -> String -> String -> String -> IO CachedKernel
lookupOrCompileKernel runtime kernelKey kernelName kernelSource =
  modifyMVar (arrayRuntimeKernelCache runtime) $ \cache ->
    case Map.lookup kernelKey cache of
      Just cachedKernel -> pure (cache, cachedKernel)
      Nothing -> do
        kernelId <- modifyMVar (arrayRuntimeNextKernelId runtime) $ \nextKernelId -> pure (nextKernelId + 1, nextKernelId)
        codeObject <- compileHipKernel (contextDeviceId (arrayRuntimeContext runtime)) kernelSource kernelName []
        loadedKernel <- loadHipKernel codeObject kernelName
        let cachedKernel = CachedKernel {cachedKernelId = kernelId, cachedKernelHandle = loadedKernel}
        pure (Map.insert kernelKey cachedKernel cache, cachedKernel)

reduceToSingleBuffer :: forall a ix. (HasCallStack, ArrayScalar a, A.Index ix) => CachedKernel -> Stream -> DeviceArray ix a -> IO (Buffer 'Device a, Bool, [Buffer 'Device a])
reduceToSingleBuffer cachedKernel stream input = go (deviceArrayBuffer input) False (bufferLength (deviceArrayBuffer input)) []
  where
    blockSize = runtimeBlockSize

    go :: Buffer 'Device a -> Bool -> Int -> [Buffer 'Device a] -> IO (Buffer 'Device a, Bool, [Buffer 'Device a])
    go currentBuffer currentOwned currentLength temporaryBuffers
      | currentLength <= 1 = pure (currentBuffer, currentOwned, temporaryBuffers)
      | otherwise = do
          let nextLength = ceilDiv currentLength (2 * blockSize)
          nextBuffer <- newDeviceBufferOn (streamDeviceId stream) nextLength
          withDevicePtr currentBuffer $ \currentPtr ->
            withDevicePtr nextBuffer $ \nextPtr ->
              launchReduceKernel cachedKernel stream currentPtr nextPtr currentLength
          let temporaryBuffers' =
                if currentOwned
                  then currentBuffer : temporaryBuffers
                  else temporaryBuffers
          go nextBuffer True nextLength temporaryBuffers'

launchFillKernel :: (HasCallStack, ArrayScalar a) => CachedKernel -> Stream -> DevicePtr a -> Int -> a -> IO ()
launchFillKernel cachedKernel stream (DevicePtr outputPtr) elementCount value =
  with outputPtr $ \pOutput ->
    with (fromIntegral elementCount :: Word64) $ \pLength ->
      with value $ \pValue ->
        withArray [castPtr pOutput, castPtr pLength, castPtr pValue] $ \kernelParams ->
          launch1dKernel (cachedKernelHandle cachedKernel) stream elementCount 0 kernelParams

launchMapKernel :: (HasCallStack, ArrayScalar a, ArrayScalar b) => CachedKernel -> Stream -> DevicePtr a -> DevicePtr b -> Int -> IO ()
launchMapKernel cachedKernel stream (DevicePtr inputPtr) (DevicePtr outputPtr) elementCount =
  with inputPtr $ \pInput ->
    with outputPtr $ \pOutput ->
      with (fromIntegral elementCount :: Word64) $ \pLength ->
        withArray [castPtr pInput, castPtr pOutput, castPtr pLength] $ \kernelParams ->
          launch1dKernel (cachedKernelHandle cachedKernel) stream elementCount 0 kernelParams

launchZipKernel :: (HasCallStack, ArrayScalar a, ArrayScalar b, ArrayScalar c) => CachedKernel -> Stream -> DevicePtr a -> DevicePtr b -> DevicePtr c -> Int -> IO ()
launchZipKernel cachedKernel stream (DevicePtr leftPtr) (DevicePtr rightPtr) (DevicePtr outputPtr) elementCount =
  with leftPtr $ \pLeft ->
    with rightPtr $ \pRight ->
      with outputPtr $ \pOutput ->
        with (fromIntegral elementCount :: Word64) $ \pLength ->
          withArray [castPtr pLeft, castPtr pRight, castPtr pOutput, castPtr pLength] $ \kernelParams ->
            launch1dKernel (cachedKernelHandle cachedKernel) stream elementCount 0 kernelParams

launchAxisReduceKernel :: HasCallStack => CachedKernel -> Stream -> DevicePtr a -> DevicePtr a -> Int -> Int -> Int -> IO ()
launchAxisReduceKernel cachedKernel stream (DevicePtr inputPtr) (DevicePtr outputPtr) outputLength rows cols =
  with inputPtr $ \pInput ->
    with outputPtr $ \pOutput ->
      with (fromIntegral rows :: Word64) $ \pRows ->
        with (fromIntegral cols :: Word64) $ \pCols ->
          withArray [castPtr pInput, castPtr pOutput, castPtr pRows, castPtr pCols] $ \kernelParams ->
            launch1dKernel (cachedKernelHandle cachedKernel) stream outputLength 0 kernelParams

launchAxisBroadcastKernel :: HasCallStack => CachedKernel -> Stream -> DevicePtr a -> DevicePtr a -> Int -> Int -> Int -> IO ()
launchAxisBroadcastKernel cachedKernel stream (DevicePtr inputPtr) (DevicePtr outputPtr) outputLength rows cols =
  with inputPtr $ \pInput ->
    with outputPtr $ \pOutput ->
      with (fromIntegral rows :: Word64) $ \pRows ->
        with (fromIntegral cols :: Word64) $ \pCols ->
          withArray [castPtr pInput, castPtr pOutput, castPtr pRows, castPtr pCols] $ \kernelParams ->
            launch1dKernel (cachedKernelHandle cachedKernel) stream outputLength 0 kernelParams

launchReduceKernel :: forall a. (HasCallStack, ArrayScalar a) => CachedKernel -> Stream -> DevicePtr a -> DevicePtr a -> Int -> IO ()
launchReduceKernel cachedKernel stream (DevicePtr inputPtr) (DevicePtr outputPtr) elementCount =
  with inputPtr $ \pInput ->
    with outputPtr $ \pOutput ->
      with (fromIntegral elementCount :: Word64) $ \pLength ->
        withArray [castPtr pInput, castPtr pOutput, castPtr pLength] $ \kernelParams ->
          launchReductionKernel (cachedKernelHandle cachedKernel) stream elementCount (fromIntegral (runtimeBlockSize * sizeOf (undefined :: a))) kernelParams

launchReduceCombineKernel :: (HasCallStack, ArrayScalar a) => CachedKernel -> Stream -> a -> DevicePtr a -> DevicePtr a -> IO ()
launchReduceCombineKernel cachedKernel stream initialValue (DevicePtr reducedPtr) (DevicePtr outputPtr) =
  with initialValue $ \pInitial ->
    with reducedPtr $ \pReduced ->
      with outputPtr $ \pOutput ->
        withArray [castPtr pInitial, castPtr pReduced, castPtr pOutput] $ \kernelParams ->
          withRawHipStream stream $ \rawStream ->
            hipModuleLaunchKernel
              (loadedHipKernelFunction (cachedKernelHandle cachedKernel))
              (HipDim3 1 1 1)
              (HipDim3 1 1 1)
              0
              (Just rawStream)
              kernelParams
              nullKernelExtras

launch1dKernel :: HasCallStack => LoadedHipKernel -> Stream -> Int -> Word32 -> Ptr (Ptr ()) -> IO ()
launch1dKernel loadedKernel stream elementCount sharedBytes kernelParams =
  withRawHipStream stream $ \rawStream ->
    hipModuleLaunchKernel
      (loadedHipKernelFunction loadedKernel)
      (HipDim3 (fromIntegral (ceilDiv elementCount runtimeBlockSize)) 1 1)
      (HipDim3 (fromIntegral runtimeBlockSize) 1 1)
      sharedBytes
      (Just rawStream)
      kernelParams
      nullKernelExtras

launchReductionKernel :: HasCallStack => LoadedHipKernel -> Stream -> Int -> Word32 -> Ptr (Ptr ()) -> IO ()
launchReductionKernel loadedKernel stream elementCount sharedBytes kernelParams =
  withRawHipStream stream $ \rawStream ->
    hipModuleLaunchKernel
      (loadedHipKernelFunction loadedKernel)
      (HipDim3 (fromIntegral (ceilDiv elementCount (2 * runtimeBlockSize))) 1 1)
      (HipDim3 (fromIntegral runtimeBlockSize) 1 1)
      sharedBytes
      (Just rawStream)
      kernelParams
      nullKernelExtras

runtimeBlockSize :: Int
runtimeBlockSize = 256

data AxisReduceDirection
  = AxisRowsReduce
  | AxisColsReduce
  deriving (Eq, Show)

data AxisBroadcastDirection
  = AxisRowsBroadcast
  | AxisColsBroadcast
  deriving (Eq, Show)

data AxisReduceKind
  = AxisReduceSum
  | AxisReduceMax
  deriving (Eq, Show)

fillKernelName :: String
fillKernelName = "molten_fill"

mapKernelName :: String
mapKernelName = "molten_map"

zipKernelName :: String
zipKernelName = "molten_zipWith"

reduceKernelName :: String
reduceKernelName = "molten_reduce_step"

reduceCombineKernelName :: String
reduceCombineKernelName = "molten_reduce_combine"

axisReduceKernelName :: AxisReduceDirection -> AxisReduceKind -> String
axisReduceKernelName direction reduceKind =
  case (direction, reduceKind) of
    (AxisRowsReduce, AxisReduceSum) -> "molten_sum_rows"
    (AxisColsReduce, AxisReduceSum) -> "molten_sum_cols"
    (AxisRowsReduce, AxisReduceMax) -> "molten_max_rows"
    (AxisColsReduce, AxisReduceMax) -> "molten_max_cols"

axisBroadcastKernelName :: AxisBroadcastDirection -> String
axisBroadcastKernelName direction =
  case direction of
    AxisRowsBroadcast -> "molten_broadcast_rows"
    AxisColsBroadcast -> "molten_broadcast_cols"

fillKernelKey :: forall a. ArrayScalar a => a -> String
fillKernelKey _ = "fill:" <> arrayScalarCType (Proxy :: Proxy a)

mapKernelKey :: forall a b. (ArrayScalar a, ArrayScalar b) => Unary a b -> String
mapKernelKey unary =
  unwords
    [ "map"
    , arrayScalarCType (Proxy :: Proxy a)
    , arrayScalarCType (Proxy :: Proxy b)
    , renderUnaryExpression unary
    ]

zipKernelKey :: forall a b c. (ArrayScalar a, ArrayScalar b, ArrayScalar c) => Binary a b c -> String
zipKernelKey binary =
  unwords
    [ "zipWith"
    , arrayScalarCType (Proxy :: Proxy a)
    , arrayScalarCType (Proxy :: Proxy b)
    , arrayScalarCType (Proxy :: Proxy c)
    , renderBinaryExpression binary
    ]

reduceKernelKey :: forall a. ArrayScalar a => Binary a a a -> String
reduceKernelKey binary =
  unwords
    [ "reduce"
    , arrayScalarCType (Proxy :: Proxy a)
    , renderBinaryExpression binary
    ]

reduceCombineKernelKey :: forall a. ArrayScalar a => Binary a a a -> String
reduceCombineKernelKey binary =
  unwords
    [ "reduce-combine"
    , arrayScalarCType (Proxy @a)
    , renderBinaryExpression binary
    ]

axisReduceKernelKey :: forall a. ArrayScalar a => Proxy a -> AxisReduceDirection -> AxisReduceKind -> String
axisReduceKernelKey _ direction reduceKind =
  unwords
    [ "axis-reduce"
    , show direction
    , show reduceKind
    , arrayScalarCType (Proxy @a)
    ]

axisBroadcastKernelKey :: forall a. ArrayScalar a => Proxy a -> AxisBroadcastDirection -> String
axisBroadcastKernelKey _ direction =
  unwords
    [ "axis-broadcast"
    , show direction
    , arrayScalarCType (Proxy @a)
    ]

kernelSupportSource :: String
kernelSupportSource =
  unlines
    [ "__device__ inline float2 molten_add_float2(float2 left, float2 right) {"
    , "  return make_float2(left.x + right.x, left.y + right.y);"
    , "}"
    , "__device__ inline float2 molten_mul_float2(float2 left, float2 right) {"
    , "  return make_float2(left.x * right.x - left.y * right.y, left.x * right.y + left.y * right.x);"
    , "}"
    , "__device__ inline double2 molten_add_double2(double2 left, double2 right) {"
    , "  return make_double2(left.x + right.x, left.y + right.y);"
    , "}"
    , "__device__ inline double2 molten_mul_double2(double2 left, double2 right) {"
    , "  return make_double2(left.x * right.x - left.y * right.y, left.x * right.y + left.y * right.x);"
    , "}"
    ]

fillKernelSource :: forall a. ArrayScalar a => Proxy a -> String
fillKernelSource _ =
  kernelSupportSource
    <> unlines
      [ "extern \"C\" __global__ void " <> fillKernelName <> "(" <> scalarType <> "* output, unsigned long long n, " <> scalarType <> " value) {"
    , "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;"
    , "  if (idx < n) {"
    , "    output[idx] = value;"
    , "  }"
    , "}"
    ]
  where
    scalarType = arrayScalarCType (Proxy :: Proxy a)

mapKernelSource :: forall a b. (ArrayScalar a, ArrayScalar b) => Proxy a -> Proxy b -> Unary a b -> String
mapKernelSource _ _ unary =
  kernelSupportSource
    <> unlines
      [ "extern \"C\" __global__ void " <> mapKernelName <> "(const " <> inputType <> "* input, " <> outputType <> "* output, unsigned long long n) {"
    , "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;"
    , "  if (idx < n) {"
    , "    " <> inputType <> " x0 = input[idx];"
    , "    output[idx] = " <> renderUnaryExpression unary <> ";"
    , "  }"
    , "}"
    ]
  where
    inputType = arrayScalarCType (Proxy :: Proxy a)
    outputType = arrayScalarCType (Proxy :: Proxy b)

zipKernelSource :: forall a b c. (ArrayScalar a, ArrayScalar b, ArrayScalar c) => Proxy a -> Proxy b -> Proxy c -> Binary a b c -> String
zipKernelSource _ _ _ binary =
  kernelSupportSource
    <> unlines
      [ "extern \"C\" __global__ void " <> zipKernelName <> "(const " <> leftType <> "* left, const " <> rightType <> "* right, " <> outputType <> "* output, unsigned long long n) {"
    , "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;"
    , "  if (idx < n) {"
    , "    " <> leftType <> " x0 = left[idx];"
    , "    " <> rightType <> " x1 = right[idx];"
    , "    output[idx] = " <> renderBinaryExpression binary <> ";"
    , "  }"
    , "}"
    ]
  where
    leftType = arrayScalarCType (Proxy :: Proxy a)
    rightType = arrayScalarCType (Proxy :: Proxy b)
    outputType = arrayScalarCType (Proxy :: Proxy c)

reduceKernelSource :: forall a. ArrayScalar a => Proxy a -> Binary a a a -> String
reduceKernelSource _ binary =
  kernelSupportSource
    <> unlines
      [ "__device__ inline " <> scalarType <> " molten_reduce_op(" <> scalarType <> " x0, " <> scalarType <> " x1) {"
    , "  return " <> renderBinaryExpression binary <> ";"
    , "}"
    , "extern \"C\" __global__ void " <> reduceKernelName <> "(const " <> scalarType <> "* input, " <> scalarType <> "* output, unsigned long long n) {"
    , "  extern __shared__ unsigned char raw_shared[];"
    , "  " <> scalarType <> "* shared = reinterpret_cast<" <> scalarType <> "*>(raw_shared);"
    , "  unsigned int tid = threadIdx.x;"
    , "  unsigned long long blockStart = 2ULL * blockIdx.x * blockDim.x;"
    , "  unsigned long long remaining = blockStart < n ? (n - blockStart) : 0ULL;"
    , "  unsigned int active = remaining == 0ULL ? 0U : (remaining >= blockDim.x ? blockDim.x : static_cast<unsigned int>(remaining));"
    , "  if (tid < active) {"
    , "    unsigned long long idx = blockStart + tid;"
    , "    " <> scalarType <> " acc = input[idx];"
    , "    unsigned long long pairIdx = idx + blockDim.x;"
    , "    if (pairIdx < n) {"
    , "      acc = molten_reduce_op(acc, input[pairIdx]);"
    , "    }"
    , "    shared[tid] = acc;"
    , "  }"
    , "  __syncthreads();"
    , "  unsigned int stride = 1U;"
    , "  while (stride < active) {"
    , "    stride <<= 1U;"
    , "  }"
    , "  while (stride > 1U) {"
    , "    stride >>= 1U;"
    , "    if (tid < stride && tid + stride < active) {"
    , "      shared[tid] = molten_reduce_op(shared[tid], shared[tid + stride]);"
    , "    }"
    , "    __syncthreads();"
    , "  }"
    , "  if (tid == 0U && active > 0U) {"
    , "    output[blockIdx.x] = shared[0];"
    , "  }"
    , "}"
    ]
  where
    scalarType = arrayScalarCType (Proxy :: Proxy a)

reduceCombineKernelSource :: forall a. ArrayScalar a => Proxy a -> Binary a a a -> String
reduceCombineKernelSource _ binary =
  kernelSupportSource
    <> unlines
      [ "extern \"C\" __global__ void " <> reduceCombineKernelName <> "(" <> scalarType <> " initialValue, const " <> scalarType <> "* reduced, " <> scalarType <> "* output) {"
    , "  if (blockIdx.x == 0 && threadIdx.x == 0) {"
    , "    " <> scalarType <> " x0 = initialValue;"
    , "    " <> scalarType <> " x1 = reduced[0];"
    , "    output[0] = " <> renderBinaryExpression binary <> ";"
    , "  }"
    , "}"
    ]
  where
    scalarType = arrayScalarCType (Proxy :: Proxy a)

axisReduceKernelSource :: forall a. ArrayScalar a => Proxy a -> AxisReduceDirection -> AxisReduceKind -> String
axisReduceKernelSource _ direction reduceKind =
  kernelSupportSource
    <> unlines
      [ axisReduceCombineSource reduceKind scalarType
      , "extern \"C\" __global__ void " <> axisReduceKernelName direction reduceKind <> "(const " <> scalarType <> "* input, " <> scalarType <> "* output, unsigned long long rows, unsigned long long cols) {"
      , "  unsigned long long outIdx = blockIdx.x * blockDim.x + threadIdx.x;"
      , "  unsigned long long outputLength = " <> outputLengthExpression direction <> ";"
      , "  if (outIdx < outputLength) {"
      , axisReduceInitialization direction scalarType
      , axisReduceLoop direction reduceKind
      , "    output[outIdx] = acc;"
      , "  }"
      , "}"
      ]
  where
    scalarType = arrayScalarCType (Proxy :: Proxy a)

axisBroadcastKernelSource :: forall a. ArrayScalar a => Proxy a -> AxisBroadcastDirection -> String
axisBroadcastKernelSource _ direction =
  kernelSupportSource
    <> unlines
      [ "extern \"C\" __global__ void " <> axisBroadcastKernelName direction <> "(const " <> scalarType <> "* input, " <> scalarType <> "* output, unsigned long long rows, unsigned long long cols) {"
      , "  unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;"
      , "  unsigned long long total = rows * cols;"
      , "  if (idx < total) {"
      , axisBroadcastIndexSource direction
      , "  }"
      , "}"
      ]
  where
    scalarType = arrayScalarCType (Proxy :: Proxy a)

axisReduceCombineSource :: AxisReduceKind -> String -> String
axisReduceCombineSource reduceKind scalarType =
  case reduceKind of
    AxisReduceSum ->
      "__device__ inline " <> scalarType <> " molten_axis_reduce_combine(" <> scalarType <> " left, " <> scalarType <> " right) { return left + right; }"
    AxisReduceMax ->
      "__device__ inline " <> scalarType <> " molten_axis_reduce_combine(" <> scalarType <> " left, " <> scalarType <> " right) { return left < right ? right : left; }"

outputLengthExpression :: AxisReduceDirection -> String
outputLengthExpression direction =
  case direction of
    AxisRowsReduce -> "rows"
    AxisColsReduce -> "cols"

axisReduceInitialization :: AxisReduceDirection -> String -> String
axisReduceInitialization direction scalarType =
  case direction of
    AxisRowsReduce ->
      "    unsigned long long base = outIdx * cols;\n    " <> scalarType <> " acc = input[base];"
    AxisColsReduce ->
      "    " <> scalarType <> " acc = input[outIdx];"

axisReduceLoop :: AxisReduceDirection -> AxisReduceKind -> String
axisReduceLoop direction reduceKind =
  case (direction, reduceKind) of
    (AxisRowsReduce, _) ->
      unlines
        [ "    for (unsigned long long col = 1ULL; col < cols; ++col) {"
        , "      acc = molten_axis_reduce_combine(acc, input[base + col]);"
        , "    }"
        ]
    (AxisColsReduce, _) ->
      unlines
        [ "    for (unsigned long long row = 1ULL; row < rows; ++row) {"
        , "      acc = molten_axis_reduce_combine(acc, input[row * cols + outIdx]);"
        , "    }"
        ]

axisBroadcastIndexSource :: AxisBroadcastDirection -> String
axisBroadcastIndexSource direction =
  case direction of
    AxisRowsBroadcast ->
      "    unsigned long long row = cols == 0ULL ? 0ULL : (idx / cols);\n    output[idx] = input[row];"
    AxisColsBroadcast ->
      "    unsigned long long col = cols == 0ULL ? 0ULL : (idx % cols);\n    output[idx] = input[col];"

ceilDiv :: Int -> Int -> Int
ceilDiv numerator denominator =
  (numerator + denominator - 1) `div` denominator

nullKernelExtras :: Ptr (Ptr ())
nullKernelExtras = castPtr (nullPtr :: Ptr ())
