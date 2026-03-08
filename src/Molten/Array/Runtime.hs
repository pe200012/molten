{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Molten.Array.Runtime
  ( ArrayRuntime
  , arrayRuntimeKernelCount
  , fillArray
  , fillArrayOn
  , mapArray
  , mapArrayOn
  , reduceAllArray
  , reduceAllArrayOn
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
import Foreign.C.Types (CSize)
import Foreign.Marshal.Array (withArray)
import Foreign.Marshal.Utils (with)
import Foreign.Ptr (Ptr, castPtr, nullPtr)
import Foreign.Storable (Storable, sizeOf)
import GHC.Stack (HasCallStack)
import Molten.Array.Device (DeviceArray, deviceArrayBuffer, deviceArraySize, mkDeviceArray, mkDeviceVector)
import Molten.Array.Expr
  ( ArrayScalar(arrayScalarCType)
  , Binary(..)
  , NumericExp
  , Unary(..)
  , renderBinaryExpression
  , renderUnaryExpression
  )
import Molten.Core.Buffer
  ( Buffer
  , Location(..)
  , bufferLength
  , bufferSizeInBytes
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

ceilDiv :: Int -> Int -> Int
ceilDiv numerator denominator =
  (numerator + denominator - 1) `div` denominator

nullKernelExtras :: Ptr (Ptr ())
nullKernelExtras = castPtr (nullPtr :: Ptr ())
