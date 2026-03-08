{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Molten.Array.Program
  ( MatrixGemmValue(..)
  , Program
  , ProgramBuilder
  , ProgramOutput(..)
  , ProgramRuntime
  , ReferenceOutput(..)
  , Value
  , axpyVectorP
  , buildProgram
  , fillArrayP
  , fftForwardP
  , fftInverseP
  , gemmMatrixP
  , inputArray
  , inputDeviceArray
  , mapExpr
  , programNodeDependencies
  , programNodeIds
  , programResultValue
  , randNormalP
  , randUniformP
  , reduceAll
  , reshapeValue
  , runProgram
  , runProgramCpu
  , scheduleProgram
  , valueId
  , valueSize
  , withProgramRuntime
  , zipWithExpr
  ) where

import Control.Exception (bracket)
import Control.Monad (foldM, forM_, when)
import Control.Monad.IO.Class (MonadIO, liftIO)
import Control.Monad.State.Strict (StateT, get, modify', runStateT)
import qualified Data.Dynamic as Dynamic
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Typeable (Typeable)
import qualified Data.Massiv.Array as A
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Array.Device (DeviceArray, deviceArraySize, mkDeviceArray)
import Molten.Array.Expr (ArrayScalar, Binary, NumericExp, Unary)
import Molten.Array.Runtime
  ( ArrayRuntime
  , fillArrayOn
  , mapArrayOn
  , reduceAllArrayOn
  , withArrayRuntime
  , zipWithArrayOn
  )
import Molten.Array.Transfer (cloneDeviceArray, copyHostArrayToDevice, reshapeDeviceArray)
import Molten.Internal.Reference.Array (fillArrayRef, mapArrayRef, reduceAllArrayRef, reshapeArrayRef, zipWithArrayRef)
import Molten.Internal.Reference.BLAS (MatrixGemmRef(..), axpyVectorRef, gemmMatrixRef)
import Molten.BLAS (MatrixGemm(..), axpyVector, gemmMatrix)
import Molten.BLAS.Types (BlasElement, Transpose)
import Molten.Core.Buffer (newDeviceBufferOn)
import Molten.Core.Context (Context, contextDefaultStream, contextDeviceId)
import Molten.Core.Event (Event, createEvent, destroyEvent, recordEvent, waitEvent)
import Molten.Core.Stream (Stream, createStream, destroyStream, streamDeviceId, synchronizeStream)
import Molten.FFT (FftComplex, FftShape, fftForwardC2C, fftInverseC2C)
import Molten.FFT.Runtime (FftRuntime, withFftRuntime)
import Molten.Internal.Scheduler
  ( ScheduledNode(..)
  , SchedulerNode(..)
  , SchedulerResource(..)
  , scheduleNodes
  )
import Molten.RAND (RandNormal, RandUniform, randNormalOn, randUniformOn)
import Molten.RAND.Runtime (RandGeneratorConfig, RandRuntime, withRandRuntime)
import ROCm.FFI.Core.Exception (throwArgumentError)

newtype ProgramBuilder a = ProgramBuilder {unProgramBuilder :: StateT BuilderState IO a}
  deriving newtype (Functor, Applicative, Monad, MonadIO)

data Value ix a = Value
  { valueId :: !Int
  , valueSize :: !(A.Sz ix)
  }
  deriving (Eq, Show)

data MatrixGemmValue a = MatrixGemmValue
  { matrixGemmValueTransA :: !Transpose
  , matrixGemmValueTransB :: !Transpose
  , matrixGemmValueAlpha :: !a
  , matrixGemmValueA :: !(Value A.Ix2 a)
  , matrixGemmValueB :: !(Value A.Ix2 a)
  , matrixGemmValueBeta :: !a
  , matrixGemmValueC :: !(Value A.Ix2 a)
  }
  deriving (Eq, Show)

data Program a = Program
  { programNodes :: ![ProgramNode]
  , programResult :: !a
  }

data ProgramRuntime = ProgramRuntime
  { programRuntimeContext :: !Context
  , programRuntimeArrayRuntime :: !ArrayRuntime
  , programRuntimeFftRuntime :: !FftRuntime
  , programRuntimeRandRuntime :: !RandRuntime
  , programRuntimeStreams :: ![Stream]
  }

data BuilderState = BuilderState
  { builderNextValueId :: !Int
  , builderNodesRev :: ![ProgramNode]
  }

data ProgramNode
  = forall ix a.
    (A.Index ix, Typeable ix, Typeable a, Storable a) =>
    InputNode !(Value ix a) !(DeviceArray ix a)
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, Storable a) =>
    HostInputNode !(Value ix a) !(A.Array A.S ix a)
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, ArrayScalar a) =>
    FillNode !(Value ix a) !a
  | forall ix a b.
    (A.Index ix, Typeable ix, Typeable a, Typeable b, ArrayScalar a, ArrayScalar b, Storable b) =>
    MapNode !(Value ix b) !(Unary a b) !(Value ix a)
  | forall ix a b c.
    (A.Index ix, Typeable ix, Typeable a, Typeable b, Typeable c, ArrayScalar a, ArrayScalar b, ArrayScalar c, Storable c) =>
    ZipNode !(Value ix c) !(Binary a b c) !(Value ix a) !(Value ix b)
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, ArrayScalar a, NumericExp a) =>
    ReduceNode !(Value A.Ix1 a) !(Binary a a a) !a !(Value ix a)
  | forall ix ix' a.
    (A.Index ix, A.Index ix', Typeable ix, Typeable ix', Typeable a, Storable a) =>
    ReshapeNode !(Value ix' a) !(Value ix a)
  | forall a.
    (Typeable a, BlasElement a, Num a, Storable a) =>
    AxpyNode !(Value A.Ix1 a) !a !(Value A.Ix1 a) !(Value A.Ix1 a)
  | forall a.
    (Typeable a, BlasElement a, Num a, Storable a) =>
    GemmNode !(Value A.Ix2 a) !(MatrixGemmValue a)
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, FftShape ix, FftComplex a) =>
    FftForwardNode !(Value ix a) !(Value ix a)
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, FftShape ix, FftComplex a) =>
    FftInverseNode !(Value ix a) !(Value ix a)
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, RandUniform a, ArrayScalar a) =>
    RandUniformNode !(Value ix a) !RandGeneratorConfig
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, RandNormal a, ArrayScalar a) =>
    RandNormalNode !(Value ix a) !RandGeneratorConfig !a !a

class ProgramOutput a where
  type ProgramResult a
  resolveProgramOutput :: ValueStore -> a -> IO (ProgramResult a)

instance (A.Index ix, Typeable ix, Typeable a, Storable a) => ProgramOutput (Value ix a) where
  type ProgramResult (Value ix a) = DeviceArray ix a
  resolveProgramOutput store value = lookupStoredValue "resolveProgramOutput" store value

instance ProgramOutput () where
  type ProgramResult () = ()
  resolveProgramOutput _ () = pure ()

instance (ProgramOutput a, ProgramOutput b) => ProgramOutput (a, b) where
  type ProgramResult (a, b) = (ProgramResult a, ProgramResult b)
  resolveProgramOutput store (a, b) = do
    left <- resolveProgramOutput store a
    right <- resolveProgramOutput store b
    pure (left, right)

class ReferenceOutput a where
  type ReferenceResult a
  resolveReferenceOutput :: ReferenceStore -> a -> IO (ReferenceResult a)

instance (A.Index ix, Typeable ix, Typeable a, Storable a) => ReferenceOutput (Value ix a) where
  type ReferenceResult (Value ix a) = A.Array A.S ix a
  resolveReferenceOutput store value = lookupStoredReferenceValue "resolveReferenceOutput" store value

instance ReferenceOutput () where
  type ReferenceResult () = ()
  resolveReferenceOutput _ () = pure ()

instance (ReferenceOutput a, ReferenceOutput b) => ReferenceOutput (a, b) where
  type ReferenceResult (a, b) = (ReferenceResult a, ReferenceResult b)
  resolveReferenceOutput store (a, b) = do
    left <- resolveReferenceOutput store a
    right <- resolveReferenceOutput store b
    pure (left, right)

withProgramRuntime :: HasCallStack => Context -> (ProgramRuntime -> IO a) -> IO a
withProgramRuntime ctx action =
  withArrayRuntime ctx $ \arrayRuntime ->
    withFftRuntime ctx $ \fftRuntime ->
      withRandRuntime ctx $ \randRuntime ->
        bracket (createProgramStreams ctx) destroyProgramStreams $ \streams ->
          action
            ProgramRuntime
              { programRuntimeContext = ctx
              , programRuntimeArrayRuntime = arrayRuntime
              , programRuntimeFftRuntime = fftRuntime
              , programRuntimeRandRuntime = randRuntime
              , programRuntimeStreams = streams
              }

buildProgram :: HasCallStack => ProgramBuilder a -> IO (Program a)
buildProgram builder = do
  (result, finalState) <- runStateT (unProgramBuilder builder) initialBuilderState
  pure Program {programNodes = reverse (builderNodesRev finalState), programResult = result}

initialBuilderState :: BuilderState
initialBuilderState = BuilderState {builderNextValueId = 0, builderNodesRev = []}

inputDeviceArray :: (A.Index ix, Typeable ix, Typeable a, Storable a) => DeviceArray ix a -> ProgramBuilder (Value ix a)
inputDeviceArray deviceArray = do
  value <- freshValue (deviceArraySize deviceArray)
  emitNode (InputNode value deviceArray)
  pure value

inputArray :: (A.Index ix, Typeable ix, Typeable a, Storable a) => A.Array A.S ix a -> ProgramBuilder (Value ix a)
inputArray array = do
  value <- freshValue (A.size array)
  emitNode (HostInputNode value array)
  pure value

fillArrayP :: (A.Index ix, Typeable ix, Typeable a, ArrayScalar a) => a -> A.Sz ix -> ProgramBuilder (Value ix a)
fillArrayP value size = do
  builtValue <- freshValue size
  emitNode (FillNode builtValue value)
  pure builtValue

mapExpr :: (A.Index ix, Typeable ix, Typeable a, Typeable b, ArrayScalar a, ArrayScalar b, Storable b) => Unary a b -> Value ix a -> ProgramBuilder (Value ix b)
mapExpr unary inputValue = do
  outputValue <- freshValue (valueSize inputValue)
  emitNode (MapNode outputValue unary inputValue)
  pure outputValue

zipWithExpr :: (A.Index ix, Typeable ix, Typeable a, Typeable b, Typeable c, ArrayScalar a, ArrayScalar b, ArrayScalar c, Storable c) => Binary a b c -> Value ix a -> Value ix b -> ProgramBuilder (Value ix c)
zipWithExpr binary leftValue rightValue = do
  if valueSize leftValue /= valueSize rightValue
    then liftIO (throwArgumentError "zipWithExpr" "input values must have the same shape")
    else pure ()
  outputValue <- freshValue (valueSize leftValue)
  emitNode (ZipNode outputValue binary leftValue rightValue)
  pure outputValue

reduceAll :: (A.Index ix, Typeable ix, Typeable a, ArrayScalar a, NumericExp a) => Binary a a a -> a -> Value ix a -> ProgramBuilder (Value A.Ix1 a)
reduceAll binary initialValue inputValue = do
  outputValue <- freshValue (A.Sz1 1)
  emitNode (ReduceNode outputValue binary initialValue inputValue)
  pure outputValue

reshapeValue :: (A.Index ix, A.Index ix', Typeable ix, Typeable ix', Typeable a, Storable a) => A.Sz ix' -> Value ix a -> ProgramBuilder (Value ix' a)
reshapeValue targetSize inputValue = do
  if A.totalElem targetSize /= A.totalElem (valueSize inputValue)
    then liftIO (throwArgumentError "reshapeValue" "reshape must preserve totalElem")
    else pure ()
  outputValue <- freshValue targetSize
  emitNode (ReshapeNode outputValue inputValue)
  pure outputValue

axpyVectorP :: (Typeable a, BlasElement a, Num a, Storable a) => a -> Value A.Ix1 a -> Value A.Ix1 a -> ProgramBuilder (Value A.Ix1 a)
axpyVectorP alpha xValue yValue = do
  when (valueSize xValue /= valueSize yValue) (liftIO (throwArgumentError "axpyVectorP" "input vectors must have the same shape"))
  outputValue <- freshValue (valueSize yValue)
  emitNode (AxpyNode outputValue alpha xValue yValue)
  pure outputValue

gemmMatrixP :: (Typeable a, BlasElement a, Num a, Storable a) => MatrixGemmValue a -> ProgramBuilder (Value A.Ix2 a)
gemmMatrixP matrixSpec = do
  outputValue <- freshValue (valueSize (matrixGemmValueC matrixSpec))
  emitNode (GemmNode outputValue matrixSpec)
  pure outputValue

fftForwardP :: (A.Index ix, Typeable ix, Typeable a, FftShape ix, FftComplex a) => Value ix a -> ProgramBuilder (Value ix a)
fftForwardP inputValue = do
  outputValue <- freshValue (valueSize inputValue)
  emitNode (FftForwardNode outputValue inputValue)
  pure outputValue

fftInverseP :: (A.Index ix, Typeable ix, Typeable a, FftShape ix, FftComplex a) => Value ix a -> ProgramBuilder (Value ix a)
fftInverseP inputValue = do
  outputValue <- freshValue (valueSize inputValue)
  emitNode (FftInverseNode outputValue inputValue)
  pure outputValue

randUniformP :: (A.Index ix, Typeable ix, Typeable a, RandUniform a, ArrayScalar a) => RandGeneratorConfig -> A.Sz ix -> ProgramBuilder (Value ix a)
randUniformP config size = do
  outputValue <- freshValue size
  emitNode (RandUniformNode outputValue config)
  pure outputValue

randNormalP :: (A.Index ix, Typeable ix, Typeable a, RandNormal a, ArrayScalar a) => RandGeneratorConfig -> a -> a -> A.Sz ix -> ProgramBuilder (Value ix a)
randNormalP config meanValue stddevValue size = do
  outputValue <- freshValue size
  emitNode (RandNormalNode outputValue config meanValue stddevValue)
  pure outputValue

programNodeIds :: Program a -> [Int]
programNodeIds = fmap programNodeOutputId . programNodes

programNodeDependencies :: Program a -> [(Int, [Int])]
programNodeDependencies program =
  [ (programNodeOutputId node, programNodeInputIds node)
  | node <- programNodes program
  ]

programResultValue :: Program a -> a
programResultValue = programResult

scheduleProgram :: HasCallStack => ProgramRuntime -> Program a -> IO [ScheduledNode]
scheduleProgram runtime program =
  scheduleNodes (length (programRuntimeStreams runtime)) (fmap toSchedulerNode (programNodes program))

runProgram :: (HasCallStack, ProgramOutput a) => ProgramRuntime -> Program a -> IO (ProgramResult a)
runProgram runtime program = do
  scheduledNodes <- scheduleProgram runtime program
  valueStore <- executeScheduledProgram runtime scheduledNodes (programNodes program)
  resolveProgramOutput valueStore (programResult program)

runProgramCpu :: (HasCallStack, ReferenceOutput a) => Program a -> IO (ReferenceResult a)
runProgramCpu program = do
  referenceStore <- executeReferenceProgram (programNodes program)
  resolveReferenceOutput referenceStore (programResult program)

freshValue :: A.Sz ix -> ProgramBuilder (Value ix a)
freshValue size =
  ProgramBuilder $ do
    state <- get
    let nextId = builderNextValueId state
    modify' $ \currentState -> currentState {builderNextValueId = nextId + 1}
    pure Value {valueId = nextId, valueSize = size}

emitNode :: ProgramNode -> ProgramBuilder ()
emitNode node =
  ProgramBuilder $ modify' $ \state -> state {builderNodesRev = node : builderNodesRev state}

createProgramStreams :: HasCallStack => Context -> IO [Stream]
createProgramStreams ctx = do
  stream1 <- createStream (contextDeviceId ctx)
  stream2 <- createStream (contextDeviceId ctx)
  stream3 <- createStream (contextDeviceId ctx)
  pure [contextDefaultStream ctx, stream1, stream2, stream3]

destroyProgramStreams :: [Stream] -> IO ()
destroyProgramStreams streams =
  mapM_ destroyStream (drop 1 streams)

type ValueStore = Map Int Dynamic.Dynamic

type ReferenceStore = Map Int Dynamic.Dynamic

executeReferenceProgram :: HasCallStack => [ProgramNode] -> IO ReferenceStore
executeReferenceProgram = foldM executeReferenceNode Map.empty

executeReferenceNode :: HasCallStack => ReferenceStore -> ProgramNode -> IO ReferenceStore
executeReferenceNode store node =
  case node of
    InputNode _ _ ->
      throwArgumentError "runProgramCpu" "device-only inputs are not supported by the CPU reference evaluator"
    HostInputNode outputValue hostArray ->
      pure (storeReferenceValue store outputValue hostArray)
    FillNode outputValue fillValue -> do
      hostArray <- fillArrayRef fillValue (valueSize outputValue)
      pure (storeReferenceValue store outputValue hostArray)
    MapNode outputValue unary inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- mapArrayRef unary hostInput
      pure (storeReferenceValue store outputValue hostArray)
    ZipNode outputValue binary leftValue rightValue -> do
      leftArray <- lookupStoredReferenceValue "runProgramCpu" store leftValue
      rightArray <- lookupStoredReferenceValue "runProgramCpu" store rightValue
      hostArray <- zipWithArrayRef binary leftArray rightArray
      pure (storeReferenceValue store outputValue hostArray)
    ReduceNode outputValue binary initialValue inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- reduceAllArrayRef binary initialValue hostInput
      pure (storeReferenceValue store outputValue hostArray)
    ReshapeNode outputValue inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- reshapeArrayRef (valueSize outputValue) hostInput
      pure (storeReferenceValue store outputValue hostArray)
    AxpyNode outputValue alpha xValue yValue -> do
      xArray <- lookupStoredReferenceValue "runProgramCpu" store xValue
      yArray <- lookupStoredReferenceValue "runProgramCpu" store yValue
      hostArray <- axpyVectorRef alpha xArray yArray
      pure (storeReferenceValue store outputValue hostArray)
    GemmNode outputValue matrixSpec -> do
      aArray <- lookupStoredReferenceValue "runProgramCpu" store (matrixGemmValueA matrixSpec)
      bArray <- lookupStoredReferenceValue "runProgramCpu" store (matrixGemmValueB matrixSpec)
      cArray <- lookupStoredReferenceValue "runProgramCpu" store (matrixGemmValueC matrixSpec)
      hostArray <-
        gemmMatrixRef
          MatrixGemmRef
            { matrixGemmRefTransA = matrixGemmValueTransA matrixSpec
            , matrixGemmRefTransB = matrixGemmValueTransB matrixSpec
            , matrixGemmRefAlpha = matrixGemmValueAlpha matrixSpec
            , matrixGemmRefA = aArray
            , matrixGemmRefB = bArray
            , matrixGemmRefBeta = matrixGemmValueBeta matrixSpec
            , matrixGemmRefC = cArray
            }
      pure (storeReferenceValue store outputValue hostArray)
    FftForwardNode _ _ ->
      throwArgumentError "runProgramCpu" "unsupported reference node: FftForwardNode"
    FftInverseNode _ _ ->
      throwArgumentError "runProgramCpu" "unsupported reference node: FftInverseNode"
    RandUniformNode _ _ ->
      throwArgumentError "runProgramCpu" "unsupported reference node: RandUniformNode"
    RandNormalNode _ _ _ _ ->
      throwArgumentError "runProgramCpu" "unsupported reference node: RandNormalNode"

executeScheduledProgram :: HasCallStack => ProgramRuntime -> [ScheduledNode] -> [ProgramNode] -> IO ValueStore
executeScheduledProgram runtime scheduledNodes nodes = do
  let nodeMap = Map.fromList [(programNodeOutputId node, node) | node <- nodes]
      executeOne :: (ValueStore, Map Int Event, Map Int Int) -> ScheduledNode -> IO (ValueStore, Map Int Event, Map Int Int)
      executeOne (store, eventMap, slotMap) scheduledNode = do
        let streamSlot = scheduledNodeStreamSlot scheduledNode
            stream = programRuntimeStreams runtime !! streamSlot
        waitForDependencies stream streamSlot eventMap slotMap (scheduledNodeDependencies scheduledNode)
        node <- lookupProgramNode nodeMap (scheduledNodeId scheduledNode)
        (store', maybeEvent) <- executeNode runtime stream node store
        let propagatedEvent =
              case maybeEvent of
                Just event -> Just event
                Nothing -> inheritDependencyEvent eventMap node
            eventMap' = maybe eventMap (\event -> Map.insert (scheduledNodeId scheduledNode) event eventMap) propagatedEvent
            slotMap' = Map.insert (scheduledNodeId scheduledNode) streamSlot slotMap
        pure (store', eventMap', slotMap')
  (store, eventMap, _) <- foldM executeOne (Map.empty, Map.empty, Map.empty) scheduledNodes
  mapM_ synchronizeStream (programRuntimeStreams runtime)
  mapM_ destroyEvent (Map.elems eventMap)
  pure store

waitForDependencies :: Stream -> Int -> Map Int Event -> Map Int Int -> [Int] -> IO ()
waitForDependencies stream streamSlot eventMap slotMap dependencyIds =
  forM_ dependencyIds $ \dependencyId ->
    case (Map.lookup dependencyId slotMap, Map.lookup dependencyId eventMap) of
      (Just dependencySlot, Just dependencyEvent)
        | dependencySlot /= streamSlot -> waitEvent stream dependencyEvent
      _ -> pure ()

executeNode :: HasCallStack => ProgramRuntime -> Stream -> ProgramNode -> ValueStore -> IO (ValueStore, Maybe Event)
executeNode runtime stream node store =
  case node of
    InputNode outputValue deviceArray ->
      pure (storeValue store outputValue deviceArray, Nothing)
    HostInputNode outputValue hostArray -> do
      deviceArray <- copyHostArrayToDevice (programRuntimeContext runtime) hostArray
      pure (storeValue store outputValue deviceArray, Nothing)
    FillNode outputValue fillValue -> do
      deviceArray <- fillArrayOn stream (programRuntimeArrayRuntime runtime) fillValue (valueSize outputValue)
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    MapNode outputValue unary inputValue -> do
      inputArray <- lookupStoredValue "MapNode" store inputValue
      deviceArray <- mapArrayOn stream (programRuntimeArrayRuntime runtime) unary inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    ZipNode outputValue binary leftValue rightValue -> do
      leftArray <- lookupStoredValue "ZipNode" store leftValue
      rightArray <- lookupStoredValue "ZipNode" store rightValue
      deviceArray <- zipWithArrayOn stream (programRuntimeArrayRuntime runtime) binary leftArray rightArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    ReduceNode outputValue binary initialValue inputValue -> do
      inputArray <- lookupStoredValue "ReduceNode" store inputValue
      deviceArray <- reduceAllArrayOn stream (programRuntimeArrayRuntime runtime) binary initialValue inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    ReshapeNode outputValue inputValue -> do
      inputArray <- lookupStoredValue "ReshapeNode" store inputValue
      deviceArray <- reshapeDeviceArray (valueSize outputValue) inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    AxpyNode outputValue alpha xValue yValue -> do
      xArray <- lookupStoredValue "AxpyNode" store xValue
      yArray <- lookupStoredValue "AxpyNode" store yValue
      clonedY <- cloneDeviceArray (programRuntimeContext runtime) yArray
      axpyVector (programRuntimeContext runtime) alpha xArray clonedY
      event <- recordNodeEvent (contextDefaultStream (programRuntimeContext runtime))
      pure (storeValue store outputValue clonedY, Just event)
    GemmNode outputValue matrixSpec -> do
      aArray <- lookupStoredValue "GemmNode" store (matrixGemmValueA matrixSpec)
      bArray <- lookupStoredValue "GemmNode" store (matrixGemmValueB matrixSpec)
      cArray <- lookupStoredValue "GemmNode" store (matrixGemmValueC matrixSpec)
      clonedC <- cloneDeviceArray (programRuntimeContext runtime) cArray
      gemmMatrix
        (programRuntimeContext runtime)
        MatrixGemm
          { matrixGemmTransA = matrixGemmValueTransA matrixSpec
          , matrixGemmTransB = matrixGemmValueTransB matrixSpec
          , matrixGemmAlpha = matrixGemmValueAlpha matrixSpec
          , matrixGemmA = aArray
          , matrixGemmB = bArray
          , matrixGemmBeta = matrixGemmValueBeta matrixSpec
          , matrixGemmC = clonedC
          }
      event <- recordNodeEvent (contextDefaultStream (programRuntimeContext runtime))
      pure (storeValue store outputValue clonedC, Just event)
    FftForwardNode outputValue inputValue -> do
      inputArray <- lookupStoredValue "FftForwardNode" store inputValue
      deviceArray <- fftForwardC2C (programRuntimeFftRuntime runtime) inputArray
      event <- recordNodeEvent (contextDefaultStream (programRuntimeContext runtime))
      pure (storeValue store outputValue deviceArray, Just event)
    FftInverseNode outputValue inputValue -> do
      inputArray <- lookupStoredValue "FftInverseNode" store inputValue
      deviceArray <- fftInverseC2C (programRuntimeFftRuntime runtime) inputArray
      event <- recordNodeEvent (contextDefaultStream (programRuntimeContext runtime))
      pure (storeValue store outputValue deviceArray, Just event)
    RandUniformNode outputValue config -> do
      deviceArray <- allocateProgramArray stream outputValue
      randUniformOn stream (programRuntimeRandRuntime runtime) config deviceArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    RandNormalNode outputValue config meanValue stddevValue -> do
      deviceArray <- allocateProgramArray stream outputValue
      randNormalOn stream (programRuntimeRandRuntime runtime) config meanValue stddevValue deviceArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)

allocateProgramArray :: (A.Index ix, Storable a) => Stream -> Value ix a -> IO (DeviceArray ix a)
allocateProgramArray stream outputValue = do
  deviceBuffer <- newDeviceBufferOn (streamDeviceId stream) (A.totalElem (valueSize outputValue))
  mkDeviceArray (valueSize outputValue) deviceBuffer

recordNodeEvent :: HasCallStack => Stream -> IO Event
recordNodeEvent stream = do
  event <- createEvent (streamDeviceId stream)
  recordEvent event stream
  pure event

inheritDependencyEvent :: Map Int Event -> ProgramNode -> Maybe Event
inheritDependencyEvent eventMap node =
  case programNodeInputIds node of
    dependencyId : _ -> Map.lookup dependencyId eventMap
    [] -> Nothing

lookupProgramNode :: HasCallStack => Map Int ProgramNode -> Int -> IO ProgramNode
lookupProgramNode nodeMap nodeId =
  case Map.lookup nodeId nodeMap of
    Just node -> pure node
    Nothing -> throwArgumentError "lookupProgramNode" ("unknown node id: " <> show nodeId)

lookupStoredValue :: forall ix a. (A.Index ix, Typeable ix, Typeable a, Storable a) => String -> ValueStore -> Value ix a -> IO (DeviceArray ix a)
lookupStoredValue functionName store value =
  case Map.lookup (valueId value) store >>= Dynamic.fromDynamic of
    Just deviceArray -> pure deviceArray
    Nothing -> throwArgumentError functionName ("missing value: " <> show (valueId value))

lookupStoredReferenceValue :: forall ix a. (A.Index ix, Typeable ix, Typeable a, Storable a) => String -> ReferenceStore -> Value ix a -> IO (A.Array A.S ix a)
lookupStoredReferenceValue functionName store value =
  case Map.lookup (valueId value) store >>= Dynamic.fromDynamic of
    Just hostArray -> pure hostArray
    Nothing -> throwArgumentError functionName ("missing reference value: " <> show (valueId value))

storeValue :: (Typeable ix, Typeable a) => ValueStore -> Value ix a -> DeviceArray ix a -> ValueStore
storeValue store outputValue deviceArray =
  Map.insert (valueId outputValue) (Dynamic.toDyn deviceArray) store

storeReferenceValue :: (Typeable ix, Typeable a) => ReferenceStore -> Value ix a -> A.Array A.S ix a -> ReferenceStore
storeReferenceValue store outputValue hostArray =
  Map.insert (valueId outputValue) (Dynamic.toDyn hostArray) store

programNodeOutputId :: ProgramNode -> Int
programNodeOutputId node =
  case node of
    InputNode outputValue _ -> valueId outputValue
    HostInputNode outputValue _ -> valueId outputValue
    FillNode outputValue _ -> valueId outputValue
    MapNode outputValue _ _ -> valueId outputValue
    ZipNode outputValue _ _ _ -> valueId outputValue
    ReduceNode outputValue _ _ _ -> valueId outputValue
    ReshapeNode outputValue _ -> valueId outputValue
    AxpyNode outputValue _ _ _ -> valueId outputValue
    GemmNode outputValue _ -> valueId outputValue
    FftForwardNode outputValue _ -> valueId outputValue
    FftInverseNode outputValue _ -> valueId outputValue
    RandUniformNode outputValue _ -> valueId outputValue
    RandNormalNode outputValue _ _ _ -> valueId outputValue

programNodeInputIds :: ProgramNode -> [Int]
programNodeInputIds node =
  case node of
    InputNode _ _ -> []
    HostInputNode _ _ -> []
    FillNode _ _ -> []
    MapNode _ _ inputValue -> [valueId inputValue]
    ZipNode _ _ leftValue rightValue -> [valueId leftValue, valueId rightValue]
    ReduceNode _ _ _ inputValue -> [valueId inputValue]
    ReshapeNode _ inputValue -> [valueId inputValue]
    AxpyNode _ _ xValue yValue -> [valueId xValue, valueId yValue]
    GemmNode _ matrixSpec -> [valueId (matrixGemmValueA matrixSpec), valueId (matrixGemmValueB matrixSpec), valueId (matrixGemmValueC matrixSpec)]
    FftForwardNode _ inputValue -> [valueId inputValue]
    FftInverseNode _ inputValue -> [valueId inputValue]
    RandUniformNode _ _ -> []
    RandNormalNode _ _ _ _ -> []

toSchedulerNode :: ProgramNode -> SchedulerNode
toSchedulerNode node =
  SchedulerNode
    { schedulerNodeId = programNodeOutputId node
    , schedulerNodeDependencies = programNodeInputIds node
    , schedulerNodeResource =
        case node of
          InputNode _ _ -> SchedulerResourceInput
          HostInputNode _ _ -> SchedulerResourceInput
          FillNode _ _ -> SchedulerResourceJit
          MapNode _ _ _ -> SchedulerResourceJit
          ZipNode _ _ _ _ -> SchedulerResourceJit
          ReduceNode _ _ _ _ -> SchedulerResourceJit
          ReshapeNode _ _ -> SchedulerResourceShape
          AxpyNode _ _ _ _ -> SchedulerResourceBlas
          GemmNode _ _ -> SchedulerResourceBlas
          FftForwardNode _ _ -> SchedulerResourceFft
          FftInverseNode _ _ -> SchedulerResourceFft
          RandUniformNode _ _ -> SchedulerResourceRand
          RandNormalNode _ _ _ _ -> SchedulerResourceRand
    }
