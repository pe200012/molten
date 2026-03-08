{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeFamilies #-}

module Molten.Array.Program
  ( LoopState
  , MatrixGemmValue(..)
  , Program
  , ProgramBuilder
  , ProgramOutput(..)
  , ProgramRuntime
  , ReferenceOutput(..)
  , Value
  , axpyVectorP
  , broadcastColsP
  , broadcastRowsP
  , buildProgram
  , fillArrayP
  , fftForwardP
  , fftInverseP
  , forLoopP
  , gemmMatrixP
  , inputArray
  , inputDeviceArray
  , mapExpr
  , maxColsP
  , maxRowsP
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
  , softmaxRowsP
  , sumColsP
  , sumRowsP
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
import qualified Data.Set as Set
import Data.Typeable (Typeable, cast)
import qualified Data.Massiv.Array as A
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Array.Device (DeviceArray, deviceArraySize, mkDeviceArray)
import Molten.Array.Expr
  ( ArrayScalar
  , Binary
  , Comparable
  , FloatingExp
  , NumericExp
  , Unary
  , binary
  , constant
  , expE
  , recipE
  , unary
  , (.+.)
  , (.*.)
  )
import Molten.Array.Runtime
  ( ArrayRuntime
  , broadcastColsArrayOn
  , broadcastRowsArrayOn
  , fillArrayOn
  , mapArrayOn
  , maxColsArrayOn
  , maxRowsArrayOn
  , reduceAllArrayOn
  , sumColsArrayOn
  , sumRowsArrayOn
  , withArrayRuntime
  , zipWithArrayOn
  )
import Molten.Array.Transfer (cloneDeviceArray, copyHostArrayToDevice, reshapeDeviceArray)
import Molten.Internal.Reference.Array (fillArrayRef, mapArrayRef, reduceAllArrayRef, reshapeArrayRef, zipWithArrayRef)
import Molten.Internal.Reference.Axis2D
  ( broadcastColsRef
  , broadcastRowsRef
  , maxColsRef
  , maxRowsRef
  , sumColsRef
  , sumRowsRef
  )
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
  , builderInsideLoop :: !Bool
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
    (Typeable a, ArrayScalar a, NumericExp a) =>
    SumRowsNode !(Value A.Ix1 a) !(Value A.Ix2 a)
  | forall a.
    (Typeable a, ArrayScalar a, NumericExp a) =>
    SumColsNode !(Value A.Ix1 a) !(Value A.Ix2 a)
  | forall a.
    (Typeable a, ArrayScalar a, Comparable a) =>
    MaxRowsNode !(Value A.Ix1 a) !(Value A.Ix2 a)
  | forall a.
    (Typeable a, ArrayScalar a, Comparable a) =>
    MaxColsNode !(Value A.Ix1 a) !(Value A.Ix2 a)
  | forall a.
    (Typeable a, ArrayScalar a) =>
    BroadcastRowsNode !(Value A.Ix2 a) !Int !(Value A.Ix1 a)
  | forall a.
    (Typeable a, ArrayScalar a) =>
    BroadcastColsNode !(Value A.Ix2 a) !Int !(Value A.Ix1 a)
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
  | LoopNode !LoopToken !LoopSpec
  | LoopResultNode !SomeValue !LoopToken !Int

newtype LoopToken = LoopToken {loopTokenId :: Int}
  deriving (Eq, Show)

data SomeValue = forall ix a.
  (A.Index ix, Typeable ix, Typeable a, Storable a) =>
  SomeValue !(Value ix a)

class LoopState s where
  traverseLoopState :: Applicative f => (forall ix a. (A.Index ix, Typeable ix, Typeable a, Storable a) => Value ix a -> f (Value ix a)) -> s -> f s
  flattenLoopState :: s -> [SomeValue]

instance LoopState () where
  traverseLoopState _ () = pure ()
  flattenLoopState () = []

instance (A.Index ix, Typeable ix, Typeable a, Storable a) => LoopState (Value ix a) where
  traverseLoopState function value = function value
  flattenLoopState value = [SomeValue value]

instance (LoopState a, LoopState b) => LoopState (a, b) where
  traverseLoopState function (left, right) = (,) <$> traverseLoopState function left <*> traverseLoopState function right
  flattenLoopState (left, right) = flattenLoopState left <> flattenLoopState right

data LoopSpec = LoopSpec
  { loopIterations :: !Int
  , loopInitialState :: ![SomeValue]
  , loopPlaceholderState :: ![SomeValue]
  , loopFinalState :: ![SomeValue]
  , loopResultTargets :: ![SomeValue]
  , loopCapturedIds :: ![Int]
  , loopBodyNodes :: ![ProgramNode]
  }

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
initialBuilderState = BuilderState {builderNextValueId = 0, builderInsideLoop = False, builderNodesRev = []}

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

sumRowsP :: (Typeable a, ArrayScalar a, NumericExp a) => Value A.Ix2 a -> ProgramBuilder (Value A.Ix1 a)
sumRowsP inputValue = do
  let A.Sz2 rows _ = valueSize inputValue
  outputValue <- freshValue (A.Sz1 rows)
  emitNode (SumRowsNode outputValue inputValue)
  pure outputValue

sumColsP :: (Typeable a, ArrayScalar a, NumericExp a) => Value A.Ix2 a -> ProgramBuilder (Value A.Ix1 a)
sumColsP inputValue = do
  let A.Sz2 _ cols = valueSize inputValue
  outputValue <- freshValue (A.Sz1 cols)
  emitNode (SumColsNode outputValue inputValue)
  pure outputValue

maxRowsP :: (Typeable a, ArrayScalar a, Comparable a) => Value A.Ix2 a -> ProgramBuilder (Value A.Ix1 a)
maxRowsP inputValue = do
  let A.Sz2 rows _ = valueSize inputValue
  outputValue <- freshValue (A.Sz1 rows)
  emitNode (MaxRowsNode outputValue inputValue)
  pure outputValue

maxColsP :: (Typeable a, ArrayScalar a, Comparable a) => Value A.Ix2 a -> ProgramBuilder (Value A.Ix1 a)
maxColsP inputValue = do
  let A.Sz2 _ cols = valueSize inputValue
  outputValue <- freshValue (A.Sz1 cols)
  emitNode (MaxColsNode outputValue inputValue)
  pure outputValue

broadcastRowsP :: (Typeable a, ArrayScalar a) => Int -> Value A.Ix1 a -> ProgramBuilder (Value A.Ix2 a)
broadcastRowsP cols inputValue = do
  when (cols < 0) (liftIO (throwArgumentError "broadcastRowsP" "column count must be non-negative"))
  let A.Sz1 rows = valueSize inputValue
  outputValue <- freshValue (A.Sz2 rows cols)
  emitNode (BroadcastRowsNode outputValue cols inputValue)
  pure outputValue

broadcastColsP :: (Typeable a, ArrayScalar a) => Int -> Value A.Ix1 a -> ProgramBuilder (Value A.Ix2 a)
broadcastColsP rows inputValue = do
  when (rows < 0) (liftIO (throwArgumentError "broadcastColsP" "row count must be non-negative"))
  let A.Sz1 cols = valueSize inputValue
  outputValue <- freshValue (A.Sz2 rows cols)
  emitNode (BroadcastColsNode outputValue rows inputValue)
  pure outputValue

softmaxRowsP :: forall a. (Typeable a, ArrayScalar a, FloatingExp a, Comparable a, Num a) => Value A.Ix2 a -> ProgramBuilder (Value A.Ix2 a)
softmaxRowsP inputValue = do
  let A.Sz2 _ cols = valueSize inputValue
      centerBinary :: Binary a a a
      centerBinary = binary (\x y -> x .+. (constant (-1) .*. y))
      scaleBinary :: Binary a a a
      scaleBinary = binary (\x y -> x .*. y)
  rowMax <- maxRowsP inputValue
  rowMaxMatrix <- broadcastRowsP cols rowMax
  centered <- zipWithExpr centerBinary inputValue rowMaxMatrix
  exponentiated <- mapExpr (unary expE) centered
  rowSums <- sumRowsP exponentiated
  inverseRowSums <- mapExpr (unary recipE) rowSums
  inverseRowSumsMatrix <- broadcastRowsP cols inverseRowSums
  zipWithExpr scaleBinary exponentiated inverseRowSumsMatrix

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

forLoopP :: LoopState s => Int -> s -> (s -> ProgramBuilder s) -> ProgramBuilder s
forLoopP iterations initialState bodyBuilder = do
  when (iterations < 0) (liftIO (throwArgumentError "forLoopP" "iteration count must be non-negative"))
  insideLoop <- ProgramBuilder (builderInsideLoop <$> get)
  when insideLoop (liftIO (throwArgumentError "forLoopP" "nested loops are not supported"))
  token <- freshLoopToken
  placeholderState <- traverseLoopState freshLike initialState
  outerState <- ProgramBuilder get
  (finalState, loopBuilderState) <-
    liftIO $
      runStateT
        (unProgramBuilder (bodyBuilder placeholderState))
        BuilderState
          { builderNextValueId = builderNextValueId outerState
          , builderInsideLoop = True
          , builderNodesRev = []
          }
  let initialValues = flattenLoopState initialState
      finalValues = flattenLoopState finalState
  when (not (loopStateShapesMatch initialValues finalValues)) (liftIO (throwArgumentError "forLoopP" "loop body must preserve state shape"))
  ProgramBuilder $ modify' $ \state -> state {builderNextValueId = builderNextValueId loopBuilderState}
  resultState <- traverseLoopState freshLike finalState
  let placeholderValues = flattenLoopState placeholderState
      resultValues = flattenLoopState resultState
      bodyNodes = reverse (builderNodesRev loopBuilderState)
      placeholderIds = Set.fromList (fmap someValueId placeholderValues)
      bodyOutputIds = Set.fromList (fmap programNodeOutputId bodyNodes)
      capturedIds =
        Set.toAscList
          ( Set.fromList (concatMap programNodeInputIds bodyNodes)
              `Set.difference` placeholderIds
              `Set.difference` bodyOutputIds
          )
      loopSpec =
        LoopSpec
          { loopIterations = iterations
          , loopInitialState = initialValues
          , loopPlaceholderState = placeholderValues
          , loopFinalState = finalValues
          , loopResultTargets = resultValues
          , loopCapturedIds = capturedIds
          , loopBodyNodes = bodyNodes
          }
  emitNode (LoopNode token loopSpec)
  forM_ (zip [0 ..] resultValues) $ \(resultIndex, resultValue) ->
    emitNode (LoopResultNode resultValue token resultIndex)
  pure resultState

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
freshValue size = do
  nextId <- freshBuilderId
  pure Value {valueId = nextId, valueSize = size}

freshLoopToken :: ProgramBuilder LoopToken
freshLoopToken = LoopToken <$> freshBuilderId

freshLike :: Value ix a -> ProgramBuilder (Value ix a)
freshLike value = freshValue (valueSize value)

freshBuilderId :: ProgramBuilder Int
freshBuilderId =
  ProgramBuilder $ do
    state <- get
    let nextId = builderNextValueId state
    modify' $ \currentState -> currentState {builderNextValueId = nextId + 1}
    pure nextId

someValueId :: SomeValue -> Int
someValueId (SomeValue value) = valueId value

loopStateShapesMatch :: [SomeValue] -> [SomeValue] -> Bool
loopStateShapesMatch leftValues rightValues =
  length leftValues == length rightValues
    && and (zipWith sameLoopShape leftValues rightValues)

sameLoopShape :: SomeValue -> SomeValue -> Bool
sameLoopShape (SomeValue (leftValue :: Value ix a)) (SomeValue rightValue) =
  case cast rightValue of
    Just (castValue :: Value ix a) -> valueSize leftValue == valueSize castValue
    Nothing -> False

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
executeReferenceProgram = executeReferenceProgramFromStore Map.empty

executeReferenceProgramFromStore :: HasCallStack => ReferenceStore -> [ProgramNode] -> IO ReferenceStore
executeReferenceProgramFromStore initialStore = foldM executeReferenceNode initialStore

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
    SumRowsNode outputValue inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- sumRowsRef hostInput
      pure (storeReferenceValue store outputValue hostArray)
    SumColsNode outputValue inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- sumColsRef hostInput
      pure (storeReferenceValue store outputValue hostArray)
    MaxRowsNode outputValue inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- maxRowsRef hostInput
      pure (storeReferenceValue store outputValue hostArray)
    MaxColsNode outputValue inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- maxColsRef hostInput
      pure (storeReferenceValue store outputValue hostArray)
    BroadcastRowsNode outputValue cols inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- broadcastRowsRef cols hostInput
      pure (storeReferenceValue store outputValue hostArray)
    BroadcastColsNode outputValue rows inputValue -> do
      hostInput <- lookupStoredReferenceValue "runProgramCpu" store inputValue
      hostArray <- broadcastColsRef rows hostInput
      pure (storeReferenceValue store outputValue hostArray)
    LoopNode _ loopSpec -> do
      loopResults <- executeReferenceLoop store loopSpec
      pure (storeLoopResults store loopResults)
    LoopResultNode _ _ _ ->
      pure store
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

executeReferenceLoop :: HasCallStack => ReferenceStore -> LoopSpec -> IO [(SomeValue, Dynamic.Dynamic)]
executeReferenceLoop outerStore loopSpec = do
  initialDynamics <- mapM (lookupStoredDynamic "runProgramCpu" outerStore) (loopInitialState loopSpec)
  finalDynamics <- go initialDynamics
  pure (zip (loopResultTargets loopSpec) finalDynamics)
  where
    capturedStore = Map.restrictKeys outerStore (Set.fromList (loopCapturedIds loopSpec))

    go :: [Dynamic.Dynamic] -> IO [Dynamic.Dynamic]
    go currentStateDynamics =
      if loopIterations loopSpec <= 0
        then pure currentStateDynamics
        else iterateLoop (loopIterations loopSpec) currentStateDynamics

    iterateLoop :: Int -> [Dynamic.Dynamic] -> IO [Dynamic.Dynamic]
    iterateLoop remaining currentStateDynamics = do
      let iterationStore = Map.union (bindLoopDynamics (loopPlaceholderState loopSpec) currentStateDynamics) capturedStore
      bodyStore <- executeReferenceProgramFromStore iterationStore (loopBodyNodes loopSpec)
      nextStateDynamics <- mapM (lookupStoredDynamic "runProgramCpu" bodyStore) (loopFinalState loopSpec)
      if remaining == 1
        then pure nextStateDynamics
        else iterateLoop (remaining - 1) nextStateDynamics

executeScheduledProgram :: HasCallStack => ProgramRuntime -> [ScheduledNode] -> [ProgramNode] -> IO ValueStore
executeScheduledProgram runtime scheduledNodes nodes =
  executeScheduledProgramFromStore runtime Map.empty scheduledNodes nodes

executeScheduledProgramFromStore :: HasCallStack => ProgramRuntime -> ValueStore -> [ScheduledNode] -> [ProgramNode] -> IO ValueStore
executeScheduledProgramFromStore runtime initialStore scheduledNodes nodes = do
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
  (store, eventMap, _) <- foldM executeOne (initialStore, Map.empty, Map.empty) scheduledNodes
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
    SumRowsNode outputValue inputValue -> do
      inputArray <- lookupStoredValue "SumRowsNode" store inputValue
      deviceArray <- sumRowsArrayOn stream (programRuntimeArrayRuntime runtime) inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    SumColsNode outputValue inputValue -> do
      inputArray <- lookupStoredValue "SumColsNode" store inputValue
      deviceArray <- sumColsArrayOn stream (programRuntimeArrayRuntime runtime) inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    MaxRowsNode outputValue inputValue -> do
      inputArray <- lookupStoredValue "MaxRowsNode" store inputValue
      deviceArray <- maxRowsArrayOn stream (programRuntimeArrayRuntime runtime) inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    MaxColsNode outputValue inputValue -> do
      inputArray <- lookupStoredValue "MaxColsNode" store inputValue
      deviceArray <- maxColsArrayOn stream (programRuntimeArrayRuntime runtime) inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    BroadcastRowsNode outputValue cols inputValue -> do
      inputArray <- lookupStoredValue "BroadcastRowsNode" store inputValue
      deviceArray <- broadcastRowsArrayOn stream (programRuntimeArrayRuntime runtime) cols inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    BroadcastColsNode outputValue rows inputValue -> do
      inputArray <- lookupStoredValue "BroadcastColsNode" store inputValue
      deviceArray <- broadcastColsArrayOn stream (programRuntimeArrayRuntime runtime) rows inputArray
      event <- recordNodeEvent stream
      pure (storeValue store outputValue deviceArray, Just event)
    LoopNode _ loopSpec -> do
      loopResults <- executeScheduledLoop runtime store loopSpec
      event <- recordNodeEvent (contextDefaultStream (programRuntimeContext runtime))
      pure (storeLoopResults store loopResults, Just event)
    LoopResultNode _ _ _ ->
      pure (store, Nothing)
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

executeScheduledLoop :: HasCallStack => ProgramRuntime -> ValueStore -> LoopSpec -> IO [(SomeValue, Dynamic.Dynamic)]
executeScheduledLoop runtime outerStore loopSpec = do
  initialDynamics <- mapM (lookupStoredDynamic "LoopNode" outerStore) (loopInitialState loopSpec)
  scheduledBodyNodes <- scheduleLoopBody runtime (loopBodyNodes loopSpec)
  finalDynamics <- go scheduledBodyNodes initialDynamics
  pure (zip (loopResultTargets loopSpec) finalDynamics)
  where
    capturedStore = Map.restrictKeys outerStore (Set.fromList (loopCapturedIds loopSpec))

    go :: [ScheduledNode] -> [Dynamic.Dynamic] -> IO [Dynamic.Dynamic]
    go scheduledBodyNodes currentStateDynamics =
      if loopIterations loopSpec <= 0
        then pure currentStateDynamics
        else iterateLoop scheduledBodyNodes (loopIterations loopSpec) currentStateDynamics

    iterateLoop :: [ScheduledNode] -> Int -> [Dynamic.Dynamic] -> IO [Dynamic.Dynamic]
    iterateLoop scheduledBodyNodes remaining currentStateDynamics = do
      let iterationStore = Map.union (bindLoopDynamics (loopPlaceholderState loopSpec) currentStateDynamics) capturedStore
      bodyStore <- executeScheduledProgramFromStore runtime iterationStore scheduledBodyNodes (loopBodyNodes loopSpec)
      nextStateDynamics <- mapM (lookupStoredDynamic "LoopNode" bodyStore) (loopFinalState loopSpec)
      if remaining == 1
        then pure nextStateDynamics
        else iterateLoop scheduledBodyNodes (remaining - 1) nextStateDynamics

scheduleLoopBody :: HasCallStack => ProgramRuntime -> [ProgramNode] -> IO [ScheduledNode]
scheduleLoopBody runtime bodyNodes =
  scheduleNodes (length (programRuntimeStreams runtime)) schedulerNodes
  where
    bodyNodeIds = Set.fromList (fmap programNodeOutputId bodyNodes)
    toLoopSchedulerNode :: ProgramNode -> SchedulerNode
    toLoopSchedulerNode node =
      let schedulerNode = toSchedulerNode node
       in schedulerNode {schedulerNodeDependencies = filter (`Set.member` bodyNodeIds) (schedulerNodeDependencies schedulerNode)}
    schedulerNodes = fmap toLoopSchedulerNode bodyNodes

bindLoopDynamics :: [SomeValue] -> [Dynamic.Dynamic] -> Map Int Dynamic.Dynamic
bindLoopDynamics placeholders dynamics =
  Map.fromList (zipWith (\placeholder dynamicValue -> (someValueId placeholder, dynamicValue)) placeholders dynamics)

lookupStoredDynamic :: HasCallStack => String -> Map Int Dynamic.Dynamic -> SomeValue -> IO Dynamic.Dynamic
lookupStoredDynamic functionName store someValue =
  case Map.lookup (someValueId someValue) store of
    Just dynamicValue -> pure dynamicValue
    Nothing -> throwArgumentError functionName ("missing loop value: " <> show (someValueId someValue))

storeLoopResults :: Map Int Dynamic.Dynamic -> [(SomeValue, Dynamic.Dynamic)] -> Map Int Dynamic.Dynamic
storeLoopResults = foldl storeOne
  where
    storeOne currentStore (someValue, dynamicValue) =
      Map.insert (someValueId someValue) dynamicValue currentStore

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
    SumRowsNode outputValue _ -> valueId outputValue
    SumColsNode outputValue _ -> valueId outputValue
    MaxRowsNode outputValue _ -> valueId outputValue
    MaxColsNode outputValue _ -> valueId outputValue
    BroadcastRowsNode outputValue _ _ -> valueId outputValue
    BroadcastColsNode outputValue _ _ -> valueId outputValue
    AxpyNode outputValue _ _ _ -> valueId outputValue
    GemmNode outputValue _ -> valueId outputValue
    FftForwardNode outputValue _ -> valueId outputValue
    FftInverseNode outputValue _ -> valueId outputValue
    RandUniformNode outputValue _ -> valueId outputValue
    RandNormalNode outputValue _ _ _ -> valueId outputValue
    LoopNode loopToken _ -> loopTokenId loopToken
    LoopResultNode (SomeValue outputValue) _ _ -> valueId outputValue

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
    SumRowsNode _ inputValue -> [valueId inputValue]
    SumColsNode _ inputValue -> [valueId inputValue]
    MaxRowsNode _ inputValue -> [valueId inputValue]
    MaxColsNode _ inputValue -> [valueId inputValue]
    BroadcastRowsNode _ _ inputValue -> [valueId inputValue]
    BroadcastColsNode _ _ inputValue -> [valueId inputValue]
    AxpyNode _ _ xValue yValue -> [valueId xValue, valueId yValue]
    GemmNode _ matrixSpec -> [valueId (matrixGemmValueA matrixSpec), valueId (matrixGemmValueB matrixSpec), valueId (matrixGemmValueC matrixSpec)]
    FftForwardNode _ inputValue -> [valueId inputValue]
    FftInverseNode _ inputValue -> [valueId inputValue]
    RandUniformNode _ _ -> []
    RandNormalNode _ _ _ _ -> []
    LoopNode _ loopSpec -> fmap someValueId (loopInitialState loopSpec) <> loopCapturedIds loopSpec
    LoopResultNode _ loopToken _ -> [loopTokenId loopToken]

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
          SumRowsNode _ _ -> SchedulerResourceJit
          SumColsNode _ _ -> SchedulerResourceJit
          MaxRowsNode _ _ -> SchedulerResourceJit
          MaxColsNode _ _ -> SchedulerResourceJit
          BroadcastRowsNode _ _ _ -> SchedulerResourceJit
          BroadcastColsNode _ _ _ -> SchedulerResourceJit
          AxpyNode _ _ _ _ -> SchedulerResourceBlas
          GemmNode _ _ -> SchedulerResourceBlas
          FftForwardNode _ _ -> SchedulerResourceFft
          FftInverseNode _ _ -> SchedulerResourceFft
          RandUniformNode _ _ -> SchedulerResourceRand
          RandNormalNode _ _ _ _ -> SchedulerResourceRand
          LoopNode _ _ -> SchedulerResourceRand
          LoopResultNode _ _ _ -> SchedulerResourceShape
    }
