module Molten.Internal.Scheduler
  ( ScheduledNode(..)
  , SchedulerNode(..)
  , SchedulerResource(..)
  , scheduleNodes
  ) where

import Data.List (insertBy, sortOn)
import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import GHC.Stack (HasCallStack)
import ROCm.FFI.Core.Exception (throwArgumentError)

data SchedulerResource
  = SchedulerResourceInput
  | SchedulerResourceShape
  | SchedulerResourceJit
  | SchedulerResourceBlas
  | SchedulerResourceFft
  | SchedulerResourceRand
  deriving (Eq, Ord, Show)

data SchedulerNode = SchedulerNode
  { schedulerNodeId :: !Int
  , schedulerNodeDependencies :: ![Int]
  , schedulerNodeResource :: !SchedulerResource
  }
  deriving (Eq, Show)

data ScheduledNode = ScheduledNode
  { scheduledNodeId :: !Int
  , scheduledNodeDependencies :: ![Int]
  , scheduledNodeResource :: !SchedulerResource
  , scheduledNodeStreamSlot :: !Int
  }
  deriving (Eq, Show)

scheduleNodes :: HasCallStack => Int -> [SchedulerNode] -> IO [ScheduledNode]
scheduleNodes streamCount nodes
  | streamCount <= 0 = throwArgumentError "scheduleNodes" "streamCount must be > 0"
  | otherwise = go initialReady indegreeMap 0 []
  where
    nodeMap = Map.fromList [(schedulerNodeId node, node) | node <- nodes]
    sortedNodes = sortOn schedulerNodeId nodes
    indegreeMap = Map.fromList [(schedulerNodeId node, length (schedulerNodeDependencies node)) | node <- sortedNodes]
    reverseEdges =
      Map.fromListWith (++)
        [ (dependencyId, [schedulerNodeId node])
        | node <- sortedNodes
        , dependencyId <- schedulerNodeDependencies node
        ]
    initialReady = [schedulerNodeId node | node <- sortedNodes, null (schedulerNodeDependencies node)]

    go :: [Int] -> Map Int Int -> Int -> [ScheduledNode] -> IO [ScheduledNode]
    go [] remainingIndegree nextSharedSlot scheduled
      | length scheduled == length nodes = pure (reverse scheduled)
      | otherwise = throwArgumentError "scheduleNodes" "dependency graph contains a cycle or unknown dependency"
    go (nodeId : readyTail) remainingIndegree nextSharedSlot scheduled =
      case Map.lookup nodeId nodeMap of
        Nothing -> throwArgumentError "scheduleNodes" ("unknown node id: " <> show nodeId)
        Just node -> do
          let (streamSlot, nextSharedSlot') = chooseStreamSlot streamCount nextSharedSlot node
              scheduledNode =
                ScheduledNode
                  { scheduledNodeId = nodeId
                  , scheduledNodeDependencies = schedulerNodeDependencies node
                  , scheduledNodeResource = schedulerNodeResource node
                  , scheduledNodeStreamSlot = streamSlot
                  }
              dependents = Map.findWithDefault [] nodeId reverseEdges
              (remainingIndegree', readyTail') = releaseDependents dependents remainingIndegree readyTail
          go readyTail' remainingIndegree' nextSharedSlot' (scheduledNode : scheduled)

chooseStreamSlot :: Int -> Int -> SchedulerNode -> (Int, Int)
chooseStreamSlot streamCount nextSharedSlot node =
  case schedulerNodeResource node of
    SchedulerResourceJit ->
      let slot = nextSharedSlot `mod` streamCount
       in (slot, nextSharedSlot + 1)
    SchedulerResourceInput -> (0, nextSharedSlot)
    SchedulerResourceShape -> (0, nextSharedSlot)
    SchedulerResourceBlas -> (0, nextSharedSlot)
    SchedulerResourceFft -> (0, nextSharedSlot)
    SchedulerResourceRand -> (0, nextSharedSlot)

releaseDependents :: [Int] -> Map Int Int -> [Int] -> (Map Int Int, [Int])
releaseDependents dependentIds indegreeMap ready =
  foldl releaseOne (indegreeMap, ready) dependentIds
  where
    releaseOne :: (Map Int Int, [Int]) -> Int -> (Map Int Int, [Int])
    releaseOne (currentIndegree, currentReady) dependentId =
      case Map.lookup dependentId currentIndegree of
        Nothing -> (currentIndegree, currentReady)
        Just indegree ->
          let indegree' = indegree - 1
              currentIndegree' = Map.insert dependentId indegree' currentIndegree
              currentReady' =
                if indegree' == 0
                  then insertReady dependentId currentReady
                  else currentReady
           in (currentIndegree', currentReady')

insertReady :: Int -> [Int] -> [Int]
insertReady nodeId = insertBy compare nodeId
