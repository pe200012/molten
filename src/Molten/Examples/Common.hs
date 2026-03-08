module Molten.Examples.Common
  ( Timed(..)
  , approxEqAbsolute
  , approxEqRelative
  , failWhen
  , measureOnce
  , measureRepeated
  , parseFlagMap
  ) where

import qualified Data.Map.Strict as Map
import Data.Map.Strict (Map)
import Data.Word (Word64)
import GHC.Clock (getMonotonicTimeNSec)

data Timed a = Timed
  { timedValue :: !a
  , timedSeconds :: !Double
  }
  deriving (Eq, Show)

measureOnce :: IO a -> IO (Timed a)
measureOnce action = do
  startNs <- getMonotonicTimeNSec
  value <- action
  endNs <- getMonotonicTimeNSec
  pure Timed {timedValue = value, timedSeconds = elapsedSeconds startNs endNs}

measureRepeated :: Int -> IO a -> IO [Timed a]
measureRepeated n action
  | n <= 0 = pure []
  | otherwise = sequence (replicate n (measureOnce action))

approxEqAbsolute :: (Ord a, Num a) => a -> a -> a -> Bool
approxEqAbsolute tolerance left right = abs (left - right) <= tolerance

approxEqRelative :: (Ord a, Fractional a) => a -> a -> a -> Bool
approxEqRelative tolerance left right =
  let scale = max 1 (max (abs left) (abs right))
   in abs (left - right) <= tolerance * scale

parseFlagMap :: [String] -> Either String (Map String String)
parseFlagMap = go Map.empty
  where
    go :: Map String String -> [String] -> Either String (Map String String)
    go acc [] = Right acc
    go _ [dangling] = Left ("missing value for flag " <> dangling)
    go acc (flag:value:rest)
      | Just name <- stripFlag flag = go (Map.insert name value acc) rest
      | otherwise = Left ("expected --flag, got " <> flag)

failWhen :: String -> Bool -> IO ()
failWhen message predicate =
  if predicate
    then fail message
    else pure ()

stripFlag :: String -> Maybe String
stripFlag ('-':'-':name)
  | not (null name) = Just name
stripFlag _ = Nothing

elapsedSeconds :: Word64 -> Word64 -> Double
elapsedSeconds startNs endNs = fromIntegral (endNs - startNs) / 1.0e9
