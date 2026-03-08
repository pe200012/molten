module Molten.Internal.Reference.Array
  ( evalBinaryRef
  , evalUnaryRef
  , fillArrayRef
  , mapArrayRef
  , reduceAllArrayRef
  , reshapeArrayRef
  , zipWithArrayRef
  ) where

import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS
import Foreign.Storable (Storable)
import GHC.Stack (HasCallStack)
import Molten.Array.Expr
  ( ArrayScalar
  , Binary
  , NumericExp
  , Unary
  , evaluateBinaryExpression
  , evaluateUnaryExpression
  )
import ROCm.FFI.Core.Exception (throwArgumentError)

evalUnaryRef :: (ArrayScalar a, ArrayScalar b) => Unary a b -> a -> b
evalUnaryRef = evaluateUnaryExpression

evalBinaryRef :: (ArrayScalar a, ArrayScalar b, ArrayScalar c) => Binary a b c -> a -> b -> c
evalBinaryRef = evaluateBinaryExpression

fillArrayRef :: (ArrayScalar a, A.Index ix) => a -> A.Sz ix -> IO (A.Array A.S ix a)
fillArrayRef value size =
  pure (AMV.fromVector' A.Seq size (VS.replicate (A.totalElem size) value))

mapArrayRef :: (ArrayScalar a, ArrayScalar b, A.Index ix) => Unary a b -> A.Array A.S ix a -> IO (A.Array A.S ix b)
mapArrayRef unary input =
  pure
    ( AMV.fromVector' A.Seq (A.size input)
        (VS.map (evalUnaryRef unary) (AM.toStorableVector input))
    )

zipWithArrayRef :: (HasCallStack, ArrayScalar a, ArrayScalar b, ArrayScalar c, A.Index ix) => Binary a b c -> A.Array A.S ix a -> A.Array A.S ix b -> IO (A.Array A.S ix c)
zipWithArrayRef binary left right =
  if A.size left == A.size right
    then
      pure
        ( AMV.fromVector' A.Seq (A.size left)
            (VS.zipWith (evalBinaryRef binary) (AM.toStorableVector left) (AM.toStorableVector right))
        )
    else throwArgumentError "zipWithArrayRef" "input arrays must have the same shape"

reduceAllArrayRef :: (ArrayScalar a, NumericExp a, A.Index ix) => Binary a a a -> a -> A.Array A.S ix a -> IO (A.Array A.S A.Ix1 a)
reduceAllArrayRef binary initialValue input =
  pure
    ( AMV.fromVector' A.Seq (A.Sz1 1)
        (VS.singleton (VS.foldl' (evalBinaryRef binary) initialValue (AM.toStorableVector input)))
    )

reshapeArrayRef :: (HasCallStack, A.Index ix, A.Index ix', Storable a) => A.Sz ix' -> A.Array A.S ix a -> IO (A.Array A.S ix' a)
reshapeArrayRef targetSize input =
  if A.totalElem targetSize == A.totalElem (A.size input)
    then pure (AMV.fromVector' A.Seq targetSize (AM.toStorableVector input))
    else throwArgumentError "reshapeArrayRef" "reshape must preserve totalElem"
