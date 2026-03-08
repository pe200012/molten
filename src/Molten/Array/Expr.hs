{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Molten.Array.Expr
  ( ArrayScalar(..)
  , Binary(..)
  , Castable(..)
  , Comparable(..)
  , Exp
  , FloatingExp(..)
  , FractionalExp(..)
  , NumericExp(..)
  , Unary(..)
  , cast
  , constant
  , evaluateBinaryExpression
  , evaluateUnaryExpression
  , expE
  , recipE
  , renderBinaryExpression
  , renderExpression
  , renderUnaryExpression
  , select
  , unary
  , binary
  , (.+.)
  , (.*.)
  , (.<.)
  ) where

import Data.Complex (Complex((:+)))
import Data.Int (Int32, Int64)
import Data.Proxy (Proxy(..))
import Data.Word (Word32)
import Foreign.Storable (Storable)
import Numeric (showFFloat)

class Storable a => ArrayScalar a where
  arrayScalarTypeName :: proxy a -> String
  arrayScalarCType :: proxy a -> String
  renderScalarLiteral :: a -> String

class ArrayScalar a => NumericExp a where
  addScalar :: a -> a -> a
  mulScalar :: a -> a -> a

class NumericExp a => FractionalExp a where
  recipScalar :: a -> a

class FractionalExp a => FloatingExp a where
  expScalar :: a -> a

class ArrayScalar a => Comparable a where
  lessThanScalar :: a -> a -> Bool

class (ArrayScalar a, ArrayScalar b) => Castable a b where
  castScalar :: a -> b

infixl 6 .+.
infixl 7 .*.
infix 4 .<.

instance ArrayScalar Float where
  arrayScalarTypeName _ = "Float"
  arrayScalarCType _ = "float"
  renderScalarLiteral value = renderFloatLiteral value <> "f"

instance ArrayScalar Double where
  arrayScalarTypeName _ = "Double"
  arrayScalarCType _ = "double"
  renderScalarLiteral = renderFloatLiteral

instance ArrayScalar Int32 where
  arrayScalarTypeName _ = "Int32"
  arrayScalarCType _ = "int"
  renderScalarLiteral = show

instance ArrayScalar Int64 where
  arrayScalarTypeName _ = "Int64"
  arrayScalarCType _ = "long long"
  renderScalarLiteral value = show value <> "ll"

instance ArrayScalar Word32 where
  arrayScalarTypeName _ = "Word32"
  arrayScalarCType _ = "unsigned int"
  renderScalarLiteral value = show value <> "u"

instance ArrayScalar Bool where
  arrayScalarTypeName _ = "Bool"
  arrayScalarCType _ = "bool"
  renderScalarLiteral value =
    if value
      then "true"
      else "false"

instance ArrayScalar (Complex Float) where
  arrayScalarTypeName _ = "ComplexFloat"
  arrayScalarCType _ = "float2"
  renderScalarLiteral (realPart :+ imagPart) =
    "make_float2(" <> renderFloatLiteral realPart <> "f, " <> renderFloatLiteral imagPart <> "f)"

instance ArrayScalar (Complex Double) where
  arrayScalarTypeName _ = "ComplexDouble"
  arrayScalarCType _ = "double2"
  renderScalarLiteral (realPart :+ imagPart) =
    "make_double2(" <> renderFloatLiteral realPart <> ", " <> renderFloatLiteral imagPart <> ")"

instance NumericExp Float where
  addScalar = (+)
  mulScalar = (*)

instance FractionalExp Float where
  recipScalar = recip

instance FloatingExp Float where
  expScalar = exp

instance NumericExp Double where
  addScalar = (+)
  mulScalar = (*)

instance FractionalExp Double where
  recipScalar = recip

instance FloatingExp Double where
  expScalar = exp

instance NumericExp Int32 where
  addScalar = (+)
  mulScalar = (*)

instance NumericExp Int64 where
  addScalar = (+)
  mulScalar = (*)

instance NumericExp Word32 where
  addScalar = (+)
  mulScalar = (*)

instance NumericExp (Complex Float) where
  addScalar = (+)
  mulScalar = (*)

instance NumericExp (Complex Double) where
  addScalar = (+)
  mulScalar = (*)

instance Comparable Float where
  lessThanScalar = (<)

instance Comparable Double where
  lessThanScalar = (<)

instance Comparable Int32 where
  lessThanScalar = (<)

instance Comparable Int64 where
  lessThanScalar = (<)

instance Comparable Word32 where
  lessThanScalar = (<)

instance Comparable Bool where
  lessThanScalar = (<)

instance Castable Float Float where
  castScalar = id

instance Castable Double Double where
  castScalar = id

instance Castable Int32 Int32 where
  castScalar = id

instance Castable Int64 Int64 where
  castScalar = id

instance Castable Word32 Word32 where
  castScalar = id

instance Castable Bool Bool where
  castScalar = id

instance Castable (Complex Float) (Complex Float) where
  castScalar = id

instance Castable (Complex Double) (Complex Double) where
  castScalar = id

instance Castable Float Double where
  castScalar = realToFrac

instance Castable Double Float where
  castScalar = realToFrac

instance Castable Int32 Float where
  castScalar = fromIntegral

instance Castable Int32 Double where
  castScalar = fromIntegral

instance Castable Int32 Int64 where
  castScalar = fromIntegral

instance Castable Int32 Word32 where
  castScalar = fromIntegral

instance Castable Int64 Float where
  castScalar = fromIntegral

instance Castable Int64 Double where
  castScalar = fromIntegral

instance Castable Int64 Int32 where
  castScalar = fromIntegral

instance Castable Int64 Word32 where
  castScalar = fromIntegral

instance Castable Word32 Float where
  castScalar = fromIntegral

instance Castable Word32 Double where
  castScalar = fromIntegral

instance Castable Word32 Int32 where
  castScalar = fromIntegral

instance Castable Word32 Int64 where
  castScalar = fromIntegral

instance Castable Float Int32 where
  castScalar = truncate

instance Castable Float Int64 where
  castScalar = truncate

instance Castable Float Word32 where
  castScalar = truncate

instance Castable Double Int32 where
  castScalar = truncate

instance Castable Double Int64 where
  castScalar = truncate

instance Castable Double Word32 where
  castScalar = truncate

instance Castable (Complex Float) (Complex Double) where
  castScalar (realPart :+ imagPart) = realToFrac realPart :+ realToFrac imagPart

instance Castable (Complex Double) (Complex Float) where
  castScalar (realPart :+ imagPart) = realToFrac realPart :+ realToFrac imagPart

data Exp a where
  VarExp :: ArrayScalar a => String -> Exp a
  ConstantExp :: ArrayScalar a => a -> Exp a
  CastExp :: Castable a b => Exp a -> Exp b
  SelectExp :: ArrayScalar a => Exp Bool -> Exp a -> Exp a -> Exp a
  AddExp :: NumericExp a => Exp a -> Exp a -> Exp a
  MulExp :: NumericExp a => Exp a -> Exp a -> Exp a
  RecipExp :: FractionalExp a => Exp a -> Exp a
  ExpExp :: FloatingExp a => Exp a -> Exp a
  LessThanExp :: Comparable a => Exp a -> Exp a -> Exp Bool

newtype Unary a b = Unary {runUnary :: Exp a -> Exp b}

newtype Binary a b c = Binary {runBinary :: Exp a -> Exp b -> Exp c}

constant :: ArrayScalar a => a -> Exp a
constant = ConstantExp

cast :: Castable a b => Exp a -> Exp b
cast = CastExp

select :: ArrayScalar a => Exp Bool -> Exp a -> Exp a -> Exp a
select = SelectExp

unary :: (Exp a -> Exp b) -> Unary a b
unary = Unary

binary :: (Exp a -> Exp b -> Exp c) -> Binary a b c
binary = Binary

(.+.) :: NumericExp a => Exp a -> Exp a -> Exp a
(.+.) = AddExp

(.*.) :: NumericExp a => Exp a -> Exp a -> Exp a
(.*.) = MulExp

(.<.) :: Comparable a => Exp a -> Exp a -> Exp Bool
(.<.) = LessThanExp

recipE :: FractionalExp a => Exp a -> Exp a
recipE = RecipExp

expE :: FloatingExp a => Exp a -> Exp a
expE = ExpExp

evaluateUnaryExpression :: ArrayScalar a => Unary a b -> a -> b
evaluateUnaryExpression (Unary function) value =
  evaluateExpression (function (ConstantExp value))

evaluateBinaryExpression :: (ArrayScalar a, ArrayScalar b) => Binary a b c -> a -> b -> c
evaluateBinaryExpression (Binary function) leftValue rightValue =
  evaluateExpression (function (ConstantExp leftValue) (ConstantExp rightValue))

evaluateExpression :: Exp a -> a
evaluateExpression expression =
  case expression of
    VarExp name -> error ("evaluateExpression: free variable " <> name)
    ConstantExp value -> value
    CastExp inner -> castScalar (evaluateExpression inner)
    SelectExp predicate ifTrue ifFalse ->
      if evaluateExpression predicate
        then evaluateExpression ifTrue
        else evaluateExpression ifFalse
    AddExp left right -> addScalar (evaluateExpression left) (evaluateExpression right)
    MulExp left right -> mulScalar (evaluateExpression left) (evaluateExpression right)
    RecipExp inner -> recipScalar (evaluateExpression inner)
    ExpExp inner -> expScalar (evaluateExpression inner)
    LessThanExp left right -> lessThanScalar (evaluateExpression left) (evaluateExpression right)

renderExpression :: Exp a -> String
renderExpression expression =
  case expression of
    VarExp name -> name
    ConstantExp value -> renderScalarLiteral value
    CastExp inner -> "((" <> expressionTypeCName expression <> ")(" <> renderExpression inner <> "))"
    SelectExp predicate ifTrue ifFalse ->
      "(" <> renderExpression predicate <> " ? " <> wrapRenderedExpression ifTrue <> " : " <> wrapRenderedExpression ifFalse <> ")"
    AddExp left right -> renderAddExpression left right
    MulExp left right -> renderMulExpression left right
    RecipExp inner -> renderRecipExpression inner
    ExpExp inner -> renderExpExpression inner
    LessThanExp left right -> binaryOperator "<" left right

renderUnaryExpression :: ArrayScalar a => Unary a b -> String
renderUnaryExpression (Unary function) = renderExpression (function (VarExp "x0"))

renderBinaryExpression :: (ArrayScalar a, ArrayScalar b) => Binary a b c -> String
renderBinaryExpression (Binary function) = renderExpression (function (VarExp "x0") (VarExp "x1"))

wrapRenderedExpression :: Exp a -> String
wrapRenderedExpression expression =
  let rendered = renderExpression expression
   in case rendered of
        '(' : _ | last rendered == ')' -> rendered
        _ -> "(" <> rendered <> ")"

expressionTypeCName :: forall a. Exp a -> String
expressionTypeCName expression =
  case expression of
    VarExp _ -> arrayScalarCType (Proxy :: Proxy a)
    ConstantExp _ -> arrayScalarCType (Proxy :: Proxy a)
    CastExp _ -> arrayScalarCType (Proxy :: Proxy a)
    SelectExp _ _ _ -> arrayScalarCType (Proxy :: Proxy a)
    AddExp _ _ -> arrayScalarCType (Proxy :: Proxy a)
    MulExp _ _ -> arrayScalarCType (Proxy :: Proxy a)
    RecipExp _ -> arrayScalarCType (Proxy :: Proxy a)
    ExpExp _ -> arrayScalarCType (Proxy :: Proxy a)
    LessThanExp _ _ -> arrayScalarCType (Proxy :: Proxy a)

binaryOperator :: String -> Exp a -> Exp a -> String
binaryOperator symbol left right =
  "((" <> renderExpression left <> ") " <> symbol <> " (" <> renderExpression right <> "))"

renderAddExpression :: forall a. NumericExp a => Exp a -> Exp a -> String
renderAddExpression left right =
  case arrayScalarTypeName (Proxy :: Proxy a) of
    "ComplexFloat" -> renderFunctionCall "molten_add_float2" left right
    "ComplexDouble" -> renderFunctionCall "molten_add_double2" left right
    _ -> binaryOperator "+" left right

renderMulExpression :: forall a. NumericExp a => Exp a -> Exp a -> String
renderMulExpression left right =
  case arrayScalarTypeName (Proxy :: Proxy a) of
    "ComplexFloat" -> renderFunctionCall "molten_mul_float2" left right
    "ComplexDouble" -> renderFunctionCall "molten_mul_double2" left right
    _ -> binaryOperator "*" left right

renderRecipExpression :: forall a. FractionalExp a => Exp a -> String
renderRecipExpression inner =
  "((" <> renderUnitLiteral (Proxy :: Proxy a) <> ") / (" <> renderExpression inner <> "))"

renderExpExpression :: forall a. FloatingExp a => Exp a -> String
renderExpExpression inner =
  case arrayScalarTypeName (Proxy :: Proxy a) of
    "Float" -> "expf(" <> renderExpression inner <> ")"
    "Double" -> "exp(" <> renderExpression inner <> ")"
    unsupportedType -> error ("renderExpExpression: unsupported scalar type " <> unsupportedType)

renderFunctionCall :: String -> Exp a -> Exp a -> String
renderFunctionCall functionName left right =
  functionName <> "(" <> renderExpression left <> ", " <> renderExpression right <> ")"

renderUnitLiteral :: forall a proxy. ArrayScalar a => proxy a -> String
renderUnitLiteral _ =
  case arrayScalarTypeName (Proxy :: Proxy a) of
    "Float" -> renderScalarLiteral (1.0 :: Float)
    "Double" -> renderScalarLiteral (1.0 :: Double)
    unsupportedType -> error ("renderUnitLiteral: unsupported scalar type " <> unsupportedType)

renderFloatLiteral :: RealFloat a => a -> String
renderFloatLiteral value =
  case showFFloat Nothing value "" of
    literal
      | '.' `elem` literal || 'e' `elem` literal || 'E' `elem` literal -> literal
      | otherwise -> literal <> ".0"
