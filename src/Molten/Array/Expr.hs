{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE ScopedTypeVariables #-}

module Molten.Array.Expr
  ( ArrayScalar(..)
  , Binary(..)
  , Castable
  , Comparable
  , Exp
  , NumericExp
  , Unary(..)
  , cast
  , constant
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

class ArrayScalar a => NumericExp a

class ArrayScalar a => Comparable a

class (ArrayScalar a, ArrayScalar b) => Castable a b

instance (ArrayScalar a, ArrayScalar b) => Castable a b

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

instance NumericExp Float
instance NumericExp Double
instance NumericExp Int32
instance NumericExp Int64
instance NumericExp Word32

instance Comparable Float
instance Comparable Double
instance Comparable Int32
instance Comparable Int64
instance Comparable Word32
instance Comparable Bool

data Exp a where
  VarExp :: ArrayScalar a => String -> Exp a
  ConstantExp :: ArrayScalar a => a -> Exp a
  CastExp :: (ArrayScalar a, ArrayScalar b, Castable a b) => Exp a -> Exp b
  SelectExp :: ArrayScalar a => Exp Bool -> Exp a -> Exp a -> Exp a
  AddExp :: NumericExp a => Exp a -> Exp a -> Exp a
  MulExp :: NumericExp a => Exp a -> Exp a -> Exp a
  LessThanExp :: Comparable a => Exp a -> Exp a -> Exp Bool

newtype Unary a b = Unary { runUnary :: Exp a -> Exp b }

newtype Binary a b c = Binary { runBinary :: Exp a -> Exp b -> Exp c }

constant :: ArrayScalar a => a -> Exp a
constant = ConstantExp

cast :: (ArrayScalar a, ArrayScalar b, Castable a b) => Exp a -> Exp b
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

renderExpression :: Exp a -> String
renderExpression expression =
  case expression of
    VarExp name -> name
    ConstantExp value -> renderScalarLiteral value
    CastExp inner -> "((" <> expressionTypeCName expression <> ")(" <> renderExpression inner <> "))"
    SelectExp predicate ifTrue ifFalse ->
      "(" <> renderExpression predicate <> " ? " <> wrapRenderedExpression ifTrue <> " : " <> wrapRenderedExpression ifFalse <> ")"
    AddExp left right -> binaryOperator "+" left right
    MulExp left right -> binaryOperator "*" left right
    LessThanExp left right -> binaryOperator "<" left right

renderUnaryExpression :: (ArrayScalar a, ArrayScalar b) => Unary a b -> String
renderUnaryExpression (Unary function) = renderExpression (function (VarExp "x0"))

renderBinaryExpression :: (ArrayScalar a, ArrayScalar b, ArrayScalar c) => Binary a b c -> String
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
    LessThanExp _ _ -> arrayScalarCType (Proxy :: Proxy a)

binaryOperator :: String -> Exp a -> Exp a -> String
binaryOperator symbol left right =
  "((" <> renderExpression left <> ") " <> symbol <> " (" <> renderExpression right <> "))"

renderFloatLiteral :: RealFloat a => a -> String
renderFloatLiteral value =
  case showFFloat Nothing value "" of
    literal
      | '.' `elem` literal || 'e' `elem` literal || 'E' `elem` literal -> literal
      | otherwise -> literal <> ".0"
