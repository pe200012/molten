{-# LANGUAGE TypeApplications #-}

module Molten.Array.ExprSpec (spec) where

import Data.Complex (Complex)
import Data.Int (Int32)
import Data.Proxy (Proxy(..))
import Data.Word (Word32)
import Molten.Array.Expr
  ( ArrayScalar(arrayScalarCType, arrayScalarTypeName)
  , Exp
  , Unary(..)
  , cast
  , constant
  , renderUnaryExpression
  , select
  , (.+.)
  , (.<.)
  )
import Test.Hspec (Spec, describe, it, shouldBe)

spec :: Spec
spec = do
  describe "renderUnaryExpression" $ do
    it "renders comparisons and select into a typed ternary expression" $ do
      let expression :: Unary Float Float
          expression = Unary (\x -> select (x .<. constant 0.5) (x .+. constant 1.0) (constant 0.0))
      renderUnaryExpression expression `shouldBe` "(((x0) < (0.5f)) ? ((x0) + (1.0f)) : (0.0f))"

    it "renders explicit casts with the target HIP C type" $ do
      let expression :: Unary Int32 Float
          expression = Unary (\x -> cast x :: Exp Float)
      renderUnaryExpression expression `shouldBe` "((float)(x0))"

  describe "ArrayScalar metadata" $ do
    it "exposes scalar names and HIP C type names for the supported surface types" $ do
      arrayScalarTypeName (Proxy @(Complex Float)) `shouldBe` "ComplexFloat"
      arrayScalarCType (Proxy @(Complex Float)) `shouldBe` "float2"
      arrayScalarTypeName (Proxy @Int32) `shouldBe` "Int32"
      arrayScalarCType (Proxy @Int32) `shouldBe` "int"
      arrayScalarTypeName (Proxy @Word32) `shouldBe` "Word32"
      arrayScalarCType (Proxy @Word32) `shouldBe` "unsigned int"
      arrayScalarTypeName (Proxy @Bool) `shouldBe` "Bool"
      arrayScalarCType (Proxy @Bool) `shouldBe` "bool"
