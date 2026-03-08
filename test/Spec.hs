module Main (main) where

import Test.Hspec (Spec, describe, hspec)
import qualified Molten.BLAS.HighLevelSpec as HighLevelSpec
import qualified Molten.BLAS.NativeSpec as NativeSpec
import qualified Molten.Core.BufferSpec as BufferSpec
import qualified Molten.Core.ContextSpec as ContextSpec
import qualified Molten.Core.GemmSpec as GemmSpec
import qualified Molten.Core.TransferSpec as TransferSpec

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "Molten.Core.Buffer" BufferSpec.spec
  describe "Molten.Core.Context" ContextSpec.spec
  describe "Molten.Core.Transfer" TransferSpec.spec
  describe "Molten.Core.Gemm" GemmSpec.spec
  describe "Molten.BLAS.Native" NativeSpec.spec
  describe "Molten.BLAS.HighLevel" HighLevelSpec.spec
