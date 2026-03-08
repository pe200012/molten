module Main (main) where

import Test.Hspec (Spec, describe, hspec)
import qualified Molten.Array.DeviceSpec as DeviceSpec
import qualified Molten.Array.ExprSpec as ExprSpec
import qualified Molten.Array.ProgramGpuSpec as ProgramGpuSpec
import qualified Molten.Array.ProgramSpec as ProgramSpec
import qualified Molten.Array.RuntimeSpec as RuntimeSpec
import qualified Molten.Array.TransferSpec as ArrayTransferSpec
import qualified Molten.BLAS.ArraySpec as ArrayBlasSpec
import qualified Molten.BLAS.HighLevelSpec as HighLevelSpec
import qualified Molten.BLAS.NativeSpec as NativeSpec
import qualified Molten.Core.BufferSpec as BufferSpec
import qualified Molten.Core.ContextSpec as ContextSpec
import qualified Molten.Core.GemmSpec as GemmSpec
import qualified Molten.Core.TransferSpec as TransferSpec
import qualified Molten.Examples.CommonSpec as ExampleCommonSpec
import qualified Molten.Examples.Heat2dFftSpec as Heat2dFftSpec
import qualified Molten.Examples.MlpForwardSpec as MlpForwardSpec
import qualified Molten.Examples.MonteCarloBachelierSpec as MonteCarloBachelierSpec
import qualified Molten.FFT.GpuSpec as FftGpuSpec
import qualified Molten.FFT.RuntimeSpec as FftRuntimeSpec
import qualified Molten.Interop.MassivGpuSpec as MassivGpuSpec
import qualified Molten.Interop.MassivSpec as MassivSpec
import qualified Molten.RAND.GpuSpec as RandGpuSpec
import qualified Molten.RAND.RuntimeSpec as RandRuntimeSpec
import qualified Molten.Reference.ArraySpec as ReferenceArraySpec
import qualified Molten.Reference.BlasSpec as ReferenceBlasSpec
import qualified Molten.Reference.ProgramSpec as ReferenceProgramSpec

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
  describe "Molten.Array.Device" DeviceSpec.spec
  describe "Molten.Array.Expr" ExprSpec.spec
  describe "Molten.Array.Program" ProgramSpec.spec
  describe "Molten.Array.Program.GPU" ProgramGpuSpec.spec
  describe "Molten.Array.Runtime" RuntimeSpec.spec
  describe "Molten.Array.Transfer" ArrayTransferSpec.spec
  describe "Molten.BLAS.Array" ArrayBlasSpec.spec
  describe "Molten.Examples.Common" ExampleCommonSpec.spec
  describe "Molten.Examples.MlpForward" MlpForwardSpec.spec
  describe "Molten.Examples.Heat2dFft" Heat2dFftSpec.spec
  describe "Molten.Examples.MonteCarloBachelier" MonteCarloBachelierSpec.spec
  describe "Molten.FFT.Runtime" FftRuntimeSpec.spec
  describe "Molten.FFT.GPU" FftGpuSpec.spec
  describe "Molten.Interop.Massiv" MassivSpec.spec
  describe "Molten.Interop.Massiv.GPU" MassivGpuSpec.spec
  describe "Molten.RAND.Runtime" RandRuntimeSpec.spec
  describe "Molten.RAND.GPU" RandGpuSpec.spec
  describe "Molten.Reference.Array" ReferenceArraySpec.spec
  describe "Molten.Reference.Blas" ReferenceBlasSpec.spec
  describe "Molten.Reference.Program" ReferenceProgramSpec.spec
