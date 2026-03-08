# Molten CPU Reference Evaluator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `molten` 增加测试用 CPU reference evaluator：支持 `Program` 的 host 输入、基础数组节点的 CPU 解释、shape-aware BLAS 的 reference helper，以及 CPU-vs-GPU 对照测试。

**Architecture:** 维持 GPU runtime 语义不变；新增一条独立的 reference evaluation layer，内部以 `massiv Array S` 作为唯一宿主表示，并通过 `ReferenceStore` 按稳定拓扑顺序解释 `Program`。公开层只暴露薄入口 `Molten.Reference`；内部实现拆到 `Molten.Internal.Reference.Array`、`Molten.Internal.Reference.BLAS`、`Molten.Internal.Reference.Program`。

**Tech Stack:** Haskell, Stack, Hspec, massiv, vector, existing `Molten.Array.Expr` / `Molten.Array.Program` / `Molten.BLAS`

---

### Task 1: 增加 reference 模块与测试骨架

**Files:**
- Modify: `package.yaml`
- Modify: `src/Molten.hs`
- Modify: `test/Spec.hs`
- Create: `src/Molten/Reference.hs`
- Create: `src/Molten/Internal/Reference/Array.hs`
- Create: `src/Molten/Internal/Reference/BLAS.hs`
- Create: `src/Molten/Internal/Reference/Program.hs`
- Create: `test/Molten/Reference/ArraySpec.hs`
- Create: `test/Molten/Reference/BlasSpec.hs`
- Create: `test/Molten/Reference/ProgramSpec.hs`

**Step 1: 写会失败的测试入口**

把新 spec 注册进 `test/Spec.hs`：

```haskell
import qualified Molten.Reference.ArraySpec as ReferenceArraySpec
import qualified Molten.Reference.BlasSpec as ReferenceBlasSpec
import qualified Molten.Reference.ProgramSpec as ReferenceProgramSpec

spec :: Spec
spec = do
  -- existing specs...
  describe "Molten.Reference.Array" ReferenceArraySpec.spec
  describe "Molten.Reference.Blas" ReferenceBlasSpec.spec
  describe "Molten.Reference.Program" ReferenceProgramSpec.spec
```

创建最小测试骨架并直接 import 尚未存在的实现模块：

```haskell
module Molten.Reference.ArraySpec (spec) where

import Molten.Reference ()
import Test.Hspec (Spec, describe, it, shouldBe)

spec :: Spec
spec = describe "wiring" (it "loads the module" (True `shouldBe` True))
```

`BlasSpec` 和 `ProgramSpec` 同理。

**Step 2: 运行测试确认 RED**

Run:

```bash
stack test --test-arguments '--match "Molten.Reference"'
```

Expected: FAIL，原因是缺少 reference 模块或导出。

**Step 3: 写最小构建骨架**

在 `package.yaml` 中加入：

```yaml
extra-source-files:
- docs/plans/2026-03-08-molten-cpu-reference-evaluator-design.md
- docs/plans/2026-03-08-molten-cpu-reference-evaluator-implementation.md

library:
  exposed-modules:
  - Molten
  - Molten.Reference
```

并创建空模块：

```haskell
module Molten.Reference () where
module Molten.Internal.Reference.Array () where
module Molten.Internal.Reference.BLAS () where
module Molten.Internal.Reference.Program () where
```

更新 `src/Molten.hs`：

```haskell
module Molten
  ( ...
  , module Molten.Reference
  ) where

import Molten.Reference
```

**Step 4: 再跑一次确认失败点前进到缺少具体 API**

Run:

```bash
stack test --test-arguments '--match "Molten.Reference"'
```

Expected: FAIL，报缺少 `runProgramCpu` / `axpyVectorRef` / `dotVectorRef` / `gemmMatrixRef` 等，而不是构建配置问题。

**Step 5: Commit**

```bash
git add package.yaml src/Molten.hs src/Molten/Reference.hs src/Molten/Internal/Reference test/Spec.hs test/Molten/Reference
git commit -m "feat: scaffold cpu reference evaluator modules"
```

### Task 2: 给 Program 增加 host 输入边界 `inputArray`

**Files:**
- Modify: `src/Molten/Array/Program.hs`
- Modify: `test/Molten/Array/ProgramSpec.hs`
- Modify: `test/Molten/Array/ProgramGpuSpec.hs`

**Step 1: 写失败测试**

在 `test/Molten/Array/ProgramSpec.hs` 追加：

```haskell
it "accepts host array inputs in Program" $ do
  let arr = A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz1 4) (fromIntegral @Int @Int32) :: A.Array A.D A.Ix1 Int32)
  program <-
    buildProgram $ do
      input0 <- inputArray arr
      mapExpr (Unary (\x -> x .+. constant 1)) input0
  programNodeIds program `shouldBe` [0, 1]
  programNodeDependencies program `shouldBe` [(0, []), (1, [0])]
```

在 `test/Molten/Array/ProgramGpuSpec.hs` 追加：

```haskell
it "uploads inputArray inputs before running on GPU" $
  withGpuContext $ \ctx ->
    withProgramRuntime ctx $ \runtime -> do
      let arr = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
      program <-
        buildProgram $ do
          input0 <- inputArray arr
          mapExpr (Unary (\x -> x .+. constant 1)) input0
      resultArray <- runProgram runtime program
      hostArray <- readDeviceArrayToHostArray ctx resultArray
      AM.toStorableVector hostArray `shouldBe` VS.fromList [2, 3, 4, 5 :: Int32]
```

**Step 2: 运行 targeted tests 确认 RED**

Run:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "accepts host array inputs in Program|uploads inputArray inputs before running on GPU"'
```

Expected: FAIL，提示 `inputArray` 不存在。

**Step 3: 写最小实现**

在 `src/Molten/Array/Program.hs`：

1. 导出 `inputArray`
2. 给 `ProgramNode` 增加 host 输入节点，例如：

```haskell
data ProgramNode
  = ...
  | forall ix a.
    (A.Index ix, Typeable ix, Typeable a, Storable a) =>
    HostInputNode !(Value ix a) !(A.Array A.S ix a)
```

3. 增加 builder：

```haskell
inputArray
  :: (A.Index ix, Typeable ix, Typeable a, Storable a)
  => A.Array A.S ix a
  -> ProgramBuilder (Value ix a)
inputArray array = do
  value <- freshValue (A.size array)
  emitNode (HostInputNode value array)
  pure value
```

4. 让 `programNodeDependencies` / `scheduleProgram` 视它为零依赖输入节点。
5. 在 `runProgram` 的执行阶段，把 `HostInputNode` 解释成：

```haskell
hostInput <- copyHostArrayToDevice (programRuntimeContext runtime) array
pure (storeValue store outputValue hostInput, Nothing)
```

不要改变现有 `inputDeviceArray` 语义。

**Step 4: 跑测试确认 GREEN**

Run:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "accepts host array inputs in Program|uploads inputArray inputs before running on GPU"'
```

Expected: PASS。

**Step 5: Commit**

```bash
git add src/Molten/Array/Program.hs test/Molten/Array/ProgramSpec.hs test/Molten/Array/ProgramGpuSpec.hs
git commit -m "feat: add host array inputs to programs"
```

### Task 3: 实现数组 reference helper 与 `Exp` 纯解释器

**Files:**
- Create: `src/Molten/Internal/Reference/Array.hs`
- Create: `test/Molten/Reference/ArraySpec.hs`
- Modify: `src/Molten/Reference.hs`

**Step 1: 写失败测试**

在 `test/Molten/Reference/ArraySpec.hs` 覆盖：

```haskell
it "maps over an Ix1 array with the Exp interpreter" $ do
  let input = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
  output <- mapArrayRef (Unary (\x -> x .+. constant 1)) input
  AM.toStorableVector output `shouldBe` VS.fromList [2, 3, 4, 5 :: Int32]

it "zips two arrays with the Exp interpreter" $ do
  let left = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
      right = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [5, 6, 7, 8 :: Int32])
  output <- zipWithArrayRef (Binary (\x y -> x .+. y)) left right
  AM.toStorableVector output `shouldBe` VS.fromList [6, 8, 10, 12 :: Int32]

it "reduces in row-major linear order" $ do
  let input = AMV.fromVector' A.Seq (A.Sz2 2 3) (VS.fromList [1, 2, 3, 4, 5, 6 :: Int32])
  output <- reduceAllArrayRef (Binary (\x y -> x .+. y)) 0 input
  AM.toStorableVector output `shouldBe` VS.fromList [21 :: Int32]

it "reshapes without changing linear order" $ do
  let input = AMV.fromVector' A.Seq (A.Sz1 6) (VS.fromList [1, 2, 3, 4, 5, 6 :: Int32])
  output <- reshapeArrayRef (A.Sz2 2 3) input
  AM.toStorableVector output `shouldBe` VS.fromList [1, 2, 3, 4, 5, 6 :: Int32]

it "rejects zipWithArrayRef shape mismatches" $ do
  let left = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
      right = AMV.fromVector' A.Seq (A.Sz1 5) (VS.fromList [1, 2, 3, 4, 5 :: Int32])
  zipWithArrayRef (Binary (\x y -> x .+. y)) left right `shouldThrow` isArgumentError "zipWithArrayRef" "same shape"
```

**Step 2: 运行 targeted tests 确认 RED**

Run:

```bash
stack test --test-arguments '--match "Molten.Reference.Array"'
```

Expected: FAIL，提示 `mapArrayRef` / `zipWithArrayRef` / `reduceAllArrayRef` / `reshapeArrayRef` 不存在。

**Step 3: 写最小实现**

在 `src/Molten/Internal/Reference/Array.hs` 实现：

```haskell
module Molten.Internal.Reference.Array
  ( evalBinaryRef
  , evalUnaryRef
  , fillArrayRef
  , mapArrayRef
  , reduceAllArrayRef
  , reshapeArrayRef
  , zipWithArrayRef
  ) where
```

建议实现结构：

```haskell
evalUnaryRef :: (ArrayScalar a, ArrayScalar b) => Unary a b -> a -> b
evalBinaryRef :: (ArrayScalar a, ArrayScalar b, ArrayScalar c) => Binary a b c -> a -> b -> c

fillArrayRef :: (ArrayScalar a, A.Index ix) => a -> A.Sz ix -> IO (A.Array A.S ix a)
mapArrayRef :: (ArrayScalar a, ArrayScalar b, A.Index ix) => Unary a b -> A.Array A.S ix a -> IO (A.Array A.S ix b)
zipWithArrayRef :: (ArrayScalar a, ArrayScalar b, ArrayScalar c, A.Index ix) => Binary a b c -> A.Array A.S ix a -> A.Array A.S ix b -> IO (A.Array A.S ix c)
reduceAllArrayRef :: (ArrayScalar a, NumericExp a, A.Index ix) => Binary a a a -> a -> A.Array A.S ix a -> IO (A.Array A.S A.Ix1 a)
reshapeArrayRef :: (A.Index ix, A.Index ix') => A.Sz ix' -> A.Array A.S ix a -> IO (A.Array A.S ix' a)
```

实现要求：

- 不重走 GPU 路线
- 直接用 `AM.toStorableVector` / `AMV.fromVector'` 保持线性顺序
- `reduceAllArrayRef` 明确按线性顺序左折叠
- shape 错误用 `throwArgumentError`
- `evalUnaryRef` / `evalBinaryRef` 通过小型纯解释器解释现有 `Exp`

**Step 4: 跑测试确认 GREEN**

Run:

```bash
stack test --test-arguments '--match "Molten.Reference.Array"'
```

Expected: PASS。

**Step 5: Commit**

```bash
git add src/Molten/Internal/Reference/Array.hs src/Molten/Reference.hs test/Molten/Reference/ArraySpec.hs
git commit -m "feat: add cpu reference array helpers"
```

### Task 4: 实现 shape-aware BLAS reference helper

**Files:**
- Create: `src/Molten/Internal/Reference/BLAS.hs`
- Create: `test/Molten/Reference/BlasSpec.hs`
- Modify: `src/Molten/Reference.hs`

**Step 1: 写失败测试**

在 `test/Molten/Reference/BlasSpec.hs` 覆盖：

```haskell
it "computes axpyVectorRef" $ do
  let x = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Float])
      y = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [10, 20, 30, 40 :: Float])
  output <- axpyVectorRef 2 x y
  AM.toStorableVector output `shouldBe` VS.fromList [12, 24, 36, 48 :: Float]

it "computes dotVectorRef" $ do
  let x = AMV.fromVector' A.Seq (A.Sz1 3) (VS.fromList [2, 4, 6 :: Float])
      y = AMV.fromVector' A.Seq (A.Sz1 3) (VS.fromList [4, 5, 6 :: Float])
  dotVectorRef x y `shouldReturn` 64

it "computes gemmMatrixRef with row-major semantics" $ do
  let a = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [1, 2, 3, 4 :: Float])
      b = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [5, 6, 7, 8 :: Float])
      c = AMV.fromVector' A.Seq (A.Sz2 2 2) (VS.fromList [0, 0, 0, 0 :: Float])
  output <- gemmMatrixRef MatrixGemmRef
    { matrixGemmRefTransA = NoTranspose
    , matrixGemmRefTransB = NoTranspose
    , matrixGemmRefAlpha = 1
    , matrixGemmRefA = a
    , matrixGemmRefB = b
    , matrixGemmRefBeta = 0
    , matrixGemmRefC = c
    }
  AM.toStorableVector output `shouldBe` VS.fromList [19, 22, 43, 50 :: Float]

it "rejects gemmMatrixRef shape mismatches" $ do
  -- mismatched inner dimensions should throw ArgumentError
```

**Step 2: 运行 targeted tests 确认 RED**

Run:

```bash
stack test --test-arguments '--match "Molten.Reference.Blas"'
```

Expected: FAIL。

**Step 3: 写最小实现**

在 `src/Molten/Internal/Reference/BLAS.hs` 实现：

```haskell
data MatrixGemmRef a = MatrixGemmRef
  { matrixGemmRefTransA :: !Transpose
  , matrixGemmRefTransB :: !Transpose
  , matrixGemmRefAlpha :: !a
  , matrixGemmRefA :: !(A.Array A.S A.Ix2 a)
  , matrixGemmRefB :: !(A.Array A.S A.Ix2 a)
  , matrixGemmRefBeta :: !a
  , matrixGemmRefC :: !(A.Array A.S A.Ix2 a)
  }

axpyVectorRef :: (HasCallStack, Num a) => a -> A.Array A.S A.Ix1 a -> A.Array A.S A.Ix1 a -> IO (A.Array A.S A.Ix1 a)
dotVectorRef :: (HasCallStack, Num a) => A.Array A.S A.Ix1 a -> A.Array A.S A.Ix1 a -> IO a
gemmMatrixRef :: (HasCallStack, Num a) => MatrixGemmRef a -> IO (A.Array A.S A.Ix2 a)
```

实现要求：

- shape 校验规则与 `src/Molten/BLAS.hs` 保持一致
- `gemmMatrixRef` 的 row-major 解释与现有 `gemmMatrix` 完全对齐
- 不隐式改布局
- `beta` 作用到输入 `C`

**Step 4: 跑测试确认 GREEN**

Run:

```bash
stack test --test-arguments '--match "Molten.Reference.Blas"'
```

Expected: PASS。

**Step 5: Commit**

```bash
git add src/Molten/Internal/Reference/BLAS.hs src/Molten/Reference.hs test/Molten/Reference/BlasSpec.hs
git commit -m "feat: add cpu reference blas helpers"
```

### Task 5: 实现 `runProgramCpu` 与 `ReferenceOutput`

**Files:**
- Create: `src/Molten/Internal/Reference/Program.hs`
- Modify: `src/Molten/Reference.hs`
- Create: `test/Molten/Reference/ProgramSpec.hs`
- Modify: `src/Molten/Array/Program.hs`

**Step 1: 写失败测试**

在 `test/Molten/Reference/ProgramSpec.hs` 覆盖：

```haskell
it "runs fill -> map -> zipWith -> reduceAll on CPU" $ do
  program <-
    buildProgram $ do
      base <- fillArrayP @A.Ix1 @Int32 2 (A.Sz1 4)
      mapped <- mapExpr (Unary (\x -> x .+. constant 1)) base
      zipped <- zipWithExpr (Binary (\x y -> x .*. y)) mapped base
      reduceAll (Binary (\x y -> x .+. y)) 0 zipped
  resultArray <- runProgramCpu program
  AM.toStorableVector resultArray `shouldBe` VS.fromList [24 :: Int32]

it "runs inputArray -> mapExpr on CPU" $ do
  let input = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
  program <-
    buildProgram $ do
      value0 <- inputArray input
      mapExpr (Unary (\x -> x .+. constant 1)) value0
  resultArray <- runProgramCpu program
  AM.toStorableVector resultArray `shouldBe` VS.fromList [2, 3, 4, 5 :: Int32]

it "runs gemmMatrixP on CPU" $ do
  -- build a small Program with inputArray or fillArrayP and compare to expected

it "rejects device-only inputs in runProgramCpu" $
  withGpuContext $ \ctx -> do
    deviceInput <- copyHostArrayToDevice ctx (AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32]))
    program <- buildProgram (inputDeviceArray deviceInput)
    runProgramCpu program `shouldThrow` isArgumentError "runProgramCpu" "device-only inputs"
```

**Step 2: 运行 targeted tests 确认 RED**

Run:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Reference.Program"'
```

Expected: FAIL。

**Step 3: 写最小实现**

在 `src/Molten/Internal/Reference/Program.hs` 实现：

```haskell
type ReferenceStore = Map Int Dynamic.Dynamic

class ReferenceOutput a where
  type ReferenceResult a
  resolveReferenceOutput :: ReferenceStore -> a -> IO (ReferenceResult a)

runProgramCpu :: (HasCallStack, ReferenceOutput a) => Program a -> IO (ReferenceResult a)
```

需要在 `src/Molten/Array/Program.hs` 提供足够的内部可见信息给 reference evaluator。最小做法：

- 导出 `programNodes` 或提供安全 accessor
- 或新增：

```haskell
programNodesInOrder :: Program a -> [ProgramNodeView]
```

其中 `ProgramNodeView` 暴露执行所需字段，但不暴露不必要的 runtime 细节。

`runProgramCpu` 的节点解释规则：

- `HostInputNode` -> 放入 `ReferenceStore`
- `InputNode` -> `throwArgumentError "runProgramCpu" "device-only inputs are not supported by the CPU reference evaluator"`
- `FillNode` -> `fillArrayRef`
- `MapNode` -> `mapArrayRef`
- `ZipNode` -> `zipWithArrayRef`
- `ReduceNode` -> `reduceAllArrayRef`
- `ReshapeNode` -> `reshapeArrayRef`
- `AxpyNode` -> `axpyVectorRef`
- `GemmNode` -> `gemmMatrixRef`
- 其他节点（FFT / RAND / future unsupported nodes） -> `throwArgumentError "runProgramCpu" "unsupported reference node"`

不要尝试模拟：

- stream
- event
- async
- FFT
- RAND

**Step 4: 跑测试确认 GREEN**

Run:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "Molten.Reference.Program"'
```

Expected: PASS。

**Step 5: Commit**

```bash
git add src/Molten/Array/Program.hs src/Molten/Internal/Reference/Program.hs src/Molten/Reference.hs test/Molten/Reference/ProgramSpec.hs
git commit -m "feat: add cpu program reference evaluator"
```

### Task 6: 增加 CPU-vs-GPU 对照测试

**Files:**
- Modify: `test/Molten/Array/ProgramGpuSpec.hs`
- Modify: `test/Molten/BLAS/ArraySpec.hs`
- Optionally Modify: `test/Molten/Array/ProgramSpec.hs`

**Step 1: 写失败测试**

在 `test/Molten/Array/ProgramGpuSpec.hs` 增加至少两个对照测试：

```haskell
it "matches CPU reference for inputArray -> mapExpr -> reduceAll" $
  withGpuContext $ \ctx ->
    withProgramRuntime ctx $ \runtime -> do
      let input = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int32])
      program <-
        buildProgram $ do
          value0 <- inputArray input
          value1 <- mapExpr (Unary (\x -> x .+. constant 1)) value0
          reduceAll (Binary (\x y -> x .+. y)) 0 value1
      cpuResult <- runProgramCpu program
      gpuResult <- runProgram runtime program
      gpuHost <- readDeviceArrayToHostArray ctx gpuResult
      AM.toStorableVector gpuHost `shouldBe` AM.toStorableVector cpuResult

it "matches CPU reference for inputArray -> gemmMatrixP" $ ...
```

在 `test/Molten/BLAS/ArraySpec.hs` 增加：

```haskell
it "matches gemmMatrixRef" $ ...
it "matches axpyVectorRef" $ ...
it "matches dotVectorRef" $ ...
```

**Step 2: 运行 targeted tests 确认 RED**

Run:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "matches CPU reference|matches gemmMatrixRef|matches axpyVectorRef|matches dotVectorRef"'
```

Expected: FAIL。

**Step 3: 写最小实现/调整测试辅助**

如需辅助函数，可在测试模块内部加：

- `compareProgramResultWithReference`
- `compareArrayVector`

但不要把测试辅助误提升为生产 API。

**Step 4: 跑测试确认 GREEN**

Run:

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test --test-arguments '--match "matches CPU reference|matches gemmMatrixRef|matches axpyVectorRef|matches dotVectorRef"'
```

Expected: PASS。

**Step 5: Commit**

```bash
git add test/Molten/Array/ProgramGpuSpec.hs test/Molten/BLAS/ArraySpec.hs
git commit -m "test: cross check gpu execution with cpu reference evaluator"
```

### Task 7: 文档、导出与最终验证

**Files:**
- Modify: `README.md`
- Modify: `src/Molten.hs`
- Modify: `package.yaml`
- Modify: `docs/plans/2026-03-08-molten-cpu-reference-evaluator-design.md`

**Step 1: 补文档**

在 README 中补一段：

- `Molten.Reference` 的定位
- `runProgramCpu` 只支持 host 输入与基础节点/BLAS
- GPU-vs-CPU reference 的基本示例

**Step 2: 追加设计文件的 `Implementation Results`**

记录：

- 新增模块
- `inputArray` 的最终落地方式
- CPU evaluator 目前不支持的节点
- 验证命令与结果

**Step 3: 跑最终验证**

Run:

```bash
stack build
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run
```

Expected:

- `stack build` 成功
- `HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test` 成功
- `HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run` 不回退

**Step 4: Commit**

```bash
git add README.md src/Molten.hs package.yaml docs/plans/2026-03-08-molten-cpu-reference-evaluator-design.md
git commit -m "docs: finalize cpu reference evaluator"
```
