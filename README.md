# molten

`molten` 是建立在 `haskell-rocm` 低层绑定之上的高层 Haskell ROCm API 原型。当前已实现四层能力：

- **运行时 + BLAS MVP**
  - 显式 `Context`
  - `Host` / `PinnedHost` / `Device` 线性 `Buffer`
  - 同步 / 异步传输与 `GpuFuture`
  - rocBLAS `axpy` / `dot` / `dotInto` / `gemmNative`
  - 高层 row-major `gemm`
- **数组互操作层**
  - `vector` 互操作
  - `massiv` 多维数组互操作
  - 轻量 `DeviceArray ix a` 设备侧 shape wrapper
  - eager `DeviceArray` reshape / clone / host / pinned roundtrip
  - shape-aware BLAS：`axpyVector` / `dotVector` / `gemmMatrix`
- **独立运行时**
  - `ArrayRuntime`：typed EDSL + hipRTC JIT one-shot kernels
  - `FftRuntime`：rocFFT plan cache / workspace 管理
  - `RandRuntime`：rocRAND generator cache / seed / stream 绑定
- **staged Program**
  - SSA `Value ix a`
  - JIT / BLAS / FFT / RAND 混合节点
  - DAG 调度与保守多 stream 执行
- **CPU reference evaluator**
  - `inputArray` host 输入边界
  - `runProgramCpu` 解释基础数组节点与 shape-aware BLAS
  - `axpyVectorRef` / `dotVectorRef` / `gemmMatrixRef`
  - 便于 GPU-vs-CPU correctness 对照

## 目录结构

- `src/Molten/Core/*`：context、stream、event、buffer、future、transfer
- `src/Molten/BLAS*`：native BLAS、shape-aware BLAS 与 GEMM 类型
- `src/Molten/Array/Device.hs`：`DeviceArray ix a`
- `src/Molten/Array/Transfer.hs`：eager device-array copy / reshape / clone
- `src/Molten/Array/Expr.hs`：typed array EDSL
- `src/Molten/Array/Runtime.hs`：hipRTC JIT one-shot kernels
- `src/Molten/Array/Program.hs`：SSA + DAG `Program`
- `src/Molten/FFT*`：rocFFT runtime 与 eager FFT API
- `src/Molten/RAND*`：rocRAND runtime 与 eager RAND API
- `src/Molten/Reference.hs`：CPU reference evaluator 公开入口
- `src/Molten/Internal/Reference/*`：CPU reference helper 内部实现
- `src/Molten/Interop/Vector.hs`：`Data.Vector.Storable` 互操作
- `src/Molten/Interop/Massiv.hs`：`massiv` 多维数组互操作
- `test/*`：纯测试与 GPU 集成测试
- `docs/plans/2026-03-08-molten-highlevel-rocm-design.md`：高层 ROCm 设计
- `docs/plans/2026-03-08-molten-massiv-adapter-design.md`：`massiv` adapter 设计
- `docs/plans/2026-03-08-molten-array-program-runtime-design.md`：array runtime / FFT / RAND / Program 设计

## 构建

本项目通过 `stack` 构建，并把 `../haskell-rocm/packages` 中的以下本地包作为依赖：

- `rocm-ffi-core`
- `rocm-hip-runtime`
- `rocm-rocblas`
- `rocm-rocfft`
- `rocm-rocrand`

当前 `stack.yaml` 还显式加入了：

- `/usr/include`
- `/usr/lib`
- `/usr/lib64`

这对应本机的 ROCm 安装位置。如果你的 ROCm 安装不在这些目录，需要同步调整 `stack.yaml`。

常用命令：

```bash
stack build
stack test
stack run
```

在当前 `gfx1103` 环境上，为了让 eager `Context` 初始化下的 rocBLAS 测试与 demo 正常运行，可使用：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack test
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run
```

## Examples

仓库现在提供五个默认就会跑 **stress workload + self-check** 的 example executables：

### 1. `molten-example-mlp-forward`

覆盖：`Program`、shape-aware BLAS、JIT elementwise、CPU-vs-GPU shadow check。

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-mlp-forward
```

可选参数：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-mlp-forward -- --batch 4096 --in 1024 --hidden 2048 --out 512
```

默认先跑一个小 shadow case 做 CPU-vs-GPU 对照，再跑大规模 GPU stress case。若 shadow check 或 stress summary 失败，程序会直接退出失败。

### 2. `molten-example-heat2d-fft`

覆盖：`FftRuntime`、复数 pointwise JIT kernel、二维 FFT repeated execution。

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-heat2d-fft
```

可选参数：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-heat2d-fft -- --nx 1024 --ny 1024 --steps 100 --alpha 0.05 --dt 1e-3
```

默认会检查两次重复运行的稳定性，以及热扩散 stepper 的能量不增长条件。失败时直接退出失败。

### 3. `molten-example-monte-carlo-bachelier`

覆盖：`randNormalP`、大规模 payoff / reduction、固定 seed 的统计稳定性检查。

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-monte-carlo-bachelier
```

可选参数：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-monte-carlo-bachelier -- --paths 8000000 --seed 12345 --spot 100 --strike 100 --sigma 20 --sqrtT 1
```

默认会重复两次同 seed 运行，检查结果重现性、非负价格与 95% 置信区间的基本一致性。失败时直接退出失败。

### 4. `molten-example-attention-forward`

覆盖：`softmaxRowsP`、2D row-wise reduction / broadcast、`Program` attention forward、CPU-vs-GPU shadow check。

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-attention-forward
```

可选参数：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-attention-forward -- --tokens 2048 --model 128 --value 128
```

默认先跑一个小规模 shadow case，对比 CPU reference 与 GPU attention forward；随后再跑较大的 GPU stress case，并检查 softmax 每行和接近 1。失败时直接退出失败。

### 5. `molten-example-black-scholes-asian`

覆盖：`forLoopP`、`LoopNode`、多时间步 Black-Scholes path simulation、Arithmetic Asian payoff、shadow + stress 双路径验证。

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-black-scholes-asian
```

可选参数：

```bash
HSA_OVERRIDE_GFX_VERSION=11.0.0 stack run molten-example-black-scholes-asian -- --paths 1000000 --steps 252 --spot 100 --strike 100 --rate 0.05 --vol 0.2 --maturity 1 --seed 12345
```

默认先跑 host normals 驱动的 CPU-vs-GPU shadow case，再跑真实 `RAND + LoopNode` 的 stress case，并检查非负价格、有限标准误差、95% 置信区间与 fixed-seed 重复性。失败时直接退出失败。

## 测试策略

测试分成三类：

1. **纯测试**：不依赖 GPU，覆盖参数校验、row-major → column-major 的参数改写，以及 `massiv` shape / rebuild 校验。
2. **GPU 运行时测试**：覆盖 sync / async transfer。
3. **GPU 集成测试**：覆盖 BLAS 与 `massiv` roundtrip。

如果当前机器没有可用 ROCm GPU，相关测试会被跳过。

GPU 相关测试假设当前环境可以成功创建 `Context`。如果 ROCm/rocBLAS 运行时本身不可用，测试会直接失败，这被视为环境问题而不是测试层的可恢复条件。

## `massiv` 互操作

`Molten.Interop.Massiv` 提供两类 bridge：

- `withHostBufferFromArray` / `readHostBufferToArray`
- `withPinnedBufferFromArray` / `readPinnedBufferToArray`

输入数组会先显式物化到 `Manifest`，再按 **row-major linear order** 进入 `Buffer` 世界。输出统一恢复为 `Array S ix e`。

设备侧多维 shape 由 `DeviceArray ix a` 保存：

```haskell
data DeviceArray ix a
```

它是对底层 `Buffer 'Device a` 的轻量 wrapper，不重做资源管理。

## 最小 `massiv` 示例

```haskell
import Molten
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest as AM

main :: IO ()
main =
  withContext (DeviceId 0) $ \ctx -> do
    let arr =
          (A.computeAs A.S (A.makeArrayLinear A.Seq (A.Sz2 2 3) fromIntegral :: A.Array A.D A.Ix2 Int))
            :: A.Array A.S A.Ix2 Int

    withPinnedBufferFromArray arr $ \sz pinnedIn ->
      withDeviceArray ctx sz $ \dev ->
        withPinnedBufferFromArray arr $ \_ pinnedOut -> do
          upload <- copyH2DAsync (contextDefaultStream ctx) (deviceArrayBuffer dev) pinnedIn
          () <- await upload
          download <- copyD2HAsync (contextDefaultStream ctx) pinnedOut (deviceArrayBuffer dev)
          () <- await download
          out <- readPinnedBufferToArray sz pinnedOut
          print (AM.toStorableVector out)
```

## 最小 Program 示例

```haskell
import Molten
import qualified Data.Massiv.Array as A

main :: IO ()
main =
  withContext (DeviceId 0) $ \ctx ->
    withProgramRuntime ctx $ \rt -> do
      prog <- buildProgram $ do
        x <- fillArrayP @A.Ix1 @Float 2 (A.Sz1 8)
        y <- mapExpr (Unary (\v -> v .+. constant 1)) x
        reduceAll (Binary (\a b -> a .+. b)) 0 y
      out <- runProgram rt prog
      host <- readDeviceArrayToHostArray ctx out
      print host
```

Program 层当前支持：

- JIT `fill` / `map` / `zipWith` / `reduceAll`
- shape-aware BLAS 节点
- FFT forward / inverse 节点
- RAND uniform / normal 节点
- 保守 DAG + 多 stream 调度

## CPU reference evaluator

`Molten.Reference` 提供测试用 CPU reference evaluator。它不是正式 CPU backend，也不模拟 stream、event 或 async 语义。它的职责是：用 `massiv Array S` 解释同一份 `Program`，产出可与 GPU 读回结果直接比较的 host 结果。

当前公开入口包括：

- `runProgramCpu`
- `mapArrayRef`
- `zipWithArrayRef`
- `reduceAllArrayRef`
- `reshapeArrayRef`
- `axpyVectorRef`
- `dotVectorRef`
- `gemmMatrixRef`

当前限制：

- `runProgramCpu` 只支持 `inputArray`
- 遇到 `inputDeviceArray` 会直接报错
- 只解释基础数组节点与 shape-aware BLAS
- 还不支持 FFT / RAND 节点

最小示例：

```haskell
import Molten
import qualified Data.Massiv.Array as A
import qualified Data.Massiv.Array.Manifest.Vector as AMV
import qualified Data.Vector.Storable as VS

main :: IO ()
main = do
  let input = AMV.fromVector' A.Seq (A.Sz1 4) (VS.fromList [1, 2, 3, 4 :: Int])
  prog <- buildProgram $ do
    x <- inputArray input
    y <- mapExpr (Unary (\v -> v .+. constant 1)) x
    reduceAll (Binary (\a b -> a .+. b)) 0 y
  print =<< runProgramCpu prog
```

## 设计取舍

- `Context` 在创建时立即初始化默认 BLAS handle。这样 `Context` 的可用性与 BLAS 运行时状态保持一致。
- 高层 `dot` 返回主机可见标量，因此它是显式同步边界；`dotInto` 则把结果留在 device buffer。
- 高层 `gemm` 默认按 row-major 解释；`gemmNative` 保留 column-major 语义。
- `massiv` 输入先显式物化到 `Manifest`；设备侧 shape 由 `DeviceArray ix a` 保存；host/pinned/device 之间继续复用既有 `Buffer` 与 transfer 层。
