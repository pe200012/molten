# molten

`molten` 是建立在 `haskell-rocm` 低层绑定之上的高层 Haskell ROCm API 原型。当前 MVP 已实现：

- 显式 `Context`
- `Host` / `PinnedHost` / `Device` 线性 `Buffer`
- 同步 / 异步传输与 `GpuFuture`
- `vector` 互操作
- rocBLAS `axpy` / `dot` / `dotInto` / `gemmNative`
- 高层 row-major `gemm`

## 目录结构

- `src/Molten/Core/*`：context、stream、event、buffer、future、transfer
- `src/Molten/BLAS*`：native BLAS 与高层 BLAS
- `src/Molten/Interop/Vector.hs`：`Data.Vector.Storable` 互操作
- `test/*`：纯测试与 GPU 集成测试
- `docs/plans/2026-03-08-molten-highlevel-rocm-design.md`：设计文件

## 构建

本项目通过 `stack` 构建，并把 `../haskell-rocm/packages` 中的以下本地包作为依赖：

- `rocm-ffi-core`
- `rocm-hip-runtime`
- `rocm-rocblas`

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

## 测试策略

测试分成两类：

1. **纯测试**：不依赖 GPU，覆盖参数校验与 row-major → column-major 的参数改写。
2. **GPU 集成测试**：覆盖 sync / async transfer 与 BLAS 数值正确性。

如果当前机器没有可用 ROCm GPU，相关测试会被跳过。

如果 `Context` 在当前环境下无法完成初始化，GPU 相关测试会以 `pending` 形式报告，并直接显示底层异常信息。

## 最小示例

```haskell
import Molten
import qualified Data.Vector.Storable as VS

main :: IO ()
main =
  withContext (DeviceId 0) $ \ctx -> do
    let input = VS.fromList [1 .. 8 :: Float]
        n = VS.length input

    withHostBufferFromVector input $ \hostIn ->
      withDeviceBuffer ctx n $ \deviceBuf ->
        withHostBuffer n $ \hostOut -> do
          copyH2D ctx deviceBuf hostIn
          copyD2H ctx hostOut deviceBuf
          out <- readHostBufferToVector hostOut
          print out
```

## 设计取舍

- `Context` 现在在创建时就立即初始化默认 BLAS handle。这样 `Context` 的可用性与 BLAS 运行时状态保持一致。
- 高层 `dot` 返回主机可见标量，因此它是显式同步边界；`dotInto` 则把结果留在 device buffer。
- 高层 `gemm` 默认按 row-major 解释；`gemmNative` 保留 column-major 语义。
