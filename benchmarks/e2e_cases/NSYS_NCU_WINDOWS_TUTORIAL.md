# nsys / ncu 入门教程（Windows 原生环境）

> 目标：在 **Windows 主机**上学会使用 NVIDIA Nsight Systems（nsys）和 Nsight Compute（ncu），
> 为将来在裸 Linux 上分析 vLLM 推理框架打基础。
>
> **为什么在 Windows 上学**：
> - nsys / ncu 都有完整的 Windows 原生版本，且 Windows 是 NVIDIA 驱动的原生环境，
>   CUPTI / NVPC profiling 事件**完整可用**（不像 WSL2 dxg 透传那样被挡掉）；
> - 学的是「工具用法 + 报告解读」这套通用能力，学会后到任何裸 Linux 上都能直接复用。
>
> **本教程不涉及 vLLM**：vLLM 官方不支持 Windows 原生运行，所以这里用一个最小的
> CUDA / PyTorch 程序作为「被分析对象」。工具学会了，分析对象换成 vLLM 即可。

---

## 0. 前置条件

- Windows 10/11，已安装 NVIDIA 驱动（你的 RTX 5070 Ti 就在 Windows 上）
- 能访问 NVIDIA 官网下载安装包

> 提醒：请用**管理员权限**运行 PowerShell / 命令行，避免 profiling 时因权限不足而失败。

---

## 0.5 快速开始（本机验证过的 Windows 完整命令）

> 本小节是「直接抄作业」版本，命令在本机（RTX 5070 Ti + Windows + 管理员终端）实测通过。
> 详细原理见下文各节。

### 环境准备（一次性）

```powershell
# 1) Python 独立 venv + CUDA 版 PyTorch
python -m venv C:\profiling-lab
C:\profiling-lab\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128

# 2) 基准脚本复制到 venv 目录
Copy-Item "d:\code\vllm-serving-optimization\benchmarks\e2e_cases\roofline_bench.py" "C:\profiling-lab\roofline_bench.py"

# 3) 允许非管理员访问 GPU 性能计数器（管理员 PowerShell 执行，重启后生效）
reg add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" /v "RestrictProfilingToAdminUsers" /t REG_DWORD /d 0 /f
```

> 注意：注册表改完要**重启电脑**才生效；如果不想改注册表，直接全程用管理员终端跑 ncu 也行。

### 快速命令（每次直接用）

```powershell
# A) 先跑基准，看各算子达成率
& "C:\profiling-lab\Scripts\python.exe" "C:\profiling-lab\roofline_bench.py"

# B) nsys 采集（宏观：时间线 + kernel 耗时排行；无需管理员）
& "D:\Program Files\host\target-windows-x64\nsys.exe" profile --trace=cuda,nvtx --output=C:\profiling-lab\roofline_nsys --force-overwrite=true --stats=true "C:\profiling-lab\Scripts\python.exe" "C:\profiling-lab\roofline_bench.py"

# 查看方式 1：GUI 时间线（最直观；看 GPU 忙不忙、有没有 idle 用这个）
& "D:\Program Files\host-windows-x64\nsys-ui.exe" C:\profiling-lab\roofline_nsys.nsys-rep

# 查看方式 2：命令行导出各汇总表（stats --report 指定表名；--force-export 强制重新生成）
#   2a) kernel 耗时排行（重点，找出热点 kernel）
& "D:\Program Files\host\target-windows-x64\nsys.exe" stats --force-export=true --report cuda_gpu_kern_sum C:\profiling-lab\roofline_nsys.nsys-rep

#   2b) CUDA API 调用耗时（看 CPU 侧调度开销，可选）
& "D:\Program Files\host\target-windows-x64\nsys.exe" stats --report cuda_api_sum C:\profiling-lab\roofline_nsys.nsys-rep
#   2c) 列出所有可用报表名（nsys 没有一次全量输出，先查有哪些表再逐个看；对应 ncu 的 --list-sections）
& "D:\Program Files\host\target-windows-x64\nsys.exe" stats --help-reports C:\profiling-lab\roofline_nsys.nsys-rep

# C) ncu 采集 fp32 GEMM（微观：GPU Speed Of Light，需管理员终端）
#     用 --set full 采完整指标集（含 FLOP，是看 Roofline / 算 AI 的前提；basic 没存 FLOP）
#     定位 kernel：--kernel-name "*sgemm*" 在 Windows 上经常匹配不到（报 No kernels were profiled），
#     稳妥用 --launch-skip 2 跳过前 2 个 randn kernel，直采 GEMM（预热第 1 次）。
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --launch-skip 2 --launch-count 1 --set full -f --export "C:\profiling-lab\roofline_fp32_sgemm_full" "C:\profiling-lab\Scripts\python.exe" "C:\profiling-lab\roofline_bench.py" --kind fp32

# 查看方式 1：GUI 打开完整报告（看 Occupancy / Warp State / 时间线等所有 section 最直观）
& "D:\Program Files\host\windows-desktop-win7-x64\ncu-ui.exe" "C:\profiling-lab\roofline_fp32_sgemm_full.ncu-rep"

# 查看方式 2：命令行直接打印全部内容（--import 不加 --section 就是全量输出，无需管理员）
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --import "C:\profiling-lab\roofline_fp32_sgemm_full.ncu-rep"

# 查看方式 3：命令行只看某几个 section（用 --section 指定，多个 section 需分次执行）
#   3a) 关键：GPU Speed Of Light（算力/带宽/DRAM 吞吐百分比，判断瓶颈类型）
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --import "C:\profiling-lab\roofline_fp32_sgemm_full.ncu-rep" --section SpeedOfLight

#   3b) Roofline 结论（报告里给出「达到 FP32 峰值的 %」，如 "70% of FP32 peak"）
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --import "C:\profiling-lab\roofline_fp32_sgemm_full.ncu-rep" --section SpeedOfLight_RooflineChart

#   3c) FLOPs（算 AI 分子）与 Bytes（算 AI 分母），手动算 AI = FLOPs ÷ Bytes
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --import "C:\profiling-lab\roofline_fp32_sgemm_full.ncu-rep" --section ComputeWorkloadAnalysis
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --import "C:\profiling-lab\roofline_fp32_sgemm_full.ncu-rep" --section MemoryWorkloadAnalysis

#   3d) Occupancy（看 warp 是否填满 SM，低则找寄存器/共享内存超限原因）
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --import "C:\profiling-lab\roofline_fp32_sgemm_full.ncu-rep" --section Occupancy

# 列出所有可用 section 名（确认正确的名字，如 SpeedOfLight_RooflineChart，避免用错名报错）
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --list-sections
```

> **文件怎么看**：
> - `nsys` 采集生成 `.nsys-rep`（主报告）+ `.sqlite`（数据库）。日常看热点：查看方式 2a（`cuda_gpu_kern_sum`）看哪个 kernel 最耗时；想直观看时间线用查看方式 1（GUI）。
> - `ncu` 采集生成 `.ncu-rep`。日常看瓶颈类型：查看方式 3a（`--section SpeedOfLight`）看 Compute/Memory 哪个接近 100%；想直观看全部用查看方式 2 或 1。
> - **想看 Roofline（AI + roof 达成率）**：必须先 **`--set full`** 采集（`basic` 没存 FLOP 指标，算不了 AI），再用查看方式 3b（`--section SpeedOfLight_RooflineChart`）。

> 工具安装路径若不同，把 `D:\Program Files\...` 换成你自己的实际路径（`nsys --version` / `ncu --version` 的输出会告诉你）。

### 解读报告（跑完快速命令后怎么看）

**先看 A 的输出（roofline_bench.py 的达成率表）**——它是「结果」，后面两个是「原因」：

```
算子           算术强度          实测值          理论峰值      达成率  判定
memcpy            0.0      xxx.x GB/s      896.0 GB/s     xx.x%  memory-bound
fp32_matmul     682.7      xx.xx TFLOPS     55.4 TFLOPS    xx.x%  compute-bound
fp16_matmul    1365.3     xxx.xx TFLOPS    110.0 TFLOPS    xx.x%  compute-bound
fp32_reduce       0.2      xxx.x GB/s      896.0 GB/s     xx.x%  memory-bound
```

- 达成率 **>80%**：算子跑得不错，不用深挖；
- **某个算子明显低**（比如 memcpy 只有 30%）：这就是 profiling 的切入点，去查为什么。

**再看 B'（nsys kernel 耗时排行）**——回答「哪个 kernel 最耗时」：

```
Time (%)  Total Time  Instances  Avg  Name
  65.2    23.4 ms     1          23.4ms  cutlass_80_...sgemm_256x128  ← 热点在这里
   ...
```

- 按 `Time (%)` 从大到小看，**第一行就是热点 kernel**；
- 想直观看 GPU 忙不忙：`nsys-ui C:\profiling-lab\roofline_nsys.nsys-rep`（实际路径 `D:\Program Files\host-windows-x64\nsys-ui.exe`），看时间线顶部 GPU 活动条，空白 = idle；
- 详细概念见 §3.3。

**最后看 C'（ncu Speed Of Light）**——回答「热点 kernel 为什么慢，是算力还是访存瓶颈」：

命令行 `ncu --import ...` 输出里重点看这一段：

```
GPU Speed Of Light Throughput:
    Compute (SM) Throughput : 60.1%
    Memory Throughput       : 12.4%
```

- **Compute (SM) % 高（接近 100）** → 算力瓶颈，SM 算力快用满了，是「算力屋顶」限制；
- **Memory % 高（接近 100）** → 访存瓶颈，被带宽卡住；
- **两个都低**（比如都 40%）→ 有「空闲」，常见原因：Occupancy 不足（warp 没填满 SM）、kernel 太短有启动/收尾开销、launch 间隙；
- 想深入看 Occupancy：`ncu-ui C:\profiling-lab\roofline_fp32_ncu.ncu-rep`（实际路径 `D:\Program Files\host\windows-desktop-win7-x64\ncu-ui.exe`）打开 GUI 看详细 section；
- 详细概念见 §4.3。

> **一句话闭环**：A 告诉你「离屋顶差百分之几」，B 告诉你「哪个 kernel 最该查」，C 告诉你「那个 kernel 差在哪」。详细原理见 §3 和 §4。

---

## 1. 安装 nsys / ncu

### 1.1 下载安装包

去 NVIDIA 官网下载 **Windows 版**安装包（`.exe`）：

- Nsight Systems：https://developer.nvidia.com/nsight-systems
- Nsight Compute：https://developer.nvidia.com/nsight-compute

> 两个都要下载。它们包含两部分：
> - **GUI**（`nsys-ui.exe` / `ncu-ui.exe`）：图形界面，看时间线/报告用
> - **CLI**（`nsys.exe` / `ncu.exe`）：命令行，本教程主要用这个

### 1.2 安装

双击 `.exe`，一路 Next 默认安装即可。默认安装路径：

```
C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.x\
C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.x\
```

### 1.3 验证

打开 **PowerShell**，确认能调用到（若提示找不到命令，用完整路径，或把安装目录加到系统 PATH）：

```powershell
nsys --version
ncu --version
```

> 如果 `nsys` / `ncu` 命令找不到，用完整路径测试，例如：
> ```powershell
> & "C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3\target-windows-x64\nsys.exe" --version
> ```
>
> **本机实际路径速查**（CLI 与 GUI 所在目录不同，容易搞混）：
>
> | 工具 | 实际路径 |
> |------|---------|
> | `nsys.exe`（命令行） | `D:\Program Files\host\target-windows-x64\nsys.exe` |
> | `nsys-ui.exe`（GUI） | `D:\Program Files\host-windows-x64\nsys-ui.exe` |
> | `ncu.exe`（命令行） | `D:\Program Files\target\windows-desktop-win7-x64\ncu.exe` |
> | `ncu-ui.exe`（GUI） | `D:\Program Files\host\windows-desktop-win7-x64\ncu-ui.exe` |

---

## 2. 准备一个「被分析对象」（Roofline 基准脚本）

nsys/ncu 分析的是「跑在 GPU 上的 CUDA 程序」。我们用 **PyTorch（Windows + CUDA 版）** 写一个
基于 **Roofline 模型**的基准脚本，这样不仅能观察 kernel，还能量化「每个算子跑到硬件理论峰值的百分之几」。

> **为什么要用 Roofline**：Roofline 模型用「算术强度（AI = FLOPs / Bytes）」判断一个算子受什么限制：
> - AI 低（每个字节访存算不了几下）→ **memory-bound**，性能受显存带宽限制；
> - AI 高 → **compute-bound**，性能受算力限制；
> - 拐点（ridge point）= 峰值算力 ÷ 带宽。
>
> 用 Roofline 设计脚本，就能**主动构造**不同算术强度的算子，分别去撞「带宽屋顶」和「算力屋顶」，
> 再配合 nsys/ncu 验证「实测值离理论峰值差多少」，比随便跑个 matmul 有信息量得多。

### 2.1 装 PyTorch（Windows + CUDA 版）

```powershell
# 建议用独立的 venv，避免污染系统 Python
python -m venv C:\profiling-lab
C:\profiling-lab\Scripts\activate

# 安装 CUDA 版 PyTorch（按你的 CUDA 版本选 cu126 / cu128）
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 2.2 拿测试脚本

脚本已在项目里：`benchmarks/e2e_cases/roofline_bench.py`。把它复制到 Windows 侧（例如
`C:\profiling-lab\roofline_bench.py`），或用下面的命令直接下载/复制。

> 该脚本针对 **RTX 5070 Ti** 内置了理论峰值（FP32 ≈ 55.4 TFLOPS、FP16 ≈ 110 TFLOPS、
> 带宽 ≈ 896 GB/s），并自动打印「实测值 vs 理论峰值」的达成率。

### 2.3 先跑一遍，看 Roofline 达成情况

```powershell
python C:\profiling-lab\roofline_bench.py
# 或只跑某一项：--kind memcpy / fp32 / fp16 / reduce
```

预期输出类似：

```
Roofline 模型上下文 — RTX 5070 Ti (Blackwell, sm_120)
  理论 FP32 峰值算力 : 55.4 TFLOPS
  理论 FP16 峰值算力 : 110.0 TFLOPS (Tensor Core)
  理论显存带宽       : 896.0 GB/s
  Ridge Point (FP32) : 61.8 FLOP/Byte

算子           算术强度          实测值          理论峰值      达成率  判定
------------------------------------------------------------------------
memcpy            0.0      xxx.x GB/s      896.0 GB/s     xx.x%  memory-bound
fp32_matmul     682.7      xx.xx TFLOPS     55.4 TFLOPS    xx.x%  compute-bound
fp16_matmul    1365.3     xxx.xx TFLOPS    110.0 TFLOPS    xx.x%  compute-bound
fp32_reduce       0.2      xxx.x GB/s      896.0 GB/s     xx.x%  memory-bound
```

### 2.4 Roofline 达成率怎么解读

| 算子 | 算术强度 | 目标屋顶 | 达成率的含义 |
|------|---------|---------|-------------|
| `memcpy` | ≈ 0 | 带宽屋顶（896 GB/s） | 达成率高（>80%）→ 访存路径没被拖慢 |
| `fp32_reduce` | ≈ 0.25 | 带宽屋顶 | 与 memcpy 对比，看加一点算力后带宽是否掉 |
| `fp32_matmul` | ≈ 683 | FP32 算力屋顶（55.4 T） | 达成率高 → GEMM 吃满 SM 算力 |
| `fp16_matmul` | ≈ 1365 | FP16 Tensor Core 屋顶（110 T） | 达成率高 → Tensor Core 正常启用 |

> **本教程的核心目标**：能回答「每个算子跑到对应 roof 的百分之几」。
> 如果某个算子达成率异常低（比如 memcpy 只有 30%），说明有非模型因素（驱动、频率、脚本 bug），
> 值得用 nsys/ncu 深入查——这就是 profiling 的切入点。

### 2.5 重要：为什么一定要在 Windows 原生（而非 WSL）跑这个脚本

这个脚本**必须在 Windows 原生环境跑**，不要在 WSL2 里跑。原因是有实测数据支撑的：

| 环境 | memcpy 实测带宽 | fp32 GEMM 实测算力 | 说明 |
|------|----------------|-------------------|------|
| **WSL2（dxg 透传）** | 214 GB/s（理论 896） | 7.3 TFLOPS（理论 55.4） | 只有理论峰值 ~24%，转译层损耗巨大 |
| **Windows 原生**（本教程环境） | 应接近 896 GB/s | 应接近 55.4 TFLOPS | 原生驱动，无转译损耗 |

> WSL2 里每个 CUDA 操作都要经 dxg 层转发给 Windows 驱动，这个转发开销在访存/计算密集型
> 操作上被放大，导致实测只有理论峰值的 ~24%。这和之前 `ERR_NVGPUCTRPERM`（profiling 采不到）
> 是**同一个根源的两个表现**。
>
> **结论**：Roofline 基准的意义在于「测真实的硬件峰值」，只有 Windows 原生（或裸 Linux）才能测准。
> 在 WSL2 里跑出来的低达成率是**环境损耗**，不是 GPU 的真实水平，会误导你的判断。

---

## 3. 用 nsys 分析（宏观：时间线 + GPU 利用率）

### 3.1 采集

```powershell
nsys profile `
    --trace=cuda,nvtx,osrt `
    --output=C:\profiling-lab\roofline_nsys `
    --force-overwrite=true `
    --stats=true `
    python C:\profiling-lab\roofline_bench.py
```

> PowerShell 里换行用反引号 `` ` ``，也可以写成一行。

采集完成后会生成 `roofline_nsys.nsys-rep` 和 `roofline_nsys.sqlite`，并直接在终端打印统计摘要。

> **Roofline 视角的采集建议**：全跑一次（4 个算子），时间线里会依次出现
> memcpy → fp32 GEMM → fp16 GEMM → reduce 四段。你可以分别看每段的 GPU 活动，
> 对照 `roofline_bench.py` 打印的达成率，观察「达成率高的算子 GPU 是否更忙」。

### 3.2 看报告（两种方式）

**方式 A：图形界面（推荐，直观）**

```powershell
& "D:\Program Files\host-windows-x64\nsys-ui.exe" C:\profiling-lab\roofline_nsys.nsys-rep
```

> `nsys-ui.exe` 在 `host-windows-x64\` 目录（与 `nsys.exe` 不在同一目录），本机实际路径见上。
> 若 GUI 打开报告立即退出/崩溃，用「方式 B」命令行替代（功能不受影响）。

会弹出 Nsight Systems GUI，时间线里能看到：
- 顶部：GPU 各 kernel 的执行条（看 GPU 忙不忙、有没有 idle 间隙）
- 中间：CPU 线程的 CUDA API 调用
- 底部：kernel 名、耗时

**方式 B：命令行导出摘要（无 GUI 时）**

```powershell
# GPU kernel 耗时排行（重点看这个）
nsys stats --report cuda_gpu_kern_sum C:\profiling-lab\roofline_nsys.nsys-rep

# CUDA API 调用耗时
nsys stats --report cuda_api_sum C:\profiling-lab\roofline_nsys.nsys-rep
```

### 3.3 怎么看懂（nsys 核心概念）

| 要看什么 | 在报告哪里 | 说明 |
|---------|-----------|------|
| **GPU 利用率** | 时间线顶部 GPU 活动条 | 条越满 = GPU 越忙；空白 = idle（空闲） |
| **哪个 kernel 最耗时** | `cuda_gpu_kern_sum` 报告，按 `Time (%)` 排序 | 排第一的往往是热点 |
| **kernel 调了多少次** | `cuda_gpu_kern_sum` 的 `Instances` 列 | 本例 fp32/fp16 GEMM 各调一次，但单次耗时大 |
| **CPU 侧在干嘛** | `cuda_api_sum` 报告 | 看 `cudaLaunchKernel`、`cudaMemcpy` 等 API 的耗时 |

> **Roofline 视角的 nsys 解读**：
> - `memcpy` / `reduce` 是 memory-bound：时间线里 GPU 计算条短、但能看到显存读写活动密集；
> - `fp32_matmul` / `fp16_matmul` 是 compute-bound：GPU 计算条长且连续，几乎无 idle；
> - 对照 `roofline_bench.py` 的达成率，达成率高的算子，其对应的 GPU 活动应该更「满」。

> **入门目标**：能从 `cuda_gpu_kern_sum` 里读出「哪个 kernel 最耗时」，能在时间线上看到「GPU 有没有空闲」，就算 nsys 入门了。

---

## 4. 用 ncu 分析（微观：单 kernel 的 SM 占用率 / 访存）

### 4.1 采集（只 profile 一个 kernel）

ncu 开销大，通常只针对**一个 kernel**分析。用 `--kernel-name` 过滤。

> **Roofline 视角的选择**：优先用 ncu 分析 **fp32_matmul 或 fp16_matmul** 这两个
> compute-bound 算子——它们最接近「算力屋顶」，ncu 能告诉你为什么达成率到不了 100%
> （是 SM 没填满？还是访存拖累？）。

```powershell
# 只跑 fp32 GEMM，并用 ncu 分析（PyTorch matmul 走 cutlass 的 sgemm kernel）
ncu `
    --launch-skip 1 `
    --launch-count 1 `
    --set full `
    --export C:\profiling-lab\roofline_fp32_ncu `
    python C:\profiling-lab\roofline_bench.py --kind fp32
```

> - `--launch-skip 1`：跳过前 1 个 kernel launch。`roofline_bench.py --kind fp32`
>   的第一个 kernel 是 `torch.randn` 的随机数生成 kernel，第二个起才是 fp32 GEMM。
>   跳过它就能精准采到 GEMM。
>   **不要用 `--kernel-name regex:gemm`**：在 Windows 上 ncu 的 `--kernel-name`
>   过滤经常失效——ncu 按 demangled 的 C++ 模板名匹配，与 nsys 显示的简化名
>   （如 `cutlass_80_simt_sgemm_256x128_8x4`）不一致，`regex:gemm` 会匹配不到，
>   报 `No kernels were profiled`。改用 `--launch-skip` 按 launch 顺序定位最可靠。
> - `--launch-count 1`：只采集 1 次 launch（ncu 会重放多次取统计，1 次 launch 已能出报告）
> - `--set full`：完整指标集。嫌慢可换 `basic`
> - `--kind fp32`：只跑 fp32 GEMM，让 ncu 聚焦，避免把 memcpy/reduce 也采进去

### 4.2 看报告

```powershell
& "D:\Program Files\host\windows-desktop-win7-x64\ncu-ui.exe" C:\profiling-lab\roofline_fp32_ncu.ncu-rep
```

> `ncu-ui.exe` 在 `host\windows-desktop-win7-x64\` 目录（与 `ncu.exe` 的 `target\` 目录不同），本机实际路径见上。
> 若 GUI 打开报告立即退出/崩溃，用下方命令行替代（功能不受影响）。

或命令行直接打印关键指标：

```powershell
& "D:\Program Files\target\windows-desktop-win7-x64\ncu.exe" --import C:\profiling-lab\roofline_fp32_ncu.ncu-rep
```

### 4.3 怎么看懂（ncu 核心概念）

ncu 报告里有几个 section，重点是前三个：

| Section | 关键指标 | 怎么判断瓶颈 |
|---------|---------|-------------|
| **GPU Speed Of Light** | Compute (SM) Throughput / Memory Throughput | SM% 高→算力瓶颈；Memory% 高→访存瓶颈 |
| **Occupancy** | Achieved vs Theoretical Occupancy | 低→warp 没填满 SM（寄存器/共享内存超限） |
| **Warp State Statistics** | Stall Long Scoreboard / Stall Wait 等 | 看 warp 主要卡在什么（访存延迟 / 同步 / 算力） |
| **Memory Workload Analysis** | 各内存吞吐、cache 命中率 | 访存瓶颈时定位是全局内存还是 L2 |

> **用 ncu 验证 Roofline 结论**：
> 1. `fp32_matmul`（AI≈683，理论上 compute-bound）→ ncu 的 `GPU Speed Of Light` 里
>    **Compute (SM) Throughput 应接近 100%**，Memory Throughput 较低；
> 2. 如果反过来 Memory Throughput 更高，说明实际访存比预想多（可能矩阵布局不佳），
>    这解释了为什么 `roofline_bench.py` 里 fp32 达成率上不去；
> 3. 同理，想看 memory-bound 的算子，可用 ncu 分析 `memcpy`，看 Memory Throughput 是否接近 100%。

> **入门目标**：能说出「这个 kernel 是算力瓶颈还是访存瓶颈」——看 `GPU Speed Of Light` 里
> SM Throughput 和 Memory Throughput 哪个更接近 100%。

---

## 5. nsys 和 ncu 的配合套路

```
第 0 步：roofline_bench.py 打印各算子达成率（先看「谁没到 roof」）
第 1 步：nsys 全采 → 找到「哪个 kernel 最耗时」（宏观定位）
第 2 步：ncu 单采那个 kernel → 分析「它为什么没到 roof」（微观归因）
```

**例子**：跑完 `roofline_bench.py` 后：
1. 发现 `fp32_matmul` 达成率只有 60%（没到 55.4 TFLOPS 屋顶）
2. nsys 的 `cuda_gpu_kern_sum` 确认 GEMM kernel 占主导耗时
3. 用 ncu `--launch-skip 1 --launch-count 1` 单采 GEMM kernel（跳过 randn kernel，见 §4.1）
4. ncu 报告显示 **Compute (SM) Throughput 只有 60%**、Occupancy 偏低 → 结论：
   这个 GEMM 没吃满 SM 算力，可能是矩阵尺寸/block 配置没让 SM 满负荷，而非访存瓶颈

> **Roofline + profiling 的闭环**：Roofline 告诉你「离屋顶差多少」，nsys/ncu 告诉你「为什么差」。
> 两者结合，才能从「现象（达成率低）」走到「根因（SM 没填满 / 访存拖累）」。

---

## 6. 常见坑

| 现象 | 原因 | 解决 |
|------|------|------|
| `ncu` 报 `ERR_NVGPUCTRPERM` | 没以管理员运行，或驱动限制 | 用管理员 PowerShell 运行；或改注册表 `RestrictProfilingToAdminUsers=0`（见下） |
| nsys 报告里 GPU kernel 为空 | 没加 `--trace=cuda`，或程序根本没跑 CUDA | 确认脚本真的在 GPU 上跑了（`torch.cuda.is_available()`） |
| `nsys` / `ncu` 命令找不到 | 安装目录没进 PATH | 用完整路径，或手动加 PATH |
| ncu 采集很慢 | `--set full` + 多个 kernel | 加 `--launch-skip` 过滤 + `--launch-count 1` |
| `ncu --kernel-name regex:gemm` 报 `No kernels were profiled` | Windows 上 ncu 按 demangled 名匹配 kernel，与 nsys 简化名不一致 | 改用 `--launch-skip 1` 跳过 randn kernel 直接采 GEMM（见 §4.1） |
| 时间线里 GPU 全空 | 采集窗口没覆盖到计算阶段 | 确认 `--capture-range` 没配错（本教程没配，全程采） |
| `nsys-ui` / `ncu-ui` 打开报告「闪退」，但 `Get-Process` 能看到进程在跑 | GUI 是**单实例**程序：后台已有旧实例时，新启动的实例检测到就**正常退出（exit 0）**，报告被转交给后台实例加载，窗口稍后才弹出，看起来像闪退 | 先看任务栏有没有已存在的 Nsight 窗口；或先 `Stop-Process -Name nsys-ui -Force`（ncu-ui 同理）清掉旧实例，再重新启动 |

### 关于 `ERR_NVGPUCTRPERM`（Windows 原生下的解法）

在 Windows **原生**环境（非 WSL）下，这个错误通常是**权限问题**，解法有效：

```powershell
# 管理员 PowerShell 执行：允许所有用户访问 GPU 性能计数器
reg add "HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak" `
    /v "RestrictProfilingToAdminUsers" /t REG_DWORD /d 0 /f
# 重启后生效
```

> 注意：这个解法在 Windows 原生环境**有效**；但之前在 WSL2 里试过**无效**，
> 因为 WSL2 的 dxg 透传根本没暴露 NVPC 接口（详见 `SCHEDULING_BENCHMARK_GUIDE.md` §8.6）。

---

## 7. 学完之后怎么迁移到 Linux（分析 vLLM）

在 Windows 上学会的**命令和报告解读能力是通用的**，迁移到裸 Linux 只需：

1. 命令里把 Windows 路径换成 Linux 路径，`python` 换成 `.venv/bin/python`
2. 被分析对象从 `bench.py` 换成 vLLM 服务端进程
3. 具体命令见 `SCHEDULING_BENCHMARK_GUIDE.md` §8.3（nsys）/ §8.4（ncu）

> 核心差异只有一点：**Linux 上 profile vLLM 要 profile「服务端进程」，不是客户端**。
> 其余 `--trace=cuda`、`--kernel-name`、`--set full` 等参数完全一样。

---

## 8. 本教程的「最小验证清单」

完成以下 6 步，就算学完 nsys/ncu 入门：

- [ ] `nsys --version` 和 `ncu --version` 能输出版本号
- [ ] `roofline_bench.py` 能在 GPU 上跑通，打印出 4 个算子的达成率表
- [ ] 能回答：「memcpy / fp32_matmul / fp16_matmul / reduce 各自的算术强度是多少？分别命中哪个屋顶？」
- [ ] `nsys profile ...` 能生成 `.nsys-rep`，且 `cuda_gpu_kern_sum` 报告能看到 GEMM kernel
- [ ] `ncu --launch-skip 1 ...` 能生成 `.ncu-rep`，且能看到 `GPU Speed Of Light` section
- [ ] 能回答：「fp32_matmul 里最耗时的 kernel 是什么？它是算力瓶颈还是访存瓶颈？离算力屋顶差多少？」

---

## 附：Roofline 模型速记

```
                算力屋顶 (55.4 TFLOPS fp32 / 110 TFLOPS fp16)
  性能           ╱────────────────────────────
  (FLOP/s)      ╱
               ╱ ← 斜率为带宽的线（AI × 带宽）
              ╱
  显存带宽 ───╱──────────────────────────────
  屋顶      ╱
           ╱
          ─┴──────────────────────────────────
           拐点(ridge)              算术强度 (FLOP/Byte)

  - 在拐点左侧（低 AI）：memory-bound，性能 = AI × 带宽
  - 在拐点右侧（高 AI）：compute-bound，性能 = 峰值算力
  - 拐点 = 峰值算力 / 带宽
```

| 概念 | 公式 | RTX 5070 Ti 数值 |
|------|------|-----------------|
| FP32 峰值算力 | cores × 2 FLOP × clock | 8960 × 2 × 3.09 GHz ≈ 55.4 TFLOPS |
| FP16 峰值算力 | Tensor Core，≈ 2× FP32 | ≈ 110 TFLOPS |
| 显存带宽 | GDDR7 16GB | ≈ 896 GB/s |
| 拐点（FP32） | 算力 / 带宽 | ≈ 61.8 FLOP/Byte |
