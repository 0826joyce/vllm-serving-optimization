# 在 RTX 5070 Ti（Ubuntu 虚拟机）上拉起并验证本项目的优化

> 目标读者：拿到一台带 **RTX 5070 Ti** 的 Ubuntu 虚拟机，想把本仓库（基于 vLLM V1 二次开发）里的优化跑起来，
> 实际测一下 `benchmarks/basic_optimization` 下三类优化的效果。
>
> 本文覆盖：**环境初始化 → 编译/安装本框架 → 把它当推理服务拉起来 → 跑各项优化 → 评估对比**。

---

## 0. 先认清你要测的是什么

`benchmarks/basic_optimization/` 下是 **设计文档**，不是脚本。真正的代码改动已经落在 `vllm/` 源码里。三类优化的接入方式不同，**这决定了你"怎么开启"它们**：

| 优化文档 | 代码落点 | 如何启用 | 状态 |
|---|---|---|---|
| `prefix-cache-scheduling-optimization.md`（缓存感知调度 + 频率感知驱逐 + 抢占缓存保护） | `vllm/v1/core/scheduler.py`、`kv_cache_manager.py`、`kv_cache_utils.py` | **随 `--enable-prefix-caching` 自动生效**（代码里绑定了 `enable_prefix_caching`） | ✅ 已实现 |
| `suffix-decoding-optimization.md`（后缀树 / 增量 SAM / 自适应匹配 Proposer） | `vllm/v1/spec_decode/suffix_proposer.py`、`suffix_automaton_proposer.py`、`adaptive_suffix_proposer.py`，接入点 `vllm/v1/worker/gpu_model_runner.py` | **环境变量 `VLLM_SPEC_PROPOSER` + `--speculative-config`** 选择 proposer | ✅ 已实现 |
| `pd-disaggregation-optimization.md`（PD 分离） | 设计为主，V1 接入未完成 | 暂不在单卡复现范围 | ⬜ 设计态 |

> 结论：在单张 5070 Ti 上，**可以完整复现「调度+KV Cache」和「后缀投机解码」两类优化**；PD 分离不在单卡复现范围内（需要至少 2 张卡，且 V1 接入是设计态）。

---

## 1. RTX 5070 Ti 的关键前提（务必先读）

5070 Ti 是 **Blackwell 架构（计算能力 sm_120）**，这点直接决定软件栈版本：

| 项目 | 要求 | 说明 |
|---|---|---|
| 显存 | **16 GB GDDR7** | 跑 1.5B/3B 模型 + 本项目 workload 足够；7B 需缩 `--max-model-len` |
| NVIDIA 驱动 | **≥ 570**（建议 575+） | Blackwell 必须用新驱动，老驱动认不出这张卡 |
| CUDA | **≥ 12.8** | sm_120 从 CUDA 12.8 起才被支持；CUDA 12.4 编不出 5070 Ti 的 kernel |
| PyTorch | **cu128 构建**（torch ≥ 2.7） | 必须装带 sm_120 的 wheel，否则报 `no kernel image is available` |
| vLLM | **本仓库源码**（基于 v0.7.3 改） | 注意：v0.7.3 默认对 Blackwell 支持不全，下面给规避方案 |

> ⚠️ 最大的坑：本仓库基线是 **vLLM v0.7.3**，那个时间点对 Blackwell（sm_120）支持很弱。
> 强烈建议**用 PyTorch 自带的 cu128 + `VLLM_USE_PRECOMPILED` 跳过本地编译 C++/CUDA**（见 §3 方案 A），
> 只让你的 **Python 层改动（scheduler / spec_decode / kv_cache_manager 全是纯 Python）** 生效——
> 而本项目所有已实现优化恰好都在 Python 层，**不需要重新编译 CUDA 算子**。

---

## 2. 初始化 Ubuntu 虚拟机环境

### 2.1 确认 GPU 直通与驱动

```bash
# 1) 确认虚拟机能看到 5070 Ti（PCI 直通必须已配好；纯虚拟显卡跑不了 CUDA）
lspci | grep -i nvidia

# 2) 确认驱动已装且版本 ≥ 570
nvidia-smi
#   关注 Driver Version 和 CUDA Version 两列，CUDA Version 应 ≥ 12.8
```

如果 `nvidia-smi` 不可用，先装驱动：

```bash
sudo apt update
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices                 # 查看推荐驱动
sudo ubuntu-drivers autoinstall        # 或指定: sudo apt install nvidia-driver-575
sudo reboot
```

> 虚拟机要跑 CUDA，**宿主机必须把物理 GPU 以 PCI passthrough/vGPU 方式直通给虚拟机**。普通的虚拟显示适配器无法运行 CUDA。

### 2.2 系统依赖

```bash
sudo apt update
sudo apt install -y build-essential git curl ca-certificates python3-dev
```

### 2.3 安装 uv（项目 AGENTS.md 指定的 Python 环境管理器）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env          # 让当前 shell 识别 uv

uv venv --python 3.12
source .venv/bin/activate
```

### 2.4 安装匹配 Blackwell 的 PyTorch（cu128）

```bash
# 必须是 cu128 构建，带 sm_120 kernel
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 验证能识别到 5070 Ti
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
#   期望输出类似: 2.7.x+cu128 True NVIDIA GeForce RTX 5070 Ti
```

如果 `torch.cuda.is_available()` 为 False 或报 `no kernel image`，说明 torch/CUDA/驱动版本不匹配，回到 §2.1 / §2.4 重新对齐版本。

---

## 3. 安装本框架（关键步骤）

进入仓库根目录：

```bash
cd /data/home/nikiijiang/expolore/serving-optimization
source .venv/bin/activate
```

### 方案 A（强烈推荐）：用预编译内核，跳过本地 CUDA 编译

本项目**所有已实现优化都是纯 Python**（scheduler、kv_cache、spec_decode proposer），不碰 CUDA 算子，所以没必要本地编译：

```bash
# 只做 Python 改动时：复用官方预编译 wheel 里的 C++/CUDA，安装当前源码为可编辑包
VLLM_USE_PRECOMPILED=1 uv pip install -e . --no-build-isolation
```

- 这样会把 `vllm/v1/core/scheduler.py` 等你的 Python 改动以 editable 模式装入环境，**改完即生效**。
- 预编译内核来自上游对应版本；若提示找不到匹配 commit 的 wheel，改用方案 B。

### 方案 B（兜底）：本地全量编译

仅当方案 A 因 Blackwell/版本问题失败时使用，耗时较长（10–40 分钟）：

```bash
# 让编译器为 5070 Ti（sm_120）生成 kernel
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=$(nproc)

uv pip install -e . --no-build-isolation
```

> 若编译报 sm_120 相关错误，说明本地 CUDA toolkit < 12.8，需要先升级 CUDA toolkit 到 12.8+ 再编。能用方案 A 就别走 B。

### 3.1 装测试依赖（用于跑单元测试验证优化逻辑）

```bash
uv pip install pytest pytest-asyncio tblib
# 跑 workload 压测脚本还需要：
uv pip install numpy aiohttp openai
```

### 3.2 安装后自检

```bash
python -c "import vllm; print('vllm import OK:', vllm.__file__)"
```

---

## 4. 把本框架当推理服务拉起来

下面用 **Qwen2.5-1.5B-Instruct** 做示例（16 GB 显存友好；首次会自动从 HuggingFace 下载，可设国内镜像）。

```bash
# 可选：国内加速
export HF_ENDPOINT=https://hf-mirror.com
```

### 4.1 基线服务（不带任何本项目优化的"对照组"）

为了做 A/B 对比，先起一个**关闭 prefix caching、用原生 ngram proposer** 的对照：

```bash
# 终端 1：基线（对照组）
VLLM_SPEC_PROPOSER=ngram \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

> 注意：不加 `--enable-prefix-caching` 时，缓存感知调度/抢占保护等优化**不会生效**，这正是对照组想要的。

起来后另开终端验证服务通了：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct",
       "messages":[{"role":"user","content":"用一句话介绍你自己"}],
       "max_tokens":50}'
```

### 4.2 启用「调度 + KV Cache」优化的服务

```bash
# 终端 1：实验组（开启 prefix caching → 自动启用缓存感知调度 + Segmented LRU + 抢占缓存保护）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

### 4.3 启用「后缀投机解码」优化的服务

通过环境变量 `VLLM_SPEC_PROPOSER` 选择 proposer（`ngram` / `suffix` / `suffix_automaton` / `adaptive`）：

```bash
# 终端 1：用自适应后缀 proposer（多候选评分，接受率最高）
VLLM_SPEC_PROPOSER=adaptive \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"ngram_prompt_lookup_min":3,"ngram_prompt_lookup_max":5}' \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

> - `--speculative-config` 必须带 ngram 字段（代码里 `assert ngram_prompt_lookup_min`），proposer 的具体实现由 `VLLM_SPEC_PROPOSER` 决定。
> - 想跑对照：`VLLM_SPEC_PROPOSER=ngram`（原生）↔ `=suffix_automaton`（增量 SAM）↔ `=adaptive`（自适应评分）。

---

## 5. 怎么跑这些优化（两条路线）

### 路线一：单元测试 —— 验证优化逻辑正确性（不需要 GPU 也能跑）

每个优化都有对应测试，直接确认实现无误：

```bash
cd /data/home/nikiijiang/expolore/serving-optimization
source .venv/bin/activate

# 调度 + KV Cache 三个优化
pytest tests/v1/core/test_cache_aware_scheduling.py     -v
pytest tests/v1/core/test_frequency_aware_eviction.py   -v
pytest tests/v1/core/test_preemption_cache_shield.py    -v
pytest tests/v1/core/test_cache_version_management.py   -v

# 后缀投机解码三个 proposer
pytest tests/v1/spec_decode/test_suffix_proposer.py            -v
pytest tests/v1/spec_decode/test_suffix_automaton_proposer.py  -v
pytest tests/v1/spec_decode/test_adaptive_suffix_proposer.py   -v
```

测试全绿 = 优化逻辑实现正确。这一步在虚拟机 CPU 上即可完成，建议**先跑这步确认代码 OK，再上 GPU 压测**。

### 路线二：端到端压测 —— 验证优化的实际效果（需要 GPU）

仓库自带的多租户压测脚本在 `benchmarks/e2e_cases/workload.py`，它会自动造 5 个 Phase 的流量并采集 TTFT / SLA 违约率：

```bash
# 终端 2：对着 §4 起好的服务打流量（脚本自带 prompt，无需你输入请求内容）
python benchmarks/e2e_cases/workload.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --duration 300 \
    --output-dir results/exp
```

> 脚本本身只是 HTTP 客户端、不吃显存；它会输出按 Phase × 租户的 P50/P95/P99 TTFT 和违约率，并落盘 JSON 到 `--output-dir`。

---

## 6. 怎么评估这些优化（A/B 对比方法）

核心思路：**控制变量，只改一个开关，跑同一份 workload，对比同一组指标。**

### 6.1 「调度 + KV Cache」优化评估

| 组别 | 服务启动参数 | 输出目录 |
|---|---|---|
| 对照组 | §4.1（**不加** `--enable-prefix-caching`） | `results/baseline` |
| 实验组 | §4.2（加 `--enable-prefix-caching`） | `results/prefix_opt` |

```bash
# 分别对两组服务各跑一次，注意切换 --output-dir
python benchmarks/e2e_cases/workload.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 --port 8000 --duration 300 --output-dir results/baseline
# （重启服务为实验组后）
python benchmarks/e2e_cases/workload.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 --port 8000 --duration 300 --output-dir results/prefix_opt
```

**关注指标**（脚本会直接打印，也在 JSON 里）：
- **P99 TTFT**：高缓存命中请求是否变快（优化点 1，预期 ↓30–50%）
- **Phase 3 Silver 租户 TTFT 退化幅度**：租户隔离效果
- **SLA 违约率**：Phase 2/4/5 下违约数是否下降
- vLLM 服务端日志里的 **prefix cache hit rate**：频率感知驱逐让命中率更稳（优化点 2，预期 ~85–95%）

### 6.2 「后缀投机解码」优化评估

固定服务其它参数，只切 `VLLM_SPEC_PROPOSER`，各跑一次：

| 组别 | 环境变量 | 输出目录 |
|---|---|---|
| 原生 N-gram | `VLLM_SPEC_PROPOSER=ngram` | `results/spec_ngram` |
| 增量 SAM | `VLLM_SPEC_PROPOSER=suffix_automaton` | `results/spec_sam` |
| 自适应 | `VLLM_SPEC_PROPOSER=adaptive` | `results/spec_adaptive` |

**关注指标**：
- **Draft 接受率**（accepted / proposed）：自适应预期比 ngram 高 15–30%
- **每步有效 token 数 / 平均接受长度**：越大说明 decode 越快
- **Gold-B 代码补全场景 E2E 延迟**：重复性高的场景后缀解码收益最明显
- **TTFT 不应明显劣化**：proposer 在关键路径上，确认没引入额外延迟

### 6.3 评估产物

每次跑完，`--output-dir` 下会生成 `results_single.json`，每条记录含 `ttft_ms` / `e2e_ms` / `sla_violated` / `phase` / `tenant` 等字段。把不同组的同名指标列成对比表即可（参考 `e2e_cases/LANDING_PLAN.md` 里的对比表格式）。

---

## 7. 显存与参数速查（5070 Ti / 16 GB）

| 模型 | 推荐 `--max-model-len` | 估算占用 | 备注 |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 8192 | ~6–8 GB | 最稳，迭代最快 |
| **Qwen2.5-1.5B-Instruct** | 8192 | ~10–12 GB | **推荐默认** |
| Qwen2.5-3B-Instruct | 4096 | ~13–15 GB | 需调小 `--max-model-len` |
| 7B/8B | ❌ | 16 GB 装不下推理 + 充足 KV | 不建议在 5070 Ti 上跑 |

通用降显存手段：调低 `--gpu-memory-utilization`（如 0.8）、调小 `--max-model-len` 与 `--max-num-batched-tokens`。

---

## 8. 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `no kernel image is available for execution` | torch 不是 cu128 / 不含 sm_120 | 重装 §2.4 的 cu128 torch |
| `torch.cuda.is_available()` 为 False | 驱动 < 570 或 GPU 未直通 | 升驱动（§2.1）/ 检查 PCI passthrough |
| 本地编译报 sm_120 错误 | CUDA toolkit < 12.8 | 优先改用方案 A（`VLLM_USE_PRECOMPILED=1`） |
| 启动即 OOM | 模型太大 / KV 池太大 | 降 `--gpu-memory-utilization`、换小模型、缩 `--max-model-len` |
| Phase 5 大量 OOM/超时 | 过载阶段并发过高 | 这是 workload 故意制造的过载，属预期；看违约率而非追求不 OOM |
| suffix proposer 没生效 | 漏了 `--speculative-config` 或环境变量 | 二者都要带（§4.3） |
| 优化好像没区别 | 对照组也开了 prefix caching | A/B 必须严格控制变量（§6.1） |

---

## 9. 最短上手路径（TL;DR）

```bash
# 1. 环境
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
cd /data/home/nikiijiang/expolore/serving-optimization
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
VLLM_USE_PRECOMPILED=1 uv pip install -e . --no-build-isolation
uv pip install pytest pytest-asyncio tblib numpy aiohttp openai

# 2. 先验证优化逻辑（CPU 即可）
pytest tests/v1/core/test_cache_aware_scheduling.py tests/v1/spec_decode/test_adaptive_suffix_proposer.py -v

# 3. 起服务（实验组：开启调度+KV优化）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-chunked-prefill --enable-prefix-caching \
    --gpu-memory-utilization 0.85 --port 8000

# 4. 另开终端压测并对比
python benchmarks/e2e_cases/workload.py --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 --port 8000 --duration 300 --output-dir results/prefix_opt
```

> 评估时记得对照组（不开 `--enable-prefix-caching` / `VLLM_SPEC_PROPOSER=ngram`）也跑一遍，比对 `results/*` 下的 TTFT 与 SLA 违约率。
