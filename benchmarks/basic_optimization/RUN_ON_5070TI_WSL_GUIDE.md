# 在 WSL2 Ubuntu + RTX 5070 Ti 上搭建并运行 vLLM 推理框架（实操指南·已验证）

> 本文档记录了**实际跑通**的完整步骤，包含踩坑经验和解决办法。
> 环境：Windows 11 + WSL2 (Ubuntu) + RTX 5070 Ti (16GB, Blackwell sm_120)
> 日期：2026-06-30
> 状态：✅ **已验证跑通**，推理服务正常返回结果

---

## 0. 背景与关键认知

### 0.1 你要跑的是什么

本仓库基于 vLLM V1 二次开发，包含三类优化：

| 优化 | 代码落点 | 如何启用 | 状态 |
|---|---|---|---|
| 缓存感知调度 + 频率感知驱逐 + 抢占缓存保护 | `vllm/v1/core/scheduler.py` 等 | `--enable-prefix-caching` 自动生效 | ✅ 已实现 |
| 后缀投机解码（后缀树/增量SAM/自适应） | `vllm/v1/spec_decode/suffix_proposer.py` 等 | 环境变量 `VLLM_SPEC_PROPOSER` + `--speculative-config` | ✅ 已实现 |
| PD 分离 | 设计为主 | 需多卡 | ⬜ 设计态 |

**关键点：所有已实现优化都是纯 Python，不需要重新编译 CUDA 算子。**

### 0.2 5070 Ti 的硬件前提

| 项目 | 要求 | 说明 |
|---|---|---|
| 显存 | 16 GB GDDR7 | 跑 1.5B 模型足够；7B 装不下 |
| NVIDIA 驱动 | ≥ 570（WSL 下 610+ 也可） | Blackwell 必须新驱动 |
| CUDA | ≥ 12.8（实测用 CUDA 13 / cu130） | sm_120 从 CUDA 12.8 起支持 |

---

## 1. 确认 WSL 与 GPU 直通

WSL2 支持 GPU 直通，前提是 Windows 宿主机已装好 NVIDIA 驱动。

```bash
nvidia-smi
```

期望输出包含 `NVIDIA GeForce RTX 5070 Ti`，显存 16303MiB。

> 如果不可用，回到 Windows 宿主机装最新 NVIDIA 驱动（≥570），重启后再试。

---

## 2. 安装系统依赖（gcc 必须！）

> ⚠️ **gcc 是必须的**——vLLM 的 triton 库需要 gcc 来编译 CUDA kernel。
> 没装 gcc 会报 `RuntimeError: Failed to find C compiler`。

```bash
sudo apt update
sudo apt install -y gcc build-essential git curl ca-certificates python3-dev
```

验证：

```bash
gcc --version   # 期望显示版本号，如 gcc 15.2.0
```

---

## 3. 安装 uv 并克隆仓库到原生 Linux 文件系统

> ⚠️ **不要在 `/mnt/d/`、`/mnt/c/` 等 NTFS 挂载目录上做 Python 编译/安装**，会极慢且有权限问题。代码必须放在 WSL 原生 Linux 文件系统。

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 克隆仓库到原生 Linux 文件系统
cd ~
git clone https://github.com/0826joyce/vllm-serving-optimization.git vllm-serving-optimization
cd ~/vllm-serving-optimization
```

---

## 4. 创建 Python 3.12 虚拟环境

vLLM 要求 Python 3.10–3.13，推荐 3.12。WSL 默认 Python 可能是 3.14（太新），用 uv 指定版本：

```bash
uv venv --python 3.12
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

python --version   # 期望: Python 3.12.x
```

---

## 5. 安装 cu130 PyTorch 并验证 GPU

> ⚠️ **关键：必须用 cu130，不是 cu128！**
>
> PyPI 上的 vLLM 0.20.2 wheel 是用 CUDA 13 编译的（`.so` 文件依赖 `libcudart.so.13`）。
> 如果装 cu128 的 torch，会报 `libcudart.so.13: cannot open shared object file`。
>
> 另外**必须指定 torch==2.11.0**，不能装最新的 2.12.x，否则 C++ ABI 不兼容会报 `undefined symbol`。

```bash
uv pip install 'torch==2.11.0' 'torchvision==0.26.0' \
    --index-url https://download.pytorch.org/whl/cu130
```

验证能识别到 5070 Ti：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

期望输出：`2.11.0+cu130 True NVIDIA GeForce RTX 5070 Ti`

---

## 6. 安装官方 vLLM 0.20.2（关键步骤）

> ⚠️ **这是本指南最关键的步骤，与原始 `RUN_ON_5070TI.md` 差异最大。**
>
> 原始文档用 `VLLM_USE_PRECOMPILED=1` 自动从 `wheels.vllm.ai` 下载匹配 commit 的 wheel。
> 但本仓库是 fork，commit 不在上游 wheel 仓库里，会报 **404 Not Found**。
>
> 而且本仓库基于 v0.7.3，与新版 vLLM 的 Python API 有大量不兼容（`Dict` 未导入、dataclass 默认值等）。
>
> **解决方案：直接安装官方 vLLM 0.20.2 完整包**（Python 代码 + 预编译 .so 完全匹配），
> 然后手动覆盖本仓库的优化文件（第 8 步）。

### 6.1 下载官方 vLLM 0.20.2 wheel

```bash
mkdir -p /tmp/vllm-wheels
curl -L -o /tmp/vllm-wheels/vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl \
  'https://files.pythonhosted.org/packages/c5/aa/4488d49c481a2184e6e285b8d3f937905205f52cd5ac30fb348770494b6e/vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl'
```

> 如果链接失效，去 https://pypi.org/simple/vllm/ 找 `vllm-0.20.2` 的 `manylinux_x86_64.whl`。

### 6.2 安装官方 vLLM（--no-deps 不动 torch）

> ⚠️ **必须用 `--no-deps`！** 不加的话 uv 会解析 172 个依赖，可能降级 `nvidia-nccl-cu13` 等，
> 导致 `libtorch_cuda.so: undefined symbol: ncclDevCommCreate`。

```bash
uv pip install /tmp/vllm-wheels/vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl --no-deps
```

### 6.3 安装运行时依赖（排除 torch / nvidia-*）

```bash
uv pip install \
  regex xformers sentencepiece tokenizers transformers accelerate \
  ray prometheus-fastapi-instrumentator uvicorn fastapi pydantic pyzmq \
  aiohttp openai tiktoken huggingface-hub safetensors gguf \
  compressed-tensors mistral_common parso \
  cycler contourpy kiwisolver matplotlib pyparsing fonttools \
  py-cpuinfo psutil orjson msgspec lm-format-enforcer \
  partial-json-parser numba 'numpy<2.3'
```

> 注意几个坑：
> - `partial-json` 的正确包名是 `partial-json-parser`
> - `numba` 需要 `numpy<2.3`，否则报 `Numba needs NumPy 2.2 or less`
> - 不要装 `torch` / `nvidia-*`，避免破坏已装好的 cu130 torch

### 6.4 验证框架能导入

```bash
# ⚠️ 必须在非仓库目录下执行，否则会加载仓库源码而非 site-packages 的官方包
cd ~
python -c "import vllm; print('vllm OK:', vllm.__file__, vllm.__version__)"
```

期望输出：`vllm OK: .../site-packages/vllm/__init__.py 0.20.2`

---

## 7. 拉起推理服务（已验证跑通）

> ⚠️ **必须加 `--enforce-eager`**（除非装了 gcc 且想用 torch.compile）。
> 不加会触发 torch.compile → triton 编译 → 需要 gcc。
> 加了 `--enforce-eager` 可禁用 torch.compile，但 triton sampling kernel 仍需要 gcc，
> 所以 **gcc 是无论如何都要装的**（见第 2 步）。
>
> ⚠️ **启动命令必须在非仓库目录下执行**（如 `cd ~`），否则 Python 会优先加载仓库源码
> 而非 site-packages 的官方包，导致各种 `NameError`。

```bash
cd ~
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
```

启动过程：下载模型（首次约 6 分钟，3GB）→ 加载到 GPU → 启动 HTTP 服务。
看到 `Application startup complete` 和 `Available routes` 说明服务起来了。

### 验证推理

另开一个 WSL 终端：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "用一句话介绍你自己"}],
    "max_tokens": 50
  }'
```

期望返回 JSON，`choices[0].message.content` 是模型回复。

---

## 8. 应用本仓库的优化（覆盖官方代码）

> 当前安装的是**官方 vLLM 0.20.2**，还没有本仓库的优化代码。
> 要启用优化，需要把仓库里的优化文件覆盖到 site-packages 的 vllm 目录。
>
> ⚠️ **注意**：本仓库基于 v0.7.3，与 0.20.2 的 Python API 有差异，
> 直接覆盖可能报错。需要逐个文件验证兼容性。
>
> **建议先用官方 0.20.2 跑通基线**（第 7 步），确认环境没问题后，
> 再尝试覆盖优化文件。如果覆盖后报错，回退到官方代码即可。

本仓库的三类优化文件：

```bash
# 调度 + KV Cache 优化
cp ~/vllm-serving-optimization/vllm/v1/core/scheduler.py \
   ~/.venv/lib/python3.12/site-packages/vllm/v1/core/scheduler.py

# 后缀投机解码 proposer（3 个文件）
cp ~/vllm-serving-optimization/vllm/v1/spec_decode/suffix_proposer.py \
   ~/.venv/lib/python3.12/site-packages/vllm/v1/spec_decode/
cp ~/vllm-serving-optimization/vllm/v1/spec_decode/suffix_automaton_proposer.py \
   ~/.venv/lib/python3.12/site-packages/vllm/v1/spec_decode/
cp ~/vllm-serving-optimization/vllm/v1/spec_decode/adaptive_suffix_proposer.py \
   ~/.venv/lib/python3.12/site-packages/vllm/v1/spec_decode/
```

> 覆盖后如果启动报错，说明该文件与 0.20.2 API 不兼容，需要手动适配或回退。

---

## 9. A/B 对比配置

### 9.1 基线对照组（关闭优化）

```bash
cd ~
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
```

### 9.2 调度 + KV Cache 优化组

```bash
cd ~
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
```

### 9.3 后缀投机解码优化组

```bash
cd ~
VLLM_SPEC_PROPOSER=adaptive \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"ngram_prompt_lookup_min":3,"ngram_prompt_lookup_max":5}' \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
```

---

## 10. 端到端压测

```bash
cd ~/vllm-serving-optimization
python benchmarks/e2e_cases/workload.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --duration 300 \
    --output-dir results/prefix_opt
```

A/B 对比时，对照组和实验组各跑一次，切换 `--output-dir`，最后对比 TTFT / SLA 违约率。

---

## 11. 显存与参数速查（5070 Ti / 16 GB）

| 模型 | 推荐 `--max-model-len` | 估算占用 | 备注 |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 8192 | ~6–8 GB | 最稳 |
| **Qwen2.5-1.5B-Instruct** | 8192 | ~10–12 GB | **推荐默认** |
| Qwen2.5-3B-Instruct | 4096 | ~13–15 GB | 需调小 `--max-model-len` |
| 7B/8B | ❌ | 16 GB 装不下 | 不建议 |

---

## 12. 常见问题排查

| 现象 | 原因 | 解决 |
|---|---|---|
| `no kernel image is available` | torch 不含 sm_120 kernel | 用 cu130 的 torch（第 5 步） |
| `torch.cuda.is_available()` 为 False | 驱动太旧或 GPU 未直通 | 装 Windows 宿主机最新 NVIDIA 驱动 |
| `VLLM_USE_PRECOMPILED` 报 404 | fork 的 commit 不在 wheels.vllm.ai | 直接安装官方 0.20.2 wheel（第 6 步） |
| `undefined symbol: ncclDevCommCreate` | uv 装 vllm 依赖时降级了 nvidia 库 | 用 `--no-deps` 安装（第 6.2 步） |
| `undefined symbol: ...torch...` | torch 版本与 wheel 不匹配 | 必须用 `torch==2.11.0`（第 5 步） |
| `libcudart.so.13: cannot open` | torch 是 cu128，wheel 需要 cu130 | 装 cu130 的 torch（第 5 步） |
| `Failed to find C compiler` | WSL 里没装 gcc | `sudo apt install gcc`（第 2 步） |
| `Numba needs NumPy 2.2 or less` | numpy 版本太新 | `uv pip install 'numpy<2.3'` |
| `Dict is not defined` / `Optional is not defined` | 在仓库目录启动，加载了旧版源码 | `cd ~` 后再启动服务（第 7 步） |
| `partial-json was not found` | 包名错误 | 正确包名是 `partial-json-parser` |
| 启动即 OOM | 模型太大 / KV 池太大 | 降 `--gpu-memory-utilization`、换小模型 |
| HuggingFace 下载失败 | hf-mirror.com 重定向回 huggingface.co | 不设 HF_ENDPOINT，直接用 huggingface.co |

---

## 13. 一键速查（TL;DR）

```bash
# === 1. 系统依赖（gcc 必须！）===
sudo apt update && sudo apt install -y gcc build-essential git curl

# === 2. uv + 仓库 ===
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
cd ~ && git clone https://github.com/0826joyce/vllm-serving-optimization.git vllm-serving-optimization
cd ~/vllm-serving-optimization

# === 3. Python 3.12 虚拟环境 ===
uv venv --python 3.12
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

# === 4. PyTorch cu130（必须 2.11.0 + cu130）===
uv pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# === 5. 下载官方 vLLM 0.20.2 wheel ===
mkdir -p /tmp/vllm-wheels
curl -L -o /tmp/vllm-wheels/vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl \
  'https://files.pythonhosted.org/packages/c5/aa/4488d49c481a2184e6e285b8d3f937905205f52cd5ac30fb348770494b6e/vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl'

# === 6. 安装 vLLM（--no-deps 避免破坏 torch）===
uv pip install /tmp/vllm-wheels/vllm-0.20.2-cp38-abi3-manylinux_2_35_x86_64.whl --no-deps

# === 7. 运行时依赖（不含 torch / nvidia-*）===
uv pip install \
  regex xformers sentencepiece tokenizers transformers accelerate \
  ray prometheus-fastapi-instrumentator uvicorn fastapi pydantic pyzmq \
  aiohttp openai tiktoken huggingface-hub safetensors gguf \
  compressed-tensors mistral_common parso \
  cycler contourpy kiwisolver matplotlib pyparsing fonttools \
  py-cpuinfo psutil orjson msgspec lm-format-enforcer \
  partial-json-parser numba 'numpy<2.3'

# === 8. 验证（必须在非仓库目录）===
cd ~
python -c "import vllm; print('vllm OK:', vllm.__version__)"

# === 9. 起服务（--enforce-eager + cd ~）===
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-prefix-caching --gpu-memory-utilization 0.85 \
    --enforce-eager --port 8000

# === 10. 另开终端验证 ===
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

---

## 附：与原始 RUN_ON_5070TI.md 的差异说明

原始文档假设 `VLLM_USE_PRECOMPILED=1` 能直接工作，但实际在 fork 仓库上遇到大量问题。本指南的关键改动：

1. **预编译 wheel 404**：fork commit 不在 wheels.vllm.ai → 直接装官方 0.20.2 wheel
2. **版本不匹配**：本仓库 v0.7.3 Python 代码与 0.20.2 不兼容 → 先用官方 0.20.2 跑通，再覆盖优化文件
3. **CUDA 版本**：原始说 cu128，实际 0.20.2 wheel 需要 cu130 → 装 torch 2.11.0+cu130
4. **torch 版本**：不能装最新 2.12.x，必须 2.11.0（ABI 匹配）
5. **gcc 必须**：triton 编译需要 gcc → `sudo apt install gcc`
6. **启动目录**：必须在非仓库目录启动（`cd ~`），否则加载旧版源码报 NameError
7. **--enforce-eager**：避免 torch.compile 问题（加了也要 gcc，因 triton sampling kernel）
8. **nccl 冲突**：`--no-deps` 安装避免降级 nvidia 库
9. **numba/numpy**：numba 需要 numpy<2.3
10. **包名**：`partial-json` → `partial-json-parser`
