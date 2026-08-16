# 重启后快速拉起推理服务（复制即用）

> 前提：之前已按 `RUN_ON_5070TI_WSL_GUIDE.md` 完成安装，环境已就绪。

## 1. 进入 WSL 并激活环境

```bash
cd ~/vllm-serving-optimization
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
```

## 2. 启动推理服务（前台运行，可看日志）

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

> 模型已缓存，约 10 秒启动。看到 `Application startup complete` 即就绪。
> 此终端保持开着，服务在前台运行。

## 3. 另开一个 WSL 终端，发请求

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","messages":[{"role":"user","content":"你好"}],"max_tokens":50}'
```

## 4. 停止服务

在服务终端按 `Ctrl+C`。

---

## 可选：切换优化模式

```bash
# 基线（无优化）—— 去掉 --enable-prefix-caching
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --enforce-eager --port 8000

# 后缀投机解码
VLLM_SPEC_PROPOSER=adaptive \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-prefix-caching \
    --speculative-config '{"method":"ngram","num_speculative_tokens":5,"ngram_prompt_lookup_min":3,"ngram_prompt_lookup_max":5}' \
    --gpu-memory-utilization 0.85 --enforce-eager --port 8000
```
