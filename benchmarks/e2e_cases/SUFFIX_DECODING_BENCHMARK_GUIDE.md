# 后缀解码优化 — 对比测试方案

> 目标：验证 [`suffix-decoding-optimization.md`](../basic_optimization/suffix-decoding-optimization.md)
> 中的后缀解码优化（**AdaptiveSuffixProposer**）相对基线（**NgramProposer**）的效果。
>
> 核心指标：**投机解码接受率（Acceptance Rate）**、平均接受长度、吞吐、端到端延迟。

## 0. 背景与原理（先理解再测）

### 0.1 投机解码（Speculative Decoding）机制

投机解码用一个小/轻量的 "drafter" 先生成多个候选 token（draft），再由主模型一次验证：

```
每个 decode step：
  drafter 提出 k 个 draft tokens（如 k=4）
     ↓
  主模型一次前向验证这 k 个 token
     ↓
  接受前 m 个（m ≤ k），丢弃其余
     ↓
  实际产出 m+1 个 token（含 1 个 bonus token）
```

**关键指标——接受率**：
```
接受率 = 被接受的 draft tokens 数 / 提出的 draft tokens 总数
```

接受率越高，说明 drafter 提出的 draft 越准，投机解码的收益越大（同样的 GPU 前向次数产出更多 token）。

### 0.2 对比对象

| 方案 | 说明 | 触发方式 |
|------|------|---------|
| **A 基线：NgramProposer** | vLLM V1 官方 N-gram 匹配，固定窗口 + 只取首个匹配 | 默认（`method="ngram"`） |
| **B 优化：AdaptiveSuffixProposer** | 后缀自动机（SAM）+ 多候选评分 + 接受率反馈 | `VLLM_SPEC_PROPOSER=adaptive` |

### 0.3 预期效果（来自优化文档）

| 指标 | 预期 | 说明 |
|------|------|------|
| 接受率 | **+15~30%** | 从"首个匹配"变为"最优匹配"（多候选评分） |
| 匹配率 | 提升 | 自适应回退减少"0 draft"情况 |
| 吞吐 | 提升 | 接受率↑ → 同样前向次数产出更多 token |
| 单步耗时 | 略增 | SAM 构建/评分有额外开销（需权衡） |

> ⚠️ **核心权衡**：Adaptive 的收益来自"接受率↑"，代价是"proposer 本身耗时↑"（SAM 构建 + 多候选评分）。
> 因此**净收益 = 接受率提升带来的吞吐增益 − proposer 额外开销**，必须实测验证净效果为正。

---

## 1. 测试环境与前提

- RTX 5070 Ti（Blackwell，sm_120），单卡
- Qwen2.5-1.5B-Instruct
- **无 `--enforce-eager`**（用默认 CUDA graph，避免 prefill 慢的坑，见调度测试经验）
- 投机解码配置：`--speculative-config '{"method":"ngram","prompt_lookup_max":3,"num_speculative_tokens":4}'`

### 1.1 两个 proposer 如何切换

```bash
# A 基线（默认 ngram）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-config '{"method":"ngram","prompt_lookup_max":3,"num_speculative_tokens":4}' \
    ...

# B 优化（adaptive suffix）
VLLM_SPEC_PROPOSER=adaptive python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-config '{"method":"ngram","prompt_lookup_max":3,"num_speculative_tokens":4}' \
    ...
```

> 两者共用 `method="ngram"` 配置（因为 Adaptive 是 ngram 的替代品，复用相同的 prompt_lookup 参数），
> 区别仅在于 `VLLM_SPEC_PROPOSER` 环境变量。

---

## 2. 核心指标与数据源

### 2.1 投机解码专属指标（最关键）

vLLM 服务端 metrics 已内置投机解码指标：

| 指标 | Prometheus / 日志 | 含义 |
|------|------------------|------|
| **接受率** | `vllm:spec_decode_num_accepted_tokens / num_draft_tokens` | 核心指标 |
| **平均接受长度** | `1 + num_accepted_tokens / num_drafts` | 含 bonus token |
| 草案次数 | `vllm:spec_decode_num_drafts` | 每 step 的 draft 数 |
| draft tokens | `vllm:spec_decode_num_draft_tokens` | 提出的 draft 总数 |
| 接受 tokens | `vllm:spec_decode_num_accepted_tokens` | 被接受的 token 总数 |

> 服务端日志也会周期性打印 `SpecDecoding metrics: Mean acceptance length: X.XX, Avg Draft acceptance rate: XX%`，
> 这是最直接的接受率数据。

### 2.2 端到端性能指标

| 指标 | 数据源 | 说明 |
|------|--------|------|
| 吞吐（tok/s） | `vllm bench serve` | 投机解码收益的直接体现 |
| TTFT / E2E | `vllm bench serve` | 延迟 |
| TPOT | `vllm bench serve` | 每 token 时间（投机解码应降低） |

### 2.3 指标关系（为什么测这些）

```
接受率↑ → 每 step 产出 token 多 → 相同前向次数吞吐↑
                                    ↓
proposer 耗时↑（SAM 构建） → 每 step 开销↑ → 吞吐↓
                                    ↓
                        净效果 = 两者权衡，必须实测
```

---

## 3. 测试设计

### 3.1 测试矩阵

| 维度 | 取值 | 说明 |
|------|------|------|
| proposer | A（ngram）/ B（adaptive） | 核心对比 |
| 负载 | 恒定 QPS（10/50/100） | 从轻到重 |
| 数据集 | random / 有重复模式 | 后缀解码对重复模式更友好 |

> **关键**：后缀解码的效果**强依赖输入是否有重复模式**（如代码、模板化文本、对话）。
> 纯随机 token 的输入几乎没有可匹配的后缀，接受率会很低。因此测试必须包含**有重复模式的负载**。

### 3.2 数据集选择（重要）

`vllm bench serve` 内置了**专门为投机解码设计**的数据集，应优先使用：

| 数据集 | 特点 | 适合测什么 |
|--------|------|-----------|
| **`spec_bench`** | vLLM 官方投机解码基准，含多类别（重复/语法/翻译等） | **主测**，最能体现投机解码差异 |
| **`prefix_repetition`** | 前缀重复数据集（`--prefix-repetition-prefix-len` 控制） | 后缀解码的理想场景（高重复） |
| `random` | 完全随机 token | 下限场景（接受率最低） |
| `sharegpt` | 真实对话，有模板重复 | 真实场景 |

> **推荐测试组合**：
> 1. `spec_bench`（主测，官方投机解码基准，最权威）
> 2. `prefix_repetition`（后缀解码的理想场景，验证上限）
> 3. `random`（下限，确认优化不会在无重复场景下恶化）

> **关键参数**：
> - `spec_bench`：`--spec-bench-category`（None=全部类别）、`--spec-bench-output-len 256`
> - `prefix_repetition`：`--prefix-repetition-prefix-len 256 --prefix-repetition-suffix-len 1024`

---

## 4. 测试步骤

### Step 1：启动 A 基线（ngram）

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"ngram","prompt_lookup_max":3,"num_speculative_tokens":4}' \
    --port 8000
```

等待 `Application startup complete`。

### Step 2：跑 A 基线基准测试

```bash
# spec_bench 数据集（vLLM 官方投机解码基准，最权威）
vllm bench serve \
    --backend openai \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --dataset-name spec_bench \
    --spec-bench-output-len 256 \
    --num-prompts 500 \
    --request-rate 10 \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,95,99 \
    --save-result --result-dir results/suffix_decoding/ \
    --result-filename baseline_ngram.json
```

### Step 3：记录 A 基线的接受率

压测结束后，从服务端日志或 metrics 提取接受率：

```bash
# 方式一：服务端日志（最直接）
grep "SpecDecoding metrics" /path/to/server.log | tail -3

# 方式二：metrics 端点
curl -s http://127.0.0.1:8000/metrics | grep spec_decode
```

### Step 4：停 A，启动 B 优化（adaptive）

```bash
VLLM_SPEC_PROPOSER=adaptive python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"ngram","prompt_lookup_max":3,"num_speculative_tokens":4}' \
    --port 8000
```

### Step 5：跑 B 优化基准测试（同 Step 2 参数）

```bash
vllm bench serve \
    --backend openai \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --dataset-name spec_bench \
    --spec-bench-output-len 256 \
    --num-prompts 500 \
    --request-rate 10 \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,95,99 \
    --save-result --result-dir results/suffix_decoding/ \
    --result-filename optimized_adaptive.json
```

### Step 6：记录 B 的接受率（同 Step 3）

---

## 5. 结果记录与判据

### 5.1 结果表

| 指标 | A 基线（ngram） | B 优化（adaptive） | 差异 | 判定 |
|------|----------------|-------------------|------|------|
| 接受率 | ~76%（71~83%） | ~74%（69~78%） | **-2pp（基本持平）** | ❌ 未达 +15~30% |
| 平均接受长度 | ~4.0 | ~3.9 | 略降 | ❌ |
| Mean TPOT | 6.39 ms | 6.92 ms | +8.3% | ⚠️ 略差 |
| P99 TPOT | 16.99 ms | 20.83 ms | +22.6% | ❌ 恶化 |
| P99 TTFT | 386.62 ms | 246.82 ms | -36% | ⚠️ 疑似波动 |
| Mean E2EL | 894.13 ms | 944.53 ms | +5.6% | ⚠️ 略差 |

> 说明：上表为**第一轮测试**（`prefix_repetition` 数据集）结果。P99 TTFT 的 -36% 与
> TPOT/E2EL 的恶化矛盾，判断为两次压测的时序波动，非优化收益。详见 §5.4。

### 5.2 判定标准

| 结果 | 判定 |
|------|------|
| 接受率 +15~30%，吞吐提升，TTFT 不恶化 | ✅ 优化有效 |
| 接受率提升但吞吐持平/下降 | ⚠️ proposer 开销抵消了收益，需优化 SAM 构建 |
| 接受率无提升 | ❌ 在测试负载下优化无效，需换数据集或调参 |

### 5.3 关键注意事项

1. **数据集决定结论**：随机 token 输入几乎无重复模式，后缀解码接受率会很低（可能接近 ngram）。
   必须用有重复模式的真实数据（sharegpt）才能体现优化价值。

2. **proposer 开销 vs 收益**：Adaptive 的 SAM 构建和多候选评分有额外 CPU 开销。在**短上下文**下，
   这个开销可能超过接受率提升带来的收益。需在**长上下文**（如 4K token）下测试，SAM 的增量优势才能体现。

3. **接受率指标的准确性**：服务端日志的 `Avg Draft acceptance rate` 是权威数据，比客户端推算更准。

### 5.4 第一轮测试结论（2026-08-26，prefix_repetition 数据集）

> 测试配置：`prefix_repetition`（prefix-len 256 / suffix-len 1024 / output 128 / 8 prefixes），
> 500 请求，10 QPS，无 eager，num_speculative_tokens=4。

#### 结果

- **接受率**：A 基线 ~76% vs B 优化 ~74%，**基本持平（-2pp），未达预期 +15~30%**
- **平均接受长度**：~4.0 vs ~3.9，略降
- **TPOT**：6.39 vs 6.92 ms（+8.3%），B 略差——SAM 构建开销存在但无收益补偿
- **E2EL**：894 vs 945 ms（+5.6%），B 略差

#### 根因分析

**在 `prefix_repetition` 数据集上，优化无效，原因是"基线已经接近上限"**：

1. `prefix_repetition` 是**高度规则的前缀重复**数据集，Ngram 的固定窗口匹配（prompt_lookup_max=3）
   已经能稳定命中（接受率高达 76%），Adaptive 的"多候选评分 + recency 加权"没有额外提升空间。

2. **Adaptive 的 proposer 开销纯属损失**：SAM 构建 + 多候选线性扫描 + 四因子评分，在"接受率没提升"
   的情况下，这些 CPU 开销直接反映为 TPOT 上升（+8.3%）。

3. **P99 TTFT 的 -36% 是假象**：与 TPOT/E2EL 的恶化矛盾，判断为两次压测的时序抖动，非优化收益。

#### 结论

**在规则重复数据集下，AdaptiveSuffixProposer 相对 NgramProposer 无优势（甚至略差）。**
这与 §5.3 的预警一致——优化价值取决于数据集，规则重复场景 Ngram 已足够。

#### 下一步

需换**不规则但可预测**的数据集重新验证，才能体现 Adaptive 的"多候选评分 + recency 加权"优势：

- `spec_bench`（官方投机解码基准，需 `--dataset-path` 下载 HF 数据集）
- `sharegpt`（真实对话，自然语言的不规则重复）

> ⚠️ **预期调整**：文档 §0.3 的"+15~30%"预期是基于"从首个匹配变为最优匹配"的理论推演，
> 但该收益**仅在不规则、可预测的真实文本上成立**。规则重复数据下 Ngram 已足够，无优化空间。

---

## 6. 待办与结论记录

> 测试结果记录于此。

- [x] A 基线（ngram）接受率 + 吞吐（§5.4，接受率 ~76%）
- [x] B 优化（adaptive）接受率 + 吞吐（§5.4，接受率 ~74%）
- [x] 第一轮对比分析（prefix_repetition，优化无效，见 §5.4）
- [ ] 第二轮：spec_bench / sharegpt 数据集（不规则真实文本）验证
- [ ] 结论：优化在什么数据集/负载下有效
