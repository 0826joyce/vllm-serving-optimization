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

> **核心变化**：本次测试把优化文档的 **6 个优化点（1/2/3/4/5/7）合并成一个"B 大优化"**，
> 一次性开启全部优化，与 A 基线做整体对比，而不是逐个优化点单独测。

| 方案 | 说明 | 触发方式 |
|------|------|---------|
| **A 基线：NgramProposer** | vLLM V1 官方 N-gram 匹配，固定窗口 + 只取首个匹配，k 固定 | 默认（`method="ngram"`），无任何后缀优化 |
| **B 大优化：AdaptiveSuffixProposer（全开）** | 后缀自动机（SAM）+ 多候选评分 + 接受率反馈 + 增量更新 + 全局池 + 动态投机长度 | 三个环境变量全开（见下） |

**B 大优化 = 优化 1/2/3/4/5/7 全部开启**：

```bash
VLLM_SPEC_PROPOSER=adaptive \        # 优化1/2/3：自适应后缀 proposer（SAM + 多候选 + 增量更新）
VLLM_SUFFIX_GLOBAL_POOL=1 \          # 优化4：跨请求共享 + 频率感知筛选
VLLM_SUFFIX_DYNAMIC_K=1 \            # 优化7：负载感知的动态投机长度
```

| 优化点 | 开关 | 作用 |
|--------|------|------|
| 1 基础实现 | `VLLM_SPEC_PROPOSER=adaptive` | 后缀自动机替代固定窗口匹配 |
| 2 增量更新 | （同上） | O(1) 增量后缀结构，长上下文下避免每步重建 |
| 3 自适应匹配 + 多候选评分 | （同上） | 回退策略 + 多候选最优匹配，提升接受率 |
| 4 跨请求共享 + 频率筛选 | `VLLM_SUFFIX_GLOBAL_POOL=1` | 全局模式池复用 + 按接受频率筛质量 |
| 5 可观测性 | （默认开启） | 内部行为指标埋点（匹配率/回退/动态k 等） |
| 7 动态投机长度 | `VLLM_SUFFIX_DYNAMIC_K=1` | 轻负载激进投机、重负载保守 |

> 注：优化点 8（树状多候选）已从优化文档移除，不纳入 B。优化 4 在 `prefix_repetition`
> 下价值有限（§3.4 说明），但作为"最大优化"仍开启，用于评估其在真实数据下的边界价值。

### 0.3 预期效果（来自优化文档）

> 以下为优化文档的理论预期，**实际是否达到需 A/B 实测验证**（尤其是净收益是否为正）。

| 指标 | 预期 | 说明 |
|------|------|------|
| 接受率 | **+15~30%** | 从"首个匹配"变为"最优匹配"（多候选评分，优化 3） |
| 匹配率 | 提升 | 自适应回退减少"0 draft"情况（优化 3） |
| 平均接受长度 | ~2-3 → ~3-5 | 多候选选最长/最优匹配（优化 3） |
| 吞吐 | 提升 | 接受率↑ → 同样前向次数产出更多 token |
| TPOT | **降低** | 接受率↑ + 动态 k（优化 7）在低负载下激进投机 |
| 单步耗时 | 略增 | SAM 构建/评分有额外开销（需权衡，优化 2 增量更新缓解） |

> ⚠️ **核心权衡**：Adaptive 的收益来自"接受率↑"，代价是"proposer 本身耗时↑"（SAM 构建 + 多候选评分）。
> 因此**净收益 = 接受率提升带来的吞吐增益 − proposer 额外开销**，必须实测验证净效果为正。
> 优化 7（动态 k）进一步把"投机算力"按负载分配——**低负载激进、高负载保守**，需在高/低负载下分别验证。

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

# B 大优化（adaptive suffix 全开：优化 1/2/3/4/5/7）
VLLM_SPEC_PROPOSER=adaptive VLLM_SUFFIX_GLOBAL_POOL=1 VLLM_SUFFIX_DYNAMIC_K=1 \
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-config '{"method":"ngram","prompt_lookup_max":3,"num_speculative_tokens":4}' \
    ...
```

> 两者共用 `method="ngram"` 配置（因为 Adaptive 是 ngram 的替代品，复用相同的 prompt_lookup 参数），
> 区别仅在于环境变量：A 无任何后缀优化开关；B 开启 `VLLM_SPEC_PROPOSER` + `VLLM_SUFFIX_GLOBAL_POOL` + `VLLM_SUFFIX_DYNAMIC_K`。

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

### 2.4 proposer 内部行为观测项（关键补充，用于归因）

> 早期探索测试（§5.5 历史过程数据）只看了"结果指标"（接受率/TPOT），得出"优化无效"后**只能靠推测归因**。
> 要真正说清楚"为什么无效/有效"，必须观测 proposer 的**内部行为**——这对应
> [`suffix-decoding-optimization.md`](../basic_optimization/suffix-decoding-optimization.md) 优化点 5 设计的 `SuffixDecodeMetrics`。
> 这些指标目前**未采集**，是本测试最大的观测盲区，需在 proposer 里埋点后补测。

| 观测项 | 回答的问题 | 为什么必须（归因价值） | 采集方式 |
|--------|-----------|---------------------|---------|
| **匹配率（match rate）** | 有多少次 propose **根本没匹配到**？ | 区分"没匹配到"vs"匹配到但没被接受"——两者的优化方向完全不同 | proposer 埋点 `match_found / (found+not_found)` |
| **平均匹配长度** | Adaptive 是否真的匹配到**更长**的模式？ | 验证"多候选评分选最优"是否真的选了更长匹配 | 埋点 `match_lengths_sum / match_found` |
| **回退次数（fallback）** | 自适应回退（n→n/2）**触发了几次**？ | 验证优化点 3 的"多级回退"是否真的在工作 | 埋点 `fallback_count` |
| **多候选命中次数** | 有多少次**选了非首个匹配**（recency/评分起作用）？ | 这是 Adaptive 相对 Ngram（只取首个匹配）的**核心差异点**，不测则无法证明优化真的生效 | 埋点 `best_candidate_from_recent` |
| **SAM 构建耗时 / 查询耗时** | proposer 的**额外 CPU 开销到底多大**？ | 净收益归因的关键——TPOT +8.3% 里有多少来自 proposer 而非 GPU | 埋点 `sam_build_time_ms` / `sam_query_time_ms` |
| **每步有效 token 数（tokens/step）** | 投机解码的**最终收益度量** | = (accepted + bonus + 1)/step，比接受率更能反映真实吞吐收益 | metrics 推算 |
| **proposer 耗时占单步比例** | proposer 开销在整个 decode step 里占比 | 判断"开销是否被 GPU forward 掩盖"（低并发时占比高） | `sam_time / step_time` |

> **归因逻辑**：只有同时看这些，才能区分下面几种"优化无效"的不同原因，对症下药：
> - 匹配率低 → 数据集没有可匹配模式（换数据集）；
> - 匹配率高但多候选命中为 0 → 评分策略没起作用（首个匹配已是最优，如 §5.5 的规则重复场景）；
> - 接受率提升但 TPOT 上升 → proposer 开销吃掉了收益（优化 SAM 构建）。

> **采集方式**：`SuffixDecodeMetrics`（优化点 5）**已实现**。正式主测时需在 `AdaptiveSuffixProposer.propose()` 里
> 采集这些内部指标，周期性打日志或接入 Prometheus（见 §4 Step 3 的采集方式），用于 §5.4 归因。

---

## 3. 测试设计

### 3.1 测试矩阵

| 维度 | 主测取值 | 说明 |
|------|---------|------|
| proposer | A（ngram）/ B（adaptive 全开） | 核心对比（B 定义见 §0.2） |
| 负载 | QPS **10（低）/ 50（高）** | 低负载下 proposer 开销占比更突出；高负载验证优化 7 负载自适应 |
| 数据集 | **prefix_repetition / spec_bench** | 覆盖"规则重复→不规则真实"两种分布（§3.3 主测） |
| k | 固定 **4** | 主测统一用 k=4（对比 B 的动态 k） |

> **补充实验（可选，归因用）**——以下维度**不在主测**，仅当 §5.3 需要深入归因时再跑：
> - **k 扫描（2/4/8）**：验证 Adaptive 是否在"长 draft"（k 大）时才显优势（ngram 固定窗口够不到长匹配）。
> - **上下文长度（512 vs 4K）**：验证增量 SAM（优化点 2）的 O(1) 更新优势是否只在长上下文体现。

> **关键**：后缀解码的效果**强依赖输入是否有重复模式**（如代码、模板化文本、对话）。
> 纯随机 token 的输入几乎没有可匹配的后缀，接受率会很低。因此主测必须包含**有重复模式的负载**
> （`prefix_repetition` 规则重复、`spec_bench` 不规则真实）。

### 3.2 数据集选择（重要）

`vllm bench serve` 内置了**专门为投机解码设计**的数据集，应优先使用：

| 数据集 | 特点 | 适合测什么 |
|--------|------|-----------|
| **`spec_bench`** | vLLM 官方投机解码基准，含多类别（重复/语法/翻译等） | **主测**，最能体现投机解码差异 |
| **`prefix_repetition`** | 前缀重复数据集（`--prefix-repetition-prefix-len` 控制） | 后缀解码的理想场景（高重复） |
| `random` | 完全随机 token | 下限场景（接受率最低） |
| `sharegpt` | 真实对话，有模板重复 | 真实场景 |

> **主测数据集（§3.3）**：
> 1. `prefix_repetition`（规则重复，验证上限）
> 2. `spec_bench`（不规则真实文本，**核心轮次**，最能体现投机解码差异）
>
> **可选补充**：`random`（无重复下限）——仅当主测显示 B 有收益时，用于确认优化不会在无重复场景恶化。

> **关键参数**：
> - `spec_bench`：`--spec-bench-category`（None=全部类别）、`--spec-bench-output-len 256`
> - `prefix_repetition`：`--prefix-repetition-prefix-len 256 --prefix-repetition-suffix-len 1024`

### 3.3 B 大优化 vs A 基线：主测矩阵

> **设计原则**：B 是 6 个优化点合一的大优化，因此不再逐优化点单独测，而是**在每个数据集下做一次完整的 A/B 对比**。
> 通过**多个数据集（覆盖不同数据分布）+ 高/低负载**，给出 B 的**有效性边界**（在什么场景有效、什么场景无效）。

| 轮次 | 数据集 | 数据分布 | QPS | 验证目标 | 状态 |
|------|--------|---------|-----|---------|------|
| **1** | **prefix_repetition** | 规则前缀重复（高度可预测） | **10 / 50** | **规则重复场景**下 B 是否优于 A（基线已接近上限，预期收益有限） | ⬜ 待做 |
| **2** | **spec_bench** | 不规则真实文本（vLLM 官方投机基准） | **10 / 50** | **不规则真实场景**下 B 是否优于 A（**验证优化主张的核心轮次**） | ⬜ 待做 |

> **有效性边界怎么判**：
> - 若**规则重复（轮次 1）B ≈ A、不规则真实（轮次 2）B > A** → 证明 B 的价值在"不规则但可预测"的真实场景，边界清晰。
> - 若两个数据集 B 都 ≈ A 或更差 → 说明 B 的 proposer 开销抵消了收益，需要从 §2.4 内部指标归因。
> - 若两个数据集 B 都 > A → 优化全面有效。
>
> **负载维度（高/低）**：低负载（QPS 10）下 proposer 开销占比更突出（不被 GPU 排队掩盖），
> 且优化 7（动态 k）会激进投机；高负载（QPS 50）下验证动态 k 是否"保守投机、避免拖累"。

> **控制变量提醒**：每轮 A/B 必须除三个环境变量（`VLLM_SPEC_PROPOSER`/`VLLM_SUFFIX_GLOBAL_POOL`/`VLLM_SUFFIX_DYNAMIC_K`）
> 外**所有参数完全一致**（同数据集、同 k、同 QPS、同 seed、同上下文长度），否则对比无效。
> 每轮都需同时采集 **§2.4 的 proposer 内部指标**，否则只能靠推测归因。

### 3.4 B 大优化中的负载维度：动态投机长度（优化点 7）

> **说明**：优化点 7（动态 k）是 B 大优化里**唯一随负载变化的维度**（其余 5 个优化点与负载无关）。
> 因此在 §3.3 的两轮主测里，通过 **高/低 QPS** 同时验证 B 的负载自适应能力。

> 机制：`set_load()` 每步传入系统负载（running/max），`_effective_k()` 动态调 k：
> - 轻负载（load<0.5）：k ×1.5（激进投机，用空闲算力换低延迟）
> - 中负载（0.5-0.8）：k ×1.0
> - 重负载（load>0.8）：k ×0.5（保守，避免失败投机挤占算力）

| 项 | 设计 |
|----|------|
| 对照 | A（ngram 固定 k=4）vs B（adaptive + 动态 k）|
| 数据集/负载 | prefix_repetition / spec_bench；**QPS 10（低）/ 50（高）** |
| 验证目标 | 低负载下 B 是否激进投机降延迟；**高负载下 B 是否避免"投机拖累"** |
| 关键指标 | 各负载下吞吐、TPOT、平均 draft 长度（avg_dyn_len）、skips（k 调为 0 的次数）|
| 内部指标来源 | `SuffixDecodeMetrics`：`Dyn Spec Len: avg=.., adjusts=.., skips(k=0)=..` |

**核心判据（负载维度）**：
- 低负载：B 动态 k 把 draft 调大（×1.5），TPOT 应**低于** A 的固定 k=4（激进投机收益）；
- 高负载：B 动态 k 应主动调小（×0.5），**吞吐不塌**；A 固定 k=4 可能因失败投机挤占算力而吞吐下降。

> **共同要求**：每轮 A/B 都需同时采集 §2.4 的 proposer 内部指标（尤其"平均 draft 长度""无效投机率"），否则无法归因。


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

### Step 4：停 A，启动 B 大优化（adaptive 全开）

```bash
VLLM_SPEC_PROPOSER=adaptive VLLM_SUFFIX_GLOBAL_POOL=1 VLLM_SUFFIX_DYNAMIC_K=1 \
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method":"ngram","prompt_lookup_max":3,"num_speculative_tokens":4}' \
    --port 8000
```

### Step 5：跑 B 大优化基准测试（同 Step 2 参数）

> 按 §3.3 两轮主测：`spec_bench`（本示例）与 `prefix_repetition` 各跑一次。
> 高负载（QPS 50）时把 `--request-rate` 改为 50 即可。

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

> **本节定位**：B 是 6 个优化点合一的大优化，因此结果不再按"优化点逐个"记录，
> 而是按 **§3.3 的两轮主测（prefix_repetition + spec_bench）** 给出 B vs A 的整体结论与**有效性边界**。

### 5.1 判定标准（B 大优化 vs A 基线）

> 判据分**三个层面**：① 是否有效（净收益）；② 在哪些数据分布下有效（边界）；③ 内部指标能否支撑归因。

**① 有效性判据（净收益）**：

| 观测 | 判定 |
|------|------|
| 接受率提升 + 吞吐提升 + TPOT/E2EL 不恶化 | ✅ **B 有效** |
| 接受率提升但吞吐/TPOT 恶化 | ⚠️ proposer 开销吃掉收益（需优化 2 增量更新/归因） |
| 接受率无提升，吞吐持平或下降 | ❌ 该数据分布下 B 无效 |

**② 有效性边界判据（跨数据集）**：

| 模式 | 结论 |
|------|------|
| 规则重复（prefix_repetition）B≈A，不规则真实（spec_bench）B>A | **B 的价值在"不规则但可预测"的真实场景**，边界清晰 ✅ |
| 两数据集 B 均 >A | B 全面有效 ✅ |
| 两数据集 B 均 ≈A 或更差 | B 的 proposer 开销抵消收益，需归因或接受"无优势" ❌ |

**③ 负载维度判据（优化 7）**：
- 低负载：B 动态 k 激进投机，TPOT 应 < A 固定 k=4；
- 高负载：B 动态 k 保守，**吞吐不塌**；A 固定 k=4 可能因失败投机挤占算力而吞吐下降。

### 5.2 B 大优化 vs A 基线：主测结果表（2026-08-29 已测）

> 测试日期 2026-08-29，QPS10 与 QPS50 已全部跑完，A/B 各 4 轮，控制变量完全一致。

**轮次 1：prefix_repetition（规则重复）**

| 指标 | 负载 | A 基线（ngram） | B 大优化（adaptive 全开） | 差异 |
|------|------|----------------|--------------------------|------|
| 接受率 | QPS10 | 77.8% | 57.9% | **-19.9pp** ❌ |
| 接受率 | QPS50 | 86.6% | 57.5% | **-29.1pp** ❌ |
| 平均接受长度 | QPS10 | 4.06 | 4.20 | +0.14（略升）|
| Output 吞吐（tok/s）| QPS10 | 1256.4 | 1268.1 | +0.9%（持平）|
| Output 吞吐（tok/s）| QPS50 | 2521.0 | 1919.6 | **-23.9%** ❌ |
| Mean / P99 TPOT（ms）| QPS10 | 5.93 / 14.95 | 5.95 / 18.48 | 持平 / +23.6% ⚠️ |
| Mean / P99 TPOT（ms）| QPS50 | 26.87 / 70.36 | 32.86 / 95.13 | **+22% / +35%** ❌ |
| Mean / P99 E2EL（ms）| QPS10 | 837 / 2032 | 838 / 2434 | 持平 / +20% ⚠️ |
| Mean / P99 E2EL（ms）| QPS50 | 4982 / 9941 | 5977 / 12831 | **+20% / +29%** ❌ |
| 成功/失败 | QPS50 | 248 / 252 | 219 / 281 | B 少 29 个成功 ❌ |

**轮次 2：spec_bench（不规则真实文本，核心轮次）**

| 指标 | 负载 | A 基线（ngram） | B 大优化（adaptive 全开） | 差异 |
|------|------|----------------|--------------------------|------|
| 接受率 | QPS10 | 41.1% | 25.0% | **-16.1pp** ❌ |
| 接受率 | QPS50 | 40.3% | 26.3% | **-14.0pp** ❌ |
| 平均接受长度 | QPS10 | 2.64 | 2.33 | -0.31 ❌ |
| Output 吞吐（tok/s）| QPS10 | 1227.5 | 1232.6 | +0.4%（持平）|
| Output 吞吐（tok/s）| QPS50 | 5108.1 | 4737.6 | **-7.3%** ❌ |
| Mean / P99 TPOT（ms）| QPS10 | 6.91 / 10.17 | 6.74 / 11.09 | -2.5% / +9% ⚠️ |
| Mean / P99 TPOT（ms）| QPS50 | 13.68 / 19.29 | 21.86 / 36.54 | **+60% / +89%** ❌ |
| Mean / P99 E2EL（ms）| QPS10 | 854 / 2138 | 826 / 2168 | -3.3% / +1% ⚠️ |
| Mean / P99 E2EL（ms）| QPS50 | 1675 / 4341 | 2596 / 7564 | **+55% / +74%** ❌ |
| 成功/失败 | QPS50 | 500 / 0 | 500 / 0 | 持平 |

> **总览**：B 在全部 4 个场景下**接受率都大幅下降**（-14~-29pp），低负载吞吐持平（±1%），
> **高负载下吞吐/延迟全面恶化**。与 §0.3"接受率 +15~30%、吞吐提升"的预期**完全相反**。

### 5.3 有效性边界结论（2026-08-29/30 实测，含修复后复测）

**B 大优化在两种数据分布、高低负载下均未优于 A 基线，且高负载下明显劣于 A。**
（修复动态 k 负载信号后的复测见 §5.4b，结论不变。）

- **低负载（QPS10）**：B 吞吐与 A 持平（+0.4%~+0.9%），但接受率大幅下降（-16~-20pp）。
  → B 用"更大的 draft 量（动态 k=6）"补偿了低接受率，净吞吐无收益。
- **高负载（QPS50）**：B 全面劣于 A——prefix_repetition 吞吐 -23.9%、spec_bench 延迟 +55%，
  失败请求更多。**B 在高负载下是净损失**。
- **修复后（§5.4b）**：负载信号改用系统 unfinished 后，接受率改善（57.5%→~67%），但
  **吞吐/延迟仍不如 A，B 依旧无优势**。

**结论**：优化文档主张的"接受率 +15~30%"在实测中**未得到验证**，反而系统性下降（-14~-29pp）。
归因结果（§5.4a/§5.4b）：① adaptive 相对 ngram 有固有劣势（prefixrep TPOT +27.7%）；② 动态 k
相对 adaptive 固定 k 有效，但其"高负载保守"信号在当前 vLLM 准入控制架构下**失效**（已知限制）。
**B 大优化（6 优化点全开）当前无法证明相对 A 基线有正向收益。**

### 5.4 内部指标归因（用 §2.4 观测项解释结果）

> 本次实测从服务端 `SuffixDecodeMetrics` 采到了关键内部指标，直接解释 B 为何接受率下降但吞吐持平。

| 观测 | 实测值（B） | 归因 |
|------|------------|------|
| 动态 k：avg_dyn_len | **QPS10 与 QPS50 均为 ~5.9~6.0** | ⚠️ **核心 bug**：动态 k 始终激进（~6，×1.5），**高负载下未按设计调小（应 ~2）**，负载自适应未生效 |
| 动态 k：adjusts / skips | adjusts 数万次，**skips(k=0)=0** | 动态 k 在工作（有调整），但从没调小到 0，说明负载信号未触发保守模式 |
| 接受率 | 57.9%（prefixrep）/ 25.0%（specbench） | **k 调大 → 更难全接受** → 接受率系统性下降 |
| 每步有效 token 数 | 低负载吞吐持平 | B 用"更多 draft 量（k=6）"补偿低接受率，净吞吐无收益 |
| SAM 构建/查询耗时 | 低负载 TPOT 持平 | 短上下文下 proposer 开销被 GPU 掩盖；高负载时才显性（TPOT +22~60%） |

**归因逻辑链**：

```
B 动态 k 恒定 ~6（负载自适应未生效）
   ↓
k 越大，draft 越难全部被接受 → 接受率系统性下降（-14~-29pp）
   ↓
低负载：draft 量大补偿了低接受率 → 吞吐持平（看似无害）
   ↓
高负载：失败投机挤占算力 + 大量无效验证 → 吞吐 -24%、TPOT +22~60%（净损失）
```

> **根因定位（已确认 bug）**：动态 k 的负载信号**设计缺陷**，导致高负载下永远激进投机。

> **Bug 机制**（`gpu_model_runner.py:4576-4579`）：
> ```python
> num_running = len(self.input_batch.req_ids)   # 当前 step 的 batch 大小
> self.drafter.set_load(num_running / max(1, self.max_num_reqs))  # max_num_reqs=128
> ```
> - load 信号用的是**"当前 step 的 batch 请求数 / max_num_seqs(128)"**；
> - 但**单步 batch 大小几乎永远 < 128**（decode 阶段受 token 预算/抢占限制，不会每步塞满），
>   实测 load 总在 **0.2~0.6** 区间；
> - 按 `_effective_k` 阈值（<0.5→×1.5，0.5-0.8→×1.0，≥0.8→×0.5），**load 几乎总是 <0.5 → 永远激进 ×1.5**，
>   **高负载下永不触发保守分支（需 load≥0.8=batch≥102，几乎不可能）**。
>
> **应改用真正的系统负载信号**：
> - scheduler 的 `num_running_reqs + num_waiting_reqs`（有无请求排队 = 是否过载）；
> - 或 KV cache 占用率；或等待队列深度。
> 当前 `gpu_model_runner.py` 无法直接访问 scheduler 统计，需跨模块传递（见 §6 待办）。

### 5.4a 隔离实验：拆解 "adaptive" 与 "动态 k" 的各自贡献（2026-08-29）

> 为厘清 B 差在哪，做三方案隔离对比（QPS10 低负载），**全关 GLOBAL_POOL**：
> A = ngram 固定 k；C = adaptive 固定 k（只开优化 1/2/3/5）；D = adaptive + 动态 k（优化 1/2/3/5/7）。
> 结论：**"动态 k 相对 adaptive 固定 k 有效"得到复现 ✅；"adaptive 相对 ngram 有固有劣势"得到确认 ❌。**

| 数据集 | 指标 | A(ngram) | C(adaptive) | D(+dynk) | C vs A | D vs C |
|--------|------|----------|-------------|----------|--------|--------|
| prefixrep | 接受率 | 77.8% | 71.6% | 65.3% | **-6.2pp** | -6.3pp |
| prefixrep | 接受长度 | 4.06 | 3.80 | 4.77 | -0.26 | **+0.97** |
| prefixrep | 吞吐(tok/s) | 1256.4 | 1261.9 | 1257.0 | +0.4% | 持平 |
| prefixrep | Mean TPOT | 5.93 | 7.57 | 5.95 | **+27.7%** ❌ | **-21.4%** ✅ |
| prefixrep | Mean E2EL | 837 | 1044 | 835 | +24.7% ❌ | **-20.1%** ✅ |
| specbench | 接受率 | 41.1% | 32.3% | 24.6% | **-8.8pp** | -7.7pp |
| specbench | 吞吐(tok/s) | 1227.5 | 1237.0 | 1238.8 | +0.8% | 持平 |
| specbench | Mean TPOT | 6.91 | 7.40 | 6.75 | +7% ⚠️ | **-8.8%** ✅ |
| specbench | Mean E2EL | 854 | 913 | 837 | +6.9% ⚠️ | **-8.3%** ✅ |

**两个独立结论**：

1. **adaptive 相对 ngram 有固有劣势（C vs A）**：
   - 接受率系统性低 6~9pp；
   - **proposer 开销在规则重复数据下显著**：prefix_repetition 下 TPOT +27.7%、E2EL +24.7%
     （SAM 构建 + 多候选评分纯属损失，因为 ngram 已接近上限）。
   - 这是 B 低负载"吞吐持平但接受率降"的直接来源。

2. **动态 k 相对 adaptive 固定 k 确实有效（D vs C）✅——复现之前 §5.5 结论**：
   - TPOT -8.8%~-21.4%、E2EL -8.3%~-20.1%（与 §5.5 的 -15.8%/-14.5% 方向一致）；
   - 接受长度显著提升（prefixrep 3.80→4.77，动态 k=6 每步接受更多 token）。

**总链**：`B 差 = adaptive 固有开销（负，-20~28%） + 动态 k 收益（正，-8~20%）≈ 净持平/略差`。
之前测"优化 7 有效"正确（那是相对 adaptive 固定 k）；但相对 ngram（A），adaptive 本身的
固有差距更大，动态 k 的收益不足以扭转。

### 5.4b Bug 1 修复尝试与复测结果（2026-08-30）

> 针对 §5.4 定位的"动态 k 负载信号缺陷"，实施了一轮修复并复测。**修复使接受率改善，但
> 仍无法让动态 k 在高负载下保守化，B 相对 A 依旧无优势。** 作为已知限制收尾。

**修复内容**（负载信号从"当前 batch"改为"系统 unfinished"）：
- `SchedulerOutput` 新增 `num_unfinished_reqs` 字段（scheduler 每步填 `get_num_unfinished_requests()`）；
- `gpu_model_runner.py` 的 `set_load()` 改用 `num_unfinished_reqs / max_num_seqs`（原为
  `len(input_batch.req_ids) / max_num_seqs`）。

**debug 实测暴露了更深的问题**（prefix_repetition QPS50）：

```
SUFFIX_DEBUG unfinished=111 batch=73 max=256 load=0.434
```

- `unfinished`（running+waiting）= **111**，但 `max_num_seqs` = **256** → load = 0.434 **< 0.5**，
  动态 k 仍走激进分支（×1.5），**保守分支（需 load≥0.8）依旧无法触发**；
- 根因：**`max_num_seqs(256)` 远大于系统实际并发能力（~111，受 `max_num_batched_tokens`/KV 限制）**，
  且 **vLLM 准入控制把过载请求直接拒绝（QPS50 下 282 失败），waiting 队列不会堆积**，
  导致"unfinished/容量"这个信号在任何负载下都到不了 0.8 阈值。

**复测结果**（B-fix，prefix_repetition QPS50）：

| 指标 | B 原（修复前） | B-fix（修复后） | A 基线 |
|------|---------------|----------------|--------|
| 接受率 | 57.5% | **66.5~68%**（↑）| 86.6% |
| 接受长度 | 4.04 | **4.74~5.03**（↑）| 4.39 |
| Output 吞吐（tok/s）| 1919.6 | 1890~1959 | **2521.0** |
| Mean TPOT（ms）| 32.86 | 27.9~29.9 | **26.87** |
| Mean E2EL（ms）| 5977 | 5649~5795 | **4982** |
| 成功/失败 | 219/281 | 218~222/278~282 | 248/252 |

**结论**：
- 修复使**接受率改善**（57.5%→~67%，动态 k 略收敛 + GLOBAL_POOL 作用），但**吞吐/延迟仍不如 A**，
  **B 相对 A 依旧无优势**；
- 修复暴露的**更深限制**：vLLM 准入控制 + `max_num_seqs` 容量虚高，使"running+waiting / 容量"
  型负载信号**在此架构下天然无法触发保守阈值**。需改用 **KV cache 占用率 / 请求失败率** 才可能
  反映真实过载——但这属于重新设计优化 7 的信号源，收益不确定，**本轮不再深入**。

> **最终判定（诚实边界）**：B 大优化（6 优化点全开）在全部测试场景下**均未优于 A 基线**。
> 优化 7（动态 k）相对 adaptive 固定 k 有效（§5.4a），但无法扭转 adaptive 相对 ngram 的
> 固有劣势；其"高负载保守投机"设计在当前 vLLM 准入控制架构下**信号失效**，为已知限制。

### 5.5 历史过程数据（先验参考，非正式 B 大优化结果）

> 以下为正式主测前的**探索性单点测试**，作为先验参考。它们只测了**部分优化点**（非 B 全开），
> 结论不等价于 B 大优化，但揭示了关键趋势，已融入 §0.3 预期。

**5.5.1 优化点 7（动态 k）低负载（prefix_repetition，QPS 10，对照组=固定 k=4）：**
TPOT **-15.8%**、E2EL **-14.5%**，吞吐持平（-0.25% 无回归）。
内部指标：动态 k 把 draft 调到 5.97（×1.5），adjusts=3752，**skips=0**。

**5.5.2 优化点 7 高负载（QPS 50，~60% 失败）：**
TPOT **-12.2%**、E2EL **-9.7%**、吞吐 +3.4%。动态 k 主动调小（保守投机），避免拖累。

**5.5.3 修复的 bug（动态 k 触发）**：`metrics.py observe_draft` 硬断言
`num_accepted_tokens <= num_spec_tokens`（固定 4），动态 k 调大后接受 >4 崩溃。
已改为"接受 ≤ 提出" + 数组动态扩展，修复后不再崩溃。

> ⚠️ 以上仅覆盖**优化点 7** 与**基础 Adaptive（优化 1/2/3/5）**，且未启用优化点 4（GLOBAL_POOL）。
> **B 大优化（全开）的正式结论必须以 §5.2 的两轮主测为准。**

### 5.6 已知限制与诚实边界

1. **数据集决定结论**：优化价值强依赖输入分布。随机 token 几乎无重复 → 接受率低、B 无优势。
2. **proposer 开销 vs 收益**：SAM 构建 + 多候选评分有额外 CPU 开销，短上下文下可能超过收益。
3. **接受率指标准确性**：服务端日志 `Avg Draft acceptance rate` 比客户端推算更权威。
4. **B 大优化的解释性**：多优化点同时开启时，若整体有效/无效，需用 §2.4 内部指标 + §5.4 归因
   定位是哪个优化点在起作用，否则无法指导后续调优。

---

> 测试结果记录于此。

- [x] A 基线（ngram）接受率 + 吞吐（prefix_repetition，接受率 ~76%）→ 迁至 §5.5 历史参考
- [x] B 优化（adaptive）接受率 + 吞吐（prefix_repetition，接受率 ~74%）→ 迁至 §5.5 历史参考
- [x] 第一轮对比分析（prefix_repetition，优化无效）→ 根因（规则重复下 ngram 已接近上限）迁至 §5.5
- [x] spec_bench 数据集已跑通，修复 SpecBench.sample bug
- [x] 优化点 7（动态 k）：低负载 TPOT -15.8%、高负载 TPOT -12.2% → 迁至 §5.5 历史参考
- [x] 修复 metrics.py 断言崩溃 bug（动态 k 触发）
- [x] **主测轮次 1：prefix_repetition，A/B 对比（QPS 10/50）** → §5.2 已填
- [x] **主测轮次 2：spec_bench，A/B 对比（QPS 10/50，核心轮次）** → §5.2 已填
- [x] 采集 §2.4 内部指标，完成 §5.3 结论 + §5.4 归因
- [x] **结论（诚实边界）**：B 大优化 4 场景下接受率全降（-14~-29pp），低负载吞吐持平，高负载净损失
- [x] **隔离实验（§5.4a）**：C(adaptive) vs A(ngram) 确认 adaptive 固有劣势（prefixrep TPOT +27.7%）；D(+dynk) vs C 复现动态 k 有效（TPOT -8.8~-21.4%）
- [x] **deep-dive 内部指标**：spec_bench 下 adaptive **Match Rate 仅 33-37%**、**Avg Match Len 仅 2.14**——多候选评分虽在工作（1.5-1.8万次）但匹配不到长模式，优化 3"匹配更长"卖点未兑现
- [x] **修复 bug 1 并复测（§5.4b）**：负载信号改为 `num_unfinished_reqs`（scheduler 传递），接受率 57.5%→~67%，但 **load=0.434<0.5 仍不触发保守**（max_num_seqs=256 ≫ 实际并发 ~111，准入控制拒绝而非排队）→ 吞吐/延迟仍不如 A，**收尾为已知限制**
- [x] **最终判定**：B 大优化全场景无优势；优化 7"高负载保守投机"信号在 vLLM 准入控制架构下失效
- [ ] **（可选，后续）bug 2（adaptive 匹配率低）**：Avg Match Len=2.14，需检查 `_propose_single` 评分是否偏向短匹配，或 spec_bench 可匹配性低——本轮不深入
- [ ] **（可选，后续）动态 k 信号重设计**：改用 KV cache 占用率 / 请求失败率，可能反映真实过载——收益不确定，本轮不深入
