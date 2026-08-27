# Prefix Cache 调度优化 — 测试方案

> 目标：验证 [`prefix-cache-scheduling-optimization.md`](../basic_optimization/prefix-cache-scheduling-optimization.md)
> 中的 Prefix Cache **调度层**优化（缓存感知调度 / 频率感知驱逐 / 抢占缓存保护）相对
> vLLM V1 **原生 Prefix Caching** 的增量效果。
>
> 核心指标：**缓存命中率**、**TTFT（首 token 时间）**、**Prefill 耗时**、**缓存震荡程度**、**抢占恢复代价**。

> ⚠️ **与 `SCHEDULING_BENCHMARK_GUIDE.md` 的关系**：调度测试文档 §1.1 已把 Prefix Cache 优化"并入"验证，
> 但那里是**借 Phase 场景顺带观测**（两轮都开原生缓存、A/B 变量是"调度优化开/关"而非"缓存优化开/关"），
> **测不出前缀缓存优化的独立增量**。本文档把它**提级为独立验证**，做针对性 A/B 对照。

---

## 0. 背景与原理（先理解再测）

### 0.1 要区分三层"缓存"

| 层次 | 是什么 | 谁提供 | 本测试的角色 |
|------|--------|--------|-------------|
| **原生 Prefix Caching** | 相同前缀的 KV block 复用（hash 链 + 哈希表） | vLLM V1 官方 | **控制变量**（两轮都开） |
| **缓存感知调度**（优化①） | 调度器优先挑"命中率高"的请求 | 本项目 | **被测变量** |
| **缓存管理策略**（优化②③） | 频率感知驱逐 + 抢占缓存保护 | 本项目 | **被测变量** |

> **核心澄清**：本测试**不是测"有没有前缀缓存"**（那是官方能力，两轮都开），
> 而是测"**调度器/缓存管理器如何更聪明地利用缓存**"的增量。

### 0.2 被测的优化点

对应优化文档，本测试覆盖 3 个**已实现**优化点 + 2 个**方案设计**：

| 优化点 | 状态 | 一句话 |
|--------|------|--------|
| ① Cache-Aware Scheduling（缓存感知调度） | ✅ 已实现 | MLFQ 同层内优先调度命中缓存多的请求（扫描窗口 K=8） |
| ② Frequency-Aware Eviction（频率感知驱逐 / Segmented LRU） | ⚠️ 见 §1.2 | Probation/Protected 双区，高频前缀不被误驱逐 |
| ③ Preemption Cache Shield（抢占缓存保护 / free_partial） | ⚠️ 见 §1.2 | 抢占时保留前缀 block，只释放尾部 |
| ④ 主动缓存预热 | ⬜ 方案设计 | 冷启动预热常见 System Prompt |
| ⑤ 缓存效率可观测性 | ⬜ 部分 | 多维缓存健康度指标 |

### 0.3 对比对象

| 方案 | 说明 |
|------|------|
| **A 基线** | 原生 Prefix Caching + 原生调度（关闭本项目缓存优化） |
| **B 优化** | 原生 Prefix Caching + 本项目缓存感知调度 / 频率驱逐 / 抢占保护 |

### 0.4 预期效果（来自优化文档）

| 指标 | 预期 |
|------|------|
| 高命中请求 TTFT | 降低约 30~50% |
| 高频前缀命中率 | 从 ~50-70% 提升到 ~85-95%（减少缓存震荡） |
| 抢占恢复代价 | 从全量 Recompute（秒级）降至部分 Recompute（百毫秒级） |

---

## 1. 前提与实现现状（重要：先读）

### 1.1 环境

- RTX 5070 Ti（Blackwell，sm_120），单卡
- Qwen2.5-1.5B-Instruct（或更大模型如 Qwen3-32B 更能体现长 prompt 命中收益）
- **无 `--enforce-eager`**（用默认 CUDA graph）
- **两轮都必须 `--enable-prefix-caching`**（原生缓存作为控制变量）

### 1.2 实现现状与开关方式（据实说明，避免踩坑）

做测试前必须正视——当前分支的实现状态并不完整，直接影响能否做 A/B：

| 优化点 | 代码现状（当前分支核对结果） | 开/关方式 |
|--------|---------------------------|-----------|
| ① 缓存感知调度 | ✅ 已实现：`scheduler.py` 的 `enable_cache_aware_scheduling` + `_cache_aware_select_next()`（扫描窗口 `cache_aware_scan_window=8`） | ✅ **已加环境变量开关 `VLLM_DISABLE_CACHE_AWARE`**——设 `=1` 关闭、不设则跟随 `--enable-prefix-caching` 开启，可直接做 A/B（见 §4 Step 1） |
| ② 频率感知驱逐（Segmented LRU） | ❌ **当前分支未检索到** probation/protected 实现（`block_pool.py` 只有普通 LRU） | 需先落地代码才能测（可能在另一分支，同 PD 情况） |
| ③ 抢占缓存保护（free_partial） | ❌ **当前分支未检索到** `free_partial`（`kv_cache_manager.py` 只有普通 `free()`） | 需先落地代码才能测 |

> ⚠️ **结论**：当前分支**只有优化①（缓存感知调度）可测**，且需改代码切开关；优化②③在当前分支未落地。
> 本文档作为**完整测试设计**，②③部分待代码同步后执行（参考 PD 文档 Step 0 的"先确认功能可运行"思路）。

---

## 2. 核心指标与数据源

### 2.1 缓存专属指标（最关键）

| 指标 | 数据源 | 含义 | 优化预期 |
|------|--------|------|---------|
| **缓存命中率** | `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` | 命中的 block 占比 | B 应更高（尤其高频前缀） |
| **命中带来的 TTFT 下降** | `vllm bench serve` TTFT | 命中 vs 未命中的首 token 时间 | 命中请求 TTFT 显著更低 |
| **Prefill 耗时** | `vllm:request_prefill_time_seconds` | 命中时应变短 | 命中→prefill token 少→耗时短 |
| **缓存震荡（高频前缀命中率波动）** | 命中率时间序列 | 高频 System Prompt 是否被误驱逐 | B（Segmented LRU）应更平稳 |
| **抢占恢复 recompute token 数** | 服务端日志 / 埋点 | 抢占恢复时重算多少 token | B（free_partial）应更少（只重算尾部） |

### 2.2 端到端指标

| 指标 | 数据源 |
|------|--------|
| 吞吐（req/s, tok/s） | `vllm bench serve` |
| P99 TTFT / TPOT / E2EL | `vllm bench serve` |
| 同 token_budget 下服务的请求数 | 调度器日志 |

### 2.3 服务端内部指标（Prometheus + Grafana）

复用调度测试文档已搭好的监控：
- `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`（命中率曲线）
- KV Cache 使用率、请求等待时间、抢占次数

---

## 3. 测试设计

### 3.1 测试矩阵

| 维度 | 取值 | 说明 |
|------|------|------|
| 方案 | A（基线）/ B（缓存优化） | 核心对比 |
| 负载 | 恒定 QPS（10/50/100） | 命中收益在有并发共享前缀时更明显 |
| **前缀共享度** | 高（多请求共享长 System Prompt）/ 低（各请求前缀不同） | ⭐关键：缓存优化的收益**强依赖前缀是否被共享** |
| **prompt 长度** | 短(256) / 长(2K+，如 4k) | 长共享前缀命中收益最大（优化文档在 Qwen3-32B 4k 上测得 TTFT-50%） |
| 缓存压力 | 低（缓存放得下）/ 高（触发驱逐） | 高压力才能测出②频率驱逐、③抢占保护的价值 |

> **核心设计思想**：Prefix Cache 优化的收益**强依赖"前缀被共享 + 缓存有压力"**。
> - 前缀不共享 → 没缓存可复用 → 优化无从发挥；
> - 缓存放得下（无驱逐/抢占）→ 优化②③无用武之地。
> 因此测试必须构造 **"高前缀共享 + 高缓存压力"** 的场景，才能暴露优化价值。

### 3.2 针对性对照实验清单（每项单独验证一个优化点）

> 这是本文档相对调度测试的关键增量——**对每个优化点做开/关 A/B**，测出独立增量。

| 轮次 | 被测优化 | 对照 | 场景 | 验证目标 | 关键指标 | 状态 |
|------|---------|------|------|---------|---------|------|
| **1** | ① 缓存感知调度 | 关闭 vs 开启 `enable_cache_aware_scheduling` | 多请求共享长 System Prompt，QPS 50 | 优先调度命中请求是否提升整体效率 | 命中率、同 budget 服务请求数、P99 TTFT | ⬜ 待做 |
| **2** | ② 频率感知驱逐 | 原生 LRU vs Segmented LRU | 高频 System Prompt + 大量低频长请求（制造缓存震荡） | 高频前缀是否不再被误驱逐 | 高频前缀命中率、命中率波动幅度 | ⬜ 待代码 |
| **3** | ③ 抢占缓存保护 | 全量 free vs free_partial | 高过载触发大量抢占（Phase 5 类场景） | 抢占恢复是否只重算尾部 | 抢占恢复 recompute token 数、恢复耗时、抢占后 TTFT | ⬜ 待代码 |
| **4** | 命中→TTFT 因果 | 同请求首次(未命中) vs 后续(命中) | 重复 prompt | 坐实"命中→TTFT 下降"的量化关系 | 首次 vs 后续的 TTFT/Prefill 耗时对比 | ⬜ 待做 |

#### 各轮次核心判据

- **轮次 1（缓存感知调度）**：核心看**同 token_budget 下服务的请求数**——缓存感知优先调度命中请求，相同预算能多服务几个请求；同时看整体 P99 TTFT 是否下降。
- **轮次 2（频率感知驱逐）**：核心看**高频前缀的命中率稳定性**——制造"高频 System Prompt + 大量低频长请求"的缓存震荡场景，验证 Segmented LRU 下高频前缀命中率是否保持稳定（不被低频请求挤出）。
- **轮次 3（抢占缓存保护）**：核心看**抢占恢复的 recompute token 数**——高过载触发抢占后，free_partial 应只重算尾部（百毫秒级）而非全量（秒级）。
- **轮次 4**：不是 A/B，而是在同一轮里对比"同一 prompt 首次(冷)vs 后续(热)"的 TTFT，坐实缓存命中的因果收益。

> **共同要求**：所有轮次用 `vllm:prefix_cache_hits/queries` 采集命中率曲线；A/B 两轮除被测开关外所有参数完全一致（同数据集、同 QPS、同 seed、同 prompt）。

---

## 4. 测试步骤

### Step 0：确认各优化点是否可运行

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate

# ① 缓存感知调度（应存在）
grep -n "enable_cache_aware_scheduling\|_cache_aware_select_next" vllm/v1/core/sched/scheduler.py

# ② 频率感知驱逐（当前分支可能缺失）
grep -rn "probation\|protected\|segmented" vllm/v1/core/block_pool.py

# ③ 抢占缓存保护（当前分支可能缺失）
grep -rn "free_partial" vllm/v1/core/kv_cache_manager.py
```

> 若 ②③ 无输出 → 当前分支未落地，仅执行轮次 1/4；②③ 待代码同步后补测。

### Step 1：准备 A/B 两个配置（缓存感知调度的开/关）

`enable_cache_aware_scheduling` 已支持环境变量开关 `VLLM_DISABLE_CACHE_AWARE`（已加入 `scheduler.py`），
**无需改代码**，直接用环境变量切换 A/B：

```python
# vllm/v1/core/sched/scheduler.py（已实现）
self.enable_cache_aware_scheduling = (
    self.cache_config.enable_prefix_caching
    and os.environ.get("VLLM_DISABLE_CACHE_AWARE") != "1")
```

- **A 基线（关闭缓存感知）**：启动时设 `VLLM_DISABLE_CACHE_AWARE=1`
- **B 优化（开启缓存感知）**：不设该变量（默认跟随 `--enable-prefix-caching` 开启）

### Step 2：启动服务（两轮都开 prefix caching）

```bash
# A 基线：关闭缓存感知调度
VLLM_DISABLE_CACHE_AWARE=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --enable-prefix-caching \
    --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --port 8000 2>&1 | tee server_A.log

# B 优化：开启缓存感知调度（不设 VLLM_DISABLE_CACHE_AWARE）
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --enable-prefix-caching \
    --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --port 8000 2>&1 | tee server_B.log
```

### Step 3：启动监控

```bash
# 复用 SCHEDULING_BENCHMARK_GUIDE.md 的 Prometheus + Grafana
# 重点看 vllm:prefix_cache_hits / vllm:prefix_cache_queries
```

### Step 4：跑压测（构造高前缀共享场景）

```bash
# 用 sharegpt（自然共享对话前缀）或 random + 固定长 prefix 构造高共享度
vllm bench serve \
    --backend openai --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --dataset-name prefix_repetition \
    --prefix-repetition-prefix-len 1024 --prefix-repetition-suffix-len 256 \
    --num-prompts 1000 --request-rate 50 --temperature 0 --ignore-eos \
    --percentile-metrics ttft,tpot,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir results/prefix_cache_<A|B>/ \
    --result-filename bench.json --metadata variant=<baseline|cache_aware>
```

### Step 5：分析

```bash
# 对比 A/B 的命中率、P99 TTFT、同 budget 服务请求数
grep -i "prefix_cache" server_A.log server_B.log
# Grafana 看命中率曲线是否 B 更高更稳
```

---

## 5. 测试结果记录与结论

> 待执行。按下表记录（A=基线，B=缓存优化）：

### 5.1 缓存感知调度对比（轮次 1）

| 指标 | A 基线 | B 缓存感知 | 差异 | 判定 |
|------|--------|-----------|------|------|
| 缓存命中率 | 待测 | 待测 | | B 更高=有效 |
| P99 TTFT (ms) | 待测 | 待测 | | B 更低=有效 |
| 同 budget 服务请求数 | 待测 | 待测 | | B 更多=有效 |
| 吞吐 (tok/s) | 待测 | 待测 | | 不应退化 |

### 5.2 结论要点（待填）

- [ ] 缓存感知调度是否提升了命中率 / 降低了 TTFT？（轮次1）
- [ ] Segmented LRU 是否减少了高频前缀的缓存震荡？（轮次2，待代码）
- [ ] free_partial 是否降低了抢占恢复代价？（轮次3，待代码）
- [ ] 命中带来的 TTFT 下降有多大？（轮次4）
- [ ] 优化在什么"前缀共享度/缓存压力"下才有价值？（边界结论）

---

## 6. 待办清单

- [ ] Step 0：确认 ②③ 优化点是否在当前分支落地（缺则从另一分支同步）
- [x] 给 `enable_cache_aware_scheduling` 加环境变量开关 `VLLM_DISABLE_CACHE_AWARE`（已完成）
- [ ] 轮次 1：缓存感知调度 A/B（当前可做，用 `VLLM_DISABLE_CACHE_AWARE` 切换）
- [ ] 轮次 4：命中→TTFT 因果验证（当前可做）
- [ ] 轮次 2：频率感知驱逐 A/B（待代码）
- [ ] 轮次 3：抢占缓存保护 A/B（待代码）
- [ ] 结论：前缀缓存优化的有效场景与边界（诚实结论）

---

## 7. 与其他测试文档的关系

| 文档 | 关系 |
|------|------|
| [`SCHEDULING_BENCHMARK_GUIDE.md`](./SCHEDULING_BENCHMARK_GUIDE.md) | §1.1 已"顺带观测"前缀缓存优化（两轮都开缓存、A/B 变量是调度优化）；本文档做**针对性 A/B**，是其补充与提级 |
| [`SUFFIX_DECODING_BENCHMARK_GUIDE.md`](./SUFFIX_DECODING_BENCHMARK_GUIDE.md) / [`PD_DISAGGREGATION_BENCHMARK_GUIDE.md`](./PD_DISAGGREGATION_BENCHMARK_GUIDE.md) | 同为"优化点独立对比测试"的结构范式 |
| [`prefix-cache-scheduling-optimization.md`](../basic_optimization/prefix-cache-scheduling-optimization.md) | 被测优化的设计文档；优化④⑤为方案设计（未实现），本测试只覆盖已落地部分 |

> **诚实定位**：Prefix Cache 复用是 vLLM 官方成熟能力，本项目优化的是"**调度器/缓存管理器如何更聪明地利用缓存**"
> （缓存感知调度、频率驱逐、抢占保护），属于**对官方机制的调度层增强**，非原创。测试目的是验证这些**增量**的效果与边界。
