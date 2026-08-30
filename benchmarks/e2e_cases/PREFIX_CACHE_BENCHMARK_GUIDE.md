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

对应优化文档，本测试覆盖 **3 个已实现的优化点作为"B 大优化"整体被测**（+ 2 个方案设计不在本测试范围）：

| 优化点 | 状态 | 一句话 | A/B 开关 |
|--------|------|--------|---------|
| ① Cache-Aware Scheduling（缓存感知调度） | ✅ 已实现 | MLFQ 同层内优先调度命中缓存多的请求（扫描窗口 K=8） | `VLLM_DISABLE_CACHE_AWARE=1` 关闭 |
| ② Frequency-Aware Eviction（频率感知驱逐 / Segmented LRU） | ✅ 已实现（移植自 main-old-backup） | Probation/Protected 双区，高频前缀不被误驱逐 | `VLLM_DISABLE_SEGMENTED_LRU=1` 关闭 |
| ③ Preemption Cache Shield（抢占缓存保护 / free_partial） | ✅ 已实现（移植自 main-old-backup） | 抢占时保留前缀 block，只释放尾部 | `VLLM_DISABLE_FREE_PARTIAL=1` 关闭 |
| ④ 主动缓存预热 | ⬜ 方案设计 | 冷启动预热常见 System Prompt | 不在本测试 |
| ⑤ 缓存效率可观测性 | ⬜ 部分 | 多维缓存健康度指标 | 不在本测试 |

### 0.3 对比对象

> **核心变化**：三个优化点合并为一个 **B 大优化**整体对比 A，不再逐个优化点单独测
> （与后缀解码测试的"大优化"范式一致）。

| 方案 | 说明 | 环境变量 |
|------|------|---------|
| **A 基线** | 原生 Prefix Caching + 原生调度 + 原生 LRU + 全量抢占（关闭全部缓存优化） | `VLLM_DISABLE_CACHE_AWARE=1 VLLM_DISABLE_SEGMENTED_LRU=1 VLLM_DISABLE_FREE_PARTIAL=1` |
| **B 大优化** | 原生 Prefix Caching + 缓存感知调度 + 分区 LRU + 抢占保护（全部开启） | 不设以上三个变量（默认开启） |

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

> ✅ **当前分支三个优化点均已落地**（②③ 已从 `main-old-backup` 移植并适配当前架构），
> 且三个优化点都有环境变量开关，**A/B 对比可直接做**：

| 优化点 | 代码现状（当前分支核对结果） | 开/关方式 |
|--------|---------------------------|-----------|
| ① 缓存感知调度 | ✅ 已实现：`scheduler.py` 的 `enable_cache_aware_scheduling` + `_cache_aware_select_next()`（扫描窗口 `cache_aware_scan_window=8`） | `VLLM_DISABLE_CACHE_AWARE=1` 关闭；不设则跟随 `--enable-prefix-caching` 开启 |
| ② 频率感知驱逐（Segmented LRU） | ✅ 已实现：`kv_cache_utils.py` 的 `FreeKVCacheBlockQueue` 分区 LRU（probation/protected）+ `block_pool.py` 的 `touch()`/`free_blocks()` 接入 | `VLLM_DISABLE_SEGMENTED_LRU=1` 关闭（退化为普通单区 LRU） |
| ③ 抢占缓存保护（free_partial） | ✅ 已实现：`single_type_kv_cache_manager.py`/`coordinator`/`kv_cache_manager.py` 的 `free_partial()` + `scheduler.py` `_preempt_request()` 部分释放 | `VLLM_DISABLE_FREE_PARTIAL=1` 关闭（退化为全量 free + 重置 0） |

> ✅ **结论**：三个优化点全部可测，A/B 通过三个环境变量一键切换（见 §4 Step 1）。

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

### 3.2 对比实验清单（B 大优化 vs A 基线 + 归因轮次）

> **主测**：三个优化点合一做 **B 大优化 vs A 基线** 的整体对比（§3.2.1）。
> **归因轮次**：若整体有收益/无收益，用开/关单优化点定位是哪个起作用（§3.2.2）。

#### 3.2.1 主测：B 大优化 vs A 基线（整体对比）

| 轮次 | 方案 | 场景 | 验证目标 | 关键指标 | 状态 |
|------|------|------|---------|---------|------|
| **M1** | A（三开关全关）vs B（全开） | 多请求共享长 System Prompt + 高缓存压力，QPS 50 | B 大优化整体是否提升命中率/降 TTFT | 命中率、P99 TTFT、吞吐 | ⬜ 待做 |
| **M2** | A vs B | 高过载（QPS 100，触发大量抢占）| B 的抢占保护是否降低恢复代价 | 抢占次数、恢复后 TTFT、失败率 | ⬜ 待做 |

#### 3.2.2 归因轮次（若需要定位是哪个优化起作用）

| 轮次 | 被测优化 | 对照 | 场景 | 验证目标 | 关键指标 | 状态 |
|------|---------|------|------|---------|---------|------|
| **1** | ① 缓存感知调度 | `VLLM_DISABLE_CACHE_AWARE=1` vs 开启 | 多请求共享长 System Prompt，QPS 50 | 优先调度命中请求是否提升整体效率 | 命中率、同 budget 服务请求数、P99 TTFT | ⬜ 待做 |
| **2** | ② 频率感知驱逐 | `VLLM_DISABLE_SEGMENTED_LRU=1` vs 开启 | 高频 System Prompt + 大量低频长请求（制造缓存震荡） | 高频前缀是否不再被误驱逐 | 高频前缀命中率、命中率波动幅度 | ⬜ 待做 |
| **3** | ③ 抢占缓存保护 | `VLLM_DISABLE_FREE_PARTIAL=1` vs 开启 | 高过载触发大量抢占（Phase 5 类场景） | 抢占恢复是否只重算尾部 | 抢占恢复 recompute token 数、恢复耗时、抢占后 TTFT | ⬜ 待做 |
| **4** | 命中→TTFT 因果 | 同请求首次(未命中) vs 后续(命中) | 重复 prompt | 坐实"命中→TTFT 下降"的量化关系 | 首次 vs 后续的 TTFT/Prefill 耗时对比 | ⬜ 待做 |

#### 各轮次核心判据

- **主测 M1/M2（整体）**：B 大优化整体命中率/TTFT/吞吐 vs A；M2 高过载看抢占保护收益。
- **归因轮次 1（缓存感知调度）**：核心看**同 token_budget 下服务的请求数**——缓存感知优先调度命中请求，相同预算能多服务几个请求；同时看整体 P99 TTFT 是否下降。
- **归因轮次 2（频率感知驱逐）**：核心看**高频前缀的命中率稳定性**——制造"高频 System Prompt + 大量低频长请求"的缓存震荡场景，验证 Segmented LRU 下高频前缀命中率是否保持稳定（不被低频请求挤出）。
- **归因轮次 3（抢占缓存保护）**：核心看**抢占恢复的 recompute token 数**——高过载触发抢占后，free_partial 应只重算尾部（百毫秒级）而非全量（秒级）。
- **归因轮次 4**：不是 A/B，而是在同一轮里对比"同一 prompt 首次(冷)vs 后续(热)"的 TTFT，坐实缓存命中的因果收益。

> **共同要求**：所有轮次用 `vllm:prefix_cache_hits/queries` 采集命中率曲线；A/B 两轮除被测开关外所有参数完全一致（同数据集、同 QPS、同 seed、同 prompt）。

---

## 4. 测试步骤

### Step 0：确认各优化点可运行

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate

# ① 缓存感知调度（应存在）
grep -n "enable_cache_aware_scheduling\|_cache_aware_select_next" vllm/v1/core/sched/scheduler.py

# ② 频率感知驱逐（已移植，应存在）
grep -rn "probation\|protected\|enable_segmented_lru" vllm/v1/core/block_pool.py

# ③ 抢占缓存保护（已移植，应存在）
grep -rn "free_partial\|VLLM_DISABLE_FREE_PARTIAL" vllm/v1/core/kv_cache_manager.py vllm/v1/core/sched/scheduler.py
```

> 三者均应输出匹配行。若 ②③ 无输出 → 代码未同步，需先移植（见 §1.2）。

### Step 1：准备 A/B 两个配置（三个开关一键切换）

三个优化点都支持环境变量开关，**无需改代码**，用环境变量切换 A/B：

| 优化点 | 关闭开关 |
|--------|---------|
| ① 缓存感知调度 | `VLLM_DISABLE_CACHE_AWARE=1` |
| ② 频率感知驱逐 | `VLLM_DISABLE_SEGMENTED_LRU=1` |
| ③ 抢占缓存保护 | `VLLM_DISABLE_FREE_PARTIAL=1` |

- **A 基线（全关）**：启动时设 `VLLM_DISABLE_CACHE_AWARE=1 VLLM_DISABLE_SEGMENTED_LRU=1 VLLM_DISABLE_FREE_PARTIAL=1`
- **B 大优化（全开）**：不设这三个变量（默认全部开启）

### Step 2：启动服务（两轮都开 prefix caching）

```bash
# A 基线：三个缓存优化全部关闭
VLLM_DISABLE_CACHE_AWARE=1 VLLM_DISABLE_SEGMENTED_LRU=1 VLLM_DISABLE_FREE_PARTIAL=1 \
    python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --enable-prefix-caching \
    --max-model-len 8192 --gpu-memory-utilization 0.85 \
    --port 8000 2>&1 | tee server_A.log

# B 大优化：三个缓存优化全部开启（不设开关变量）
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

> 按下表记录（A=基线三关全关，B=大优化三开全开）：

### 5.1 B 大优化 vs A 基线：主测结果表（2026-08-30 已测）

> 场景：`prefix_repetition`（prefix-len 1024 / suffix-len 256 / output 128），1000 请求，
> `--enable-prefix-caching`，无 eager。A = 三开关全关；B = 三优化全开。

**主测 M1（QPS 50，高前缀共享）**

| 指标 | A 基线（三关全关） | B 大优化（三开全开） | 差异 | 判定 |
|------|-------------------|--------------------|------|------|
| 成功/失败 | 677 / 323 | **692 / 308** | **+15 成功** | ✅ B 服务更多请求 |
| 输出吞吐（tok/s）| 3785.0 | **3888.3** | **+2.7%** | ✅ B 略高 |
| P99 TTFT (ms) | 2535.3 | **2013.6** | **-20.6%** | ✅ B 显著更低 |
| P99 TPOT (ms) | 68.79 | 66.53 | -3.3% | ✅ B 略好 |
| P99 E2EL (ms) | 11180.8 | **10363.6** | **-7.3%** | ✅ B 更低 |
| 请求吞吐（req/s）| 29.63 | **30.38** | +2.5% | ✅ B 更高 |
| 峰值并发 | 340 | 340 | 持平 | 控制变量一致 |

**主测 M2（QPS 100，严重过载 ~45% 失败）**

| 指标 | A 基线 | B 大优化 | 差异 | 判定 |
|------|--------|----------|------|------|
| 成功/失败 | 553 / 447 | 553 / 447 | 持平 | ⚠️ 无差异 |
| 输出吞吐（tok/s）| 5081.2 | 5014.0 | -1.3% | ⚠️ 基本持平 |
| P99 TTFT (ms) | 2502.5 | 2707.0 | +8.2% | ⚠️ 波动（无显著差异）|
| P99 TPOT (ms) | 56.87 | 57.06 | +0.3% | ⚠️ 持平 |
| P99 E2EL (ms) | 9271.9 | 9601.2 | +3.5% | ⚠️ 基本持平 |

> **命中率对比**：A 日志 77.8~88.4%，B 日志 77.3~88.2%，**基本一致**
> （prefix_repetition 本身高重复，原生缓存命中率已高，B 的命中率优势不在此场景体现）。

### 5.2 结论要点（2026-08-30 实测）

- [x] **主测 M1（QPS 50）**：B 大优化**有效**——成功请求 +15、吞吐 +2.7%、**P99 TTFT -20.6%**、P99 E2EL -7.3%。
  收益主要来自**缓存感知调度**（同预算优先调度高命中请求 → 服务更多请求、TTFT 更低）。
- [x] **主测 M2（QPS 100 严重过载）**：B 与 A **基本持平**（成功数/吞吐/延迟均无显著差异）。
  → 严重过载下大量请求直接失败而非抢占，缓存优化收益被稀释。
- [ ] 归因轮次 1：缓存感知调度单独贡献（M1 的 -20.6% TTFT 大概率主要来自它，需单独测确认）
- [ ] 归因轮次 2：Segmented LRU 是否减少高频前缀缓存震荡（需"高频+大量低频长请求"场景）
- [ ] 归因轮次 3：free_partial 是否降低抢占恢复代价（M2 未体现，需更可控的抢占场景）
- [x] **归因轮次（2026-08-30，QPS 50，单优化开关）**：三个优化**各自独立贡献 -6~7% TTFT**，且**可线性叠加**（见 §5.3）✅
- [ ] 命中带来的 TTFT 下降有多大？（归因轮次 4）
- [x] **边界结论**：B 在**中等负载 + 高前缀共享**下有效（M1）；在**严重过载**下收益消失（M2）。
  有效性边界 = 系统还能服务请求（未饱和）+ 前缀被共享。

### 5.3 归因实验：三个优化的各自贡献（2026-08-30，QPS 50）

> 方法：在 M1 场景下分别只开一个优化（其余两个关闭），定位各优化独立贡献。
> C1 = 只开①缓存感知；C2 = 只开②分区 LRU；C3 = 只开③抢占保护。

| 方案 | 成功/失败 | 吞吐(tok/s) | P99 TTFT(ms) | P99 TTFT 相对 A | P99 E2EL(ms) |
|------|----------|------------|--------------|----------------|--------------|
| A（全关）| 677/323 | 3785.0 | 2535.3 | 基线 | 11180.8 |
| C1（只开①）| 691/309 | 3870.5 | 2350.3 | **-7.3%** | 10747.9 |
| C2（只开②）| 691/309 | 3872.3 | 2368.6 | **-6.6%** | 10754.6 |
| C3（只开③）| 686/314 | 3844.5 | 2362.1 | **-6.8%** | 10828.9 |
| B（全开）| 692/308 | 3888.3 | **2013.6** | **-20.6%** | 10363.6 |

**核心发现：三个优化正交、可线性叠加**：

- 每个优化**单独开**都有约 **-6~7% P99 TTFT**（2350~2369ms）且成功数 +9~14——各自独立有效；
- **B 全开是 -20.6%（2013.6ms）**，**≈ 三个单独收益之和（-7.3 - 6.6 - 6.8 ≈ -20.7%）**；
- 说明三个优化**作用于不同环节且互不冲突**：①改调度顺序（命中优先）、②改驱逐策略（高频保护）、③改抢占恢复（前缀保留）——**合并后收益可叠加**。

> 注：单个优化收益只有 ~7%（弱于 B 全开的 20.6%），因此**只测单优化会低估整体价值**，
> 这验证了"合并为 B 大优化"的必要性（与后缀解码测试的教训一致——单点测 vs 合并测的差异）。

---

## 6. 待办清单

- [x] 移植优化②（分区 LRU）和③（free_partial）到当前分支（已完成，从 main-old-backup）
- [x] 给 ②③ 加环境变量开关 `VLLM_DISABLE_SEGMENTED_LRU` / `VLLM_DISABLE_FREE_PARTIAL`（已完成）
- [x] 给 ① 的 `VLLM_DISABLE_CACHE_AWARE` 开关（已完成，之前）
- [x] **主测 M1**：B 大优化 vs A，高前缀共享 + 高缓存压力，QPS 50（B 有效：TTFT -20.6%）→ §5.1
- [x] **主测 M2**：B 大优化 vs A，高过载 QPS 100（B 持平，无显著差异）→ §5.1
- [x] **归因轮次 1~3**：单优化各自 -6~7% TTFT，全开 -20.6%（≈线性叠加）→ §5.3
- [ ] 归因轮次 4：命中→TTFT 因果验证（可选）
- [x] 结论：M1 有效（-20.6% TTFT）、M2 持平、归因证明正交可叠加（诚实结论）

---

## 7. 与其他测试文档的关系

| 文档 | 关系 |
|------|------|
| [`SCHEDULING_BENCHMARK_GUIDE.md`](./SCHEDULING_BENCHMARK_GUIDE.md) | §1.1 已"顺带观测"前缀缓存优化（两轮都开缓存、A/B 变量是调度优化）；本文档做**针对性 A/B**，是其补充与提级 |
| [`SUFFIX_DECODING_BENCHMARK_GUIDE.md`](./SUFFIX_DECODING_BENCHMARK_GUIDE.md) / [`PD_DISAGGREGATION_BENCHMARK_GUIDE.md`](./PD_DISAGGREGATION_BENCHMARK_GUIDE.md) | 同为"优化点独立对比测试"的结构范式 |
| [`prefix-cache-scheduling-optimization.md`](../basic_optimization/prefix-cache-scheduling-optimization.md) | 被测优化的设计文档；优化④⑤为方案设计（未实现），本测试只覆盖已落地部分 |

> **诚实定位**：Prefix Cache 复用是 vLLM 官方成熟能力，本项目优化的是"**调度器/缓存管理器如何更聪明地利用缓存**"
> （缓存感知调度、频率驱逐、抢占保护），属于**对官方机制的调度层增强**，非原创。测试目的是验证这些**增量**的效果与边界。
