# 基于 vLLM V1 的 LLM 推理引擎全栈优化

> 覆盖调度与资源管理、KV Cache 管理、投机解码、分布式架构（PD 分离）四大方向，基于 vLLM V1 源码二次开发

## 项目概述

本项目基于 **vLLM V1 架构**（v0.7.3），对 LLM 推理引擎进行系统级优化。将云网络（流量调度/优先级转发）、云存储（时延优化/限速流控/分层存储）的核心 Infra 经验迁移至大模型推理场景，从**调度策略、KV Cache 管理、投机解码、分布式架构**四大维度全面提升推理服务的时延、吞吐和稳定性。

### 项目定位

LLM 推理优化的完整图谱包含 6 大方向。本项目聚焦其中 4 个，覆盖推理链路的全部关键环节：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         LLM 推理优化全景图                                     │
├────────────────────────┬─────────────────────────────────────────────────────┤
│ 1. 调度与资源管理       │ ✅ 本项目核心方向 — 8 个优化点（3 已实现）            │
│ 2. KV Cache 管理       │ ✅ 本项目核心方向 — 5 个优化点（3 已实现）            │
│ 3. 投机解码            │ ✅ 本项目设计方向 — 5 个优化点（设计完成）            │
│ 4. 分布式架构(PD分离)   │ ✅ 本项目规划方向 — 6 个优化点（设计完成）            │
│ 5. 模型压缩/量化       │ ⬜ 不在当前范围（需模型侧配合）                       │
│ 6. Attention/算子优化   │ ⬜ 不在当前范围（FlashAttention 等已由社区覆盖）      │
├────────────────────────┼─────────────────────────────────────────────────────┤
│ ★ 端到端业务场景验证    │ ✅ 5 阶段递进加压 + 5 项增量修复（workload 已完成）   │
│                        │    从框架优化 → 业务优化的完整闭环                    │
└────────────────────────┴─────────────────────────────────────────────────────┘
```

### V1 架构关键特征（本项目基于此开发）

| 特征 | V1 现状 | 本项目优化 |
|------|---------|-----------|
| 调度策略 | 仅 FCFS（先来先服务） | QoS 分级 + MLFQ 自适应 + Token 限速 |
| 抢占策略 | LIFO（Recompute 代价极高） | 优先级感知抢占 + 缓存保护 |
| KV Cache 管理 | Prefix Caching + 简单 LRU | Cache-Aware 调度 + Segmented LRU + 抢占保护 |
| 投机解码 | 仅 N-gram Proposer | 后缀树 Proposer + 增量后缀自动机 |
| PD 分离 | 仅 V0 支持，V1 无任何代码 | V1 适配 + 智能路由 + 传输优化 |

---

## 优化方向总览

本项目包含 **4 大优化方向 + 1 个端到端验证框架**，共 **24 个优化点**。每个方向有独立的详细设计文档：

### 方向一：调度与资源管理（8 个优化点）

> **核心思路**：将云网络/存储的调度、限速、时延优化能力迁移至推理引擎调度器

| # | 优化点 | 优先级 | 状态 | Infra 能力对标 |
|---|--------|--------|------|---------------|
| 1 | QoS 分级调度 | P0 | ✅ 已实现 | 网关优先级队列 + 高优包优先转发 |
| 2 | KV 缓存显存水位线与配额流控 | P0 | 🔲 未实现 | 存储水位线 + IO 配额 |
| 3 | 请求准入控制 | P1 | 🔲 未实现 | ECN 显式拥塞通知 + RED 随机早期丢弃 |
| 4 | Token 级速率控制 | P1 | ✅ 已实现 | 令牌桶 / 漏桶限速 |
| 5 | Deadline-aware 调度 | P2 | 🔲 未实现 | EDF + fq_codel / HFSC |
| 6 | 加权公平队列 WFQ 调度 | P2 | 🔲 未实现 | WFQ / DRR + per-tenant 配额 |
| 7 | MLFQ 多级反馈队列 | P3 | ✅ 已实现 | OS MLFQ / CFS |
| 8 | KV Cache 分层存储与智能迁移 | P1 | 🔲 未实现 | 存储热温冷分层 + 预取 |

**已实现优化的核心设计**：

- **QoS 分级调度**：综合 API 传入的业务优先级 + prompt 长度分级 + 等待时间衰减因子，动态计算 `effective_priority`，保障短请求低时延
- **Token 限速**：Per-request 令牌桶机制，高负载下低优请求限速（rate=8-50 tokens/step），高优请求不限速，避免硬性抢占
- **MLFQ 多级反馈**：4 级队列（Interactive/Standard/Batch/Background），请求按实际 token 消耗自动降级，无需人工标注优先级

**调度四层架构**：

```
请求到达
    │
    ▼
优化 6（WFQ）：按 tenant_id 分流 → 保证租户间公平
    │
    ▼
优化 1（QoS）：同一租户内按 effective_priority 排序
    │
    ▼
优化 7（MLFQ）：同优先级内自适应排序（短请求自动优先）
    │
    ▼
请求被选中，进入 running 状态
    │
    ▼
优化 4（Token 限速）：低优请求限速，为高优让出 token_budget
    │
    ▼
输出 tokens
```

**核心修改文件**：
- `vllm/v1/core/scheduler.py` — 调度器核心逻辑
- `vllm/v1/request.py` — Request 类扩展（优先级、MLFQ 级别、速率限制器）

> 📄 调度与资源管理的详细设计见本文件下方的[详细设计章节](#调度与资源管理详细设计)

---

### 方向二：KV Cache 管理（5 个优化点）

> **核心思路**：围绕 vLLM V1 的 Prefix Caching 机制做深度优化，提升缓存命中率、降低 TTFT

| # | 优化点 | 优先级 | 状态 | 核心价值 |
|---|--------|--------|------|---------|
| 1 | Cache-Aware Scheduling | P0 | ✅ 已实现 | MLFQ 层内按缓存命中率排序，token_budget 利用率↑ |
| 2 | Frequency-Aware Eviction (Segmented LRU) | P0 | ✅ 已实现 | 高频前缀不被误驱逐，缓存命中率↑ |
| 3 | Preemption Cache Shield | P0 | ✅ 已实现 | 抢占时保留前缀缓存，恢复代价↓ |
| 4 | Proactive Cache Warming | P1 | 🔲 未实现 | 冷启动时主动预热高频前缀 |
| 5 | Cache Efficiency Dashboard | P2 | 🔲 未实现 | 全链路缓存指标可观测 |

**已实现优化的核心设计**：

- **Cache-Aware Scheduling**：调度时优先调度 Prefix Cache 命中率高的请求（`computed_tokens / prompt_tokens`），让命中的请求少做 Prefill 计算，节省 token_budget
- **Segmented LRU**：将 `FreeKVCacheBlockQueue` 分为 probation 和 protected 两段，高频 block 自动晋升到 protected，不被首次驱逐
- **Preemption Cache Shield**：抢占时只释放尾部 blocks，保留前缀（System Prompt）缓存，恢复时只需重算尾部

**核心修改文件**：
- `vllm/v1/core/kv_cache_manager.py` — KV Cache 管理器
- `vllm/v1/core/kv_cache_utils.py` — 缓存工具函数

> 📄 详细设计文档：[`prefix-cache-scheduling-optimization.md`](prefix-cache-scheduling-optimization.md)

---

### 方向三：投机解码 — 后缀解码（5 个优化点）

> **核心思路**：用后缀树（Suffix Tree）替换 vLLM V1 的 N-gram Proposer，提升 Decode 阶段的投机效率

| # | 优化点 | 优先级 | 状态 | 核心价值 |
|---|--------|--------|------|---------|
| 1 | SuffixTreeProposer | P0 | 🔲 设计完成 | 最长后缀匹配，替换固定 N-gram |
| 2 | 增量后缀自动机 (Incremental SAM) | P0 | 🔲 设计完成 | O(1) 增量更新，避免每步重建 |
| 3 | 自适应匹配策略 | P1 | 🔲 设计完成 | 动态调整匹配长度和 draft 数量 |
| 4 | 跨请求共享后缀树 | P2 | 🔲 设计完成 | 利用相似对话模板的跨请求模式 |
| 5 | 投机解码可观测性 | P2 | 🔲 设计完成 | 全链路指标：接受率、匹配长度、加速比 |

**设计要点**：
- **后缀解码仅在 Decode 阶段生效**（不影响 Prefill），与 PD 分离天然兼容
- 当前 vLLM V1 仅支持 N-gram Proposer（固定窗口、无状态搜索、只取首次匹配），后缀树在匹配长度和接受率上显著优于 N-gram

> 📄 详细设计文档：[`suffix-decoding-optimization.md`](suffix-decoding-optimization.md)

---

### 方向四：分布式架构 — PD 分离（6 个优化点）

> **核心思路**：将 Prefill（计算密集）和 Decode（访存密集）分离到不同实例，独立调优 TTFT 和 ITL

| # | 优化点 | 优先级 | 状态 | 核心价值 |
|---|--------|--------|------|---------|
| 1 | V1 引擎 PD 基础适配 | P0 | 🔲 设计完成 | V1 的 GPUModelRunner 支持 KV 收发 |
| 2 | 智能请求路由/代理 | P0 | 🔲 设计完成 | 负载感知 + 请求分类 + 故障转移 |
| 3 | 调度器 PD 感知 | P1 | 🔲 设计完成 | Prefill-only / Decode-only 专属调度策略 |
| 4 | KV Cache 传输优化 | P1 | 🔲 设计完成 | 增量传输 + FP8 压缩 + 流水线化 |
| 5 | Prefix Cache 与 PD 协同 | P2 | 🔲 设计完成 | 跨实例缓存共享，避免重复传输 |
| 6 | 多实例协调与可观测性 | P2 | 🔲 设计完成 | NP:MD 弹性部署 + 全链路指标 |

**关键背景**：
- vLLM 已有完整的 PD 分离三层抽象（Pipe → LookupBuffer → Connector），但**仅在 V0 引擎中实现**
- V1 引擎（`vllm/v1/`）中**零 PD 相关代码**，需要全新适配
- 现有路由是简单的 HTTP Proxy，无智能决策能力

> 📄 详细设计文档：[`pd-disaggregation-optimization.md`](pd-disaggregation-optimization.md)

---

### 端到端业务场景验证框架

> **核心思路**：设计一个 5 阶段递进加压的综合压测，暴露已有优化的盲区，驱动增量修复

**框架概述**：模拟企业级 AI 平台，一个 vLLM 实例同时服务 7 个租户（Gold/Silver/Bronze），涵盖短对话、代码补全、长文档 RAG 等业务类型。

```
时间线：
0s─────60s─────120s─────180s─────240s─────300s
│ Phase 1 │ Phase 2 │ Phase 3  │ Phase 4  │ Phase 5  │
│ 稳态预热 │ Prompt  │ Gold-A   │ 长文档   │ 全面     │
│ 建立基线 │ 版本切换 │ 流量暴增  │ 暴增     │ 过载     │
```

**暴露的 5 个盲区 → 对应的 5 个增量修复**：

| Phase | 暴露问题 | 增量修复 | 状态 |
|-------|---------|---------|------|
| 全程 | 代码补全取消后缓存未保留 | 修复 1：取消感知缓存保留 | 🔲 未实现 |
| Phase 2 | Prompt 切换时旧缓存占据 protected zone | 修复 2：缓存版本管理 | 🔲 未实现 |
| Phase 3 | MLFQ/QoS 无租户级隔离 | 修复 3：Prefill 预算隔离 | 🔲 未实现 |
| Phase 4 | 长文档 Prefill 挤占短对话 budget | 修复 4：租户级资源隔离 | 🔲 未实现 |
| Phase 5 | 无准入控制和过期丢弃 | 修复 5：过载管理 | 🔲 未实现 |

**文件**：
- `benchmarks/e2e_business_cases/workload.py` — 压测工作负载生成器 ✅ 已完成
- `benchmarks/e2e_business_cases/LANDING_PLAN.md` — 分阶段实施方案 ✅ 已完成

> 📄 详细设计文档：[`benchmarks/e2e_business_cases/README.md`](benchmarks/e2e_business_cases/README.md)
> 📄 落地方案：[`benchmarks/e2e_business_cases/LANDING_PLAN.md`](benchmarks/e2e_business_cases/LANDING_PLAN.md)

---

## 四大方向协同关系

```
                    ┌────────────────────────────────────────────┐
                    │       方向一：调度与资源管理                   │
                    │  QoS 分级 / MLFQ / Token 限速 / 准入控制    │
                    │  ★ 决定"谁先被调度"+"跑多快"                 │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │       方向二：KV Cache 管理                   │
                    │  Cache-Aware / Segmented LRU / 抢占保护      │
                    │  ★ 通过前缀复用减少 Prefill 计算量 → TTFT↓   │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │       方向三：投机解码（后缀解码）              │
                    │  SuffixTree / 增量 SAM / 自适应匹配           │
                    │  ★ Decode 阶段每步有效 token↑ → TPOT↓        │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │       方向四：分布式架构（PD 分离）             │
                    │  V1 适配 / 智能路由 / 传输优化 / Cache 协同   │
                    │  ★ P/D 独立部署，TTFT 和 ITL 互不干扰        │
                    └──────────────────┬─────────────────────────┘
                                       │
                    ┌──────────────────▼─────────────────────────┐
                    │      端到端业务场景验证                        │
                    │  5 阶段递进加压 + 5 项增量修复                 │
                    │  ★ 从框架优化到业务优化的完整闭环              │
                    └────────────────────────────────────────────┘
```

**具体协同**：

| 组合 | 协同效果 |
|------|---------|
| 调度 + KV Cache | Cache-Aware 调度优先处理缓存命中高的请求 → Prefill 计算量↓ → token_budget 利用率↑ |
| 调度 + 投机解码 | MLFQ 对投机解码成功的请求自动保持高优（每步产出多 token → 降级慢） |
| KV Cache + PD 分离 | Prefix Cache 命中的 KV blocks 不重复传输 → 传输量↓ 50-80% |
| 投机解码 + PD 分离 | 后缀解码只在 Decode 实例运行 → 不干扰 Prefill 实例的计算密度 |
| 全部 + 端到端验证 | 综合压测暴露各优化的盲区 → 驱动增量修复 → 形成完整闭环 |

---

## 实现进度总览

### 已实现（6 项）

| # | 优化点 | 方向 | 核心修改文件 |
|---|--------|------|-------------|
| 1 | QoS 分级调度 | 调度 | `vllm/v1/core/scheduler.py`, `vllm/v1/request.py` |
| 2 | Token 级速率控制 | 调度 | `vllm/v1/core/scheduler.py`, `vllm/v1/request.py` |
| 3 | MLFQ 多级反馈队列 | 调度 | `vllm/v1/core/scheduler.py`, `vllm/v1/request.py` |
| 4 | Cache-Aware Scheduling | KV Cache | `vllm/v1/core/kv_cache_manager.py` |
| 5 | Frequency-Aware Eviction (Segmented LRU) | KV Cache | `vllm/v1/core/kv_cache_utils.py` |
| 6 | Preemption Cache Shield | KV Cache | `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/scheduler.py` |

### 设计完成、待实现（18 项）

**调度方向**（5 项）：KV 水位线流控、准入控制、Deadline/EDF、WFQ 公平调度、KV Cache 分层存储

**KV Cache 方向**（2 项）：Proactive Cache Warming、Cache Efficiency Dashboard

**投机解码方向**（5 项）：SuffixTreeProposer、增量 SAM、自适应匹配、跨请求共享、可观测性

**PD 分离方向**（6 项）：V1 适配、智能路由、调度器感知、传输优化、Cache 协同、多实例协调

### 端到端验证框架

- [x] workload.py 工作负载生成器
- [x] LANDING_PLAN.md 分阶段实施方案
- [ ] 修复 1：取消感知缓存保留
- [ ] 修复 2：缓存版本管理
- [ ] 修复 3：Prefill 预算隔离
- [ ] 修复 4：租户级资源隔离
- [ ] 修复 5：过载管理

---

## 预期性能目标

| 指标 | 原生 vLLM V1 | 优化后（预期） | 主要优化来源 |
|------|-------------|---------------|-------------|
| 短请求 TTFT P99 | 基准 | ↓ 30-50% | QoS 分级 + Cache-Aware 调度 |
| 短请求 TPOT 抖动 | 基准 | ↓ 40%+ | Token 限速 + MLFQ |
| 缓存命中率 | 基准 | ↑ 20-40% | Segmented LRU + Cache-Aware + Preemption Shield |
| 抢占频率 | 基准 | ↓ 50%+ | 水位线流控 + Token 限速 + 分层存储 |
| 抢占恢复耗时 | Recompute（秒级） | CPU→GPU memcpy（毫秒级） | KV Cache 分层存储 |
| Decode 加速比 | 1× (无投机) | 1.5-3× | 后缀解码（SuffixTreeProposer） |
| ITL 尾延迟（PD 分离） | Prefill 干扰导致尖峰 | 完全稳定 | PD 分离 |
| 多租户公平性 | 无保障 | Jain's Index > 0.9 | WFQ + 租户隔离 |

---

## 项目亮点

1. **全栈覆盖**：同时优化调度、KV Cache、投机解码、分布式架构四大方向，形成完整的推理优化体系
2. **技术迁移性强**：将云网络/存储的调度、限速、时延优化、分层存储能力直接复用至推理引擎
3. **基于 V1 架构**：所有优化直接在 vLLM V1（最新主力架构）上开发，确保前瞻性
4. **渐进式实现**：24 个优化点按优先级逐步实现，每个独立可用、可测试、可量化
5. **端到端验证**：5 阶段递进加压的综合压测框架，从框架优化到业务优化的完整闭环
6. **填补社区空白**：V1 PD 分离、MLFQ、Token 限速、Cache-Aware 调度等均为社区首创

---

## 文档索引

| 文档 | 内容 | 优化点数 |
|------|------|---------|
| 📄 本文件 (README.md) | 全局总览 + 调度方向详细设计 | 8 (调度) |
| 📄 [`prefix-cache-scheduling-optimization.md`](prefix-cache-scheduling-optimization.md) | Prefix Cache 感知调度与复用优化 | 5 (KV Cache) |
| 📄 [`suffix-decoding-optimization.md`](suffix-decoding-optimization.md) | 后缀解码（投机解码优化） | 5 (投机解码) |
| 📄 [`pd-disaggregation-optimization.md`](pd-disaggregation-optimization.md) | Prefill-Decode 分离架构优化 | 6 (PD 分离) |
| 📄 [`benchmarks/e2e_business_cases/README.md`](benchmarks/e2e_business_cases/README.md) | 端到端业务场景综合压测设计 | 5 (增量修复) |
| 📄 [`benchmarks/e2e_business_cases/LANDING_PLAN.md`](benchmarks/e2e_business_cases/LANDING_PLAN.md) | 端到端压测分阶段实施方案 | — |

---

## 如何运行

### 环境准备

1. GPU 实例（A10/T4/L4 均可），Ubuntu 20.04+，CUDA 12.1+
2. 安装依赖：
   ```bash
   git clone <本项目地址>
   cd vllm-serving-optimization
   pip install -e .  # 开发模式安装，改代码实时生效
   ```

### 启动推理服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> \
    --max-model-len 4096 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill
```

### 运行基准测试

```bash
# 端到端业务场景综合压测
python benchmarks/e2e_business_cases/workload.py

# 待实现：对比测试脚本
python benchmarks/qos_benchmark.py --baseline --optimized --output results/
```

---

## 调度与资源管理详细设计

> 以下为方向一（调度与资源管理）8 个优化点的详细设计。其他方向的详细设计请参见对应的独立文档。

### 优化 1：推理请求 QoS 分级调度 `[P0]` `[已实现]`

> 对应能力迁移：**云网络虚拟网关 → 优先级队列 + 高优包优先转发**

#### 核心思路

类比网关**高优包优先转发、低时延保障**机制，对推理请求按长度/业务类型分级，保障短对话/搜索类高优请求的低时延体验。

#### vLLM V1 现状分析（v0.7.3）

- V1 调度器（`vllm/v1/core/scheduler.py`）仅支持 FCFS 调度，没有任何优先级机制
- `Request` 类没有 `priority` 字段，所有请求被平等对待
- 抢占策略为简单的 LIFO（`self.running.pop()`），不考虑请求重要性
- **缺少多维优先级计算**：没有综合 prompt 长度分级和等待时间防饿死因子

#### 实现细节

- 修改文件：
  - `vllm/v1/request.py` — Request 类新增多维优先级计算（`effective_priority` 属性）
  - `vllm/v1/core/scheduler.py` — 调度器集成动态优先级更新
- 核心逻辑：
  1. **多维优先级计算**：综合 API 传入的业务优先级（`priority` 字段）+ `num_prompt_tokens` 长度分级（短请求 <512 token 为高优）+ 等待时间衰减因子（防饿死）；
  2. **动态优先级更新**：每个调度步开始时，更新所有 waiting 请求的 `effective_priority`，使等待时间越长的请求优先级自动提升；
  3. **优先级感知调度**：`waiting` 队列每步按 `effective_priority` 重新排序，最高优先级请求排在队首；
  4. **优先级感知抢占**：抢占时选择 `self.running` 中 `effective_priority` 值最大（优先级最低）的请求。

---

### 优化 2：KV 缓存显存水位线与配额流控 `[P0]` `[未实现]`

> 对应能力迁移：**云存储 IO QoS → 水位线流控 + per-request 带宽配额**

#### 核心思路

借鉴云存储**流量整形、分级水位线、per-request 带宽配额**思想，对 KV 缓存实现显存资源管控。

#### 实现细节

- 修改文件：`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/scheduler.py`
- 核心逻辑：
  1. **多级水位线**：🟢 绿色（<70%）自由调度 → 🟡 黄色（70-85%）拒绝低优新请求 → 🔴 红色（≥85%）仅允许高优短请求
  2. **单请求 KV 块动态配额**：`max_blocks_per_req = base_quota × priority_factor × (1 - usage)`
  3. **主动回收**：水位超过黄色阈值时，主动释放 `ref_cnt == 0` 的 prefix cache blocks

---

### 优化 3：请求准入控制 `[P1]` `[未实现]`

> 对应能力迁移：**网络流量整形 Traffic Shaping → ECN + RED**

#### 实现细节

- 修改文件：`vllm/v1/engine/processor.py`, `vllm/v1/core/kv_cache_manager.py`
- 核心逻辑：
  1. **基于 KV Cache 水位的准入**：红色水位仅放行高优请求
  2. **基于队列深度的准入**：`len(waiting) > MAX_QUEUE_DEPTH` 时返回 503 + Retry-After
  3. **拒绝策略**：返回明确错误码和重试建议（类比 ECN 通知上游降速）

---

### 优化 4：Token 级速率控制 `[P1]` `[已实现]`

> 对应能力迁移：**云存储 IO 限速 → 令牌桶（Token Bucket）/ 漏桶（Leaky Bucket）**

#### 实现细节

- 修改文件：`vllm/v1/core/scheduler.py`, `vllm/v1/request.py`
- 核心逻辑：
  1. **Per-request 令牌桶**：`TokenRateLimiter(rate, burst, tokens)`
  2. **差异化限速**：高优不限速，低优根据系统负载动态调整 rate（8-64 tokens/step）
  3. **集成到调度器**：`schedule()` 中 `num_new_tokens = rate_limiter.consume(num_new_tokens)`

---

### 优化 5：Deadline-aware 调度 `[P2]` `[未实现]`

> 对应能力迁移：**网络 QoS Deadline 队列 → fq_codel / HFSC 调度**

#### 实现细节

- 修改文件：`vllm/v1/request.py`, `vllm/v1/core/scheduler.py`, `vllm/v1/engine/processor.py`
- 核心逻辑：
  1. **EDF（Earliest Deadline First）调度**：deadline 最早的请求最先被调度
  2. **松弛时间感知**：`slack_time = deadline - now - estimated_remaining_time`
  3. **超时处理**：超过 deadline 未开始 → 408 Timeout；已在生成中 → 标记低优

---

### 优化 6：加权公平队列 WFQ 调度 `[P2]` `[未实现]`

> 对应能力迁移：**网络 WFQ/DRR 流量调度 → 多租户加权公平**

#### 实现细节

- 修改文件：`vllm/v1/core/scheduler.py`, 新增 `vllm/v1/core/tenant_manager.py`
- 核心逻辑：
  1. **WFQ 虚拟时间戳调度**：选 `virtual_time` 最小的租户的队首请求
  2. **租户级 Token Budget 池化**：全局 budget 按权重划分给各租户
  3. **Budget 借用机制**：空闲租户的 budget 可被借用（work-conserving）

---

### 优化 7：MLFQ 多级反馈队列 `[P3]` `[已实现]`

> 对应能力迁移：**OS 进程调度 CFS → 多级反馈 + 自适应优先级**

#### 实现细节

- 修改文件：`vllm/v1/request.py`, `vllm/v1/core/scheduler.py`
- 核心逻辑：
  1. **4 级队列**：L0(interactive, 128) → L1(standard, 512) → L2(batch, 2048) → L3(background, ∞)
  2. **降级规则**：累计消耗 token 超过当前级配额 → 自动降级（`mlfq_account_tokens()`）
  3. **升级规则**：被抢占 → 升一级（`mlfq_promote()`），token 消耗不重置
  4. **调度顺序**：从 L0 扫描，有请求则优先调度，级别内 FCFS

---

### 优化 8：KV Cache 分层存储与智能迁移 `[P1]` `[未实现]`

> 对应能力迁移：**存储系统热温冷分层 + 块迁移 + 预取**

#### 实现细节

- 新增/修改文件：`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/kv_cache_utils.py`, 新增 `vllm/v1/core/block_migrator.py`
- 核心逻辑：
  1. **三级分层**：L1 GPU 显存（热）→ L2 CPU DRAM（温）→ L3 NVMe SSD（冷）
  2. **块级粒度迁移**：CUDA 异步 memcpy（不阻塞 GPU 计算），对比 V0 的全量 swap
  3. **智能降级**：高优请求只降到 CPU（保证快速恢复），低优可降到磁盘
  4. **智能预取**：预测下一步可能调度的 top-K 请求，异步预取其 KV Cache
  5. **抢占改造**：抢占不再 Recompute，而是降级到 CPU 保存

```
改造后（三级分层）：

  ┌───────────────────────────┐
  │   L1: GPU 显存（热层）      │  ← 活跃请求的 KV Cache
  ├───────────────────────────┤
  │   L2: CPU DRAM（温层）      │  ← 被抢占请求的 KV Cache（保留，不丢弃）
  ├───────────────────────────┤
  │   L3: NVMe SSD（冷层）      │  ← 长上下文历史 KV Cache
  └───────────────────────────┘
```

---

## 优化点依赖关系

```
方向一（调度）:
  优化 1 (QoS) ──┬──→ 优化 3 (准入控制) ──→ 优化 5 (Deadline/EDF)
  优化 2 (水位线) ─┤         │
  优化 8 (分层) ───┘         ├──→ 优化 4 (Token 限速)
                    优化 6 (WFQ) ──→ 优化 7 (MLFQ)

方向四（PD 分离）:
  优化 1 (V1 适配) ──┬──→ 优化 2 (路由) ──→ 优化 6 (多实例)
                     ├──→ 优化 3 (调度感知)
                     └──→ 优化 4 (传输) ──→ 优化 5 (Cache 协同)

跨方向:
  调度·QoS + KV Cache·Cache-Aware → 缓存命中优先调度
  KV Cache·Segmented LRU + PD·传输优化 → 缓存命中部分不传输
  投机解码 + PD·Decode-only → 投机解码不干扰 Prefill
```
