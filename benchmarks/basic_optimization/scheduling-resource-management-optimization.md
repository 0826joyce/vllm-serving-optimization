# 基于 vLLM V1 的调度与资源管理优化

> 将云网络/存储的调度、限速、时延优化能力迁移至 LLM 推理引擎调度器，从"谁先被调度"+"跑多快"两个维度全面优化推理服务的时延、吞吐和公平性

## 一、项目背景与动机

### 1.1 问题引入

vLLM V1 的调度器（`vllm/v1/core/scheduler.py`）默认采用 **FCFS（先来先服务）** 调度策略，对推理请求一律平等对待。在云网络/存储中已经被反复验证有效的调度理念（优先级队列、流量整形、限速流控、公平队列）在推理引擎调度器上几乎缺失：

> **调度器仅支持 FCFS，无优先级机制；抢占策略为简单 LIFO（Recompute 代价极高）；无请求准入控制、无速率限制、无租户公平性保障。**

### 1.2 vLLM V1 调度机制梳理（基于源码分析）

#### 1.2.1 核心流程

每个请求从到达系统到完成生成，经历以下调度决策点：

```
请求到达
    │
    ▼
调度入口（谁先被选中）           运行时（选中后跑多快）
    │                              │
优化 6（WFQ）→ 优化 1（QoS）→ 优化 7（MLFQ）
    │  租户分流      优先级排序     自适应排序    │
    ▼                                       ▼
请求被选中进入 running → 优化 4（Token 限速）→ 完成
                             控制每步生成多少 token
```

```
┌─────── 调度入口（谁先被选中）───────┐     ┌── 运行时（选中后跑多快）──┐
│                                     │     │                          │
到达 → 优化6 → 优化1 → 优化7 → 被选中调度 → │ →  优化4（Token 限速）      │ → 完成
│  租户分流   优先级排序  自适应排序     │     │  控制每步生成多少 token   │
└──────────────────────────────────────┘     └──────────────────────────┘
      "决定调度顺序"                              "控制资源消耗速率"
```

#### 1.2.2 现状问题

- **无优先级机制**：`Request` 类没有 `priority` 字段，所有请求被平等对待
- **抢占策略粗暴**：`self.running.pop()` 简单 LIFO，不考虑请求重要性，Recompute 代价极高
- **无准入控制**：队列无长度限制，过载时全部请求一起拥塞
- **无速率控制**：请求一旦进入 running 就以系统最大速度消耗 `token_budget`，低优请求拖慢高优请求
- **无租户隔离**：多租户场景下没有公平性保障，一个租户可占满全部资源

### 1.3 需要优化的问题

| 现状问题 | 具体表现 | 影响 |
|---------|---------|------|
| **无优先级调度** | FCFS 先来先服务，短请求被长请求堵住 | 短对话/搜索类高优请求时延恶化 |
| **抢占策略粗暴** | LIFO 弹出，不考虑请求重要性 | 高优请求可能被抢占，恢复需 Recompute |
| **无准入控制** | 队列无长度限制 | 过载时全体拥塞，服务质量雪崩 |
| **无速率限制** | 请求全速消耗 token_budget | 低优请求拖慢高优请求的 TPOT |
| **无租户公平** | 一个租户可占满全部资源 | 多租户场景公平性无保障 |

### 1.4 优化主题定位

本项目聚焦于**调度策略与资源管理的系统化改造**，通过 8 个递进的优化点（其中 6 个已实现），形成一套完整的调度与资源管控方案：

```
原始 vLLM V1:
┌─────────────────────────────────────────┐
│  FCFS 调度 ──→ 全速生成 ──→ 无准入 ──→ 无租户隔离 │
└─────────────────────────────────────────┘

优化后：
┌─────────────────────────────────────────────────────────┐
│  WFQ 租户公平 → QoS 分级 → MLFQ 自适应 → Token 限速 → 准入控制 │
│  ──→ 水位线流控 ──→ Deadline/EDF ──→ KV Cache 分层迁移        │
└─────────────────────────────────────────────────────────┘
```

---

## 二、优化点总览

| # | 优化点 | 优先级 | 状态 | Infra 能力对标 |
|---|--------|--------|------|---------------|
| 1 | QoS 分级调度 | P0 | ✅ 已实现 | 网关优先级队列 + 高优包优先转发 |
| 2 | KV 缓存显存水位线与配额流控 | P0 | 🔲 未实现 | 存储水位线 + IO 配额 |
| 3 | 请求准入控制 | P1 | ✅ 已实现 | ECN 显式拥塞通知 + RED 随机早期丢弃 |
| 4 | Token 级速率控制 | P1 | ✅ 已实现 | 令牌桶 / 漏桶限速 |
| 5 | Deadline-aware 调度 | P2 | ✅ 已实现 | EDF + fq_codel / HFSC |
| 6 | 加权公平队列 WFQ 调度 | P2 | ✅ 已实现 | WFQ / DRR + per-tenant 配额 |
| 7 | MLFQ 多级反馈队列 | P3 | ✅ 已实现 | OS MLFQ / CFS |
| 8 | KV Cache 分层存储与智能迁移 | P1 | 🔲 未实现 | 存储热温冷分层 + 预取 |

---

## 三、已实现优化详解

### 优化 1：推理请求 QoS 分级调度 `[P0]` `[已实现]`

> 对应能力迁移：**云网络虚拟网关 → 优先级队列 + 高优包优先转发**

#### 3.1 核心思路

类比网关**高优包优先转发、低时延保障**机制，对推理请求按长度/业务类型分级，保障短对话/搜索类高优请求的低时延体验。

#### 3.2 vLLM V1 现状分析（v0.7.3）

- V1 调度器仅支持 FCFS 调度，没有任何优先级机制
- `Request` 类没有 `priority` 字段，所有请求被平等对待
- 抢占策略为简单的 LIFO（`self.running.pop()`），不考虑请求重要性
- **缺少多维优先级计算**：没有综合 prompt 长度分级和等待时间防饿死因子

#### 3.3 实现细节

- **修改文件**：
  - `vllm/v1/request.py` — Request 类新增多维优先级计算（`effective_priority` 属性）
  - `vllm/v1/core/scheduler.py` — 调度器集成动态优先级更新
- **核心逻辑**：
  1. **多维优先级计算**：综合 API 传入的业务优先级（`priority` 字段）+ `num_prompt_tokens` 长度分级（短请求 <512 token 为高优）+ 等待时间衰减因子（防饿死）
  2. **动态优先级更新**：每个调度步开始时，更新所有 waiting 请求的 `effective_priority`，使等待时间越长的请求优先级自动提升
  3. **优先级感知调度**：`waiting` 队列每步按 `effective_priority` 重新排序，最高优先级请求排在队首
  4. **优先级感知抢占**：抢占时选择 `self.running` 中 `effective_priority` 值最大（优先级最低）的请求

#### 3.4 价值

短请求 TTFT 显著降低，同时通过等待时间衰减因子避免低优请求饿死。

---

### 优化 4：Token 级速率控制 `[P1]` `[已实现]`

> 对应能力迁移：**云存储 IO 限速 → 令牌桶（Token Bucket）/ 漏桶（Leaky Bucket）**

#### 4.1 核心思路

为什么有了 QoS/MLFQ 还需要 Token 限速？

QoS/MLFQ 解决了"谁先上路"，但请求一旦被选中进入 running 状态，它就以系统最大速度消耗 `token_budget`。在高负载下，低优请求会消耗大量 budget，拖慢高优请求的 TPOT。Token 限速在**运行时**控制生成速率，为高优请求让出预算。

```
场景：10 个请求在 running 中，全局 token_budget = 2048

没有 Token 限速：
  高优请求 × 3: 各消耗 ~200 tokens/step → 合计 600
  低优请求 × 7: 各消耗 ~200 tokens/step → 合计 1400
  → 低优请求消耗了 68% 的 budget！高优请求的 TPOT 被拖慢

有 Token 限速：
  高优请求 × 3: 各消耗 ~200 tokens/step → 不限速
  低优请求 × 7: 各消耗 ~20 tokens/step  → 限速为高优让路
  → 低优请求只消耗 140 tokens，高优请求不受影响
```

#### 4.2 实现细节

- **修改文件**：`vllm/v1/core/scheduler.py`, `vllm/v1/request.py`
- **核心逻辑**：
  1. **Per-request 令牌桶**：`TokenRateLimiter(rate, burst, tokens)`
  2. **差异化限速**：高优不限速，低优根据系统负载动态调整 rate（8-64 tokens/step）
  3. **集成到调度器**：`schedule()` 中 `num_new_tokens = rate_limiter.consume(num_new_tokens)`

#### 4.3 关键区别

| 维度 | 优化 1/6/7（调度入口） | 优化 4（运行时控制） |
|------|---------------------|-------------------|
| **作用时机** | 请求从 waiting → running 时 | 请求已在 running 中 |
| **控制对象** | 调度**顺序** | 生成**速率** |
| **类比** | 高速公路收费站：决定谁先上路 | 限速牌：已上路的车能开多快 |
| **解决的问题** | 谁先拿到 GPU 时间 | 拿到 GPU 时间后用多少 |

---

### 优化 7：MLFQ 多级反馈队列 `[P3]` `[已实现]`

> 对应能力迁移：**OS 进程调度 CFS → 多级反馈 + 自适应优先级**

#### 7.1 核心思路

借鉴操作系统多级反馈队列（MLFQ），无需人工标注优先级，请求按实际 token 消耗自动降级，实现"短请求自动优先、长请求自动让位"的自适应调度。

#### 7.2 实现细节

- **修改文件**：`vllm/v1/request.py`, `vllm/v1/core/scheduler.py`
- **核心逻辑**：
  1. **4 级队列**：L0(interactive, 128) → L1(standard, 512) → L2(batch, 2048) → L3(background, ∞)
  2. **降级规则**：累计消耗 token 超过当前级配额 → 自动降级（`mlfq_account_tokens()`）
  3. **升级规则**：被抢占 → 升一级（`mlfq_promote()`），token 消耗不重置
  4. **调度顺序**：从 L0 扫描，有请求则优先调度，级别内 FCFS

#### 7.3 价值

无需人工标注优先级，通过请求自身的 token 消耗行为自动分级，短交互请求天然保持高优。

---

## 四、待实现优化设计

### 优化 2：KV 缓存显存水位线与配额流控 `[P0]` `[未实现]`

> 对应能力迁移：**云存储 IO QoS → 水位线流控 + per-request 带宽配额**

#### 核心思路

借鉴云存储**流量整形、分级水位线、per-request 带宽配额**思想，对 KV 缓存实现显存资源管控。

#### 实现细节

- **修改文件**：`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/scheduler.py`
- **核心逻辑**：
  1. **多级水位线**：🟢 绿色（<70%）自由调度 → 🟡 黄色（70-85%）拒绝低优新请求 → 🔴 红色（≥85%）仅允许高优短请求
  2. **单请求 KV 块动态配额**：`max_blocks_per_req = base_quota × priority_factor × (1 - usage)`
  3. **主动回收**：水位超过黄色阈值时，主动释放 `ref_cnt == 0` 的 prefix cache blocks

---

### 优化 3：请求准入控制 `[P1]` `[已实现]`

> 对应能力迁移：**网络流量整形 Traffic Shaping → ECN + RED**

#### 实现细节

- **修改文件**：`vllm/v1/core/sched/scheduler.py`, `vllm/v1/engine/core.py`, `vllm/v1/request.py`
- **核心逻辑**：
  1. **基于队列深度的准入**：`len(waiting) >= max_queue_depth`（默认 40）时拒绝低优新请求
  2. **基于 SLA 违约率的准入**：滑动窗口（maxlen=25）内违约率 ≥ `overload_violation_threshold`
     （默认 0.25）时拒绝低优请求
  3. **优先级保护**：仅 `priority < 0`（gold 租户）永不拒绝；silver/bronze 受准入控制约束
  4. **拒绝路径完整化**：被拒请求立即 `_free_request` + EngineCore 发 `ERROR` 输出
     （`FinishReason.ERROR`），客户端收到 503 而非挂起超时
  5. **SLA 违约窗口填充**：请求完成时 `append(request.is_sla_violated())`，驱动违约率门控

#### 关键修复（落地过程中发现并解决的 4 个 bug）

| Bug | 影响 | 修复 |
|-----|------|------|
| `_sla_violation_window` 从未被填充 | SLA 违约率门控永远不触发 | 请求完成时 append 违约结果 |
| 被拒请求未通知客户端 | 挂起直到超时 + 内存泄漏 | `_free_request` + `_send_error_outputs_to_client` |
| `FINISHED_REJECTED` 无 finish_reason | `get_finished_reason()` 返回 None | 映射到 `FinishReason.ERROR` |
| 短 prompt 都判为高优 | 准入控制形同虚设 | 改为仅 `priority < 0` 永不拒绝 |

#### 验证结论（A/B 实测）

| Phase 4 违约率 | A 基线(fcfs) | B 优化(priority) | 效果 |
|---------------|-------------|-----------------|------|
| gold（最高优） | 83.2% | 96.6% | 无法保护（自身过载 + 不可拒绝） |
| silver | 76.3% | **20.0%** | 违约率 -56.3pp ✅ |
| bronze（最低优） | 69.0% | **16.0%** | 违约率 -53.0pp ✅ |

> **核心认知**：准入控制能「决定谁被拒绝」，不能「凭空增加算力」。gold 违约源于
> 自身高 QPS + 最高优先级不可拒绝的物理约束，调度无法解决。准入控制的本质是把过载
> 代价从高优先级租户转移到低优先级租户（通过拒绝实现保护，而非更快完成）。

#### 配套机制：长文档 Prefill 让位 `[已实现]`

在准入控制之外，额外实现了「长文档 continuation prefill 让位」
（`enable_long_prefill_yield`，`VLLM_DISABLE_LONG_PREFILL_YIELD=1` 关闭）：
识别 RUNNING 中仍处理大量 prompt（>1024 token）的长文档 continuation，每个 step 限流
此类请求（默认最多 1 个），让 token_budget 留给 waiting 的短请求，使短请求能在长文档
prefill 的 chunk 之间插队。

---

### 优化 5：Deadline-aware 调度 `[P2]` `[已实现]`

> 对应能力迁移：**网络 QoS Deadline 队列 → fq_codel / HFSC 调度**

#### 实现细节

- **修改文件**：`vllm/v1/request.py`, `vllm/v1/core/sched/scheduler.py`
- **核心逻辑**：
  1. **EDF（Earliest Deadline First）调度**：deadline 最早的请求最先被调度
     （`_deadline_aware_sort_waiting()`，`enable_deadline_aware_scheduling` 开关，
     `VLLM_DISABLE_DEADLINE_AWARE=1` 关闭）
  2. **松弛时间感知**：`slack_time = deadline - now`，`sla_urgency` 计算紧急度
  3. **SLA 感知抢占**：`_select_preemption_victim()` 优先抢占已违约（`is_sla_violated()`）
     的请求，其次按 `effective_priority` 抢占最低优先级

---

### 优化 6：加权公平队列 WFQ 调度 `[P2]` `[已实现]`

> 对应能力迁移：**网络 WFQ/DRR 流量调度 → 多租户加权公平**

#### 实现细节

- **修改文件**：`vllm/v1/core/sched/scheduler.py`, 新增 `vllm/v1/core/sched/tenant_manager.py`
- **核心逻辑**：
  1. **租户级并发上限**：`TenantManager.can_schedule(tenant_id)` 检查租户是否达到
     `max_running` 并发限制（`enable_tenant_isolation` 开关，`VLLM_DISABLE_TENANT_ISO=1` 关闭）
  2. **WFQ 加权调度**：按租户权重分配调度机会
  3. **租户级 Token Budget 池化**：全局 budget 按权重划分给各租户

---

### 优化 8：KV Cache 分层存储与智能迁移 `[P1]` `[未实现]`

> 对应能力迁移：**存储系统热温冷分层 + 块迁移 + 预取**

#### 实现细节

- **新增/修改文件**：`vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/kv_cache_utils.py`, 新增 `vllm/v1/core/block_migrator.py`
- **核心逻辑**：
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

## 五、调度四层架构

8 个优化点共同构成一个完整的调度流水线，每个优化负责一个决策环节：

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

---

## 六、优化点依赖关系

```
优化 1 (QoS) ──┬──→ 优化 3 (准入控制) ──→ 优化 5 (Deadline/EDF)
优化 2 (水位线) ─┤         │
优化 8 (分层) ───┘         ├──→ 优化 4 (Token 限速)
                 优化 6 (WFQ) ──→ 优化 7 (MLFQ)
```

---

## 七、核心修改文件

| 文件 | 涉及优化点 | 说明 |
|------|-----------|------|
| `vllm/v1/core/scheduler.py` | 1, 2, 4, 5, 6, 7, 8 | 调度器核心逻辑 |
| `vllm/v1/request.py` | 1, 4, 5, 7 | Request 类扩展（优先级、MLFQ 级别、速率限制器、deadline） |
| `vllm/v1/core/kv_cache_manager.py` | 2, 8 | KV Cache 管理器 |
| `vllm/v1/core/kv_cache_utils.py` | 8 | 缓存工具函数 |
| `vllm/v1/engine/processor.py` | 3, 5 | 请求处理与准入 |
| 新增 `vllm/v1/core/tenant_manager.py` | 6 | 租户管理与配额 |
| 新增 `vllm/v1/core/block_migrator.py` | 8 | 块级迁移器 |

---

## 八、预期性能目标

| 指标 | 原生 vLLM V1 | 优化后（预期） | 主要优化来源 |
|------|-------------|---------------|-------------|
| 短请求 TTFT P99 | 基准 | ↓ 30-50% | QoS 分级 + Cache-Aware 调度 |
| 短请求 TPOT 抖动 | 基准 | ↓ 40%+ | Token 限速 + MLFQ |
| 抢占频率 | 基准 | ↓ 50%+ | 水位线流控 + Token 限速 + 分层存储 |
| 抢占恢复耗时 | Recompute（秒级） | CPU→GPU memcpy（毫秒级） | KV Cache 分层存储 |
| 多租户公平性 | 无保障 | Jain's Index > 0.9 | WFQ + 租户隔离 |

---

## 九、与其他方向的协同

| 组合 | 协同效果 |
|------|---------|
| 调度 + KV Cache | Cache-Aware 调度优先处理缓存命中高的请求 → Prefill 计算量↓ → token_budget 利用率↑ |
| 调度 + 投机解码 | MLFQ 对投机解码成功的请求自动保持高优（每步产出多 token → 降级慢） |
| 调度 + PD 分离 | Deadline-aware 调度可区分 Prefill-only / Decode-only 请求 |

---

## 十、实现进度

- ✅ **已实现（6 项）**：
  1. QoS 分级调度（优化 1）
  2. 请求准入控制（优化 3，含 SLA 违约率门控 + 拒绝路径完整化）
  3. Token 级速率控制（优化 4）
  4. Deadline-aware 调度（优化 5）
  5. 加权公平队列 WFQ + 租户隔离（优化 6）
  6. MLFQ 多级反馈队列（优化 7）
- ✅ **配套机制**：长文档 continuation prefill 让位（`enable_long_prefill_yield`）
- 🔲 **待实现（2 项）**：KV 水位线流控（优化 2）、KV Cache 分层存储（优化 8）

> **验证状态**：优化 1/3/4/5/6/7 已通过 A/B 压测验证（详见
> `benchmarks/e2e_cases/SCHEDULING_BENCHMARK_GUIDE.md` §7）。核心结论：优先级调度 +
> 准入控制能有效保护 silver/bronze 租户（违约率 -53~-56pp），但无法保护 gold
> （最高优先级自身过载，物理无解）。
