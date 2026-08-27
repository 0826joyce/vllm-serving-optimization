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

### 优化 9：KV 到达事件驱动调度（PD 分离场景） `[P2]` `[方案设计]`

> 对应能力迁移：**事件驱动 I/O（epoll/事件循环）→ KV 就绪事件触发调度**
> 呼应 [`pd-disaggregation-optimization.md`](./pd-disaggregation-optimization.md) 的 PD 分离场景。

#### 问题分析

当调度器与 PD 分离结合时，Decode 实例的请求需要**等待 Prefill 实例把 KV cache 传过来**才能开始 decode。当前调度是"轮询式"——每步都去检查 KV 到了没，浪费调度开销；且请求在等 KV 时若仍留在 waiting 参与排序，会干扰正常调度。

#### 设计方案

引入一个**等待远端 KV 的独立状态**（`WAITING_FOR_REMOTE_KVS`），配合 KV 到达事件驱动：

```python
class KVReceiveMonitor:
    """监控 PD 分离下 KV cache 的到达，事件驱动地唤醒等待的请求。"""

    def __init__(self, timeout_s: float = 30.0):
        self._waiting_for_kv: dict[str, float] = {}   # req_id → 开始等待的时间

    def on_request_waiting_kv(self, request):
        """请求进入"等待远端 KV"状态，暂时移出正常调度队列。"""
        request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        self._waiting_for_kv[request.request_id] = time.monotonic()

    def poll_ready(self, connector) -> list[str]:
        """事件驱动：返回 KV 已到达、可以恢复调度的请求。"""
        ready = []
        for req_id, start in list(self._waiting_for_kv.items()):
            if connector.is_kv_ready(req_id):        # KV 到达事件
                ready.append(req_id)
                del self._waiting_for_kv[req_id]
            elif time.monotonic() - start > self.timeout_s:
                # 超时回退：KV 迟迟不到 → 降级为本地重算（安全网）
                ready.append(req_id)                  # 唤醒但标记为需本地 recompute
                del self._waiting_for_kv[req_id]
        return ready
```

**关键设计**：
1. **独立状态隔离**：等 KV 的请求移出正常 waiting 队列，不干扰 QoS/MLFQ 排序。
2. **事件驱动而非轮询**：KV 到达才唤醒，减少无效调度检查。
3. **超时回退安全网**：KV 传输失败/超时 → 降级为本地重算，保证请求不会永久卡死（这是工程健壮性关键）。

#### 修改文件
- `vllm/v1/core/sched/scheduler.py` — 集成 `WAITING_FOR_REMOTE_KVS` 状态处理
- 新增 `vllm/v1/core/sched/kv_receive_monitor.py`

#### 预期效果
- PD 分离下调度开销降低（不再轮询等 KV）
- 等 KV 的请求不占算力、不干扰正常调度
- KV 传输故障时有降级路径，不会卡死请求

---

### 优化 10：多模型实例的资源池化调度 `[P3]` `[方案设计]`

> 对应能力迁移：**云资源池化 / 容器编排（K8s bin-packing）→ 单实例多模型的 GPU 资源复用**

#### 问题分析

单实例单模型时，若该模型负载低，GPU 算力和显存大量闲置。若能在一个实例里**按需调度多个模型**、空闲模型让出资源，可大幅提升利用率、降低成本。

#### 设计方案

在调度器之上加一层**模型级资源仲裁**：

```python
class MultiModelResourceArbiter:
    """多模型资源仲裁：在模型间分配 GPU 显存/算力配额。"""

    def __init__(self, models: dict, total_memory: int):
        self.models = models                      # model_id → 模型实例
        self.memory_quota: dict[str, int] = {}    # 每个模型的显存配额

    def rebalance(self, load_stats: dict[str, float]):
        """按各模型近期负载动态调整显存配额（用得多的多分）。"""
        total_load = sum(load_stats.values()) or 1.0
        for model_id, load in load_stats.items():
            self.memory_quota[model_id] = int(self.total_memory * load / total_load)

    def should_sleep(self, model_id: str) -> bool:
        """长时间无请求的模型 → sleep 释放显存（对应 vLLM sleep/wake_up）。"""
        return self._idle_time(model_id) > self.sleep_threshold_s
```

> **据实说明**：这一项在 vLLM 里对应官方的 **sleep/wake_up API** + 显存 offload 机制。真正的
> "虚拟显存 + 多模型统一接管"（如 xLLM PR #861 的 Global XTensor）需要底层显存虚拟化（CUDA VMM），
> **属于比本项目调度层更深的基础设施改造**，本优化仅在调度层做"配额仲裁 + sleep/wake 触发"，
> 不涉及底层显存虚拟化。

#### 修改文件
- 新增 `vllm/v1/core/sched/multi_model_arbiter.py`
- 依赖 vLLM 官方 sleep/wake_up 能力

#### 预期效果
- 多模型共享单卡，闲时让出资源，提升 GPU 利用率、降成本
- ⚠️ 局限：受限于单卡显存与官方 offload 能力，深度虚拟显存需底层改造

---

## 五、调度四层架构

6 个已实现优化点共同构成一个完整的调度流水线，每个优化负责一个决策环节：

```
请求到达
   │
   ▼  优化6 WFQ ── 按 tenant_id 分流，检查租户并发上限（保证租户间公平）
   │
   ▼  优化3 准入控制 ── 过载时？队列满(40)/违约率高(0.25)→ 拒绝低优（gold 放行）
   │
   ▼  优化1 QoS ── 算 effective_priority，waiting 队列按优先级排序
   │
   ▼  优化7 MLFQ ── 同优先级内按 token 消耗自动分级（短请求优先）
   │
   ▼  优化5 Deadline ── 紧急请求（快到期）再插队；抢占时先抢已违约的
   │
   ▼  请求被选中，进入 running
   │
   ▼  优化4 Token 限速 ── 低优请求限速(8~64 tok/step)，把预算让给高优
   │  （配套：长文档 prefill 让位，每步最多 1 个长 continuation）
   │
   ▼  输出 tokens
```

> **分层理解**：
> - **入口关卡**（决定"能不能进/进哪"）：优化 6 租户分流 → 优化 3 过载准入。
> - **waiting 排序**（决定"谁先进 running"）：优化 1 QoS（主键）→ 优化 7 MLFQ（次键）→ 优化 5 Deadline（紧急插队）。
> - **running 运行时**（已在 GPU 上跑）：优化 4 Token 限速（控速）+ 抢占（缺资源时先踢已违约/最低优的）+ 长文档 prefill 让位。

### 各优化作用的队列/阶段（重要）

vLLM 调度有两个核心队列：**waiting**（等待被调度的请求）和 **running**（已在 GPU 上并行生成 token 的请求）。每个优化明确作用在哪个阶段：

| 优化 | 作用队列/阶段 | 具体做什么 | 备注 |
|------|-------------|-----------|------|
| **优化 6 WFQ/租户隔离** | **入口：waiting → running 准入** | 检查该租户在 running 的并发数是否达上限（默认 64），达到则该请求这步不进 running（留 waiting） | 控制"每个租户能有多少请求同时在 running" |
| **优化 3 准入控制** | **入口：进 waiting 之前** | 过载时（队列深度≥40 或违约率>0.25）直接拒绝低优新请求（503），gold 放行 | 请求根本进不了系统，不占队列 |
| **优化 1 QoS** | **waiting 排序**（+ running 抢占） | 算 `effective_priority`，作为 waiting 出队的**第一排序键**；running 里作为抢占选择依据 | waiting 主键；running 里不排序、只定"抢占谁" |
| **优化 7 MLFQ** | **waiting 排序** | 按累计 token 消耗自动降级，作为 waiting 出队的**第二排序键**（QoS 相等时比 level） | 计数记在请求对象上，跨 waiting/running 往返不丢失 |
| **优化 5 Deadline** | **waiting 排序**（+ running 抢占） | 紧急（快到期）请求在 waiting 里插队；抢占时先踢 running 里已违约的 | waiting 排序 + running 抢占对象选择 |
| **优化 4 Token 限速** | **running 运行时** | 对 running 里每个请求，按档位+负载限制它每步能生成的 token 数 | 唯一真正作用在 running 内部的"控速"机制 |
| **配套 长文档 prefill 让位** | **running 运行时** | 限制 running 里长文档 continuation prefill 每步最多 1 个，让 budget 给 waiting 的短请求插队 | 作用在 running 的调度分配 |

> **一句话记忆**：
> - **"排队顺序"类**（QoS / MLFQ / Deadline 的排序部分）→ 全作用在 **waiting**，决定"谁先进 running"。
> - **"准入名额"类**（WFQ 租户隔离 / 准入控制）→ 作用在 **入口**，决定"能不能进、进不进得来"。
> - **"运行时控速"类**（Token 限速 / 长文档让位 / 抢占）→ 作用在 **running**，决定"已在跑的请求各跑多快、缺资源时踢谁"。
> - **running 内部不做"重新排队"**——里面的请求是并行 forward 的，没有先后顺序，只有"限速"和"抢占淘汰"。

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

---

## 十一、优化点详解速查（含真实参数与代码落点）

> 本节把 6 个已实现优化点逐个讲清「原理 → 解决什么 → 真实参数 → 代码落点」，
> 参数值均来自源码（`vllm/v1/request.py`、`vllm/v1/core/sched/scheduler.py`、
> `vllm/v1/core/sched/tenant_manager.py`），可直接对照。

### 11.1 参数总表

| 优化 | 开关 | 核心参数（真实值） | 是否在 §7 调过 |
|------|------|-------------------|--------------|
| 1 QoS 分级 | `--scheduling-policy priority`（`fcfs` 关） | 短 prompt 线 `SHORT_PROMPT_THRESHOLD=512`、中 `2048`；boost 短 `-2`/中 `-1`/长 `+1`；防饿死 `STARVATION_DECAY_INTERVAL=5s` | ❌ 固定 |
| 3 准入控制 | `VLLM_DISABLE_OVERLOAD_MGMT=1` 关 | **`max_queue_depth=40`、`overload_violation_threshold=0.25`、`_sla_violation_window` maxlen`=25`**；高优判定 `priority < 0` | ✅ **反复调**（§7.4/§7.6） |
| 4 Token 限速 | `VLLM_DISABLE_RATE_LIMIT=1` 关 | 低优 rate 动态 `8~64 tokens/step`；per-request 令牌桶 `TokenRateLimiter` | ❌ 固定 |
| 5 Deadline | `VLLM_DISABLE_DEADLINE_AWARE=1` 关（依附过载管理） | `deadline = arrival_time + sla_ttft_ms`；`slack_time`/`sla_urgency` | ❌ 固定 |
| 6 WFQ/租户隔离 | `VLLM_DISABLE_TENANT_ISO=1` 关 | **租户并发上限 `default_max_running=64`**、`default_weight=1.0`；`effective_weight = weight / max(1, running)` | ❌ 固定 |
| 7 MLFQ | `VLLM_DISABLE_MLFQ=1` 关 | 4 级配额 L0`=128` / L1`=512` / L2`=2048` / L3`=∞` | ❌ 固定 |
| 配套 长文档让位 | `VLLM_DISABLE_LONG_PREFILL_YIELD=1` 关 | 长文档线 `>1024 token`、每步 `max_long_continuation_per_step=1` | ❌ 固定（改的是测试数据） |

### 11.2 WFQ / 租户隔离怎么保证公平（优化 6）

**代码**：`tenant_manager.py` 的 `TenantManager`

公平靠**两个机制**，不是靠"平均分配请求数"：

**机制 1：租户并发上限（硬隔离）**

```python
default_max_running: int = 64   # 每个租户默认最多 64 个请求同时在 running
def can_schedule(self, tenant_id):
    return self.tenant_running.get(tenant_id, 0) < max_allowed
```

- 每个租户在 `running` 里的请求数有上限（默认 **64**，可 per-tenant 通过 `register_tenant` 覆盖）。
- 达到上限后，该租户的新请求这一步**被跳过**（留在 waiting 等下一步），**不会被拒绝**。
- 作用：防止单个租户占满全部 running 槽位，饿死其他租户。

**机制 2：WFQ 动态权重（软公平）**

```python
def get_scheduling_weight(self, tenant_id):
    return weight / max(1, running_count)   # 用得越多，权重越低
```

- 一个租户当前占用的 running 槽越多，它的**有效调度权重越低**，让"用得少"的租户获得优先。
- 这是"加权公平"——不是绝对平均，而是**动态倾斜给欠服务的租户**。

> **回答你的疑问：每个用户允许的请求数是多少？是固定的吗？**
> - **默认是固定的 `max_running=64`**（`default_max_running`）——这是每个租户在 running 里的**并发上限**（不是"总共能发多少请求"，是"同时在跑几个"）。
> - 但它**可 per-tenant 配置**：`register_tenant(tenant_id, max_running=X, weight=Y)` 能给不同租户设不同上限和权重（比如 gold 租户给 128、bronze 给 16）。不配置就用默认 64。
> - 注意这个"64"和准入控制的 `max_queue_depth=40` **无关**：前者管"每个租户在 running 的并发数"，后者管"整个 waiting 队列的总深度"。

### 11.3 QoS 与 MLFQ 的关系（优化 1 vs 优化 7）—— 排序键的主次

这是最容易混淆的点。**关键事实：在 `priority` 策略下，`waiting` 是一个统一的优先级堆（`PriorityRequestQueue`），出队顺序由 `Request.__lt__` 决定，而 `__lt__` 里 QoS 和 MLFQ 是「主键 + 次键」的关系**：

```python
# vllm/v1/request.py 的 Request.__lt__（决定谁排队首）
def __lt__(self, other):
    # ① 第一关键字：QoS 的 effective_priority（值小 = 优先）
    if self.effective_priority != other.effective_priority:
        return self.effective_priority < other.effective_priority
    # ② 第二关键字：MLFQ level（同优先级时，level 小 = 优先）
    if self.mlfq_level != other.mlfq_level:
        return self.mlfq_level < other.mlfq_level
    # ③ 第三关键字：到达时间（同 level 时 FCFS）
    if self.arrival_time != other.arrival_time:
        return self.arrival_time < other.arrival_time
    return self.request_id < other.request_id
```

**所以关系是：QoS 定「大组」，MLFQ 在「大组内」再排序，同组同级才看到达时间。**

- **QoS（`effective_priority`）是第一决定因素**：先按它分大档。
- **MLFQ（`mlfq_level`）是第二决定因素**：只有当两个请求 `effective_priority` **相等**时，才比 MLFQ level（level 小的优先）。
- **到达时间是第三兜底**：QoS 和 MLFQ 都相等时，才 FCFS。
