# 基于 vLLM V1 的推理引擎 QoS 与时延优化

> 结合云网络虚拟网关流量调度 + 云存储限速/时延优化经验，实现大模型推理引擎的高优保障与资源可控性

## 项目背景

大模型推理引擎本质是**高并发请求的调度、资源管控与时延抖动控制**，与云网络虚拟网关（流量转发/优先级调度）、云存储（时延优化/限速流控）工作高度同源。

本项目基于 **vLLM V1 架构**源码二次开发，将云原生 Infra 的核心能力平移至 LLM 推理场景，解决线上推理的核心痛点：

1. 长请求占满资源，导致短请求尾时延飙升（Head-of-Line Blocking）；
2. KV 缓存显存抢占严重，V1 无水位线/配额控制，抢占代价极高（需从头 Recompute）；
3. KV Cache 仅驻留 GPU 显存，无分层存储，长上下文（128K+）易打爆单卡显存；
4. 资源无配额限制，无准入控制，服务质量不可控；
5. 缺少时间感知调度，无 Deadline/SLA 机制；
6. 缺少多租户公平性保障，无加权调度与配额隔离能力。

### V1 架构关键特征（本项目基于此开发）

| 特征 | V1 现状 | 优化空间 |
|------|---------|----------|
| 调度策略 | 仅 FCFS（先来先服务） | 无优先级、无公平性、无时间感知 |
| 优先级支持 | 无（Request 类无 priority 字段） | 需引入优先级框架 |
| 抢占策略 | LIFO（代价极高，Recompute） | 需按优先级抢占，减少抢占频率 |
| KV Cache 管理 | 有 `usage` 上报，无水位线/配额 | V0 有 watermark，V1 缺失 |
| KV Cache 存储 | 仅 GPU 显存，无分层 | V0 有 swap（已移除），需三级分层 |
| 准入控制 | 无（任何请求直接入队） | 无过载保护机制 |
| 速率控制 | 无 token 级限速 | 无法精细化控制单请求资源消耗 |

## 技术核心 & 实现方案

> 📋 **实现规划**：共 8 个优化点，按优先级从 P0 到 P3 逐步实现。每个优化点独立可用，高优先级的先实现。

---

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

#### 预期效果

- 短请求（<512 token）平均 TTFT（首 token 时延）显著降低
- 消除长 prefill 对短请求的 Head-of-Line Blocking
- 等待时间衰减因子确保低优请求不会被永远饿死

---

### 优化 2：KV 缓存显存水位线与配额流控 `[P0]` `[未实现]`

> 对应能力迁移：**云存储 IO QoS → 水位线流控 + per-request 带宽配额**

#### 核心思路

借鉴云存储**流量整形、分级水位线、per-request 带宽配额**思想，对 KV 缓存实现显存资源管控，避免显存抢占与抖动。

#### vLLM V1 现状分析

- V1 的 `KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`）只有 `usage` 属性上报使用率，**没有任何水位线机制**
- V0 的 `BlockManager`（`vllm/core/block_manager.py`）有 `watermark_blocks = int(watermark * num_gpu_blocks)` 水位线，但 **V1 完全移除了**
- V1 分配失败时直接触发抢占（Recompute 代价极高），没有任何缓冲/降级/限流机制
- 没有单请求 KV 块配额限制，单个长请求可占满所有显存

#### 实现细节

- 修改文件：
  - `vllm/v1/core/kv_cache_manager.py` — 水位线和配额逻辑
  - `vllm/v1/core/scheduler.py` — 调度器读取水位线状态做决策
- 核心逻辑：
  1. **多级水位线机制**（类比存储高低水位）：
     - 🟢 绿色（usage < 70%）：自由调度，所有请求正常分配
     - 🟡 黄色（70% ≤ usage < 85%）：拒绝新的低优请求入队，限制单请求 block 分配增速
     - 🔴 红色（usage ≥ 85%）：仅允许高优短请求，触发主动 LRU 回收
  2. **单请求 KV 块动态配额**：配额不是固定值，而是根据请求优先级 × 当前水位动态调整：
     - `max_blocks_per_req = base_quota × priority_factor × (1 - usage)`
  3. **主动回收**：水位超过黄色阈值时，主动释放 `FreeKVCacheBlockQueue` 中 `ref_cnt == 0` 的 prefix cache blocks，不等到分配失败才触发抢占。

#### 预期效果

- 显存占用峰值显著降低（从被动抢占变为主动控制）
- OOM 概率从抢占式兜底降至水位线防御，稳定性大幅提升
- 抢占频率降低（水位线提前拦截，减少 Recompute 开销）

---

### 优化 3：请求准入控制 `[P1]` `[未实现]`

> 对应能力迁移：**网络流量整形 Traffic Shaping → ECN 显式拥塞通知 + RED 随机早期丢弃**

#### 核心思路

从请求入口处做准入判断，在系统过载前主动拒绝/降级，避免所有请求挤入队列后才被动抢占。类比网络 ECN + RED：**在拥塞发生前就开始控制流入**。

#### vLLM V1 现状分析

- 任何请求到达后直接进入 `waiting` 队列，没有任何准入门槛
- 系统过载时只能依赖调度器的抢占机制（代价极高）来事后补救
- 没有队列深度限制，waiting 队列可以无限增长

#### 实现细节

- 修改文件：
  - `vllm/v1/engine/processor.py` — 请求入口处新增准入判断
  - `vllm/v1/core/kv_cache_manager.py` — 暴露水位线状态接口
- 核心逻辑：
  1. **基于 KV Cache 水位的准入控制**：
     ```
     if kv_usage > RED_WATERMARK:
         仅放行 priority == HIGH 的请求
     elif kv_usage > YELLOW_WATERMARK:
         拒绝预估 KV 占用大于阈值的新请求（按 prompt_token_count 预估）
     else:
         自由放行
     ```
  2. **基于队列深度的准入控制**：
     ```
     if len(waiting) > MAX_QUEUE_DEPTH:
         拒绝请求，返回 503 Service Unavailable（带 Retry-After）
     ```
  3. **拒绝策略**：被拒绝的请求不是直接丢弃，而是返回明确的错误码和重试建议（类比网络 ECN 通知上游降速）。

#### 预期效果

- 从源头控制系统负载，避免抢占风暴
- 系统在高负载下保持稳定的服务质量，而非全面退化

---

### 优化 4：Token 级速率控制 `[P1]` `[未实现]`

> 对应能力迁移：**云存储 IO 限速 → 令牌桶（Token Bucket）/ 漏桶（Leaky Bucket）**

#### 核心思路

借鉴存储 IO QoS 中的**令牌桶限速**机制，对低优请求实施 token 生成速率限制，将计算资源让给高优请求，而非简单粗暴地抢占（抢占 = Recompute，代价极高）。

#### vLLM V1 现状分析

- 没有任何 token 级速率控制
- 每个请求被调度后，以系统最大速度消耗 token_budget，无法控制单请求的资源消耗速率
- 调度器的 `token_budget` 是全局共享的，没有 per-request 预算分配机制

#### 实现细节

- 修改文件：
  - `vllm/v1/core/scheduler.py` — 调度时按速率限制分配 token 数
  - `vllm/v1/request.py` — Request 新增速率控制状态
- 核心逻辑：
  1. **Per-request 令牌桶**：
     ```python
     class TokenRateLimiter:
         rate: float    # 每步允许的平均 token 数
         burst: int     # 突发容量
         tokens: float  # 当前桶内余量
     ```
  2. **差异化限速**：高优请求不限速（rate=∞），低优请求根据当前系统负载动态调整 rate：
     - 系统空闲时：低优也不限速（充分利用资源）
     - 系统繁忙时：低优限速，为高优让出 token_budget
  3. **集成到调度器**：在 `schedule()` 中计算 `num_new_tokens` 时，叠加速率限制：
     ```python
     num_new_tokens = min(num_new_tokens, token_budget, rate_limiter.available())
     ```

#### 预期效果

- 高负载下高优请求的 TPOT（每 token 时延）更稳定
- 减少抢占频率（通过限速实现软性资源调节，避免硬性抢占）
- 整体吞吐不降低（空闲时限速自动放开）

---

### 优化 5：Deadline-aware 调度 `[P2]` `[未实现]`

> 对应能力迁移：**网络 QoS Deadline 队列 → fq_codel / HFSC 调度**

#### 核心思路

引入时间感知调度，为每个请求设定 SLA 目标（TTFT 首 token 时延、TPOT 每 token 时延），调度器感知请求的**松弛时间（slack time）**，优先调度最紧急的请求。

#### vLLM V1 现状分析

- **完全没有时间感知**：无 deadline、无 timeout、无 SLA 机制
- 请求无论等多久都只能被动等待，没有紧急程度概念
- 没有区分"快要超时的请求"和"刚到的请求"

#### 实现细节

- 修改文件：
  - `vllm/v1/request.py` — 新增 deadline 和时间感知字段
  - `vllm/v1/core/scheduler.py` — EDF 调度逻辑
  - `vllm/v1/engine/processor.py` — API 层支持 deadline / SLA 参数
- 核心逻辑：
  1. **请求 SLA 定义**：
     ```python
     class Request:
         deadline: Optional[float]       # 绝对截止时间（秒级时间戳）
         ttft_target: Optional[float]    # TTFT 目标（如 200ms）
         tpot_target: Optional[float]    # TPOT 目标（如 50ms/token）
         sla_class: str = "best_effort"  # SLA 等级: "realtime" / "interactive" / "batch" / "best_effort"
     ```
  2. **EDF（Earliest Deadline First）调度算法**：
     - 核心思想：**deadline 最早的请求最先被调度**，这是实时系统中理论最优的单处理器调度算法
     - 对 `waiting` 队列按 deadline 排序，deadline 最早的排队首
     - 对于没有显式 deadline 的请求，根据 `sla_class` 自动推算：
       ```python
       def compute_deadline(request: Request) -> float:
           if request.deadline is not None:
               return request.deadline
           # 根据 SLA 等级自动推算
           base_latency = {
               "realtime": 0.1,      # 100ms 内必须出首 token
               "interactive": 0.5,   # 500ms 内
               "batch": 5.0,         # 5s 内
               "best_effort": float('inf'),  # 无期限
           }
           return request.arrival_time + base_latency[request.sla_class]
       ```
  3. **松弛时间（Slack Time）感知**：
     ```
     slack_time = deadline - now - estimated_remaining_time
     ```
     - slack_time > 0：还有余量，可以等待
     - slack_time ≈ 0：紧急，必须立即调度
     - slack_time < 0：已超期，触发降级或主动释放
  4. **超时处理策略**（类比网络丢弃过期包）：
     - 超过 deadline 且尚未开始 prefill → 主动丢弃，返回 408 Timeout
     - 超过 deadline 但已在生成中 → 标记为低优，不再抢占其他请求的资源
     - 所有超时释放的资源立即回收给紧急请求

#### 预期效果

- TTFT P99 尾时延显著降低（EDF 保证最紧急请求最先处理）
- 资源不浪费在注定超时的请求上
- 支持精细化 SLA 分级（realtime / interactive / batch / best_effort）
- 这是 vLLM 社区目前**完全没有人在做**的方向

---

### 优化 6：加权公平队列 WFQ 调度 `[P2]` `[未实现]`

> 对应能力迁移：**网络 WFQ/DRR 流量调度 → 多租户加权公平**

#### 核心思路

借鉴网络调度中的 **WFQ（Weighted Fair Queuing）/ DRR（Deficit Round Robin）** 算法，在多租户推理场景下，按权重公平分配 GPU 计算资源。

#### vLLM V1 现状分析

- 只有 FCFS 和静态 Priority 两种策略
- 没有任何公平性保障：一个租户的大量请求可以饿死其他租户
- 没有"租户"维度的概念

#### 实现细节

- 修改文件：
  - `vllm/v1/core/scheduler.py` — WFQ 调度逻辑 + token budget 池化
  - `vllm/v1/request.py` — 新增 tenant_id 字段
  - `vllm/v1/engine/processor.py` — API 层支持 tenant_id 参数
  - 新增 `vllm/v1/core/tenant_manager.py` — 租户配额管理
- 核心逻辑：
  1. **租户队列**：按 `tenant_id` 分组，每个租户独立 waiting 队列
  2. **虚拟时间戳调度（WFQ）**：
     ```python
     class WFQScheduler:
         queues: Dict[str, Deque[Request]]  # 租户 → 请求队列
         weights: Dict[str, float]          # 租户 → 权重
         virtual_time: Dict[str, float]     # 租户 → 虚拟时间戳
         
         def select_next(self) -> Request:
             # 选 virtual_time 最小的租户的队首请求
             tenant = min(active_tenants, key=lambda t: virtual_time[t])
             req = queues[tenant].popleft()
             virtual_time[tenant] += cost / weights[tenant]
             return req
     ```
  3. **租户级 Token Budget 池化**（类比存储 QoS 的 per-tenant IOPS 配额）：
     - 每个租户分配独立的 token budget 池，全局 `max_num_batched_tokens` 按权重划分：
       ```python
       class TenantManager:
           tenant_budgets: Dict[str, int]  # 租户 → 当前可用 token budget
           
           def allocate_budgets(self, total_budget: int):
               """每个调度步开始时，按权重分配 token budget"""
               total_weight = sum(self.weights.values())
               for tenant_id, weight in self.weights.items():
                   self.tenant_budgets[tenant_id] = int(
                       total_budget * weight / total_weight
                   )
           
           def try_consume(self, tenant_id: str, tokens: int) -> int:
               """租户消耗 token budget，返回实际可用量"""
               available = self.tenant_budgets[tenant_id]
               consumed = min(tokens, available)
               self.tenant_budgets[tenant_id] -= consumed
               return consumed
       ```
     - **Budget 借用机制**：当某租户空闲时，其未使用的 budget 可被其他租户借用（work-conserving），但有借用上限（防止突发霸占）
  4. **KV Cache 配额隔离**：每个租户的 KV blocks 总占用不超过配额上限
     ```python
     max_kv_blocks_per_tenant = total_gpu_blocks * weight / total_weight * 1.2  # 允许 20% 超售
     ```
  5. **权重可配置**：通过 API 或配置文件设定租户权重，支持动态调整（热更新）

#### 预期效果

- 多租户场景下各租户按权重获得 GPU 时间片
- 避免单个租户的突发流量饿死其他租户
- 适用于推理服务平台（一个 vLLM 实例服务多个业务线）

---

### 优化 7：MLFQ 多级反馈队列 `[P3]` `[已实现]`

> 对应能力迁移：**OS 进程调度 CFS → 多级反馈 + 自适应优先级**

#### 核心思路

借鉴 OS 的 **MLFQ（Multi-Level Feedback Queue）** 调度思想：请求初始进入最高优先级队列，随着消耗的 token_budget 增加逐级降低优先级。短请求天然在高优级别完成，长请求逐渐降级——**无需显式标注优先级即可实现自适应调度**。

#### vLLM V1 现状分析

- 请求优先级是静态的（如果有的话），不会随运行状态变化
- 没有根据请求运行时"行为"动态调整调度优先级的机制
- 长请求和短请求在调度上完全平等（FCFS）

#### 实现细节

- 修改文件：
  - `vllm/v1/request.py` — 新增 `MLFQLevel` 配置类、`MLFQ_LEVELS` 全局级别定义、Request 新增 `mlfq_level`/`mlfq_tokens_consumed` 字段和 `mlfq_account_tokens()`/`mlfq_promote()` 方法
  - `vllm/v1/core/scheduler.py` — 新增 `mlfq_queues`（N 级 deque）、`_mlfq_peek_next()`/`_mlfq_pop_next()`/`_mlfq_remove_from_level()` 辅助方法；调度循环从最高级队列优先取请求；抢占时调用 `mlfq_promote()` 升级；`update_from_output()` 中调用 `mlfq_account_tokens()` 实现自动降级
- 核心逻辑：
  1. **多级队列**：定义 4 个优先级级别，每级有不同的 token 配额（时间片）：
     ```python
     MLFQ_LEVELS = [
         MLFQLevel(level=0, name="interactive", token_quota=128),    # L0: 短对话
         MLFQLevel(level=1, name="standard",    token_quota=512),    # L1: 普通请求
         MLFQLevel(level=2, name="batch",       token_quota=2048),   # L2: 长请求
         MLFQLevel(level=3, name="background",  token_quota=inf),    # L3: 后台任务
     ]
     ```
  2. **降级规则**：请求在当前级别累计消耗的 output token 超过该级配额后，自动降到下一级（`mlfq_account_tokens()`）
  3. **升级规则**：被抢占的请求升一级（`mlfq_promote()`），防止饿死；token 消耗计数不重置，防止反复抢占刷级
  4. **调度顺序**：从 L0 开始扫描，当前级别有请求则优先调度，级别内 FCFS

#### 预期效果

- 无需人工标注优先级，系统自动对短请求提供低时延保障
- 长请求自然降级但不会饿死（有升级兜底）
- 可作为优化 1（静态 QoS 分级）的最终演进形态

---

### 优化 8：KV Cache 分层存储与智能迁移 `[P1]` `[未实现]`

> 对应能力迁移：**存储系统热温冷分层（Hot/Warm/Cold Tiering）+ 块迁移（Block Migration）+ 预取（Prefetch）**

#### 核心思路

将存储系统的**分级存储（Tiered Storage）** 思想引入 KV Cache 管理：把 GPU 显存当 L1 热层、CPU DRAM 当 L2 温层、NVMe SSD 当 L3 冷层，实现 KV Cache 的块级粒度迁移和智能预取，从根本上解决 V1 显存不足时只能 Recompute 的问题。

#### vLLM V1 现状分析

- KV Cache **全量驻留 GPU 显存**，没有任何分层存储机制
- V0 曾支持 CPU swap（GPU ↔ CPU 全量搬运），但 V1 架构中**完全移除了 swap 功能**
- 显存不足时唯一选项是 **Recompute**（`num_computed_tokens = 0`，丢弃全部 KV Cache 从头重算），代价极高
- 长上下文请求（128K+ tokens）容易打爆单卡显存，无法支撑
- `FreeKVCacheBlockQueue` 只管理 GPU 上的 blocks，没有多级存储抽象

#### 架构设计

```
当前 V1 架构（单层）：

  ┌───────────────────────────┐
  │     GPU 显存（唯一层）      │  ← 所有 KV Cache
  │   PagedAttention Blocks    │  ← 不够 → Recompute（全量重算）
  └───────────────────────────┘

改造后（三级分层）：

  ┌───────────────────────────┐
  │   L1: GPU 显存（热层）      │  ← 活跃请求的 KV Cache
  │   延迟 ~ns, 带宽 ~TB/s     │  ← 直接参与 Attention 计算
  ├───────────────────────────┤
  │   L2: CPU DRAM（温层）      │  ← 被抢占/暂停请求的 KV Cache
  │   延迟 ~μs, PCIe ~64GB/s  │  ← 保留而非丢弃，恢复无需重算
  ├───────────────────────────┤
  │   L3: NVMe SSD（冷层）      │  ← 长上下文历史 KV Cache
  │   延迟 ~ms, 带宽 ~7GB/s    │  ← 128K context 不再打爆显存
  └───────────────────────────┘

  数据流：
  ↑ 预取（Prefetch）：L3→L2→L1（请求即将被调度时提前搬运）
  ↓ 降级（Demote）  ：L1→L2→L3（按优先级/访问热度逐级下沉）
```

#### 实现细节

- 新增/修改文件：
  - `vllm/v1/core/kv_cache_manager.py` — 扩展为多级存储管理
  - `vllm/v1/core/kv_cache_utils.py` — 新增 `TieredBlockPool`，管理三级 block 池
  - `vllm/v1/core/scheduler.py` — 抢占时降级而非丢弃，调度时触发预取
  - 新增 `vllm/v1/core/block_migrator.py` — 块迁移引擎
- 核心逻辑：
  1. **多级 Block 池**（类比存储分层）：
     ```python
     class TieredKVCacheManager:
         gpu_pool: FreeKVCacheBlockQueue    # L1: GPU 显存
         cpu_pool: CPUBlockPool             # L2: CPU DRAM（pinned memory）
         disk_pool: DiskBlockPool           # L3: NVMe SSD（异步 IO）
         
         block_location: Dict[int, Tier]    # block_id → 当前所在层级
     ```
  2. **块级粒度迁移**（对比 V0 的全量 swap）：
     - V0 swap 是把整个请求的所有 KV blocks 一次性搬到 CPU，粒度粗、延迟高
     - 本方案按**单个 block** 粒度迁移，类比存储系统的 **page migration**
     - 使用 CUDA 异步 memcpy（`cudaMemcpyAsync` + dedicated stream），不阻塞 GPU 计算
  3. **智能降级策略**（对比简单 LRU）：
     - 被抢占的请求：KV Cache **降级到 L2（CPU DRAM）** 而非丢弃
     - 长时间未被调度的温层数据：进一步降级到 L3（NVMe）
     - 降级决策综合考虑：请求优先级、block 访问热度（last_accessed）、剩余生成量
     ```python
     def demote_policy(block: KVCacheBlock) -> Tier:
         if block.request.priority == HIGH:
             return Tier.CPU    # 高优请求只降到 CPU，保证快速恢复
         if time.now() - block.last_accessed > COLD_THRESHOLD:
             return Tier.DISK   # 长时间未访问降到磁盘
         return Tier.CPU        # 默认降到 CPU
     ```
  4. **智能预取**（类比存储预读）：
     - 当 waiting 队列中的请求即将被调度时，提前把其 KV Cache 从 L2/L3 搬回 L1
     - 预取时机：调度器的 `schedule()` 每步结束时，预测下一步可能调度的 top-K 请求，异步发起预取
     - 预取带宽控制：不能让预取占满 PCIe 带宽影响正常推理，设置预取带宽上限
     ```python
     def prefetch_candidates(self) -> List[Request]:
         """预测下一步最可能被调度的请求"""
         candidates = sorted(self.waiting, key=lambda r: r.effective_priority)
         return candidates[:PREFETCH_TOP_K]
     ```
  5. **抢占改造**：抢占不再 Recompute，而是降级保存：
     ```python
     # 改造前（V1 原版）：
     preempted_req.num_computed_tokens = 0  # 丢弃，全量重算
     
     # 改造后：
     self.kv_cache_manager.demote_to_cpu(preempted_req)  # 降级到 CPU
     preempted_req.kv_tier = Tier.CPU  # 标记 KV 所在层级
     # 恢复时：prefetch 从 CPU 搬回 GPU，无需重算
     ```

#### 与其他优化点的协同

- **+ 优化 2（水位线流控）**：L1（GPU）水位线达到黄色时，主动触发降级迁移到 L2，而非等到分配失败
- **+ 优化 1（QoS 分级）**：降级时优先降低优先级的请求的 KV Cache
- **+ 优化 3（准入控制）**：准入判断时考虑 L2/L3 的可用容量，而非只看 L1

#### 预期效果

- **抢占代价大幅降低**：从 Recompute（全量重算）降为 CPU→GPU memcpy（毫秒级恢复）
- **支撑长上下文**：128K+ token 请求不再打爆单卡显存，历史 KV Cache 可下沉到 CPU/NVMe
- **显存利用率提升**：GPU 显存集中服务活跃请求，非活跃数据自动下沉
- **整体吞吐提升**：减少 Recompute 的 GPU 算力浪费，计算资源更多用于有效推理

---

## 优化点总览与依赖关系

```
优化 1 (QoS 分级调度) ──┐
                         ├──→ 优化 3 (准入控制) ──→ 优化 5 (Deadline/EDF 调度)
优化 2 (KV 水位线流控) ──┤         │
                         │         ├──→ 优化 4 (Token 限速)
优化 8 (KV 分层存储) ────┘         │
                         优化 6 (WFQ 加权公平) ──→ 优化 7 (MLFQ 多级反馈)
```

**依赖说明**：
- 优化 8（KV 分层存储）与优化 2（水位线流控）紧密协同：水位线触发降级迁移
- 优化 8 与优化 1（QoS 分级）协同：降级时按优先级选择目标
- 优化 5（EDF 调度）是优化 1 的时间维度增强，需要优先级框架作为基础
- 优化 6 和优化 7 相对独立，可与其他优化并行开发

### 优化 1 / 6 / 7 三者协同关系（调度三层架构）

优化 1（QoS 分级调度）、优化 6（WFQ 加权公平）、优化 7（MLFQ 多级反馈队列）解决的是**三个完全不同维度**的调度问题，它们分别回答：

| 维度 | 优化点 | 核心问题 | 类比 |
|------|--------|---------|------|
| **租户间** | 优化 6（WFQ） | 每个租户该分到多少 GPU 资源？ | 高速公路按城市分配车道数 |
| **租户内·请求间** | 优化 1（QoS） | 哪个请求更重要？ | 救护车走应急车道 |
| **同优先级·请求间** | 优化 7（MLFQ） | 哪个请求更快能完成？ | 超市快速结账通道 |

**为什么三者缺一不可：**

1. **只有优化 1（QoS 分级）不够**：
   - 优化 1 依赖 API 调用方**显式传入 `priority`**，但并非所有场景都能标注（SaaS 平台、统一网关转发等）
   - 优先级基于**预估**（如 prompt 长度），但输入短 ≠ 输出短（"写一篇 5000 字论文"只有 15 tokens prompt）
   - 没有**租户维度**隔离，单租户可发起大量同优先级请求霸占所有资源

2. **只有优化 7（MLFQ）不够**：
   - MLFQ 根据请求实际 token 消耗**自动推断**优先级，无需人工标注
   - 但请求刚到达时都从 L0 开始，**无法在入口处区分 VIP 和普通请求**
   - 同样没有**租户维度**隔离能力

3. **只有优化 6（WFQ）不够**：
   - WFQ 按租户权重公平分配 GPU 时间，但**租户内部**的请求全部 FCFS
   - 无法区分同一租户内的高优和低优请求

**三者协同的最终形态：**

```
请求到达
    │
    ▼
优化 6（WFQ）：按 tenant_id 分流到各租户队列
    │          保证每个租户拿到其权重对应的 GPU 份额
    │
    ├── 租户 A 队列（权重 50%）──┐
    ├── 租户 B 队列（权重 30%）──┼── 每个租户队列内部：
    └── 租户 C 队列（权重 20%）──┘
            │
            ▼
    优化 1（QoS 分级）：同一租户内，高优请求先调度
            │
            ▼
    优化 7（MLFQ）：同一优先级内，短请求自动优先
```

| 层级 | 谁来决定 | 决定什么 | 防护的场景 |
|------|---------|---------|-----------|
| **租户间** | 优化 6（WFQ） | A 拿 50%，B 拿 30%，C 拿 20% | 大租户突发流量饿死小租户 |
| **租户内·请求间** | 优化 1（QoS） | VIP 请求先于普通请求 | 低优请求挡住高优请求 |
| **同优先级·请求间** | 优化 7（MLFQ） | 短请求自动优先于长请求 | 长请求 Head-of-Line Blocking |

### 优化 4（Token 限速）与优化 1/6/7 的关系：调度入口 vs 运行时控制

优化 1/6/7 解决的都是同一个问题的不同维度：**"谁先被调度"（调度入口决策）**。而优化 4 解决的是一个完全不同阶段的问题：**"被调度之后跑多快"（运行时资源消耗控制）**。

```
请求的完整生命周期：

        ┌─────── 调度入口（谁先被选中）──────┐     ┌── 运行时（选中后跑多快）──┐
        │                                    │     │                          │
到达 → 优化6 → 优化1 → 优化7 → 被选中调度 → │ →  │  优化4（Token 限速）       │ → 完成
        │ 租户分流  优先级排序  自适应排序     │     │  控制每步生成多少 token    │
        └────────────────────────────────────┘     └──────────────────────────┘
             "决定调度顺序"                              "控制资源消耗速率"
```

**关键区别**：

| 维度 | 优化 1/6/7（调度入口） | 优化 4（运行时控制） |
|------|---------------------|-------------------|
| **作用时机** | 请求从 waiting → running 时 | 请求已在 running 中 |
| **控制对象** | 调度**顺序** | 生成**速率** |
| **类比** | 高速公路收费站：决定谁先上路 | 限速牌：已上路的车能开多快 |
| **解决的问题** | 谁先拿到 GPU 时间 | 拿到 GPU 时间后用多少 |

**为什么有了优化 1/6/7 还需要优化 4？**

优化 1/6/7 解决了"谁先上路"，但一旦请求被选中进入 running 状态，它就以系统最大速度消耗 `token_budget`。这在高负载下会导致问题：

```
场景：10 个请求在 running 中，全局 token_budget = 2048

没有优化 4：
  高优请求 × 3: 各消耗 ~200 tokens/step → 合计 600
  低优请求 × 7: 各消耗 ~200 tokens/step → 合计 1400
  → 低优请求消耗了 68% 的 budget！高优请求的 TPOT 被拖慢

有了优化 4：
  高优请求 × 3: rate=无限 → 各消耗 ~200 tokens/step → 合计 600
  低优请求 × 7: rate=50 tokens/step → 各消耗 50 → 合计 350
  → 剩余 budget 全部留给高优请求，高优 TPOT 稳定
```

**优化 4 的核心价值**：不是把低优请求踢出去（抢占代价太高），而是**让它慢慢跑**，把计算资源让给高优请求。这比抢占（Recompute）要优雅得多：

| 策略 | 对低优请求的影响 | 代价 |
|------|---------------|------|
| 抢占（优化 1 的极端情况） | 被踢出 running，KV Cache 全部丢失，需从头重算 | 极高（Recompute） |
| 限速（优化 4） | 留在 running 中，只是跑慢一点 | 几乎为零 |

**四者协同的完整调度架构**：

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
优化 7（MLFQ）：同优先级内自适应排序
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

| 优化点 | 优先级 | 状态 | 核心修改文件 | Infra 能力对标 |
|--------|--------|------|-------------|---------------|
| 1. QoS 分级调度 | P0 | ✅ 已实现 | `vllm/v1/core/scheduler.py`, `vllm/v1/request.py` | 网关优先级队列 |
| 2. KV 水位线流控 | P0 | 🔲 未实现 | `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/scheduler.py` | 存储水位线 + IO 配额 |
| 3. 准入控制 | P1 | 🔲 未实现 | `vllm/v1/engine/processor.py`, `vllm/v1/core/kv_cache_manager.py` | ECN + RED 拥塞控制 |
| 4. Token 限速 | P1 | 🔲 未实现 | `vllm/v1/core/scheduler.py`, `vllm/v1/request.py` | 令牌桶 / 漏桶限速 |
| 5. Deadline/EDF 调度 | P2 | 🔲 未实现 | `vllm/v1/request.py`, `vllm/v1/core/scheduler.py`, `vllm/v1/engine/processor.py` | EDF + fq_codel / HFSC |
| 6. WFQ 公平调度 | P2 | 🔲 未实现 | `vllm/v1/core/scheduler.py`, `vllm/v1/request.py`, 新增 `vllm/v1/core/tenant_manager.py` | WFQ / DRR + per-tenant 配额 |
| 7. MLFQ 多级反馈 | P3 | ✅ 已实现 | `vllm/v1/core/scheduler.py`, `vllm/v1/request.py` | OS MLFQ / CFS |
| 8. KV Cache 分层存储 | P1 | 🔲 未实现 | `vllm/v1/core/kv_cache_manager.py`, `vllm/v1/core/kv_cache_utils.py`, 新增 `vllm/v1/core/block_migrator.py` | 存储热温冷分层 + 预取 |

## 项目亮点

1. **技术迁移性强**：将云网络/存储的调度、限速、时延优化、分层存储能力直接复用至推理引擎，非纯 AI 调参，具备工程落地价值；
2. **基于 V1 架构**：V1 是 vLLM 的最新主力架构（多进程 ZMQ IPC），所有优化直接在 V1 上开发，确保前瞻性；
3. **渐进式实现**：8 个优化点从 P0 到 P3 逐步实现，每个独立可用、可测试、可量化；
4. **填补社区空白**：KV Cache 分层存储、Deadline/EDF 调度、Token 限速、MLFQ 等机制在 vLLM V1 社区均无人实现，差异化明显；
5. **两大核心方向**：调度器优化（优化 1/3/4/5/6/7）+ KV Cache 存储优化（优化 2/8），覆盖推理引擎两大核心子系统；
6. **贴近线上场景**：解决的都是推理引擎上线的真实痛点（时延抖动、OOM、资源抢占、长上下文支撑），而非玩具项目。

## 性能测试对比

> ⚠️ 以下为预期性能目标，待各优化点实现后逐步补充实测数据。

| 指标 | 原生 vLLM V1 | 优化后（预期） | 优化来源 |
|------|-------------|---------------|----------|
| 短请求 TTFT P99 | 基准 | ↓ 30-50% | 优化 1 + 优化 5 |
| 短请求 TPOT 抖动 | 基准 | ↓ 40%+ | 优化 4 + 优化 2 |
| 最大显存占用 | 基准 | ↓ 20-30% | 优化 2 + 优化 3 |
| 长请求 OOM 概率 | 存在 | → 0% | 优化 2 + 优化 3 + 优化 8 |
| 抢占频率 | 基准 | ↓ 50%+ | 优化 2 + 优化 4 + 优化 8 |
| 抢占恢复耗时 | Recompute（秒级） | CPU→GPU memcpy（毫秒级） | 优化 8 |
| 长上下文支撑（128K+） | 单卡易 OOM | 三级分层，稳定运行 | 优化 8 |
| 多租户公平性 | 无保障 | Jain's Index > 0.9 | 优化 6 |

## 如何运行

### 环境准备

1. GPU 实例（A10/T4/L4 均可），Ubuntu 20.04+，CUDA 12.1+；
2. 安装依赖：
   ```bash
   git clone https://github.com/你的用户名/vllm-inference-qos-latency-optimization.git
   cd vllm-inference-qos-latency-optimization
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
# 待实现：对比测试脚本
python benchmarks/qos_benchmark.py --baseline --optimized --output results/
```

## 实现进度追踪

- [x] 优化 1：QoS 分级调度（P0）
- [ ] 优化 2：KV 缓存水位线流控（P0）
- [ ] 优化 8：KV Cache 分层存储与智能迁移（P1）
- [ ] 优化 3：请求准入控制（P1）
- [ ] 优化 4：Token 级速率控制（P1）
- [ ] 优化 5：Deadline/EDF 调度（P2）
- [ ] 优化 6：WFQ 加权公平调度（P2）
- [x] 优化 7：MLFQ 多级反馈队列（P3）
