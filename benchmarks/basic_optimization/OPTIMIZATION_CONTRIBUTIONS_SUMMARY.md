# vLLM V1 推理服务优化 — 核心贡献摘要

> 本文档以"背景与挑战 / 核心贡献 / 收益"的格式，汇总在 vLLM V1 上完成的两大类推理服务优化工作。
> 定位：**已有成熟技术理念在 vLLM V1 上的系统性迁移、整合与工程落地**（非原创算法），
> 价值在于工程系统能力——正确、成体系地落地到生产级引擎并验证清楚效果与边界。
>
> 详细设计见：
> [`scheduling-resource-management-optimization.md`](./scheduling-resource-management-optimization.md)、
> [`prefix-cache-scheduling-optimization.md`](./prefix-cache-scheduling-optimization.md)、
> [`suffix-decoding-optimization.md`](./suffix-decoding-optimization.md)；
> 测试方案见 `benchmarks/e2e_cases/` 下对应 GUIDE 文档。

---

## 一、调度与资源管理优化（跨领域调度理念的系统性迁移）

### 背景与挑战

vLLM V1 原生调度以 FCFS + 基础 priority 为主，在**多租户、混合负载、过载**场景下缺乏精细的资源治理能力：高优请求会被大批长请求阻塞、低优请求过载时无准入保护、单租户可占满全部资源、长文档 Prefill 饿死短请求。业界网络（QoS/WFQ/令牌桶/ECN）与操作系统（MLFQ/EDF）领域早有成熟的调度治理理念，但尚未被**成体系地**迁移进 LLM 推理调度器。

### 核心贡献

独立设计并实现了一套**跨领域调度理念迁移**的完整调度流水线，将网络/OS 的成熟调度算法系统性整合进 vLLM V1 调度器（`vllm/v1/core/sched/scheduler.py`），形成**入口准入 → waiting 排序 → running 控速**三层协同：

- **QoS 分级调度**：多维有效优先级（业务优先级 + prompt 长度分级 + 等待衰减防饿死），决定 waiting 出队顺序（对标网络优先级队列）。
- **MLFQ 多级反馈队列**：按请求累计 token 消耗自动降级（128/512/2048/∞ 四级），无需人工标注即可让长请求自动让位（对标 OS CFS/MLFQ）。
- **Token 级速率控制**：per-request 令牌桶 + 负载感知动态限速（依据 MLFQ 分档，高负载下压低低优请求的每步 token 数），在 running 内部做算力再分配（对标云存储 IO 限速）。
- **请求准入控制**：过载时（队列深度 ≥ 40 / SLA 违约率 > 0.25）主动拒绝低优请求、保护高优（对标网络 ECN/RED 流量整形）。
- **Deadline-aware（EDF）调度**：按截止时间紧急度排序，抢占时优先牺牲已违约请求（对标网络 EDF/fq_codel）。
- **WFQ 加权公平 + 租户隔离**：per-租户并发上限（默认 64）+ 动态权重，防止单租户占满资源（对标网络 WFQ/DRR）。
- **配套：长文档 Prefill 让位**：限制长 continuation prefill 每步并发数，缓解长文档饿死短请求。

工程落地上，处理了旧版 → v0.20.2 的 `RequestQueue` 抽象移植、优先级队列堆不变量、被拒请求拒绝路径完整化（修复 SLA 窗口填充、挂起请求泄漏等 4 个真实 bug）、状态跨 waiting↔running 往返维护等难题；每个优化提供独立开关（`VLLM_DISABLE_*`）支持 A/B 消融。

### 收益

通过一整套 A/B 压测**诚实验证了优化效果与能力边界**（详见 `SCHEDULING_BENCHMARK_GUIDE.md` §7）：调度优化能显著保护 silver/bronze 租户（临界过载区间 SLA 违约率 **-53~-56pp**），同时明确指出其边界——**无法解决物理算力约束**（最高优先级 gold 自身过载时调度无解）。验证过程包含"实测 → 证伪 → 修正"闭环（如更正早先"GPU 算力耗尽"的误判、定位 `--enforce-eager` 与长文档为真实根因），沉淀出可复现的调优参数与评估方法学。

---

## 二、Prefix Cache 感知调度与 KV Cache 复用优化

### 背景与挑战

vLLM V1 的 Prefix Caching 机制本身已较完善（同步写入、同步骤内可共享物理 block），但在**调度策略与缓存管理策略**层面仍有明显空间：调度器不感知缓存命中率导致 token_budget 利用率低、纯 LRU 驱逐让高频 System Prompt 被低频长请求挤出（缓存震荡）、抢占过于激进导致恢复代价高、冷启动无预热。

### 核心贡献

独立设计并实现了一套 **Prefix Cache 与调度器协同**的 KV Cache 复用优化（`vllm/v1/core/scheduler.py`、`kv_cache_manager.py`、`kv_cache_utils.py`）：

- **Cache-Aware Scheduling（缓存感知调度）**：在 MLFQ 同层级内引入缓存命中感知排序，优先调度实际 prefill token 少（命中率高）的请求，用扫描窗口（K=8）控制开销，同 token_budget 下服务更多请求。
- **Frequency-Aware Eviction（频率感知驱逐）**：将 LRU 空闲队列改造为 **Segmented LRU（Probation/Protected 双区）**，被再次命中的 block 升级到保护区，避免高频前缀被误驱逐；所有操作维持 O(1)。
- **Preemption Cache Shield（抢占缓存保护）**：新增 `free_partial()`，抢占时保留连续有 hash 的前缀 block、只释放尾部，恢复时从部分 recompute 起步；含"释放量不足时降级为全量释放"的抢占前进保障，避免空转。
- **主动缓存预热 / 缓存效率可观测性**（方案设计）：冷启动预热常见 System Prompt、多维缓存健康度指标（token 级节省率、健康驱逐率等）。

### 收益（预期，部分已实现验证中）

高缓存命中请求的 TTFT 降低约 30~50%、高频前缀命中率从 ~50-70% 提升到 ~85-95%（减少缓存震荡）、抢占恢复从全量 Recompute（秒级）降至部分 Recompute（百毫秒级）。与调度优化协同：MLFQ 层级内叠加缓存感知排序、Token 限速腾出的 budget 让更多高命中请求被调度、缓存保护避免抢占波及高频前缀。

---

## 三、后缀解码（Suffix Decoding）投机推理优化

### 背景与挑战

vLLM V1 的投机解码在 V1 架构下**仅支持 N-gram Proposer**（源码 assert 限制），且 N-gram 存在固定窗口（n 选大匹配不到、选小接受率低）、只取首次匹配、无状态每次全量搜索 O(context_len)、仅搜本请求上下文等局限。在重复性、结构性强的场景（对话、代码补全、Agentic Coding）中，基于历史统计的后缀匹配本可取得更高接受率，但这一方向在 V1 尚未落地。

### 核心贡献

独立设计并实现了**基于后缀结构的投机解码方案**，接入 vLLM V1 的 `generate_draft_token_ids` 链路（`vllm/v1/spec_decode/`），并做了多层递进增强：

- **SuffixTreeProposer（后缀数组 + 二分）**：用 Numba JIT 的后缀数组替换 KMP，支持自适应回退与最优匹配选择，与 `NgramProposer` 接口完全兼容可直接替换。
- **增量后缀自动机（SAM）**：实现在线 O(1) 均摊追加的后缀自动机，per-request 维护、跨步复用，避免每步 O(context_len) 全量重建；处理了抢占后 context 收缩重建、请求生命周期清理等工程细节。
- **自适应匹配 + 多候选评分**：多级回退 + 四因子加权评分（匹配长度 / 续接长度 / 位置新旧 / **历史接受率**），带滑动窗口接受率反馈闭环，从"首次匹配"升级为"最优匹配"。
- **进阶方案设计**：全局后缀树 + 频率感知 draft 质量筛选（按期望收益筛选，减少无效投机）、动态投机长度（负载感知调 k，与调度优化共用负载信号）、树状多候选并行验证（接入 vLLM 树注意力，多押路径提命中率）。

工程上全部采用 Numba JIT 实现、每个 Proposer 配完整测试（累计上千行测试覆盖，含与 NgramProposer 的对拍、大上下文与高并发压力测试）。

### 收益（含诚实的边界结论）

后缀匹配查询复杂度从 O(context_len) 降至 O(pattern_len)、支持 O(1) 增量更新；在重复性高的场景预期每步有效 token 数从 ~1.5 提升到 ~3.0（等效约 2x Decode 加速）。测试中**诚实记录了负面结论**（详见 `SUFFIX_DECODING_BENCHMARK_GUIDE.md`）：在规则重复数据集上，因首次匹配已是最优，多候选评分未带来接受率提升——据此明确了"优化的有效场景是**不规则真实文本**（对话/Agentic Coding），需在 spec_bench/sharegpt 等数据集、并配合 proposer 内部指标归因后才能形成完整结论"。

---

## 四、总结：定位与价值

| 优化大类 | 对标业界方向 | 是否原创 | 核心工程价值 |
|---------|-------------|---------|-------------|
| 调度与资源管理 | 网络/OS 经典调度（QoS/WFQ/MLFQ/EDF/令牌桶/准入） | ❌ 移植整合 | 成体系整合成协同流水线 + 诚实验证能力边界 |
| Prefix Cache 复用 | KV 前缀缓存 + 缓存替换策略 | ❌ 增强官方机制 | 调度与缓存协同 + Segmented LRU/部分释放的工程实现 |
| 后缀解码 | SuffixDecoding / SGLang 后缀投机 | ❌ V1 移植增强 | 在只支持 ngram 的 V1 上移植后缀结构 + 四因子评分等增量 |

> **统一定位**：三大类工作均为"**成熟技术理念在 vLLM V1 上的系统性迁移、整合与工程落地**"，
> 而非原创算法。价值体现在：① 系统性整合（多机制协同不冲突）；② 生产级引擎落地（趟过移植/堆不变量/拒绝路径等真实工程坑）；
> ③ 可控实验与诚实验证（含证伪自己、讲清能力边界）；④ 跨领域/跨模块知识迁移（网络↔调度、后缀结构↔投机、负载信号跨模块复用）。

---

## 五、各已实现优化的核心逻辑伪代码

> 以下伪代码提炼每个**已实现**优化点的核心逻辑（省略工程细节），用于快速理解"每个优化到底做了什么"。
> 真实实现见对应源码文件。

### 5.0 请求完整生命周期主流程（各优化点的嵌入位置）

> 这段主流程把一个请求从**到达 → 调度 → forward → 采样/投机 → KV 管理 → 产出**的完整链路串起来，
> 并用 `# ★[优化X]` 标注每个已实现优化嵌入的位置。各优化的核心逻辑见 §5.1-§5.3 展开。

```python
# ============ 引擎主循环：每个 step 处理一批请求 ============
def engine_step():
    # ---------- 阶段 A：请求入口（新请求到达时）----------
    for req in newly_arrived_requests:
        if not should_admit(req, waiting, violation_window):   # ★[准入控制] 过载拒绝低优
            reject(req, code=503); continue
        if not tenant_mgr.can_schedule(req.tenant_id):         # ★[WFQ/租户隔离] 租户名额
            hold(req); continue
        waiting.add(req)                                       # 进等待队列

    # ---------- 阶段 B：调度（决定这一步谁进 running、各跑多少 token）----------
    scheduler_output = schedule()
    # ---------- 阶段 C：模型前向 forward（GPU 上并行执行）----------
    hidden = model.forward(scheduler_output.token_ids,         # 只算被调度的 token
                           attn_metadata=scheduler_output.attn_meta)
    # ---------- 阶段 D：采样 + 投机验证 ----------
    sampled = sample(hidden, scheduler_output.sampling_meta)

    if spec_decode_enabled:                                    # ★[后缀解码] 投机
        for req in running:
            suffix_sam[req.id].extend(sampled[req.id])         # ★ 增量 SAM 更新
            draft = adaptive_propose(req.context, ...)         # ★ 匹配+评分选 draft
            req.pending_draft = draft
        # 下一步 forward 会把 draft 一起验证；本步验证上一轮 draft：
        accepted = rejection_sample(prev_drafts, sampled)      # ★ 接受最长合法前缀
        update_accept_rate_tracker(accepted)                   # ★ 反馈闭环

    # ---------- 阶段 E：状态更新 + KV 管理 ----------
    for req in running:
        m = num_tokens_generated(req, sampled, accepted)
        req.mlfq_account_tokens(m)                             # ★[MLFQ] 累计消耗→可能降级
        if req.finished():
            block_pool.free(req)                               # ★[Segmented LRU] 释放进分区
            violation_window.append(req.is_sla_violated(now))  # 供准入控制算违约率
    return outputs


# ============ 调度核心：阶段 B 展开 ============
def schedule():
    now = time()
    # (1) waiting 队列排序：QoS 主键 → MLFQ 次键 → Deadline 插队
    for req in waiting:
        req.effective_priority = compute_effective_priority(req, now)  # ★[QoS]
    waiting.sort()                              # __lt__: QoS → mlfq_level → arrival ★[MLFQ]
    deadline_aware_reorder(waiting, now)        # ★[Deadline] 紧急请求插队

    token_budget = MAX_NUM_BATCHED_TOKENS
    scheduled = []
    # (2) 先调度已在 running 的请求（continuous batching：running 优先补位）
    update_rate_limiters(running, max_running)  # ★[Token 限速] 按负载+MLFQ 设 rate
    for req in running:
        n = min(remaining_need(req), token_budget)
        n = req.rate_limiter.consume(n)         # ★[Token 限速] 低优这步少给 token
        if is_long_continuation(req) and long_count >= 1:      # ★[长文档让位]
            continue
        alloc = allocate_slots(req, n)          # ★[PagedAttention] 分页 KV 分配
        if alloc is None:                       # KV 不够 → 抢占
            victim = select_preemption_victim(running)         # ★[Deadline] 先踢已违约
            free_partial(victim)                # ★[抢占缓存保护] 只释放尾部、留前缀
        scheduled.append((req, n)); token_budget -= n

    # (3) 再从 waiting 挑新请求进 running（缓存感知）
    while token_budget > 0 and waiting:
        req = cache_aware_select_next(mlfq_queues, scan_window=8)  # ★[缓存感知调度] 选命中多的
        if not tenant_mgr.can_schedule(req.tenant_id):             # ★[租户隔离] 再查名额
            continue
        computed = get_computed_blocks(req)     # ★[前缀缓存] 命中则复用（touch，见 §5.2）
        n = req.rate_limiter.consume(min(req.num_tokens - computed, token_budget))
        allocate_slots(req, n); running.add(req)
        scheduled.append((req, n)); token_budget -= n

    return build_scheduler_output(scheduled)
```

> **读法**：`engine_step()` 是引擎每步的主干（A 入口 → B 调度 → C forward → D 采样/投机 → E 状态/KV）；
> `schedule()` 是阶段 B 的展开。每处 `# ★[优化X]` 就是该优化嵌入主流程的位置——
> 下面 §5.1-§5.3 是这些 `★` 函数各自的核心逻辑。

---

### 5.1 调度与资源管理

**协作流程（一个请求从到达到产出的串联）**：

```
请求到达
  │
  ├─(1) 准入控制 should_admit()      过载时？gold 放行 / 低优拒绝(503)  ← 入口闸门
  │        通过 ↓
  ├─(2) 租户隔离 can_schedule()      该租户 running 未满 64 才放进      ← 入口名额
  │        通过 ↓
  ├─ 进入 waiting 队列，每步调度时排序：
  │     (3) QoS compute_effective_priority()  算主排序键
  │     (4) MLFQ mlfq_level                    同优先级时的次排序键
  │     (5) Deadline 紧急请求插队 + 抢占时先踢已违约的
  │        → 按 __lt__(QoS→MLFQ→到达时间) 出队，选中进入 running
  │        ↓
  ├─(6) running 内：update_rate_limiters() 按负载+MLFQ档给每个请求设 rate
  │     Token 限速 consume() 决定每请求这步能生成几个 token
  │     （配套：长文档让位，长 continuation 每步最多 1 个）
  │        ↓
  └─ 产出 token → mlfq_account_tokens() 累计消耗 → 可能触发 (4) 降级 → 回到下一步排序
```

> 关键衔接点：**MLFQ 的 level 是枢纽**——它既作为 (3)(4) 的 waiting 次排序键，又喂给 (6) Token 限速分档；
> 请求产出 token 后累计消耗回写 MLFQ，形成"跑得越多→降级→排队靠后+被限速"的闭环。

#### QoS 分级调度（多维有效优先级）

```python
# —— 主流程（调度→forward→采样→更新完整链路，★ 标本优化调用点）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    # B 调度：
    for req in waiting:
        req.effective_priority = compute_effective_priority(req, now)  # ★ 本优化调用点
    waiting.sort()                                    # ★ 按 QoS 主键出队
    scheduled = pick_requests(waiting, running, token_budget)
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样
    update_states(running, sampled)                    # E 状态更新
    return outputs

# —— 核心函数 ——
def compute_effective_priority(req, now):
    # 值越小 = 越优先。三部分叠加
    priority = req.base_priority                      # gold=-2 / silver=0 / bronze=2
    # 长度调整：短 prompt 加优先级，长 prompt 降优先级
    if req.num_prompt_tokens < 512:   priority -= 2   # 短
    elif req.num_prompt_tokens < 2048: priority -= 1  # 中
    else:                              priority += 1   # 长
    # 防饿死：每等待 5 秒，优先级 -1（有上限）
    priority -= min((now - req.arrival_time) // 5, STARVATION_CAP)
    return priority

# waiting 出队顺序：QoS 主键 → MLFQ 次键 → 到达时间兜底
def __lt__(self, other):
    if self.effective_priority != other.effective_priority:
        return self.effective_priority < other.effective_priority
    if self.mlfq_level != other.mlfq_level:
        return self.mlfq_level < other.mlfq_level
    return self.arrival_time < other.arrival_time
```

#### MLFQ 多级反馈队列（按累计 token 自动降级）

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    scheduled = schedule(waiting, running)             # B 调度（排序用到 mlfq_level 次键）
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样
    # E 状态更新：产出 token 后记账
    for req in running:
        mlfq_account_tokens(req, num_generated[req.id])  # ★ 本优化调用点：累计→可能降级
    return outputs

# —— 核心函数 ——
MLFQ_QUOTA = {0: 128, 1: 512, 2: 2048, 3: INF}   # 每级累计配额

def mlfq_account_tokens(req, num_tokens):
    req.mlfq_tokens_consumed += num_tokens            # 累计，只增不减
    # 累计消耗越过当前级配额 → 降级（长请求自动下沉）
    if req.mlfq_tokens_consumed >= MLFQ_QUOTA[req.mlfq_level]:
        req.mlfq_level = min(req.mlfq_level + 1, 3)

def mlfq_promote(req):                                # 被抢占时补偿升级
    req.mlfq_level = max(req.mlfq_level - 1, 0)        # 注意：token 计数不重置（防刷）
```

#### Token 级速率控制（负载感知令牌桶）

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    # B 调度：给 running 请求分配本步 token 时限速
    update_rate_limiters(running, max_running)        # ★ 本优化调用点：按负载设 rate
    scheduled = []
    for req in running:
        n = req.rate_limiter.consume(min(remaining_need(req), token_budget))  # ★ 限每步 token
        if n > 0:
            scheduled.append((req, n))
    hidden = model.forward(gather_tokens(scheduled))  # C forward（只算被放行的 token）
    sampled = sample(hidden)                           # D 采样
    update_states(running, sampled)                    # E 状态更新
    return outputs

# —— 核心函数 ——
def update_rate_limiters(running, max_running):
    load = len(running) / max(1, max_running)         # 系统负载 = running 占用率
    for req in running:
        tier = get_priority_tier(req)                 # 优先看 mlfq_level，其次 qos
        if load < 0.5 or tier == "HIGH":
            rate = INF                                # 轻载或高优：不限速
        elif load < 0.8:
            rate = 64 if tier == "NORMAL" else 16
        else:                                         # 高载：限得最狠
            rate = 32 if tier == "NORMAL" else 8
        req.rate_limiter.set_rate(rate)

# 调度时：桶里有多少令牌，这步最多生成多少 token
def consume(bucket, requested):
    bucket.tokens = min(bucket.tokens + bucket.rate, bucket.burst)  # refill
    allowed = min(requested, int(bucket.tokens))
    bucket.tokens -= allowed
    return allowed                                    # 0 = 这步不参与 forward batch
```

#### 请求准入控制（过载闸门）

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    # A 入口：新请求进 waiting 前先过闸门
    for req in newly_arrived:
        if should_admit(req, waiting, violation_window):   # ★ 本优化调用点
            waiting.add(req)
        else:
            reject(req, code=503)                          # 过载拒绝低优
    scheduled = schedule(waiting, running)             # B 调度
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样
    # E 状态更新：完成的请求把违约情况填进窗口，供准入控制算违约率
    for req in finished(running, sampled):
        violation_window.append(req.is_sla_violated(now))
    return outputs

# —— 核心函数 ——
def should_admit(req, waiting, violation_window):
    if req.priority < 0:                              # gold 永远放行
        return True
    # 门1：队列深度
    if len(waiting) >= 40:                            # max_queue_depth
        return False                                  # 拒绝低优（503）
    # 门2：近期 SLA 违约率
    violation_rate = sum(violation_window) / max(1, len(violation_window))
    if violation_rate > 0.25:                         # overload_violation_threshold
        return False
    return True
```

#### Deadline-aware 调度（EDF + SLA 感知抢占）

```python
# —— 主流程（★ 标本优化调用点，两处）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    # B 调度：
    waiting.sort(key=lambda r: r.deadline)            # ★ 本优化调用点(a)：EDF 紧急插队
    for req in waiting_or_running_needing_kv:
        if not allocate_slots(req):                   # KV 不够 → 抢占
            victim = select_preemption_victim(running)  # ★ 本优化调用点(b)：先踢已违约
            free_partial(victim)
    scheduled = build_output(...)
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样
    update_states(running, sampled)                    # E 状态更新
    return outputs

# —— 核心函数 ——
def is_sla_violated(req, now):
    return (req.deadline - now) <= 0                  # deadline = arrival + sla_ttft_ms

def select_preemption_victim(running):
    # 优先抢占"已违约"的（反正救不回，牺牲它不额外亏 SLA）
    violated = [r for r in running if r.is_sla_violated(now)]
    pool = violated if violated else running
    # 其次抢 effective_priority 最低（最不重要）的
    return max(pool, key=lambda r: (r.effective_priority, r.arrival_time))
```

#### WFQ 加权公平 + 租户隔离

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    # B 调度：从 waiting 挑请求进 running 时加一道租户名额检查
    for req in waiting_sorted:
        if not can_schedule(req.tenant_id, tenant_running):  # ★ 本优化调用点：租户名额
            continue                                          # 该租户满了→跳过，留 waiting
        tenant_running[req.tenant_id] += 1
        running.add(req)
    hidden = model.forward(gather_tokens(running))    # C forward
    sampled = sample(hidden)                           # D 采样
    for req in finished(running):                      # E 状态更新
        tenant_running[req.tenant_id] -= 1
    return outputs

# —— 核心函数 ——
def can_schedule(tenant_id, tenant_running):
    return tenant_running[tenant_id] < 64             # 每租户 running 并发上限（默认64）

def scheduling_weight(tenant_id, base_weight, running_count):
    return base_weight / max(1, running_count)        # 用得越多权重越低 → 倾斜给欠服务租户
```

#### 长文档 Prefill 让位

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    # B 调度：遍历 running 时对长文档 continuation 限流
    state.long_continuation_count = 0                 # 每步重置
    scheduled = []
    for req in running:
        if long_prefill_guard(req, state) == SKIP:    # ★ 本优化调用点：长文档让位
            continue                                  # budget 留给短请求
        scheduled.append(req)
    hidden = model.forward(gather_tokens(scheduled))  # C forward
    sampled = sample(hidden)                           # D 采样
    update_states(running, sampled)                    # E 状态更新
    return outputs

# —— 核心函数 ——
def long_prefill_guard(chunk, state):
    is_long = chunk.request.num_prompt_tokens > 1024              # 长文档判定
    if is_long and state.long_continuation_count >= 1:           # 每步最多 1 个长 continuation
        return SKIP                                              # 让位，把 budget 留给短请求
    if is_long:
        state.long_continuation_count += 1
    return SCHEDULE
```

### 5.2 Prefix Cache 调度优化

**协作流程（缓存的用、留、驱三个环节）**：

```
请求进调度（在 §5.1 的 waiting 排序内）
  │
  ├─(1) 缓存感知调度 cache_aware_select_next()
  │        同 MLFQ 层级内，优先选"实际 prefill token 最少"（命中最多）的请求
  │        ↓
  ├─ 命中的 block 靠 touch() 复用（ref_cnt+1，见 §5.1 的分页机制）
  │        并被 (2) 标记 _promoted → 释放时进保护区
  │        ↓
  ├─(3) 抢占发生时（缺 KV block，见 §5.1 Deadline 抢占）
  │        free_partial() 只释放尾部、保留有 hash 的前缀 → 恢复时少重算
  │        ↓
  └─(2) 需要驱逐时 Segmented LRU：优先驱逐 probation 区
           高频前缀在 protected 区，不易被低频长请求挤出（抗缓存震荡）
```

> 关键衔接点：三者都围绕 §5.1 的 **block + ref_cnt 分页模型**协作——(1) 决定"优先服务谁的缓存"，
> (2) 决定"缓存满了先驱逐谁"，(3) 决定"抢占时保留哪些缓存"，共同放大原生 Prefix Caching 的复用率。

#### 缓存感知调度（同层内优先调度命中多的）

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    # B 调度：从 waiting 挑请求进 running 时，优先挑命中缓存多的
    while token_budget > 0 and waiting:
        req = cache_aware_select_next(mlfq_queues, scan_window=8)  # ★ 本优化调用点
        computed = get_computed_blocks(req)           # 命中前缀→touch 复用
        n = min(req.num_tokens - computed, token_budget)
        allocate_slots(req, n); running.add(req); token_budget -= n
    hidden = model.forward(gather_tokens(running))    # C forward（命中越多→prefill 越少）
    sampled = sample(hidden)                           # D 采样
    update_states(running, sampled)                    # E 状态更新
    return outputs

# —— 核心函数 ——
def cache_aware_select_next(mlfq_queues, scan_window=8):
    for level_queue in mlfq_queues:                   # 高层级仍优先（不破坏 MLFQ 隔离）
        if not level_queue:
            continue
        best = None
        for req in level_queue[:scan_window]:         # 只扫前 K 个，控制开销
            _, num_computed = get_computed_blocks(req)
            actual_prefill = req.num_tokens - num_computed  # 实际要 prefill 的 token
            if best is None or actual_prefill < best.prefill:
                best = (req, actual_prefill)          # 选实际 prefill 最少（命中最多）的
        return best.req                               # 同 budget 下能多服务请求
    return None
```

#### 频率感知驱逐（Segmented LRU）

```python
# —— 主流程（★ 标本优化调用点，三个时机）——
def engine_step():
    scheduled = schedule(...)                          # B 调度：命中 block 时 touch() ★复用
    for b in hit_blocks: touch(b)                      # ★ 时机1：cache hit 复用
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样
    for req in finished(running, sampled):             # E 状态更新
        for b in req.blocks: free_block(b)             # ★ 时机2：请求完成释放进分区
    # 分配新 block 缺空闲时：popleft() 优先驱逐试用区    # ★ 时机3：驱逐
    return outputs

# —— 核心函数 ——
# free queue 分两区：probation（试用，先驱逐）/ protected（保护，后驱逐）
def touch(block):                                     # cache hit（从 free queue 抢救）
    if block.ref_cnt == 0:
        block._promoted = True                        # 标记：下次释放进 protected
        free_queue.remove(block)
    block.ref_cnt += 1

def free_block(block):
    block.ref_cnt -= 1
    if block.ref_cnt == 0:
        if block._promoted:
            free_queue.append_protected(block)        # 高频 → 保护区（不易被挤出）
            block._promoted = False
        else:
            free_queue.append(block)                  # 普通 → 试用区

def popleft():                                        # 分配新 block 时驱逐
    return pop_probation() if num_probation > 0 else pop_protected()  # 优先驱逐试用区
```

#### 抢占缓存保护（free_partial）

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    admit_new_requests(newly_arrived, waiting)        # A 入口
    # B 调度：KV 不够触发抢占时，用 free_partial 代替普通 free
    for req in running_needing_kv:
        if not allocate_slots(req):
            victim = select_preemption_victim(running)
            preempt(victim, block_size)               # ★ 本优化调用点：只释放尾部、留前缀
    scheduled = build_output(...)
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样（被抢占者恢复时从前缀起步）
    update_states(running, sampled)                    # E 状态更新
    return outputs

# —— 核心函数 ——
def preempt(req, block_size):
    blocks = req_to_blocks[req.id]
    # 保留连续有 hash 的前缀 block（有缓存价值），只释放尾部
    keep = 0
    for b in blocks:
        if b.block_hash is not None: keep += 1
        else: break                                   # hash chain 断裂即停
    would_free = len(blocks) - keep
    if keep > 0 and would_free >= 1:
        free_partial(req, keep_prefix_blocks=keep)    # 部分释放：保留前缀
        req.num_computed_tokens = keep * block_size   # 恢复时从前缀起步（不从 0）
    else:
        free(req)                                     # 降级：释放量不足则全量释放（防空转）
        req.num_computed_tokens = 0
```

### 5.3 后缀解码

**协作流程（一个 decode step 里 draft → 验证的串联）**：

```
每个 decode step：
  │
  ├─(1) 维护 SAM：把上一步新生成的 token 增量 extend() 进后缀自动机（O(1)，不重建）
  │        ↓
  ├─(2) 匹配：suffix_propose() 用后缀数组/SAM 在历史里找当前后缀的匹配
  │        找到多个候选 ↓
  ├─(3) 评分：adaptive_propose() 四因子加权，选期望收益最高的候选作为 draft
  │        ↓
  ├─ draft 的 k 个 token 一次性喂给 Target Model forward 验证（复用 vLLM 投机验证路径）
  │        ↓
  ├─ rejection_sample 接受最长合法前缀 → 接受了 m 个 token（一步产出 1+m 个）
  │        ↓
  └─ 把接受结果回写 (3) 的接受率 tracker（反馈闭环）+ 新 token 回到 (1) 更新 SAM
```

> 关键衔接点：(1) 增量 SAM 是**性能基础**（避免每步 O(n) 重建），(2)(3) 是**质量核心**（找最优 draft）；
> 验证结果通过"接受率 tracker"反哺 (3) 的评分，形成"投得准→接受率高→更多采纳该类 draft"的闭环。
> 与调度的联动：draft token 也会进入 forward batch 占算力，因此高负载下应配合调度侧的负载信号收缩投机（见进阶优化 7）。

#### SuffixTreeProposer（后缀数组 + 自适应回退）

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    scheduled = schedule(waiting, running)             # B 调度
    # C forward：把上一轮 draft 和正常 token 一起前向，一次验证
    hidden = model.forward(scheduled.token_ids + prev_drafts)
    sampled = sample(hidden)                           # D 采样
    accepted = rejection_sample(prev_drafts, sampled)  # 验证 draft
    # D'：为每个请求生成下一轮 draft
    for req in running:
        suffix_sam[req.id].extend(sampled[req.id])     # [增量SAM] 追加 token
        cands = suffix_propose(req.context, MIN_N, MAX_N, K)   # ★ 本优化调用点：后缀匹配
        req.draft = adaptive_propose(req.context, cands, tracker)  # [评分] 选最优
    update_states(running, sampled, accepted)          # E 状态更新
    return outputs

# —— 核心函数 ——
def suffix_propose(context, min_n, max_n, k):
    # 从长到短尝试匹配后缀（自适应回退）
    for n in range(max_n, min_n - 1, -1):
        pattern = context[-n:]
        pos = suffix_array_search(context, pattern)   # 后缀数组 + 二分，O(pattern_len)
        if pos is not None:
            return context[pos + n : pos + n + k]     # 匹配后的 k 个 token 作为 draft
    return []                                         # 没匹配到 → 不投机
```

#### 增量后缀自动机（SAM，O(1) 均摊追加）

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    scheduled = schedule(waiting, running)             # B 调度
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样
    # D'：把新生成的 token 增量喂进各请求的 SAM（不重建），再匹配起草
    for req in running:
        suffix_sam[req.id].extend(sampled[req.id])     # ★ 本优化调用点：O(1) 追加
        req.draft = suffix_propose(req.context, ...)   # 基于更新后的 SAM 匹配
    update_states(running, sampled)                    # E 状态更新
    return outputs

# —— 核心函数 ——
class IncrementalSuffixAutomaton:
    def extend(self, c):                              # 每新增 1 个 token，O(1) 均摊更新
        cur = new_state(len=self.last.len + 1)
        p = self.last
        while p and c not in p.next:
            p.next[c] = cur; p = p.link
        if p is None:
            cur.link = self.root
        else:
            q = p.next[c]
            if p.len + 1 == q.len:
                cur.link = q
            else:                                     # clone 分裂（SAM 经典逻辑）
                clone = copy(q); clone.len = p.len + 1
                while p and p.next[c] == q:
                    p.next[c] = clone; p = p.link
                q.link = cur.link = clone
        self.last = cur
    # 跨步复用：只在末尾追加新生成的 token，不重建整棵结构
```

#### 自适应匹配 + 多候选评分

```python
# —— 主流程（★ 标本优化调用点）——
def engine_step():
    scheduled = schedule(waiting, running)             # B 调度
    hidden = model.forward(scheduled.token_ids)       # C forward
    sampled = sample(hidden)                           # D 采样
    accepted = rejection_sample(prev_drafts, sampled)  # 验证上一轮 draft
    # D'：多候选评分选最优 draft
    for req in running:
        cands = suffix_propose(req.context, ...)       # 后缀匹配的多个候选
        req.draft = adaptive_propose(req.context, cands, tracker)  # ★ 本优化调用点
    tracker.update(accepted)                           # E：验证结果回写接受率（反馈闭环）
    return outputs

# —— 核心函数 ——
def score_candidate(cand, recent_accept_rate):
    # 四因子加权：从"首次匹配"升级为"最优匹配"
    return (0.25 * cand.match_len          # 匹配长度
          + 0.20 * cand.continuation_len   # 可续接长度
          + 0.25 * cand.recency            # 位置新旧（越近越可能相关）
          + 0.30 * recent_accept_rate)     # 历史接受率（反馈闭环）

def adaptive_propose(context, candidates, tracker):
    best = max(candidates, key=lambda c: score_candidate(c, tracker.rate(c)))
    return best.tokens                                # 选期望收益最高的，而非第一个匹配
```

> 以上均为**已实现**优化点的核心逻辑。进阶优化点（后缀解码 6/7/8、调度 9/10、前缀缓存预热/可观测性）
> 为方案设计阶段，其伪代码见各自设计文档，实现后再补入本节。

---

## 六、Insight：这三类优化的本质与在业界优化体系中的位置

### 6.1 三类优化的本质归类

把三大类优化抽象后，会发现它们**本质上都不碰模型本身（不改权重、不改精度、不动模型结构），而是优化"请求/资源/计算的组织方式"**——属于 **serving 层（推理服务层）的系统优化**，而非模型层优化。

| 优化大类 | 本质类型 | 一句话本质 | 优化的"资源" |
|---------|---------|-----------|-------------|
| 调度与资源管理 | **资源调度与仲裁（Resource Scheduling）** | 在请求间**分配有限的算力/显存/时间** | GPU 算力、token_budget、running 名额 |
| Prefix Cache 复用 | **计算复用与缓存管理（Compute Reuse / Caching）** | **避免重复计算**已算过的前缀 KV | 已计算的 KV Cache（时间换空间的复用） |
| 后缀解码 | **计算模式变换（并行化 / Compute Reshaping）** | 把**串行 decode 变成并行验证**，用便宜的猜测换 GPU 并行度 | GPU 的并行吞吐能力（访存 → 算力的再平衡） |

**三者共同的本质**——都是围绕一个核心矛盾做文章：

> **GPU 算力/显存有限，而请求是动态、异构、有优先级的。**
> 三类优化分别从**"怎么分配"（调度）、"怎么少算"（缓存复用）、"怎么算得更并行"（投机）** 三个角度，
> 在**不改变模型和输出正确性**的前提下，压榨同一份硬件的有效利用率。

### 6.2 在业界推理优化体系中的位置

业界 LLM 推理优化大致分四层，从底到上：

```
┌─────────────────────────────────────────────────────────┐
│ L4  服务/调度层  Serving & Scheduling                     │  ← 本项目三类优化都在这里
│     Continuous Batching / 调度策略 / Prefix Cache /        │     （最贴近"多请求、生产部署"）
│     投机解码 / PD 分离 / 多租户 QoS                         │
├─────────────────────────────────────────────────────────┤
│ L3  显存/KV 管理层  Memory Management                      │  ← Prefix Cache 触及此层
│     PagedAttention / KV Cache 分页、量化、offload          │
├─────────────────────────────────────────────────────────┤
│ L2  计算/Kernel 层  Compute & Kernels                     │
│     FlashAttention / 融合 kernel / CUDA Graph              │
├─────────────────────────────────────────────────────────┤
│ L1  模型/算法层  Model & Algorithm                        │
│     量化(GPTQ/AWQ/FP8) / MoE / MLA / 蒸馏 / 模型结构        │
└─────────────────────────────────────────────────────────┘
```

**本项目的三类优化全部位于最上层 L4（服务/调度层）**，特点是：

- **通用、正交、可叠加**：不依赖特定模型/硬件，可与下层（量化、FlashAttention、PagedAttention）**任意组合叠加**收益——这也是它们工程价值高的原因。
- **无损**：不改变模型输出的正确性（投机解码用拒绝采样保证无损、缓存复用是精确复用、调度只改顺序），区别于 L1 量化那种"用精度换速度"的有损优化。
- **面向"多请求、真实负载、生产部署"**：L1-L3 大多优化"单次前向多快"，而 L4 优化的是"**一堆并发请求整体服务得多好**"（吞吐、尾延迟、公平性、SLA）——越靠近真实线上服务，L4 越关键。

### 6.3 三类优化各自的差异化定位（L4 内部再细分）

即使同在 L4，三类优化针对的性能目标也不同：

| 优化 | 主要改善的指标 | 生效前提（边界） | 与业界对标 |
|------|--------------|----------------|-----------|
| 调度与资源管理 | **公平性、尾延迟、SLA 达成率**（不主要提吞吐） | 有**负载压力 + 优先级差异**才有意义；救不了物理算力不足 | 借鉴网络/OS 调度（QoS/WFQ/MLFQ/EDF），迁移到推理 |
| Prefix Cache 复用 | **TTFT、Prefill 成本** | 有**前缀共享**（System Prompt/多轮对话/RAG）才有效 | RadixAttention / vLLM 官方 prefix caching 的调度层增强 |
| 后缀解码 | **TPOT、Decode 吞吐** | 有**重复/结构化模式**（对话/代码/Agentic）才有效 | SuffixDecoding / SGLang 后缀投机 |

> **一个统一视角**：三类优化恰好覆盖了一个请求延迟的两大组成——**TTFT（首字延迟）由 Prefix Cache + 调度优化改善，TPOT（生成延迟）由后缀解码改善**，而调度层再从全局把它们组织起来、按优先级分配。三者合起来是一套**面向多租户在线服务的、无损的、服务层组合优化**。

### 6.4 一句话总结

> 这三类优化本质上都是 **serving 层（L4）的系统级、无损、正交优化**——不碰模型本身，而是通过
> **更聪明地"调度分配、复用计算、并行猜测"** 来压榨同一份 GPU 的有效利用率。
> 它们在业界推理优化体系中处于**最贴近生产部署的顶层**：不追求"单次前向更快"（那是 L1-L3 的事），
> 而是解决"**大量并发、异构、有优先级的真实请求，如何被整体服务得又快又公平又不违约**"——
> 这正是 LLM 从"能跑"走向"规模化在线服务"时最关键、也最能体现工程系统能力的一层。
