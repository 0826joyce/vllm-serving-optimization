# 基于 vLLM V1 的 Prefix Cache 感知调度与 KV Cache 复用优化

> 围绕 vLLM V1 的 Prefix Caching 机制做深度优化，提升缓存命中率、降低 TTFT、减少 KV Cache 的冗余计算与显存浪费

## 一、项目背景与动机

### 1.1 问题引入

vLLM V1 的 Prefix Caching 机制本身已经较为完善——缓存写入是同步的，同一调度步内多个请求可以自然共享物理 block，不存在"批次内无法共享"的问题。但在**调度策略**和**缓存管理策略**层面，仍有明显的优化空间：

> **调度器不感知缓存命中率，导致调度顺序不佳、token_budget 利用率低；LRU 驱逐策略不考虑访问频率，导致高频前缀被误驱逐；抢占机制过于激进，恢复代价高；冷启动无预热机制。**

### 1.2 vLLM V1 Prefix Caching 机制梳理（基于源码分析）

#### 1.2.1 核心流程

每个 WAITING 请求在调度时经历以下流程：

```
get_computed_blocks(request)
  → hash_request_tokens() 计算 block hash（链式：parent_hash → child_hash）
  → 逐个查找 cached_block_hash_to_block → 遇到 miss 即 break
  → 返回 computed_blocks + num_computed_tokens

allocate_slots(request, num_new_tokens, computed_blocks)
  → _touch(computed_blocks)            // 将命中的 block 从 free queue 移出，ref_cnt++
  → _get_new_blocks(num_new_blocks)    // 分配新 block（可能触发驱逐）
  → _cache_full_blocks(...)            // 同步将新的 full block 的 hash 写入缓存
```

`_cache_full_blocks()` 在 `allocate_slots()` 内部**同步执行**，因此同一调度步内，先调度的请求 A 写入的缓存，后调度的请求 B 可以立即查到并共享 A 的物理 block（通过 `_touch()` 增加 `ref_cnt`）。

#### 1.2.2 不去重设计

源码注释（`kv_cache_manager.py`）指出当前不做 block 去重：当一个 block 被 `_cache_full_blocks()` 写入时，如果 hash 已存在，不会合并物理 block。但 `_get_cached_block()` 总是返回第一个匹配的 block，所以后续请求仍然共享同一个物理 block，不会产生重复分配。

#### 1.2.3 同步骤内的缓存共享示例

```
缓存为空时，同一步内调度请求 A 和 B（相同 System Prompt）：

请求 A (System Prompt 200 tokens + User 2 tokens)：
  → get_computed_blocks(A) → 缓存为空 → 命中 0 blocks
  → allocate_slots(A, 202) → 分配 13 blocks，_cache_full_blocks() 注册 12 个 full block hash

请求 B (System Prompt 200 tokens + User 3 tokens)：
  → get_computed_blocks(B) → 命中 A 注册的 12 blocks ✅
  → allocate_slots(B, 3) → _touch() 共享 A 的 12 blocks，仅分配 1 个新 block
  → B 只需 prefill 3 tokens（而非 203 tokens）
```

GPU 侧，A 的全量 prefill 和 B 的增量 prefill 在同一个 forward pass 中执行，Attention 内核能正确处理共享 block 的并行填充与读取。

### 1.3 需要优化的问题

| 现状问题 | 具体表现 | 影响 |
|---------|---------|------|
| **调度器不感知缓存命中率** | 调度器按 MLFQ/FCFS 排序，不考虑请求的缓存命中率 | 高命中率的请求不会被优先调度，低效消耗 token_budget |
| **缓存驱逐策略单一** | 仅基于 LRU 驱逐，不考虑 block 的复用概率 | 高频共享前缀（如 System Prompt）可能被低频长请求挤出 |
| **被抢占请求恢复代价高** | `num_computed_tokens = 0`，blocks 全部释放 | 恢复时若缓存被驱逐则需全量 Recompute |
| **无缓存预热机制** | 冷启动时缓存为空 | 冷启动阶段 TTFT 极高 |
| **缓存可观测性不足** | 仅有基础命中率指标 | 无法量化优化效果、难以发现缓存异常 |

### 1.4 优化主题定位

本项目聚焦于 **Prefix Cache 与调度器的协同优化**，通过 5 个递进的优化点，形成一套完整的 KV Cache 复用优化方案。

```
优化目标：

  原始 vLLM V1:
  ┌─────────────────────────────────────────┐
  │  请求调度（MLFQ/FCFS）──→ Prefill ──→ Decode │
  │       ↑                     ↑               │
  │   不感知缓存            缓存命中靠运气         │
  └─────────────────────────────────────────┘

  优化后：
  ┌───────────────────────────────────────────────────────┐
  │  缓存感知调度 ──→ 智能驱逐保护 ──→ 高效 Prefill         │
  │       ↑               ↑               ↑               │
  │  高命中优先调度    频率感知驱逐     增量计算             │
  │  + 缓存预热       + 抢占保护      + 缓存效率可观测       │
  └───────────────────────────────────────────────────────┘
```

---

## 二、优化点详细设计

### 优化点 1：Cache-Aware Scheduling（缓存感知调度）`[核心]` ✅ 已实现

#### 问题分析

当前调度器在选取 WAITING 请求时，调用 `get_computed_blocks()` 获取缓存命中信息，但这个信息**仅用于减少 prefill token 数，不影响调度顺序**。

```python
# 当前代码 (scheduler.py, WAITING 调度循环)
# 1. 按 MLFQ 层级 → 层内 FCFS 顺序取出请求
request = self._mlfq_peek_next()
# 2. 查缓存（但不影响调度顺序）
computed_blocks, num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
# 3. 计算需要 prefill 的 token 数
num_new_tokens = request.num_tokens - num_computed_tokens
```

**问题**：假设 MLFQ L0 中有两个请求 A 和 B：
- A：1000 tokens，缓存命中 0 blocks → 需要 prefill 1000 tokens
- B：1000 tokens，缓存命中 900 tokens → 只需 prefill 100 tokens

如果 A 到达时间早于 B（按 FCFS），调度器会先选 A，消耗大量 token_budget，可能导致 B 在本步无法被调度。但如果先调度 B，只需 100 tokens 就能让 B 进入 running，极大提升系统效率。

**量化分析**：
- 先调度 A：消耗 1000 tokens budget，B 可能延迟 1 步 → B 的 TTFT += 1 step latency
- 先调度 B：消耗 100 tokens budget，A 还剩 900 tokens budget 可用 → 两者都能在本步被调度
- 净收益：相同 token_budget 下多服务了 1 个请求

#### 设计方案

**在 MLFQ 同层级内，引入缓存感知排序**：

```python
def _schedule_waiting_cache_aware(self):
    """缓存感知的 WAITING 请求调度。

    在 MLFQ 同一层级内，不再使用纯 FCFS，而是优先调度
    缓存命中率高（实际 prefill tokens 少）的请求。
    
    实现方式：两阶段选择
    1. 扫描同层级的前 K 个候选请求，获取各自的缓存命中信息
    2. 按 actual_prefill_tokens 升序排列，优先调度"性价比"最高的请求
    """
    SCAN_WINDOW = 8  # 扫描窗口大小，避免对所有 WAITING 请求调用 get_computed_blocks

    for level_queue in self.mlfq_queues:
        if not level_queue:
            continue
        
        # 取本层级前 SCAN_WINDOW 个候选
        candidates = list(itertools.islice(level_queue, SCAN_WINDOW))
        
        # 预查缓存命中情况
        scored = []
        for req in candidates:
            computed_blocks, num_computed_tokens = \
                self.kv_cache_manager.get_computed_blocks(req)
            actual_prefill = req.num_tokens - num_computed_tokens
            scored.append((actual_prefill, req, computed_blocks, num_computed_tokens))
        
        # 按 actual_prefill 升序（缓存命中多的排前面）
        scored.sort(key=lambda x: x[0])
        
        for actual_prefill, req, computed_blocks, num_computed_tokens in scored:
            if token_budget <= 0:
                break
            # ... 正常的 allocate_slots 逻辑 ...
```

**关键设计考量**：
1. **不破坏 MLFQ 的层级隔离**：只在同层级内排序，高层级仍然优先于低层级
2. **扫描窗口限制**：只看前 K 个候选，避免对所有请求调用 `get_computed_blocks()` 的开销
3. **`get_computed_blocks()` 的副作用**：该函数会更新 `req_to_block_hashes`（缓存 hash），但不修改引用计数，所以安全
4. **与 QoS 优先级的兼容**：可以在优先级排序基础上叠加缓存感知因子

#### 性能影响分析

**开销**：
- 每步额外调用 K 次 `get_computed_blocks()`（每次是 O(num_blocks) 的 hash 查找）
- 对 K 个候选排序：O(K log K)

**收益**：
- 同样的 token_budget 下能服务更多请求
- 高缓存命中的请求 TTFT 降低
- 整体 prefill 阶段 GPU 时间减少

#### 修改文件
- `vllm/v1/core/scheduler.py` — WAITING 调度循环中引入缓存感知排序

#### 涉及的 vLLM 知识点
- `get_computed_blocks()` 的链式 hash 查找机制（遇到 miss 即 break）
- `hash_request_tokens()` 的结果缓存在 `req_to_block_hashes` 中（避免重复计算）
- MLFQ 层级内的 FCFS 语义
- `token_budget` 的共享模型

#### 实现记录

**已实现**，修改文件：`vllm/v1/core/scheduler.py`

##### 1. 新增配置项（`__init__` 方法）

```python
# ---- Cache-Aware Scheduling ----
self.enable_cache_aware_scheduling: bool = (
    self.cache_config.enable_prefix_caching)
self.cache_aware_scan_window: int = 8
```

- `enable_cache_aware_scheduling`：当 prefix caching 开启时自动启用，否则降级为原始 FCFS
- `cache_aware_scan_window`：每个 MLFQ 层级内扫描的候选数，默认 8

##### 2. WAITING 调度循环改造

原始路径（纯 FCFS）：
```python
request = self._mlfq_peek_next()
computed_blocks, num_computed_tokens = self.kv_cache_manager.get_computed_blocks(request)
```

改造后（缓存感知）：
```python
if self.enable_cache_aware_scheduling and self.enable_mlfq:
    selection = self._cache_aware_select_next()
    if selection is None:
        break
    (request, computed_blocks, num_computed_tokens) = selection
    cache_info_precomputed = True
elif self.enable_mlfq:
    request = self._mlfq_peek_next()
    cache_info_precomputed = False
else:
    request = self.waiting[0]
    cache_info_precomputed = False

if not cache_info_precomputed:
    computed_blocks, num_computed_tokens = \
        self.kv_cache_manager.get_computed_blocks(request)
```

关键改动：
- 当 cache-aware 启用时，`_cache_aware_select_next()` 已经返回了缓存信息，避免重复调用 `get_computed_blocks()`
- 从队列移除时使用 `mlfq_queues[request.mlfq_level].remove(request)` 精确移除（因为选中的不一定是队首）

##### 3. 新增 `_cache_aware_select_next()` 方法

```python
def _cache_aware_select_next(
    self,
) -> Optional[Tuple[Request, List, int]]:
    """Select the next WAITING request using cache-aware ordering.

    Within the highest non-empty MLFQ level, scan up to
    cache_aware_scan_window candidates and return the one that
    requires the fewest new prefill tokens (i.e. highest cache hit
    rate).
    """
    for level_queue in self.mlfq_queues:
        if not level_queue:
            continue

        k = min(self.cache_aware_scan_window, len(level_queue))
        candidates = list(itertools.islice(level_queue, k))

        best: Optional[Tuple[int, Request, List, int]] = None
        for req in candidates:
            computed_blocks, num_computed_tokens = (
                self.kv_cache_manager.get_computed_blocks(req))
            actual_prefill = req.num_tokens - num_computed_tokens
            if best is None or actual_prefill < best[0]:
                best = (actual_prefill, req, computed_blocks,
                        num_computed_tokens)

        if best is not None:
            _, req, comp_blocks, n_computed = best
            return (req, comp_blocks, n_computed)

    return None
```

**设计保证**：
| 约束 | 保证 |
|------|------|
| 不破坏 MLFQ 层级隔离 | 只在同层级内选择，高层级仍优先 |
| 扫描开销可控 | 只扫描前 K=8 个候选 |
| `get_computed_blocks()` 无副作用 | 只做 hash 查找和缓存，不修改 ref_cnt |
| prefix caching 未启用时降级 | 绑定 `enable_prefix_caching`，未启用走 FCFS |
| 向后兼容 | 非 MLFQ 模式路径完全保留 |

##### 4. 测试文件
- `tests/v1/core/test_cache_aware_scheduling.py` — 缓存感知调度单元测试

---

### 优化点 2：Frequency-Aware Cache Eviction（频率感知缓存驱逐）`[核心]` ✅ 已实现（方案A）

#### 问题分析

当前 vLLM V1 的缓存驱逐策略是**纯 LRU（Least Recently Used）**：

```python
# FreeKVCacheBlockQueue 是一个双向链表：
# - 头部 (popleft) = 最久未使用的 block → 最先被驱逐
# - 尾部 (append)  = 最近释放的 block → 最后被驱逐
#
# 释放时逆序进队列（尾部 block 先进 = LRU 序），分配时从头部取
```

**LRU 的问题——"频率盲"**：

```
场景：System Prompt "你是一个有帮助的 AI 助手..." 的 blocks（hash=0xABC）
  
  t=0: 请求 1 完成 → System Prompt blocks 的 ref_cnt 从 1 降为 0 → 进入 free queue 尾部
  t=1: 一个长上下文请求（8K tokens）被调度 → 分配大量新 blocks
       → 从 free queue 头部不断 popleft → 其他旧 blocks 被驱逐
  t=2: 长请求完成 → 释放 500 个 blocks → 全部进入 free queue 尾部
       → System Prompt blocks 被推到了 free queue 中部偏前位置
  t=3: 又一个长请求 → System Prompt blocks 被驱逐！
  t=4: 新一批使用相同 System Prompt 的请求到达 → 全部 miss → 全量 prefill

  期望行为：System Prompt blocks 每分钟被 100+ 请求访问，应该有更高的"保护权"
```

**根因**：LRU 只看"最近一次使用时间"，不看"历史使用频率"。一个被使用过 1000 次的 System Prompt block 和一个只用过 1 次的普通 block，在 free queue 中的优先级完全取决于谁最后被释放。

#### 设计方案

**方案 A：Segmented LRU（分区 LRU）**—— 推荐，实现简单，效果明确

```
空闲队列分两段：

  ┌──────────────────┬───────────────────────┐
  │  Probation Zone  │   Protected Zone      │
  │  (试用区)         │   (保护区)             │
  │  新释放的 block    │   高频访问的 block      │
  │  ← 优先驱逐       │   ← 最后驱逐           │
  └──────────────────┴───────────────────────┘
  
  规则：
  1. block 第一次释放（ref_cnt → 0）时，进入 Probation Zone 尾部
  2. 如果在 Probation Zone 中被 _touch()（再次命中），升级到 Protected Zone 尾部
  3. 分配新 block 时优先从 Probation Zone 头部取（驱逐）
  4. Probation Zone 为空时，从 Protected Zone 头部取
  5. Protected Zone 的容量上限 = free_blocks 总数 × protected_ratio（如 50%）
  6. Protected Zone 满时，Protected 头部的 block 降级到 Probation 头部
```

**实现核心**：

```python
class FreeKVCacheBlockQueue:
    """改造为 Segmented LRU 的空闲队列"""
    
    def __init__(self, blocks: List[KVCacheBlock], protected_ratio: float = 0.5):
        # 两个双向链表（使用 block 的 prev/next 指针）
        self.probation_head: Optional[KVCacheBlock] = None
        self.probation_tail: Optional[KVCacheBlock] = None
        self.protected_head: Optional[KVCacheBlock] = None
        self.protected_tail: Optional[KVCacheBlock] = None
        
        self.num_probation_blocks: int = 0
        self.num_protected_blocks: int = 0
        self.max_protected_blocks: int = int(len(blocks) * protected_ratio)
        
        # 初始时所有 block 进入 probation zone
        # ... 初始化链表 ...
    
    @property
    def num_free_blocks(self) -> int:
        return self.num_probation_blocks + self.num_protected_blocks
    
    def popleft(self) -> KVCacheBlock:
        """优先从 probation zone 驱逐"""
        if self.num_probation_blocks > 0:
            return self._pop_from_probation()
        else:
            return self._pop_from_protected()
    
    def append(self, block: KVCacheBlock) -> None:
        """释放时进入 probation zone"""
        self._append_to_probation(block)
    
    def promote(self, block: KVCacheBlock) -> None:
        """从 probation 升级到 protected（被 _touch 时调用）"""
        self._remove_from_probation(block)
        if self.num_protected_blocks >= self.max_protected_blocks:
            # protected 满了，将最旧的降级
            demoted = self._pop_from_protected()
            self._prepend_to_probation(demoted)
        self._append_to_protected(block)
```

**与 `_touch()` 的集成**：

```python
def _touch(self, blocks: List[KVCacheBlock]) -> None:
    for block in blocks:
        if block.ref_cnt == 0:
            # block 在 free queue 中被再次命中
            if block in probation_zone:  # 通过标记位判断
                self.free_block_queue.promote(block)  # 升级到 protected
            else:
                self.free_block_queue.remove(block)  # 已在 protected，直接移除
        block.incr_ref()
```

**方案 B：加权驱逐（Weighted Eviction）**—— 备选

```python
@dataclass
class KVCacheBlock:
    # 新增字段
    access_count: int = 0  # 历史被 _touch 的总次数
    
class FreeKVCacheBlockQueue:
    def popleft_weighted(self) -> KVCacheBlock:
        """扫描前 N 个候选，驱逐 access_count 最低的"""
        SCAN_WINDOW = 8
        candidates = self._get_head_n(SCAN_WINDOW)
        victim = min(candidates, key=lambda b: b.access_count)
        self.remove(victim)
        return victim
```

方案 B 更简单但打破了 O(1) 的 popleft 性能保证。

#### 修改文件
- `vllm/v1/core/kv_cache_utils.py` — `FreeKVCacheBlockQueue` 改造为分区 LRU，`KVCacheBlock` 新增 zone 标记
- `vllm/v1/core/kv_cache_manager.py` — `_touch()` 中调用 promote，驱逐时使用新策略

#### 预期效果
- System Prompt 等高频前缀的缓存命中率从 ~50% 提升到 ~90%
- "缓存震荡"（cache thrashing）大幅减少——高频 block 不再被低频长请求挤出
- 关键指标：缓存命中率方差降低（更稳定）

#### 风险评估
- **复杂度增加**：双链表变为两个双链表，但操作仍为 O(1)
- **protected_ratio 参数敏感性**：需要根据工作负载调优
- **内存开销**：每个 block 增加 1 个标记位，可忽略

#### 实现记录

**已实现（方案A：Segmented LRU）**，修改文件：
- `vllm/v1/core/kv_cache_utils.py` — `FreeKVCacheBlockQueue` 改造为分区 LRU，`KVCacheBlock` 新增 zone 标记
- `vllm/v1/core/kv_cache_manager.py` — `_touch()` 集成 promote 标记，`free()` 根据标记选择 zone

##### 1. `KVCacheBlock` 新增字段

```python
# ---- Segmented LRU zone tracking ----
free_zone: Optional[str] = None    # "probation" / "protected" / None
_promoted: bool = False            # 被 _touch() 命中时设为 True
```

- `free_zone`：标记 block 当前所属的 free queue zone
- `_promoted`：当 block 在 free queue 中被 cache hit（`_touch()`），标记为 True。后续释放时进入 protected zone

##### 2. `FreeKVCacheBlockQueue` 改造为 Segmented LRU

```
空闲队列分两段：

  ┌──────────────────┬───────────────────────┐
  │  Probation Zone  │   Protected Zone      │
  │  (试用区)         │   (保护区)             │
  │  新释放的 block    │   高频访问的 block      │
  │  ← 优先驱逐       │   ← 最后驱逐           │
  └──────────────────┴───────────────────────┘
```

核心 API：
| 方法 | 行为 |
|------|------|
| `popleft()` | 优先从 probation 头部驱逐，probation 为空时从 protected 头部驱逐 |
| `append(block)` | 将新释放的 block 放入 probation 尾部 |
| `append_protected(block)` | 将被 promote 过的 block 放入 protected 尾部，protected 满时降级最旧的 protected block 到 probation 头部 |
| `remove(block)` | 从所在 zone 移除（O(1)，基于 `free_zone` 标记） |
| `promote(block)` | 将 probation block 移到 protected（promote 方法本身可用于直接 zone 间迁移） |

初始化时所有 block 进入 probation zone，`protected_ratio` 默认 0.5（50% 总 block 数作为 protected 上限）。

##### 3. `_touch()` 改造

```python
def _touch(self, blocks):
    for block in blocks:
        if block.ref_cnt == 0:
            # 标记为已 promote，下次 free 时进入 protected zone
            block._promoted = True
            self.free_block_queue.remove(block)
        block.incr_ref()
```

关键语义：当一个 free block（eviction candidate）被 cache hit 时，设置 `_promoted = True`。这个标记跟随 block 的整个生命周期，直到它再次被释放。

##### 4. `free()` 改造

```python
def free(self, request):
    # ...
    for block in ordered_blocks:
        block.decr_ref()
        if block.ref_cnt == 0:
            if block._promoted:
                # 被 cache-hit 过的 block → protected zone
                self.free_block_queue.append_protected(block)
                block._promoted = False
            else:
                # 普通 block → probation zone
                self.free_block_queue.append(block)
```

##### 5. 设计保证

| 约束 | 保证 |
|------|------|
| 所有操作 O(1) | 双链表操作 + zone 标记，无扫描 |
| System Prompt 保护 | 高频 block 被 touch → 进 protected → 最后驱逐 |
| Protected 满时降级 | 最旧 protected block 降级到 probation 头部 |
| 向后兼容 | probation zone 行为与原 LRU 完全一致 |
| 无运行时开销 | `_promoted` 仅在 `_touch` 和 `free` 时读写 |
| 可配置 | `protected_ratio` 参数控制保护区大小比例 |

##### 6. 测试文件
- `tests/v1/core/test_frequency_aware_eviction.py` — Segmented LRU 单元测试

---

### 优化点 3：Preemption Cache Shield（抢占缓存保护）`[重要]` ✅ 已实现

#### 问题分析

当前 vLLM V1 的抢占逻辑非常激进：

```python
# scheduler.py 第 232-234 行
self.kv_cache_manager.free(preempted_req)       # 释放所有 blocks
preempted_req.status = RequestStatus.PREEMPTED
preempted_req.num_computed_tokens = 0            # 重置为 0！
```

`free()` 的实现（`kv_cache_manager.py` 第 270-291 行）：
```python
def free(self, request: Request) -> None:
    blocks = self.req_to_blocks.pop(request.request_id, [])
    # 逆序释放
    for block in reversed(blocks):
        block.decr_ref()
        if block.ref_cnt == 0:
            self.free_block_queue.append(block)
    self.num_cached_block.pop(request.request_id, None)
```

**三重损失**：
1. **计算损失**：`num_computed_tokens = 0` 意味着恢复时从头开始，即使前缀可能仍在缓存中
2. **缓存暴露**：释放后 block 的 `ref_cnt` 可能降为 0，成为驱逐候选
3. **连锁反应**：如果前缀 block 被驱逐，不仅影响被抢占请求，还影响**所有共享该前缀的其他请求**

**但有一个缓解因素**：`free()` 只是 `decr_ref()`，不清除 hash。如果 block 的 `ref_cnt` 在释放后仍 > 0（被其他请求共享），则不会进入 free queue。即使降为 0，hash 仍保留，恢复时 `get_computed_blocks()` 仍可能命中。

**真正的问题**：`num_computed_tokens = 0` 的重置过于保守。实际上可以设置为一个更合理的值——至少是**有 hash 的 block 数量 × block_size**，让 `get_computed_blocks()` 在恢复时有更好的起点。

#### 设计方案

**核心思想**：抢占时不改变 `free()` 行为（保持简单），但优化**恢复逻辑**。

```python
# 改造后的抢占逻辑（scheduler.py）

# 方案 1：保留 block hashes，不重置 num_computed_tokens 为 0
preempted_req = self.running.pop()  # 选最低优先级的
self.kv_cache_manager.free(preempted_req)
preempted_req.status = RequestStatus.PREEMPTED
# 不设置 num_computed_tokens = 0
# 而是保留当前值，恢复时 get_computed_blocks() 会重新校准
# 但需要注意：num_computed_tokens 在恢复时会被 scheduler 重新设置
```

**更安全的方案 2：部分释放 + 降级保障**

```python
def free_partial(self, request: Request, keep_prefix_blocks: int) -> int:
    """部分释放：保留前 keep_prefix_blocks 个 blocks，仅释放尾部。

    被保留的 block 维持 ref_cnt > 0，不会成为驱逐候选。
    
    Args:
        request: 被抢占的请求
        keep_prefix_blocks: 保留的前缀 blocks 数
    Returns:
        实际释放的 blocks 数
    """
    blocks = self.req_to_blocks[request.request_id]
    
    # 保留的 blocks
    kept_blocks = blocks[:keep_prefix_blocks]
    freed_blocks = blocks[keep_prefix_blocks:]
    
    # 仅释放尾部 blocks
    for block in reversed(freed_blocks):
        block.decr_ref()
        if block.ref_cnt == 0:
            self.free_block_queue.append(block)
    
    # 更新请求的 block 列表（保留前缀部分）
    self.req_to_blocks[request.request_id] = kept_blocks
    self.num_cached_block[request.request_id] = min(
        self.num_cached_block.get(request.request_id, 0),
        keep_prefix_blocks
    )
    
    return len(freed_blocks)
```

**调度器集成（含降级保障）**：

```python
_PREEMPT_MIN_FREE_BLOCKS = 1  # partial free 至少要释放的 block 数

# 计算应保留的前缀 block 数
blocks = self.kv_cache_manager.req_to_blocks[preempted_req.request_id]
keep_count = 0
for b in blocks:
    if b.block_hash is not None:
        keep_count += 1
    else:
        break  # hash chain 断裂，后续没有意义

would_free = len(blocks) - keep_count
if keep_count > 0 and would_free >= _PREEMPT_MIN_FREE_BLOCKS:
    # 尾部有足够的 blocks 可释放 → 部分释放
    self.kv_cache_manager.free_partial(preempted_req, keep_count)
    preempted_req.num_computed_tokens = keep_count * self.block_size
else:
    # 全部 blocks 都有 hash（或释放量太少）→ 降级为全量释放
    # 保证抢占循环能推进，不会空转
    self.kv_cache_manager.free(preempted_req)
    preempted_req.num_computed_tokens = 0
```

**为什么需要降级？** 如果被抢占请求的 blocks 几乎全部有 hash（比如刚 prefill 完还没生成几个 token），`free_partial` 只能释放 0~1 个 block。此时抢占循环会反复抢占多个请求却释放不出足够空间，导致"空转"。降级为全量释放可以保证每次抢占都能回收有效空间。

#### 可行性分析

**风险**：
- `free_partial` 后 `req_to_blocks` 仍包含被保留的 blocks，但请求已不在 running 中
- 恢复时 `allocate_slots()` 需要正确处理已有的 `req_to_blocks`
- 现有代码的 `allocate_slots()` 已经通过 `len(req_blocks)` 计算增量，所以**兼容**

**关键一致性检查**：
- `num_cached_block` 需要同步更新 → ✅ 已在方案中处理
- 被保留的 blocks 的 `ref_cnt` 保持 > 0 → ✅ 不进入 free queue
- 恢复时 `get_computed_blocks()` 可能返回已保留的 blocks 的子集（如果部分被驱逐） → ✅ 安全

#### 修改文件
- `vllm/v1/core/kv_cache_manager.py` — 新增 `free_partial()` 方法
- `vllm/v1/core/scheduler.py` — 抢占逻辑改造，使用部分释放

#### 预期效果
- 抢占恢复时间从全量 Recompute 降低到部分 Recompute
- 被抢占请求的共享前缀得到保护，间接保护其他请求的缓存命中
- 抢占代价降低 50-80%（取决于前缀长度占比）

#### 实现记录

**已实现**，修改文件：
- `vllm/v1/core/kv_cache_manager.py` — 新增 `free_partial()` 方法
- `vllm/v1/core/scheduler.py` — 抢占逻辑改造为部分释放 + 智能 `num_computed_tokens` 设置

##### 1. `KVCacheManager.free_partial()` 方法

```python
def free_partial(self, request: Request, keep_prefix_blocks: int) -> int:
    """部分释放：仅释放尾部 blocks，保留前 keep_prefix_blocks 个。

    被保留的 block 维持 ref_cnt > 0，不会成为驱逐候选。
    req_to_blocks 更新为仅包含保留的 blocks。
    req_to_block_hashes 不裁剪（hash chain 可复用）。
    num_cached_block 同步更新。

    Returns:
        实际释放的 blocks 数。
    """
```

关键设计：
- 尾部 blocks 按逆序释放（与 `free()` 一致）
- 释放时复用 Segmented LRU 逻辑（`_promoted` → protected zone）
- `req_to_block_hashes` 保留完整，恢复时 `get_computed_blocks()` 可直接利用

##### 2. Scheduler 抢占逻辑改造

```python
# 原始逻辑：
self.kv_cache_manager.free(preempted_req)           # 全量释放
preempted_req.num_computed_tokens = 0                # 重置为 0

# 改造后（含降级保障）：
preempt_blocks = self.kv_cache_manager.req_to_blocks.get(...)
keep_count = 0
if self.kv_cache_manager.enable_caching:
    for blk in preempt_blocks:
        if blk.block_hash is not None:
            keep_count += 1
        else:
            break  # hash chain 断裂

would_free = len(preempt_blocks) - keep_count
if keep_count > 0 and would_free >= _PREEMPT_MIN_FREE_BLOCKS:
    # 尾部可释放量足够 → 部分释放
    self.kv_cache_manager.free_partial(preempted_req, keep_count)
    preempted_req.num_computed_tokens = keep_count * self.block_size
else:
    # 释放量不足 → 降级为全量释放，保证抢占推进
    self.kv_cache_manager.free(preempted_req)
    preempted_req.num_computed_tokens = 0
```

保留 block 的选择逻辑：遍历请求的 block 列表，连续有 `block_hash` 的 block 都保留（它们有缓存价值）。遇到第一个 `block_hash is None` 即 break（hash chain 特性：后续 block 的 hash 依赖前面的）。

**降级保障**：当 `would_free`（尾部可释放的 block 数）小于 `_PREEMPT_MIN_FREE_BLOCKS`（默认 1）时，说明该请求几乎全是有 hash 的 blocks，partial free 释放不出有效空间。此时退化为全量释放，确保抢占循环中每次迭代都能回收空间，避免"空转"（反复抢占却释放不出 blocks）。

##### 3. 恢复路径兼容性分析

| 恢复时调用 | 行为 | 兼容性 |
|-----------|------|--------|
| `get_computed_blocks()` | 查找 `cached_block_hash_to_block` | ✅ hash 未清除，保留的 block 可被命中 |
| `allocate_slots()` | `num_required - len(req_to_blocks)` 计算增量 | ✅ `req_to_blocks` 已缩减为 kept_blocks |
| `_touch(computed_blocks)` | 对已保留的 block `incr_ref` | ✅ 这些 block ref_cnt ≥ 1，不在 free queue 中 |

##### 4. 设计保证

| 约束 | 保证 |
|------|------|
| 向后兼容 | caching 未启用时降级为 `free()` + `num_computed_tokens = 0` |
| 无 hash chain 破坏 | 只保留连续有 hash 的前缀，不跳跃 |
| Segmented LRU 集成 | 释放的 block 遵循 `_promoted` 逻辑 |
| `req_to_block_hashes` 完整 | 恢复时无需重算 hash |
| `num_cached_block` 一致 | 同步更新为 `min(prev, keep_count)` |
| 抢占前进保障 | 释放量不足时降级为全量释放，避免抢占循环空转 |

##### 5. 测试文件
- `tests/v1/core/test_preemption_cache_shield.py` — 抢占缓存保护单元测试

---

### 优化点 4：Proactive Cache Warming（主动缓存预热）`[进阶]`

#### 问题分析

系统冷启动时缓存为空，所有请求都需完整 prefill。在实际在线服务中，System Prompt 的种类有限且可预知（通常 5-20 种）。

**与 V1 现有机制的关系**：V1 的 `_cache_full_blocks()` 是"被动缓存"——只有请求实际执行了 prefill 后才写入。预热是"主动缓存"——在没有用户请求时提前计算 KV。

#### 设计方案

**方案 A：配置文件驱动的预热**（推荐）

```python
class CacheWarmingManager:
    """缓存预热管理器。

    功能：
    1. 从配置文件加载常见 System Prompt
    2. 系统启动后或空闲时，构造虚拟请求执行 prefill
    3. 将计算好的 KV blocks 写入缓存
    """
    
    def __init__(self, config_path: str, kv_cache_manager: KVCacheManager):
        self.kv_cache_manager = kv_cache_manager
        self.warmup_prompts = self._load_warmup_prompts(config_path)
    
    def _load_warmup_prompts(self, path: str) -> List[Dict]:
        """从 YAML/JSON 配置加载预热 prompt 列表。
        
        配置格式：
        warmup_prompts:
          - name: "general_assistant"
            tokens: [token_id_1, token_id_2, ...]
            # 或
            text: "你是一个有帮助的AI助手..."
          - name: "code_assistant"
            tokens: [...]
        """
        # ... 加载逻辑 ...
    
    def warmup(self, model_runner) -> int:
        """执行预热，返回预热的 block 数。
        
        通过构造只含 prompt 的虚拟请求，调用 model_runner 
        执行 forward pass，然后将 KV 写入 cache blocks。
        """
        total_warmed_blocks = 0
        for prompt_config in self.warmup_prompts:
            # 1. 构造虚拟 Request
            virtual_req = self._make_virtual_request(prompt_config)
            
            # 2. 预计算 block hashes
            block_hashes = hash_request_tokens(
                self.kv_cache_manager.block_size, virtual_req)
            
            # 3. 检查是否已在缓存中
            already_cached = all(
                self.kv_cache_manager._get_cached_block(h) is not None
                for h in block_hashes
            )
            if already_cached:
                continue
            
            # 4. 分配 blocks 并执行 prefill
            # ... 通过 scheduler 的特殊接口注入 ...
            
            total_warmed_blocks += len(block_hashes)
        
        return total_warmed_blocks
```

**方案 B：统计驱动的自动预热**

```python
def should_warmup(self) -> bool:
    """判断是否应执行缓存预热"""
    return (
        len(self.scheduler.running) == 0  # 系统空闲
        and self.kv_cache_manager.usage < 0.3  # 缓存利用率低
        and time.monotonic() - self.last_warmup_time > 60  # 距上次预热间隔足够
    )
```

#### 实现复杂度评估

**难点**：
1. 预热需要实际在 GPU 上执行 forward pass 才能生成有效的 KV 数据
2. 仅在 CPU 上注册 hash 没有意义（block 里没有 KV 数据，其他请求引用它会得到垃圾数据）
3. 需要在 scheduler 和 model_runner 之间增加协调机制

**结论**：预热的实现需要跨越 scheduler（CPU）和 model_runner（GPU）两层，复杂度较高，建议作为 P2 优化。

#### 修改文件
- 新增 `vllm/v1/core/cache_warming.py` — 缓存预热管理器
- `vllm/v1/core/scheduler.py` — 集成预热触发逻辑
- `vllm/v1/engine/core.py` — 启动时调用预热

#### 预期效果
- 冷启动后首批请求的 TTFT 从"完整 prefill"降低到"只需 prefill 用户输入部分"
- System Prompt 的缓存命中率在服务启动后立即接近 100%

---

### 优化点 5：Cache Efficiency Dashboard（缓存效率量化与可观测性）`[辅助]`

#### 问题分析

当前 vLLM V1 的缓存指标（`PrefixCacheStats`）非常有限：

```python
# 现有指标
class PrefixCacheStats:
    requests: int = 0   # 查询的请求数
    queries: int = 0    # 查询的 block 总数
    hits: int = 0       # 命中的 block 总数
    reset: bool = False
```

`PrefixCachingMetrics` 提供了滑动窗口的 hit_rate，但缺少以下关键维度：
- **Token 级节省量**：命中了多少 blocks 并不直观，节省了多少 prefill tokens 更有意义
- **驱逐原因分析**：被驱逐的 block 中，有多少是高频 block？
- **缓存容量利用率**：有 hash 的 block 占比、protected zone 利用率等
- **抢占影响量化**：抢占导致了多少缓存损失？

#### 设计方案

```python
@dataclass
class EnhancedPrefixCacheStats:
    """增强的缓存统计指标"""
    
    # === 基础命中指标 ===
    total_queries: int = 0          # 总查询请求数
    total_block_queries: int = 0    # 总查询 block 数
    total_block_hits: int = 0       # 总命中 block 数
    
    # === Token 级效率指标 ===
    tokens_saved_by_cache: int = 0     # 因缓存命中节省的 prefill tokens
    tokens_computed: int = 0           # 实际 prefill 的 tokens 数
    
    # === 驱逐指标 ===
    eviction_count: int = 0            # 总驱逐次数
    eviction_from_probation: int = 0   # 从 probation zone 驱逐（正常）
    eviction_from_protected: int = 0   # 从 protected zone 驱逐（不健康信号）
    
    # === 共享指标 ===
    blocks_with_sharing: int = 0       # ref_cnt > 1 的 block 数（被共享）
    max_block_ref_cnt: int = 0         # 最大 ref_cnt（衡量热度）
    
    # === 抢占影响指标 ===
    preemption_blocks_freed: int = 0   # 抢占时释放的 block 数
    preemption_blocks_preserved: int = 0  # 抢占时保留的 block 数（如果启用部分释放）
    
    @property
    def hit_rate(self) -> float:
        """Block 级命中率"""
        total = self.total_block_hits + (self.total_block_queries - self.total_block_hits)
        return self.total_block_hits / max(1, self.total_block_queries)
    
    @property
    def token_saving_rate(self) -> float:
        """Token 级节省率 = 节省的 / (节省的 + 实际计算的)"""
        total = self.tokens_saved_by_cache + self.tokens_computed
        return self.tokens_saved_by_cache / max(1, total)
    
    @property
    def healthy_eviction_rate(self) -> float:
        """健康驱逐率 = probation 驱逐 / 总驱逐"""
        return self.eviction_from_probation / max(1, self.eviction_count)
```

**埋点位置**：

| 埋点位置 | 收集的指标 |
|---------|----------|
| `get_computed_blocks()` | block_queries, block_hits, tokens_saved |
| `allocate_slots()` | tokens_computed（num_tokens 参数） |
| `_get_new_blocks()` → `_maybe_evict_cached_block()` | eviction_count, eviction_zone |
| `free()` / `free_partial()` | preemption 相关 |
| 周期性扫描 `block_pool` | blocks_with_sharing, max_ref_cnt |

#### 修改文件
- `vllm/v1/metrics/stats.py` — 增强缓存统计类
- `vllm/v1/core/kv_cache_manager.py` — 关键路径埋点
- `vllm/v1/core/scheduler.py` — 输出增强指标

---

## 三、实现路线图与依赖关系

```
优化点 5 (可观测性)  ←── 辅助所有其他优化的效果验证
    ↑
    │
优化点 1 (缓存感知调度) ─────→ 优化点 4 (缓存预热)
    │                              
    ↓                              
优化点 2 (频率感知驱逐) ──→ 优化点 3 (抢占缓存保护)
```

**推荐实现顺序**：

| 阶段 | 优化点 | 优先级 | 预计工作量 | 核心收益 | 风险 |
|------|--------|--------|-----------|---------|------|
| 阶段 1 | 优化点 5：可观测性 | P0 | 小 | 为后续优化提供量化基础 | 低 |
| 阶段 2 | 优化点 1：缓存感知调度 | P0 | 中 | 直接降低 TTFT，提升吞吐 | 低（只影响调度顺序） |
| 阶段 3 | 优化点 2：频率感知驱逐 | P0 | 中 | 提升缓存命中率，减少 thrashing | 中（需调 protected_ratio） |
| 阶段 4 | 优化点 3：抢占缓存保护 | P1 | 中 | 降低抢占恢复代价 | 中（需处理一致性） |
| 阶段 5 | 优化点 4：缓存预热 | P2 | 大 | 解决冷启动问题 | 高（需跨 scheduler/GPU） |

---

## 四、核心收益总结

| 指标 | 优化前（vLLM V1 原生） | 优化后（预期） | 来源 |
|------|---------------------|--------------|------|
| 高缓存命中请求的 TTFT | 受调度顺序影响，可能被低效请求阻塞 | 优先调度，TTFT ↓ 30-50% | 优化点 1 |
| 高频前缀缓存命中率 | ~50-70%（受 LRU thrashing 影响） | ~85-95%（分区 LRU 保护） | 优化点 2 |
| 抢占恢复耗时 | 全量 Recompute（秒级） | 部分 Recompute（百毫秒级） | 优化点 3 |
| 冷启动首批 TTFT | 完整 prefill | 仅用户输入 prefill | 优化点 4 |
| 缓存效果可见性 | 仅基础 hit_rate | 多维度指标 + 健康度分析 | 优化点 5 |

---

## 五、学习价值

通过这 5 个优化点的实现，你将深入理解：

1. **vLLM V1 的 KV Cache 管理全流程**：block 分配 → hash 计算 → 缓存查找 → 引用计数 → LRU 驱逐 → 块释放
2. **Prefix Caching 的链式 hash 机制**：`parent_hash → child_hash` 的链式依赖，以及为什么中间 break 后续全部 miss
3. **调度器与 KV Cache 的交互时序**：
   - WAITING 请求：`get_computed_blocks()` → `allocate_slots()` → `_cache_full_blocks()` → `_touch()`
   - RUNNING 请求：`allocate_slots()` → `_cache_full_blocks()`（增量 block）
4. **`FreeKVCacheBlockQueue` 的双向链表实现**：为什么不用 Python deque（需要 O(1) 中间删除 `_touch`）
5. **引用计数的精确语义**：`ref_cnt == 0` = 驱逐候选（hash 仍保留），`ref_cnt > 0` = 使用中（不可驱逐）
6. **`_touch()` 的关键作用**：从空闲队列中"抢救"一个即将被驱逐的 block
7. **不去重设计的原因**：保证 block table 是 append-only 的，简化 model_runner 的 block table 管理
8. **抢占机制的代价分析**：`num_computed_tokens = 0` + `free()` 是最昂贵的操作，但也是最简单的实现
9. **`num_computed_tokens` 必须是 `block_size` 倍数的限制**：源码注释已标记为可优化项

---

## 六、与当前 README 已有优化的关系

本项目（Prefix Cache 优化）聚焦于 **KV Cache 复用效率**，与 README 中已有的优化互补：

```
README 已有优化（调度侧）：
  优化 1: QoS 分级调度 ─────── 决定"谁先被调度"（基于优先级）
  优化 4: Token 限速 ────────── 控制"跑多快"（限制低优请求的生成速率）
  优化 7: MLFQ 多级反馈 ─────── 自适应优先级（长请求自动降级）

本项目（KV Cache 侧）：
  优化点 1: 缓存感知调度 ─────── 决定"谁更值得被调度"（缓存命中高=性价比高）
  优化点 2: 频率感知驱逐 ─────── 保护"高频缓存"（防止 System Prompt 被驱逐）
  优化点 3: 抢占缓存保护 ─────── 降低"抢占代价"（部分释放 + 快速恢复）
  优化点 4: 缓存预热 ──────────── 解决"冷启动"（主动预计算 KV）
  优化点 5: 可观测性 ──────────── 量化"缓存效果"（多维指标 + 健康度）

两者协同：
  MLFQ 层级内 FCFS → 叠加缓存感知排序 → 同层级内"高命中先跑"
  Token 限速让低优慢跑 → 腾出 token_budget → 更多高命中请求可被调度
  QoS 抢占低优请求 → 缓存保护避免抢占波及高频前缀
  MLFQ 降级长请求 → 长请求的 blocks 在 probation zone → 频率感知驱逐优先回收
```
