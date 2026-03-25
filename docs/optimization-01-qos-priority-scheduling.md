# 优化 1：推理请求 QoS 分级调度 — 实现总结

> **状态**：✅ 已实现  
> **优先级**：P0  
> **Infra 能力对标**：云网络虚拟网关 → 优先级队列 + 高优包优先转发

---

## 1. 问题背景

### 原生 vLLM V1 的调度问题

vLLM V1 调度器虽然已支持 FCFS 和 Priority 两种策略，但 Priority 模式仅使用 **静态的 `priority` 整数值**——API 调用方传什么就用什么，没有任何智能计算。

这带来两个核心问题：

**问题 1：短请求被长请求阻塞（Head-of-Line Blocking）**

```
时间线:
t0: 长请求 A 到达（prompt 4000 tokens, priority=0）
t1: 短请求 B 到达（prompt 50 tokens, priority=0）
t2: 短请求 C 到达（prompt 30 tokens, priority=0）

静态优先级视角：三个请求 priority 都是 0，完全相同
调度顺序：A → B → C（先来先服务）
结果：短请求 B、C 必须等长请求 A 完成 prefill 后才能被调度
```

**问题 2：低优请求可能被永远饿死**

```
持续高负载场景：
t0: 低优请求 X（priority=5）入队
t1: 高优请求 Y（priority=0）入队 → 排在 X 前面
t2: 高优请求 Z（priority=0）入队 → 排在 X 前面
...
t100: 仍然不断有高优请求到达
结果：X 永远无法被调度 → 饿死
```

### 设计目标

1. **自动识别短请求**：无需人工标注，短 prompt 请求自动获得更高调度优先级
2. **防止饿死**：等待时间越长的请求，优先级自动提升
3. **向后兼容**：不传 `priority` 参数也能正常工作，且自动享受长度分级的好处
4. **动态调整**：每个调度步都重新计算优先级，反映最新的等待时间

---

## 2. 核心设计：多维优先级公式

```
effective_priority = base_priority + length_adjustment - starvation_boost
```

| 分量 | 含义 | 取值范围 | 效果 |
|------|------|---------|------|
| `base_priority` | API 调用方传入的静态优先级 | 0（默认，最高优先） | 越小越优先 |
| `length_adjustment` | 根据 prompt 长度自动调整 | -2 ~ +1 | 短请求减小（更优先），长请求增大 |
| `starvation_boost` | 根据等待时间自动提升 | 0 ~ 10 | 被减去 → 等越久越优先 |

### 长度分级规则

| Prompt 长度 | 分级 | `length_adjustment` | 效果 |
|------------|------|-------------------|------|
| < 512 tokens | 短请求 | **-2** | 显著提升优先级 |
| 512 ~ 2048 tokens | 中等请求 | **-1** | 适度提升优先级 |
| ≥ 2048 tokens | 长请求 | **+1** | 轻微降低优先级 |

### 等待时间防饿死规则

```
starvation_boost = min(waiting_seconds / 5.0, 10)
```

| 等待时间 | `starvation_boost` | 效果 |
|---------|-------------------|------|
| 0 ~ 5 秒 | 0 | 无额外提升 |
| 5 ~ 10 秒 | 1 | 优先级提升 1 |
| 10 ~ 15 秒 | 2 | 优先级提升 2 |
| ... | ... | ... |
| ≥ 50 秒 | 10（封顶） | 最大提升，几乎不可能继续被饿死 |

### 实际效果举例

```
场景：3 个请求同时在 waiting 队列，等待 12 秒后的优先级计算

请求 A: priority=0, prompt=4000 tokens (长请求)
  effective = 0 + (+1) - 2 = -1

请求 B: priority=0, prompt=100 tokens (短请求)
  effective = 0 + (-2) - 2 = -4  ← 最高优先级

请求 C: priority=3, prompt=200 tokens (短请求但 API 标了低优)
  effective = 3 + (-2) - 2 = -1

调度顺序：B(-4) → A(-1) = C(-1) → 按到达时间决定
```

```
场景：低优请求等待 30 秒后

请求 X: priority=5, prompt=100 tokens, 等待 30 秒
  effective = 5 + (-2) - 6 = -3  ← 等待时间拯救了它

请求 Y: priority=0, prompt=100 tokens, 刚到达
  effective = 0 + (-2) - 0 = -2

调度顺序：X(-3) → Y(-2)  （等了 30 秒的低优请求反超了刚到的高优请求）
```

---

## 3. 代码改动详解

### 修改文件清单

| 文件 | 改动类型 | 改动内容 |
|------|---------|---------|
| `vllm/v1/request.py` | 修改 | Request 类新增 QoS 多维优先级 |
| `vllm/v1/core/sched/scheduler.py` | 修改 | 调度器集成动态优先级更新 |
| `vllm/v1/core/sched/request_queue.py` | 修改 | PriorityRequestQueue 新增 reheapify |

### 3.1 `vllm/v1/request.py` — Request 类增强

#### 新增：类级别 QoS 配置常量

```python
class Request:
    # ---- QoS Priority Configuration ----
    SHORT_PROMPT_THRESHOLD: int = 512      # tokens
    MEDIUM_PROMPT_THRESHOLD: int = 2048    # tokens

    SHORT_PROMPT_BOOST: int = 2    # short requests get significant boost
    MEDIUM_PROMPT_BOOST: int = 1   # medium requests get moderate boost
    LONG_PROMPT_PENALTY: int = 1   # long requests get slight penalty

    STARVATION_DECAY_INTERVAL: float = 5.0  # seconds
    MAX_STARVATION_BOOST: int = 10
```

**设计考量**：使用类级别常量而非配置文件，方便后续迁移到配置驱动。所有阈值都可以直接调整。

#### 新增：`_effective_priority` 缓存字段

```python
def __init__(self, ...):
    ...
    self.priority = priority
    # QoS: cached effective priority (updated by scheduler each step)
    self._effective_priority: int | None = None
```

**为什么用缓存而非每次计算？**  
- `__lt__` 比较方法在堆排序中被频繁调用（每次 heapify 都会多次比较）
- 如果每次 `__lt__` 都调用 `time.time()` 重新计算，性能开销大
- 在调度步开始时统一计算一次，后续直接用缓存值

#### 新增：`effective_priority` 属性（只读）

```python
@property
def effective_priority(self) -> int:
    if self._effective_priority is not None:
        return self._effective_priority
    return self.priority  # 未计算时回退到静态优先级
```

#### 新增：`compute_effective_priority()` 方法

```python
def compute_effective_priority(self, now: float | None = None) -> int:
    if now is None:
        now = time.time()

    base = self.priority

    # 1. 长度分级
    num_prompt = self.num_prompt_tokens
    if num_prompt < self.SHORT_PROMPT_THRESHOLD:
        length_adjustment = -self.SHORT_PROMPT_BOOST
    elif num_prompt < self.MEDIUM_PROMPT_THRESHOLD:
        length_adjustment = -self.MEDIUM_PROMPT_BOOST
    else:
        length_adjustment = self.LONG_PROMPT_PENALTY

    # 2. 防饿死衰减
    waiting_time = now - self.arrival_time
    starvation_boost = min(
        int(waiting_time / self.STARVATION_DECAY_INTERVAL),
        self.MAX_STARVATION_BOOST,
    )

    self._effective_priority = base + length_adjustment - starvation_boost
    return self._effective_priority
```

#### 修改：`__lt__()` 比较方法

```python
# 改造前：使用静态 priority
def __lt__(self, other):
    if self.priority != other.priority:
        return self.priority < other.priority
    ...

# 改造后：使用 effective_priority
def __lt__(self, other):
    self_prio = self.effective_priority
    other_prio = other.effective_priority
    if self_prio != other_prio:
        return self_prio < other_prio
    ...
```

**影响范围**：`PriorityRequestQueue` 内部的 `heapq` 依赖 `__lt__` 做排序，修改后所有堆操作自动使用多维优先级。

---

### 3.2 `vllm/v1/core/sched/scheduler.py` — 调度器集成

#### 修改：`schedule()` 方法 — 每步开始时更新优先级

```python
def schedule(self) -> SchedulerOutput:
    ...
    self.kv_cache_manager.new_step_starts()

    # [新增] QoS: 每个调度步开始时更新所有请求的动态优先级
    if self.policy == SchedulingPolicy.PRIORITY:
        self._update_effective_priorities()

    # 后续正常调度逻辑...
```

**时机选择**：放在 `new_step_starts()` 之后、调度 RUNNING 请求之前，确保本步所有调度决策都使用最新的优先级值。

#### 修改：抢占逻辑 — 使用 `effective_priority`

```python
# 改造前：使用静态 priority
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time),
)

# 改造后：使用 effective_priority
preempted_req = max(
    self.running,
    key=lambda r: (r.effective_priority, r.arrival_time),
)
```

**效果**：抢占时选择 `effective_priority` 值最大（优先级最低）的请求，考虑了长度和等待时间因素。

#### 新增：`_update_effective_priorities()` 方法

```python
def _update_effective_priorities(self) -> None:
    now = time.time()

    # 更新 waiting 队列中所有请求的优先级
    for request in self.waiting:
        request.compute_effective_priority(now)
    # 重建堆序（优先级变了，堆序可能失效）
    if isinstance(self.waiting, PriorityRequestQueue):
        self.waiting._reheapify()

    # 更新 skipped_waiting 队列
    for request in self.skipped_waiting:
        request.compute_effective_priority(now)
    if isinstance(self.skipped_waiting, PriorityRequestQueue):
        self.skipped_waiting._reheapify()

    # 更新 running 请求（用于抢占决策）
    for request in self.running:
        request.compute_effective_priority(now)
```

**为什么要 reheapify？**  
- `PriorityRequestQueue` 内部是堆（`heapq`），堆的有序性依赖元素的 `__lt__` 比较结果
- 更新 `_effective_priority` 后，请求间的相对大小关系可能改变
- 必须调用 `heapq.heapify()` 重建堆序，时间复杂度 O(n)

---

### 3.3 `vllm/v1/core/sched/request_queue.py` — 优先级队列支持

#### 新增：`PriorityRequestQueue._reheapify()` 方法

```python
def _reheapify(self) -> None:
    """Re-heapify the internal heap after external priority changes."""
    heapq.heapify(self._heap)
```

**设计考量**：这是一个内部方法（`_` 前缀），只应由调度器在更新完所有请求优先级后调用，避免外部代码滥用。

---

## 4. 调度流程图

```
每个调度步（schedule() 被调用）：

┌────────────────────────────────────────────────────┐
│ Step 1: 更新动态优先级                               │
│                                                     │
│   for req in waiting + skipped_waiting + running:   │
│       req.compute_effective_priority(now)            │
│                                                     │
│   waiting.reheapify()  # 重建堆序                    │
├────────────────────────────────────────────────────┤
│ Step 2: 调度 RUNNING 请求（续写 decode tokens）       │
│                                                     │
│   每个 running 请求分配 1 个 decode token 的 budget   │
│   分配 KV blocks，失败则触发抢占：                     │
│     → 选择 effective_priority 最大的 running 请求抢占  │
├────────────────────────────────────────────────────┤
│ Step 3: 调度 WAITING 请求（新请求 prefill）           │
│                                                     │
│   从 waiting 堆中按 effective_priority 弹出           │
│   短请求(eff_prio=-2) 排在长请求(eff_prio=+1) 前面    │
│   等了很久的请求(starvation_boost) 自动前移            │
│                                                     │
│   分配 KV blocks，失败则触发抢占（同上）               │
├────────────────────────────────────────────────────┤
│ Step 4: 构建 SchedulerOutput 并返回                  │
└────────────────────────────────────────────────────┘
```

---

## 5. 向后兼容性

| 场景 | 行为 | 说明 |
|------|------|------|
| 不传 `priority` 参数 | `base=0` → 长度分级仍然生效 | 短请求自动排前面 |
| 使用 FCFS 调度策略 | `_update_effective_priorities()` 不被调用 | 完全无影响 |
| 所有请求 prompt 长度相同 | `length_adjustment` 相同 → 退化为按到达时间 | 行为与原版一致 |
| 低负载（无等待） | `starvation_boost=0` → 只有长度分级 | 开销极小 |

---

## 6. 性能开销分析

| 操作 | 频率 | 时间复杂度 | 实际开销 |
|------|------|----------|---------|
| `compute_effective_priority()` | 每步 × 每个请求 | O(1) | 几次算术运算，忽略不计 |
| `heapify(waiting)` | 每步 × 1 次 | O(n) | n 为 waiting 长度，通常 <1000 |
| `time.time()` | 每步 × 1 次 | O(1) | 系统调用，~100ns |
| `__lt__` 增加的开销 | 每次堆比较 | O(1) | 多一次属性访问，忽略不计 |

**总结**：对于典型场景（waiting 队列 < 1000 个请求），每步额外开销在微秒级，相比模型推理的毫秒级耗时完全可忽略。

---

## 7. 可配置参数一览

| 参数 | 默认值 | 含义 | 调优建议 |
|------|--------|------|---------|
| `SHORT_PROMPT_THRESHOLD` | 512 | 短请求阈值（tokens） | 根据业务平均 prompt 长度调整 |
| `MEDIUM_PROMPT_THRESHOLD` | 2048 | 中等请求阈值（tokens） | 根据模型 max_model_len 调整 |
| `SHORT_PROMPT_BOOST` | 2 | 短请求优先级提升值 | 增大 → 短请求更优先 |
| `MEDIUM_PROMPT_BOOST` | 1 | 中等请求优先级提升值 | - |
| `LONG_PROMPT_PENALTY` | 1 | 长请求优先级惩罚值 | 增大 → 长请求更靠后 |
| `STARVATION_DECAY_INTERVAL` | 5.0s | 每多久提升 1 级优先级 | 减小 → 防饿死更激进 |
| `MAX_STARVATION_BOOST` | 10 | 等待时间提升的上限 | 增大 → 允许更大的提升幅度 |

---

## 8. 与其他优化的关系

### 与优化 4（Token 限速）的关系

优化 1 决定**谁先被调度**，优化 4 决定**被调度后跑多快**。二者是调度的两个阶段：

```
优化 1（入口）→ 决定调度顺序 → 请求被选中执行 → 优化 4（运行时）→ 控制 token 生成速率
```

详见 README 中的优化关系说明。

### 与优化 6（WFQ）的关系

优化 6 在优化 1 **之上**工作：先按租户分配资源份额，再在每个租户内使用优化 1 的多维优先级排序。

### 与优化 7（MLFQ）的关系

优化 7 可作为优化 1 的**运行时补充**：优化 1 基于 prompt 长度**预估**优先级，优化 7 根据实际 token 消耗**事后调整**。

---

## 9. 如何验证

### 启动 Priority 调度模式

```bash
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> \
    --scheduling-policy priority \
    --max-model-len 4096
```

### 发送带优先级的请求

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1")

# 高优请求
response = client.chat.completions.create(
    model="xxx",
    messages=[{"role": "user", "content": "你好"}],
    extra_body={"priority": 0}  # 0 = 最高优先级
)

# 低优请求
response = client.chat.completions.create(
    model="xxx",
    messages=[{"role": "user", "content": "写一篇论文..."}],
    extra_body={"priority": 5}  # 数值越大优先级越低
)
```

### 验证效果

1. 同时发送多个长请求和短请求，观察短请求的 TTFT 是否显著降低
2. 持续发送高优请求，观察低优请求是否在等待足够时间后最终被调度（防饿死）
3. 对比 FCFS 和 Priority 模式下的 P50/P99 TTFT 差异
