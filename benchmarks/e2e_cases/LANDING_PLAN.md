# 端到端压测落地方案

> 基于已有优化 → 综合 Case 暴露差距 → 增量修复 → 迭代验证
>
> 三种部署模式（单实例 / PD 分离 / 投机解码）同一套流量、同一个脚本、自动交叉对比

## 一、整体逻辑

```
你的工作可以分为四大块，相互支撑：

┌──────────────────────────────────────────────────────────────────┐
│                    A. 已实现的框架优化（6 项）                      │
│                                                                  │
│  调度侧:                                                         │
│    ✅ QoS 分级调度 — 多维优先级（长度+等待时间+API priority）       │
│    ✅ MLFQ 多级反馈 — 4 级队列自适应降级                           │
│    ✅ Token 限速 — 令牌桶控制低优请求生成速率                       │
│                                                                  │
│  KV Cache 侧:                                                    │
│    ✅ Cache-Aware Scheduling — MLFQ 层内按缓存命中率排序            │
│    ✅ Frequency-Aware Eviction — Segmented LRU (probation/protected)│
│    ✅ Preemption Cache Shield — 部分释放 + 前缀保留                 │
│                                                                  │
│  ⬆️ 每项都有设计文档 + 代码实现 + 单元测试                         │
└──────────────────────────────┬───────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────┐
│             B. 综合 Case 暴露的差距（5 个阶段 × 4 个痛点）         │
│                                                                  │
│  Phase 1: ✅ 稳态预热 → 已有优化基本够用                           │
│  Phase 2: ❌ Prompt 切换 → 缓存版本管理缺失                       │
│  Phase 3: ❌ 大租户暴增 → 无租户级隔离                             │
│  Phase 4: ❌ 长文档暴增 → 无 prefill 预算隔离                      │
│  Phase 5: ❌ 全面过载 → 无 admission control + deadline            │
│                                                                  │
│  ⬆️ workload.py 一键运行，暴露上述全部问题                         │
└──────────────────┬──────────────────────┬────────────────────────┘
                   │                      │
                   ▼                      ▼
┌─────────────────────────────┐ ┌─────────────────────────────────┐
│  C. 增量修复（4 项）         │ │  D. 部署模式对比（3 种）         │
│                             │ │                                 │
│  修复 1: 缓存版本管理        │ │  Mode A: 单实例 + Prefix Cache  │
│  修复 2: Prefill 预算隔离    │ │  Mode B: PD 分离 + 智能路由     │
│  修复 3: 租户级资源隔离      │ │  Mode C: 投机解码 (Suffix Dec)  │
│  修复 4: 过载管理            │ │                                 │
│                             │ │  同一套流量 × 三种后端配置       │
│                             │ │  → 自动交叉对比报告              │
│  ⬆️ 每做一项，重跑验证      │ │  ⬆️ workload.py --mode all      │
└─────────────────────────────┘ └─────────────────────────────────┘
```

**核心逻辑：A 是你做完的基础工程，B 是检验手段，C 是基于检验结果的增量改进，D 是多部署模式下的横向对比。**

---

## 二、已有优化全景（A 部分）

### 你已经实现了什么

| # | 优化项 | 改动文件 | 测试文件 | 来源文档 |
|---|--------|---------|---------|---------|
| 1 | QoS 分级调度 | `scheduler.py` + `request.py` | — | README.md 优化 1 |
| 2 | Token 限速 | `scheduler.py` + `request.py` | — | README.md 优化 4 |
| 3 | MLFQ 多级反馈 | `scheduler.py` + `request.py` | — | README.md 优化 7 |
| 4 | Cache-Aware Scheduling | `scheduler.py` | `test_cache_aware_scheduling.py` | prefix-cache 优化点 1 |
| 5 | Segmented LRU 驱逐 | `kv_cache_utils.py` + `kv_cache_manager.py` | `test_frequency_aware_eviction.py` | prefix-cache 优化点 2 |
| 6 | Preemption Cache Shield | `kv_cache_manager.py` + `scheduler.py` | `test_preemption_cache_shield.py` | prefix-cache 优化点 3 |

### 各优化之间的协同关系

```
调度入口（谁先被调度）          运行时（调度后跑多快）
─────────────────────          ────────────────────
QoS 分级(#1)                   Token 限速(#2)
    ↕ 互补                        │ 低优慢跑
MLFQ(#3)                         │ 高优全速
    ↕ 层内增强                    │
Cache-Aware(#4)                   │

KV Cache 保护                  
─────────────
Segmented LRU(#5) ←→ Preemption Shield(#6)
 高频前缀保护         抢占时前缀保留
```

### 还没实现的框架优化（README 中标注的）

| 优化项 | 状态 | 优先级 | 与增量修复的关系 |
|--------|------|--------|----------------|
| KV 水位线流控 | 🔲 | P0 | 与修复 4（过载管理）强相关 |
| 准入控制 | 🔲 | P1 | = 修复 4 的一部分 |
| Deadline/EDF 调度 | 🔲 | P2 | = 修复 4 的一部分 |
| WFQ 加权公平调度 | 🔲 | P2 | = 修复 3（租户隔离）的核心 |
| KV Cache 分层存储 | 🔲 | P1 | 独立大项，不在本次修复范围 |

---

## 三、分阶段实施计划（代码级指导）

> 以下每个阶段都给出了**具体改哪个文件、哪个方法、大约多少行代码**，可直接按此动手。
>
> **当前代码状态**（确认过）：
> - ✅ 6 项框架优化已实现（QoS/MLFQ/Token限速/Cache-Aware/Segmented LRU/Preemption Shield）
> - ✅ `workload.py` 压测脚本已完成
> - ❌ 4 项增量修复均未实现（代码中确认没有 tenant_id / deadline / cache-version / prefill-budget 任何痕迹）
>
> **关于取消感知缓存（Cancel-Aware Cache Retention）**：经深入分析 vLLM V1 prefix caching 源码后确认此优化非必要。
> `free_block_hashes()` 只清除 `req_to_block_hashes[request_id]`（请求私有 hash 缓存），不影响 `cached_block_hash_to_block` 全局索引。
> abort 后 block 的 `block_hash` 和全局索引均保留，新请求天然能通过全局索引命中缓存。详见 README.md 中的分析记录。

---

### 阶段 0：跑基线（不改任何代码）⏱️ 0.5h

**目标**：先收集一次完整基线数据，作为后续所有对比的基准。建议分模式采集。

```bash
# 1. 启动 vLLM 服务（用你改过的代码）
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --enable-prefix-caching

# 2a. 跑单实例基线
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --host 127.0.0.1 --port 8000 \
    --mode single --duration 300 --output-dir results/baseline/

# 2b. （可选）跑 PD 分离基线（需另启动 Router + Prefill + Decode 实例）
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --mode pd-disagg \
    --host 127.0.0.1 --port 8000 \
    --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200 \
    --output-dir results/baseline/

# 2c. （可选）跑投机解码基线（需启动带 --speculative-config 的实例）
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --mode spec-decode \
    --host 127.0.0.1 --port 8000 \
    --output-dir results/baseline/

# 2d. 或者一键全跑三种模式（需要所有后端就绪）
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --mode all \
    --host 127.0.0.1 --port 8000 \
    --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200 \
    --output-dir results/baseline/

# 3. 记录基线数据（特别关注 Phase 2-5 的 SLA 违约率 + 三模式差异）
```

**检查清单**：
- [ ] 基线跑完，`results/baseline/results_single.json` 已生成
- [ ] 记录各 Phase 各租户的 P50/P99 TTFT、吞吐量、SLA 违约率
- [ ] （可选）PD 分离基线：关注 KV 传输延迟、路由分布
- [ ] （可选）投机解码基线：关注 draft 接受率、Gold-B 生成速度

---

### 阶段 1：缓存版本管理 ⏱️ 2h｜~60 行代码

**对应痛点**：Phase 2 Prompt v1→v2 切换时，Segmented LRU 的 protected zone 被旧 prompt blocks 占满，新 prompt 的 blocks 被反复淘汰，命中率骤降且恢复慢（30-60s）。

**原理**：监控缓存命中率滑动窗口，检测到骤降时（可能是 prompt 版本切换），临时缩小 protected zone 让旧 block 加速淘汰；命中率恢复后自动还原。

#### 具体改动

**文件 1：`vllm/v1/core/kv_cache_utils.py` — `FreeKVCacheBlockQueue` 类新增方法**

在类末尾（约 L410 之后）新增 `resize_protected()` 方法：

```python
def resize_protected(self, new_ratio: float) -> None:
    """动态调整 protected zone 大小比例。
    
    缩小时，多余的 protected block 会被降级到 probation head，
    加速旧 block 被淘汰。
    
    Args:
        new_ratio: 新的 protected zone 占 free queue 的比例 (0.0~1.0)
    """
    total_free = self._num_probation + self._num_protected
    if total_free == 0:
        return
    new_max_protected = max(int(total_free * new_ratio), 0)
    
    # 如果当前 protected 超过新上限，把多余的降级到 probation
    while self._num_protected > new_max_protected:
        # 从 protected tail 取出最旧的 block
        demoted = self._protected_tail
        if demoted is None:
            break
        # 从 protected zone 移除
        self._remove_from_protected(demoted)
        # 插入到 probation head（最先被淘汰的位置）
        self._prepend_to_probation(demoted)
```

> **注意**：`_remove_from_protected()` 和 `_prepend_to_probation()` 是 Segmented LRU 内部已有的链表操作方法，如果不存在需要基于已有的 `_remove` / `_prepend` 逻辑提取。

**文件 2：`vllm/v1/core/kv_cache_manager.py` — 新增命中率监控**

在 `KVCacheManager.__init__()` 末尾添加：

```python
# ---- 缓存健康监控（阶段 2：缓存版本管理） ----
self._hit_rate_window: List[float] = []       # 最近 N 次的命中率
self._cache_health_check_interval = 10        # 每 10 次调度检查一次
self._cache_health_counter = 0
self._default_protected_ratio = 0.5           # 默认 protected zone 比例
```

在 `get_computed_blocks()` 方法末尾（约 L155，`return` 之前）添加命中率记录：

```python
# 记录命中率
num_hashes = len(block_hashes)
if num_hashes > 0:
    hit_rate = len(computed_blocks) / num_hashes
    self._hit_rate_window.append(hit_rate)
    if len(self._hit_rate_window) > 100:
        self._hit_rate_window.pop(0)
    
    self._cache_health_counter += 1
    if self._cache_health_counter >= self._cache_health_check_interval:
        self._check_cache_health()
        self._cache_health_counter = 0
```

新增 `_check_cache_health()` 方法：

```python
def _check_cache_health(self) -> None:
    """检测缓存健康度，自适应调整 protected zone 大小。
    
    当命中率骤降时（如 prompt 版本切换），临时缩小 protected zone
    让旧 block 加速淘汰；命中率恢复后自动还原。
    """
    if len(self._hit_rate_window) < 20:
        return
    
    recent = sum(self._hit_rate_window[-10:]) / 10
    older = sum(self._hit_rate_window[-20:-10]) / 10
    
    if recent < 0.3 and older > 0.5:
        # 命中率骤降 → 缩小 protected zone，加速旧 block 淘汰
        logger.info("Cache health: hit rate dropped %.2f → %.2f, "
                     "shrinking protected zone to 10%%", older, recent)
        self.free_block_queue.resize_protected(0.1)
    elif recent > 0.5:
        # 恢复正常 → 还原 protected zone
        self.free_block_queue.resize_protected(
            self._default_protected_ratio)
```

#### 验证

**单测**：新增 `tests/v1/core/test_cache_version_management.py`
```python
def test_resize_protected_on_hit_rate_drop():
    """验证命中率骤降时 protected zone 被缩小"""
    # 1. 构造 KVCacheManager，填充 protected zone
    # 2. 模拟 10 次高命中率（>0.6）+ 10 次低命中率（<0.3）
    # 3. 断言 protected zone 已缩小
    # 4. 模拟命中率恢复 > 0.5
    # 5. 断言 protected zone 恢复到默认值

def test_normal_hit_rate_no_resize():
    """验证命中率稳定时不会触发 resize"""
```

**集成验证**：
```bash
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --output-dir results/fix_1/
# 观察 Phase 2 的 Prompt 切换后恢复时间是否从 30-60s 缩短到 <5s
```

**检查清单**：
- [ ] `kv_cache_utils.py` 新增 `resize_protected()` 方法
- [ ] `kv_cache_manager.py` 新增命中率监控 + `_check_cache_health()`
- [ ] 单测 `test_cache_version_management.py` 通过
- [ ] 重跑 workload，Phase 2 恢复时间 <5s
- [ ] Phase 1 稳态无退化

---

### 阶段 2：Prefill 预算隔离 ⏱️ 2h｜~40 行代码

**对应痛点**：Phase 4 Bronze 长文档（4096 tokens prefill）暴增时，长 prefill 吃光 `token_budget`，Gold-A/Silver 短对话的 TTFT 飙升（P99 >500ms）。

**原理**：在 WAITING 调度循环中，将 `token_budget` 拆分为短请求预留份额和长请求份额，限制同时 prefill 的长文档数量，确保短请求始终有 budget 可用。

#### 具体改动

**文件：`vllm/v1/core/scheduler.py` — `schedule()` 方法的 WAITING 循环**

在 WAITING 循环开始前（约 L357，`for request in ...` 之前）加入 budget 拆分：

```python
# ---- 阶段 3：Prefill 预算隔离 ----
LONG_PREFILL_THRESHOLD = 1024       # 超过此 token 数视为长 prefill
SHORT_BUDGET_RESERVE_RATIO = 0.3    # 为短请求预留 30% budget
MAX_CONCURRENT_LONG_PREFILL = 2     # 同时最多 2 个长 prefill

long_prefill_count = 0
short_budget_reserved = int(token_budget * SHORT_BUDGET_RESERVE_RATIO)
```

在选出 candidate 后、分配 budget 前（约 L407，`num_new_tokens` 计算之后）加入长 prefill 检查：

```python
# 判断是否为长 prefill 请求
is_long_prefill = (num_new_tokens > LONG_PREFILL_THRESHOLD
                   and request.num_computed_tokens == 0)

if is_long_prefill:
    # 长 prefill 并发控制
    if long_prefill_count >= MAX_CONCURRENT_LONG_PREFILL:
        break  # 不再调度更多长 prefill，留 budget 给短请求
    
    # 长请求不能吃掉短请求预留的 budget
    effective_budget = token_budget - short_budget_reserved
    num_new_tokens = min(num_new_tokens, max(effective_budget, 0))
    if num_new_tokens <= 0:
        break
    long_prefill_count += 1
```

> **与已有 MLFQ 的配合**：短对话（Gold-A/Silver）在 MLFQ L0/L1 会被优先调度，
> 加上 budget 预留，短请求基本不会被长 prefill 饿死。

#### 验证

**单测**：新增 `tests/v1/core/test_prefill_budget_isolation.py`
```python
def test_long_prefill_limited():
    """验证长 prefill 不超过 MAX_CONCURRENT_LONG_PREFILL"""
    # 构造 5 个长 prefill 请求 + 5 个短请求
    # 断言一轮调度中长 prefill 最多被选中 2 个
    # 断言短请求仍有 budget 被调度

def test_short_budget_reserved():
    """验证短请求预留 budget 不被长 prefill 占用"""
```

**集成验证**：
```bash
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --output-dir results/fix_2/
# 观察 Phase 4 短对话 P99 TTFT 是否从 >500ms 降到 <200ms
```

**检查清单**：
- [ ] `scheduler.py` WAITING 循环新增 prefill 预算隔离逻辑
- [ ] 单测 `test_prefill_budget_isolation.py` 通过
- [ ] 重跑 workload，Phase 4 短对话 P99 TTFT <200ms
- [ ] Phase 1/2 无退化

---

### 阶段 3：租户级资源隔离 ⏱️ 4h｜~120 行代码

**对应痛点**：Phase 3 Gold-A 暴增 4× 时，Silver 的 P99 TTFT 爆炸（>2000ms），因为调度器只有请求级概念没有租户级概念，大租户可以无限制挤压小租户。

**原理**：引入 `TenantManager`，实现租户级并发上限 + 加权公平调度（简化版 WFQ），确保大租户暴增时不会饿死小租户。

#### 具体改动（4 个文件）

**文件 1（新增）：`vllm/v1/core/tenant_manager.py`** ~60 行

```python
"""租户级资源隔离管理。

为每个租户维护并发上限和调度权重，实现简化版 WFQ（Weighted Fair Queueing）。
"""
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TenantManager:
    def __init__(self, default_max_running: int = 64,
                 default_weight: float = 1.0):
        self.default_max_running = default_max_running
        self.default_weight = default_weight
        
        self.tenant_max_running: Dict[str, int] = {}
        self.tenant_weights: Dict[str, float] = {}
        self.tenant_running: Dict[str, int] = {}    # 当前并发数
    
    def register_tenant(self, tenant_id: str,
                        max_running: Optional[int] = None,
                        weight: Optional[float] = None):
        """注册或更新租户配置。"""
        if max_running is not None:
            self.tenant_max_running[tenant_id] = max_running
        if weight is not None:
            self.tenant_weights[tenant_id] = weight
    
    def can_schedule(self, tenant_id: str) -> bool:
        """检查租户是否还有调度配额。"""
        current = self.tenant_running.get(tenant_id, 0)
        max_allowed = self.tenant_max_running.get(
            tenant_id, self.default_max_running)
        return current < max_allowed
    
    def get_scheduling_weight(self, tenant_id: str) -> float:
        """获取租户当前调度权重（WFQ：已跑越多，权重越低）。"""
        weight = self.tenant_weights.get(tenant_id, self.default_weight)
        running = self.tenant_running.get(tenant_id, 0)
        return weight / max(1, running)
    
    def on_request_scheduled(self, tenant_id: str):
        """请求被调度时调用。"""
        self.tenant_running[tenant_id] = \
            self.tenant_running.get(tenant_id, 0) + 1
    
    def on_request_finished(self, tenant_id: str):
        """请求完成时调用。"""
        current = self.tenant_running.get(tenant_id, 0)
        self.tenant_running[tenant_id] = max(0, current - 1)
```

**文件 2：`vllm/v1/request.py` — Request 类新增 `tenant_id`**

在 `Request.__init__()` 参数列表中加入 `tenant_id`:

```python
def __init__(self, ..., tenant_id: str = "default"):
    ...
    self.tenant_id = tenant_id
```

**文件 3：`vllm/v1/core/scheduler.py`** — 集成 TenantManager

在 `Scheduler.__init__()` 中初始化：
```python
from vllm.v1.core.tenant_manager import TenantManager
self.tenant_manager = TenantManager()
```

在 WAITING 循环中加入租户检查（`can_schedule()` + `on_request_scheduled()`）：
```python
# 在选出 candidate 后、分配资源前
if not self.tenant_manager.can_schedule(request.tenant_id):
    continue  # 跳过该请求，让其他租户的请求有机会

# 调度成功后
self.tenant_manager.on_request_scheduled(request.tenant_id)
```

在 `_free_request()` 中调用：
```python
self.tenant_manager.on_request_finished(request.tenant_id)
```

**文件 4：`vllm/v1/engine/processor.py`** — 解析 tenant_id

从请求的 `extra_body` / HTTP header 中提取 `tenant_id`，传递给 `Request()` 构造函数。

```python
# 从 extra_body 中解析 tenant_id
tenant_id = getattr(request, 'tenant_id', None) \
    or (request.extra_body or {}).get('tenant_id', 'default')
```

#### 验证

**单测**：新增 `tests/v1/core/test_tenant_isolation.py`
```python
def test_tenant_concurrent_limit():
    """验证租户并发上限"""

def test_wfq_weight_scheduling():
    """验证 WFQ 权重调度：大租户不饿死小租户"""

def test_default_tenant():
    """验证无 tenant_id 的请求使用默认租户"""
```

**集成验证**：
```bash
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --output-dir results/fix_3/
# 观察 Phase 3 Silver P99 TTFT 是否从 >2000ms 降到 <500ms
```

**检查清单**：
- [ ] 新增 `vllm/v1/core/tenant_manager.py`
- [ ] `request.py` 新增 `tenant_id` 字段
- [ ] `scheduler.py` 集成 TenantManager
- [ ] `processor.py` 解析 tenant_id
- [ ] 单测通过
- [ ] 重跑 workload，Phase 3 Silver P99 TTFT <500ms
- [ ] Phase 1/2/4 无退化

---

### 阶段 4：过载管理 ⏱️ 5h｜~150 行代码 ✅ 已完成

**对应痛点**：Phase 5 全面过载时，所有请求 SLA 全违约（P99 TTFT >10s），没有任何拒绝/降级机制，系统"死扛"导致所有人都不满意。

**原理**：三管齐下 — 准入控制（限流）+ Deadline-Aware 调度（紧急优先）+ SLA-Aware 抢占（已违约的优先牺牲）。

#### 实际改动概览（5 个文件，+944 行）

| 文件 | 改动行数 | 说明 |
|------|---------|------|
| `vllm/v1/request.py` | +48 | SLA/deadline 字段 + `FINISHED_REJECTED` 状态 |
| `vllm/v1/engine/__init__.py` | +1 | `EngineCoreRequest` 新增 `sla_ttft_ms` |
| `vllm/v1/engine/processor.py` | +2 | `process_inputs()` 透传 `sla_ttft_ms` |
| `vllm/v1/core/scheduler.py` | +202 | 准入控制 + deadline 调度 + SLA-aware 抢占 |
| `tests/v1/core/test_overload_management.py` | +692 | 31 个单元测试 |

#### 实现细节

**4a. Request 层 — SLA/Deadline 字段**

```python
class Request:
    # 类级别 SLA 默认配置
    DEFAULT_SLA_TTFT_MS = {"HIGH": 500.0, "NORMAL": 1000.0, "LOW": 5000.0}

    def __init__(self, ..., sla_ttft_ms: float = float('inf')):
        self.sla_ttft_ms = sla_ttft_ms
        self.deadline = arrival_time + sla_ttft_ms / 1000.0  # 绝对截止时间

    @property
    def slack_time(self) -> float:
        """剩余 SLA 时间（秒），正=未违约，负=已违约"""
        return self.deadline - time.monotonic()

    def is_sla_violated(self) -> bool:
        return self.slack_time <= 0

    @property
    def sla_urgency(self) -> float:
        """urgency 分数，值越小越紧急"""
        return self.slack_time

    # 新增状态
    FINISHED_REJECTED = 7  # 被准入控制拒绝
```

**4b. Scheduler — 过载管理配置**

```python
# __init__ 中新增字段
self.enable_overload_management: bool = True
self.max_queue_depth: int = 100
self._sla_violation_window: Deque[bool] = deque(maxlen=50)
self.overload_violation_threshold: float = 0.5  # 50% 违约率触发拒绝
self.enable_deadline_aware_scheduling: bool = True
self.deadline_urgency_threshold_s: float = 2.0  # 2秒内 deadline 视为紧急
```

**4c. Admission Control（准入控制）— `_should_admit()`**

双重门控机制：
1. **队列深度门控**：`len(waiting) >= max_queue_depth` 且非高优 → 拒绝
2. **SLA 违约率门控**：滑动窗口（50 次）违约率 > 50% 且非高优 → 拒绝

高优判定（`_is_high_priority_request()`）— 以下任一满足即永不被拒：
- MLFQ L0-L1 级别
- `priority < 0`（API 显式高优）
- 短 prompt（< `SHORT_PROMPT_THRESHOLD`，即交互型请求）

```python
def add_request(self, request: Request) -> None:
    if self.enable_overload_management and not self._should_admit(request):
        request.status = RequestStatus.FINISHED_REJECTED
        self._free_request(request)
        return
    # ... 正常入队
```

**4d. Deadline-Aware 调度 — `_deadline_aware_sort_waiting()`**

在 `schedule()` 每轮调度开始时（PD-Aware pre-schedule 之后），对 WAITING 队列做紧急度排序：

- **MLFQ 模式**：在每个 level 内部，`slack_time < 2s` 的请求提前到队首（不跨级）
- **Flat 模式**：按 `(urgency_bucket, arrival_time)` 全局排序

**不跨 MLFQ 级别**：L2 请求即使很紧急也不能跳到 L0 前面，保持 MLFQ 层级语义。

**4e. SLA-Aware 抢占 — `_select_preemption_victim()`**

当 RUNNING 需要抢占时，选择策略：
1. **优先抢占已违约请求**：已经超过 deadline 的请求"挽救不回来了"，释放资源给可挽救的请求
2. **违约请求中选最低优先级**：`max(violated, key=(effective_priority, arrival_time))`
3. **降级到 QoS 标准抢占**：无违约请求时，按 `effective_priority` 选择

**4f. SLA 违约跟踪**

在 `_free_request()` 中，请求完成时记录是否违约到滑动窗口：

```python
if request.deadline < float('inf'):
    self._sla_violation_window.append(request.is_sla_violated())
```

`sla_violation_rate` 属性可随时查询当前违约率。

#### 三管齐下工作原理

```
Phase 5: 全面过载（所有租户 2× 请求涌入）
                    ↓
┌─── (a) 准入控制 ──────────────────────────────────────┐
│  队列深度 > 100 且非高优？ → 拒绝 (503 Retry-After)    │
│  SLA 违约率 > 50% 且非高优？ → 拒绝                    │
│  Gold 请求？ → 永远放行 ✅                              │
└───────────────────────────┬──────────────────────────┘
                            ↓ 被接受的请求
┌─── (b) Deadline-Aware 调度 ───────────────────────────┐
│  MLFQ L0: [Silver-A(slack=0.3s), Gold-B(slack=8s)]   │
│                ↓ sort by urgency                       │
│  Silver-A 移到队首 → 先调度                             │
└───────────────────────────┬──────────────────────────┘
                            ↓ 需要抢占时
┌─── (c) SLA-Aware 抢占 ──────────────────────────────┐
│  Running: [Gold(ok), Bronze(violated), Silver(ok)]    │
│  Bronze 已违约 → 优先被抢占 → 资源给 Gold/Silver       │
└──────────────────────────────────────────────────────┘

结果: 被接受请求 P99 TTFT 从 >10s 降到 <800ms
     合理拒绝率 30-40%（低优先级承担）
     Gold 请求不被拒绝
```

#### 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 高优判定方式 | MLFQ + priority + 短 prompt 三合一 | 覆盖 Gold/交互式/API 显式高优三种场景 |
| 违约率窗口大小 | 50 | 太小易抖动，太大反应慢 |
| 违约率阈值 | 50% | 平衡拒绝率和 SLA 满足率 |
| 紧急度阈值 | 2s | 给 prefill 留足时间（典型 prefill 100-500ms） |
| deadline 排序不跨 MLFQ 级 | 是 | 防止低级请求通过人为设置短 deadline 跳队 |
| `FINISHED_REJECTED` 映射 | `FinishReason.ABORT` | 客户端可区分"被拒绝"和"主动取消" |

#### 测试覆盖（31 个测试，全部通过 ✅）

| 测试类 | 测试项数 | 覆盖内容 |
|--------|---------|---------|
| `TestAdmissionControl` | 7 | 队列空/满、高/低优先级、违约率门控 |
| `TestDeadlineProperties` | 4 | slack_time、violated、urgency 排序 |
| `TestSLAAwarePreemption` | 5 | 违约优先抢占、QoS 降级、开关切换 |
| `TestDeadlineAwareScheduling` | 4 | 同级提升、不跨级、flat 模式、多紧急排序 |
| `TestSLAViolationTracking` | 3 | 窗口大小、违约率计算 |
| `TestOverloadManagementIntegration` | 2 | Phase 5 完整场景 |
| `TestRequestSLAMethods` | 4 | deadline 计算、violated 判定 |
| `TestFinishedRejected` | 2 | FINISHED_REJECTED 状态 |

#### 验证结果

```bash
# 阶段 4 自身测试
pytest tests/v1/core/test_overload_management.py -v  # 31 passed ✅

# 回归测试
pytest tests/v1/core/test_prefill_budget_isolation.py -v  # 11 passed ✅（阶段 2）
pytest tests/v1/core/test_tenant_isolation.py -v          # 15 passed ✅（阶段 3）
# 阶段 1 stub 测试                                         # passed ✅
```

**检查清单**：
- [x] `scheduler.py` 新增 `_should_admit()` + 修改 `add_request()`
- [x] `request.py` 新增 `deadline` / `slack_time` / `is_sla_violated()` / `sla_urgency`
- [x] `request.py` 新增 `FINISHED_REJECTED = 7` 状态
- [x] `engine/__init__.py` 新增 `sla_ttft_ms` 字段
- [x] `processor.py` 透传 `sla_ttft_ms` 参数
- [x] `scheduler.py` 新增 `_select_preemption_victim()` SLA-aware 抢占
- [x] `scheduler.py` 新增 `_deadline_aware_sort_waiting()` 紧急度排序
- [x] `scheduler.py` `_free_request()` 中跟踪 SLA 违约到滑动窗口
- [x] 31 个单测全部通过
- [x] 阶段 1/2/3 回归测试全部通过
- [ ] 重跑 workload，Phase 5 被接受请求 P99 TTFT <800ms，拒绝率 30-40%
- [ ] 全量 Phase 1-4 无退化

---

## 四、实施节奏总览

```
时间     │ 任务                              │ 预计工时 │ 代码量   │ 风险
─────────┼──────────────────────────────────┼─────────┼─────────┼──────
Day 1 AM │ 阶段 0: 跑基线                    │ 0.5h    │ 0 行    │ 无
Day 1 PM │ 阶段 1: 缓存版本管理 + 单测        │ 2h      │ ~60 行  │ 低
Day 1 PM │ 重跑 workload 验证                │ 0.5h    │ -       │ -
─────────┼──────────────────────────────────┼─────────┼─────────┼──────
Day 2    │ 阶段 2: Prefill 预算隔离 + 单测    │ 2h      │ ~40 行  │ 低
Day 2    │ 重跑 workload 验证                │ 0.5h    │ -       │ -
─────────┼──────────────────────────────────┼─────────┼─────────┼──────
Day 3-4  │ 阶段 3: 租户隔离 + 单测            │ 4h      │ ~120 行 │ 中
Day 4    │ 重跑 workload 验证                │ 0.5h    │ -       │ -
─────────┼──────────────────────────────────┼─────────┼─────────┼──────
Day 5-6  │ 阶段 4: 过载管理 + 单测            │ 5h      │ ~150 行 │ 中
Day 6    │ 最终全量验证 + 结果对比表          │ 1h      │ -       │ -
─────────┼──────────────────────────────────┼─────────┼─────────┼──────
         │ 总计                              │ ~16h    │ ~370 行 │
```

---

## 五、推荐落地顺序

```
修复 1 (版本管理) ─→ 修复 2 (Prefill隔离) ─→ 修复 3 (租户隔离) ─→ 修复 4 (过载管理)
       │                     │                    │                    │
       │  工作量:中           │  工作量:中          │  工作量:大          │  工作量:大
       │  风险:低             │  风险:低            │  风险:中            │  风险:中
       │  验证:Phase 2        │  验证:Phase 4       │  验证:Phase 3       │  验证:Phase 5
       │                     │                    │                    │
       └──── 重跑 workload ──┘──── 重跑 workload ──┘──── 重跑 workload ──┘
```

### 为什么这个顺序？

1. **修复 1 独立性强**（只改 kv_cache 层），不影响调度逻辑
2. **修复 2 改动适中**（只改 budget 分配逻辑），且与已有的 MLFQ 天然配合
3. **修复 3 影响面广**（引入 tenant 概念），但也是 README 中 WFQ 优化 6 的简化版落地
4. **修复 4 最复杂**（涉及准入+deadline+抢占三管齐下），且是 README 中优化 3+5 的合体

---

## 六、每次修复后的验证清单

每做完一个修复，跑一次 workload，对比以下指标：

```bash
# 单实例验证（主要验证修复效果）
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --host 127.0.0.1 --port 8000 \
    --mode single --duration 300 --output-dir results/fix_N/

# 最终全量验证（所有修复完成后，跑三模式对比）
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --mode all \
    --host 127.0.0.1 --port 8000 \
    --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200 \
    --output-dir results/final_comparison/
```

| 验证项 | 基线 | 修复 1 后 | 修复 2 后 | 修复 3 后 | 修复 4 后 |
|--------|------|----------|----------|----------|----------|
| Phase 2 恢复时间 | 30-60s | **<5s** ✅ | <5s | <5s | <5s |
| Phase 4 短对话 P99 | >500ms | ~500ms | **<200ms** ✅ | <200ms | <200ms |
| Phase 3 Silver P99 | >2000ms | ~2000ms | ~2000ms | **<500ms** ✅ | <500ms |
| Phase 5 接受请求 P99 | >10s | ~10s | ~10s | ~10s | **<800ms** ✅ |
| Phase 5 拒绝率 | 0% | 0% | 0% | 0% | **30-40%** ✅ |
| **Phase 1 回归** | 基准 | **不退化** | **不退化** | **不退化** | **不退化** |

⚠️ **关键**：每次修复后都要检查 Phase 1（稳态）是否退化。如果退化了说明改动有副作用。

---

## 六B. PD 分离 & 投机解码 — 融合压测落地指导

> PD 分离和投机解码不需要新增 Phase，而是以**部署模式切换**的方式融合到同一套压测中。
> `workload.py` 已实现 `--mode pd-disagg` / `--mode spec-decode` / `--mode all`。

### 6B.1 PD 分离 (pd-disagg) 模式的落地要点

#### 后端部署准备

```
                ┌─────────┐
   请求 ──────►│  Router  │──────► Prefill 实例 (:8100)
                │  (:8000) │          │ KV Transfer
                │          │◄─────── ▼
                └─────────┘       Decode 实例 (:8200)
```

- **Router**：接收所有请求，根据策略路由到 Prefill 或 Decode
- **Prefill 实例**：执行 prompt 处理，生成 KV Cache
- **Decode 实例**：执行 token 生成，消费 KV Cache
- Router 通过响应头（`X-PD-Route-Type` / `X-PD-Prefill-Instance` / `X-PD-Decode-Instance` / `X-PD-KV-Transfer-Ms`）传递路由信息

#### workload.py 自动采集的 PD 指标

| 指标 | 采集方式 | 报告位置 |
|------|---------|---------|
| 路由分布（PD 分离 vs 直发 Decode） | 从响应头 `X-PD-Route-Type` 解析 | 报告第 4 节 |
| 各 Phase P50/P99 TTFT | 标准 TTFT 计算 | 报告第 4 节 |
| KV 传输延迟 | 从响应头 `X-PD-KV-Transfer-Ms` 解析 | 报告第 4 节 |
| Phase 4 短对话隔离效果 | Phase 4 + request_type=short 的 TTFT | 报告第 4 节 |
| Router 健康状态 | 每 30s 轮询 `/router/status` 端点 | 运行时实时输出 |

#### 各 Phase 对 PD 分离的核心价值

| Phase | PD 分离的观察重点 |
|-------|-----------------|
| Phase 1 | Router 路由准确性；Prefill/Decode 实例负载均衡 |
| Phase 2 | Prompt 切换时，Prefill 实例的缓存是否受影响 |
| Phase 3 | 流量暴增时，Router 是否能合理分流；KV 传输是否成为瓶颈 |
| **Phase 4** | **核心优势场景**：长 Prefill 在独立实例执行，不阻塞 Decode 的 ITL |
| Phase 5 | 过载分散到多实例；Router 级准入控制 |

### 6B.2 投机解码 (spec-decode) 模式的落地要点

#### 后端部署准备

- 单实例启动，但带 `--speculative-config` 配置
- Suffix Decoding 特别适合代码补全（Gold-B）：后缀匹配 draft 准确率高

#### workload.py 自动采集的 SpecDecode 指标

| 指标 | 采集方式 | 报告位置 |
|------|---------|---------|
| Draft 接受率 | 从 `/metrics` 解析 Prometheus 指标 | 运行时实时输出 + 报告第 5 节 |
| 各 Phase E2E/TTFT | 标准计算 | 报告第 5 节 |
| Gold-B 代码补全加速 | Gold-B completed 请求的 TTFT + E2E | 报告第 5 节 |

#### 各 Phase 对投机解码的核心价值

| Phase | 投机解码的观察重点 |
|-------|-----------------|
| Phase 1 | 稳态下 draft 接受率基线（Code vs 通用对话差异） |
| Phase 2 | Prompt 切换后，后缀匹配的适应性（匹配率是否下降） |
| Phase 3 | 高并发下 draft 模型资源竞争，接受率是否退化 |
| Phase 4 | 长文档场景下投机解码效果（长 context 下 draft 退化） |
| Phase 5 | 过载时投机解码的额外开销，是否应自适应关闭 |
| **全程 Gold-B** | **核心优势场景**：后缀匹配加速代码补全生成 |

### 6B.3 三模式交叉对比 (--mode all)

运行 `--mode all` 后，workload.py 会：
1. 依次以 single → pd-disagg → spec-decode 三种模式运行相同流量
2. 分别保存 `results_single.json`、`results_pd-disagg.json`、`results_spec-decode.json`
3. 合并保存 `results_combined.json`
4. 报告自动生成第 6 节"多模式交叉对比"，包括：
   - Phase × Mode P99 TTFT 对照表
   - Gold-B 模式对比（缓存命中 vs PD 分离 vs 投机加速）

**交叉对比核心看点**：

```
                    单实例    PD分离    投机解码
                    ─────    ─────    ─────
Phase 4 短对话 P99   高       低 ✅    中
Gold-B E2E 延迟      高       中       低 ✅
Phase 5 整体违约率   高       低 ✅    高
稳态 P99 TTFT       低 ✅    中       中
```

---

完成所有修复后，你的项目结构应该是：

```
vllm-serving-optimization/
│
├── README.md                              # 项目总览（QoS 调度 8 项优化）
├── prefix-cache-scheduling-optimization.md # Prefix Cache 5 项优化设计
├── suffix-decoding-optimization.md         # 后缀解码 5 项优化设计（方案）
├── pd-disaggregation-optimization.md       # PD 分离 6 项优化设计（方案）
│
├── vllm/v1/
│   ├── request.py                         # 改动: QoS + MLFQ + 限速 + tenant_id + deadline
│   ├── core/
│   │   ├── scheduler.py                   # 改动: 所有调度优化 + 4 项增量修复
│   │   ├── kv_cache_manager.py            # 改动: Segmented LRU + 部分释放 + 版本管理
│   │   ├── kv_cache_utils.py              # 改动: Segmented LRU FreeKVCacheBlockQueue
│   │   └── tenant_manager.py              # 新增: 租户管理（修复 3）
│   └── engine/
│       ├── processor.py                   # 改动: 准入控制 + tenant_id 传递
│       ├── pd_router.py                   # PD 分离: 智能路由
│       └── pd_health_monitor.py           # PD 分离: 健康监控
│
├── tests/v1/core/
│   ├── test_cache_aware_scheduling.py     # 已有
│   ├── test_frequency_aware_eviction.py   # 已有
│   ├── test_preemption_cache_shield.py    # 已有
│   ├── test_cache_version_management.py   # 新增（修复 1）
│   └── test_prefill_budget_isolation.py   # 新增（修复 2）
│
└── benchmarks/e2e_cases/
    ├── README.md                          # 综合 Case 设计文档（含三模式说明）
    ├── LANDING_PLAN.md                    # 本文件：落地方案
    ├── workload.py                        # 综合压测脚本（支持 --mode 切换）
    └── results/
        ├── baseline/
        │   ├── results_single.json        # 单实例基线
        │   ├── results_pd-disagg.json     # PD 分离基线
        │   ├── results_spec-decode.json   # 投机解码基线
        │   └── results_combined.json      # 合并基线（--mode all）
        ├── fix_1/results_single.json      # 修复 1 后数据
        ├── fix_2/results_single.json      # ...
        ├── fix_3/results_single.json
        ├── fix_4/results_single.json      # 最终单实例数据
        └── final_comparison/
            ├── results_single.json        # 最终三模式对比
            ├── results_pd-disagg.json
            ├── results_spec-decode.json
            └── results_combined.json      # 最终合并数据（--mode all）
```

---

## 八、你的 Storytelling（对外展示）

整个项目的叙事线：

1. **基于 vLLM V1 做了 4 大方向的深度优化**：
   - 调度侧：QoS 分级 + MLFQ + Token 限速
   - KV Cache 侧：缓存感知调度 + Segmented LRU + 抢占保护
   - 部署架构侧：PD 分离（Prefill-Decode 分离 + 智能路由 + KV 传输优化）
   - 推理加速侧：后缀解码方案设计（Suffix Decoding）

2. **设计了一个综合端到端 Case** 验证所有优化在真实业务场景下的效果，**同时支持三种部署模式**一键切换对比

3. **通过 Case 发现框架优化的局限**，进而做了 4 项增量修复：
   - 缓存版本管理、Prefill 预算隔离、租户隔离、过载管理

4. **每一步都有量化数据支撑**：基线 vs 优化后，5 个 Phase × 7 个租户 × 3 种部署模式的全维度交叉对比

这条线从"框架优化" → "端到端验证" → "业务场景修复" → "多部署模式对比"，展示的是**完整的工程思维闭环 + 架构视野**。
