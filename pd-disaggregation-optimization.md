# 基于 vLLM V1 的 Prefill-Decode 分离（PD Disaggregation）优化

> 将 vLLM V0 的 PD 分离能力迁移到 V1 架构，并在调度感知、智能路由、KV 传输效率、Prefix Cache 协同等层面做深度优化

## 一、项目背景与动机

### 1.1 为什么需要 PD 分离

大模型推理包含两个计算特性截然不同的阶段：

| 阶段 | 计算特性 | 瓶颈 | GPU 利用模式 |
|------|---------|------|-------------|
| **Prefill** | 一次性处理整个 prompt，大量矩阵乘法 | **计算密集型（Compute-bound）** | GPU 算力拉满，显存带宽相对空闲 |
| **Decode** | 逐 token 自回归生成，每步只算 1 token | **访存密集型（Memory-bandwidth-bound）** | GPU 算力大量闲置，显存带宽拉满（读 KV Cache） |

**混合部署的问题**：当 Prefill 和 Decode 混合在同一实例上运行时：
- Prefill 任务的大量计算会**插入 Decode 批次**，导致 Decode 的 ITL（Inter-Token Latency）出现尖峰
- Chunked Prefill 可以缓解，但需要精确调参 `chunk_size`，且无法彻底消除 Prefill 干扰
- **无法独立调优 TTFT 和 ITL**：提升 TTFT 需要更大的 TP，但这会浪费 Decode 阶段的 GPU 资源

**PD 分离的核心价值**：
1. **独立调优 TTFT 和 ITL**：为 Prefill 和 Decode 分配不同的并行策略（如 Prefill 用 TP=4，Decode 用 TP=1）
2. **控制尾部 ITL**：Decode 实例上不运行任何 Prefill 任务，ITL 完全稳定
3. **异构硬件适配**：Prefill 用计算卡（A100），Decode 用推理卡（L40S/T4），成本更优

> ⚠️ 注意：PD 分离**不提升吞吐量**（总计算量不变），其核心价值在于**时延控制和资源效率**。

### 1.2 vLLM 现有 PD 分离能力（v0.7.3 源码分析）

通过深入阅读 vLLM 源码，现有 PD 分离功能的完整画像如下：

#### 1.2.1 三层抽象架构

```
vllm/distributed/kv_transfer/
├── kv_pipe/                          # L1: 传输管道层
│   ├── base.py                       # KVPipeBase 抽象基类
│   ├── pynccl_pipe.py                # PyNccl 实现（NCCL GPU 直传）
│   └── mooncake_pipe.py              # Mooncake 实现（RDMA/TCP Transfer Engine）
├── kv_lookup_buffer/                 # L2: 查找缓冲层
│   ├── base.py                       # KVLookupBufferBase 抽象基类
│   └── simple_buffer.py              # SimpleBuffer 实现（deque + 前缀匹配）
├── kv_connector/                     # L3: 连接器层
│   ├── base.py                       # KVConnectorBase 抽象基类
│   ├── factory.py                    # 工厂 + 注册机制
│   └── simple_connector.py           # SimpleConnector（组合 Pipe + Buffer）
└── kv_transfer_agent.py              # 入口代理（shim wrapper）
```

**三层职责**：

| 层级 | 核心 API | 职责 |
|------|---------|------|
| **KVPipe** | `send_tensor()` / `recv_tensor()` | 单向 FIFO 管道，传输 `torch.Tensor` |
| **KVLookupBuffer** | `insert()` / `drop_select()` | 键值查找缓冲（类 SQL 语义），解决乱序问题 |
| **KVConnector** | `send_kv_caches_and_hidden_states()` / `recv_kv_caches_and_hidden_states()` | 与 vLLM ModelRunner 集成 |

**为什么需要 LookupBuffer**：Prefill 实例可能按 A→B→C 顺序处理请求，但 Decode 实例可能先处理 C。FIFO 管道无法处理乱序，所以需要 LookupBuffer 做 token 前缀匹配，找到正确的 KV Cache。

#### 1.2.2 现有实现的关键特征

通过源码分析（`simple_connector.py`, `simple_buffer.py`, `pynccl_pipe.py`）：

1. **仅支持 v0 引擎**：在 `vllm/v1/` 中搜索 `disagg`、`kv_transfer`、`kv_connector` — **零匹配**，V1 完全不支持 PD 分离
2. **仅支持 1P1D**：`KVTransferConfig` 注释明确 "Currently only 1P1D is supported"（1 Prefill + 1 Decode）
3. **两种传输后端**：
   - `PyNcclConnector` → PyNccl NCCL 通信（需要 `kv_parallel_size=2`）
   - `MooncakeConnector` → Mooncake Transfer Engine（RDMA/TCP）
4. **路由层是外部的**：通过简单的 HTTP Proxy（`disagg_prefill_proxy_server.py`，Quart 实现）转发请求
5. **Buffer 匹配是 O(n)**：`SimpleBuffer._matches()` 通过 token 前缀匹配，遍历 buffer 逐个比较
6. **假设所有请求都是 Prefill**：代码中有多个 `FIXME: This assumes all requests are prefill`

#### 1.2.3 集成点（ModelRunner v0）

```python
# vllm/worker/model_runner.py

# 在模型前向计算之前，尝试接收 KV Cache
if self.need_recv_kv(model_input, kv_caches):
    hidden_states, bypass_model_exec, model_input = \
        get_kv_transfer_group().recv_kv_caches_and_hidden_states(
            model_executable, model_input, kv_caches)

# 如果成功接收所有 KV → bypass_model_exec=True → 跳过前向计算
# 如果有任何 KV 缺失 → bypass_model_exec=False → 回退到正常 Prefill

# 在模型前向计算之后，发送 KV Cache
if self.need_send_kv(model_input, kv_caches):
    get_kv_transfer_group().send_kv_caches_and_hidden_states(
        model_executable, model_input, kv_caches, hidden_states)
```

判断条件：
- `need_recv_kv` → `is_kv_consumer` + 非 profiling + 是 prefill
- `need_send_kv` → `is_kv_producer` + 非 profiling + 是 prefill

#### 1.2.4 现有代理（Proxy）工作流

```python
# benchmarks/disagg_benchmarks/disagg_prefill_proxy_server.py

@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    # 1. 复制请求，设置 max_tokens=1，发给 Prefill 实例 (port 8100)
    prefill_request['max_tokens'] = 1
    async for _ in forward_request('http://localhost:8100/v1/completions', prefill_request):
        continue
    # 2. Prefill 完成后（KV Cache 已传输），将原始请求转发给 Decode 实例 (port 8200)
    generator = forward_request('http://localhost:8200/v1/completions', original_request_data)
    response = await make_response(generator)
    return response
```

### 1.3 现有实现的局限性总结

| 局限性 | 具体表现 | 影响 |
|--------|---------|------|
| **V1 不支持** | `vllm/v1/` 中零 PD 相关代码 | 无法利用 V1 的多进程架构和改进的调度器 |
| **仅 1P1D** | 不支持 NP:MD 弹性部署 | 无法根据负载动态调整 P:D 比例 |
| **路由无智能** | 简单 HTTP 代理，先 Prefill 再 Decode | 无负载感知、无请求分类、无故障转移 |
| **调度器无感知** | Scheduler 不知道 PD 分离的存在 | 无法根据 KV 传输状态优化调度决策 |
| **Buffer O(n) 匹配** | `SimpleBuffer` 逐个遍历比较 token | 高并发下成为性能瓶颈 |
| **全量传输** | 每个请求的完整 KV Cache 都要传输 | 不利用 Prefix Cache，重复传输共享前缀 |
| **无 Prefix Cache 协同** | Prefill 端的缓存对 Decode 端不可见 | 缓存命中率优化无法跨实例生效 |
| **无背压联动** | Prefill 端不感知 Decode 端的负载 | 可能导致 Decode 端过载 |

### 1.4 优化主题定位

本项目聚焦于 **将 PD 分离能力迁移到 vLLM V1 架构，并在迁移过程中解决上述局限性**。通过 6 个递进的优化点，从 V1 基础适配、智能路由、调度器感知、传输优化、Prefix Cache 协同、到多实例协调，形成一套完整的 PD 分离优化方案。

```
优化路径：

  优化 1: V1 引擎 PD 基础适配
    │   （让 V1 的 GPUModelRunner 支持 KV 收发）
    ▼
  优化 2: 智能请求路由/代理
    │   （负载感知、请求分类、故障转移）
    ▼
  优化 3: 调度器 PD 感知
    │   （Scheduler 感知 KV 传输状态和 P/D 角色）
    ▼
  优化 4: KV Cache 传输优化
    │   （增量传输、压缩、流水线化）
    ▼
  优化 5: Prefix Cache 与 PD 分离协同
    │   （跨实例缓存共享、避免重复传输）
    ▼
  优化 6: 多实例协调与可观测性
      （NP:MD 弹性部署、全链路指标）
```

---

## 二、优化点详细设计

### 优化 1：V1 引擎 PD 基础适配 `[P0]` `[已实现]`

> **目标**：让 vLLM V1 的 GPUModelRunner 和 Scheduler 支持 KV Cache 的发送和接收，实现最基本的 1P1D 功能

#### 为什么需要数据传输

PD 分离的部署架构中，Producer（Prefill 实例）和 Consumer（Decode 实例）运行在**不同的 GPU** 上（可能甚至在不同的物理机器上）。每个 GPU 有自己独立的显存，**KV Cache 是存在 GPU 显存里的 tensor，两块 GPU 的显存互相看不到**。

```
┌──────────────────────────┐      ┌──────────────────────────┐
│   机器 A（或 GPU 0）       │      │   机器 B（或 GPU 1）       │
│                          │      │                          │
│   Producer 进程           │      │   Consumer 进程           │
│   ├── 自己的 vLLM Engine  │      │   ├── 自己的 vLLM Engine  │
│   ├── 自己的 Model        │      │   ├── 自己的 Model        │
│   ├── 自己的 GPU 显存      │      │   ├── 自己的 GPU 显存      │
│   └── 自己的 KV Cache     │      │   └── 自己的 KV Cache     │
│       ┌─────────────┐    │      │       ┌─────────────┐    │
│       │ Block 0: ██ │    │      │       │ Block 0: 空 │    │
│       │ Block 1: ██ │    │      │       │ Block 1: 空 │    │
│       │ Block 2: ██ │    │      │       │ Block 2: 空 │    │
│       └─────────────┘    │      │       └─────────────┘    │
└──────────────────────────┘      └──────────────────────────┘
         两块 GPU 的显存是完全隔离的
```

如果不传输 KV，Consumer 必须自己重新跑一遍 Prefill，完全失去了 PD 分离的意义。底层传输通过 **NCCL**（NVLink/PCIe 直连）或 **RDMA**（跨机器网络）实现。

#### PD 分离的并行原理

**单个请求内部**：必须先 Prefill 再 Decode，严格串行（因果依赖——Decode 需要 Prefill 产生的 KV Cache）。

**多个请求之间**：这才是 PD 分离真正的价值所在。

没有 PD 分离（单 GPU 混合部署）：
```
GPU:  [A-Prefill][A-Decode][A-Decode]...[B-Prefill][B-Decode][B-Decode]...[C-Prefill]...
       ████████  █ █ █ █              ████████  █ █ █ █              ████████
问题：B 排队等 A 做完 prefill，A 的 decode 又被 B 的 prefill 挤占
```

有 PD 分离（两块 GPU 各司其职）：
```
Producer GPU:  [A-Prefill] [B-Prefill] [C-Prefill] [D-Prefill]
                ████████    ████████    ████████    ████████
                   │           │           │           │
                   │ 传KV      │ 传KV      │ 传KV      │ 传KV
                   ▼           ▼           ▼           ▼
Consumer GPU:     [A-Dec][A-Dec][A-Dec][B-Dec][B-Dec][C-Dec][D-Dec]...
                   █ █ █  █ █ █  █ █ █  █ █ █  █ █ █  █ █ █  █ █ █
核心收益：A 在 decode 的同时，B 在 prefill，互不干扰
```

#### vLLM V1 现状分析

- V1 使用全新的 `GPUModelRunner`（`vllm/v1/worker/gpu_model_runner.py`），与 V0 的 `ModelRunner` 完全不同
- V1 的 KV Cache 管理使用 `KVCacheManager`（`vllm/v1/core/kv_cache_manager.py`），基于 hash chain 的 Prefix Caching
- V1 的 Scheduler（`vllm/v1/core/scheduler.py`）是 two-phase 调度（不再有 SWAPPED 状态）
- V1 使用 ZMQ IPC 多进程架构，Engine 和 Worker 分离

#### 实现方案

**修改/新增的文件**：

| 文件 | 类型 | 说明 |
|------|------|------|
| `vllm/distributed/kv_transfer/kv_connector/v1_connector.py` | **新建** | V1 兼容的 KV Connector（传输层） |
| `vllm/distributed/kv_transfer/kv_connector/factory.py` | 修改 | 注册 V1 Connector |
| `vllm/v1/worker/gpu_model_runner.py` | 修改 | `execute_model()` 中加入 KV send/recv 钩子 |
| `vllm/v1/core/scheduler.py` | 修改 | 接受 PD 配置，存储角色标志 |
| `vllm/v1/engine/core.py` | 修改 | 初始化 KV Transfer，prefill 完成标记 |
| `vllm/v1/core/kv_cache_manager.py` | 修改 | 新增 `register_received_blocks()` |
| `tests/v1/core/test_pd_disaggregation.py` | **新建** | 70 个测试（基于 AST 解析） |

#### 🔵 Producer 端（Prefill 实例）完整数据流

以一条请求 "Hello world tell me a story"（6 个 token）为例：

**第 1 步：EngineCore.step() — 驱动调度和执行**

```python
# vllm/v1/engine/core.py
def step(self) -> EngineCoreOutputs:
    scheduler_output = self.scheduler.schedule()          # 调度
    output = self.model_executor.execute_model(scheduler_output)  # 执行
    engine_core_outputs = self.scheduler.update_from_output(...)  # 更新

    # Producer 特有：prefill 完了就标记请求结束
    if self.is_kv_producer:
        self._finish_prefill_only_requests(scheduler_output)
    return engine_core_outputs
```

**第 2 步：GPUModelRunner.execute_model() — 正常跑 forward**

Producer 端 `is_kv_consumer = False`，不尝试接收 KV，直接跑模型 forward pass：

```python
# vllm/v1/worker/gpu_model_runner.py
def execute_model(self, scheduler_output):
    self._update_states(scheduler_output)

    # bypass_model_exec = False（Producer 不是 consumer，不接收 KV）
    # 正常走 forward
    with set_forward_context(attn_metadata, self.vllm_config):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.kv_caches,  # ← forward 完成后，KV Cache 被填充
            ...
        )
```

Forward 完成后，6 个 token 的 KV Cache 已经填入了 GPU 显存中的 paged KV cache。

**第 3 步：Forward 之后 → 发送 KV**

```python
    # Producer: forward 完了就发 KV
    if self.is_kv_producer and self.kv_connector is not None:
        self._send_kv_caches_for_producer(scheduler_output, hidden_states)
```

`_send_kv_caches_for_producer()` 的核心逻辑：

1. 遍历当前 batch 中的每个请求
2. 识别正在 prefill 的请求（`num_computed_tokens < num_tokens`）
3. 构建 token_ids tensor
4. 调用 `kv_connector.send_kv_caches_v1()`

**第 4 步：V1KVConnector.send_kv_caches_v1() — 从 KV Cache 抽取数据并发送**

V1 的 KV Cache 是**分页存储**的，需要用 `block_ids` + `block_size` 算术来定位每个 token 的 KV：

```
KV Cache 布局: [2, num_blocks, block_size, num_kv_heads, head_size]
                ↑                  ↑
                K和V两部分         每个 block 存 block_size 个 token

假设 block_size=4，6 个 token 分布在 2 个物理 block 中：
Block 0: [token0, token1, token2, token3]   ← block_ids[0] = 7（物理编号）
Block 1: [token4, token5, _空_, _空_]       ← block_ids[1] = 15（物理编号）
```

抽取逻辑（对每一层 attention）：

```python
for token_idx in range(total_tokens):     # 0, 1, 2, 3, 4, 5
    block_idx = token_idx // block_size    # 0, 0, 0, 0, 1, 1  → 逻辑 block 序号
    block_offset = token_idx % block_size  # 0, 1, 2, 3, 0, 1  → block 内偏移
    block_id = block_ids[block_idx]        # 7, 7, 7, 7, 15, 15 → 物理 block 编号

    key = key_cache[block_id, block_offset]     # [num_kv_heads, head_size]
    value = value_cache[block_id, block_offset]  # [num_kv_heads, head_size]
```

所有层的 KV 拼成大 tensor 后，通过 `producer_buffer.insert()` → 底层 NCCL/RDMA 发送到网络。

**第 5 步：标记请求为已完成**

回到 EngineCore，`_finish_prefill_only_requests()` 标记请求为 `FINISHED_STOPPED`：

```python
def _finish_prefill_only_requests(self, scheduler_output):
    for req_id in list(scheduler_output.num_scheduled_tokens.keys()):
        request = self.scheduler.requests.get(req_id)
        # prefill 做完了 → 标记为 FINISHED_STOPPED
        if request.num_computed_tokens >= request.num_tokens:
            self.scheduler.finish_requests([req_id], RequestStatus.FINISHED_STOPPED)
```

**为什么要标记完成？** Producer 只做 Prefill，不做 Decode。如果不标记完成，请求会留在调度器里被安排去 decode 产生 token，但这不是 Producer 的职责。

#### 🟢 Consumer 端（Decode 实例）完整数据流

**第 1 步：同样进入 execute_model()**

**第 2 步：Forward 之前 → 尝试接收 KV**

```python
# vllm/v1/worker/gpu_model_runner.py
def execute_model(self, scheduler_output):
    self._update_states(scheduler_output)

    # Consumer: forward 前先试着接收 KV
    bypass_model_exec = False
    if self.is_kv_consumer and self.kv_connector is not None:
        bypass_model_exec, kv_recv_success_map = (
            self._recv_kv_caches_for_consumer(scheduler_output))
```

`_recv_kv_caches_for_consumer()` 的核心逻辑：

1. 遍历 `scheduled_new_reqs`（新到的请求）
2. 对每个请求，调用 `kv_connector.recv_kv_caches_v1()`
3. 如果收到完整 KV → 更新 `num_computed_tokens = num_tokens`（告诉系统"这些 token 已经算过了"）
4. 如果**所有请求**的 KV 都收到了 → `bypass_model_exec = True`

**第 3 步：V1KVConnector.recv_kv_caches_v1() — 接收并写入 KV Cache**

和 send 是镜像操作：

```python
# 1. 从 buffer 中按 token 前缀匹配取出 KV 数据
ret = self.consumer_buffer.drop_select(input_tokens, roi)

# 2. 收到了 → 写入本地的 paged KV cache（和 send 完全镜像）
for layer_id in range(num_layers):
    for token_idx in range(num_recv_tokens):
        block_idx = token_idx // block_size
        block_offset = token_idx % block_size
        block_id = block_ids[block_idx]

        key_cache[block_id, block_offset].copy_(layer_keys[token_idx])
        value_cache[block_id, block_offset].copy_(layer_values[token_idx])
```

**第 4 步：判断是否跳过 forward pass**

```python
    if not bypass_model_exec:
        # 有请求没收到 KV → 正常跑 forward
        hidden_states = self.model(...)
    else:
        # 全部 KV 都收到了 → 跳过 prefill 的 forward pass
        hidden_states = torch.zeros(num_input_tokens, self.hidden_size, ...)
```

**第 5 步：进入正常 Decode 循环**

之后 Consumer 正常进入 decode 循环，一个 token 一个 token 地生成。

#### 完整数据流总览

```
                    Producer 端                                 Consumer 端
                    ─────────                                   ─────────
1. 收到请求                                          1. 收到同一个请求
2. Scheduler.schedule()                              2. Scheduler.schedule()
3. execute_model():                                  3. execute_model():
   ├── _update_states()                                 ├── _update_states()
   ├── (不是consumer，跳过recv)                          ├── _recv_kv_caches_for_consumer()
   │                                                    │    └── kv_connector.recv_kv_caches_v1()
   │                                                    │         → 从网络接收 KV
   │                                                    │         → 写入本地 paged KV cache
   │                                                    │         → 返回 (success, num_tokens)
   │                                                    │    └── 更新 num_computed_tokens
   │                                                    ├── bypass_model_exec = True? 跳过 forward
   ├── model.forward()  ← 正常跑                        │
   │    → KV Cache 被填充                                ├── 直接进入 sampling/decode
   ├── _send_kv_caches_for_producer()                   │
   │    └── kv_connector.send_kv_caches_v1()            │
   │         → 从 KV cache 抽取数据                      │
   │         → 通过 NCCL/RDMA 发送                       │
   │                                                    │
4. _finish_prefill_only_requests()                   4. 正常 decode 生成 tokens
   → 标记请求 FINISHED_STOPPED
   → 请求不会进入 decode 循环
```

#### KVCacheManager.register_received_blocks — 前缀缓存一致性

V1 有**前缀缓存**（Prefix Caching）：如果两个请求有相同的 prompt 前缀，第二个请求可以复用第一个请求的 KV Cache。前缀缓存靠 **hash chain** 实现——把每个 block 的内容做 hash，存到 `cached_block_hash_to_block` 字典里。

**问题**：Consumer 从网络收到的 KV 数据直接写入了物理 block，但**没有注册 hash**。后续有相同前缀的请求时，前缀缓存会查不到（hash 表里没有）。

**解决方案**：`register_received_blocks()` 在 KV 写入后，用标准的 hash chain 算法补上 hash 注册：

```python
def register_received_blocks(self, request, num_received_tokens):
    # 用标准算法算出每个 block 的 hash
    block_hashes = hash_request_tokens(self.block_size, request)
    # 只注册满了的 block（不完整的 block 没法被复用）
    num_full_blocks = num_received_tokens // self.block_size
    for blk_idx in range(num_already_cached, num_full_blocks):
        block = req_blocks[blk_idx]
        block_hash = block_hashes[blk_idx]
        if block.block_hash is None:
            block.block_hash = block_hash
            self.cached_block_hash_to_block[block_hash][block.block_id] = block
```

这样后续有相同前缀的请求就能直接命中缓存，不需要再次从 Producer 传输。

#### 关键设计决策

1. **新建 V1KVConnector 而非修改 V0 的 SimpleConnector**：V1 使用完全不同的输入结构（`SchedulerOutput` 而非 `ModelInputForGPUWithSamplingMetadata`），需要独立的 Connector
2. **V0 抽象方法 raise NotImplementedError**：保持继承关系（工厂模式需要），但引导使用 V1 原生 API（`send_kv_caches_v1()` / `recv_kv_caches_v1()`）
3. **`bypass_model_exec` 模式**：复用 V0 的设计 —— 如果所有 KV 都从 Prefill 实例收到，直接跳过 forward pass
4. **完全向后兼容**：`kv_transfer_config=None`（默认值）时所有 PD 代码路径都不执行，不影响非 PD 场景

#### 关键挑战

- **V1 没有 `ModelInputForGPUWithSamplingMetadata`**：V0 的 KV Transfer API 依赖这个结构，V1 使用完全不同的输入格式（`SchedulerOutput`），需要重新设计 Connector 接口
- **V1 的 Prefix Caching 需要协调**：接收到的 KV blocks 需要正确注册 hash，否则会导致缓存不一致
- **V1 的 num_computed_tokens 机制**：V1 通过 `num_computed_tokens` 实现 Chunked Prefill 和缓存复用，PD 分离需要与此机制兼容

#### 预期效果

- V1 引擎支持基本的 1P1D PD 分离
- Prefill 实例可以将 KV Cache 传输到 Decode 实例
- Decode 实例成功接收 KV Cache 后跳过 Prefill 前向计算

---

### 优化 2：智能请求路由/代理 `[P0]` `[已实现]`

> **目标**：替换现有的简单 HTTP 代理，实现负载感知、请求分类、故障转移的智能路由

#### vLLM 现状分析

现有代理（`disagg_prefill_proxy_server.py`）的问题：
- **串行处理**：先发请求到 Prefill（等待完成），再转发到 Decode
- **无负载感知**：不知道 Prefill/Decode 实例的当前负载
- **无请求分类**：所有请求走相同的 Prefill→Decode 路径
- **无故障转移**：Prefill 实例宕机则全部失败
- **仅支持 Completions API**：不支持 Chat API

#### 实现方案

**新增文件**：
- `vllm/v1/engine/pd_router.py` — 智能路由器核心
- `vllm/v1/engine/pd_health_monitor.py` — 健康监测与负载采集

**核心逻辑**：

1. **负载感知路由**：

```python
class PDRouter:
    """智能 PD 路由器"""

    def __init__(self, prefill_endpoints: List[str], decode_endpoints: List[str]):
        self.prefill_pool = EndpointPool(prefill_endpoints)
        self.decode_pool = EndpointPool(decode_endpoints)
        self.health_monitor = HealthMonitor(self.prefill_pool, self.decode_pool)

    async def route_request(self, request: dict) -> AsyncGenerator:
        # 1. 请求分类
        request_type = self._classify_request(request)

        # 2. 选择策略
        if request_type == "short_decode_only":
            # 短请求（<128 tokens）直接发给 Decode 实例做 Prefill+Decode
            # 避免 KV 传输开销大于 Prefill 本身
            endpoint = self._select_decode_endpoint(request)
            async for chunk in self._forward(endpoint, request):
                yield chunk

        elif request_type == "long_prefill":
            # 长请求走完整 PD 分离路径
            prefill_ep = self._select_prefill_endpoint(request)
            decode_ep = self._select_decode_endpoint(request)
            await self._prefill_phase(prefill_ep, request)
            async for chunk in self._decode_phase(decode_ep, request):
                yield chunk

    def _classify_request(self, request: dict) -> str:
        """根据 prompt 长度判断是否值得走 PD 分离"""
        prompt_tokens = estimate_token_count(request["prompt"])
        if prompt_tokens < self.short_threshold:  # 默认 128
            return "short_decode_only"
        return "long_prefill"

    def _select_prefill_endpoint(self, request: dict) -> str:
        """选择负载最低的 Prefill 实例"""
        candidates = self.prefill_pool.get_healthy_endpoints()
        return min(candidates, key=lambda ep: ep.current_load)
```

2. **健康监测**：

```python
class HealthMonitor:
    """周期性采集 Prefill/Decode 实例的负载指标"""

    async def collect_metrics(self, endpoint: str) -> EndpointMetrics:
        # 调用 vLLM 的 /metrics 端点
        response = await self.session.get(f"{endpoint}/metrics")
        return EndpointMetrics(
            running_requests=...,
            waiting_requests=...,
            kv_cache_usage=...,
            gpu_utilization=...,
        )

    def is_healthy(self, endpoint: str) -> bool:
        metrics = self.latest_metrics[endpoint]
        return (
            metrics.response_time < self.timeout_threshold
            and metrics.error_rate < self.error_threshold
        )
```

3. **故障转移**：

```python
async def _prefill_with_fallback(self, request: dict):
    """Prefill 失败时自动回退"""
    for prefill_ep in self.prefill_pool.get_by_priority():
        try:
            await self._prefill_phase(prefill_ep, request)
            return
        except (ConnectionError, TimeoutError):
            self.health_monitor.mark_unhealthy(prefill_ep)
            continue

    # 所有 Prefill 实例都不可用，回退到 Decode 实例自行 Prefill
    logger.warning("All prefill instances unavailable, falling back to decode-side prefill")
    decode_ep = self._select_decode_endpoint(request)
    yield self._forward(decode_ep, request)
```

#### 预期效果

- 短请求（<128 tokens）绕过 PD 分离路径，避免不必要的 KV 传输开销
- 请求自动路由到负载最低的实例
- 实例故障时自动转移，不影响服务可用性
- 支持 NP:MD 的灵活部署（多个 Prefill + 多个 Decode）

#### 实际实现

##### 改动文件清单

| 文件 | 改动类型 | 说明 |
|------|----------|------|
| `vllm/v1/engine/pd_router.py` | 新建 | 智能路由器核心（~400行） |
| `vllm/v1/engine/pd_health_monitor.py` | 新建 | 健康监测与负载采集（~300行） |
| `tests/v1/engine/test_pd_router.py` | 新建 | 单元测试（~500行，30+ 测试用例） |

##### 架构设计

```
Client Request
       │
       ▼
┌──────────────────────────────────────────────────┐
│                   PDRouter                        │
│  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Request      │  │   HealthMonitor          │  │
│  │  Classifier   │  │  ┌──────────────────┐   │  │
│  │  ─────────    │  │  │ Prometheus /metrics│  │  │
│  │  short(<128)  │  │  │ 每5s采集一次      │   │  │
│  │  long(>=128)  │  │  └──────────────────┘   │  │
│  └──────┬───────┘  └──────────────────────────┘  │
│         │                                         │
│    ┌────┴────┐                                    │
│    │ short?  │                                    │
│    └────┬────┘                                    │
│    yes  │  no                                     │
│    │    │                                         │
│    │    ▼                                         │
│    │  ┌───────────────────────────┐               │
│    │  │ Prefill Phase (w/ retry)  │               │
│    │  │  select_prefill_endpoint()│               │
│    │  │  max_tokens=1, 等待完成   │               │
│    │  │  失败→换endpoint重试      │               │
│    │  │  全部失败→回退到Decode    │               │
│    │  └───────────┬───────────────┘               │
│    │              │                               │
│    ▼              ▼                               │
│  ┌────────────────────────────┐                   │
│  │ Decode Phase (streaming)   │                   │
│  │  select_decode_endpoint()  │                   │
│  │  流式返回给Client          │                   │
│  └────────────────────────────┘                   │
│                                                   │
│  EndpointPool(prefill)    EndpointPool(decode)    │
│  ├─ ep1: load=3, HEALTHY  ├─ ep1: load=5, HEALTHY│
│  ├─ ep2: load=8, HEALTHY  └─ ep2: load=2, HEALTHY│
│  └─ ep3: UNHEALTHY                               │
└──────────────────────────────────────────────────┘
```

##### 核心模块说明

**1. `pd_health_monitor.py` — 健康监测**

- **`EndpointMetrics`**：从 vLLM Prometheus `/metrics` 端点解析的指标数据
  - `running_requests`（`vllm:num_requests_running`）
  - `waiting_requests`（`vllm:num_requests_waiting`）
  - `gpu_cache_usage`（`vllm:gpu_cache_usage_perc`）
  - `load_score`：复合负载评分 = `running×1.0 + waiting×2.0 + gpu_cache×10.0`

- **`EndpointPool`**：管理同角色的一组端点
  - `get_least_loaded()` → 选择 load_score 最低的健康端点
  - `get_by_priority()` → 健康端点优先，按负载排序
  - `mark_healthy()` / `mark_unhealthy()` → 手动状态控制

- **`HealthMonitor`**：后台异步循环
  - 每5秒（可配置）对所有端点发 GET /metrics
  - 解析 Prometheus 文本格式，提取关键指标
  - 健康状态带滞后：连续3次失败→UNHEALTHY，连续2次成功→HEALTHY
  - 避免网络抖动导致的频繁状态切换

**2. `pd_router.py` — 智能路由**

- **请求分类**：
  - 估算 prompt token 数（字符数 / 4）
  - `< short_prompt_threshold`（默认128）→ 短请求，直发 Decode
  - `>= threshold` → 长请求，走 Prefill→Decode 分离路径
  - 支持 Completions API（`prompt` 字段）和 Chat API（`messages` 字段）

- **负载感知路由**：
  - `_select_prefill_endpoint()` / `_select_decode_endpoint()` 都选负载最低的
  - 基于 `EndpointMetrics.load_score` 排序

- **故障转移（`_prefill_with_fallback`）**：
  1. 按负载排序尝试 Prefill 端点
  2. 每次失败后 `mark_unhealthy()`，换下一个端点
  3. 最多重试 `max_retries` 次（默认2）
  4. 所有 Prefill 不可用 → 回退到 Decode 实例自行 Prefill+Decode

- **API 端点**：
  - `POST /v1/completions` — OpenAI Completions API
  - `POST /v1/chat/completions` — OpenAI Chat API
  - `GET /health` — 健康检查（healthy / degraded / unhealthy）
  - `GET /router/status` — 路由器状态与统计指标

##### 使用方式

```bash
# 启动 Prefill 实例（端口 8100/8101）
vllm serve <model> --port 8100 --kv-transfer-config '{"kv_role":"kv_producer",...}'
vllm serve <model> --port 8101 --kv-transfer-config '{"kv_role":"kv_producer",...}'

# 启动 Decode 实例（端口 8200/8201）
vllm serve <model> --port 8200 --kv-transfer-config '{"kv_role":"kv_consumer",...}'
vllm serve <model> --port 8201 --kv-transfer-config '{"kv_role":"kv_consumer",...}'

# 启动智能路由代理（2P2D 配置）
python -m vllm.v1.engine.pd_router \
    --prefill-endpoints http://localhost:8100 http://localhost:8101 \
    --decode-endpoints http://localhost:8200 http://localhost:8201 \
    --port 8000 \
    --short-prompt-threshold 128

# 客户端请求（和普通 OpenAI API 完全兼容）
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3-8B", "prompt": "Hello", "max_tokens": 100}'
```

##### 与优化1的关系

优化2（路由代理）工作在 **HTTP 层**，与优化1（V1引擎内部PD适配）是正交的：
- 优化1处理单个 vLLM 实例内部的 KV Cache 传输（Producer→Consumer）
- 优化2处理多个 vLLM 实例之间的请求调度（哪个请求发给哪个实例）
- 两者配合工作：优化2负责"宏观调度"，优化1负责"微观传输"

---

### 优化 3：调度器 PD 感知 `[P1]` `[未实现]`

> **目标**：让 V1 调度器感知 PD 分离角色，针对 Prefill-only 和 Decode-only 场景做专门优化

#### vLLM V1 现状分析

- V1 调度器（`scheduler.py`）将 Prefill 和 Decode 请求**混合调度**，使用统一的 `token_budget` 控制
- `max_num_batched_tokens` 在 Prefill 和 Decode 之间共享，Prefill 占用大量 budget 会挤压 Decode
- 调度器不知道当前实例是 Prefill-only 还是 Decode-only

#### 实现方案

**修改文件**：
- `vllm/v1/core/scheduler.py` — 根据 PD 角色调整调度策略

**核心逻辑**：

1. **Prefill-only 实例的调度优化**：

```python
class Scheduler:
    def _schedule_prefill_only(self) -> SchedulerOutput:
        """Prefill 实例的专属调度逻辑"""
        # Prefill 实例不需要处理 running 中的 Decode 请求
        # 可以将全部 token_budget 用于 Prefill

        # 1. 按 prompt 长度排序（短请求优先 → 更高并发度）
        sorted_waiting = sorted(
            self.waiting,
            key=lambda r: r.num_prompt_tokens
        )

        # 2. 贪心分配：尽可能多地并发 Prefill
        scheduled = []
        remaining_budget = self.max_num_batched_tokens
        for req in sorted_waiting:
            prefill_tokens = req.num_prompt_tokens - req.num_computed_tokens
            if prefill_tokens <= remaining_budget:
                scheduled.append(req)
                remaining_budget -= prefill_tokens

        # 3. Prefill 完成的请求不进入 running（发送 KV 后直接完成）
        return self._build_output(scheduled)
```

2. **Decode-only 实例的调度优化**：

```python
def _schedule_decode_only(self) -> SchedulerOutput:
    """Decode 实例的专属调度逻辑"""
    # Decode 实例不运行 Prefill（KV 从 Prefill 实例接收）
    # 可以最大化 Decode 批次大小

    # 1. 新请求进入时，先检查 KV Cache 是否已到达
    for req in self.waiting:
        if self.kv_receiver.has_kv_for(req):
            # KV 已到达，可以直接加入 running（跳过 Prefill）
            req.num_computed_tokens = req.num_prompt_tokens
            self.running.add(req)
        else:
            # KV 未到达，暂时保持 waiting
            pass

    # 2. 全部 token_budget 分配给 Decode 步骤
    #    每个 running 请求每步生成 1 token
    decode_batch_size = min(len(self.running), self.max_num_batched_tokens)
    return self._build_decode_output(decode_batch_size)
```

3. **KV 到达事件驱动**：

```python
class KVReceiveMonitor:
    """监控 KV Cache 的到达状态"""

    def __init__(self, scheduler: Scheduler):
        self.pending_requests: Dict[str, Request] = {}
        self.scheduler = scheduler

    def register_pending(self, request: Request):
        """注册等待 KV 的请求"""
        self.pending_requests[request.request_id] = request

    def on_kv_received(self, request_id: str):
        """KV 到达回调 → 通知调度器"""
        if request_id in self.pending_requests:
            req = self.pending_requests.pop(request_id)
            req.num_computed_tokens = req.num_prompt_tokens
            self.scheduler.move_to_running(req)
```

#### 预期效果

- Prefill 实例：全部 token_budget 用于 Prefill，吞吐最大化
- Decode 实例：不被 Prefill 干扰，ITL 完全稳定
- KV 到达驱动调度，减少 Decode 端的等待时间

---

### 优化 4：KV Cache 传输优化 `[P1]` `[未实现]`

> **目标**：减少 KV Cache 在 Prefill→Decode 之间的传输延迟和带宽占用

#### vLLM 现状分析

- 当前 `SimpleConnector.send_kv_caches_and_hidden_states()` 传输**完整的 KV Cache**
- 每个请求的每一层的 key 和 value 都要通过 NCCL/Mooncake 传输
- 对于长 prompt（如 4096 tokens），KV Cache 大小可达数百 MB
- 传输是逐请求、逐层进行的，没有流水线化

#### 实现方案

**修改文件**：
- `vllm/distributed/kv_transfer/kv_connector/v1_connector.py` — 新增 V1 专用 Connector
- `vllm/distributed/kv_transfer/kv_pipe/pynccl_pipe.py` — 增强传输效率

**核心逻辑**：

1. **增量 KV 传输**（配合 Prefix Caching）：

```python
class V1IncrementalConnector(KVConnectorBase):
    """只传输未缓存的 KV blocks"""

    def send_kv_caches_incremental(self, request, kv_caches, computed_blocks):
        """
        Prefix Cache 命中的 blocks 不传输
        只传输新计算的 KV blocks
        """
        # computed_blocks = Prefix Cache 已命中的 blocks（两端都有）
        # new_blocks = 这次 Prefill 新计算的 blocks（只有 Prefill 端有）
        all_blocks = request.block_table
        new_blocks = [b for b in all_blocks if b not in computed_blocks]

        # 只发送 new_blocks 对应的 KV Cache
        for block_id in new_blocks:
            for layer_id in range(self.num_layers):
                key_data = kv_caches[layer_id][0][block_id]
                value_data = kv_caches[layer_id][1][block_id]
                self.pipe.send_tensor(key_data)
                self.pipe.send_tensor(value_data)

        # 发送 block 映射元信息（decode 端需要知道如何安放这些 blocks）
        self.pipe.send_tensor(torch.tensor(new_blocks))
```

2. **KV Cache 压缩传输**：

```python
class CompressedKVPipe(KVPipeBase):
    """支持 KV Cache 压缩的传输管道"""

    def send_tensor_compressed(self, tensor: torch.Tensor) -> None:
        # FP16 → FP8 量化压缩（减少 50% 传输量）
        if self.compression_enabled:
            compressed = self._quantize_to_fp8(tensor)
            metadata = {
                "original_dtype": tensor.dtype,
                "scale": self._compute_scale(tensor),
                "shape": tensor.shape,
            }
            self._send_metadata(metadata)
            self._send_impl(compressed)
        else:
            self.send_tensor(tensor)

    def recv_tensor_decompressed(self) -> torch.Tensor:
        metadata = self._recv_metadata()
        compressed = self._recv_impl()
        if "scale" in metadata:
            return self._dequantize_from_fp8(
                compressed, metadata["scale"], metadata["original_dtype"]
            )
        return compressed
```

3. **Layer-wise 流水线传输**：

```python
async def send_kv_pipelined(self, request, kv_caches):
    """逐层流水线传输：不等所有层都计算完再发送"""
    # Layer 0 计算完 → 立即开始传输 Layer 0 的 KV
    # Layer 1 计算完 → 立即开始传输 Layer 1 的 KV
    # ... 计算和传输重叠执行
    for layer_id in range(self.num_layers):
        # 异步发送当前层的 KV（不阻塞下一层的计算）
        asyncio.create_task(
            self._send_layer_kv(layer_id, kv_caches[layer_id])
        )
```

4. **高效 Buffer 查找**（替换 O(n) 匹配）：

```python
class HashedLookupBuffer(KVLookupBufferBase):
    """基于 hash 的 O(1) KV Cache 查找"""

    def __init__(self):
        self.buffer: Dict[int, List[torch.Tensor]] = {}  # token_hash → KV data

    def insert(self, input_tokens, roi, key, value, hidden):
        token_hash = self._compute_token_hash(input_tokens, roi)
        self.buffer[token_hash] = [input_tokens, roi, key, value, hidden]

    def drop_select(self, input_tokens, roi):
        token_hash = self._compute_token_hash(input_tokens, roi)
        if token_hash in self.buffer:
            return self.buffer.pop(token_hash)
        return [None] * 5

    def _compute_token_hash(self, tokens, roi):
        """复用 vLLM V1 的 hash chain 算法"""
        active_tokens = tokens[roi] if roi is not None else tokens
        return hash(active_tokens.cpu().numpy().tobytes())
```

#### 预期效果

- 增量传输：Prefix Cache 命中的部分不重复传输，传输量减少 50-80%（高缓存命中率场景）
- FP8 压缩：传输带宽减少 50%，精度损失可忽略
- 流水线传输：计算与传输重叠，总延迟 ≈ max(计算, 传输) 而非 计算 + 传输
- Hash 查找：Buffer 匹配从 O(n) 降为 O(1)

---

### 优化 5：Prefix Cache 与 PD 分离协同 `[P2]` `[未实现]`

> **目标**：让 Prefix Cache 的命中信息跨实例共享，避免重复传输已缓存的 KV blocks

#### 问题分析

在 PD 分离场景下，Prefix Cache 面临独特挑战：
- **Prefill 端缓存**：Prefill 实例计算后会缓存 KV blocks（Prefix Cache 写入）
- **Decode 端缓存**：Decode 实例接收到 KV 后也可以缓存（用于后续请求复用）
- **两端缓存不同步**：Prefill 端知道自己缓存了什么，但 Decode 端不知道；反之亦然
- **重复传输**：相同 System Prompt 的多个请求，每次都要传输完整 KV，即使 Decode 端已有缓存

#### 实现方案

**新增文件**：
- `vllm/v1/core/pd_cache_coordinator.py` — 跨实例缓存协调器

**核心逻辑**：

1. **缓存状态同步**：

```python
class PDCacheCoordinator:
    """协调 Prefill 和 Decode 实例之间的缓存状态"""

    def __init__(self):
        # Decode 端维护的已缓存 block hash 集合
        self.decode_cached_hashes: Set[int] = set()

    def compute_transfer_plan(self, request, prefill_blocks, decode_cached_hashes):
        """
        计算最优传输方案：
        - 两端都有缓存的 blocks → 不传输
        - 只有 Prefill 端有的 blocks → 传输
        """
        need_transfer = []
        skip_transfer = []

        for block in prefill_blocks:
            block_hash = self._get_block_hash(block)
            if block_hash in decode_cached_hashes:
                skip_transfer.append(block)  # Decode 端已有，不传输
            else:
                need_transfer.append(block)  # Decode 端没有，需要传输

        return TransferPlan(
            transfer_blocks=need_transfer,
            skip_blocks=skip_transfer,
            saved_bytes=len(skip_transfer) * self.block_size_bytes,
        )
```

2. **Decode 端缓存注册**：

```python
def on_kv_received(self, request, received_blocks):
    """Decode 端接收 KV 后，注册缓存 hash"""
    for block in received_blocks:
        block_hash = self._compute_block_hash(block.tokens)
        self.kv_cache_manager.register_cached_block(block_hash, block.block_id)
        self.decode_cached_hashes.add(block_hash)
```

3. **利用 V1 的 Hash Chain**：

```python
def leverage_v1_hash_chain(self, request_tokens):
    """
    复用 V1 的 hash chain 机制：
    - Prefill 端：hash_request_tokens() 计算 block hash
    - 传输时：将 hash 信息一起发送给 Decode 端
    - Decode 端：用收到的 hash 注册到 cached_block_hash_to_block
    """
    # 复用 kv_cache_utils.hash_request_tokens()
    block_hashes = hash_request_tokens(
        block_size=self.block_size,
        request=request,
    )
    # 将 hash 元信息与 KV 数据一起传输
    return block_hashes
```

#### 预期效果

- 相同 System Prompt 的请求只传输一次 KV，后续请求 Decode 端直接命中缓存
- 传输量大幅减少（System Prompt 通常占 prompt 的 80%+）
- 与优化 4（增量传输）叠加效果更好

---

### 优化 6：多实例协调与可观测性 `[P2]` `[未实现]`

> **目标**：支持 NP:MD 弹性部署，提供全链路性能指标

#### 实现方案

**新增文件**：
- `vllm/v1/engine/pd_orchestrator.py` — 多实例编排器
- `vllm/v1/engine/pd_metrics.py` — PD 分离指标采集

**核心逻辑**：

1. **NP:MD 弹性部署**：

```python
class PDOrchestrator:
    """管理多个 Prefill 和 Decode 实例"""

    def __init__(self, config: PDConfig):
        self.prefill_instances: List[InstanceInfo] = []
        self.decode_instances: List[InstanceInfo] = []

    def scale_prefill(self, target_count: int):
        """根据 Prefill 负载动态扩缩 Prefill 实例数"""
        current = len(self.prefill_instances)
        if target_count > current:
            for _ in range(target_count - current):
                self._launch_prefill_instance()
        elif target_count < current:
            for _ in range(current - target_count):
                self._drain_and_stop_prefill_instance()

    def auto_scale(self):
        """基于指标自动调整 P:D 比例"""
        prefill_load = self._avg_prefill_load()
        decode_load = self._avg_decode_load()

        # Prefill 过载：增加 Prefill 实例
        if prefill_load > 0.8 and decode_load < 0.5:
            self.scale_prefill(len(self.prefill_instances) + 1)
        # Decode 过载：增加 Decode 实例
        elif decode_load > 0.8 and prefill_load < 0.5:
            self.scale_decode(len(self.decode_instances) + 1)
```

2. **全链路指标**：

```python
class PDMetrics:
    """PD 分离全链路指标"""

    # 传输指标
    kv_transfer_latency_ms: Histogram       # KV 传输延迟分布
    kv_transfer_bytes: Counter              # KV 传输总字节数
    kv_transfer_saved_bytes: Counter        # 因缓存命中避免传输的字节数
    kv_compression_ratio: Gauge             # 压缩比

    # 路由指标
    routing_decision_count: Counter         # 路由决策计数（by type: pd_split / decode_only）
    routing_latency_ms: Histogram           # 路由决策延迟
    fallback_count: Counter                 # 故障转移次数

    # 调度指标
    prefill_instance_utilization: Gauge     # Prefill 实例 GPU 利用率
    decode_instance_utilization: Gauge      # Decode 实例 GPU 利用率
    kv_wait_time_ms: Histogram              # Decode 端等待 KV 到达的时间

    # 端到端指标
    pd_ttft_ms: Histogram                   # PD 分离模式下的 TTFT
    pd_itl_ms: Histogram                    # PD 分离模式下的 ITL
    pd_overhead_ms: Histogram               # PD 分离引入的额外开销
```

3. **Dashboard 接入**：

```python
# 暴露 Prometheus 兼容的 /metrics 端点
class PDMetricsExporter:
    def export(self) -> str:
        """导出 Prometheus 格式指标"""
        lines = []
        lines.append(f'pd_kv_transfer_latency_ms_p99 {self.metrics.kv_transfer_latency_ms.p99}')
        lines.append(f'pd_kv_transfer_bytes_total {self.metrics.kv_transfer_bytes.total}')
        lines.append(f'pd_kv_transfer_saved_bytes_total {self.metrics.kv_transfer_saved_bytes.total}')
        lines.append(f'pd_routing_fallback_total {self.metrics.fallback_count.total}')
        lines.append(f'pd_prefill_gpu_utilization {self.metrics.prefill_instance_utilization.value}')
        lines.append(f'pd_decode_gpu_utilization {self.metrics.decode_instance_utilization.value}')
        return '\n'.join(lines)
```

#### 预期效果

- 支持灵活的 NP:MD 部署比例（如 2P3D、1P4D 等）
- 基于负载指标自动调整 P:D 比例
- 全链路可观测，快速定位性能瓶颈

---

## 三、优化点总览与实施路线

### 3.1 总览表

| 优化点 | 优先级 | 状态 | 核心修改 | 依赖关系 |
|--------|--------|------|---------|---------|
| 1. V1 引擎 PD 基础适配 | P0 | ✅ 已实现 | `v1/worker/gpu_model_runner.py`, `v1/core/scheduler.py`, `v1/engine/core.py`, `v1_connector.py` | 无 |
| 2. 智能请求路由/代理 | P0 | 🔲 未实现 | 新增 `v1/engine/pd_router.py` | 优化 1 |
| 3. 调度器 PD 感知 | P1 | 🔲 未实现 | `v1/core/scheduler.py` | 优化 1 |
| 4. KV Cache 传输优化 | P1 | 🔲 未实现 | 新增 `v1_connector.py`, 修改 `pynccl_pipe.py` | 优化 1 |
| 5. Prefix Cache 与 PD 协同 | P2 | 🔲 未实现 | 新增 `pd_cache_coordinator.py` | 优化 1 + 4 |
| 6. 多实例协调与可观测性 | P2 | 🔲 未实现 | 新增 `pd_orchestrator.py`, `pd_metrics.py` | 优化 1 + 2 |

### 3.2 依赖关系图

```
优化 1 (V1 PD 基础适配) ──┬──→ 优化 2 (智能路由) ──→ 优化 6 (多实例协调)
                          │
                          ├──→ 优化 3 (调度器 PD 感知)
                          │
                          └──→ 优化 4 (传输优化) ──→ 优化 5 (Prefix Cache 协同)
```

### 3.3 推荐实施顺序

```
Phase 1（V1 跑通 PD 分离）:
  ├── 优化 1: V1 引擎基础适配（核心，优先级最高）
  └── 优化 2: 智能路由（替换简单代理）

Phase 2（性能优化）:
  ├── 优化 3: 调度器 PD 感知
  └── 优化 4: KV 传输优化

Phase 3（高级特性）:
  ├── 优化 5: Prefix Cache 协同
  └── 优化 6: 多实例协调 + 可观测性
```

---

## 四、与其他优化方向的协同

PD 分离与本项目的其他优化方向有密切的协同关系：

### 4.1 与 QoS 调度（README.md 优化 1/4/7）

```
PD 分离后，QoS 调度可以更精细化：

  Prefill 实例上的 QoS：
    ├── 高优请求（实时对话）→ 优先 Prefill → 优先传输 KV
    └── 低优请求（批处理）  → 延后 Prefill → 限速传输

  Decode 实例上的 QoS：
    ├── Token 限速只在 Decode 实例上生效（Prefill 无 Decode token 产出）
    └── MLFQ 在 Decode 实例上保证短请求低延迟完成
```

### 4.2 与 Prefix Cache 调度（prefix-cache-scheduling-optimization.md）

```
PD 分离 + Prefix Cache 的协同：

  Prefill 实例：
    ├── Cache-Aware Scheduling → 相同前缀的请求调度在一起
    ├── Segmented LRU → 高频 System Prompt 不被驱逐
    └── 计算完的 KV 注册到 Prefix Cache → 下次相同前缀免传输

  Decode 实例：
    ├── 接收到的 KV 注册到本地 Prefix Cache
    └── 后续相同前缀的请求直接复用，无需再次传输
```

### 4.3 与后缀解码（suffix-decoding-optimization.md）

```
PD 分离 + 后缀解码的协同：

  后缀解码只在 Decode 阶段生效（不影响 Prefill）：
    ├── Decode 实例上运行 SuffixTreeProposer
    ├── 不会干扰 Prefill 实例的计算
    └── Decode 实例的 ITL 稳定 → 后缀解码的投机预测更准确
```

### 4.4 与端到端业务场景（e2e_business_cases）

```
PD 分离在端到端场景中的角色：

  Phase 3（流量暴增）：PD 分离可以独立扩容 Prefill 实例应对
  Phase 4（长文档暴增）：长文档的 Prefill 不干扰短对话的 Decode
  Phase 5（全面过载）：Prefill 实例可以更早拒绝（准入控制在 Prefill 端）
```

---

## 五、预期性能收益

| 指标 | 混合部署（当前） | PD 分离（优化后） | 优化来源 |
|------|----------------|------------------|---------|
| ITL P99 抖动 | Prefill 插入导致尖峰 | 完全稳定 | 优化 1 + 3 |
| TTFT（长 prompt） | 排队等待 Decode 空闲 | 独立 Prefill 实例全力计算 | 优化 1 + 3 |
| KV 传输延迟 | N/A（不传输） | 首次传输 + 后续缓存命中免传输 | 优化 4 + 5 |
| 传输带宽占用 | N/A | ↓ 50-80%（增量传输 + FP8 压缩） | 优化 4 |
| 路由决策延迟 | N/A | < 1ms（负载感知路由） | 优化 2 |
| 故障恢复时间 | 无故障转移 | 秒级自动转移 | 优化 2 + 6 |
| 资源利用率 | GPU 利用模式不匹配 | P/D 独立调优，整体利用率↑ | 优化 3 + 6 |

---

## 六、实现进度追踪

- [x] 优化 1：V1 引擎 PD 基础适配（P0）
- [ ] 优化 2：智能请求路由/代理（P0）
- [ ] 优化 3：调度器 PD 感知（P1）
- [ ] 优化 4：KV Cache 传输优化（P1）
- [ ] 优化 5：Prefix Cache 与 PD 分离协同（P2）
- [ ] 优化 6：多实例协调与可观测性（P2）
