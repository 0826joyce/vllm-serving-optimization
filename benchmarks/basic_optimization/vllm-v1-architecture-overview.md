# vLLM V1 推理引擎架构与要点总览

> 基于 vLLM v0.7.3 V1 架构的源码分析，覆盖整体架构、核心模块、请求生命周期、KV Cache 管理、投机解码、分布式架构等关键知识点

## 一、项目定位

vLLM 是一个高性能的 LLM 推理和服务引擎。V1 架构是其最新的主力架构，相比 V0 有以下核心改进：

| 特性 | V0 | V1 |
|------|----|----|
| 调度器 | 三阶段（Prefill → Running → Swapped） | 两阶段（Waiting → Running），无 SWAPPED 状态 |
| 抢占策略 | Swap 到 CPU（I/O 开销大） | Recompute（直接丢弃、重新计算，无 I/O） |
| Chunked Prefill | 可选 | 默认启用（与 Decode 统一 token_budget） |
| KV Cache | Block-level 管理 | Block-level + Prefix Caching（hash chain） |
| PD 分离 | 有完整实现 | 零代码（需全新适配） |
| 投机解码 | 多种 proposer | 仅 N-gram Proposer |

---

## 二、整体架构（四层分离）

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          API 入口层 (Entrypoints)                        │
│  OpenAI Compatible API (FastAPI) / Anthropic / gRPC / MCP               │
│  路径: vllm/entrypoints/openai/api_server.py                            │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │ HTTP/WebSocket
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          引擎层 (Engine)                                 │
│  AsyncLLM (异步) / LLMEngine (同步)                                     │
│  ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐   │
│  │ InputProcessor   │   │ OutputProcessor  │   │ EngineCoreClient   │   │
│  │ tokenize+预处理  │   │ detokenize+后处理│   │ ZMQ 通信客户端     │   │
│  └─────────────────┘   └──────────────────┘   └────────┬───────────┘   │
│  路径: vllm/v1/engine/                                  │              │
└─────────────────────────────────────────────────────────┼──────────────┘
                                                          │ ZMQ IPC
                                                          ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       引擎核心层 (EngineCore)                            │
│  EngineCoreProc — 独立进程运行                                            │
│  ┌──────────────────────┐   ┌────────────────────────────────┐          │
│  │    Scheduler          │   │         Executor               │          │
│  │  调度决策：谁跑/跑多少  │   │  分布式编排：驱动 Worker 执行   │          │
│  │  KVCacheManager       │   │  Multiproc / Ray / Uniproc     │          │
│  │  EncoderCacheManager  │   │                                │          │
│  └──────────────────────┘   └─────────────┬──────────────────┘          │
│  路径: vllm/v1/engine/core.py              │                            │
│        vllm/v1/core/scheduler.py           │                            │
└────────────────────────────────────────────┼────────────────────────────┘
                                             │ Worker RPC / 共享内存
                                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Worker 层 (Worker)                               │
│  GPUWorker → GPUModelRunner → 模型前向计算                                │
│  ┌─────────────────┐   ┌──────────────────┐   ┌────────────────────┐   │
│  │ GPUInputBatch    │   │ BlockTable       │   │ Sampler / Logprobs │   │
│  │ 请求→Tensor 打包 │   │ Block 映射管理   │   │ 采样+logprob 计算  │   │
│  └─────────────────┘   └──────────────────┘   └────────────────────┘   │
│  路径: vllm/v1/worker/gpu_worker.py                                     │
│        vllm/v1/worker/gpu_model_runner.py                               │
└──────────────────────────────────────────────────────────────────────────┘
```

### 四层职责总结

| 层级 | 核心职责 | 关键类 | 进程模型 |
|------|---------|--------|---------|
| **API 入口** | 接收 HTTP 请求，路由到引擎 | `api_server.py` (FastAPI) | API 进程 |
| **引擎层** | 输入预处理、输出后处理、客户端通信 | `AsyncLLM`, `InputProcessor`, `OutputProcessor` | API 进程 |
| **引擎核心** | 调度决策、KV Cache 管理、驱动 Worker | `EngineCoreProc`, `Scheduler`, `Executor` | 独立进程 |
| **Worker 层** | GPU 上的模型前向计算、采样 | `GPUWorker`, `GPUModelRunner` | Worker 进程 |

---

## 三、核心模块详解

### 3.1 Scheduler — 调度器

调度器是 vLLM 的"大脑"，负责每一轮迭代（scheduling step）决定：
1. **哪些 RUNNING 请求继续计算**（分配 token_budget）
2. **哪些 WAITING 请求被调度**（从等待队列选取）
3. **需不需要抢占**（KV Cache 不足时，抢占哪个 RUNNING 请求）

```python
# vllm/v1/core/sched/scheduler.py — 上游标准 Scheduler
class Scheduler(SchedulerInterface):
    def schedule(self) -> SchedulerOutput:
        """每个 scheduling step 调用一次，返回本轮调度决策"""
        # 1. 调度 RUNNING 请求（分配 token_budget）
        # 2. 如果 KV Cache 不足 → 触发抢占
        # 3. 调度 WAITING 请求（入队新请求）
        # 4. 返回 SchedulerOutput
    
    def update_from_output(self, scheduler_output, model_runner_output):
        """模型执行完毕后，更新调度器状态"""
        # 1. 更新每个请求的 num_computed_tokens
        # 2. 检查是否完成（stop condition）
        # 3. 释放已完成请求的资源
```

**原生 V1 调度策略**：
- **调度顺序**：FCFS（先来先服务）
- **抢占策略**：LIFO（后进先出）— Recompute（丢弃 KV，重新计算）
- **Chunked Prefill**：长 prompt 被分成多个 chunk，与 decode 请求共享 `token_budget`

**SchedulerOutput 数据流**：

```
Scheduler.schedule()
     │
     ├── scheduled_new_reqs: List[NewRequestData]     # 首次调度的请求
     ├── scheduled_cached_reqs: CachedRequestData     # 已缓存的请求（增量更新）
     ├── num_scheduled_tokens: Dict[req_id, int]      # 每个请求分配的 token 数
     ├── total_num_scheduled_tokens: int               # 总 token 预算消耗
     ├── scheduled_spec_decode_tokens: Dict            # 投机解码的 draft tokens
     ├── finished_req_ids: Set[str]                    # 已完成的请求（通知 Worker 清理）
     └── ...
```

### 3.2 KVCacheManager — KV 缓存管理器

KV Cache 是推理引擎中最核心的内存资源。V1 的 KV Cache 管理基于 **Block-level 分页**：

```
┌──────────────────────────────────────────────────────────────────┐
│                    KV Cache 物理内存 (GPU)                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │ Block 0│ │ Block 1│ │ Block 2│ │ Block 3│ │ Block 4│  ...    │
│  │ 16 tok │ │ 16 tok │ │ 16 tok │ │ 16 tok │ │ 16 tok │        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│      ↑           ↑           ↑                                   │
│   req_A[0]   req_A[1]   req_B[0]    ← Block Table 映射          │
└──────────────────────────────────────────────────────────────────┘
```

**核心组件**：

| 组件 | 文件 | 职责 |
|------|------|------|
| `KVCacheManager` | `kv_cache_manager.py` | 顶层管理器：分配/释放/prefix cache 查询 |
| `BlockPool` | `block_pool.py` | 物理 block 池：free list 管理 |
| `KVCacheCoordinator` | `kv_cache_coordinator.py` | 多类型缓存协调 |
| `KVCacheBlock` | `kv_cache_utils.py` | 单个 block 的数据结构（block_id, ref_cnt, block_hash） |

**Prefix Caching 工作原理**：

```
Prefix Cache 的核心是 hash chain:

Request A: [sys_prompt(200 tok) + user_query(50 tok)]
           ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
           │Hash(B0)│→ │Hash(B1)│→ │Hash(B2)│→ │Hash(B3)│  ← 链式 hash
           │=0xAB12 │  │=0xCD34 │  │=0xEF56 │  │=0x1234 │
           └───────┘  └───────┘  └───────┘  └───────┘
                                      ↑
Request B: [sys_prompt(200 tok) + different_query]
           Hash(B0)=0xAB12 ✅ → Hash(B1)=0xCD34 ✅ → Hash(B2)=0xEF56 ✅ → miss
           共享前 3 个 block！只需 prefill 剩余 tokens

查找流程:
  get_computed_blocks(request)
    → hash_request_tokens()      # 逐 block 计算 hash（parent_hash chain）
    → lookup cached_block_hash_to_block  # 查 hash → physical block
    → 返回 computed_blocks + num_computed_tokens
  
  allocate_slots(request, num_new_tokens, computed_blocks)
    → _touch(computed_blocks)    # 共享 block: ref_cnt++
    → _get_new_blocks(...)       # 分配新 block（可能驱逐 LRU block）
    → _cache_full_blocks(...)    # 同步写入新 block 的 hash（同步！同 step 内可见）
```

**关键设计决策**：
- Block 大小默认 16 tokens
- Prefix Cache 的 hash chain 是**同步写入**的，同一 scheduling step 内的请求可以立即共享
- 驱逐策略：LRU（`ref_cnt == 0` 的 block 才可驱逐）
- 抢占时直接 Recompute（V1 无 Swap 到 CPU 的能力）

### 3.3 Request — 请求数据结构

```python
class Request:
    # ---- 核心标识 ----
    request_id: str
    prompt_token_ids: List[int]          # prompt 的 token IDs
    num_prompt_tokens: int               # prompt 长度
    
    # ---- 生成状态 ----
    num_computed_tokens: int = 0          # 已计算的 token 数
    _output_token_ids: List[int] = []    # 已生成的 output tokens
    spec_token_ids: List[int] = []       # 投机解码的 draft tokens
    
    # ---- 调度状态 ----
    status: RequestStatus                 # WAITING / RUNNING / PREEMPTED / FINISHED_*
    arrival_time: float                   # 到达时间
    
    # ---- KV Cache ----
    block_hashes: List[BlockHash]         # block 的 hash chain
    num_cached_tokens: int                # prefix cache 命中的 token 数
    
    # ---- 采样参数 ----
    sampling_params: SamplingParams       # temperature, top_p, max_tokens 等
    eos_token_id: int                     # 结束 token
    
    # ---- 关键属性 ----
    @property
    def num_tokens(self) -> int:          # = len(prompt) + len(output)
    @property
    def num_tokens_with_spec(self) -> int: # = num_tokens + len(spec_tokens)
```

**请求状态机**：

```
              add_request()        schedule()         _check_stop()
                  │                    │                    │
  ┌───────────┐   │   ┌───────────┐   │   ┌───────────┐   │   ┌────────────────┐
  │  外部请求  │──→│──→│  WAITING   │──→│──→│  RUNNING   │──→│──→│ FINISHED_*     │
  └───────────┘       └───────────┘       └───────────┘       └────────────────┘
                           ↑                    │                 FINISHED_STOPPED
                           │    preempt         │                 FINISHED_LENGTH_CAPPED
                           │   ┌───────────┐    │                 FINISHED_ABORTED
                           └───│ PREEMPTED  │←──┘                 FINISHED_IGNORED
                               └───────────┘
```

### 3.4 Engine 层 — 引擎协调

引擎层是前后端之间的桥梁，采用**多进程 + ZMQ 通信**架构：

```
┌─────────────────────────────┐         ┌──────────────────────────────┐
│      API 进程                │         │   EngineCore 进程             │
│                              │  ZMQ    │                              │
│  AsyncLLM                    │ ◄─────► │  EngineCoreProc              │
│   ├─ InputProcessor          │  IPC    │   ├─ Scheduler               │
│   │   ├─ tokenize            │         │   │   ├─ KVCacheManager      │
│   │   └─ 多模态处理          │         │   │   └─ EncoderCacheManager │
│   ├─ OutputProcessor         │         │   ├─ Executor                │
│   │   ├─ detokenize          │         │   │   └─ Worker(s)           │
│   │   ├─ logprobs            │         │   └─ StructuredOutputManager │
│   │   └─ stop string 检测    │         │                              │
│   └─ EngineCoreClient       │         │  busy loop:                  │
│       (ZMQ 发送/接收)        │         │    scheduler.schedule()      │
│                              │         │    → executor.execute_model() │
└─────────────────────────────┘         │    → scheduler.update()      │
                                         └──────────────────────────────┘
```

**关键数据结构**：

```python
class EngineCoreRequest(msgspec.Struct):
    """前端 → EngineCore 的请求"""
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float
    lora_request: Optional[LoRARequest]

class EngineCoreOutput(msgspec.Struct):
    """EngineCore → 前端的输出"""
    request_id: str
    new_token_ids: list[int]
    finish_reason: FinishReason | None
    new_logprobs: LogprobsLists | None
```

### 3.5 Worker 层 — 模型执行

```
GPUWorker
  │
  ├─ init_device()          # 初始化 GPU 设备、分布式环境
  ├─ load_model()           # 加载模型权重
  ├─ determine_available_memory()  # 探测可用显存
  │
  └─ GPUModelRunner
       │
       ├─ execute_model(scheduler_output)
       │     │
       │     ├─ 1. 准备输入 Tensor（GPUInputBatch）
       │     │     ├─ token_ids, positions, block_table
       │     │     └─ 多模态特征（如果有）
       │     │
       │     ├─ 2. 模型前向传播
       │     │     ├─ Embedding → Transformer Layers → LM Head
       │     │     └─ Attention: FlashAttention / FlashInfer / ...
       │     │
       │     ├─ 3. 采样
       │     │     ├─ Sampler（top-k, top-p, temperature）
       │     │     └─ RejectionSampler（投机解码验证）
       │     │
       │     └─ 4. 返回 ModelRunnerOutput
       │           ├─ sampled_token_ids
       │           ├─ spec_token_ids (draft for next step)
       │           └─ logprobs
       │
       └─ CUDA Graph（可选）
             └─ 对固定 batch size 使用 CUDA Graph 加速
```

### 3.6 Executor 层 — 分布式编排

```python
class Executor(ABC):
    """抽象执行器基类"""
    
    @abstractmethod
    def execute_model(self, scheduler_output) -> ModelRunnerOutput:
        """将调度决策分发给 Worker 执行"""

# 三种实现：
# 1. UniprocExecutor  — 单进程（debug / 单 GPU）
# 2. MultiprocExecutor — 多进程（TP > 1，本机多 GPU）
# 3. RayExecutor      — Ray 分布式（跨节点 TP/PP）
```

---

## 四、请求完整生命周期

```
客户端发送请求
    │
    ▼
(1) API Server 接收 HTTP 请求
    │ POST /v1/chat/completions
    ▼
(2) AsyncLLM.generate()
    │ → InputProcessor.process_inputs()
    │   ├─ tokenize prompt
    │   ├─ 处理多模态输入（图片/视频）
    │   └─ 构造 EngineCoreRequest
    │ → EngineCoreClient.send(request)
    ▼
(3) EngineCoreProc 接收请求
    │ → Request.from_engine_core_request()
    │ → Scheduler.add_request(request)
    │   └─ request.status = WAITING, 加入 waiting 队列
    ▼
(4) Scheduler.schedule() — 每个 step 调用
    │
    ├─ 4a. 调度 RUNNING 请求
    │      for req in self.running:
    │        num_new_tokens = req.num_tokens_with_spec - req.num_computed_tokens
    │        new_blocks = kv_cache_manager.allocate_slots(req, num_new_tokens)
    │        if new_blocks is None:
    │          → 触发抢占（preempt 优先级最低的 RUNNING 请求）
    │        token_budget -= num_new_tokens
    │
    ├─ 4b. 调度 WAITING 请求
    │      while waiting && token_budget > 0:
    │        req = waiting.popleft()  // FCFS
    │        computed_blocks, num_computed = kv_cache_manager.get_computed_blocks(req)
    │        num_new_tokens = req.num_tokens - num_computed  // 减去 prefix cache 命中
    │        new_blocks = kv_cache_manager.allocate_slots(req, num_new_tokens, computed_blocks)
    │        running.append(req)
    │        req.status = RUNNING
    │
    └─ 4c. 返回 SchedulerOutput
    ▼
(5) Executor.execute_model(scheduler_output)
    │ → GPUWorker.execute_model()
    │   → GPUModelRunner.execute_model()
    │     ├─ 准备 input tensors
    │     ├─ model.forward()  ← GPU 计算
    │     ├─ Sampler.forward() → sampled_token_ids
    │     └─ (可选) NgramProposer.propose() → spec_token_ids
    │ → 返回 ModelRunnerOutput
    ▼
(6) Scheduler.update_from_output(scheduler_output, model_output)
    │ for req in running:
    │   req.num_computed_tokens += num_scheduled
    │   req.append_output_token_ids(sampled_tokens)
    │   if _check_stop(req):  // EOS / max_tokens / stop_string
    │     req.status = FINISHED_STOPPED
    │     _free_request(req)  // 释放 KV Cache blocks
    │ → 返回 EngineCoreOutputs
    ▼
(7) EngineCoreClient 接收输出
    │ → OutputProcessor.process_outputs()
    │   ├─ detokenize(new_token_ids) → text
    │   ├─ 计算 logprobs
    │   └─ 检测 stop string
    │ → 构造 RequestOutput
    ▼
(8) AsyncLLM yield RequestOutput
    │ → SSE / JSON 流式返回给客户端
    ▼
客户端接收响应
```

---

## 五、关键性能指标

| 指标 | 定义 | 优化方向 |
|------|------|---------|
| **TTFT** (Time To First Token) | 从请求到达到第一个 token 生成的时间 | Prefill 速度、调度延迟、Prefix Cache 命中率 |
| **ITL / TPOT** (Inter-Token Latency) | 连续两个 token 之间的间隔 | Decode 效率、batch 大小、token_budget 分配 |
| **Throughput** (tokens/s) | 单位时间内总 token 产出 | batch 效率、GPU 利用率、投机解码加速比 |
| **缓存命中率** | Prefix Cache 命中的 block 比例 | 缓存策略（LRU/SLRU）、调度顺序 |
| **抢占频率** | 每秒触发抢占的次数 | 内存管理、水位线流控 |
| **尾延迟** (P99) | 99% 请求的延迟上界 | 公平调度、限速、过载管理 |

---

## 六、Attention 后端

vLLM V1 支持多种 Attention 实现，通过 `attention/backends/` 下的模块注册：

| 后端 | 适用场景 | 特点 |
|------|---------|------|
| **FlashAttention** | NVIDIA GPU (主力) | Tiled + Fused Softmax，O(1) 额外内存 |
| **FlashInfer** | NVIDIA GPU | 支持 Paged KV Cache 的原生 attention |
| **FlexAttention** | 灵活 mask 场景 | PyTorch 原生，支持自定义 mask |
| **Triton Attention** | 跨平台 | Triton 编写，可移植性好 |
| **CPU Attention** | CPU 推理 | 纯 CPU 实现 |
| **MLA** (Multi-head Latent Attention) | DeepSeek 模型 | 压缩 KV Cache（GQA 的进化版） |

---

## 七、投机解码 (Speculative Decoding)

投机解码通过"猜测+验证"减少 auto-regressive 的串行步数：

```
常规 Decode (每步 1 token):
  Step 1: [prompt] → token_1
  Step 2: [prompt, token_1] → token_2
  Step 3: [prompt, token_1, token_2] → token_3
  → 3 步产出 3 tokens

投机 Decode (每步 1+K tokens):
  Step 1: 
    Draft: propose [d1, d2, d3]   ← proposer 猜测 3 个 token
    Verify: model([prompt, d1, d2, d3])  ← 一次前向验证 4 个位置
    Accept: token_1=d1 ✅, token_2=d2 ✅, token_3 ≠ d3 ❌ → resample → token_3
  → 1 步产出 3 tokens (加速 ~3×)
```

**V1 支持的 Proposer**：

| Proposer | 说明 | 性能 |
|----------|------|------|
| N-gram | 在本请求 context 中固定窗口搜索 | 基础，匹配率有限 |
| EAGLE | Draft model 辅助生成 | 高接受率，额外显存开销 |
| Medusa | 多头预测 | 并行 draft，需微调 |

**投机解码在调度器中的体现**：

```python
# Scheduler 中：
request.spec_token_ids = [d1, d2, d3]  # proposer 上一步生成的 draft
request.num_tokens_with_spec = num_tokens + len(spec_token_ids)
# → schedule() 为这个请求分配 1 + 3 = 4 个 token 的 budget

# Worker 中：
# 一次前向传播同时处理 normal token + spec tokens
# RejectionSampler 验证 draft tokens
```

---

## 八、分布式架构

### 8.1 并行策略

| 策略 | 说明 | 切分方式 |
|------|------|---------|
| **Tensor Parallelism (TP)** | 同一层的权重切分到多 GPU | 行/列并行 + AllReduce |
| **Pipeline Parallelism (PP)** | 不同层分到不同 GPU | 前向传播流水线 |
| **Data Parallelism (DP)** | 相同模型副本处理不同数据 | 独立推理，无通信 |

### 8.2 PD 分离 (Prefill-Decode Disaggregation)

```
┌─────────────────────────┐    KV Transfer    ┌──────────────────────────┐
│   Prefill Instance       │   (NCCL/RDMA)     │   Decode Instance        │
│   (计算密集型)            │ ──────────────►   │   (访存密集型)            │
│                          │                    │                          │
│   ├─ 只做 Prefill        │    KV Cache       │   ├─ 只做 Decode          │
│   ├─ 大 TP (如 TP=4)     │    Tensor         │   ├─ 小 TP (如 TP=1)     │
│   └─ 优化 TTFT           │                    │   └─ 优化 ITL             │
└─────────────────────────┘                    └──────────────────────────┘

vLLM PD 分离三层抽象:
  L1: KVPipe          — 传输管道（NCCL / Mooncake RDMA）
  L2: KVLookupBuffer   — 查找缓冲（prefix match + deque）
  L3: KVConnector      — 连接器（组合 Pipe + Buffer，对接 Scheduler）
```

**注意**：V1 架构中 PD 分离需要全新适配（V0 有完整实现，V1 零代码）。

---

## 九、核心源码文件索引

### 调度与资源管理
| 文件 | 说明 |
|------|------|
| `vllm/v1/core/sched/scheduler.py` | 上游标准 Scheduler（模块化设计） |
| `vllm/v1/core/sched/interface.py` | `SchedulerInterface` 抽象基类 |
| `vllm/v1/core/sched/output.py` | `SchedulerOutput`, `NewRequestData`, `CachedRequestData` |
| `vllm/v1/core/sched/request_queue.py` | 请求队列实现（FCFS / Priority） |
| `vllm/v1/request.py` | `Request` 类、`RequestStatus` 枚举 |

### KV Cache 管理
| 文件 | 说明 |
|------|------|
| `vllm/v1/core/kv_cache_manager.py` | KV Cache 管理器（分配/释放/prefix cache） |
| `vllm/v1/core/block_pool.py` | 物理 block 池（free list 管理） |
| `vllm/v1/core/kv_cache_utils.py` | `KVCacheBlock`, `BlockHash`, hash 计算 |
| `vllm/v1/core/kv_cache_coordinator.py` | 多类型 KV Cache 协调器 |
| `vllm/v1/kv_cache_interface.py` | `KVCacheConfig`, `KVCacheSpec` 接口定义 |

### 引擎层
| 文件 | 说明 |
|------|------|
| `vllm/v1/engine/__init__.py` | `EngineCoreRequest`, `EngineCoreOutput`, `FinishReason` |
| `vllm/v1/engine/async_llm.py` | `AsyncLLM` — 异步推理入口 |
| `vllm/v1/engine/core.py` | `EngineCoreProc` — 引擎核心进程（调度循环） |
| `vllm/v1/engine/core_client.py` | ZMQ 客户端（前端 ↔ EngineCore 通信） |
| `vllm/v1/engine/input_processor.py` | 输入预处理（tokenize + 多模态） |
| `vllm/v1/engine/output_processor.py` | 输出后处理（detokenize + logprobs） |

### Worker 层
| 文件 | 说明 |
|------|------|
| `vllm/v1/worker/gpu_worker.py` | GPU Worker（设备管理 + 模型加载） |
| `vllm/v1/worker/gpu_model_runner.py` | GPU Model Runner（前向计算 + 采样） |
| `vllm/v1/worker/gpu_input_batch.py` | 输入 Tensor 打包 |
| `vllm/v1/worker/block_table.py` | Block Table 管理 |

### Executor 层
| 文件 | 说明 |
|------|------|
| `vllm/v1/executor/abstract.py` | `Executor` 抽象基类 |
| `vllm/v1/executor/uniproc_executor.py` | 单进程执行器 |
| `vllm/v1/executor/multiproc_executor.py` | 多进程执行器（本机多 GPU） |
| `vllm/v1/executor/ray_executor.py` | Ray 分布式执行器 |

### 投机解码
| 文件 | 说明 |
|------|------|
| `vllm/v1/spec_decode/ngram_proposer.py` | N-gram Proposer |
| `vllm/v1/spec_decode/eagle.py` | EAGLE Proposer |
| `vllm/v1/sample/rejection_sampler.py` | Rejection Sampler（draft 验证） |

### API 入口
| 文件 | 说明 |
|------|------|
| `vllm/entrypoints/openai/api_server.py` | OpenAI Compatible API Server |
| `vllm/entrypoints/llm.py` | 离线推理入口（`LLM` 类） |
| `vllm/entrypoints/openai/chat_completion/serving.py` | Chat Completion 处理逻辑 |

---

## 十、LLM 推理优化全景图

```
┌────────────────────────────────────────────────────────────────────────┐
│                       LLM 推理优化全景                                  │
├───────────────────────┬────────────────────────────────────────────────┤
│ 1. 调度与资源管理      │ QoS 分级、MLFQ、Token 限速、准入控制、           │
│                        │ WFQ 公平调度、Deadline/EDF、KV 水位线             │
│                        │ → 决定"谁先跑""跑多快""满了怎么办"             │
├───────────────────────┼────────────────────────────────────────────────┤
│ 2. KV Cache 管理       │ Prefix Caching、Cache-Aware 调度、              │
│                        │ Segmented LRU、抢占保护、分层存储                │
│                        │ → 提升缓存命中率，减少 Prefill 计算量            │
├───────────────────────┼────────────────────────────────────────────────┤
│ 3. 投机解码            │ N-gram / 后缀树 / EAGLE / Medusa                │
│                        │ → 每步多产 token，加速 Decode 阶段               │
├───────────────────────┼────────────────────────────────────────────────┤
│ 4. 分布式架构 (PD分离) │ V1 PD 适配、智能路由、KV 传输优化                │
│                        │ → TTFT 和 ITL 独立优化，互不干扰                 │
├───────────────────────┼────────────────────────────────────────────────┤
│ 5. 模型压缩/量化       │ GPTQ / AWQ / FP8 / W4A16                       │
│                        │ → 减少模型权重体积，加速矩阵乘法                 │
├───────────────────────┼────────────────────────────────────────────────┤
│ 6. Attention/算子优化  │ FlashAttention / FlashInfer / Paged Attention   │
│                        │ → 减少 Attention 的时间和内存开销                 │
└───────────────────────┴────────────────────────────────────────────────┘
```

---

## 十一、快速上手

### 启动服务

```bash
# 安装
git clone <repo_url>
cd vllm-serving-optimization
pip install -e .

# 启动 OpenAI Compatible API Server
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> \
    --max-model-len 4096 \
    --max-num-batched-tokens 2048 \
    --enable-chunked-prefill

# 发送请求
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "<模型名>",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }'
```

### 关键启动参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--max-model-len` | 最大序列长度（prompt + output） | 模型配置 |
| `--max-num-batched-tokens` | 每步最大 token 预算 | 2048 |
| `--max-num-seqs` | 最大并发请求数 | 256 |
| `--enable-chunked-prefill` | 启用 Chunked Prefill（V1 默认） | True |
| `--enable-prefix-caching` | 启用 Prefix Caching | False |
| `--block-size` | KV Cache block 大小 | 16 |
| `--tensor-parallel-size` | Tensor Parallelism 大小 | 1 |
| `--speculative-model` | 投机解码 draft 模型 | None |
| `--num-speculative-tokens` | 每步投机 token 数 | None |
