# 基于 vLLM V1 的后缀解码（Suffix Decoding）优化

> 在 vLLM V1 现有 N-gram Proposer 基础上，引入后缀树（Suffix Tree）实现更高效、更高接受率的投机解码

## 一、项目背景与动机

### 1.1 后缀解码的核心思想

后缀解码是一种新的投机解码策略：

> **利用后缀树（Suffix Tree / Suffix Automaton）在请求的历史 token 中高效查找最长匹配后缀，作为投机解码的 draft tokens，相比固定 N-gram 匹配能获得更高的匹配长度和接受率。**

核心思路：将 prompt + 已生成 tokens 构建为后缀结构，每次解码时在后缀树中查找当前最后几个 token 的最长匹配，将匹配后续的 tokens 作为 draft 提案。

### 1.2 vLLM V1 当前投机解码的局限性

通过深入阅读 vLLM V1 源码（`vllm/v1/spec_decode/ngram_proposer.py` + `vllm/v1/worker/gpu_model_runner.py` + `vllm/v1/core/scheduler.py` + `vllm/v1/sample/rejection_sampler.py`），发现以下可优化的点：

| 现状问题 | 具体表现 | 影响 |
|---------|---------|------|
| **V1 仅支持 N-gram** | 代码中明确 `assert speculative_config.ngram_prompt_lookup_min`，不支持任何其他 proposer | 无法使用更高级的提案策略 |
| **固定 N-gram 窗口** | 只用 `ngram_prompt_lookup_min` 单一 n 值做匹配，无回退 | n 选大了匹配不到，选小了匹配短、接受率低 |
| **只取首次匹配** | KMP 搜索找到第一个匹配就返回，不选最优 | 可能错过更长、更优质的匹配 |
| **无状态搜索** | `NgramProposer` 无状态，每次 `propose()` 都从头遍历整个 context | 重复计算，O(context_len) 时间复杂度 |
| **仅搜索本请求上下文** | 只在当前请求的 prompt + output 中搜索匹配 | 跨请求共享模式（如相似对话模板）无法利用 |
| **仅支持 Greedy 采样** | `RejectionSampler` 抛 `NotImplementedError` 如果非 greedy | 限制了适用场景 |

### 1.3 优化主题定位

本项目聚焦于 **用后缀树（Suffix Tree）替换/增强 vLLM V1 的 N-gram Proposer**，通过 5 个递进的优化点，从基础后缀树实现、增量更新、自适应匹配策略、跨请求共享、到全链路指标量化，形成一套完整的后缀解码优化方案。

```
vLLM V1 当前 spec decode 链路：

  Scheduler.schedule()        → 计算 num_tokens_with_spec
       ↓
  GPUModelRunner.execute_model()
       ↓
  模型前向传播（一次性计算 normal + spec tokens）
       ↓
  RejectionSampler.forward()  → 逐位验证 draft vs target
       ↓
  generate_draft_token_ids()  → ★ NgramProposer.propose() ★
       ↓                         ← 这里是我们要优化的核心
  ModelRunnerOutput            → spec_token_ids 传回 Scheduler

优化后：
  generate_draft_token_ids()  → ★ SuffixTreeProposer.propose() ★
                                  ├── 增量更新后缀树（O(1) per token）
                                  ├── 最长后缀匹配（O(pattern_len)）
                                  ├── 多候选评分选最优
                                  └── 自适应匹配长度
```

---

## 二、vLLM V1 Spec Decode 全链路分析

> 在开始优化之前，先完整理解现有链路，这是加深 vLLM 理解的关键

### 2.0 execute_model 完整函数调用栈（源码级）

以下是从引擎调度到模型输出的**完整调用链**，标注了每个函数的文件位置和行号（基于当前代码库），以及 spec decode 相关的关键逻辑。

#### 2.0.1 顶层调用链（Engine → Executor → Worker → ModelRunner）

```
EngineCore.step()                                    # vllm/v1/engine/core.py:146
  ├─ self.scheduler.schedule()                        # → SchedulerOutput
  │     └─ 构建 scheduled_spec_decode_tokens          # scheduler.py:178, 327-332
  │        （将每个请求上一步生成的 spec_token_ids 纳入调度）
  │
  ├─ self.model_executor.execute_model(scheduler_output)
  │     └─ Executor.execute_model()                   # vllm/v1/executor/abstract.py:71-77
  │           └─ collective_rpc("execute_model", args=(scheduler_output,))
  │                 └─ Worker.execute_model()          # vllm/v1/worker/gpu_worker.py:222-228
  │                       └─ self.model_runner.execute_model(scheduler_output)
  │                             └─ ★ GPUModelRunner.execute_model() ★
  │                                  # vllm/v1/worker/gpu_model_runner.py:867-1018
  │
  └─ self.scheduler.update_from_output(scheduler_output, output)
        └─ 处理 accepted/rejected tokens              # scheduler.py:657-679
           更新 num_computed_tokens、spec_token_ids
```

#### 2.0.2 GPUModelRunner.execute_model() 内部调用栈

```python
# ========== vllm/v1/worker/gpu_model_runner.py ==========

@torch.inference_mode()
def execute_model(scheduler_output, intermediate_tensors=None):  # L867-1018
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 1: 状态更新                                                  │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ self._update_states(scheduler_output)          # L873, 定义: L235-424
    │    ├─ 移除已完成请求                              # L246-259
    │    │   └─ input_batch.remove_request(req_id)     # gpu_input_batch.py:301-333
    │    ├─ 移除未调度请求（抢占/暂停）                   # L269-284
    │    ├─ 添加新请求                                  # L286-339
    │    │   └─ CachedRequestState 构建                 # gpu_input_batch.py:23-45
    │    │   └─ input_batch.add_request(req_state)      # gpu_input_batch.py:207-299
    │    ├─ 更新运行中请求的状态                          # L341-400
    │    │   ├─ 更新 num_computed_tokens                # L378
    │    │   ├─ token_ids_cpu[i, start:end] = new_token_ids     # L387-389
    │    │   ├─ ★ num_tokens_no_spec[i] = end_token_index ★     # L390
    │    │   │   （记录不含 spec tokens 的真实 token 数量）
    │    │   ├─ ★ 追加 spec_token_ids 到 token_ids_cpu ★        # L391-398
    │    │   │   spec_token_ids = scheduler_output.scheduled_spec_decode_tokens
    │    │   │                                                   .get(req_id, ())
    │    │   │   token_ids_cpu[i, end:end+spec_len] = spec_token_ids
    │    │   └─ ★ num_tokens[i] = end_token_index ★             # L400
    │    │       （num_tokens 包含 spec decode tokens）
    │    ├─ 压缩空洞 input_batch.condense()             # L420-421
    │    └─ refresh_sampling_metadata() → GPU           # L423-424
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 2: 多模态编码器（可选，文本模型跳过）                           │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ [可选] self._execute_encoder(scheduler_output)   # L877, 定义: L778-824
    ├─ [可选] self._gather_encoder_outputs()            # L878, 定义: L826-862
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 3: 准备模型输入                                               │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ attn_metadata, logits_indices = self._prepare_inputs(scheduler_output)
    │    │                                              # L883, 定义: L426-594
    │    ├─ block_table.commit(num_reqs)                # L437 → GPU 拷贝
    │    ├─ 构建 num_scheduled_tokens, req_indices, arange
    │    │   （利用 np.repeat + np.cumsum 向量化操作）   # L441-464
    │    ├─ 构建 positions_np (= num_computed_tokens + arange)    # L467-470
    │    ├─ 构建 token_indices → index_select 得到 input_ids     # L481-490
    │    ├─ 构建 slot_mapping（block_table_indices + offsets）    # L492-508
    │    ├─ 构建 FlashAttentionMetadata                          # L561-575
    │    │   ├─ query_start_loc, seq_lens, block_table, slot_mapping
    │    │   └─ cascade attention 检测                            # L540-559
    │    │
    │    ├─ ★ Spec Decode 关键分支 ★                              # L577-588
    │    │   if len(scheduled_spec_decode_tokens) > 0:
    │    │       logits_indices = self._calc_spec_decode_metadata()
    │    │   else:
    │    │       logits_indices = query_start_loc[1:] - 1  # 每请求只取最后 1 个
    │    │
    │    └─ _calc_spec_decode_metadata()                 # L732-776
    │         │  计算 spec decode 时需要提取 logits 的位置索引
    │         │  关键逻辑：
    │         │    num_sampled_tokens = num_spec_decode_tokens + 1
    │         │    logits_start_loc = cu_num_tokens - num_sampled_tokens
    │         │    → 向量化计算得到 spec_decode_logits_indices
    │         │
    │         │  示例：
    │         │    num_scheduled_tokens: [4, 100, 3,   100, 2]
    │         │    num_spec_tokens:      [3, 0,   2,   0,   1]
    │         │    num_sampled_tokens:   [4, 1,   3,   1,   2]
    │         │    → logits_indices: [0,1,2,3, 103, 104,105,106, 206,207,208]
    │         └─ return torch.from_numpy(indices).to(device)
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 4: CUDA Graph / Eager 模式选择                               │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ num_input_tokens = pad_for_cudagraph(...)        # L885-893
    │   （如果用 CUDA Graph，需要 pad 到预定义的 batch size）
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 5: 模型前向传播（最耗时）                                      │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ with set_forward_context(attn_metadata, vllm_config):    # L937
    │   hidden_states = self.model(                             # L938-945
    │       input_ids=input_ids,         # [num_input_tokens] int32
    │       positions=positions,          # [num_input_tokens] int64
    │       kv_caches=self.kv_caches,     # 持久化 KV Cache buffer
    │       attn_metadata=None,           # 通过 forward_context 传递
    │       intermediate_tensors=...,     # Pipeline Parallel 中间层
    │       inputs_embeds=...,            # 多模态模型使用
    │   )
    │   注：normal tokens 和 spec tokens 一次性全部经过 forward，
    │       GPU 不区分哪些是"真实"哪些是"投机"的 token
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 6: 提取 logits + 采样                                        │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ hidden_states = hidden_states[:num_scheduled_tokens]     # L950
    ├─ sample_hidden_states = hidden_states[logits_indices]     # L951
    │   （用 logits_indices 选出需要采样的位置）
    ├─ logits = self.model.compute_logits(sample_hidden_states) # L952
    │
    ├─ sampling_metadata = input_batch.get_sampling_metadata(   # L955-956
    │       scheduled_spec_decode_tokens)
    │   └─ InputBatch.get_sampling_metadata()           # gpu_input_batch.py:459-467
    │       ★ 设置 sampling_metadata.spec_token_ids ★
    │       （如果有 spec decode，则填充 spec_token_ids 列表；否则 None）
    │
    ├─ sampler_output = self.model.sample(logits, sampling_metadata)  # L957-960
    │   └─ Sampler.forward()                            # vllm/v1/sample/sampler.py:24-73
    │       │
    │       ├─ if spec_token_ids 非空:                   # L29-36
    │       │   ★ 走 RejectionSampler 路径 ★
    │       │   └─ RejectionSampler.forward()           # rejection_sampler.py:55-61
    │       │       ├─ 仅支持 greedy（否则抛 NotImplementedError）
    │       │       └─ self.forward_method(logits, sampling_metadata)
    │       │           │
    │       │           ├─ [FlashInfer] flashinfer_sample()      # L63-115
    │       │           │   ├─ argmax 得到 target_token_ids
    │       │           │   ├─ 构建 draft_probs, target_probs (one-hot)
    │       │           │   └─ fs.chain_speculative_sampling()
    │       │           │
    │       │           └─ [Native] forward_native()             # L118-166
    │       │               ├─ output_token_ids = logits.argmax()
    │       │               ├─ ★ accept_mask = (target[:,:-1] == spec).cumprod(dim=1)
    │       │               │   （逐位比较，一旦不匹配，后续全部拒绝）
    │       │               ├─ 生成 generate_mask（含 bonus token）
    │       │               │   bonus = 第一个拒绝位置用 target 的选择
    │       │               └─ output[~generate_mask] = INVALID_TOKEN_ID (-1)
    │       │
    │       └─ if spec_token_ids 为空:                   # L38-73
    │           走 普通 Sampler 路径
    │           ├─ apply_logits_bias()
    │           ├─ apply_penalties()
    │           ├─ sample() → greedy_sample / topk_topp
    │           └─ 返回 SamplerOutput(sampled_token_ids=[batch,1])
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 7: GPU→CPU 同步 + 后处理                                     │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ logprobs 处理                                     # L978-980
    ├─ prompt_logprobs 处理                              # L983-986
    │
    ├─ ★ 提取 valid sampled tokens ★                    # L988-1002
    │   if max_gen_len == 1:  # 无 spec decode
    │       valid_sampled_token_ids = sampled_token_ids.tolist()
    │   else:  # 有 spec decode
    │       valid_mask = sampled_token_ids != INVALID_TOKEN_ID
    │       valid_sampled_token_ids = [过滤后的有效 token 列表]
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 8: 生成下一步 Draft Tokens ★★★ 核心优化点 ★★★                 │
    │  └──────────────────────────────────────────────────────────────────┘
    ├─ if self.use_spec_decode:                          # L1004-1008
    │   spec_token_ids = self.generate_draft_token_ids(valid_sampled_token_ids)
    │   │
    │   └─ generate_draft_token_ids()                    # L1020-1046
    │       for i, sampled_ids in enumerate(sampled_token_ids):
    │         │
    │         ├─ start_idx = input_batch.num_tokens_no_spec[i]  # L1034
    │         ├─ token_ids_cpu[i, start:end] = sampled_ids      # L1036
    │         │   （将本步 accepted tokens 写入 CPU buffer）
    │         │
    │         ├─ ★ drafter_output = self.drafter.propose(       # L1037-1041
    │         │       token_ids_cpu[i, :end_idx],
    │         │       ngram_prompt_lookup_min,       # n: 匹配窗口
    │         │       num_speculative_tokens,        # k: 期望 draft 长度
    │         │   )
    │         │   └─ NgramProposer.propose()         # ngram_proposer.py:10-42
    │         │       └─ _find_subarray_kmp()         # ngram_proposer.py:70-103
    │         │           │  ★★ 这是我们要替换的核心函数 ★★
    │         │           ├─ pattern = context[-n:]     # 取最后 n 个 token
    │         │           ├─ lps = _kmp_lps_array(pattern)  # KMP 前缀函数
    │         │           ├─ KMP 搜索 context[:-n] 中的 pattern
    │         │           │   while i < context_len - n:
    │         │           │       if match → return context[i:i+k]
    │         │           │       （找到第一个匹配就返回后续 k 个 token）
    │         │           └─ return None  # 未找到匹配
    │         │
    │         └─ draft_token_ids.append(output.tolist() or [])
    │
    │  ┌──────────────────────────────────────────────────────────────────┐
    │  │ 阶段 9: 构建并返回输出                                             │
    │  └──────────────────────────────────────────────────────────────────┘
    └─ return ModelRunnerOutput(                          # L1010-1018
           req_ids=...,
           req_id_to_index=...,
           sampled_token_ids=valid_sampled_token_ids,    # 本步有效 tokens
           spec_token_ids=spec_token_ids,                # 下一步的 draft
           logprobs=...,
           prompt_logprobs_dict=...,
       )
       # 定义: vllm/v1/outputs.py:57-82
```

#### 2.0.3 Spec Decode 初始化路径

```
GPUModelRunner.__init__()                             # gpu_model_runner.py:48-233
  └─ if self.speculative_config:                       # L122
       self.use_spec_decode = True                     # L123
       assert speculative_config.ngram_prompt_lookup_min  # L126
       ★ "Currently, only ngram spec decode is supported in V1." ★
       self.drafter = NgramProposer()                  # L129
       # Numba JIT 预热
       self.drafter.propose(np.zeros(1024), min, num)  # L132-136
```

#### 2.0.4 关键数据结构内存布局

```
InputBatch (gpu_input_batch.py:48-547):
  │
  ├─ token_ids_cpu: np.ndarray [max_num_reqs, max_model_len] int32
  │   持久化 CPU buffer，存储每个请求的 prompt + output + spec tokens
  │   ★ spec tokens 直接追加在 output tokens 后面 ★
  │
  ├─ num_tokens: np.ndarray [max_num_reqs] int32
  │   每个请求的总 token 数（包含 spec tokens）
  │   用于 _prepare_inputs 构建模型输入
  │
  ├─ num_tokens_no_spec: np.ndarray [max_num_reqs] int32
  │   每个请求不含 spec tokens 的真实 token 数
  │   用于 generate_draft_token_ids 确定写入起点
  │
  ├─ num_computed_tokens_cpu: np.ndarray [max_num_reqs] int32
  │   每个请求已经 forward 过的 token 数
  │   用于计算 positions 和确定新调度的 token 范围
  │
  └─ block_table: BlockTable
      KV Cache 的 block 映射表

SamplingMetadata (sample/metadata.py:10-39):
  └─ spec_token_ids: Optional[List[List[int]]]
      每个请求的 spec token IDs
      None → 走普通 Sampler
      非空 → 走 RejectionSampler

ModelRunnerOutput (outputs.py:57-82):
  ├─ sampled_token_ids: List[List[int]]    # 本步有效 token（accepted + bonus）
  └─ spec_token_ids: Optional[List[List[int]]]  # 下一步的 draft tokens
      → 传回 Scheduler，由 Scheduler 在下一步调度时使用
```

#### 2.0.5 一个完整 Step 的时间线（Spec Decode 模式）

```
时间 ─────────────────────────────────────────────────────────────→

  │← _update_states ─────→│    写入 spec tokens 到 token_ids_cpu
  │                        │← _prepare_inputs ──→│   构建 GPU 张量 + logits_indices
  │                        │                      │← model.forward ──────────→│
  │                        │                      │   (GPU) ~300ms             │
  │                        │                      │← compute_logits ──→│
  │                        │                      │← Sampler ────→│
  │                        │                      │  RejectionSampler: accept/reject
  │                        │                      │← GPU→CPU sync ──→│
  │                        │                      │← generate_draft ──→│
  │                        │                      │  NgramProposer.propose()
  │                        │                      │  ★ O(context_len) ← 优化目标 ★
  │                        │                      │← 构建 Output ──→│
  │                        │                      │                   │→ 返回给 Scheduler
```

---

### 2.1 数据流全景

```
Step N 完整流程：

  ┌─ Scheduler.schedule() ──────────────────────────────────────────┐
  │ 1. 遍历 RUNNING 请求                                             │
  │ 2. num_new_tokens = num_tokens_with_spec - num_computed_tokens   │
  │    ↑ num_tokens_with_spec = prompt + output + spec_token_ids     │
  │ 3. 从 token_budget 中扣减                                        │
  │ 4. 提取 scheduled_spec_decode_tokens[req_id] = spec_token_ids   │
  └────────────────────┬────────────────────────────────────────────┘
                       ↓
  ┌─ GPUModelRunner._update_states() ──────────────────────────────┐
  │ 5. token_ids_cpu[i, start:end] = new_token_ids  (正常 tokens)   │
  │ 6. num_tokens_no_spec[i] = end                                  │
  │ 7. token_ids_cpu[i, end:end+spec_len] = spec_tokens (追加)      │
  │ 8. num_tokens[i] = end + spec_len                               │
  └────────────────────┬────────────────────────────────────────────┘
                       ↓
  ┌─ GPUModelRunner._prepare_inputs() ─────────────────────────────┐
  │ 9. 检测是否有 spec decode tokens                                 │
  │ 10. 有 spec → _calc_spec_decode_metadata():                     │
  │     计算 logits_indices（每请求 spec_len+1 个位置）              │
  │ 11. 无 spec → 每请求只取最后 1 个位置的 logits                    │
  └────────────────────┬────────────────────────────────────────────┘
                       ↓
  ┌─ 模型前向传播 ────────────────────────────────────────────────────┐
  │ 12. 一次性对所有 tokens（normal + spec）做 forward                │
  │ 13. 用 logits_indices 提取需要采样的位置的 hidden_states          │
  │ 14. 计算 logits                                                  │
  └────────────────────┬────────────────────────────────────────────┘
                       ↓
  ┌─ RejectionSampler.forward() ──────────────────────────────────┐
  │ 15. target_tokens = argmax(logits) (greedy)                    │
  │ 16. 逐位比较 target_tokens vs spec_token_ids:                  │
  │     accept_mask = (target[:, :-1] == spec).cumprod(dim=1)      │
  │     → 一旦不匹配，后续全部拒绝                                   │
  │ 17. 加上 bonus token（第一个拒绝位置用 target 的选择）           │
  │ 18. 不接受的位置填 INVALID_TOKEN_ID (-1)                        │
  └────────────────────┬────────────────────────────────────────────┘
                       ↓
  ┌─ generate_draft_token_ids() ──────────────────────────────────┐
  │ 19. 将 accepted tokens 追加到 token_ids_cpu                    │
  │ 20. 对每个请求调用 NgramProposer.propose():                    │
  │     propose(token_ids_cpu[i, :end], n=ngram_min, k=num_spec)  │
  │     → 在 context 中 KMP 搜索最后 n 个 token 的匹配             │
  │     → 返回匹配后续最多 k 个 tokens                             │
  │ 21. 返回 List[List[int]] → ModelRunnerOutput.spec_token_ids   │
  └────────────────────┬────────────────────────────────────────────┘
                       ↓
  ┌─ Scheduler.update_from_output() ──────────────────────────────┐
  │ 22. num_computed_tokens += 调度的 - 被拒绝的                    │
  │ 23. request.spec_token_ids = new_draft (为下一步准备)           │
  │ 24. 追加 generated tokens 到 output_token_ids                  │
  └─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键接口（实现后缀解码需要对接的）

```python
# ---- NgramProposer 当前接口 (ngram_proposer.py) ----
class NgramProposer:
    def propose(
        self,
        context_token_ids: np.ndarray,  # [prompt + output + sampled], int32
        n: int,                          # n-gram 匹配窗口
        k: int,                          # 期望 draft 长度
    ) -> Optional[np.ndarray]:           # [<=k] int32 或 None
        ...

# ---- 调用方 (gpu_model_runner.py, generate_draft_token_ids) ----
def generate_draft_token_ids(
    self,
    sampled_token_ids: List[List[int]],  # 本步每个请求采样的 token
) -> List[List[int]]:                     # 返回每个请求的 draft tokens
    for i in range(batch_size):
        # 追加 sampled tokens 到 token_ids_cpu
        ...
        draft = self.drafter.propose(
            token_ids_cpu[i, :end_idx],
            self.speculative_config.ngram_prompt_lookup_min,
            self.speculative_config.num_speculative_tokens,
        )
        results.append(draft.tolist() if draft is not None else [])

# ---- 调度器消费 (scheduler.py) ----
# request.spec_token_ids: List[int]  ← 来自 ModelRunnerOutput
# request.num_tokens_with_spec       ← prompt + output + spec 总长度
```

### 2.3 性能约束

`generate_draft_token_ids()` 位于**模型前向传播后、结果返回前**的关键路径上：

```
时间线：
  ├── 模型 forward ──── 300ms ────┤
  ├── sampling ──── 5ms ─────────┤
  ├── generate_draft ── ??ms ────┤ ← 这里的延迟直接增加端到端延迟
  └── 返回结果 ──────────────────┘

当前 NgramProposer: O(context_len) per request
目标 SuffixTreeProposer: O(pattern_len) per request（pattern_len << context_len）
```

---

## 三、优化点详细设计

### 优化点 1：SuffixTreeProposer 基础实现 `[核心]` `[已实现 ✅]`

#### 问题分析

当前 `NgramProposer.propose()` 的核心问题：

```python
# 当前实现（简化）
def propose(self, context, n, k):
    pattern = context[-n:]           # 取最后 n 个 token
    idx = kmp_search(context[:-n], pattern)  # O(context_len) 搜索
    if idx >= 0:
        return context[idx+n : idx+n+k]     # 返回匹配后续
    return None
```

1. **固定窗口**：只用固定的 `n` 值，如果 n=3 匹配不到就直接放弃
2. **首次匹配**：KMP 返回第一个匹配，不一定是最好的
3. **全量搜索**：每次都从头遍历 context，O(context_len)

#### 设计方案

**用后缀树（Generalized Suffix Tree）替换 KMP 搜索**：

后缀树的核心优势：
- **构建一次，查询多次**：O(n) 构建，O(m) 查询（m 为 pattern 长度）
- **天然支持可变长度匹配**：沿着树走到走不动为止，就是最长匹配
- **所有匹配位置**：叶子节点告诉你所有出现位置，可以选最优

```python
# 新文件: vllm/v1/spec_decode/suffix_proposer.py

import numpy as np
from typing import Optional, List, Tuple

class SuffixTreeNode:
    """后缀树节点（使用 Ukkonen 算法的隐式后缀树表示）"""
    __slots__ = ['children', 'suffix_link', 'start', 'end', 'leaf_index']
    
    def __init__(self, start: int = -1, end: int = -1):
        self.children: dict[int, 'SuffixTreeNode'] = {}  # token_id → child
        self.suffix_link: Optional['SuffixTreeNode'] = None
        self.start = start    # 边标签在原始序列中的起始位置
        self.end = end        # 边标签的结束位置（-1 表示活叶，动态增长）
        self.leaf_index = -1  # 叶子节点：该后缀在原始序列中的起始位置


class SuffixTreeProposer:
    """基于后缀树的投机解码 Proposer。
    
    核心优势（相比 NgramProposer）：
    1. 查询时间 O(m) 而非 O(n)，m = 匹配模式长度，n = 上下文长度
    2. 自动找最长匹配，无需指定固定 n
    3. 返回所有匹配位置，可选最优
    
    与 NgramProposer 保持相同的对外接口，可直接替换。
    """
    
    def __init__(self):
        self._root: Optional[SuffixTreeNode] = None
        self._text: Optional[np.ndarray] = None
        self._text_len: int = 0
        
        # 预热（与 NgramProposer 保持一致）
        dummy = np.zeros(64, dtype=np.int32)
        self._build_suffix_structure(dummy)
        self.propose(dummy, 3, 5)
    
    def _build_suffix_structure(self, text: np.ndarray):
        """构建后缀数组 + LCP 数组（实际实现中可能更高效）。
        
        这里使用简化的后缀数组实现（SA-IS 或 DC3 算法），
        相比 Ukkonen 后缀树更易实现且缓存友好。
        
        完整的 Ukkonen 后缀树实现可作为后续优化。
        """
        self._text = text
        self._text_len = len(text)
        
        # 构建后缀数组（Suffix Array）
        # SA[i] = 第 i 小的后缀在原文中的起始位置
        self._sa = self._build_suffix_array(text)
        
        # 构建 LCP 数组（Longest Common Prefix）
        # LCP[i] = SA[i] 和 SA[i-1] 对应后缀的最长公共前缀长度
        self._lcp = self._build_lcp_array(text, self._sa)
    
    def _build_suffix_array(self, text: np.ndarray) -> np.ndarray:
        """使用 Python sorted 构建后缀数组（简化版）。
        
        生产环境应使用 SA-IS 算法（O(n) 时间，O(n) 空间），
        或使用 Cython/Numba 加速。
        """
        n = len(text)
        # 构建 (后缀起始位置) 列表，按后缀字典序排序
        sa = sorted(range(n), key=lambda i: text[i:].tobytes())
        return np.array(sa, dtype=np.int32)
    
    def _build_lcp_array(self, text: np.ndarray, sa: np.ndarray) -> np.ndarray:
        """使用 Kasai 算法构建 LCP 数组，O(n) 时间。"""
        n = len(text)
        rank = np.zeros(n, dtype=np.int32)
        lcp = np.zeros(n, dtype=np.int32)
        
        for i in range(n):
            rank[sa[i]] = i
        
        h = 0
        for i in range(n):
            if rank[i] > 0:
                j = sa[rank[i] - 1]
                while (i + h < n and j + h < n 
                       and text[i + h] == text[j + h]):
                    h += 1
                lcp[rank[i]] = h
                if h > 0:
                    h -= 1
            else:
                h = 0
        
        return lcp
    
    def _find_longest_match(
        self, 
        pattern: np.ndarray,
        min_match_len: int = 2,
    ) -> List[Tuple[int, int]]:
        """在后缀数组中查找 pattern 的所有匹配位置。
        
        使用二分搜索在 SA 中定位 pattern 的范围，
        然后返回所有匹配位置及其后续可用长度。
        
        Args:
            pattern: 要匹配的 token 序列
            min_match_len: 最少匹配长度
            
        Returns:
            List[(match_position, available_continuation_length)]
            按 available_continuation_length 降序排列
        """
        # 二分搜索找到 pattern 在 SA 中的范围 [lo, hi)
        lo = self._sa_lower_bound(pattern)
        hi = self._sa_upper_bound(pattern)
        
        if lo >= hi:
            return []
        
        results = []
        pattern_len = len(pattern)
        
        for idx in range(lo, hi):
            pos = self._sa[idx]
            # 匹配位置之后还有多少 tokens 可用
            continuation_len = self._text_len - (pos + pattern_len)
            if continuation_len > 0:
                results.append((pos + pattern_len, continuation_len))
        
        # 按可用续接长度降序排列（优先选能提供更多 draft tokens 的匹配）
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _sa_lower_bound(self, pattern: np.ndarray) -> int:
        """在后缀数组中二分查找 pattern 的下界。"""
        lo, hi = 0, len(self._sa)
        plen = len(pattern)
        while lo < hi:
            mid = (lo + hi) // 2
            pos = self._sa[mid]
            # 比较 text[pos:pos+plen] 与 pattern
            suffix = self._text[pos:pos+plen]
            if len(suffix) < plen:
                # 后缀比 pattern 短，按字典序一定 < pattern（如果前缀相同的话）
                # 简化处理：逐元素比较
                cmp = self._compare_arrays(suffix, pattern[:len(suffix)])
                if cmp < 0 or (cmp == 0):
                    lo = mid + 1
                else:
                    hi = mid
            else:
                cmp = self._compare_arrays(suffix, pattern)
                if cmp < 0:
                    lo = mid + 1
                else:
                    hi = mid
        return lo
    
    def _sa_upper_bound(self, pattern: np.ndarray) -> int:
        """在后缀数组中二分查找 pattern 的上界。"""
        lo, hi = 0, len(self._sa)
        plen = len(pattern)
        while lo < hi:
            mid = (lo + hi) // 2
            pos = self._sa[mid]
            suffix = self._text[pos:pos+plen]
            if len(suffix) < plen:
                cmp = self._compare_arrays(suffix, pattern[:len(suffix)])
                if cmp <= 0:
                    lo = mid + 1
                else:
                    hi = mid
            else:
                cmp = self._compare_arrays(suffix, pattern)
                if cmp <= 0:
                    lo = mid + 1
                else:
                    hi = mid
        return lo
    
    @staticmethod
    def _compare_arrays(a: np.ndarray, b: np.ndarray) -> int:
        """字典序比较两个数组。返回 -1, 0, 1。"""
        min_len = min(len(a), len(b))
        for i in range(min_len):
            if a[i] < b[i]:
                return -1
            elif a[i] > b[i]:
                return 1
        if len(a) < len(b):
            return -1
        elif len(a) > len(b):
            return 1
        return 0
    
    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,                          # 保持接口兼容：最小匹配长度
        k: int,                          # 期望 draft 长度
    ) -> Optional[np.ndarray]:
        """生成 draft token 提案（与 NgramProposer 相同接口）。
        
        算法：
        1. 取 context 最后 n 个 token 作为初始 pattern
        2. 在 context[:-1] 中查找所有匹配（排除最后位置本身）
        3. 如果找到，选择续接最长的匹配，返回后续 k 个 tokens
        4. 如果未找到，缩短 pattern 重试（自适应回退）
        """
        if len(context_token_ids) < n + 1:
            return None
        
        # 构建后缀结构（基于 context[:-1]，排除最后位置避免自匹配）
        search_text = context_token_ids[:-1]
        self._build_suffix_structure(search_text)
        
        # 自适应匹配：从 n 开始尝试，逐步缩短
        for match_len in range(n, max(1, n - 2), -1):
            pattern = context_token_ids[-match_len:]
            matches = self._find_longest_match(pattern, match_len)
            
            if matches:
                # 选续接最长的匹配
                best_pos, best_cont_len = matches[0]
                actual_k = min(k, best_cont_len)
                return search_text[best_pos:best_pos + actual_k]
        
        return None
```

#### 实际实现

已实现文件：`vllm/v1/spec_decode/suffix_proposer.py`（309 行）
测试文件：`tests/v1/spec_decode/test_suffix_proposer.py`（224 行）

实现要点：
- 使用 **Numba JIT 编译** 的后缀数组 + 二分搜索，替代设计方案中的纯 Python 后缀数组
- 后缀数组构建采用 **迭代倍增 + Shell 排序**（O(n·log²n)），JIT 兼容
- 二分搜索实现 `_sa_lower_bound` / `_sa_upper_bound`，精确定位所有匹配位置
- 自适应回退：从 n 到 max(2, n//2) 逐步缩短匹配长度
- 最优匹配选择：遍历所有匹配位置，选择续接最长的（跳过自匹配）
- 保持与 `NgramProposer` 完全相同的外部接口 `propose(context, n, k)`，可直接替换

测试覆盖：
- 基础匹配、无匹配、短上下文、重复模式等功能测试
- 自适应回退行为验证
- 与 NgramProposer 的对比测试（30 轮随机数据，确保不遗漏匹配）
- 边界情况：全相同 token、大上下文（2000 tokens）、n=1、dtype 验证

#### 修改文件
- 新增 `vllm/v1/spec_decode/suffix_proposer.py` — 后缀数组 Proposer（Numba JIT）
- 新增 `tests/v1/spec_decode/test_suffix_proposer.py` — 完整测试套件

#### 预期效果
- 查询时间从 O(context_len) 降到 O(pattern_len * log(context_len))
- 自适应匹配长度：不再受固定 n 值限制
- 选择最优匹配：从"第一个匹配"变为"续接最长的匹配"

#### 涉及的 vLLM 知识点
- `NgramProposer` 的接口设计：`propose(context, n, k) → Optional[np.ndarray]`
- `GPUModelRunner.generate_draft_token_ids()` 的调用链
- `SpeculativeConfig` 的参数传递（`ngram_prompt_lookup_min`, `num_speculative_tokens`）
- V1 spec decode 的 assert 限制（当前只允许 ngram）

---

### 优化点 2：增量后缀结构更新 `[核心]` `[已实现 ✅]`

#### 问题分析

优化点 1 中的实现有一个明显问题：**每次 `propose()` 都完整重建后缀结构**。在自回归解码中，每步只新增 1-few 个 token，全量重建浪费大量时间。

```
Step 1: context = [t1, t2, ..., t100]           → 重建 SA，O(100)
Step 2: context = [t1, t2, ..., t100, t101]     → 重建 SA，O(101) ← 浪费
Step 3: context = [t1, t2, ..., t100, t101, t102] → 重建 SA，O(102) ← 更浪费
```

#### 设计方案

**引入有状态的后缀自动机（Suffix Automaton / SAM）替代后缀数组**：

后缀自动机（SAM）的核心优势：
- **O(1) 在线追加**：每添加一个字符/token，只需 O(1) 均摊时间更新
- **O(m) 查询**：查找长度为 m 的模式只需 O(m) 时间
- **空间 O(n)**：节点数 ≤ 2n，转移数 ≤ 3n

```python
class SAMNode:
    """后缀自动机节点。"""
    __slots__ = ['len', 'link', 'transitions', 'first_end_pos']
    
    def __init__(self, length: int = 0):
        self.len = length                    # 该状态对应的最长子串长度
        self.link: int = -1                  # 后缀链接（suffix link）
        self.transitions: dict[int, int] = {}  # token_id → node_id
        self.first_end_pos: int = -1         # 该状态首次出现的结束位置


class IncrementalSuffixAutomaton:
    """增量后缀自动机，支持 O(1) 在线追加 token。
    
    每次 decode 新增 token 时调用 extend()，
    查找时调用 find_longest_match()。
    """
    
    def __init__(self):
        self.nodes: List[SAMNode] = [SAMNode(0)]  # 初始节点
        self._last: int = 0  # 上一次追加后的当前节点
        self._size: int = 1
    
    def extend(self, token_id: int):
        """追加一个 token，O(1) 均摊更新后缀自动机。
        
        这是 Suffix Automaton 的经典在线构建算法。
        """
        cur = self._size
        self.nodes.append(SAMNode(self.nodes[self._last].len + 1))
        self._size += 1
        self.nodes[cur].first_end_pos = self.nodes[cur].len - 1
        
        p = self._last
        while p != -1 and token_id not in self.nodes[p].transitions:
            self.nodes[p].transitions[token_id] = cur
            p = self.nodes[p].link
        
        if p == -1:
            self.nodes[cur].link = 0
        else:
            q = self.nodes[p].transitions[token_id]
            if self.nodes[p].len + 1 == self.nodes[q].len:
                self.nodes[cur].link = q
            else:
                # Clone 节点
                clone = self._size
                self.nodes.append(SAMNode(self.nodes[p].len + 1))
                self._size += 1
                self.nodes[clone].link = self.nodes[q].link
                self.nodes[clone].transitions = dict(self.nodes[q].transitions)
                self.nodes[clone].first_end_pos = self.nodes[q].first_end_pos
                
                while p != -1 and self.nodes[p].transitions.get(token_id) == q:
                    self.nodes[p].transitions[token_id] = clone
                    p = self.nodes[p].link
                
                self.nodes[q].link = clone
                self.nodes[cur].link = clone
        
        self._last = cur
    
    def find_longest_match(
        self, 
        pattern: np.ndarray,
        text: np.ndarray,
        k: int,
    ) -> Optional[np.ndarray]:
        """在自动机中查找 pattern 的最长匹配，返回后续 k 个 tokens。
        
        Args:
            pattern: 要匹配的 token 序列（通常是最后几个 token）
            text: 原始文本（用于提取匹配后续的 tokens）
            k: 期望的 draft 长度
            
        Returns:
            匹配后续的 k 个 tokens，或 None
        """
        # 沿着 SAM 的转移边走，直到走不动
        node = 0
        matched_len = 0
        
        for token in pattern:
            if token in self.nodes[node].transitions:
                node = self.nodes[node].transitions[token]
                matched_len += 1
            else:
                break
        
        if matched_len < 2:  # 至少匹配 2 个 token 才有意义
            return None
        
        # 找到匹配位置后，提取后续 tokens
        end_pos = self.nodes[node].first_end_pos
        if end_pos < 0:
            return None
        
        next_start = end_pos + 1
        available = len(text) - next_start
        if available <= 0:
            return None
        
        actual_k = min(k, available)
        return text[next_start:next_start + actual_k]


class StatefulSuffixProposer:
    """有状态的后缀 Proposer，跨步复用后缀自动机。
    
    与 NgramProposer 的无状态设计不同，本 Proposer 为每个请求
    维护一个增量更新的后缀自动机。
    """
    
    def __init__(self):
        # 每个请求的后缀自动机
        # key: 在 input_batch 中的 index
        self._automata: dict[int, IncrementalSuffixAutomaton] = {}
        self._prev_len: dict[int, int] = {}  # 记录上次构建到的位置
    
    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
        req_index: int = 0,  # 新增参数：请求索引
    ) -> Optional[np.ndarray]:
        """生成 draft token 提案。
        
        首次调用时构建完整 SAM，后续调用只增量追加新 tokens。
        """
        text_len = len(context_token_ids)
        
        if req_index not in self._automata:
            # 首次构建
            sam = IncrementalSuffixAutomaton()
            for token in context_token_ids[:-1]:
                sam.extend(int(token))
            self._automata[req_index] = sam
            self._prev_len[req_index] = text_len - 1
        else:
            # 增量追加
            sam = self._automata[req_index]
            prev = self._prev_len[req_index]
            for token in context_token_ids[prev:text_len - 1]:
                sam.extend(int(token))
            self._prev_len[req_index] = text_len - 1
        
        # 查找匹配
        pattern = context_token_ids[-n:]
        return sam.find_longest_match(
            pattern, context_token_ids[:-1], k
        )
    
    def remove_request(self, req_index: int):
        """请求完成时清理其后缀自动机。"""
        self._automata.pop(req_index, None)
        self._prev_len.pop(req_index, None)
```

#### 实际实现

已实现文件：`vllm/v1/spec_decode/suffix_automaton_proposer.py`（359 行）
测试文件：`tests/v1/spec_decode/test_suffix_automaton_proposer.py`（581 行）

实现要点：
- 完整实现 **在线后缀自动机（SAM）**：`_SAMNode` + `IncrementalSuffixAutomaton`
- `extend()` 方法实现标准 SAM 在线构建算法，支持 O(1) 均摊追加
- 包含 clone 机制处理后缀链接分裂
- `find_longest_match()` 和 `find_all_match_lengths()` 两种查询方式
- `SuffixAutomatonProposer` 为每个请求维护独立 SAM（使用 **req_id 字符串** 而非 req_index，避免 batch condense 时索引变动）
- 增量更新：首次调用 O(n) 构建，后续调用仅追加新 token O(new_tokens)
- 上下文收缩检测：如果 context 变短（如抢占后恢复），自动重建 SAM
- `remove_request()` 清理已完成请求的 SAM 状态，防止内存泄漏

测试覆盖（581 行，7 个测试类）：
- `TestIncrementalSuffixAutomaton`：SAM 核心数据结构单元测试（9 个用例）
- `TestSuffixAutomatonProposerBasic`：基础功能测试（7 个用例）
- `TestSuffixAutomatonProposerIncremental`：增量更新正确性（5 个用例，含压力测试）
- `TestSuffixAutomatonProposerLifecycle`：请求生命周期管理（4 个用例，含 50 请求批量测试）
- `TestSuffixAutomatonProposerAdaptive`：自适应回退（2 个用例）
- `TestSuffixAutomatonVsOthers`：与 NgramProposer/SuffixTreeProposer 对比（3 个用例，200 轮随机数据）
- `TestSuffixAutomatonProposerPerformance`：大上下文（4000 tokens）、增量更新性能、100 并发请求

#### 修改文件
- 新增 `vllm/v1/spec_decode/suffix_automaton_proposer.py` — 增量 SAM + `SuffixAutomatonProposer`
- 新增 `tests/v1/spec_decode/test_suffix_automaton_proposer.py` — 完整测试套件

#### 预期效果
- 增量更新：每步 O(new_tokens) 而非 O(context_len) 重建
- 长上下文场景（4K+ tokens）性能提升显著：从 O(n) 降到 O(1) 每步
- 内存 O(n)：节点数不超过 2n

#### 涉及的 vLLM 知识点
- `GPUModelRunner.generate_draft_token_ids()` 的逐请求循环结构
- `input_batch.token_ids_cpu` 的内存布局（`[batch, max_num_tokens]` 的 numpy 数组）
- `num_tokens_no_spec` vs `num_tokens` 的区别
- 请求的生命周期：何时创建/何时完成（对应 SAM 的创建/清理时机）

---

### 优化点 3：自适应匹配策略与多候选评分 `[重要]` `[已实现 ✅]`

#### 问题分析

即使有了高效的后缀结构，匹配策略仍然可以优化：

```
场景 1：固定 n=4 匹配
  context: [..., "用", "户", "你", "好", "请", ...]
  最后 4 个 token: ["你", "好", "请", "问"]
  n=4 匹配不到 → 放弃 → 本步 0 个 spec tokens

  但 n=2 ["请", "问"] 可以在 context 中找到 → 产生 draft tokens
  → 固定 n 值错失了机会

场景 2：多个匹配位置
  context 中 "请问" 出现在位置 50、120、300
  位置 50 后面是: "天气怎么样"
  位置 120 后面是: "今天有什么新闻"
  位置 300 后面是: "你能帮我" ← 最近的匹配

  哪个是最好的 draft？不一定是续接最长的，也不一定是最近的
```

#### 设计方案

**多级回退匹配 + 历史接受率加权的候选评分**：

```python
class AdaptiveSuffixProposer(StatefulSuffixProposer):
    """自适应后缀 Proposer：多级回退 + 候选评分。"""
    
    def __init__(self):
        super().__init__()
        # 历史接受率统计（用于评估匹配质量）
        self._accept_stats: dict[int, AcceptanceTracker] = {}
    
    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,       # 最大匹配长度
        k: int,       # 期望 draft 长度
        req_index: int = 0,
    ) -> Optional[np.ndarray]:
        """自适应提案：多级回退匹配 + 候选评分。"""
        # 增量更新 SAM（复用父类逻辑）
        self._ensure_sam_updated(context_token_ids, req_index)
        sam = self._automata[req_index]
        search_text = context_token_ids[:-1]
        
        best_draft = None
        best_score = -1.0
        
        # 多级回退：从 n 开始，逐步缩短匹配长度
        for match_len in range(n, max(1, n // 2) - 1, -1):
            pattern = context_token_ids[-match_len:]
            
            # 获取所有匹配位置
            matches = self._find_all_matches(sam, pattern, search_text)
            
            for match_pos, cont_len in matches:
                # 计算候选分数
                score = self._score_candidate(
                    match_pos=match_pos,
                    match_len=match_len,
                    cont_len=cont_len,
                    context_len=len(context_token_ids),
                    req_index=req_index,
                )
                
                if score > best_score:
                    actual_k = min(k, cont_len)
                    best_draft = search_text[match_pos:match_pos + actual_k]
                    best_score = score
        
        return best_draft
    
    def _score_candidate(
        self,
        match_pos: int,
        match_len: int,
        cont_len: int,
        context_len: int,
        req_index: int,
    ) -> float:
        """候选评分函数。
        
        综合考虑：
        1. 匹配长度（越长越可能是相关上下文）
        2. 续接长度（越长能提供越多 draft tokens）
        3. 位置新旧（越近越可能与当前上下文相关）
        4. 历史接受率（该匹配长度的 draft 平均接受了多少）
        """
        # 归一化各因子到 [0, 1]
        match_score = min(match_len / 8.0, 1.0)          # 匹配长度，8 封顶
        cont_score = min(cont_len / 10.0, 1.0)           # 续接长度，10 封顶
        recency = match_pos / max(1, context_len)         # 位置越大（越近）越好
        
        # 历史接受率
        accept_rate = self._get_accept_rate(req_index, match_len)
        
        # 加权求和
        return (0.3 * match_score + 
                0.2 * cont_score + 
                0.2 * recency + 
                0.3 * accept_rate)
    
    def _get_accept_rate(self, req_index: int, match_len: int) -> float:
        """获取某匹配长度的历史接受率。"""
        tracker = self._accept_stats.get(req_index)
        if tracker is None:
            return 0.5  # 默认中性评估
        return tracker.get_rate(match_len)
    
    def update_acceptance(
        self, 
        req_index: int, 
        match_len: int,
        num_proposed: int, 
        num_accepted: int,
    ):
        """更新历史接受率（在 RejectionSampler 结果回来后调用）。"""
        if req_index not in self._accept_stats:
            self._accept_stats[req_index] = AcceptanceTracker()
        self._accept_stats[req_index].record(match_len, num_proposed, num_accepted)


class AcceptanceTracker:
    """滑动窗口接受率追踪器。"""
    
    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        # match_len → deque of (proposed, accepted)
        self._history: dict[int, list] = {}
    
    def record(self, match_len: int, proposed: int, accepted: int):
        if match_len not in self._history:
            self._history[match_len] = []
        history = self._history[match_len]
        history.append((proposed, accepted))
        if len(history) > self._window_size:
            history.pop(0)
    
    def get_rate(self, match_len: int) -> float:
        history = self._history.get(match_len, [])
        if not history:
            return 0.5
        total_proposed = sum(p for p, _ in history)
        total_accepted = sum(a for _, a in history)
        return total_accepted / max(1, total_proposed)
```

#### 实际实现

已实现文件：`vllm/v1/spec_decode/adaptive_suffix_proposer.py`（363 行）
测试文件：`tests/v1/spec_decode/test_adaptive_suffix_proposer.py`（372 行）

实现要点：
- `AdaptiveSuffixProposer` 继承 `SuffixAutomatonProposer`，复用增量 SAM 能力
- **AcceptanceTracker**：滑动窗口（默认 20）接受率追踪器，使用 `deque(maxlen=N)` 自动驱逐旧记录
- **多候选搜索** `_find_all_candidates()`：先通过 SAM 快速验证模式存在（O(m)），再线性扫描找出所有出现位置
- **四因子加权评分** `_score_candidate()`：
  - `W_MATCH=0.25`：匹配长度（cap=8）
  - `W_CONT=0.20`：续接长度（cap=10）
  - `W_RECENCY=0.25`：位置新旧（越近越好）
  - `W_ACCEPT=0.30`：历史接受率（权重最高）
- **跨级别最优选择**：遍历所有回退级别的所有候选，选全局最高分
- **接受反馈闭环** `update_acceptance()`：记录上次提案的 (match_len, num_proposed)，接收 num_accepted 更新 tracker
- `remove_request()` 清理 SAM + tracker + last_proposal 三份状态

测试覆盖（15 个测试用例）：
- 基础匹配、无匹配、多候选偏好最近位置（验证 recency 评分有效）
- 自适应回退、AcceptanceTracker 单元测试（含窗口滑动验证）
- 接受反馈闭环验证、增量更新一致性、请求生命周期
- 多请求独立性、20 轮增量 vs 重建压力测试
- 200 轮随机数据匹配率对比（Adaptive ≥ SAM）
- 大上下文性能基准（4000 tokens 初始构建 + 增量追加）

#### 修改文件
- 新增 `vllm/v1/spec_decode/adaptive_suffix_proposer.py` — `AdaptiveSuffixProposer` + `AcceptanceTracker`
- 新增 `tests/v1/spec_decode/test_adaptive_suffix_proposer.py` — 完整测试套件

#### 预期效果
- 接受率提升：从"首次匹配"到"最优匹配"，预期接受率提升 15-30%
- 自适应回退：固定 n 匹配不到时自动缩短，减少"0 draft"的情况
- 历史反馈：根据实际接受情况动态调整评分权重

#### 涉及的 vLLM 知识点
- `RejectionSampler` 的验证机制：`cumprod` 逐位比较 + bonus token
- `ModelRunnerOutput` 中 `sampled_token_ids` 的含义（accepted + bonus - rejected = valid）
- `INVALID_TOKEN_ID` 的过滤逻辑

---

### 优化点 4：跨请求共享后缀树 `[进阶]` `[方案设计 ⬜]`

#### 问题分析

优化点 1-3 的后缀结构是**每请求独立**的，无法利用跨请求的共享模式：

```
场景：多个请求使用相同的 System Prompt

  请求 A: [System Prompt 500 tokens] + [用户问题 A]
  请求 B: [System Prompt 500 tokens] + [用户问题 B]
  请求 C: [System Prompt 500 tokens] + [用户问题 C]

  当前：每个请求独立构建 SAM → 3 × O(500) 构建开销
  优化：共享 System Prompt 的 SAM，每个请求只增量追加各自的部分

更强的场景：对话服务中，很多回答具有相似的句式模板
  "好的，我来帮你..."、"根据你的描述..."、"以下是..."
  
  跨请求的共享后缀树可以利用这些模式，
  即使当前请求 context 中没有出现过，但其他请求出现过 → 可以作为 draft
```

#### 设计方案

**Global Suffix Pool：全局后缀池**

```python
class GlobalSuffixPool:
    """全局后缀池：收集所有请求的 token 序列片段，
    构建全局后缀结构供所有请求查询。
    
    设计要点：
    1. 不存储完整请求序列（隐私 + 内存考虑）
    2. 只存储"被多次验证通过的"片段（高质量 draft 来源）
    3. 容量有限，使用 LRU 驱逐低频片段
    """
    
    def __init__(self, max_segments: int = 10000, segment_max_len: int = 32):
        self.max_segments = max_segments
        self.segment_max_len = segment_max_len
        
        # 存储的片段：List[np.ndarray]
        self._segments: list[np.ndarray] = []
        
        # 全局 SAM（定期重建）
        self._global_sam: Optional[IncrementalSuffixAutomaton] = None
        self._dirty = True  # 是否有新片段加入需要重建
        
        # 片段频率统计
        self._segment_access_count: dict[int, int] = {}  # segment_idx → count
    
    def add_accepted_segment(self, tokens: np.ndarray):
        """添加一个被验证接受的 token 片段。
        
        由 GPUModelRunner 在每步验证后调用：
        如果某个 draft 被接受了 >= 3 个 tokens，将接受的部分加入池中。
        """
        if len(tokens) < 3:
            return
        
        # 截断到 max_len
        segment = tokens[:self.segment_max_len].copy()
        self._segments.append(segment)
        self._dirty = True
        
        # 容量驱逐
        if len(self._segments) > self.max_segments:
            self._evict_least_used()
    
    def query(
        self, 
        pattern: np.ndarray, 
        k: int,
    ) -> Optional[np.ndarray]:
        """在全局后缀池中查找匹配。
        
        作为 per-request SAM 的补充查询源。
        """
        if self._dirty:
            self._rebuild_global_sam()
            self._dirty = False
        
        if self._global_sam is None:
            return None
        
        # 拼接所有 segments 为一个序列（用分隔符分隔）
        return self._global_sam.find_longest_match(
            pattern, self._concat_text, k
        )
    
    def _rebuild_global_sam(self):
        """重建全局 SAM（后台异步执行）。"""
        if not self._segments:
            self._global_sam = None
            return
        
        # 用特殊分隔符连接所有 segments
        SEPARATOR = -1  # 不会出现在正常 token_id 中
        parts = []
        for seg in self._segments:
            parts.append(seg)
            parts.append(np.array([SEPARATOR], dtype=np.int32))
        
        self._concat_text = np.concatenate(parts)
        
        sam = IncrementalSuffixAutomaton()
        for token in self._concat_text:
            sam.extend(int(token))
        self._global_sam = sam
    
    def _evict_least_used(self):
        """驱逐最不常用的片段。"""
        if len(self._segments) <= self.max_segments:
            return
        # 简单策略：驱逐最早加入的（FIFO）
        excess = len(self._segments) - self.max_segments
        self._segments = self._segments[excess:]


class HybridSuffixProposer(AdaptiveSuffixProposer):
    """混合 Proposer：先查本请求 SAM，再查全局池。"""
    
    def __init__(self, global_pool: GlobalSuffixPool):
        super().__init__()
        self._global_pool = global_pool
    
    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
        req_index: int = 0,
    ) -> Optional[np.ndarray]:
        # 先查本请求的 SAM
        draft = super().propose(context_token_ids, n, k, req_index)
        
        if draft is not None and len(draft) >= k // 2:
            return draft  # 本请求匹配质量足够好
        
        # 本请求匹配不够好，查全局池
        pattern = context_token_ids[-n:]
        global_draft = self._global_pool.query(pattern, k)
        
        if global_draft is not None:
            if draft is None or len(global_draft) > len(draft):
                return global_draft
        
        return draft
```

#### 修改文件
- `vllm/v1/spec_decode/suffix_proposer.py` — 新增 `GlobalSuffixPool` 和 `HybridSuffixProposer`
- `vllm/v1/worker/gpu_model_runner.py` — 实例化全局池，验证后添加 accepted segments

#### 预期效果
- 短 context 请求也能利用全局模式产生高质量 draft
- 对话模板场景下的接受率大幅提升
- 冷启动请求（刚进入系统的）也能立即获得 draft

#### 涉及的 vLLM 知识点
- `input_batch` 的批次管理：多个请求在同一个 batch 中如何组织
- `GPUModelRunner` 的生命周期：跨步的状态维护
- Worker 进程与 Engine 进程的关系（全局池应该在 Worker 进程级别）

---

### 优化点 5：后缀解码效果量化与可观测性 `[辅助]` `[方案设计 ⬜]`

#### 问题分析

当前 vLLM V1 的 spec decode 指标非常有限。要验证后缀解码的效果，需要完善的可观测性。

#### 设计方案

```python
@dataclass
class SuffixDecodeMetrics:
    """后缀解码指标。"""
    
    # 基础指标
    total_proposals: int = 0          # 总提案次数
    total_draft_tokens: int = 0       # 总 draft tokens 数
    total_accepted_tokens: int = 0    # 总被接受的 tokens 数
    total_bonus_tokens: int = 0       # 总 bonus tokens 数
    
    # 匹配指标
    match_found_count: int = 0        # 找到匹配的次数
    match_not_found_count: int = 0    # 未找到匹配的次数
    match_lengths_sum: int = 0        # 匹配长度总和
    fallback_count: int = 0           # 回退到更短匹配的次数
    
    # 候选评分指标
    multi_candidate_count: int = 0    # 有多个候选的次数
    best_candidate_from_recent: int = 0  # 最优候选来自最近位置的次数
    
    # 全局池指标
    global_pool_queries: int = 0      # 全局池查询次数
    global_pool_hits: int = 0         # 全局池命中次数
    global_pool_segments: int = 0     # 全局池当前片段数
    
    # SAM 性能指标
    sam_build_time_ms: float = 0.0    # SAM 构建/更新总耗时
    sam_query_time_ms: float = 0.0    # SAM 查询总耗时
    
    @property
    def acceptance_rate(self) -> float:
        """接受率 = accepted / proposed"""
        return self.total_accepted_tokens / max(1, self.total_draft_tokens)
    
    @property
    def avg_accepted_length(self) -> float:
        """平均每次提案的接受长度"""
        return self.total_accepted_tokens / max(1, self.total_proposals)
    
    @property
    def tokens_per_step(self) -> float:
        """每步有效 token 数 = accepted + bonus + 1（normal token）"""
        total_steps = self.match_found_count + self.match_not_found_count
        total_tokens = (self.total_accepted_tokens + 
                       self.total_bonus_tokens + total_steps)
        return total_tokens / max(1, total_steps)
    
    @property
    def match_rate(self) -> float:
        """匹配成功率"""
        total = self.match_found_count + self.match_not_found_count
        return self.match_found_count / max(1, total)
    
    @property
    def avg_match_length(self) -> float:
        """平均匹配长度"""
        return self.match_lengths_sum / max(1, self.match_found_count)
    
    def report(self) -> str:
        """生成可读的指标报告。"""
        return (
            f"=== Suffix Decode Metrics ===\n"
            f"Acceptance Rate: {self.acceptance_rate:.1%}\n"
            f"Tokens/Step:     {self.tokens_per_step:.2f}\n"
            f"Match Rate:      {self.match_rate:.1%}\n"
            f"Avg Match Len:   {self.avg_match_length:.1f}\n"
            f"Avg Accept Len:  {self.avg_accepted_length:.1f}\n"
            f"Fallback Rate:   {self.fallback_count}/{self.total_proposals}\n"
            f"Global Pool Hit: {self.global_pool_hits}/{self.global_pool_queries}\n"
            f"SAM Build Time:  {self.sam_build_time_ms:.1f}ms\n"
            f"SAM Query Time:  {self.sam_query_time_ms:.1f}ms\n"
        )
```

#### 修改文件
- `vllm/v1/spec_decode/suffix_proposer.py` — 在各方法中埋点
- `vllm/v1/worker/gpu_model_runner.py` — 定期输出指标
- 可选：集成到 vLLM 的 Prometheus metrics 中

#### 涉及的 vLLM 知识点
- `SpecDecodeWorkerMetrics`（V0 中的指标类）如何收集和上报
- `GPUModelRunner.execute_model()` 的返回链路
- V1 的日志和监控框架

---

## 四、实现路线图与依赖关系

```
优化点 1 (基础后缀数组)
    │
    ↓
优化点 2 (增量 SAM)  ←── 核心性能保障
    │
    ↓
优化点 3 (自适应匹配 + 评分)  ←── 核心质量提升
    │
    ├──→ 优化点 4 (跨请求共享池)  ←── 进阶
    │
    └──→ 优化点 5 (可观测性)  ←── 辅助所有优化的效果验证
```

**推荐实现顺序**：

| 阶段 | 优化点 | 优先级 | 预计工作量 | 核心收益 | 状态 |
|------|--------|--------|-----------|---------|------|
| 阶段 1 | 优化点 1：基础 SuffixTreeProposer | P0 | 中 | 替换 NgramProposer，验证可行性 | ✅ 已完成 |
| 阶段 2 | 优化点 2：增量 SAM | P0 | 大 | 核心性能优化，O(1) 增量更新 | ✅ 已完成 |
| 阶段 3 | 优化点 5：可观测性 | P0 | 小 | 量化后续优化的效果 | ⬜ 未开始 |
| 阶段 4 | 优化点 3：自适应匹配 + 评分 | P1 | 中 | 提升接受率 | ✅ 已完成 |
| 阶段 5 | 优化点 4：跨请求共享池 | P2 | 大 | 多请求场景下的额外收益 | ⬜ 未开始 |

---

## 五、核心收益总结

| 指标 | N-gram（vLLM V1 当前） | 后缀解码（优化后预期） | 来源 |
|------|----------------------|---------------------|------|
| 查询时间复杂度 | O(context_len) per propose | O(pattern_len) per propose | 优化 1 ✅ / 优化 2 ✅ |
| 增量更新 | 无（每次全量搜索） | O(1) per new token | 优化 2 ✅ |
| 匹配成功率 | ~40-60%（固定 n 值） | ~70-85%（自适应回退） | 优化 3 ✅ |
| 平均接受长度 | ~2-3 tokens | ~3-5 tokens | 优化 3 ✅ |
| 每步有效 token 数 | ~1.5-2.0 | ~2.5-3.5 | 综合 |
| 跨请求利用 | 无 | 有（全局池） | 优化 4 ⬜ |

**整体效果**：在重复性较高的场景（如对话服务、代码补全、模板化回答）中，Decode 阶段的每步有效 token 数从 ~1.5 提升到 ~3.0，等效于 2x 的 Decode 速度提升。

## 六、学习价值

通过这 5 个优化点的实现，你将深入理解：

1. **vLLM V1 投机解码全链路**：Scheduler → ModelRunner → Forward → RejectionSampler → Proposer 的完整数据流
2. **NgramProposer 的 KMP 实现**：Numba JIT 加速、numpy 数组操作
3. **后缀数据结构的工程实践**：后缀数组、LCP 数组、后缀自动机的构建与查询
4. **RejectionSampler 的逐位验证机制**：`cumprod` 的妙用、bonus token 的作用
5. **Scheduler 对 spec tokens 的统一处理**：`num_tokens_with_spec` 消除了 prefill/decode 的区分
6. **`generate_draft_token_ids()` 的关键路径地位**：位于模型推理后的串行瓶颈
7. **V1 vs V0 架构差异**：V1 将 spec decode 紧密集成在 ModelRunner 中，而非独立 Worker
8. **GPU→CPU→GPU 的数据流**：spec token IDs 在 CPU 上生成（Proposer），通过 SchedulerOutput 传回 GPU

---

## 七、与当前项目已有优化的关系

```
已有优化（调度侧 + 限速侧）：
  README 优化 1: QoS 分级调度 ──── 决定"谁先被调度"
  README 优化 4: Token 限速 ─────── 控制"跑多快"
  README 优化 7: MLFQ 多级反馈 ─── 自适应优先级

prefix-cache-scheduling-optimization.md（KV Cache 侧）：
  前缀缓存感知调度 ──────────────── 让缓存命中率影响调度顺序
  批次内前缀去重 ─────────────────── 同批次共享前缀复用
  频率感知驱逐 ───────────────────── 保护高频缓存不被误驱逐

suffix-decoding-optimization.md（推理加速侧）：    ← 本文件
  后缀树 Proposer ────────────────── 更高效的 draft 生成              ✅ 已实现
  增量后缀自动机 ─────────────────── 跨步复用，O(1) 更新             ✅ 已实现
  自适应匹配 + 评分 ──────────────── 提升 draft 接受率               ✅ 已实现
  跨请求共享 ─────────────────────── 全局模式利用                    ⬜ 方案设计
  可观测性指标 ───────────────────── 量化优化效果                    ⬜ 方案设计

三者协同：
  调度器 ─── 决定请求的优先级和运行速率
       ↓
  KV Cache ── 通过前缀复用减少 Prefill 的计算量
       ↓
  后缀解码 ── 通过高质量 draft 减少 Decode 的步数
       ↓
  总效果 ──── TTFT↓(KV Cache) + Decode 速度↑(后缀解码) + QoS 保障(调度)
```
