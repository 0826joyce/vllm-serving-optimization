# 端到端业务场景综合压测

> 一个综合性 Case，模拟 **"企业级 AI 平台"** 的真实运行环境。
> 该平台同时服务多个租户、多种业务类型，且经历流量波动和系统变更。
> **同一套流量、三种部署模式**（单实例 / PD 分离 / 投机解码），一键对比不同优化方向的效果。

## 项目概述

• **端到端业务场景综合压测框架（E2E Business Case Benchmark）**

  — **背景与挑战：** 针对 vLLM V1 推理引擎在企业级多租户场景下的调度与 KV Cache 优化，单一指标（如吞吐/TTFT）无法反映真实负载下的系统瓶颈。已有的 6 项框架优化（QoS 分级调度、MLFQ 多级反馈、Token 限速、Cache-Aware 调度、Segmented LRU 频率感知驱逐、抢占缓存保护）在稳态下有效，但在 Prompt 版本切换、流量突增、长文档暴增、全面过载等真实业务场景下仍存在明显不足——缺少缓存版本管理、租户级资源隔离、Prefill 预算隔离、过载管理和取消感知缓存保留等端到端能力。同时，PD 分离（Prefill-Decode 分离）和投机解码（Suffix Decoding）等高级部署模式也需要在同样的业务场景下进行评估对比。

  — **核心贡献：** 独立设计并实现了一套 5 阶段递进加压、**3 种部署模式可切换**的综合压测框架。模拟企业级 AI 平台的真实运行环境：一个 vLLM 实例（或 PD 分离集群）同时服务 Gold（金融客服 + 代码补全）、Silver × 3（通用客服）、Bronze × 2（文档分析）共 7 个租户，涵盖短对话、高频取消、长文档 RAG 等 4 种业务类型。通过 5 个 Phase（稳态预热 → Prompt 灰度切换 → 流量暴增 4× → 长文档暴增 → 全面过载）在 300 秒内逐步引入新压力。支持单实例（Prefix Cache）、PD 分离（智能路由 + KV 传输）、投机解码（Suffix Decoding）三种模式，以及 `--mode all` 依次跑三种模式并自动生成交叉对比报告。

  — **收益：** 成功暴露了已有 6 项优化在真实多租户场景下的 5 类核心缺陷，驱动形成从"框架优化"到"业务优化"的完整闭环。同时，通过三模式交叉对比，量化不同部署架构在同一业务场景下的表现差异（如 PD 分离在长文档暴增时的 Decode 解耦优势、投机解码在代码补全场景的加速效果等）。

## 部署模式

本压测支持三种部署模式（`--mode` 参数），同一套流量在不同后端配置下跑出对比数据：

| 模式 | `--mode` 值 | 后端配置 | 核心评估点 |
|------|-------------|---------|-----------|
| **Mode A: 单实例** | `single`（默认） | 1 个 vLLM 实例 + Prefix Cache | 调度优化 + KV Cache 策略 |
| **Mode B: PD 分离** | `pd-disagg` | Router + Prefill 实例 + Decode 实例 | 智能路由 + KV 传输 + Prefill/Decode 解耦 |
| **Mode C: 投机解码** | `spec-decode` | 1 个 vLLM 实例 + `--speculative-config` | Suffix Decoding 加速 + draft 接受率 |
| **Mode D: 全部** | `all` | 依次跑 A→B→C | 自动生成三模式交叉对比报告 |

## 业务背景

你是一家 AI 平台的推理引擎负责人。平台基于 vLLM 部署，同时服务以下业务：

| 租户 | 业务类型 | SLA | 流量特征 |
|------|---------|-----|---------|
| **Gold-A（金融客服）** | 短对话 + System Prompt 会定期更新 | P99 TTFT < 200ms | 稳态 QPS=8，可突发到 32 |
| **Gold-B（代码补全）** | 高频补全 + 70% 取消率 | TTFT < 200ms | 稳态 QPS=10，连续击键触发 |
| **Silver × 3（通用客服）** | 短对话 | P99 TTFT < 500ms | 各 QPS=8，稳定 |
| **Bronze × 2（文档分析）** | 长文档 RAG | P99 TTFT < 3000ms | QPS=3，可突发到 10 |

整个测试运行 **5 分钟（300 秒）**，分 5 个阶段，每个阶段引入一种新压力：

```
时间线：
0s─────60s─────120s─────180s─────240s─────300s
│ Phase 1 │ Phase 2 │ Phase 3  │ Phase 4  │ Phase 5  │
│ 稳态预热 │ Prompt  │ Gold-A   │ 长文档   │ 全面     │
│ 建立基线 │ 版本切换 │ 流量暴增  │ 暴增     │ 过载     │
```

### Phase 1（0-60s）：稳态预热
- 所有租户正常 QPS
- **目标**：建立性能基线，观察你的优化在稳态下是否有效

### Phase 2（60-120s）：System Prompt 灰度切换
- Gold-A 的 System Prompt 从 v1 → v2（灰度 10% → 100%）
- **暴露**：Segmented LRU protected zone 被旧 prompt 的死缓存占据，TTFT 突增
- 对应原 Case 1 痛点

### Phase 3（120-180s）：Gold-A 流量暴增 4×
- Gold-A QPS 从 8 飙升到 32
- Silver 租户被挤压
- **暴露**：MLFQ/QoS 无租户隔离，大租户拖垮小租户 SLA
- 对应原 Case 3 痛点

### Phase 4（180-240s）：Bronze 长文档暴增
- Bronze 租户长文档 QPS 从 3 飙升到 10
- 长文档 prefill 阻塞短对话 TTFT
- **暴露**：Cache-Aware 在全 miss 时无效，MLFQ 不区分 prefill 长度
- 对应原 Case 4 痛点

### Phase 5（240-300s）：全面过载
- 在 Phase 3 + Phase 4 的基础上，所有租户 QPS 再增 50%
- 系统过载，观察 SLA 违约率
- **暴露**：无过载保护、无 Deadline-aware 调度
- 对应原 Case 5 痛点

### 全程存在：代码补全高频取消
- Gold-B 全程以连续击键模式发送请求，每个击键取消上一个请求
- **暴露**：取消后 KV Cache 被丢弃，prefix 无法复用
- 对应原 Case 2 痛点

### 各 Phase 与三种部署模式的关系

| Phase | 单实例 (single) 关注点 | PD 分离 (pd-disagg) 关注点 | 投机解码 (spec-decode) 关注点 |
|-------|----------------------|--------------------------|----------------------------|
| Phase 1 | 稳态基线 | Router 路由均衡性 | draft 接受率基线 |
| Phase 2 | 缓存版本切换 | Prefill 缓存与 Prompt 切换交互 | 后缀匹配随 prompt 变化的适应性 |
| Phase 3 | 租户隔离 | Router 过载分流 + KV 传输带宽瓶颈 | 高并发下 draft 模型资源竞争 |
| Phase 4 | Prefill 预算隔离 | **核心优势**：长 Prefill 不阻塞 Decode ITL | 长文档场景 draft 退化分析 |
| Phase 5 | 过载管理 | 多实例分散过载 + Router 级准入 | 过载时投机解码的开销/收益比 |
| 全程 Gold-B | 取消缓存保留 | PD 分离下取消的 KV 回收 | **核心优势**：后缀匹配加速代码补全 |

## 你当前优化的表现预测（各阶段）

| Phase | 你的优化 | 预期表现 |
|-------|---------|---------|
| Phase 1 | Cache-Aware + MLFQ + Segmented LRU | ✅ 稳态下基本有效 |
| Phase 2 | Segmented LRU | ❌ Protected zone 被死缓存占据，新 prompt 热不起来 |
| Phase 3 | MLFQ + QoS Priority | ❌ 请求级调度，无租户隔离，Silver P99 爆炸 |
| Phase 4 | Cache-Aware + MLFQ | ❌ 全 cache miss + 不区分 prefill 长度，短对话被阻塞 |
| Phase 5 | 所有优化 | ❌ 无过载保护，全部请求 SLA 违约 |
| 全程 | Preemption Cache Shield | ❌ 取消走 _free_request 不走抢占，prefix 不保留 |

## 你需要做的端到端改进

### 1. 缓存版本管理（针对 Phase 2）
- 检测 prompt 版本切换，主动降级旧版本 protected zone blocks
- 动态调整 `protected_ratio`：cache hit 率骤降时临时降到 0.1
- 灰度预热：首个 v2 请求到达时主动缓存 v2 prefix

### 2. 取消感知的缓存保留（针对 Gold-B 全程）
- `FINISHED_ABORTED` 时不调用 `free_block_hashes()`
- 保留 prefix blocks 的 hash，让后续相似请求命中
- 定期清理超过 TTL 的残留 hash

### 3. 租户级资源隔离（针对 Phase 3）
- 为 Gold/Silver/Bronze 分配调度权重（WFQ）
- 租户级并发上限（Gold max=20, Silver max=10, Bronze max=5）
- 租户级 KV Cache 预算（保证下限 + 弹性借用）

### 4. Prefill 预算隔离（针对 Phase 4）
- 短请求预留 30% prefill budget
- 限制同时进行 prefill 的长文档请求数
- Prefill 长度感知的调度优先级（Shortest-Prefill-First）

### 5. 过载管理（针对 Phase 5）
- Admission Control：基于队列深度 + SLA 违约率的主动拒绝
- Deadline-Aware：按 slack（剩余 SLA 时间）排序，EDF 调度
- 自适应 token budget：高负载时减少 prefill budget，优先完成 decode
- SLA-Aware 抢占：已违约请求优先被抢占，释放资源给还有救的

## 验收标准

### 单实例 (single) 模式

| 指标 | 优化前 | 优化后（目标） |
|------|--------|--------------|
| Phase 1: Gold-A P99 TTFT | ~100ms | < 200ms ✅ |
| Phase 2: Prompt 切换后恢复时间 | 30-60s | < 5s |
| Phase 2: 切换期间 P99 TTFT | > 1000ms | < 500ms |
| Phase 3: Silver P99 TTFT | > 2000ms | < 500ms |
| Phase 3: Gold-A 暴增时自身 P99 | ~300ms | < 400ms（受配额限制） |
| Phase 4: 短对话 P99 TTFT（长文档暴增时）| > 500ms | < 200ms |
| Phase 5: 被接受请求的 P99 TTFT | > 10s（全违约） | < 800ms |
| Phase 5: 合理拒绝率 | 0%（全收全违约）| 30-40% |
| 全程: Gold-B 连续输入缓存命中率 | ~0% | > 80% |
| 全程: Gold-B 完成请求 TTFT | ~200ms | < 50ms |

### PD 分离 (pd-disagg) 模式 — 额外指标

| 指标 | 目标 |
|------|------|
| Router 路由准确率（PD 分离 vs 直发 Decode） | 合理分布（>70% 走 PD 分离） |
| KV 传输延迟 P95 | < 50ms |
| Phase 4: PD 模式短对话 P99 TTFT | < 150ms（优于单实例：长 Prefill 不阻塞 Decode） |
| Phase 5: PD 模式 Router 级拒绝率 | 20-30%（比单实例更平滑的过载管理） |

### 投机解码 (spec-decode) 模式 — 额外指标

| 指标 | 目标 |
|------|------|
| Draft 接受率 | > 60%（Code 补全场景）/ > 40%（通用对话） |
| 全程 Gold-B E2E 延迟降低 | > 30%（相比单实例） |
| Phase 3 高并发下 draft 接受率退化 | < 15%（可接受的退化范围） |

### 三模式交叉对比 (all) 模式

| 对比维度 | 期望结论 |
|---------|---------|
| Phase 4 P99 TTFT: PD 模式 vs 单实例 | PD 模式显著更优（Prefill/Decode 解耦） |
| Gold-B E2E: SpecDec 模式 vs 单实例 | SpecDec 模式显著更快（后缀匹配加速） |
| Phase 5 整体 SLA 违约率: 三模式排序 | PD < single ≤ SpecDec（PD 可分散过载） |

## 核心权衡总结

| 权衡维度 | 两端 | 你需要找的平衡点 |
|---------|------|----------------|
| 缓存保留 vs 内存泄漏 | 不清 hash → 复用率高；清 hash → 内存干净 | TTL + LRU 清理 |
| 租户隔离 vs 资源效率 | 硬隔离 → 浪费；无隔离 → 互相干扰 | 保证下限 + 弹性借用 |
| 短请求优先 vs 长文档饿死 | SPF → 长文档永远不执行 | Aging + 最大等待时间 |
| 拒绝请求 vs 全部接受 | 拒绝 → 部分达标；全收 → 全部违约 | 自适应 admission control |
| 主动失效 vs 被动淘汰 | 主动 → 快但误判；被动 → 慢但安全 | 命中率监控 + 触发条件 |

## 使用方式

### Mode A: 单实例 + Prefix Cache（默认）

```bash
# 1. 启动 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> \
    --max-model-len 8192 \
    --max-num-batched-tokens 4096 \
    --enable-chunked-prefill \
    --enable-prefix-caching

# 2. 运行综合压测（单实例模式为默认值）
python benchmarks/e2e_cases/workload.py \
    --model <模型名> \
    --host 127.0.0.1 --port 8000 \
    --duration 300
```

### Mode B: PD 分离 + 智能路由

```bash
# 1. 启动 Prefill 实例
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> --port 8100 \
    --enable-chunked-prefill --enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"simple","kv_role":"kv_producer"}'

# 2. 启动 Decode 实例
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> --port 8200 \
    --kv-transfer-config '{"kv_connector":"simple","kv_role":"kv_consumer"}'

# 3. 启动 PD Router（假设 Router 监听 8000 端口）
python -m vllm.v1.engine.pd_router --port 8000 \
    --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200

# 4. 运行综合压测
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --mode pd-disagg \
    --host 127.0.0.1 --port 8000 \
    --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200
```

### Mode C: 单实例 + Suffix Decoding（投机解码）

```bash
# 1. 启动带投机解码的 vLLM 服务
python -m vllm.entrypoints.openai.api_server \
    --model <模型路径> \
    --max-model-len 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --speculative-config '{"method":"suffix","num_speculative_tokens":5}'

# 2. 运行综合压测
python benchmarks/e2e_cases/workload.py \
    --model <模型名> --mode spec-decode \
    --host 127.0.0.1 --port 8000
```

### Mode D: 三模式全自动对比

```bash
# 需要提前启动好所有三种后端服务
# Router@8000, Prefill@8100, Decode@8200, 单实例@8000, SpecDec@8000
# （或依次切换服务后手动运行）

python benchmarks/e2e_cases/workload.py \
    --model <模型名> --mode all \
    --host 127.0.0.1 --port 8000 \
    --prefill-host 127.0.0.1:8100 --decode-host 127.0.0.1:8200

# 运行结束后自动输出：
#   - 各 Phase 各租户的 P50/P95/P99 TTFT
#   - SLA 违约率（按租户 × 阶段交叉统计）
#   - Gold-B 代码补全的缓存命中率和取消率
#   - PD 分离专项：路由分布、KV 传输延迟
#   - Spec Decode 专项：draft 接受率、生成加速比
#   - 三模式交叉对比表（Phase × Mode P99 TTFT）
#   - 结果 JSON 文件用于后续对比分析
#     - results_single.json
#     - results_pd-disagg.json
#     - results_spec-decode.json
#     - results_combined.json（合并数据）
```

## 建议的优化实践顺序

1. **先跑一次基线**：不做任何改动，分别用 `--mode single`、`--mode pd-disagg`、`--mode spec-decode` 各运行一次（或 `--mode all` 一键全跑），收集各 Phase 的 SLA 违约数据
2. **Phase 2 → 缓存版本管理**：改动集中在 `kv_cache_manager.py`，独立验证
3. **全程 → 取消感知缓存**：改动集中在 `scheduler.py` 的 `_free_request()`，独立验证
4. **Phase 4 → Prefill 预算隔离**：改动集中在 `schedule()` 的 budget 分配逻辑
5. **Phase 3 → 租户级隔离**：影响面最广，需要 scheduler + kv_cache_manager + request 联动
6. **Phase 5 → 过载管理**：需要 admission control + deadline-aware + 自适应 budget 协同

每做完一步：
- 用 `--mode single` 重新运行 workload，对比该 Phase 的指标变化
- **同时观察其他 Phase 是否有退化**——这就是"平衡取舍"的核心体验
- 做完所有修复后，用 `--mode all` 跑一次完整的三模式交叉对比，量化不同部署架构下优化的收益差异
