# PD 分离（Prefill-Decode Disaggregation）优化 — 测试方案

> 目标：验证 [`pd-disaggregation-optimization.md`](../basic_optimization/pd-disaggregation-optimization.md)
> 中的 PD 分离优化（V1 引擎适配 / 智能路由 / 调度器 PD 感知）相对**单实例混合部署（P+D 合一）**的效果。
>
> 核心指标：**TTFT（首 token 时间）**、**TPOT（每 token 时间）**、吞吐、**Prefill/Decode 相互干扰程度**、KV 传输开销。

## 0. 背景与原理（先理解再测）

### 0.1 什么是 PD 分离

标准 vLLM 单实例把 **Prefill（处理 prompt，计算密集）** 和 **Decode（逐 token 生成，访存密集）** 混在同一个引擎里跑。二者特性冲突：

```
单实例混合部署（基线）：
  同一 GPU 上，Prefill 和 Decode 抢同一份算力/显存
     ↓
  长 prompt 的 Prefill 会阻塞正在 Decode 的请求（TPOT 抖动）
  Decode 的小 batch 又让 Prefill 的算力利用率低

PD 分离（优化）：
  Prefill 实例（算力优化）  ──KV cache 传输──▶  Decode 实例（访存/显存优化）
     ↓                                            ↓
  各自按自己的负载特性调优，互不干扰
```

**核心收益**：Prefill 和 Decode 解耦后，**Decode 的 TPOT 不再被 Prefill 抖动干扰**，两者可独立扩缩容与调优。

### 0.2 对比对象

| 方案 | 说明 | 部署形态 |
|------|------|---------|
| **A 基线：单实例混合部署** | P+D 合一（标准 vLLM），Prefill/Decode 抢同一 GPU | 1 个实例 |
| **B 优化：PD 分离** | Prefill 实例 + Decode 实例，KV cache 跨实例传输 | ≥2 个实例（1P + 1D） |

### 0.3 预期效果（来自优化文档）

| 指标 | 预期 | 说明 |
|------|------|------|
| **TPOT（Decode 时延）** | **更稳、更低** | Decode 不再被 Prefill 阻塞，抖动↓ |
| **TTFT** | 视负载 | 分离后 Prefill 专用实例可能更快；但增加了 KV 传输一跳 |
| 吞吐 | 视场景 | 高并发/长 prompt 场景收益明显；轻载可能因传输开销略降 |
| KV 传输开销 | 新增成本 | 分离引入的**额外代价**，必须测量 |

> ⚠️ **核心权衡**：PD 分离的收益来自"消除 P/D 相互干扰"，代价是"**KV cache 跨实例传输的开销 + 多实例资源成本**"。
> 因此**净收益 = 消除干扰的增益 − KV 传输开销**，必须实测验证。

---

## 1. 测试环境与前提（重要：先读约束）

### 1.1 硬件约束与实现现状（必须正视）

PD 分离**本质上需要至少 2 个实例（1 Prefill + 1 Decode）**，这与当前测试环境和实现状态存在两个硬约束，测试方案必须据实处理：

| 约束 | 现状 | 对测试的影响 |
|------|------|-------------|
| **单卡 5070 Ti** | 只有 1 张 GPU | 无法真正做"2 GPU 物理分离"。只能：① 单卡起 2 个实例（显存各分一半，模拟分离但有资源竞争）；② 或在多卡机器上补测 |
| **优化点实现状态** | 优化文档中**优化 1/2/3 标为已实现，优化 4/5/6 为方案设计（未实现）** | 只能测已实现部分；且需先确认 KV connector 在本 fork 中是否真正可用 |

> ⚠️ **前置确认（做测试前必须先验证）**：本项目代码中当前**未检索到** `kv_transfer` / `KVConnector` 相关实现。
> 因此 **Step 0 必须先确认 PD 分离功能是否真正可运行**（见 §4 Step 0），否则本测试无法执行——
> 若功能未落地，本文档作为"待实现功能的测试设计"存档，待功能可用后再执行。

### 1.2 环境

- RTX 5070 Ti（Blackwell，sm_120），单卡（或多卡机器补测）
- Qwen2.5-1.5B-Instruct
- **无 `--enforce-eager`**（用默认 CUDA graph，避免 prefill 慢的坑，见调度测试经验）
- 若单卡起双实例：每实例 `--gpu-memory-utilization 0.4`（各占一半显存）

---

## 2. 核心指标与数据源

### 2.1 PD 分离专属指标（最关键）

PD 分离的价值主要体现在"**消除 Prefill 对 Decode 的干扰**"，所以核心看 TPOT 的稳定性和 KV 传输开销：

| 指标 | 数据源 | 含义 | PD 分离的预期 |
|------|--------|------|--------------|
| **P99 TPOT** | `vllm bench serve` | 每 token 时间尾延迟 | 分离后应**更低更稳**（Decode 不被 Prefill 打断） |
| **TPOT 抖动（P99/P50）** | 由 bench 结果计算 | Decode 时延稳定性 | 分离后**抖动比应下降**——这是 PD 分离最核心的证据 |
| **P99 TTFT** | `vllm bench serve` | 首 token 时间 | 分离后 Prefill 专用，可能↓；但加了 KV 传输一跳 |
| **KV 传输耗时** | 服务端日志 / 埋点 | 每请求 KV 从 P→D 的传输时间 | 分离**新增**的开销，必须量化 |
| **KV 传输量（bytes）** | 埋点 | 传了多少 KV cache | 评估传输带宽压力 |

### 2.2 端到端性能指标

| 指标 | 数据源 | 说明 |
|------|--------|------|
| 吞吐（req/s, tok/s） | `vllm bench serve` | 整体性能 |
| P99 E2EL | `vllm bench serve` | 端到端总延迟 |
| 失败请求数 | `vllm bench serve` | 分离引入的 KV 传输失败/超时（稳定性） |

### 2.3 服务端内部指标（Prometheus + Grafana）

复用调度测试文档已搭好的 Prometheus/Grafana：

| 指标 | 含义 | 看 PD 分离的什么 |
|------|------|-----------------|
| 各实例 Prefill/Decode 耗时 | 分实例的阶段耗时 | 确认 P 实例只做 prefill、D 实例只做 decode |
| 各实例 KV Cache 使用率 | 显存占用 | P/D 实例显存分布是否合理 |
| 请求等待时间 | 排队 | D 实例是否因等 KV 而排队（`WAITING_FOR_REMOTE_KVS`） |
| KV connector 传输指标 | 传输队列/耗时 | 若 connector 已实现并暴露指标 |

### 2.4 指标关系（为什么测这些）

```
消除 P/D 干扰 → Decode 不被 Prefill 阻塞 → TPOT 抖动↓、更稳     ← 主要收益
                                              ↓
KV 跨实例传输 → 每请求多一跳传输 → TTFT/E2EL 增加、带宽压力      ← 主要代价
                                              ↓
                        净效果 = 两者权衡，必须实测（尤其看 TPOT 稳定性 vs 传输开销）
```

---

## 3. 测试设计

### 3.1 测试矩阵

| 维度 | 取值 | 说明 |
|------|------|------|
| 部署形态 | A（单实例 P+D 合一）/ B（PD 分离 1P+1D） | 核心对比 |
| 负载 | 恒定 QPS（10/50/100） | 从轻到重 |
| **prompt 长度** | 短(256) / **长(2K+)** | ⭐关键：PD 分离的收益在**长 prompt**场景（长 Prefill 对 Decode 干扰大）最明显 |
| **P:D 比例**（仅 B） | 1P:1D / 2P:1D / 1P:2D | 验证不同 P/D 配比下的平衡点（受单卡限制，多卡才好扫） |
| 数据集 | sharegpt / random（可控长度） | sharegpt 贴近真实；random 可精确控制 prompt 长度 |

> **关键设计思想**：PD 分离**不是在所有场景都赢**。它的核心价值是"长 prompt 高并发下保护 Decode 的 TPOT"。
> 因此测试必须包含**长 prompt + 有并发 decode 的混合负载**——这正是单实例最痛（Prefill 阻塞 Decode）、分离最能体现价值的场景。
> 轻载 / 短 prompt 场景下，分离的 KV 传输开销可能反而让它**净亏**，这也是要如实测出的边界。

### 3.2 关键对照：Prefill 干扰 Decode 的场景（PD 分离的主战场）

设计一个"长 prompt 请求 + 持续 decode 请求"混合的负载，最能暴露单实例的痛点：

```
混合负载：
  - 背景流：持续的短请求在 decode（模拟在线服务）
  - 冲击流：间歇性的长 prompt 请求（2K+ token）触发大 Prefill

单实例（A）：长 Prefill 抢算力 → 背景 decode 的 TPOT 尖刺（抖动大）
PD 分离（B）：长 Prefill 在 P 实例 → 背景 decode 在 D 实例不受影响（TPOT 平稳）
```

**核心判据**：对比 A/B 两轮**背景 decode 请求的 P99 TPOT 及其抖动**——若 B 的 TPOT 抖动显著低于 A，即证明 PD 分离有效。

### 3.3 完整对比实验清单（进度 + 缺口）

| 轮次 | 部署 | prompt 长度 | 负载 | 目的 | 状态 |
|------|------|------------|------|------|------|
| **0** | — | — | — | **确认 PD 功能可运行**（KV connector 是否可用） | ⬜ 前置必做 |
| **1** | A 单实例 | 短(256) | QPS 10 | 基线·轻载短 prompt | ⬜ 待做 |
| **2** | B 分离 | 短(256) | QPS 10 | 分离·轻载（预期：可能因传输开销净亏，测边界） | ⬜ 待做 |
| **3** | A 单实例 | **长(2K+)混合** | QPS 10 | 基线·长 prompt 干扰 decode（暴露痛点） | ⬜ 待做 |
| **4** | B 分离 | **长(2K+)混合** | QPS 10 | 分离·长 prompt（**核心轮次**，验证 TPOT 抖动↓） | ⬜ 待做 |
| **5** | A / B | 长混合 | **QPS 50/100** | 高并发下的收益对比 | ⬜ 待做 |
| **6** | B 分离 | 长混合 | QPS 50 | **P:D 比例扫描**（1:1 / 2:1 / 1:2，找平衡点，需多卡） | ⬜ 待做 |

> **实验充分性判断**：
> - 轮次 3/4 是**核心**——它们直接对比"长 Prefill 干扰 Decode"场景下 A/B 的 TPOT 抖动，是 PD 分离价值的关键证据。不跑无法证明分离有效。
> - 轮次 2 是**边界验证**——诚实测出"轻载短 prompt 下分离可能净亏"，避免夸大 PD 分离的普适性。
> - 轮次 6 需**多卡**才能有意义（单卡起多实例会资源互抢，扭曲结果）。

> **控制变量提醒**：A/B 两轮的**总显存、总算力应尽量对等**（例如 A 用整卡，B 的 1P+1D 各半卡），否则是"1 卡 vs 2 卡"的不公平对比。理想情况在**2 卡机器**上做：A=1 卡跑 P+D，B=2 卡分别跑 P/D——但这样 B 用了双倍资源，也要在结论里说明。**最公平的对比是同等总资源下比较**。

---

## 4. 测试步骤

### Step 0：确认 PD 分离功能可运行（前置，必做）

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate

# 1. 确认代码里是否有 KV connector 实现
grep -rn "KVConnector\|kv_transfer\|kv_connector" vllm/ | head

# 2. 确认 vllm 是否支持 --kv-transfer-config 参数
python -c "from vllm.config import KVTransferConfig; print('KV transfer supported')" 2>&1 | head

# 3. 查看官方 PD 分离示例（若存在）
ls examples/ | grep -i "disagg\|pd\|kv_transfer"
```

> **判断**：
> - 若上述有输出 → PD 功能可能可用，继续 Step 1。
> - 若全部无输出 → **本 fork 尚未落地 PD 分离**，本文档作为"待实现功能的测试设计"存档，
>   优化文档中"优化 1/2/3 已实现"的标注需复核（可能是设计稿而非落地代码）。

### Step 1：启动 A 基线（单实例 P+D 合一）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

### Step 2：启动 B 优化（PD 分离，示意——以实际 connector 用法为准）

> ⚠️ 以下命令为**示意**，具体参数取决于本 fork 实际的 KV connector 实现方式。
> 若用 vLLM 官方 `--kv-transfer-config`，P/D 两实例分别配 `kv_producer` / `kv_consumer`。

```bash
# Prefill 实例（生产者）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.4 --port 8100 \
    --kv-transfer-config '{"kv_connector":"<实际connector>","kv_role":"kv_producer"}'

# Decode 实例（消费者）
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --gpu-memory-utilization 0.4 --port 8200 \
    --kv-transfer-config '{"kv_connector":"<实际connector>","kv_role":"kv_consumer"}'

# 路由代理（把 prefill 请求发 P、decode 阶段转 D）——对应优化点 2 的智能路由
# 以实际路由脚本为准
```

### Step 3：启动监控（复用调度测试的 Prometheus/Grafana）

```bash
# 参考 SCHEDULING_BENCHMARK_GUIDE.md 的 Prometheus + Grafana 搭建
# 同时监控 P 实例（8100）和 D 实例（8200）的 /metrics
```

### Step 4：跑压测（A/B 各一轮，参数完全一致）

```bash
# 核心轮次：长 prompt 混合负载（对应 §3.3 轮次 3/4）
vllm bench serve \
    --backend openai --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:<A=8000 / B=路由端口> --endpoint /v1/completions \
    --dataset-name random --random-input-len 2048 --random-output-len 256 \
    --num-prompts 500 --request-rate 10 --temperature 0 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir results/pd_<baseline|disagg>/ \
    --result-filename bench.json --metadata deploy=<single|disagg>
```

### Step 5：分析（核心看 TPOT 抖动对比）

```bash
# 对比 A/B 两轮的 P99 TPOT 和 P99/P50 抖动比
# 若 B 的 TPOT 抖动显著低于 A → PD 分离有效
```

---

## 5. 测试结果记录与结论

> 待执行。按下表记录（A=单实例，B=PD 分离）：

### 5.1 核心对比表（长 prompt 混合负载，轮次 3/4）

| 指标 | A 单实例 | B PD 分离 | 差异 | 判定 |
|------|---------|-----------|------|------|
| P99 TPOT (ms) | 待测 | 待测 | | B 更低=有效 |
| **TPOT 抖动 (P99/P50)** | 待测 | 待测 | | **B 更小=分离消除了干扰（核心证据）** |
| P99 TTFT (ms) | 待测 | 待测 | | 视 KV 传输开销 |
| 吞吐 (tok/s) | 待测 | 待测 | | ±5% 内为持平 |
| P99 E2EL (ms) | 待测 | 待测 | | B 可能因传输略增 |
| KV 传输耗时 (ms/req) | N/A | 待测 | | 分离的新增成本 |
| 失败请求数 | 待测 | 待测 | | 分离稳定性 |

### 5.2 结论要点（待填）

- [ ] PD 分离在**长 prompt 混合负载**下是否降低了 Decode 的 TPOT 抖动？（核心结论）
- [ ] KV 传输开销有多大？是否吃掉了收益？
- [ ] 在什么场景 PD 分离**净亏**（轻载/短 prompt）？—— 边界结论
- [ ] 同等总资源下（而非 1 卡 vs 2 卡），分离是否仍有优势？

---

## 6. 待办清单

- [ ] **Step 0：确认 PD 分离功能是否真正可运行**（KV connector 是否落地）——前置阻塞项
- [ ] 复核优化文档"优化 1/2/3 已实现"的标注（代码中未检索到 kv_transfer）
- [ ] 轮次 1/2：轻载短 prompt A/B（边界验证）
- [ ] 轮次 3/4：长 prompt 混合负载 A/B（**核心，验证 TPOT 抖动**）
- [ ] 轮次 5：高并发（QPS 50/100）A/B
- [ ] 轮次 6：P:D 比例扫描（需多卡）
- [ ] 补齐 KV 传输耗时/传输量埋点（对应优化点 3 的传输指标）
- [ ] 结论：PD 分离的有效场景与边界（诚实结论，含"何时净亏"）

---

## 7. 与其他测试文档的关系

| 文档 | 关系 |
|------|------|
| [`SCHEDULING_BENCHMARK_GUIDE.md`](./SCHEDULING_BENCHMARK_GUIDE.md) | 复用其 Prometheus/Grafana 监控搭建；调度优化是**单实例内**的资源编排，PD 分离是**跨实例**的资源编排，互补 |
| [`SUFFIX_DECODING_BENCHMARK_GUIDE.md`](./SUFFIX_DECODING_BENCHMARK_GUIDE.md) | 同为"优化点对比测试"的结构范式（原理→指标→矩阵→步骤→诚实结论） |
| [`pd-disaggregation-optimization.md`](../basic_optimization/pd-disaggregation-optimization.md) | 被测优化的设计文档；注意其优化 4/5/6 为方案设计（未实现），本测试只覆盖已落地部分 |

> **诚实定位提醒**：PD 分离是业界成熟技术（DistServe/Splitwise/Mooncake/vLLM 官方均有），本项目的价值在
> "**V0→V1 移植 + 工程适配**"而非原创。测试的目的是**验证移植实现的正确性与收益边界**，而非证明新技术。
