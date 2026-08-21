# 调度优化 — 对比测试方案

> 目标：验证两类**调度相关优化**在真实混合负载下的效果：
> 1. [`scheduling-resource-management-optimization.md`](../basic_optimization/scheduling-resource-management-optimization.md) 中已实现的调度优化（QoS 分级调度 / Token 限速 / MLFQ 多级反馈队列 / 过载管理 / Deadline-aware 调度）
> 2. [`prefix-cache-scheduling-optimization.md`](../basic_optimization/prefix-cache-scheduling-optimization.md) 中已实现的 Prefix Cache 调度优化（KV Cache 命中 / 缓存感知调度 / 长上下文复用）
>
> 这两类优化本质都是**调度相关优化**，开关会同时打开，作为**一类优化**一起测试，流程相同。

## Quick Start：服务端指标采集（Prometheus + Grafana）

> 本节是**一次性环境准备**，完成后后续每次压测都能直接在 Grafana 看到服务端内部状态（队列长度、KV Cache、抢占次数等）。全程约 10 分钟。

### 前提

- Ubuntu 本地已安装 Prometheus 和 Grafana（`apt install prometheus grafana`）
- vLLM 服务已启动（`/metrics` 端点默认开启，无需额外参数）
- 当前用户已配置免密 `sudo`（否则部分命令需手动输入密码）

### Step 1：验证 vLLM `/metrics` 端点可访问

vLLM V1 引擎**默认就开启了 `/metrics`**，不需要 `--enable-metrics` 参数。服务启动后直接验证：

```bash
# 检查服务健康
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8000/health
# 预期输出: HTTP 200

# 检查 /metrics 有数据
curl -s http://127.0.0.1:8000/metrics | head -5
# 预期看到 vllm:num_requests_running 等 gauge 指标
```

> **注意**：不要加 `--disable-log-stats` 参数，否则 `/metrics` 里的很多指标会是空的。该参数默认 `False`（即 stats 默认开启），只要不显式关闭就行。

### Step 2：配置 Prometheus 抓取 vLLM

备份原配置并写入新配置：

```bash
# 备份
sudo cp /etc/prometheus/prometheus.yml /etc/prometheus/prometheus.yml.bak

# 写入新配置（保留原有 prometheus + node，新增 vllm）
sudo tee /etc/prometheus/prometheus.yml > /dev/null << 'EOF'
global:
  scrape_interval: 5s          # 5 秒抓一次（默认 15s 对 300s 压测太粗）
  evaluation_interval: 5s
  external_labels:
    monitor: 'vllm-benchmark'

scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    scrape_timeout: 5s
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']

  - job_name: 'vllm'
    metrics_path: /metrics
    static_configs:
      - targets: ['localhost:8000']
        labels:
          instance: 'vllm-single'
EOF

# 验证配置语法
promtool check config /etc/prometheus/prometheus.yml
# 预期输出: SUCCESS: ... is valid prometheus config file syntax
```

### Step 3：重启 Prometheus 并验证

```bash
# 重启
sudo systemctl restart prometheus
sleep 3

# 确认服务状态
sudo systemctl is-active prometheus
# 预期输出: active

# 验证 vLLM target 已被抓取
curl -s http://127.0.0.1:9090/api/v1/targets | python3 -c "
import json, sys
data = json.load(sys.stdin)
for t in data['data']['activeTargets']:
    print(f\"job={t['labels'].get('job',''):15s} health={t['health']:6s} url={t['scrapeUrl']}\")"
# 预期看到: job=vllm  health=up    url=http://localhost:8000/metrics

# 验证能查到 vLLM 指标
curl -s 'http://127.0.0.1:9090/api/v1/query?query=vllm:num_requests_running' | python3 -m json.tool
# 预期 result 里有值（当前没请求时为 "0"）
```

> **如果 vllm target 显示 down**：检查 vLLM 服务是否在跑（`curl http://127.0.0.1:8000/health`）、防火墙是否放行 8000 端口、Prometheus 配置里的 targets 地址是否正确。

### Step 4：配置 Grafana 数据源

Grafana 默认监听 3000 端口，账号 `admin` / `admin`。通过 API 自动添加 Prometheus 数据源（免去手动点击）：

```bash
# 确认 Grafana 在跑
curl -s -o /dev/null -w "Grafana HTTP %{http_code}\n" http://127.0.0.1:3000/api/health
# 预期输出: Grafana HTTP 200

# 通过 API 添加 Prometheus 数据源
curl -s -X POST http://admin:admin@127.0.0.1:3000/api/datasources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "vLLM-Prometheus",
    "type": "prometheus",
    "access": "proxy",
    "url": "http://127.0.0.1:9090",
    "isDefault": true,
    "editable": true
  }' | python3 -m json.tool
# 预期输出包含 "message": "Datasource added"

# 验证数据源已添加
curl -s http://admin:admin@127.0.0.1:3000/api/datasources | python3 -c "
import json, sys
for ds in json.load(sys.stdin):
    print(f\"name={ds['name']:25s} type={ds['type']:15s} url={ds['url']}\")"
# 预期看到: name=vLLM-Prometheus  type=prometheus  url=http://127.0.0.1:9090
```

> **如果已添加过会报 "data source with same name exists"**，忽略即可，或先删除：`curl -X DELETE http://admin:admin@127.0.0.1:3000/api/datasources/name/vLLM-Prometheus`

### Step 5：创建 vLLM 监控 Dashboard

通过 API 一键创建包含 8 个核心面板的 Dashboard：

```bash
cat > /tmp/vllm-dashboard.json << 'DASHBOARD'
{
  "dashboard": {
    "title": "vLLM 调度监控 — 对比测试",
    "tags": ["vllm", "scheduling"],
    "timezone": "browser",
    "refresh": "5s",
    "time": {"from": "now-30m", "to": "now"},
    "panels": [
      {
        "title": "1. 请求队列长度（Running vs Waiting）",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "targets": [
          {"expr": "vllm:num_requests_running", "legendFormat": "running"},
          {"expr": "vllm:num_requests_waiting", "legendFormat": "waiting"}
        ]
      },
      {
        "title": "2. KV Cache 使用率",
        "type": "gauge",
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 0},
        "targets": [{"expr": "vllm:kv_cache_usage_perc", "legendFormat": "usage"}],
        "fieldConfig": {"defaults": {"min": 0, "max": 1, "unit": "percentunit"}}
      },
      {
        "title": "3. 累计抢占次数",
        "type": "stat",
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
        "targets": [{"expr": "vllm:num_preemptions", "legendFormat": "preemptions"}],
        "fieldConfig": {"defaults": {"unit": "short"}}
      },
      {
        "title": "4. TTFT P50/P95/P99（服务端视角，秒）",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {"expr": "histogram_quantile(0.50, sum(rate(vllm:time_to_first_token_seconds_bucket[30s])) by (le))", "legendFormat": "P50"},
          {"expr": "histogram_quantile(0.95, sum(rate(vllm:time_to_first_token_seconds_bucket[30s])) by (le))", "legendFormat": "P95"},
          {"expr": "histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[30s])) by (le))", "legendFormat": "P99"}
        ]
      },
      {
        "title": "5. 请求排队时间 P99（客户端测不到）",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {"expr": "histogram_quantile(0.99, sum(rate(vllm:request_queue_time_seconds_bucket[30s])) by (le))", "legendFormat": "queue P99"},
          {"expr": "histogram_quantile(0.50, sum(rate(vllm:request_queue_time_seconds_bucket[30s])) by (le))", "legendFormat": "queue P50"}
        ]
      },
      {
        "title": "6. Prefill vs Decode 耗时 P95",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
        "targets": [
          {"expr": "histogram_quantile(0.95, sum(rate(vllm:request_prefill_time_seconds_bucket[30s])) by (le))", "legendFormat": "Prefill P95"},
          {"expr": "histogram_quantile(0.95, sum(rate(vllm:request_decode_time_seconds_bucket[30s])) by (le))", "legendFormat": "Decode P95"}
        ]
      },
      {
        "title": "7. ITL（Inter-Token Latency）P95/P99",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
        "targets": [
          {"expr": "histogram_quantile(0.95, sum(rate(vllm:inter_token_latency_seconds_bucket[30s])) by (le))", "legendFormat": "ITL P95"},
          {"expr": "histogram_quantile(0.99, sum(rate(vllm:inter_token_latency_seconds_bucket[30s])) by (le))", "legendFormat": "ITL P99"}
        ]
      },
      {
        "title": "8. Token 吞吐量（req/s & tok/s）",
        "type": "graph",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24},
        "targets": [
          {"expr": "sum(rate(vllm:request_success[30s]))", "legendFormat": "成功请求 req/s"},
          {"expr": "sum(rate(vllm:prompt_tokens[30s]))", "legendFormat": "输入 tok/s"},
          {"expr": "sum(rate(vllm:generation_tokens[30s]))", "legendFormat": "输出 tok/s"}
        ]
      }
    ]
  },
  "overwrite": true
}
DASHBOARD

curl -s -X POST http://admin:admin@127.0.0.1:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @/tmp/vllm-dashboard.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
if d.get('id'):
    print(f\"Dashboard 创建成功！\")
    print(f\"访问地址: http://127.0.0.1:3000{d['url']}\")
else:
    print(f\"创建失败: {d}\")"
```

创建成功后，**浏览器打开输出的地址**（类似 `http://127.0.0.1:3000/d/xxxx/vllm`），即可看到 8 个面板。

### Step 6：端到端验证

发几个测试请求，确认 Grafana 能看到数据变化：

```bash
# 发 3 个测试请求
for i in 1 2 3; do
    curl -s -X POST http://127.0.0.1:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","prompt":"Hello, my name is","max_tokens":20,"temperature":0}' \
      > /dev/null
    echo "请求 $i 完成"
done

# 等 Prometheus 抓取（5s 间隔）
sleep 6

# 验证 TTFT 采样数有变化
curl -s 'http://127.0.0.1:9090/api/v1/query?query=vllm:time_to_first_token_seconds_count' | python3 -c "
import json, sys
r = json.load(sys.stdin)['data']['result']
print(f'TTFT 采样数: {r[0][\"value\"][1] if r else \"无数据\"}')"
# 数字应该比之前增加 3 左右
```

然后回到 Grafana 浏览器页面，刷新 Dashboard，应该能看到面板 4（TTFT）和面板 8（吞吐）有数据曲线。

### Step 7：日常使用（每次压测时）

环境准备好后，每次压测只需：

1. **启动 vLLM 服务**（A 轮基线或 B 轮优化版）
2. **打开 Grafana Dashboard**，时间范围设为 `now-30m` 或 `now-1h`
3. **在另一个终端跑 `workload.py` 压测**
4. **压测结束后**，在 Grafana 里把时间范围锁定到压测时段（比如 `2026-08-12 23:00:00 to 2026-08-12 23:05:00`）
5. **截图保存**到 `results/<轮次>/` 下，用于 A/B 对比

> **Prometheus 和 Grafana 不需要重启**——它们会自动抓取当前运行的 vLLM 实例。切换 A/B 轮次时只需重启 vLLM 服务即可。

### 面板说明速查

| 面板 | 看什么 | A/B 对比要点 |
|------|--------|-------------|
| 1. 队列长度 | `waiting` 堆积 = TTFT 飙升的直接原因 | Phase 3/4 优化版 waiting 应更短 |
| 2. KV Cache | 打满 1.0 会触发抢占 | Phase 4 优化版应更平滑 |
| 3. 抢占次数 | 过载管理是否生效的硬证据 | Phase 5 优化版应更可控 |
| 4. TTFT | 服务端视角的 TTFT 分布 | 跟客户端 `workload.py` 对比，差值 = 网络开销 |
| 5. 排队时间 | **客户端测不到**，直接定位调度瓶颈 | Phase 4 优化版应明显更低 |
| 6. Prefill/Decode | 区分瓶颈在哪个阶段 | Phase 4 优化版 Prefill 应被 Token 限速削平 |
| 7. ITL | 生成流畅度 | 两轮应接近，优化不应影响 Decode |
| 8. 吞吐量 | 整体性能 | 优化版不应明显低于基线（±5% 内） |

### 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| Grafana 面板全是空白 | Prometheus 没抓到数据 | 检查 `curl http://127.0.0.1:9090/api/v1/targets` 里 vllm job 是否 up |
| vllm target 显示 down | vLLM 服务没启动或端口不对 | `curl http://127.0.0.1:8000/health` 检查服务；检查 `prometheus.yml` 里 targets 地址 |
| `/metrics` 返回但值都是 0 | 误加了 `--disable-log-stats` | 重启 vLLM 服务，去掉 `--disable-log-stats` 参数 |
| Grafana 报 "datasource not found" | 数据源未添加或名称不匹配 | 重新执行 Step 4，或检查数据源名称是否为 `vLLM-Prometheus` |
| Histogram 面板显示为空 | 没有请求被处理过 | 先发几个请求（Step 6），Histogram 需要有数据点才能算百分位 |
| Prometheus 配置改了不生效 | 没重启 Prometheus | `sudo systemctl restart prometheus` |

---

## 0. 性能评估工具选型说明

本测试涉及两类性能评估，需要不同的工具配合使用。下表先厘清主流工具的定位，再说明本项目为什么这样选。

### 0.1 主流工具对比

| 工具 | 维护方 | 测什么 | 流量模式 | 适合场景 | 本项目是否使用 |
|------|--------|--------|----------|----------|----------------|
| **`vllm bench serve`** | vLLM 官方（`vllm/benchmarks/serve.py`） | vLLM serving 的吞吐 + 延迟 | 单一流量源：ShareGPT / 随机 / 自定义数据集，恒定 QPS 或恒定并发 | 回归测试、调参对比、版本间性能对比 | ✅ 用于绝对性能基线（见 §5 扩展 0） |
| **Prometheus + Grafana** | 开源社区 | vLLM 服务端内部状态（队列长度、KV Cache、抢占次数等） | 无需发请求，被动采集 vLLM `/metrics` 端点 | 服务端瓶颈定位、调度器行为可视化 | ✅ 用于服务端指标采集（见 §5 扩展 -1） |
| **GuideLLM** | vllm-project 官方 | 生产级 LLM 部署的性能/效率/可靠性 | 支持 ramp、多种 rate 模式，更工程化 | 选型评估、容量规划、SLA 评估 | ❌ 本项目不需要 |
| **EvalScope (perf)** | 阿里 ModelScope | 通用 LLM 压测，跨框架 | 类似 vllm bench，但框架无关 | 多框架对比（vLLM / sglang / ollama） | ❌ 本项目不需要 |
| **llm-perf** | NVIDIA | 极致性能上限 | 固定输入/输出长度，扫并发 | NVIDIA GPU 性能上限基准 | ❌ 本项目不需要 |
| **`workload.py`**（本项目） | 自研 | 调度策略对多租户混合负载的公平性 | 7 租户 × 5 阶段 × 3 种请求类型，有 QPS 暴增、Prompt 灰度、击键取消 | 调度优化对比（QoS / MLFQ / Token 限速 / 过载管理） | ✅ 核心压测工具（见 §3） |

> **关键事实**：EvalScope 官方对比文档确认，在相同请求参数与并发配置下，`evalscope perf` 能与 `vllm bench serve` 达到一致的负载和指标。因此 `vllm bench serve` / GuideLLM / EvalScope 三者在"绝对性能测试"上结论一致，选哪个都行，区别主要在易用性和生态。本项目选 `vllm bench serve` 是因为它是 vLLM 原生工具，无需额外安装。

### 0.2 为什么需要三类工具

| 层 | 工具 | 回答的问题 | 指标来源 |
|----|------|-----------|---------|
| **调度公平性** | `workload.py` | "在混合负载下，Gold vs Bronze 的 SLA 保障差异有多大？" | 客户端秒表 |
| **绝对性能** | `vllm bench serve` | "这个 vLLM 实例在恒定负载下 P99 TTFT / 吞吐是多少？" | 客户端秒表 |
| **服务端内部状态** | Prometheus + Grafana | "调度器队列长度 / KV Cache 使用率 / 抢占次数如何变化？" | vLLM 服务端导出 |

三类工具**目标不同，不能互相替代**：

- `vllm bench serve` 是单一流量源，**无法模拟 7 租户同时争抢**，测不出调度策略对 Gold/Bronze 的差异化影响
- `workload.py` 是混合多源流量，但流量模式不标准，**无法回答"优化版有没有牺牲整体吞吐"** 这个 reviewer 必问的问题
- 两者都是**客户端视角**，**看不到调度器内部状态**（队列长度、KV Cache 使用率、抢占次数），而服务端指标（Prometheus + Grafana）正好补这个盲区

因此本项目采用**三层证据**：`workload.py` 测公平性 + `vllm bench serve` 测绝对性能 + Prometheus+Grafana 看服务端内部状态，三层结合才能完整说清楚"优化版在保持绝对性能不退化的前提下，改善了多租户公平性，且服务端内部状态符合预期"。

### 0.3 TTFT / TPOT / ITL 指标的统计原理（两类工具一致）

很多人误以为 TTFT 这些指标是 GPU 端给的，其实**完全是客户端用秒表 + SSE 流回调算出来的**，跟 GPU 内部状态无关。`vllm bench serve` 和 `workload.py` 的统计思路**本质相同**，都是：

```
客户端发请求 → 记录 send_time → 读 SSE 流 → 收到第一个有内容的 delta 记 first_token_time → 流结束记 complete_time
```

#### 两者的具体实现对比

| 维度 | `vllm bench serve`（`endpoint_request_func.py`） | `workload.py` |
|------|--------------------------------------------------|---------------|
| **计时函数** | `time.perf_counter()` | `time.monotonic()` |
| **TTFT 锚点** | 第一个含 `choices` 的 SSE chunk 到达时 | 第一个含非空 `content` / `text` 的 delta 到达时 |
| **TTFT 计算** | `ttft = time.perf_counter() - st`（秒） | `ttft_ms = (first_token_time - send_time) * 1000`（毫秒） |
| **E2E 计算** | `latency = most_recent_timestamp - st` | `e2e_ms = (complete_time - send_time) * 1000` |
| **output_tokens** | 从服务端 `usage.completion_tokens` 字段读取（更准） | 客户端自己数 delta 个数（可能多算 special token） |
| **ITL（token 间延迟）** | ✅ 记录每个 token 的时间戳，`itl.append(timestamp - most_recent_timestamp)` | ❌ 不记录，只有 TTFT 和 E2E |
| **TPOT** | ✅ 服务端算：`(latency - ttft) / (output_len - 1)` | ❌ 不算 |
| **SSE 解析** | `StreamedResponseHandler` 按双换行分割 + JSON 校验 | 逐行 `startswith("data: ")` 分割 |
| **统计维度** | 全局聚合：P50/P95/P99 + mean/std/median | 按 tier × phase × type 切分聚合 |
| **失败处理** | `output.success = False` + 记 error | `record.error = str(e)` + 记 `was_cancelled` |

#### 关键差异解读

1. **`vllm bench serve` 多了 ITL 和 TPOT**：这是两个重要的 Decode 阶段指标。ITL（Inter-Token Latency）反映生成的流畅度，TPOT（Time Per Output Token）反映平均单 token 耗时。`workload.py` 没有这两个指标，因为它只关心"调度公平性"而非"生成质量"。

2. **`output_tokens` 来源不同**：`vllm bench serve` 从服务端 `usage.completion_tokens` 读取（准确，含 special token 处理），`workload.py` 客户端自己数 delta（可能漏算空 content 的 special token）。对比绝对性能时以 `vllm bench serve` 为准。

3. **`time.perf_counter()` vs `time.monotonic()`**：两者都不受系统时钟跳变影响，精度都是纳秒级，实际使用中差异可忽略。`perf_counter` 是 Python 官方推荐用于性能测量的函数。

4. **统计维度不同**：`vllm bench serve` 只输出全局 P50/P95/P99，不切分租户/阶段；`workload.py` 按 tier × phase × type 切分，这是为了暴露调度策略对不同请求的差异化影响——`vllm bench serve` 的全局聚合会**掩盖**这种差异。

#### 完整的 TTFT 包含什么

无论哪个工具，客户端测到的 TTFT 都包含：

```
TTFT = 网络上行 + 排队等待 + Prefill 计算 + 首个 token 的 Decode 步 + 网络下行
```

**不包含**服务端内部指标如 KV cache 命中率、调度等待时间、Prefill/Decode 分别的耗时。要看那些得拉 vLLM 的 `/metrics`（Prometheus 端点），那是另一个层面的可观测性。

---

## 1. 测试原理

### 对比维度

同一份代码、同一个模型、同一套压测流量，**只切换调度优化的开关**，跑两轮对比：

| 轮次 | 调度策略 | QoS | MLFQ | Token 限速 | 过载管理 | 租户隔离 | Deadline-aware | Prefix Cache |
|------|---------|-----|------|-----------|---------|---------|---------------|--------------|
| **A — 基线** | `fcfs` | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| **B — 优化版** | `priority` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

> **关于 Prefix Cache**：`--enable-prefix-caching` 在两轮都开启（它是 vLLM 原生的自动前缀缓存，不属于本项目新增的调度优化），用于保证两轮在相同缓存条件下对比。本项目的 Prefix Cache **调度优化**（缓存感知调度、长上下文复用等）体现在调度器如何**利用**这个缓存，其效果通过服务端指标（`vllm:prefix_cache_hits` 等）和 Phase 2 的灰度切换场景来验证。详见下文 §1.1。

### 开关机制

调度优化通过环境变量 + 启动参数控制（已实现在 `vllm/v1/core/scheduler.py`）：

| 优化 | 关闭方式 | 默认 |
|------|---------|------|
| QoS 分级调度 | `--scheduling-policy fcfs` | fcfs（需显式指定 `priority` 才开启） |
| MLFQ 多级反馈 | `VLLM_DISABLE_MLFQ=1` | 开启 |
| Token 限速 | `VLLM_DISABLE_RATE_LIMIT=1` | 开启 |
| 租户隔离 | `VLLM_DISABLE_TENANT_ISO=1` | 开启 |
| 过载管理（准入+SLA抢占） | `VLLM_DISABLE_OVERLOAD_MGMT=1` | 开启 |
| Deadline-aware 调度 | `VLLM_DISABLE_DEADLINE_AWARE=1` | 开启 |
| Prefix Cache（自动前缀缓存） | 无开关（vLLM 原生） | `--enable-prefix-caching` 显式开启 |

> **基线模式** = 所有 `VLLM_DISABLE_*` 都设为 `1` + `--scheduling-policy fcfs`，模拟原生 vLLM V1 的纯 FCFS 行为。`--enable-prefix-caching` 在两轮**都开启**，保证缓存条件一致。

### 1.1 Prefix Cache 调度优化点

本项目除了 QoS / MLFQ / Token 限速等调度优化外，还实现了 Prefix Cache 相关的调度优化（详见 [`prefix-cache-scheduling-optimization.md`](../basic_optimization/prefix-cache-scheduling-optimization.md)）。它和前面的调度优化**本质同类**（都是调度器如何排序/复用资源），因此并入本测试一起验证。

Prefix Cache 优化的核心目标：**让调度器更聪明地复用已计算的 KV Cache**，减少重复 Prefill，从而降低 TTFT 和显存压力。

| 优化点 | 作用 | 在本测试中的验证场景 |
|--------|------|---------------------|
| 缓存感知调度（Cache-aware Scheduling） | 优先调度命中缓存的请求，减少 Prefill 开销 | Phase 2 灰度切换 + Phase 1 稳态 |
| 长上下文复用 | 长文档（Bronze）的共享前缀只算一次 | Phase 4 长文档暴增 |
| 缓存块高效管理 | 减少显存碎片，提升缓存利用率 | 全程（看 KV Cache 使用率） |
| 缓存命中统计 | 暴露命中率指标供可观测性分析 | Prometheus 指标 `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` |
| 缓存驱逐优化 | 高价值缓存优先保留，低价值先驱逐 | Phase 4→5 缓存压力增大时 |

> **为什么两轮都开 `--enable-prefix-caching`**：`--enable-prefix-caching` 是 vLLM 原生能力，属于"缓存机制"，本项目优化的是"调度器如何**利用**缓存"。两轮都开启原生缓存，是为了隔离变量——对比的是**调度器层面的缓存感知优化**，而非"有没有缓存"本身。

> **Prefix Cache 效果的量化指标**：
> - **命中率** = `vllm:prefix_cache_hits` / `vllm:prefix_cache_queries`（服务端指标，见 §扩展 -1）
> - **命中带来的 TTFT 下降**：Phase 2 里 v2 prompt 首次出现（未命中）vs 后续（命中）的 TTFT 对比
> - **Prefill 耗时下降**：`vllm:request_prefill_time_seconds` 在缓存命中时应明显变短

### 压测工具

复用本目录下的 [`workload.py`](workload.py)，使用 `--mode single`（单实例模式）。它模拟企业级 AI 平台，一个 vLLM 实例同时服务 7 个租户（Gold/Silver/Bronze），通过 5 个阶段递进引入压力：

| Phase | 时间 | 场景 | 暴露的优化点 |
|-------|------|------|-------------|
| 1 | 0-60s | 稳态预热 | QoS 基础优先级效果 |
| 2 | 60-120s | System Prompt 灰度切换 | 缓存版本管理 |
| 3 | 120-180s | Gold-A 流量暴增 4× | 租户隔离 + MLFQ 防饿死 |
| 4 | 180-240s | Bronze 长文档暴增 | Token 限速 + Prefill 预算隔离 |
| 5 | 240-300s | 全面过载 | 过载管理（准入 + SLA 抢占） |

---

## 2. 环境准备

### 前提

- 已按 `QUICK_START.md` 完成安装，环境就绪
- GPU 实例可用（RTX 5070 Ti / A10 / T4 / L4 均可）
- 模型 `Qwen/Qwen2.5-1.5B-Instruct` 已缓存

### 进入环境

```bash
cd ~/vllm-serving-optimization
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"
```

### 安装压测依赖

```bash
pip install aiohttp numpy
```

---

## 3. 拉起服务并压测

> ⚠️ 单卡无法同时跑两个 vLLM 实例（显存不够）。操作方式是 **先跑 A 收集数据，停掉，再跑 B 收集数据**，两轮用完全相同的流量参数（`--seed 42` 保证流量可复现）。

### 轮次 A — 基线（纯 FCFS，所有调度优化关闭）

#### A.1 启动服务

```bash
# 所有调度优化关闭，模拟原生 vLLM V1 FCFS 行为
VLLM_DISABLE_MLFQ=1 \
VLLM_DISABLE_RATE_LIMIT=1 \
VLLM_DISABLE_TENANT_ISO=1 \
VLLM_DISABLE_OVERLOAD_MGMT=1 \
VLLM_DISABLE_DEADLINE_AWARE=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --scheduling-policy fcfs \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
```

> 等待看到 `Application startup complete` 即就绪。此终端保持开着。

#### A.2 运行压测

另开一个 WSL 终端：

```bash
cd ~/vllm-serving-optimization
source .venv/bin/activate

python benchmarks/e2e_cases/workload.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --mode single \
    --duration 300 \
    --seed 42 \
    --output-dir results/baseline_fcfs/
```

> 全程约 5 分钟。结束后结果保存在 `results/baseline_fcfs/`。

#### A.3 停止服务

在服务终端按 `Ctrl+C`，等待进程退出，确认 GPU 显存已释放：

```bash
nvidia-smi  # 确认无残留 vllm 进程
```

---

### 轮次 B — 优化版（QoS + MLFQ + Token 限速全开）

#### B.1 启动服务

```bash
# 默认所有优化开启，只需指定 scheduling-policy=priority 启用 QoS
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --max-model-len 8192 \
    --enable-prefix-caching \
    --scheduling-policy priority \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --port 8000
```

> 等待看到 `Application startup complete` 即就绪。

#### B.2 运行压测

另开一个 WSL 终端：

```bash
cd ~/vllm-serving-optimization
source .venv/bin/activate

python benchmarks/e2e_cases/workload.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --host 127.0.0.1 --port 8000 \
    --mode single \
    --duration 300 \
    --seed 42 \
    --output-dir results/optimized_qos/
```

> 流量参数与轮次 A **完全相同**（`--seed 42`），唯一区别是后端调度策略。

#### B.3 停止服务

`Ctrl+C` 停止服务。

---

## 4. 结果对比

### 4.1 输出文件结构

每轮压测结束后，`results/<目录>/` 下会生成：

```
results/
├── baseline_fcfs/
│   ├── raw_metrics.json      # 原始时序数据（workload.py）
│   ├── summary.json           # 按 Phase / 租户聚合的统计（workload.py）
│   └── abs_bench/
│       └── abs_bench.json     # 绝对性能基线（vllm bench serve）
└── optimized_qos/
    ├── raw_metrics.json
    ├── summary.json
    └── abs_bench/
        └── abs_bench.json
```

### 4.2 关键对比指标

对照 `scheduling-resource-management-optimization.md` 和 `prefix-cache-scheduling-optimization.md` 的预期目标，重点看以下指标：

| 对比维度 | Phase | 基线预期 | 优化版预期 | 对应优化 |
|---------|-------|---------|-----------|---------|
| Gold-A P99 TTFT | 1 | 基准 | ↓ 30-50% | QoS 分级（短请求优先） |
| Silver P99 TTFT（Gold-A 暴增时） | 3 | 大幅恶化 | 明显更好 | MLFQ + QoS（防饿死） |
| Gold-A TPOT 抖动（P99-P50） | 3 | 大 | ↓ 40%+ | Token 限速（低优让路） |
| 被接受请求 P99 TTFT | 5 | >10s（全违约） | <800ms | 过载管理（准入控制） |
| 合理拒绝率 | 5 | 0%（全收全违约） | 30-40% | 过载管理（SLA 感知拒绝） |
| Bronze 长文档对 Gold 的影响 | 4 | 严重干扰 | 有限影响 | Token 限速 + 租户隔离 |
| **Prefix Cache 命中率** | 全程 | 基准（仅原生缓存） | 更高（缓存感知调度优先命中） | 缓存感知调度 |
| **Phase 2 v2 首现 vs 后续 TTFT** | 2 | 首现慢、后续快 | 后续命中更明显 | 缓存版本管理 + 命中复用 |
| **长文档共享前缀 Prefill 耗时** | 4 | 高（每次重复 Prefill） | 低（共享前缀只算一次） | 长上下文复用 |

> **Prefix Cache 指标的读取**：命中率、Prefill 耗时来自服务端 `/metrics`（`vllm:prefix_cache_hits`、`vllm:prefix_cache_queries`、`vllm:request_prefill_time_seconds`），需要通过 Prometheus + Grafana 查看（见 §扩展 -1）。客户端 `workload.py` 和 `vllm bench serve` 测不到这些服务端内部指标。

### 4.3 快速对比脚本

用以下命令快速提取两轮的关键指标做对比：

```bash
cd ~/vllm-serving-optimization

echo "========== 基线 (FCFS) =========="
python -c "
import json
with open('results/baseline_fcfs/summary.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=2, ensure_ascii=False))
" 2>/dev/null || echo "结果文件尚未生成"

echo ""
echo "========== 优化版 (QoS+MLFQ+Token限速) =========="
python -c "
import json
with open('results/optimized_qos/summary.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=2, ensure_ascii=False))
" 2>/dev/null || echo "结果文件尚未生成"
```

### 4.4 判断优化是否生效的信号

如果优化生效，你应该观察到：

1. **Phase 1（稳态）**：优化版 Gold-A 的 TTFT 明显低于基线（短请求被优先调度）
2. **Phase 3（Gold-A 暴增）**：
   - 基线：Silver/Bronze 请求被饿死，TTPT 飙升
   - 优化版：MLFQ 让暴增的 Gold-A 请求逐渐降级，Silver 不会被完全饿死
3. **Phase 4（长文档暴增）**：
   - 基线：Bronze 长文档的 Prefill 占满 token_budget，拖慢所有请求
   - 优化版：Token 限速让 Bronze 低速运行，Gold 不受影响
4. **Phase 5（全面过载）**：
   - 基线：所有请求都收，全部 SLA 违约，P99 TTFT > 10s
   - 优化版：过载管理拒绝低优请求，被接受请求的 P99 TTFT < 800ms

---

## 5. 注意事项

### 公平性保障

- **两轮使用相同 `--seed 42`**：保证流量模式（请求到达时间、prompt 内容、租户分布）完全一致，唯一变量是调度策略
- **两轮使用相同模型和启动参数**：`--max-model-len`、`--gpu-memory-utilization`、`--enforce-eager` 等保持一致
- **两轮之间重启服务**：避免 KV Cache 残留影响第二轮结果。务必确认 `nvidia-smi` 显示 GPU 显存已释放后再启动第二轮

### Prefix Caching 的处理

两轮都保留 `--enable-prefix-caching`。原因是：Prefix Caching 属于 KV Cache 方向的优化，不是调度方向。我们对比的是**调度策略**的效果，所以要控制变量——Prefix Caching 在两轮中保持一致（都开），只让调度策略不同。

### 如果优化效果不明显

可能的原因：
1. **负载不够高**：QoS/MLFQ/Token限速的优势在高负载（Phase 3-5）才明显，Phase 1 稳态差异可能不大
2. **模型太小**：1.5B 模型的 Prefill/Decode 都很快，调度差异被掩盖。可换更大模型（如 7B）复测
3. **`--enforce-eager` 影响**：禁用了 CUDA Graph，单步开销变大，可能放大调度效果。如需更贴近生产，可去掉 `--enforce-eager`（但显存占用会增加）

---

### 扩展 -1：服务端指标采集（Prometheus + Grafana）

#### 为什么需要这一步

前面所有工具（`workload.py` 和 `vllm bench serve`）都只能测**客户端视角**的延迟（TTFT / E2E / TPOT），但调度器**内部状态**——比如队列长度、KV Cache 使用率、running/waiting 请求数、抢占次数——客户端完全看不到。

这些服务端内部指标对解释"为什么 TTFT 飙到 80 秒"至关重要：

- Phase 4 Gold-A 的 TTFT 80s 是因为 `vllm:num_requests_waiting` 堆积？还是 `vllm:kv_cache_usage_perc` 满了触发抢占？
- 优化版的过载管理生效时，`vllm:num_preemptions` 有没有变化？
- `vllm:request_queue_time_seconds` 能直接看到请求在队列里等了多久，比客户端推算更准

vLLM 内置了 Prometheus 指标导出，配合 Prometheus + Grafana 可以可视化这些内部状态。

#### -1.1 vLLM 服务端：指标默认开启

> **重要**：当前 vLLM 版本（V1 引擎）**默认就开启了 `/metrics` 端点**，不需要 `--enable-metrics` 参数。`/metrics` 路由在 `vllm/entrypoints/serve/instrumentator/metrics.py` 中无条件挂载到 FastAPI app。

验证：服务启动后，直接 curl：

```bash
curl http://127.0.0.1:8000/metrics | head -20
```

应该能看到类似：

```
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="Qwen/Qwen2.5-1.5B-Instruct"} 0.0
# HELP vllm:num_requests_waiting Number of requests waiting in the waiting queue.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="Qwen/Qwen2.5-1.5B-Instruct"} 0.0
...
```

> **关于 `--disable-log-stats`**：这个参数默认 `False`，意味着 stats 默认开启，metrics 会被正常填充。**不要加 `--disable-log-stats`**，否则 `/metrics` 里的很多指标会是空的。如果你发现 `/metrics` 返回的指标值都是 0，检查是不是误加了这个参数。

#### -1.2 关键指标速查

vLLM V1 导出的指标（前缀 `vllm:`，完整列表见 `vllm/v1/metrics/loggers.py`）：

| 指标名 | 类型 | 含义 | 对本测试的价值 |
|--------|------|------|---------------|
| `vllm:num_requests_running` | Gauge | GPU 上正在跑的请求数 | 看并发是否打满 |
| `vllm:num_requests_waiting` | Gauge | 等待队列中的请求数 | **核心**：TTFT 飙升的直接原因 |
| `vllm:kv_cache_usage_perc` | Gauge | KV Cache 使用率（0-1） | 看是否触发抢占 / 驱逐 |
| `vllm:num_preemptions` | Counter | 累计抢占次数 | 过载管理是否生效的硬证据 |
| `vllm:time_to_first_token_seconds` | Histogram | TTFT 分布（服务端视角） | 对比客户端 TTFT，分离网络开销 |
| `vllm:inter_token_latency_seconds` | Histogram | ITL 分布 | Decode 流畅度 |
| `vllm:request_time_per_output_token_seconds` | Histogram | TPOT 分布 | 平均单 token 耗时 |
| `vllm:e2e_request_latency_seconds` | Histogram | E2E 延迟分布 | 服务端视角的总延迟 |
| `vllm:request_queue_time_seconds` | Histogram | **请求排队等待时间** | 客户端测不到，直接定位调度瓶颈 |
| `vllm:request_prefill_time_seconds` | Histogram | Prefill 阶段耗时 | 区分 Prefill vs Decode 瓶颈 |
| `vllm:request_decode_time_seconds` | Histogram | Decode 阶段耗时 | 区分 Prefill vs Decode 瓶颈 |
| `vllm:prompt_tokens_cached` | Counter | Prefix Cache 命中的 token 数 | 验证 Prefix Caching 效果 |
| `vllm:prefix_cache_queries` | Counter | Prefix Cache 查询次数 | 配合 hits 算命中率 |

> **Histogram 与客户端百分位的区别**：服务端 Histogram 用固定 bucket（如 TTFT 的 bucket 是 `[0.001, 0.005, 0.01, 0.02, 0.04, ..., 2560.0]` 秒），通过 `histogram_quantile()` 在 PromQL 里估算 P99。客户端工具则是精确计算每个请求的 TTFT 后取百分位。两者会有微小差异（Histogram 是近似），但趋势一致。

#### -1.3 配置 Prometheus 拉取

安装 Prometheus（Ubuntu 本地安装，systemd 管理）：

```bash
# 1. 更新软件包列表
sudo apt update

# 2. 安装 Prometheus
sudo apt install prometheus -y

# 3. 启动 Prometheus 服务并设置开机自启
sudo systemctl start prometheus
sudo systemctl enable prometheus

# 4. 检查服务状态（确保显示 active (running)）
sudo systemctl status prometheus

# 安装完成后，Prometheus 的配置文件位于 /etc/prometheus/prometheus.yml
```

配置 Prometheus 抓取 vLLM（直接修改 `/etc/prometheus/prometheus.yml`）：

```yaml
global:
  scrape_interval: 5s        # 5 秒拉一次（默认 15s 太粗，压测只有 300s）
  evaluation_interval: 5s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['127.0.0.1:8000']   # 本地部署直接用 127.0.0.1
        labels:
          instance: 'vllm-single'
```

修改后重启 Prometheus 使配置生效：

```bash
# 验证配置语法
promtool check config /etc/prometheus/prometheus.yml

# 重启
sudo systemctl restart prometheus
```

> **关于 `scrape_interval`**：默认 15 秒对 300 秒的压测来说太粗（每个 phase 只有 4 个采样点），建议设为 5 秒（每个 phase 有 12 个采样点）。如果磁盘允许，可以设到 2 秒。

> **关于 targets 地址**：本地部署（非 Docker）直接填 `127.0.0.1:8000` 或 `localhost:8000`。只有当你用 Docker 跑 Prometheus、vLLM 跑在宿主机时，才需要用 `host.docker.internal` 这类跨容器域名。

#### -1.4 配置 Grafana 展示

安装 Grafana（Ubuntu 本地安装，systemd 管理）：

```bash
# 1. 安装依赖工具
sudo apt-get install -y apt-transport-https wget gnupg

# 2. 添加 Grafana 的 GPG 密钥
sudo mkdir -p /etc/apt/keyrings/
sudo wget -O /etc/apt/keyrings/grafana.asc https://apt.grafana.com/gpg-full.key
sudo chmod 644 /etc/apt/keyrings/grafana.asc

# 3. 添加 Grafana 的 APT 仓库
echo "deb [signed-by=/etc/apt/keyrings/grafana.asc] https://apt.grafana.com stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list

# 4. 更新软件包列表并安装 Grafana (开源版)
sudo apt-get update
sudo apt-get install grafana -y

# 5. 启动 Grafana 服务并设置开机自启
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# 6. 检查服务状态
sudo systemctl status grafana-server

# Grafana 的配置文件位于 /etc/grafana/grafana.ini
```

配置步骤：

1. **添加数据源**：浏览器打开 `http://localhost:3000`（默认账号 `admin` / `admin`）
   - Configuration → Data Sources → Add data source → Prometheus
   - URL 填 `http://127.0.0.1:9090`（本地部署）
   - Save & Test

2. **创建 Dashboard**：可以手动建，也可以导入社区现成的。推荐手动建以下核心面板：

#### -1.5 核心 Dashboard 面板设计

按本测试的 5 阶段场景，建议建以下面板（PromQL 直接复制可用）：

**面板 1：请求队列长度（核心，暴露调度瓶颈）**

```promql
# Running vs Waiting 请求数（双线图）
vllm:num_requests_running
```
```promql
vllm:num_requests_waiting
```

> 解读：Phase 4 时如果 `waiting` 飙到 50+ 而 `running` 卡在某个值，说明 Bronze 长文档把 KV Cache 占满，新请求全部排队。

**面板 2：KV Cache 使用率**

```promql
vllm:kv_cache_usage_perc
```

> 解读：使用率达到 1.0 时触发抢占/驱逐。对比 A/B 两轮，看优化版是否通过 Token 限速让 KV Cache 使用更平滑。

**面板 3：抢占次数（累计）**

```promql
# 用 increase 算每个 phase 的增量
increase(vllm:num_preemptions[60s])
```

> 解读：按 60s 窗口算增量，正好对应 5 个 phase。优化版的过载管理应该让抢占更"有意为之"（SLA 感知）而非被动触发。

**面板 4：TTFT P99（服务端视角）**

```promql
histogram_quantile(0.99, sum(rate(vllm:time_to_first_token_seconds_bucket[30s])) by (le))
```

> 解读：跟客户端 `workload.py` 测的 TTFT 对比，差值就是网络 + 客户端处理开销。

**面板 5：请求排队时间 P99（客户端测不到）**

```promql
histogram_quantile(0.99, sum(rate(vllm:request_queue_time_seconds_bucket[30s])) by (le))
```

> 解读：这是服务端独有指标，直接显示请求在调度队列里等了多久。Phase 4 的 80s TTFT 里，如果 queue_time 占 78s，说明瓶颈在排队而非 Prefill。

**面板 6：Prefill vs Decode 耗时对比**

```promql
# Prefill P95
histogram_quantile(0.95, sum(rate(vllm:request_prefill_time_seconds_bucket[30s])) by (le))
```
```promql
# Decode P95
histogram_quantile(0.95, sum(rate(vllm:request_decode_time_seconds_bucket[30s])) by (le))
```

> 解读：Bronze 长文档的 Prefill 应该明显高于 Gold 短对话。如果优化版的 Token 限速生效，Phase 4 的 Prefill 时长应该被"削平"。

**面板 7：Prefix Cache 命中率**

```promql
# 命中率
sum(rate(vllm:prefix_cache_hits[30s])) / sum(rate(vllm:prefix_cache_queries[30s]))
```

> 解读：Phase 2 的 Prompt 灰度切换后，命中率应该从高（v1 命中）降到低（v2 未缓存）再回升。两轮表现应一致（Prefix Cache 不受调度策略影响）。

#### -1.6 完整采集流程（A/B 两轮）

```bash
cd ~/vllm-serving-optimization

# ===== 一次性准备：启动 Prometheus + Grafana =====
# 1. 配置 Prometheus（修改 /etc/prometheus/prometheus.yml，见 -1.3）
sudo tee /etc/prometheus/prometheus.yml > /dev/null << 'EOF'
global:
  scrape_interval: 5s
  evaluation_interval: 5s
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['127.0.0.1:8000']
        labels:
          instance: 'vllm-single'
EOF

# 2. 重启 Prometheus 使配置生效
promtool check config /etc/prometheus/prometheus.yml
sudo systemctl restart prometheus

# 3. 启动 Grafana（若已安装但未启动）
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# 4. Grafana 添加 Prometheus 数据源（见 -1.4）

# ===== 轮次 A：基线 =====
# 启动 vLLM 基线服务（A.1 的命令，/metrics 自动开启）
# 在 Grafana 里确认能看到 vllm:num_requests_running 等指标
# 跑 workload.py 压测
# 压测结束后，在 Grafana 里把时间范围锁定到压测时段，截图保存

# ===== 轮次 B：优化版 =====
# 重启 vLLM 为优化版配置（B.1）
# 跑相同 workload.py 压测
# Grafana 截图

# ===== 对比 =====
# 把 A/B 两轮的 Grafana 截图并排放，重点对比：
# - Phase 3/4 的 waiting 队列长度（优化版应更短）
# - Phase 4 的 kv_cache_usage_perc（优化版应更平滑）
# - Phase 5 的 num_preemptions 增量（优化版应更可控）
# - 全程的 queue_time P99（优化版应更低）
```

#### -1.7 对比要点

| 指标 | Phase | 基线预期 | 优化版预期 | 对应优化 |
|------|-------|---------|-----------|---------|
| `num_requests_waiting` 峰值 | 3/4/5 | 高（堆积严重） | 明显更低 | QoS + MLFQ 防饿死 |
| `kv_cache_usage_perc` 波动 | 4 | 频繁打满 1.0 | 更平滑 | Token 限速 + 租户隔离 |
| `num_preemptions` 增量 | 5 | 高且无序 | 低且 SLA 感知 | 过载管理（准入控制） |
| `request_queue_time_seconds` P99 | 4 | >70s | 明显更低 | Prefill 预算隔离 |
| `request_prefill_time_seconds` P95 | 4 | 极高（长文档） | 被 Token 限速削平 | Token 限速 |

#### -1.8 注意事项

1. **Prometheus 数据持久化**：本地安装（systemd）的 Prometheus 数据默认存在 `/var/lib/prometheus/`，重启服务不会丢失。如果想保留两轮数据做长期对比，直接保存即可；更简单的做法是**压测时直接 Grafana 截图**。

2. **scrape_interval 不能太短**：设到 1 秒会增加 vLLM 服务端开销（每次 scrape 都要遍历所有指标）。5 秒是平衡点。

3. **多轮对比的标签**：如果要同时保留 A/B 数据做查询对比，可以在 `prometheus.yml` 里用 `relabel_configs` 给不同轮次打不同 label。但更简单的做法是**每轮压测后导出 Grafana 截图**，避免 Prometheus 配置复杂化。

4. **指标不是越多越好**：`--kv-cache-metrics`、`--cudagraph-metrics`、`--enable-mfu-metrics` 这些开关会开启额外指标，但会增加服务端开销。本测试**不需要**开这些，默认指标已经足够。

5. **与客户端指标的协同分析**：服务端 `vllm:time_to_first_token_seconds` 和客户端 `workload.py` 的 `ttft_ms` 的差值 ≈ 网络往返 + 客户端 SSE 解析开销。如果差值异常大（>50ms），说明客户端或网络有问题，需要排查。

---

### 扩展 0：补充绝对性能基线（vllm bench serve）

#### 为什么需要这一步

`workload.py` 测的是**调度策略对多租户混合负载的公平性**（按 tier × phase × type 切分 TTFT/SLA 违约率），但它**无法回答**："优化版有没有牺牲整体吞吐？" 这个 reviewer 必问的问题。

vLLM 官方提供的 `vllm bench serve`（即 `vllm/benchmarks/serve.py`）正好补这个缺口：单一流量源、标准数据集、恒定 QPS，输出**绝对性能指标**（吞吐、P99 TTFT、P99 TPOT、ITL）。

两层证据结合才能说清楚：**优化版在保持绝对性能不退化的前提下，还改善了多租户公平性**。

#### 0.1 启动服务（复用轮次 A/B 的服务即可）

`vllm bench serve` 只是客户端工具，服务端不用改。直接复用前面 A.1 / B.1 启动的服务即可，**不要重启**——否则 KV Cache 状态会变。

如果服务已停，按 A.1 或 B.1 重新启动，等 `Application startup complete`。

#### 0.2 运行绝对性能基线（A/B 各跑一次）

> 关键：A/B 两轮用**完全相同**的 bench 参数，唯一变量是服务端调度策略。

在服务终端之外另开一个终端：

```bash
cd ~/vllm-serving-optimization
source .venv/bin/activate

# ===== 轮次 A：基线（FCFS）服务在跑时执行 =====
vllm bench serve \
    --backend openai \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 1000 \
    --request-rate 10 \
    --burstiness 1.0 \
    --temperature 0 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir results/baseline_fcfs/abs_bench/ \
    --result-filename abs_bench.json \
    --metadata tier=baseline policy=fcfs

# 停掉基线服务（Ctrl+C），启动轮次 B 优化版服务，确认就绪后：

# ===== 轮次 B：优化版（QoS+MLFQ+Token限速）服务在跑时执行 =====
vllm bench serve \
    --backend openai \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 \
    --endpoint /v1/completions \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 128 \
    --num-prompts 1000 \
    --request-rate 10 \
    --burstiness 1.0 \
    --temperature 0 \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,95,99 \
    --save-result \
    --result-dir results/optimized_qos/abs_bench/ \
    --result-filename abs_bench.json \
    --metadata tier=optimized policy=priority
```

#### 0.3 参数说明

| 参数 | 值 | 含义 |
|------|-----|------|
| `--backend openai` | OpenAI 兼容 | 走 `/v1/completions` 接口 |
| `--dataset-name random` | 随机 token 数据集 | 可控输入/输出长度，避免 ShareGPT 数据偏置 |
| `--random-input-len 1024` | 输入 1024 token | 中等长度 prompt，覆盖典型场景 |
| `--random-output-len 128` | 输出 128 token | 短输出，便于快速跑完 1000 条 |
| `--num-prompts 1000` | 1000 条请求 | 统计样本足够算稳定的 P99 |
| `--request-rate 10` | 10 QPS | 恒定速率，Poisson 到达（`burstiness=1.0`） |
| `--temperature 0` | 贪心解码 | 消除采样随机性，两轮输出完全可比 |
| `--ignore-eos` | 忽略 EOS | 强制跑到 `output_len`，避免提前结束导致长度不一致 |
| `--percentile-metrics ttft,tpot,itl,e2el` | 四个指标全报 | TTFT/TPOT/ITL/E2EL 都看 |
| `--metric-percentiles 50,95,99` | P50/P95/P99 | 覆盖中位、尾部、极端尾 |
| `--save-result` | 保存 JSON | 自动写到 `--result-dir` |
| `--metadata` | 自定义元数据 | 标记这轮是 baseline/optimized，方便后续脚本聚合 |

> **关于 QPS 选择**：10 QPS 是保守值。如果想测吞吐上限，可以加跑一组 `--request-rate inf`（不限速，所有请求瞬间发出，测最大并发处理能力）。但对比 A/B 时**两轮必须用相同 QPS**，否则没有可比性。

#### 0.4 输出解读

终端会直接打印结果表格，类似：

```
============ Serving Benchmark Result ============
Successful requests:              1000
Failed requests:                  0
Request rate configured (RPS):    10.00
Benchmark duration (s):           102.34
Total input tokens:               1024000
Total generated tokens:           128000
Request throughput (req/s):       9.77
Output token throughput (tok/s):  1250.32
Total token throughput (tok/s):   11250.45
-------------- Time to First Token --------------
Mean TTFT (ms):                   45.23
Median TTFT (ms):                 42.10
P99 TTFT (ms):                    125.67
-------------- Time per Output Token (excl. 1st token) --------------
Mean TPOT (ms):                   18.45
Median TPOT (ms):                 18.20
P99 TPOT (ms):                    32.15
-------------- Inter-token Latency --------------
Mean ITL (ms):                    18.32
Median ITL (ms):                  18.10
P99 ITL (ms):                     35.78
-------------- End-to-end Latency --------------
Mean E2EL (ms):                   2340.56
Median E2EL (ms):                 2320.10
P99 E2EL (ms):                    2680.45
==================================================
```

JSON 结果保存在 `results/<轮次>/abs_bench/abs_bench.json`。

#### 0.5 A/B 对比要点

把两轮的 `abs_bench.json` 放一起对比，重点看：

| 指标 | 期望 | 异常信号 |
|------|------|---------|
| **Request throughput (req/s)** | B ≈ A（±5% 内） | B 明显低于 A → 优化版牺牲了吞吐 |
| **Output token throughput (tok/s)** | B ≈ A | B 明显低 → 调度开销过大 |
| **P99 TTFT** | B ≤ A 或接近 | B 明显高 → QoS/MLFQ 引入了排队开销 |
| **P99 TPOT** | B ≈ A | B 明显高 → Token 限速过度限制了 Decode |
| **P99 ITL** | B ≈ A | ITL 抖动大 → 调度抢占频繁，影响生成流畅度 |
| **Failed requests** | 都为 0 | B 有失败 → 过载管理误杀 |

**判断优化是否"安全"的标准**：

- ✅ **安全优化**：B 的吞吐/TPOT/ITL 与 A 接近（±5%），同时 `workload.py` 的多租户公平性指标明显改善 → 优化版可上线
- ⚠️ **有代价优化**：B 的吞吐略低（5-10%）但公平性大幅改善 → 需权衡业务优先级
- ❌ **过度优化**：B 的吞吐/TPOT 明显恶化（>10%）→ 调度开销超过了公平性收益，需调参

#### 0.6 可选：扫多档 QPS 测吞吐曲线

如果想看完整性能曲线（而不只是单点），可以扫几档 QPS：

```bash
for qps in 5 10 20 40 80; do
    vllm bench serve \
        --backend openai \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --base-url http://127.0.0.1:8000 \
        --endpoint /v1/completions \
        --dataset-name random \
        --random-input-len 1024 --random-output-len 128 \
        --num-prompts 500 \
        --request-rate $qps \
        --temperature 0 --ignore-eos \
        --percentile-metrics ttft,tpot \
        --metric-percentiles 99 \
        --save-result \
        --result-dir results/baseline_fcfs/abs_bench/ \
        --result-filename abs_bench_${qps}qps.json \
        --metadata tier=baseline policy=fcfs qps=$qps
done
```

A/B 各扫一遍，画出 `QPS vs P99 TTFT` 和 `QPS vs throughput` 曲线，能更直观看到优化版在哪个负载水平开始有优势或劣势。

> **注意**：扫多档 QPS 很耗时（5 档 × 2 轮 × ~100s = ~17 分钟），且每档之间服务端 KV Cache 状态会变化。如果时间紧，单点 10 QPS 对比已经足够说明问题。

---

### 扩展 1：单独验证某一项优化

如果想隔离验证单项优化的贡献，可以逐个开启：

```bash
# 只开 QoS（关闭其他）
VLLM_DISABLE_MLFQ=1 VLLM_DISABLE_RATE_LIMIT=1 VLLM_DISABLE_OVERLOAD_MGMT=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-prefix-caching --scheduling-policy priority \
    --gpu-memory-utilization 0.85 --enforce-eager --port 8000

# 只开 MLFQ（关闭其他，QoS 用 fcfs）
VLLM_DISABLE_RATE_LIMIT=1 VLLM_DISABLE_OVERLOAD_MGMT=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-prefix-caching --scheduling-policy fcfs \
    --gpu-memory-utilization 0.85 --enforce-eager --port 8000

# 只开 Token 限速（关闭其他，QoS 用 fcfs）
VLLM_DISABLE_MLFQ=1 VLLM_DISABLE_OVERLOAD_MGMT=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-prefix-caching --scheduling-policy fcfs \
    --gpu-memory-utilization 0.85 --enforce-eager --port 8000
```

每轮用不同的 `--output-dir` 保存结果，即可做单项消融实验。

---

## 6. 完整流程速查

```bash
# ===== 一次性准备：启动 Prometheus + Grafana =====
cd ~/vllm-serving-optimization && source .venv/bin/activate

# 1. 配置 Prometheus（写入 /etc/prometheus/prometheus.yml）
sudo tee /etc/prometheus/prometheus.yml > /dev/null << 'EOF'
global:
  scrape_interval: 5s
  evaluation_interval: 5s
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['127.0.0.1:8000']
        labels:
          instance: 'vllm-single'
EOF

# 2. 重启 Prometheus 使配置生效
sudo systemctl restart prometheus

# 3. 启动 Grafana（若未启动）
sudo systemctl start grafana-server
sudo systemctl enable grafana-server

# 浏览器打开 http://localhost:3000 添加 Prometheus 数据源（见 §扩展 -1.4）

# ===== 轮次 A：基线 =====

# 终端 1：启动基线服务
VLLM_DISABLE_MLFQ=1 VLLM_DISABLE_RATE_LIMIT=1 VLLM_DISABLE_TENANT_ISO=1 \
VLLM_DISABLE_OVERLOAD_MGMT=1 VLLM_DISABLE_DEADLINE_AWARE=1 \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-prefix-caching --scheduling-policy fcfs \
    --gpu-memory-utilization 0.85 --enforce-eager --port 8000

# 终端 2：跑多租户混合负载压测（公平性指标）
python benchmarks/e2e_cases/workload.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 \
    --mode single --duration 300 --seed 42 \
    --output-dir results/baseline_fcfs/

# Grafana 截图：锁定压测时段，保存队列长度 / KV Cache / 抢占次数等面板截图

# 终端 2：跑绝对性能基线（吞吐/延迟指标，服务不用重启）
vllm bench serve \
    --backend openai --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 1000 --request-rate 10 --temperature 0 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir results/baseline_fcfs/abs_bench/ \
    --result-filename abs_bench.json \
    --metadata tier=baseline policy=fcfs

# 停止服务（Ctrl+C），确认显存释放
nvidia-smi

# ===== 轮次 B：优化版 =====
# 终端 1：启动优化版服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
    --enable-prefix-caching --scheduling-policy priority \
    --gpu-memory-utilization 0.85 --enforce-eager --port 8000

# 终端 2：跑多租户混合负载压测（相同流量）
python benchmarks/e2e_cases/workload.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8000 \
    --mode single --duration 300 --seed 42 \
    --output-dir results/optimized_qos/

# Grafana 截图：锁定压测时段，保存相同面板截图用于 A/B 对比

# 终端 2：跑绝对性能基线（相同 bench 参数）
vllm bench serve \
    --backend openai --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 1000 --request-rate 10 --temperature 0 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir results/optimized_qos/abs_bench/ \
    --result-filename abs_bench.json \
    --metadata tier=optimized policy=priority

# ===== 对比结果 =====
# 公平性指标（workload.py）
ls results/baseline_fcfs/ results/optimized_qos/
# 绝对性能指标（vllm bench serve）
ls results/baseline_fcfs/abs_bench/ results/optimized_qos/abs_bench/
# 服务端内部指标（Grafana 截图，手动保存到 results/ 下）
#   baseline_fcfs/grafana_queue.png       - 队列长度
#   baseline_fcfs/grafana_kv_cache.png    - KV Cache 使用率
#   baseline_fcfs/grafana_preemptions.png - 抢占次数
#   optimized_qos/grafana_*.png           - 同上
```

---

## 7. 测试结果记录与方案调整

> 本节记录实际执行的 A/B 测试结论，以及根据测试结果对方案的调整方向。
> 测试环境：RTX 5070 Ti（Blackwell，sm_120），Qwen2.5-1.5B-Instruct，
> `--enforce-eager`，单实例，5 阶段混合负载（Phase 1-5，各 60s）。

### 7.1 代码移植背景（重要）

原始优化代码基于**旧版 vLLM 调度器结构**（`vllm/v1/core/scheduler.py` 单文件），
而运行环境是**官方 vLLM v0.20.2**（调度器已重构为 `vllm/v1/core/sched/scheduler.py`
+ `RequestQueue` 抽象）。两者 API 不兼容，直接在仓库目录启动会因新旧代码混血导致大量
`ImportError` / `NameError`。

因此将全部优化**移植到基于官方 v0.20.2 的新分支** `feature/qos-optimization-v0202`：

| 优化 | 移植位置 | 开关 |
|------|---------|------|
| QoS 优先级 | `effective_priority`（v0.20.2 已内置 `_update_effective_priorities`） | `--scheduling-policy priority` |
| MLFQ 多级反馈 | `request.mlfq_level` + `update_from_output` 降级 | `VLLM_DISABLE_MLFQ` |
| Token 限速 | `TokenRateLimiter` + `schedule()` 接入 | `VLLM_DISABLE_RATE_LIMIT` |
| 租户隔离 | `TenantManager` + `schedule()` 并发上限 | `VLLM_DISABLE_TENANT_ISO` |
| 过载管理准入 | `_should_admit` + `add_request()` | `VLLM_DISABLE_OVERLOAD_MGMT` |
| Deadline-aware | `_deadline_aware_sort_waiting` + `schedule()` | `VLLM_DISABLE_DEADLINE_AWARE` |
| Prefill 预算隔离 | `schedule()` WAITING 循环接入 | — |

### 7.2 A/B 测试结论（2026-08-16）

#### 分阶段违约率对比

| 阶段 | A 基线(FCFS) | B 优化(priority) | 结论 |
|------|-------------|-----------------|------|
| Phase 1（稳态） | 48ms, 0% | 54ms, 0.1% | 相当 |
| Phase 2（Prompt 切换） | 86ms, 0% | 90ms, 0% | 相当 |
| Phase 3（Gold-A 暴增 4×） | 57ms, 0% | 132ms, 0% | 相当（B 略慢但零违约） |
| Phase 4（长文档暴增） | 94072ms, 87.9% | 87343ms, 88.1% | P99 降 9-13%，违约率持平 |
| Phase 5（全局过载） | 14690ms, 100% | 20885ms, 100% | 都崩溃 |

#### Phase 4 分租户违约率

| 租户 | A 基线 | B 优化 |
|------|--------|--------|
| gold | 89.7% | 89.9% |
| silver | 87.6% | ~88% |
| bronze | 85.1% | ~85% |

#### 核心结论

1. **Phase 1-3（稳态~适度过载）**：优化版与基线相当，无回归，零 SLA 违约 ✅
2. **Phase 4（长文档暴增）**：优化版 P99 TTFT 降低 9-13%，但违约率几乎持平 ❌
3. **Phase 5（极端过载）**：两版都 100% 违约，GPU 物理算力耗尽 ❌

### 7.3 为什么优化效果有限（根因分析）

Phase 4/5 违约率的**根本原因是长文档 Prefill 的绝对耗时**，而非调度排序问题：

1. **Bronze 长文档**（300 token 输出 + 长 prompt）在 1.5B 模型 + `--enforce-eager` 下，
   单次 Prefill 本身就要数秒到数十秒。
2. Phase 4 时 Bronze QPS 从 3 涨到 10，叠加 Gold-A 32 QPS，**总量超过 GPU 处理能力**。
3. 一旦 GPU 打满，**任何调度策略都无法让所有请求满足 SLA**——物理上算不过来。

因此：调度优化（QoS/MLFQ/Token 限速/租户隔离/Prefill 预算隔离）在**适度过载**下有效，
但在**极端过载**下受限于物理算力，无法让所有请求满足 SLA。

### 7.4 方案调整：准入控制阈值调优（下一步方向）

真正能解决 Phase 4 的是**过载管理的「准入控制拒绝」**——当系统过载时主动拒绝
低优先级请求（HTTP 503），而不是全收然后全部超时。

当前 `_should_admit()` 的阈值**过于宽松**，导致 Phase 4 没有触发拒绝：

```python
self.max_queue_depth: int = 100              # 队列深度上限（过高）
self.overload_violation_threshold: float = 0.5  # SLA 违约率阈值（过高）
self._sla_violation_window: deque = deque(maxlen=50)  # 违约统计窗口
```

#### 调整思路

| 参数 | 当前值 | 建议值 | 调整理由 |
|------|--------|--------|---------|
| `max_queue_depth` | 100 | **30~50** | Phase 4 队列深度未达 100，导致准入控制从未触发 |
| `overload_violation_threshold` | 0.5 | **0.2~0.3** | 更早识别过载，及时拒绝低优先级请求 |
| `_sla_violation_window` maxlen | 50 | **20~30** | 更灵敏地反映近期违约率 |

#### 关键补充：准入控制必须与优先级联动

准入控制的核心原则是「**保护高优先级请求，牺牲低优先级请求**」：

1. **gold（priority=-2）**：永不拒绝，保证 Gold SLA
2. **silver（priority=0）**：违约率超过阈值时拒绝
3. **bronze（priority=2）**：队列深度或违约率一超阈值即拒绝

当前 `_should_admit()` 已实现 `_is_high_priority_request()` 对高优请求放行，但
**拒绝判定依赖的队列深度/违约率阈值过高**，导致 Phase 4 时 bronze 长文档也没被拒绝，
反而占满 GPU 拖垮了 gold。

#### 预期效果

调低阈值后，Phase 4 的预期行为：

- Bronze 长文档请求在过载时被**主动拒绝**（返回 503），释放 GPU 给 Gold/Silver
- Gold 短对话违约率从 89.9% **显著下降**（理想情况下 < 10%）
- 代价：Bronze 的拒绝率上升（但这是**符合 SLA 分级的预期行为**——牺牲低优先级保高优先级）

#### 验证方法

调低阈值后重跑 B 轮，重点看两个指标：

1. **Phase 4 的 gold 违约率**：应从 ~90% 显著下降
2. **Phase 4 的 bronze 拒绝率**：应明显上升（HTTP 503 数量增加）

> **注意**：`workload.py` 目前把 `was_cancelled` 和 HTTP 错误都记为失败，需要确认
> 准入控制的 503 响应能被正确记录，以便区分「主动拒绝」和「超时失败」。

### 7.5 待办事项

- [ ] 调低准入控制阈值（`max_queue_depth`、`overload_violation_threshold`、窗口大小）
- [ ] 验证 priority 优先级在 Phase 4 是否真正生效（gold 应比 bronze 违约率更低）
- [ ] 补充 workload.py 对 503 拒绝的识别（区分主动拒绝 vs 超时）
- [ ] Cache-Aware 调度：当前因 `remove+prepend` 破坏 PriorityQueue heap 不变量而暂时
      移除，需改为通过 `effective_priority` 融入缓存命中信息
- [ ] 在 `vllm bench serve` 绝对性能基线上验证优化无吞吐回归
```

---

## 8. GPU Profiling（Nsight Systems + Nsight Compute）

> 本节介绍如何用 NVIDIA 官方 Profiler 分析**优化版框架**（轮次 B，`--scheduling-policy priority`）在压测中的 GPU 真实行为。
>
> 与前面已介绍的三层证据的关系：
>
> | 层 | 工具 | 视角 | 回答的问题 |
> |----|------|------|-----------|
> | 客户端延迟 | `workload.py` / `vllm bench serve` | 应用层 | "SLA 违约了吗？TTFT 多少？" |
> | 服务端状态 | Prometheus + Grafana | 调度器层 | "队列多长？KV Cache 用满了吗？抢占几次？" |
> | **GPU 微观** | **nsys / ncu** | **硬件层** | **"GPU 利用率多少？瓶颈在访存还是算力？哪个 kernel 最慢？"** |
>
> 前三层回答「**什么现象**」，GPU Profiling 回答「**为什么**」——例如 §7 里 Phase 4/5「GPU 物理算力耗尽」这个结论，只有靠 nsys/ncu 才能**坐实**（看到 SM 占用率、访存吞吐、kernel 耗时）。

### 8.1 工具定位

| 工具 | 全称 | 采集方式 | 输出 | 本测试用途 |
|------|------|---------|------|-----------|
| **nsys** (Nsight Systems) | 系统级时间线分析 | 采样 + 事件追踪，**低开销**，包住整段进程 | `.nsys-rep` 时间线报告 | 宏观：看 GPU 利用率、kernel 调度、Prefill/Decode 阶段切换、是否 idle |
| **ncu** (Nsight Compute) | 单 kernel 性能分析 | 重放/插桩单个 kernel，**高开销**，逐 kernel 精确计量 | 每个 kernel 的 SM 占用率、访存吞吐、warp stall 原因 | 微观：定位最慢 kernel 的瓶颈（算力 vs 访存 vs 延迟） |

> **先 nsys 后 ncu**：nsys 先找出「哪个 kernel / 哪个阶段最耗时」，再用 ncu 对那个 kernel 做深入分析。不要一上来就用 ncu 扫全部 kernel——ncu 会重放 kernel，开销极大，压测期间跑会严重失真。

### 8.2 安装（一次性）

本测试环境为 **WSL + Ubuntu 26.04**，使用 NVIDIA 官方 apt 仓库安装（不需要装完整 CUDA toolkit，profiler 独立打包）：

```bash
# 1. 添加 NVIDIA 官方 apt 仓库密钥
curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb" \
    -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update

# 2. 安装 Nsight Systems（系统级时间线）和 Nsight Compute（单 kernel 分析）
sudo apt-get install -y cuda-nsight-systems-13-0 cuda-nsight-compute-13-3

# 3. nsys 会自动软链到 /usr/local/bin/nsys；ncu 需要手动软链
sudo ln -sf /opt/nvidia/nsight-compute/2026.2.1/ncu /usr/local/bin/ncu

# 4. 验证
nsys --version   # 预期: NVIDIA Nsight Systems version 2025.x
ncu --version    # 预期: NVIDIA (R) Nsight Compute ... Version 2026.x
```

> **Ubuntu 26.04 注意**：NVIDIA 仓库目前最高只发布到 `ubuntu2404`，在 26.04 上安装 `ubuntu2404` 源的包是可行的（本节命令已验证）。若未来官方出了 `ubuntu2604` 源，改用对应地址即可。
>
> **Blackwell（sm_120）支持**：RTX 5070 Ti 是 Blackwell 架构，务必用 **CUDA 13.x 对应版本**（nsys ≥ 2025.x、ncu ≥ 2025.4），旧版 profiler 无法正确解析 sm_120 的 kernel。

### 8.3 nsys 宏观分析：整段压测全采

> ⚠️ **先读这条（本 WSL 环境重要限制）**：在 WSL2 `/dev/dxg` 透传下，nsys **采不到 GPU 设备侧的 kernel 数据**（`cuda_gpu_kern_sum` 报告为空），只能采到 CPU 侧的 CUDA API 调用。完整说明与验证过程见 §8.6。本节命令是**裸 Linux 环境的标准做法**，在 WSL 下跑能走通流程、生成报告，但 GPU kernel 时间线部分是空的。

#### 8.3.1 原理（关键：profile 谁）

nsys 用 `nsys profile` 包住一个进程，在进程运行期间低开销采样 CPU/GPU 事件，生成 `.nsys-rep` 时间线。

> ⚠️ **必须 profile 服务端进程，不是客户端**：GPU 计算（attention / GEMM / 各种 kernel）都发生在 **服务端 `api_server` 进程**里，而客户端 `workload.py` 是纯 Python `aiohttp` 程序，**自己不做任何 CUDA 计算**。如果 `nsys profile` 包住的是 `workload.py`，生成的报告里 `cuda_gpu_kern_sum` 会是空的（`does not contain CUDA kernel data`）。
>
> 正确做法：**用 nsys 包住服务端启动命令**，配合 `--capture-range` 只采集压测期间，避免把服务启动/模型加载阶段也采进去。

#### 8.3.2 操作步骤（推荐：`--capture-range` 只采压测窗口）

**终端 1：用 nsys 包住服务端启动**（服务端是优化版 B 轮配置，`/metrics` 自动开启）：

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate
mkdir -p results/optimized_qos/nsys

# nsys 包住服务端进程，cudaProfilerApi 模式：等 workload.py 触发了 start 才采
# （这样能精确框定压测窗口，避开模型加载阶段）
nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --cuda-graph-trace=node \
    --output=results/optimized_qos/nsys/optimized_qos \
    --force-overwrite=true \
    --stats=true \
    .venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --max-model-len 8192 \
        --enable-prefix-caching \
        --scheduling-policy priority \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --port 8000
```

> 等待看到 `Application startup complete` 后，再在终端 2 触发压测。

**终端 2：跑 workload.py 压测（同时触发 profiling 开始/停止）**：

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate

# 用一个小脚本：先 cudaProfilerStart，跑压测，再 cudaProfilerStop
.venv/bin/python -c "
import ctypes, subprocess, sys
cuda = ctypes.CDLL('libcuda.so.1')
cuda.cudaProfilerStart()
sys.exit(subprocess.call([
    '.venv/bin/python', 'benchmarks/e2e_cases/workload.py',
    '--model', 'Qwen/Qwen2.5-1.5B-Instruct',
    '--host', '127.0.0.1', '--port', '8000',
    '--mode', 'single',
    '--duration', '300',
    '--seed', '42',
    '--output-dir', 'results/optimized_qos/',
]))
"
```

> **原理**：`--capture-range=cudaProfilerApi` 让 nsys 处于"待命"状态，直到进程里调用 `cudaProfilerStart()` 才开始采集。这样采集窗口精确等于压测窗口（`--duration 300`），排除了服务启动、模型加载这些无关阶段。

**备选方案（更简单，但会采进模型加载阶段）**：

如果不想用 `cudaProfilerApi`，可以直接让 nsys 从启动就开始采（去掉 `--capture-range`），但这样 `.nsys-rep` 会包含数十秒的模型加载阶段，时间线里有一段是纯加载、无推理，需要手动缩放时间轴到压测段。数据量也会更大。

#### 8.3.3 参数说明

| 参数 | 含义 |
|------|------|
| `--trace=cuda,nvtx,osrt` | 采集 CUDA kernel 调用、NVTX 标记、OS 运行时线程。**不要**加 `cudnn,cublas`（vLLM 主要用自定义 kernel，加了徒增开销） |
| `--capture-range=cudaProfilerApi` | 等进程调用 `cudaProfilerStart()` 才开始采集，精确框定压测窗口 |
| `--cuda-graph-trace=node` | 采集 CUDA Graph 节点（vLLM 大量使用 CUDA Graph 加速 Decode，否则可能采不到 graph 内的 kernel） |
| `--output=<路径>` | 报告文件路径（会生成 `<路径>.nsys-rep` 和 `<路径>.sqlite`） |
| `--force-overwrite=true` | 覆盖同名旧报告 |
| `--stats=true` | 结束时在终端打印 GPU 利用率 / kernel 耗时汇总表 |
| `--duration 300` | 压测时长，与 §3 一致 |

> **可选：限制采样范围**。300s 全采的 `.nsys-rep` 文件可能数百 MB 到数 GB。如果只想看某个 Phase（如 Phase 4 长文档暴增），可以把 `--duration` 改小（如 `--duration 60` 只看前 60s）。但初次建议全采，先建立整体认知。

#### 8.3.4 查看报告

```bash
# 方式一：在 WSL 里用 nsys-ui 打开图形界面（需要 X/WSLg）
nsys-ui results/optimized_qos/nsys/optimized_qos.nsys-rep

# 方式二：命令行导出摘要（无图形界面时）
nsys stats results/optimized_qos/nsys/optimized_qos.nsys-rep
```

> **WSL 图形界面**：WSLg 自带 GUI 支持，`nsys-ui` 可直接弹出窗口。如果无法显示，用 `nsys stats` 导出文本摘要，或把 `.nsys-rep` 拷到 Windows 侧用 Nsight Systems 桌面版打开。

#### 8.3.5 时间线里看什么（对照 5 个 Phase）

| 观察点 | 怎么找 | 预期（优化版） | 对应结论 |
|--------|--------|---------------|---------|
| **GPU 利用率** | 时间线顶部 GPU 活动占比 | Phase 1-3 高（>70%），Phase 4/5 接近 100% 饱和 | 坐实 §7「Phase 4/5 算力耗尽」 |
| **GPU idle 气泡** | 时间线里 GPU 空白的间隙 | idle 间隙对应调度切换/排队，优化版 Phase 3 应比基线少 | 调度优化的 GPU 证据 |
| **Prefill vs Decode 阶段** | NVTX 标记（若 vLLM 有打标）或 kernel 名区分 | Phase 4 长文档 Prefill 段明显变长 | 长文档 Prefill 瓶颈 |
| **kernel 调用密度** | Decode 阶段大量小 kernel 高频调用 | 优化版 Decode 不应受影响 | 调度优化不应拖慢 Decode |
| **内存拷贝/同步** | `cudaMemcpy` / `cudaDeviceSynchronize` 事件 | 不应出现大量同步等待 | 是否有不必要的 host-device 同步 |

> **为什么优化版 Phase 4 仍违约**：在 nsys 时间线上，你会看到 Phase 4 开始后 GPU 利用率立刻顶到 100% 并持续到结束——这直观印证 §7.3 的根因：**不是调度排序问题，而是物理算力打满**。调度优化（QoS/MLFQ 等）只能决定「谁先上」，不能凭空增加算力。

### 8.4 ncu 微观分析：单 kernel 定位瓶颈

> ⚠️ **环境提示**：本测试环境是 WSL + Blackwell，ncu 因 `ERR_NVGPUCTRPERM` **无法使用**（详见 §8.6 已实测确认）。本节命令在**裸 Linux** 环境下为标准做法，先阅读 §8.6 了解如何在 WSL 下解决后，再回来执行。

#### 8.4.1 原理与适用场景

ncu 对**单个 kernel launch** 做精确计量（SM 占用率、访存吞吐、warp stall 原因、寄存器/共享内存使用等），它需要**重放** kernel 多次，开销巨大。因此：

- **不能**在压测期间对整个服务跑 ncu（会严重失真甚至 OOM）
- **正确用法**：先让 nsys 找出最耗时的 kernel（如 attention / GEMM / 自定义 kernel），再用 ncu **单独测那个 kernel**

#### 8.4.2 找出目标 kernel

先用 nsys 报告找出热点 kernel：

```bash
nsys stats --report cuda_gpu_kern_sum results/optimized_qos/nsys/optimized_qos.nsys-rep
```

输出会按 kernel 总耗时排序，形如：

```
Time (%)  Total Time (ns)  Instances  Avg (ns)  Kernel Name
---------------------------------------------------------------
 42.3     123456789000     5000       24691357   void marlin_moe(...)
 18.7     54567890000      3000       18189296   void paged_attention_v1(...)
  ...
```

记下占比最高的 kernel 名字（如 `paged_attention_v1` / `marlin_*` / 自定义调度相关 kernel）。

#### 8.4.3 用 ncu 分析单个 kernel

由于 vLLM 服务是常驻进程，ncu 有两种接法：

**方式 A：`--launch-skip` + `--launch-count`（推荐，服务不重启）**

```bash
# 只 profile 第 100 次之后、共 5 次的 paged_attention_v1 kernel
ncu \
    --kernel-name regex:paged_attention \
    --launch-skip 100 \
    --launch-count 5 \
    --set full \
    --target-processes all \
    --export results/optimized_qos/nsys/attention_kernel \
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --max-model-len 8192 \
        --enable-prefix-caching \
        --scheduling-policy priority \
        --gpu-memory-utilization 0.85 \
        --enforce-eager \
        --port 8000
```

> ncu 会**接管服务进程的启动**（`ncu ... python -m vllm...`）。启动后在另一个终端发少量请求触发目标 kernel，ncu 采集完指定的 launch 次数后自动结束。

**方式 B：attach 到已运行的服务**

```bash
# 先正常启动服务，再找到其 PID，用 ncu attach
ncu --kernel-name regex:paged_attention \
    --launch-count 5 \
    --set full \
    --target-processes <vllm_pid>
```

> 方式 A 更干净（避免 attach 竞态），推荐先用 A。若服务启动参数复杂、想复用已就绪的服务，再用 B。

#### 8.4.4 参数说明

| 参数 | 含义 |
|------|------|
| `--kernel-name regex:paged_attention` | 用正则匹配目标 kernel（`regex:` 前缀） |
| `--launch-skip 100` | 跳过前 100 次 launch（等预热稳定） |
| `--launch-count 5` | 采集 5 次（ncu 会多次重放取统计，5 次足够） |
| `--set full` | 采集完整指标集（SM 占用/访存/warp stall 等）。可用 `basic` 加快速度 |
| `--export <路径>` | 导出 `.ncu-rep` 报告文件 |
| `--target-processes all` | profile 所有子进程（vLLM 会 fork 多个 worker） |

#### 8.4.5 查看与解读

```bash
# 打开图形报告（WSLg）
ncu-ui results/optimized_qos/nsys/attention_kernel.ncu-rep

# 或命令行直接看关键指标（不指定 kernel 名时打印默认 section）
ncu --import results/optimized_qos/nsys/attention_kernel.ncu-rep
```

重点看以下 section，判断瓶颈类型：

| Section | 指标 | 瓶颈判断 |
|---------|------|---------|
| **GPU Speed Of Light** | Compute (SM) Throughput / Memory Throughput | SM% 高→算力瓶颈；Memory% 高→访存瓶颈 |
| **Occupancy** | Achieved / Theoretical Occupancy | 低→warp 没填满 SM，可能是寄存器/共享内存超限 |
| **Warp State Statistics** | Stall Long Scoreboard / Stall Wait | 看 warp 主要卡在什么（访存延迟 vs 同步 vs 算力） |
| **Memory Workload Analysis** | 各内存吞吐、cache 命中率 | 访存瓶颈时定位是全局内存还是 L2 |

> **对本测试的意义**：如果 ncu 显示 `paged_attention` 的 Memory Throughput 接近 100%（访存瓶颈），说明 1.5B 小模型在 5070 Ti 上受限于 KV Cache 访存，而非算力——这进一步解释 §7 里 Phase 4/5 为何调度优化帮不上忙（瓶颈在硬件访存带宽，不在调度顺序）。

### 8.5 与 Prometheus/Grafana 的协同

nsys/ncu 和 Prometheus/Grafana 是**互补**的，建议同时采：

| 维度 | Prometheus/Grafana | nsys/ncu |
|------|-------------------|----------|
| 时间分辨率 | 5s 一次（scrape 间隔） | 纳秒级（kernel 级） |
| 观察对象 | 调度器内部状态（队列/KV Cache/抢占） | GPU 硬件活动（kernel/SM/访存） |
| 回答 | "调度器在干什么" | "GPU 在干什么" |

**联合分析套路**（以 Phase 4 为例）：

1. Grafana 看到 `num_requests_waiting` 飙升、`kv_cache_usage_perc` 打满 → 确认过载发生在调度层
2. nsys 看到 GPU 利用率顶到 100%、Prefill 段变长 → 确认是真实算力/访存耗尽
3. ncu 看到热点 kernel Memory Throughput ≈ 100% → 定位具体瓶颈在访存

三步闭环，从「现象」到「根因」完整覆盖。

### 8.6 注意事项与常见坑

| 现象 | 原因 | 解决 |
|------|------|------|
| `nsys profile` 报权限错误 | WSL 下 perf 计数器受限 | `sudo nsys profile ...`，或先确认 driver 已启用 persistence mode（`nvidia-smi` 里应为 Enabled） |
| `.nsys-rep` 文件巨大 | 300s 全采 + 高频 kernel | 用 `--trace=cuda` 去掉 osrt/nvtx 减量；或缩短 `--duration` |
| `cuda_gpu_kern_sum` 报告为空 | WSL dxg 透传不暴露 GPU 设备侧事件 | **已实测确认**：WSL 下 nsys 采不到 GPU kernel，见下条 |
| ncu 报 `ERR_NVGPUCTRPERM` | WSL dxg 透传无性能计数器接口 | **已实测确认无解**，见下条 |

> ⚠️ **WSL2 `/dev/dxg` 透传下，nsys 和 ncu 都无法做 GPU 设备侧 profiling（本环境已实测确认）**：
>
> 本测试环境是 **WSL2 + RTX 5070 Ti（Blackwell sm_120）**，经排查确认为 WSL2 的 `/dev/dxg` GPU 透传模式：
>
> ```
> $ ls /dev/dxg          # → 存在（dxg 透传的标志）
> $ ls /dev/nvidia*      # → 不存在（没有原生 CUDA 设备节点）
> $ which nvidia-smi     # → /usr/lib/wsl/lib/nvidia-smi（WSL 转译版，非真驱动）
> ```
>
> 这种模式下，Linux 侧 CUDA 调用经 dxg 层**转发**给 Windows 驱动，**GPU 设备侧的 profiling 事件（CUPTI 设备事件 / NVPC 计数器）在 dxg 转译层被挡掉**。实测结论：
>
> | 工具 | CPU 侧 CUDA API 追踪 | GPU 侧 kernel 追踪 | 报错/现象 |
> |------|---------------------|-------------------|----------|
> | **nsys** | ✅ 能采到（如 `cudaLaunchKernel` 调用次数、耗时） | ❌ 采不到 | `cuda_gpu_kern_sum` 报告为空（`does not contain CUDA kernel data`） |
> | **ncu** | — | ❌ 无法用 | `ERR_NVGPUCTRPERM` |
>
> **关键实测证据**（均在本环境验证）：
> 1. `nsys profile --trace=cuda` 包住服务端跑压测 → `cuda_api_sum` 报告有数据（`cudaLaunchKernel` 190 万次），但 `cuda_gpu_kern_sum` / `cuda_gpu_mem_time_sum` 报告**全为空**；
> 2. 用最简单的 `torch` matmul 脚本（单 kernel）测 nsys → 同样采不到 GPU kernel，确认不是 vLLM 特有，而是环境限制；
> 3. `sudo ncu ...` → 报 `ERR_NVGPUCTRPERM`；
> 4. 修改 Windows 注册表 `RestrictProfilingToAdminUsers=0` + `wsl --shutdown` → ncu 仍报同样的错；
> 5. WSL 的 `libcuda.so.1`（`/usr/lib/wsl/lib/`）缺失 `cudaProfilerStart` 符号，`--capture-range=cudaProfilerApi` 也无法用。
>
> **结论**：在 WSL2 `/dev/dxg` 透传模式下，**nsys 和 ncu 都无法做 GPU 设备侧 profiling，且 WSL 内无任何配置可解**。这两个工具都依赖 CUPTI 的设备侧事件（NVPC），而 dxg 透传层没有暴露这个接口。
>
> 要真正用上 GPU profiling，必须换环境：
>
> - **裸 Linux**（物理机或 VFIO GPU 直通虚拟机）：nsys / ncu 均标准可用，本文档 §8.3 / §8.4 命令可直接复用；
> - **云 GPU 裸机实例**（AWS/GCP 等）：通常可用。
>
> **在本 WSL 环境下能拿到的 profiling 信息有限**：
> - nsys 能采到 **CPU 侧 CUDA API 调用**（kernel 的 launch 次数、API 耗时），可粗略推断 GPU 忙闲，但**看不到 kernel 在 GPU 上的实际执行时间、SM 占用率、访存吞吐**；
> - 若只是了解工具用法、熟悉命令，nsys 仍可跑通流程（会生成报告，只是 GPU kernel 部分为空）。

> **补充：`ERR_NVGPUCTRPERM` 在不同环境下的含义不同**：
> - **裸 Linux**：通常是权限问题，改 `NVreg_RestrictProfilingToAdminUsers=0` 或用 root 可解；
> - **WSL2 dxg 透传**：架构性无解（本环境），改注册表也无效；
> - `ERR_PROFILING_NOT_SUPPORTED`：驱动/硬件不支持 profiling（云 GPU、虚拟化 GPU），无解，只能换硬件。

### 8.7 完整流程速查（Profiling 版）

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate

# ===== 终端 1：用 nsys 包住服务端启动（优化版 B 轮，cudaProfilerApi 模式） =====
mkdir -p results/optimized_qos/nsys
nsys profile \
    --trace=cuda,nvtx,osrt \
    --capture-range=cudaProfilerApi \
    --cuda-graph-trace=node \
    --output=results/optimized_qos/nsys/optimized_qos \
    --force-overwrite=true --stats=true \
    .venv/bin/python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
        --enable-prefix-caching --scheduling-policy priority \
        --gpu-memory-utilization 0.85 --enforce-eager --port 8000
# 等看到 "Application startup complete" 再执行终端 2

# ===== 终端 2：触发 profiling + 跑压测 =====
.venv/bin/python -c "
import ctypes, subprocess, sys
cuda = ctypes.CDLL('libcuda.so.1')
cuda.cudaProfilerStart()
sys.exit(subprocess.call([
    '.venv/bin/python', 'benchmarks/e2e_cases/workload.py',
    '--model', 'Qwen/Qwen2.5-1.5B-Instruct',
    '--host', '127.0.0.1', '--port', '8000',
    '--mode', 'single', '--duration', '300', '--seed', '42',
    '--output-dir', 'results/optimized_qos/',
]))
"
# 压测结束后，终端 1 的 nsys 会自动停止采集并生成报告

# ===== 查看 nsys 报告 =====
nsys-ui results/optimized_qos/nsys/optimized_qos.nsys-rep   # 图形
nsys stats --report cuda_gpu_kern_sum results/optimized_qos/nsys/optimized_qos.nsys-rep  # 找热点 kernel

# ===== ncu 深入热点 kernel（微观，需另起服务） =====
ncu \
    --kernel-name regex:<热点kernel名> \
    --launch-skip 100 --launch-count 5 \
    --set full \
    --export results/optimized_qos/nsys/hot_kernel \
    python -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-1.5B-Instruct --max-model-len 8192 \
        --enable-prefix-caching --scheduling-policy priority \
        --gpu-memory-utilization 0.85 --enforce-eager --port 8000

# ===== 查看 ncu 报告 =====
ncu-ui results/optimized_qos/nsys/hot_kernel.ncu-rep
```
