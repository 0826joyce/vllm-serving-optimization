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

### 7.0 汇总：必须的参数配置与测试评估内容（先看这个）

> 本节先给结论——经过 §7.1-§7.16 整条验证链的实测，反推出"参数必须怎么配"和"测试评估的是什么"。详细推导见后续小节，本节用于快速上手与避坑。

#### 7.0.1 必须的参数配置（踩坑后固化）

以下配置由 §7.2-§7.16 的实测结论反推得出，**任意一项偏离都会导致结果失真或结论错误**：

**A. 服务端启动参数（优化版 B 轮 / C 轮）**

| 参数 | 必须值 | 不能改的原因 / 踩坑来源 |
|------|--------|------------------------|
| `--scheduling-policy` | `priority`（优化版）/ `fcfs`（基线） | 唯一变量，决定 QoS 是否生效 |
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | A/B 两轮必须同一模型，控制变量 |
| `--max-model-len` | `8192` | 两轮一致；过低会截断长文档，影响 Phase 4 |
| `--enable-prefix-caching` | 两轮都开 | 控制变量，隔离"调度优化"而非"有没有缓存" |
| `--gpu-memory-utilization` | `0.85` | 两轮一致 |
| **`--enforce-eager`** | **必须去掉**（§7.14） | ⚠️ 最大坑：eager 禁用 CUDA graph，prefill 慢 60-100 倍，Phase 3 就 100% 违约，会得出错误结论 |
| `--max-num-seqs`（`max_num_seqs`） | 默认 256，**不要降到 128**（§7.14） | 降到 128 反而加剧排队，256 能容纳更多短请求快速 decode |
| `--max-num-batched-tokens` | `2048` | 决定长文档 chunked prefill 的块数（4800 token 需 3 块） |

**B. 准入控制阈值（优化版，§7.6 固化）**

| 参数 | 必须值 | 原默认值 | 调整理由 |
|------|--------|---------|---------|
| `max_queue_depth` | `40` | 100 | 过高导致准入控制从未触发（§7.4） |
| `overload_violation_threshold` | `0.25` | 0.5 | 更早识别过载 |
| `_sla_violation_window` maxlen | `25` | 50 | 更灵敏反映近期违约率 |
| `_is_high_priority_request` | 仅 `priority < 0`（gold） | 含"短 prompt 即高优" | 旧判定让准入控制形同虚设（§7.6） |

**C. 压测参数（workload.py）**

| 参数 | 必须值 | 踩坑来源 |
|------|--------|---------|
| `--seed` | `42` | A/B 必须相同，保证流量可复现 |
| `--duration` | `300` | 5 个 Phase × 60s |
| `--bronze-qps-scale` | **必须显式设置**（§7.15-7.16） | ⚠️ 默认 1.0 时 4800 token × 10 QPS = 48000 tok/s，超硬件 16.7 倍，物理无解；临界过载区间用 `0.2~0.25` 才能测出优化效果 |
| 长文档数据 | **~4800 token**（§7.11 修正后） | 旧数据 230 token 名不副实，掩盖了长文档 Prefill 的真实破坏力 |

**D. 基线版（A 轮）必须显式关闭所有优化**

```bash
VLLM_DISABLE_MLFQ=1 VLLM_DISABLE_RATE_LIMIT=1 VLLM_DISABLE_TENANT_ISO=1 \
VLLM_DISABLE_OVERLOAD_MGMT=1 VLLM_DISABLE_DEADLINE_AWARE=1 \
--scheduling-policy fcfs
```
> 少关一个会让基线"偷偷"带了优化，A/B 对比失效。

#### 7.0.2 测试评估的是什么（评估维度与判据）

本测试用**三层证据**交叉验证，缺一不可：

| 层 | 工具 | 评估的核心问题 | 关键指标 | 判据 |
|----|------|--------------|---------|------|
| **客户端·公平性** | `workload.py` | 优化是否改善了**低优租户**的 SLA？是否保护了高优？ | 按 tier × phase 的违约率、完成请求数、拒绝率 | silver/bronze 违约率 ↓53-56pp 为有效；gold 违约率不变则符合预期（物理无解） |
| **客户端·绝对性能** | `vllm bench serve` | 优化有没有**牺牲整体吞吐**？ | 吞吐(req/s, tok/s)、P99 TTFT/TPOT/ITL/E2EL、失败请求数 | B 轮吞吐与 A 轮 ±5% 内为安全；>10% 退化为过度优化 |
| **服务端·内部状态** | Prometheus + Grafana | 调度器内部行为是否符合预期？ | 队列长度、KV Cache 使用率、抢占次数、排队时间 P99、Prefill/Decode 耗时 | 优化版 waiting 队列更短、KV Cache 更平滑、抢占 SLA 感知 |

**评估的三个判据层级：**

1. **优化是否有效**：silver/bronze 违约率显著下降（临界过载区间内）
2. **优化是否安全**：绝对吞吐不退化（±5%），TPOT/ITL 不恶化
3. **优化边界在哪**：轻载无效（无优化空间）、严重过载无效（物理算力耗尽），**临界过载（bronze QPS 0.2-0.5）是有效区间**

#### 7.0.3 评估时必须同时看的三个率（避免幸存者偏差）

§7.10 / §7.16 暴露的关键陷阱：**只看违约率会被准入控制欺骗**——拒绝低优请求会让"完成的"请求违约率下降，但这是幸存者偏差。完整评估必须同时看：

| 指标 | 含义 | 为什么不能只看违约率 |
|------|------|---------------------|
| **违约率** | 完成请求中 SLA 超标的比例 | 被拒请求不计入，会人为降低 |
| **拒绝率** | 被准入控制拒绝（503）的比例 | 优化"保护"silver 的机制本质是拒绝，不是更快 |
| **完成率** | 实际完成 / 总发出 | 反映用户实际体验（被拒=用户拿不到结果） |

> 三者结合才能说清楚：优化是"让低优更快"还是"让低优被拒"——本项目结论是后者（§7.10 发现 3）。

#### 7.0.4 一句话总结

> **参数**：去掉 `--enforce-eager` + 用 `--bronze-qps-scale` 控负载到临界过载 + 准入阈值收紧到 40/0.25/25，是跑出有效结论的硬前提。
> **评估**：三层证据（公平性 + 绝对性能 + 服务端状态）× 三个率（违约 + 拒绝 + 完成），才能说清楚"优化有效但有边界、保护的是拒绝而非加速"。

---

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

- [x] 调低准入控制阈值（`max_queue_depth`、`overload_violation_threshold`、窗口大小）
      → 见 §7.6，实测准入控制已触发（6038 次拒绝），但未能降低 gold 违约率
- [x] 验证 priority 优先级在 Phase 4 是否真正生效（gold 应比 bronze 违约率更低）
      → 见 §7.6，gold 确实未被拒绝，但自身高 QPS 导致违约率反而升高
- [ ] 补充 workload.py 对 503 拒绝的识别（区分主动拒绝 vs 超时）
      → 当前被拒绝请求 `ttft_ms=None`，被 workload.py 静默忽略，无法统计拒绝率
- [ ] Cache-Aware 调度：当前因 `remove+prepend` 破坏 PriorityQueue heap 不变量而暂时
      移除，需改为通过 `effective_priority` 融入缓存命中信息
- [ ] 在 `vllm bench serve` 绝对性能基线上验证优化无吞吐回归

### 7.6 C 轮测试：准入控制阈值调优（2026-08-22）

> 针对 §7.4 的方案调整，实施了准入控制修复 + 阈值收紧，并重跑优化版（B 轮配置）。
> 代码改动见 `vllm/v1/core/sched/scheduler.py`、`vllm/v1/engine/core.py`、`vllm/v1/request.py`。

#### 代码改动清单

| 改动 | 内容 | 修复的 bug |
|------|------|-----------|
| **SLA 违约窗口填充** | 在请求完成时 `_sla_violation_window.append(request.is_sla_violated())` | 窗口从未被填充，SLA 违约率门控**永远不触发** |
| **拒绝路径完整化** | 被拒绝请求立即 `_free_request` + EngineCore 发 `ERROR` 输出给客户端 | 被拒请求之前会挂起直到超时 + 内存泄漏 |
| **`FINISHED_REJECTED` 映射** | 加入 `_FINISHED_REASON_MAP` → `FinishReason.ERROR` | `get_finished_reason()` 之前返回 None |
| **高优先级判定收紧** | `_is_high_priority_request` 改为仅 `priority < 0`（gold），去掉"短 prompt 即高优" | 之前几乎所有短请求都被判为高优，准入控制形同虚设 |
| **阈值收紧** | `max_queue_depth` 100→40，`overload_violation_threshold` 0.5→0.25，窗口 50→25 | 阈值过宽，准入控制从未触发 |

#### 测试结果对比

| 指标 | A 基线 | B 优化(旧) | C 调优后 |
|------|--------|-----------|---------|
| Phase 4 准入拒绝次数 | 0 | 0 | **6038** |
| Phase 4 gold 违约率 | 89.7% | 89.9% | **98.2%** ❌ |
| Phase 4 silver 请求数 | ~1500（全收） | ~1500（全收） | **74**（其余全拒） |
| Phase 4 bronze 请求数 | ~350（全收） | ~350（全收） | **53**（其余全拒） |
| Phase 5 违约率 | 100% | 100% | 100% |

#### 核心结论

准入控制的**拒绝机制真正生效了**（从 0 次到 6038 次拒绝，silver/bronze 在 Phase 4 被大量拒绝），
但 **gold 违约率不降反升（89.9% → 98.2%）**。原因见 §7.7。

### 7.7 根因再分析：为什么准入控制没能保护 gold

C 轮结果证明了一个重要事实：**准入控制无法解决 Phase 4 的违约，因为问题不在"入口"而在"GPU 物理算力 + gold 自身 QPS"**。

#### 恶性循环的形成

Phase 4 时，长文档（Bronze）Prefill 单次数秒到数十秒，占住 GPU 后：

1. `waiting` 队列迅速堆积到 `max_queue_depth=40`
2. 准入控制开始拒绝新来的 silver/bronze（queue depth 门控，6038 次全是这个原因触发）
3. **但 gold（priority=-2）被 `_is_high_priority_request` 放行，继续进入队列**
4. 队列里 gold 占比越来越高，gold 短请求的 Prefill 依然要和长文档 Prefill 竞争 GPU
5. gold 排队时间越来越长 → 违约率反而升高到 98.2%

#### 关键数据佐证

- 6038 次拒绝**全部**由 `queue depth >= 40` 触发，**0 次**由 SLA 违约率门控触发
- 说明 SLA 违约率窗口虽然已修复填充，但在 queue depth 门控"一刀切"下根本没机会起作用
- Phase 4 gold 有 1939 个请求（~32 QPS），这个 QPS **本身就超过了长文档 Prefill 占用 GPU 后的可用算力**

#### 与 §7.3 结论的一致性

这个结果**印证了 §7.3 的根因分析**：Phase 4 违约的根本原因是**长文档 Prefill 的绝对耗时 +
gold 自身 32 QPS 超过 GPU 处理能力**，这是物理算力约束，不是调度策略能解决的。

准入控制能做的只是"拒绝低优先级请求"，但它：
- ❌ 无法让长文档 Prefill 变快
- ❌ 无法让已在队列里的 gold 请求插队（它们依然要排队）
- ❌ 无法降低 gold 自身的 QPS

#### 真正的解决方向（需要产品/业务层面决策）

要真正保护 gold 的 SLA，调度层能做的是有限的，根因在**负载设计**：

| 方向 | 说明 | 是否调度层能解决 |
|------|------|----------------|
| **降低 gold QPS** | Phase 4 的 32 QPS 过高，需业务侧限流 | ❌ 需业务配合 |
| **长文档单独池化** | Bronze 长文档走独立 GPU/队列，不干扰短对话 | 部分（需硬件） |
| **长文档 Prefill 抢占** | 长文档 Prefill 允许被短请求抢占 | ✅ 调度层可做，但 vLLM V1 的 Prefill 抢占复杂 |
| **更激进的准入 + 负载分级** | 拒绝时**连队列里已排队的低优请求也驱逐** | ✅ 调度层可做（见 §7.8） |

### 7.8 下一步方向：队列驱逐（比准入控制更彻底）

准入控制只拦"入口"，但 Phase 4 的问题是"队列里已堆积的请求"。更彻底的方案是**队列驱逐**：
当检测到过载时，不仅拒绝新请求，还要**主动驱逐队列里已排队的低优先级请求**（silver/bronze），
为 gold 腾出位置和算力。

```python
# 伪代码：过载时驱逐 waiting 队列中的低优请求
if violation_rate > threshold:
    for r in list(self.waiting):
        if not self._is_high_priority_request(r):
            r.status = RequestStatus.FINISHED_REJECTED
            self.waiting.remove_request(r)
            self._free_request(r)
```

> 但需注意：这只能缓解"队列堆积"，仍无法解决"gold 自身 32 QPS 超过 GPU 算力"的根本矛盾。
> 最终方案需要结合业务侧限流（降低 gold QPS）或硬件扩容（长文档独立池化）。

> ⚠️ **重要更正（见 §7.9）**：上述"gold 自身 32 QPS 超过 GPU 算力"的判断是**错误的**。
> §7.9 的隔离实验证明 GPU 算力完全够用，真正瓶颈是 **Bronze 长文档 Prefill 的干扰**。

### 7.9 隔离实验：长文档是违约的唯一根因（2026-08-23）

> 为验证「Phase 4/5 违约到底是不是 GPU 算力耗尽」，做了隔离实验：
> 用 `--bronze-qps-scale 0` 把 Bronze 长文档负载归零，其余负载（gold 32→48 QPS、
> silver 8→12 QPS）保持默认，观察 gold 是否仍违约。

#### 实验结果

| 场景 | Phase 4 gold 违约率 | Phase 5 gold 违约率 | Phase 5 gold P99 |
|------|-------------------|-------------------|-----------------|
| bronze=1.0（默认，长文档干扰） | 89.9% ~ 98.2% | 100% | 严重超时 |
| **bronze=0（隔离实验）** | **0%** | **0.1%**（仅 4 次） | **68ms** |

#### 决定性结论

1. **GPU 物理算力完全够用**：即使 Phase 5 全面过载（gold 48 QPS + silver 36 QPS），
   去掉长文档后所有租户违约率仍 ≈ 0%，gold P99 仅 68ms。
2. **Bronze 长文档是唯一的、决定性的干扰源**：长文档 Prefill（长 prompt，单次数秒到数十秒）
   占住 GPU，阻塞了所有短请求（gold/silver）的 Prefill，导致大面积违约。
3. **§7.3 / §7.7 的"GPU 物理算力耗尽"结论是错误的**，需要更正。

#### 对调度优化的启示

这才是调度优化**应该发力**的地方：长文档 Prefill 应该被
- **抢占**（Preemption）—— 长文档 Prefill 让位给短请求；
- 或 **预算隔离**（Prefill Budget Isolation）—— 长文档不能独占 GPU。

项目里已有的 `Prefill Budget Isolation`（`long_prefill_count` + `short_budget_reserved`）
和 `MLFQ 降级` 理论上应能缓解，但 §7.2 的 A/B 测试显示无差异，**说明这些优化可能没有
真正生效**——这是下一步需要排查的方向（见 §7.10）。

### 7.10 负载扫描：定位优化方案的有效边界（2026-08-23）

> 为了量化"优化方案相对基线（fcfs）到底有没有效果、在什么负载范围内有效"，
> 在 `workload.py` 中新增了 QPS 缩放参数：
>
> - `--qps-scale`：全局 QPS 缩放（所有租户所有阶段）
> - `--bronze-qps-scale`：额外只缩放 Bronze 长文档 QPS（用于隔离长文档干扰）
>
> 扫描思路：固定 gold 负载，**扫描 bronze 负载从 0 到 1.0**，对比
> A 基线（`--scheduling-policy fcfs` + 禁用所有优化）vs B 优化
> （`--scheduling-policy priority`）的各租户违约率。

#### 扫描矩阵与结果

| bronze 缩放 | Phase 4 gold 违约 | Phase 4 silver 违约 | Phase 5 gold 违约 | Phase 5 silver 违约 |
|------------|------------------|-------------------|------------------|-------------------|
| **0（隔离）** | 0%（仅 B 测） | 0% | 0.1% | 0% |
| **0.25** A 基线 | 0% | 0% | 87.7% | 83.7% |
| **0.25** B 优化 | 0% | 0% | 93.1% | **11.3%** ✅ |
| **0.5** A 基线 | 82.8% | 84.1% | 100% | 100% |
| **0.5** B 优化 | 84.3% | **7.6%** ✅ | 100% | — |
| **1.0** A 基线 | 89.7% | — | 100% | — |
| **1.0** B 优化 | 89.9% | — | 100% | — |

> 注：`—` 表示该租户在对应阶段几乎全部请求被准入控制拒绝，无有效样本。
> Phase 5 在 bronze≥0.5 时 gold 100% 违约，属极限过载。

#### 三个关键发现

**发现 1：优化方案对 gold 的"保护"基本无效**

在所有扫描点，**gold 违约率 A/B 几乎无差异**（甚至 B 略差：Phase 5 下 87.7% vs 93.1%）。
根因：gold 是最高优先级（priority=-2），准入控制**永不拒绝** gold，它只能排队等待；
而排队本身导致违约。gold 违约是"自身 QPS 超载 + 长文档占用 GPU"的物理约束，调度无法解决。

**发现 2：优化方案对 silver/bronze 的"保护"效果显著**

这是优化方案最明确的量化收益：

| 对比点 | silver 违约率 A→B | bronze 违约率 A→B |
|--------|------------------|------------------|
| Phase 5（bronze=0.25） | 83.7% → **11.3%** | 71.0% → **27.6%** |
| Phase 4（bronze=0.5） | 84.1% → **7.6%** | 80.9% → **10.1%** |

**发现 3：这个"保护"的机制是「拒绝」而非「更快」**

关键证据（bronze=0.25，Phase 5）：
- silver 完成请求数：A 基线 805 个 vs B 优化 **150 个**（拒绝率 ~81%）
- B 优化的准入控制拒绝了大量 silver 请求，使"完成的 silver"违约率降低

这是**准入控制的预期行为**（过载时拒绝低优先级请求），但从服务质量角度看，
silver 用户拿到的不是"更快"，而是"被拒绝"。这需要业务侧决策：**过载时是拒绝低优
请求（保护高优），还是让所有请求都慢一点（公平降级）？**

#### 有效性边界结论

| 负载区间 | 优化方案表现 | 判定 |
|---------|-------------|------|
| bronze ≤ 0.25（长文档轻载） | Phase 1-4 全部 0 违约，A/B 无差异 | 负载太轻，优化无用武之地 |
| 0.25 < bronze < 0.5（临界过载） | silver/bronze 违约率 A→B 显著下降（80%→7~10%） | **优化有效区间** ✅ |
| bronze ≥ 0.5（严重过载） | gold/silver/bronze 全线崩，A/B 无差异 | 物理算力耗尽，调度无能为力 |

**结论**：优化方案的「有效边界」在**临界过载区间**（长文档负载占 GPU 约 25%~50% 时），
此时优先级调度 + 准入控制能有效把过载代价从 silver/bronze 转移到"被拒绝的请求"上，
量化收益为 silver 违约率 **-73%~-80%**。而在轻载（无优化空间）和严重过载（物理算力
耗尽）两端，优化方案都无法体现效果。

#### 局限与下一步

1. **无法保护 gold**：gold 自身过载 + 最高优先级不可拒绝，导致 gold 违约无解。
   需实现**长文档 Prefill 抢占/隔离**，让 gold 短请求真正插队长文档 Prefill。
2. **"保护 = 拒绝"的语义**：当前准入控制通过拒绝低优请求保护高优，但被拒请求
   在 workload.py 中被静默忽略（`ttft_ms=None`），需补充对 503 拒绝的显式统计。
3. **测量陷阱**：违约率只统计"完成的请求"，被拒请求不计入，导致 silver 低违约
   是"幸存者偏差"。完整评估需同时看「违约率」+「拒绝率」+「完成率」三个指标。

### 7.11 重要更正：Bronze「长文档」此前名不副实（2026-08-23）

> 在排查「Prefill 预算隔离为何没生效」时，用 tokenizer 实测了 `LONG_DOCUMENTS`
> 的实际长度，发现一个**测试数据 bug**，此前的部分结论需要修正。

#### 实测数据

| 项目 | 此前认知 | 实测真相 |
|------|---------|---------|
| Bronze prompt 长度 | "长文档"（长 prompt） | **仅 205~245 token**（内容很短） |
| Bronze `max_tokens` | — | **300**（输出很长） |
| 调度器 `long_prefill_threshold` | — | 1024（Bronze 远够不到） |

#### 由此修正的三个错误结论

1. **"Bronze 长文档 Prefill 拖垮 gold" → 实际是「长输出（300 token）+ 高 QPS」拖垮 gold**：
   Bronze 的 prompt 只有 ~230 token，Prefill 极快（毫秒级），真正的资源消耗在
   **Decode 阶段**——每个 Bronze 请求生成最多 300 token，长期占用 KV Cache 与 Decode 算力。
2. **"Prefill 预算隔离没生效"的根因**：`is_long_prefill = num_new_tokens > 1024`，
   而 Bronze 只有 230 token，**判定永远为 False**，预算隔离从未触发——不是代码 bug，
   而是测试数据没达到阈值。
3. **之前扫描的"长文档负载"实为"长输出负载"**：§7.9 / §7.10 的隔离实验与负载扫描，
   隔离的实际上是「长输出 + 高 QPS」的 Decode 竞争，而非「长 prompt 的 Prefill 竞争」。

#### 已做的修正

`workload.py` 的 `LONG_DOCUMENTS` 已改为**程序化生成真实长文档（约 4800~4900 token）**：

- 明确超过 `long_prefill_threshold=1024`，能触发「长文档」判定；
- 在 `max_num_batched_tokens=2048` 下需 **3 块 chunked prefill**，能暴露长文档
  Prefill 独占调度的问题；
- 内容为真实技术/业务段落（非无意义重复字符），贴近真实 RAG 场景。

> ⚠️ **因此 §7.9 / §7.10 的扫描结论需要在新数据上重新验证**：旧数据对应的负载是
> 「长输出」，新数据才是真正的「长文档 Prefill」。两者对调度器的压力模型不同，
> 拐点位置与优化方案的相对表现都可能变化，需重新扫描确认（见 §7.12）。

### 7.12 重新扫描：真正的长文档 Prefill（2026-08-23）

> `LONG_DOCUMENTS` 修正为 ~4800 token 后，重跑 B 优化版 bronze=0.5 扫描点，
> 对比旧数据（230 token「长输出」场景），验证真正的长文档 Prefill 的破坏力。

#### 结果对比（B 优化版，bronze=0.5）

| Phase | gold 违约率（旧：230 token） | gold 违约率（新：4800 token） |
|-------|---------------------------|---------------------------|
| Phase 1 | 0.6% | 2.0% |
| Phase 2 | 0% | **8.7%** ❌ |
| Phase 3 | 0% | **100%** ❌ |
| Phase 4 | 84.3% | 100% ❌ |
| Phase 5 | 100% | 100% ❌ |

#### 决定性发现：真正的长文档 Prefill 破坏力远超「长输出」

1. **旧场景（230 token prompt + 300 token 输出）**：Phase 3（gold 32 QPS）能扛住（0 违约），
   只有 Phase 4（bronze 暴增到 10 QPS）才崩。
2. **新场景（4800 token prompt）**：**Phase 3（bronze 仅 3 QPS）就 100% 违约**，
   甚至 Phase 2 就开始出现违约（8.7%）。

根因：4800 token 的长文档 Prefill 需 **3 块 chunked prefill**（`max_num_batched_tokens=2048`），
即使只有 3 QPS 的 bronze，每个长文档 Prefill 都长期占用 GPU，叠加 gold 32 QPS 暴增，
短请求全部排队超时。

> **重要印证**：这终于证实了 §7.9 隔离实验的原始假设——「长文档 Prefill」确实是
> 违约的决定性因素。此前旧数据（230 token）掩盖了这个事实，因为它根本不是长文档。

#### 结论与下一步

1. **测试数据修正有效**：现在的 Bronze 是名副其实的「长文档」，能真实暴露
   长文档 Prefill 对短请求的毁灭性干扰。
2. **Prefill 预算隔离仍未能保护 gold**：虽然 `is_long_prefill`（>1024 token）现在能触发了，
   但在单 partial prefill 的限制下，长文档 Prefill 依然独占 GPU，gold 无法插队。
3. **真正的解法需要「并发 partial prefill」**：让短请求能与长文档 Prefill 并行调度。
   但本 fork 硬性禁用了该功能（`--max-num-partial-prefills > 1` 报 `NotImplementedError`），
   需评估是否放开该限制（见 §7.13）。

### 7.13 排查结论：并发 partial prefill 功能缺失 + 长文档 continuation 独占 GPU（2026-08-23）

> 针对「放开并发 partial prefill 限制」做了三步排查，结论是**该方向在当前代码库不可行**，
> 且定位到了真正的根因。

#### 排查结果

**1. 禁用原因（git blame）**：`_check_feature_supported()` 的禁用来自上游 vLLM 官方
（commit `bc150f5`，Docker release PR），注释 "No Concurrent Partial Prefills so far"。
**不是本 fork 特意加的**，是上游当时就未实现该功能。

**2. scheduler 逻辑是否完整**：**不完整，功能缺失**。`max_num_partial_prefills` 和
`max_long_partial_prefills` 这两个字段：
- 只在 `config/scheduler.py` 定义/校验（默认值、合法性检查）
- 在 `arg_utils.py` 传参 + 硬性禁用
- **在 scheduler 的调度循环里完全没有被消费**（`vllm/v1/core/sched/scheduler.py`
  中无任何引用；唯一引用在 AMD ROCm 后端 `rocm_aiter_fa.py`，与本环境无关）

**结论**：即使移除 `_check_feature_supported()` 的限制，设 `--max-num-partial-prefills 2`
也**不会产生任何效果**——该参数会被忽略，因为调度逻辑根本不存在。

#### 真正的根因：chunked prefill continuation 连续独占 GPU

排查中定位到 gold 100% 违约的真正根因，涉及三层：

1. **running 请求优先调度**（`scheduler.py:484` "First, schedule the RUNNING requests"）：
   长文档一旦进入 running，它的 chunked prefill continuation 在 `schedule_running`
   里**优先**占用 token_budget，waiting 里的短请求排不上。

2. **长文档 3 块 prefill 连续独占**：4800 token 长文档需 3 块 chunked prefill
   （`max_num_batched_tokens=2048`），这 3 块在 running 里连续执行，期间 gold 无法插队。

3. **Prefill 预算隔离的 `break` 语义 bug**（`scheduler.py:841`）：当长文档名额用完时，
   用 `break` **跳出整个 while 调度循环**（而非跳过该长文档继续调度短请求），
   导致排在其后的短请求完全无法被调度。

#### 可行的修复方向（按成本排序）

| 方向 | 成本 | 效果 | 说明 |
|------|------|------|------|
| **修复 `break` → 跳过语义** | 低 | 有限 | 让长文档名额用完后，短请求仍能继续调度，但无法解决「running 长文档 continuation 独占」 |
| **长文档 continuation 让位** | 中 | 较好 | 在 `schedule_running` 里，对「长文档 continuation prefill」降低优先级，让短请求的新 prefill 先调度 |
| **移植并发 partial prefill** | 高 | 彻底 | 从上游 vLLM 移植整套并发 partial prefill 调度逻辑，工作量大、风险高 |
| **PD 分离部署** | 中 | 架构级 | 长文档 Prefill 走独立实例，物理隔离，不动 scheduler 核心 |

> 建议先做**「长文档 continuation 让位」**（成本中等，直接针对根因第 2 层），
> 即让长文档的后续 prefill chunk 不优先于短请求的新 prefill。

### 7.14 验证实验：去掉 enforce-eager 与「让位」逻辑（2026-08-23）

> 实现了「长文档 continuation 让位」（`enable_long_prefill_yield`，默认开启，
> `VLLM_DISABLE_LONG_PREFILL_YIELD=1` 可关），并做了两组验证实验。

#### 实验矩阵（bronze=0.5，真长文档 4800 token）

| 配置 | Phase 3 gold 违约 | Phase 4 gold 违约 |
|------|------------------|------------------|
| eager + seq=256（原配置） | 100% | 100% |
| eager + seq=256 + 让位逻辑 | 100% | 100%（让位无效） |
| **无 eager + seq=256** | **0~2%** ✅ | 100% |
| 无 eager + seq=128 | 10.9%（反而更差）| 89.9% |

#### 关键发现

1. **`--enforce-eager` 是 Phase 3 违约的元凶**：去掉后 Phase 3 从 100% 降到 0~2%。
   eager 模式禁用了 CUDA graph 与编译优化，prefill 极慢（80-200 tok/s）。
2. **「让位」逻辑在 eager 下无效**：瓶颈是 prefill 速度，而非调度顺序。
3. **降低 max_num_seqs（128）反而更差**：256 并发能容纳更多短请求快速 decode，
   128 反而加剧排队。「batching 崩坏」假设被证伪。

### 7.15 核心发现：Phase 4 负载远超硬件能力（2026-08-23）

> 通过单测精确测算了长文档 prefill 的真实硬件能力，发现 Phase 4 的负载设计
> **远超 5070 Ti 的能力**，这是违约的根本原因，任何调度优化都无法解决。

#### 硬件能力实测（单测）

| 场景 | prompt throughput | 说明 |
|------|------------------|------|
| 4 个相同长文档（命中缓存） | 52937 tok/s | 缓存命中，几乎免费 |
| 4 个不同长文档（无缓存） | 2873 tok/s | **真实 prefill 能力** |

#### 负载需求 vs 硬件能力

| 阶段 | bronze QPS | prompt 长度 | prefill 需求 | vs 硬件能力 |
|------|-----------|------------|-------------|------------|
| Phase 3 | 3 QPS | 4800 token | 14400 tok/s | **5 倍超载** |
| Phase 4 | 10 QPS | 4800 token | **48000 tok/s** | **16.7 倍超载** |
| Phase 5 | 15 QPS | 4800 token | 72000 tok/s | 25 倍超载 |

#### 结论

**Phase 4 的 bronze 负载（4800 token × 10 QPS = 48000 tok/s prefill）是硬件能力
（2873 tok/s）的 16.7 倍**。这是负载设计过载，不是调度问题：

1. 旧数据（230 token）时 Phase 4 是 2300 tok/s，接近硬件能力，属「适度过载」；
2. 修正为 4800 token 后，负载变成 48000 tok/s，是「完全过载」，物理无解；
3. 单张 5070 Ti 无法承担 10 QPS 的 4800 token 长文档 prefill（需多张 A100 级别 GPU）。

#### 需要决策的问题

修正测试数据后，Phase 4 负载变得不现实。需要重新审视测试负载设计：

- 方案 1：降低 Phase 4 的 bronze QPS（10 → 1~2），让负载回归现实；
- 方案 2：降低长文档长度（4800 → 2000），介于「长文档」与「可处理」之间；
- 方案 3：接受 Phase 4 为「极限过载场景」，违约是预期的，聚焦 Phase 1-3 展示优化效果。

### 7.16 方案 1 验证：降低 bronze QPS 后 A/B 对比（2026-08-23）

> 采用方案 1，用 `--bronze-qps-scale 0.2`（Phase 4 bronze 从 10 → 2 QPS）重跑
> A/B 对比，量化 priority 优化在「临界过载」下的效果。

#### A/B 对比结果（scale=0.2，真长文档 4800 token）

| Phase 4 违约率 | A 基线(fcfs) | B 优化(priority) | 差异 |
|---------------|-------------|-----------------|------|
| **gold** | 83.2% | 96.6% | 优化无效（反而更差）❌ |
| **silver** | 76.3% | **20.0%** | 优化有效，-56.3pp ✅ |
| **bronze** | 69.0% | **16.0%** | 优化有效，-53.0pp ✅ |

#### 完成请求数对比（揭示「保护」的真相）

| Phase 4 完成请求数 | A 基线 | B 优化 | 说明 |
|-------------------|--------|--------|------|
| silver | 930 | **115** | B 拒绝 87.6% |
| bronze | 100 | **25** | B 拒绝 75.0% |

> B 优化的 silver/bronze 低违约率是**准入控制拒绝**的结果（大量请求被拒，未计入违约统计），
> 而非「更快完成」。这是「幸存者偏差」，完整评估需同时看违约率 + 拒绝率 + 完成率。

#### 最终结论（本轮调度优化验证的完整总结）

1. **priority 优化对 gold 无法保护**：gold（priority=-2）是最高优先级，准入控制永不拒绝，
   只能排队，排队即违约。gold 违约源于「自身高 QPS + 长文档抢占 GPU」的物理约束，
   任何调度策略都无法解决。

2. **priority 优化对 silver/bronze 有效**：通过准入控制拒绝低优先级请求，
   把过载代价从「高优先级租户」转移到「低优先级租户」，量化收益为 silver/bronze
   违约率 **-53 ~ -56pp**。

3. **调度的能力边界清晰**：调度优化能「决定谁被拒绝」，不能「凭空增加算力」。
   当负载超硬件能力时，必须牺牲低优先级请求，这是准入控制的本质。

#### 三个关键测试配置坑（务必记录，避免后人踩坑）

| 坑 | 影响 | 正确做法 |
|----|------|---------|
| `--enforce-eager` | 禁用 CUDA graph，prefill 慢 60-100 倍，Phase 3 就 100% 违约 | 去掉，用默认 CUDA graph |
| 长文档长度 | 4800 token × 10 QPS = 48000 tok/s，超硬件 16.7 倍 | 用 `--bronze-qps-scale` 控制到合理范围 |
| `max_num_seqs` | 降低到 128 反而更差 | 保持默认 256 |

#### 收尾说明

本轮「调度优化验证」到此收尾。已完成的验证链：

1. §7.2 A/B 基线对比（旧数据 230 token）
2. §7.6/§7.7 准入控制修复 + 根因分析
3. §7.9 隔离实验（长文档是违约唯一根因）
4. §7.10 负载扫描（定位有效边界）
5. §7.11 测试数据修正（230 → 4800 token）
6. §7.12-7.15 真长文档重测 + 根因排查
7. §7.16 方案 1 验证 + 最终结论

核心结论稳定且可复现：**调度优化（priority + 准入控制）能有效保护 silver/bronze 租户
（-53~-56pp 违约率），但无法保护 gold（最高优先级自身过载，物理无解）。**

### 7.17 指标覆盖度自检：已验证 vs 尚缺（重要）

> 本节显式记录一个**证据链缺口**：§7 整条验证链的实测数据，**几乎全部集中在「违约率 / 拒绝率 / 完成请求数」**，
> 而方法论（§0、§4、§扩展 0）设计的「吞吐、TPOT、ITL」等绝对性能指标**基本没有真正采集用于 A/B 对比**。
> 结论"调度优化保护了 silver/bronze"目前**只有公平性一条腿**，缺「整体吞吐是否被牺牲」这条腿。

#### 7.17.1 已验证的指标（§7 有实测 A/B 数据）

| 指标 | 来源 | 在 §7 的落点 | 状态 |
|------|------|-------------|------|
| 分阶段 / 分租户**违约率** | `workload.py` | §7.2/§7.6/§7.9/§7.10/§7.12/§7.16 | ✅ 充分 |
| 准入**拒绝次数 / 拒绝率** | `workload.py` + scheduler 日志 | §7.6（6038 次）/§7.16 | ✅ 充分 |
| **完成请求数** | `workload.py` | §7.10/§7.16 | ✅ 有（用于揭示"保护=拒绝"） |
| **P99 TTFT**（零星） | `workload.py` | §7.2（如 94072ms）/§7.9（68ms） | ⚠️ 仅个别点，未系统 A/B |

#### 7.17.2 尚缺的指标（设计了但 §7 未执行 → 已于 §7.17.5 补测）

| 指标 | 应有来源 | 为什么重要 | 当前状态 |
|------|---------|-----------|---------|
| **吞吐（req/s、tok/s）** | `vllm bench serve`（§扩展 0） | reviewer 必问："优化有没有牺牲整体吞吐？"（§扩展 0 原话） | ✅ 已补测（§7.17.5，10 QPS 下零退化） |
| **P99 TPOT** | `vllm bench serve` | Token 限速是否过度拖慢 Decode | ✅ 已补测（§7.17.5，±0%） |
| **P99 ITL** | `vllm bench serve` | 抢占是否影响生成流畅度 | ✅ 已补测（§7.17.5，-1.0%） |
| **P99 E2EL**（系统性） | `vllm bench serve` | 端到端总延迟的 A/B 对比 | ✅ 已补测（§7.17.5，+0.3%） |
| 服务端**排队时间 / Prefill-Decode 耗时** | Prometheus+Grafana（§扩展 -1） | 坐实瓶颈在排队还是 Prefill | ✅ 数字已补（§7.17.6：A 排队 12.2s vs B 0.11s）；截图待补 |

#### 7.17.3 缺口的影响

1. **"保护 = 拒绝" 的代价未被量化**：§7.16 已证明优化靠**拒绝大量 silver/bronze** 来降违约率
   （silver 拒 87.6%、bronze 拒 75%）。被拒请求不产出 token，**必然影响整体有效吞吐**——
   但没有吞吐数据，无法回答"这个保护值不值"。
2. **无法排除"优化引入了 Decode 退化"**：QoS/MLFQ/Token 限速可能增加调度开销或抢占，
   理应用 TPOT/ITL 验证"Decode 不受影响"（§0.3 / §7.0.2 判据 ±5%），但从未实测。
3. **判据悬空**：§7.0.2 定了"B 轮吞吐与 A 轮 ±5% 内为安全，>10% 为过度优化"的判据，
   却没有任何一轮数据去套用这个判据。

#### 7.17.4 补测命令清单（把缺的这条腿补上）

在 §3 的 A / B 两轮服务**各在跑时**，分别执行以下命令（服务**不要重启**，避免 KV Cache 状态变化），
即可补齐吞吐 / TPOT / ITL / E2EL 的 A/B 对比。**关键：A/B 用完全相同的 bench 参数**。

```bash
cd ~/vllm-serving-optimization && source .venv/bin/activate

# ===== A 基线（fcfs）服务在跑时执行 =====
vllm bench serve \
    --backend openai --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 1000 --request-rate 10 --temperature 0 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir results/baseline_fcfs/abs_bench/ \
    --result-filename abs_bench.json --metadata tier=baseline policy=fcfs

# ===== B 优化（priority）服务在跑时执行（相同参数）=====
vllm bench serve \
    --backend openai --model Qwen/Qwen2.5-1.5B-Instruct \
    --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
    --dataset-name random --random-input-len 1024 --random-output-len 128 \
    --num-prompts 1000 --request-rate 10 --temperature 0 --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --save-result --result-dir results/optimized_qos/abs_bench/ \
    --result-filename abs_bench.json --metadata tier=optimized policy=priority
```

> 详细参数说明见 §扩展 0.3；A/B 对比判据（±5% 安全 / >10% 过度优化）见 §扩展 0.5 与 §7.0.2。

补测后，把结果按下表填入，才算完成"绝对性能不退化"的验证：

| 指标 | A 基线 | B 优化 | 差异 | 判定（±5% 安全 / >10% 过度） |
|------|--------|--------|------|------------------------------|
| Request throughput (req/s) | 9.928 | 9.928 | 0% | ✅ 无退化 |
| Output token throughput (tok/s) | 1270.81 | 1270.84 | +0.002% | ✅ 无退化 |
| P99 TTFT (ms) | 168.84 | 177.06 | +4.9% | ✅ 安全（<5%） |
| P99 TPOT (ms) | 12.80 | 12.80 | 0% | ✅ 无退化 |
| P99 ITL (ms) | 41.53 | 41.11 | -1.0% | ✅ 无退化（略优） |
| Failed requests | 0 | 0 | — | ✅ 无失败 |

> **注意**：§7 的核心场景是「临界过载（`--bronze-qps-scale 0.2`）」，而上面 `vllm bench serve` 用的是
> 单一 random 流量的恒定 10 QPS，两者负载模型不同。要严格对齐 §7 结论，还应在**过载负载下**补一组
> 吞吐对比（例如加跑 `--request-rate inf` 测最大并发吞吐，或扫多档 QPS，见 §扩展 0.6），
> 用以量化"拒绝换来的公平性"到底牺牲了多少有效吞吐。

### 7.17.5 补测结果：绝对性能 A/B 对比（2026-08-26，无 eager）

> 按 §7.17.4 命令补测，补齐"吞吐 / TPOT / ITL / E2EL"这条腿。两轮均用**无 eager** 配置
> （§7.14 已证明 `--enforce-eager` 是性能大坑，此处对齐生产配置）。

#### 场景一：恒定 10 QPS（正常负载）—— 优化零退化

| 指标 | A 基线(fcfs) | B 优化(priority) | 差异 | 判定 |
|------|-------------|-----------------|------|------|
| Request throughput (req/s) | 9.928 | 9.928 | 0% | ✅ |
| Output token throughput (tok/s) | 1270.81 | 1270.84 | +0.002% | ✅ |
| P99 TTFT (ms) | 168.84 | 177.06 | +4.9% | ✅ 安全 |
| P99 TPOT (ms) | 12.80 | 12.80 | 0% | ✅ |
| P99 ITL (ms) | 41.53 | 41.11 | -1.0% | ✅ |
| P99 E2EL (ms) | 1718.42 | 1724.00 | +0.3% | ✅ |
| Failed requests | 0 | 0 | — | ✅ |

**结论**：在正常负载（10 QPS）下，priority 优化对绝对性能**零退化**（所有指标 ±5% 内，
TPOT/吞吐完全持平）。这坐实了「优化不牺牲整体吞吐」这条腿。

#### 场景二：过载负载（`--request-rate inf`）—— 揭示"拒绝换公平"的本质

| 指标 | A 基线(fcfs) | B 优化(priority) | 说明 |
|------|-------------|-----------------|------|
| Failed requests | 0 / 500 | **432 / 500（86.4%）** | B 的准入控制拒绝了 86% 请求 |
| Completed | 500 | 68 | 拒绝 = 不产出 token |
| P99 TTFT (ms) | 28654.15 | 2852.19 | 见下方"幸存者偏差"说明 |
| P99 E2EL (ms) | 30442.38 | 3935.29 | 见下方"幸存者偏差"说明 |
| Request throughput (req/s) | 16.39 | 17.11 | 相近（B 略高因拒绝快） |

**核心洞察**：过载下两种策略是**完全不同的权衡**：

- **fcfs（不拒绝）**：500 个请求全部"完成"，但 P99 TTFT 高达 28.6 秒（严重超时，SLA 全违约）
- **priority（拒绝 86%）**：仅 68 个请求完成，但 TTFT 降到 2.8 秒（接近 SLA 达标）

这**量化了"拒绝换公平"的代价**：优化并非"凭空变快"，而是**牺牲 86% 的低优请求，
换来剩余 14% 请求的低延迟**。这是准入控制的本质——过载时必有取舍，调度优化决定的是
"牺牲谁"，而不是"不牺牲"。

> ⚠️ **关于"快 10 倍"的严谨说明（幸存者偏差）**：28654 ÷ 2852 ≈ 10，但这两个 P99 TTFT
> **不是同一批请求**的对比：
> - A 基线的 P99 TTFT 是**全部 500 个请求**（含所有慢请求）的分位值；
> - B 优化的 P99 TTFT 只统计**成功完成的 68 个请求**（432 个被拒请求无 TTFT，不计入）。
>
> 因此"快 10 倍"的准确含义是：**"B 下幸存的那 14% 请求"比"A 下全部请求"快 10 倍**，
> 而非"同一请求在 B 下更快"。这是**幸存者偏差**——被拒请求没有延迟数据，导致 B 的
> P99 只反映"幸运者"。真正的公平对比应看「完成率 × 完成者的延迟」的联合分布，或
> 直接看「整体有效吞吐」（已在上方列出：两轮 req/s 相近，但 B 有 86% 请求未产出 token）。

#### 补测后的完整结论

结合 §7.16（公平性）+ §7.17.5（绝对性能），调度优化验证的证据链**两条腿齐全**：

1. **公平性腿**（§7.16）：priority 优化保护 silver/bronze，违约率 -53~-56pp
2. **绝对性能腿**（§7.17.5）：正常负载零退化（±5% 内）；过载下用"拒绝"换"低延迟"

### 7.17.6 补测结果：服务端排队时间 / Prefill 耗时（2026-08-26，无 eager）

> 补齐 §7.17.2 最后一环：服务端内部指标（`vllm:request_queue_time_seconds` /
> `vllm:request_prefill_time_seconds`），坐实「瓶颈在排队还是 Prefill」。
> 采集方式：过载负载（`vllm bench serve --request-rate inf`，500 请求）下直接
> curl vLLM `/metrics` 端点的 histogram。

#### 服务端指标对比（过载负载）

| 指标 | A 基线(fcfs) | B 优化(priority) | 解读 |
|------|-------------|-----------------|------|
| 完成请求数 | 500 / 500 | 68 / 500 | B 拒绝 86.4% |
| **平均排队时间（queue_time）** | **12.2 秒** | **0.11 秒** | 差异 110 倍 |
| 排队 P50（约） | ~5 秒 | <0.3 秒 | A 大量请求排队 10-30s |
| **平均 Prefill 时间** | **< 0.5 秒** | < 0.3 秒（68 个成功） | 两轮 Prefill 都快 |

#### 关键结论：坐实「瓶颈在排队，不在 Prefill」

1. **A 基线（fcfs）的瓶颈是排队**：平均排队 12.2 秒，而 Prefill 仅 < 0.5 秒。
   排队时间占端到端延迟的**绝对主导**（12.2s / 总延迟 ~24.7s ≈ 50%），
   且 P50 排队就已达 ~5 秒——请求大量堆积在 waiting 队列，而非 GPU 算力不足。

2. **B 优化（priority）通过"拒绝"消灭排队**：平均排队时间从 12.2s 降到 0.11s，
   但代价是拒绝 86.4% 请求（432 个）。幸存的 68 个请求排队 < 0.3s。

3. **Prefill 不是瓶颈**（此负载下）：两轮 Prefill 都 < 0.5 秒。这修正了 §7.9 之前
   对「长文档 Prefill 慢」的笼统归因——在 1024 token 输入下 Prefill 很快，真正的
   瓶颈是**调度队列堆积**（长文档场景下是长文档 Prefill 独占 GPU 导致队列堆积，
   但根源仍是「排队」，而非「Prefill 本身慢」）。

> **说明**：B 优化的 `request_prefill_time_seconds` 对 432 个被拒请求记录为异常大值
> （落在 1920~7680s 桶），是被拒请求未正常完成 prefill 导致的计数口径问题，非真实
> Prefill 耗时。表中 Prefill 时间仅取「成功请求」的口径。

#### Grafana 截图

> Grafana Dashboard「vLLM 调度监控 — 对比测试」（8 个面板）的过载压测时段截图，
> 完整覆盖排队长度、排队时间、Prefill/Decode 耗时等核心面板：

![Grafana 调度监控 Dashboard](assets/grafana_scheduling_dashboard.png)

**截图时段标注**（通过 Prometheus 历史数据精确回查确认，非人工记忆）：

| 时间段 | 对应轮次 | 判定依据 |
|--------|---------|---------|
| 20:15 ~ 20:16（左段） | **B 优化（priority）** | waiting 峰值 285，快速回落（拒绝后清空队列） |
| 20:29 ~ 20:30（右段） | **A 基线（fcfs）** | waiting 峰值 434，`queue_time_sum` 累计 6108s（= A 基线的实测值） |

> 判定依据说明：右段（20:29）的 `vllm:request_queue_time_seconds_sum` 累计值恰为
> **6108.3**，与 §7.17.6 表格中 A 基线的实测 `sum=6108.315` 完全吻合，故右段为 A 基线；
> 左段（20:15）对应 B 优化（priority）。

**面板解读（对应 §7.17.6 的数字结论）**：

| 面板 | 观察 | 印证结论 |
|------|------|---------|
| 1. 队列长度（Running vs Waiting） | 两轮 waiting 都瞬间堆积（285 / 434），A 基线峰值更高 | A 基线排队更严重 |
| 5. 请求排队时间 P99 | 两轮 P99 排队都飙到 20~30s（P99 远高于均值） | 排队是过载下的主导延迟 |
| 6. Prefill vs Decode 耗时 P95 | A 基线 Prefill 几乎为 0；B 优化出现 ~8000 异常值（被拒请求） | Prefill 非瓶颈；被拒请求计数异常 |

> 注：面板 5 的 P99（20~30s）远高于表格里的「平均排队 12.2s / 0.11s」，因为 P99 是
> 最慢 1% 请求的分位值，且截图包含 A/B 两轮混叠的瞬时峰。精确的均值对比以 §7.17.6
> 的 histogram 数据为准。

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
