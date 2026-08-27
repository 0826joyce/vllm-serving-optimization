#!/usr/bin/env bash
# 优化 9 多模型资源池化 · 一键测试脚本（方案 B 外部编排）
#
# 串起 4 个进程：
#   1) vLLM 实例 A（端口 8001，占 ~40% 显存）
#   2) vLLM 实例 B（端口 8002，占 ~40% 显存）
#   3) 编排器 multi_model_arbiter.py（监控负载、自动 sleep/wake）
#   4) 错峰压测 skew_workload.py（先打 A、后打 B，触发 sleep/wake）
#
# 关键前提（脚本已内置）：
#   - 每个实例启动带 --enable-sleep-mode
#   - 每个实例启动带环境变量 VLLM_SERVER_DEV_MODE=1（否则 /sleep /wake_up 路由不注册）
#
# 用法：
#   bash benchmarks/e2e_cases/run_multimodel_pool.sh
#   MODEL=Qwen/Qwen2.5-0.5B-Instruct IDLE=60 BUSY=90 CYCLES=2 QPS=5 \
#       bash benchmarks/e2e_cases/run_multimodel_pool.sh
#
# 结束：Ctrl+C 或压测跑完后，脚本会自动关闭所有拉起的进程。

set -euo pipefail

# ---------------- 可配置参数（环境变量覆盖）----------------
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT_A="${PORT_A:-8001}"
PORT_B="${PORT_B:-8002}"
GPU_UTIL="${GPU_UTIL:-0.4}"        # 单卡起两个实例，各占 40%
IDLE="${IDLE:-60}"                 # 编排器：空闲多久 sleep
POLL="${POLL:-5}"                  # 编排器：轮询间隔
DURATION="${DURATION:-600}"        # 编排器：运行总时长
BUSY="${BUSY:-90}"                 # 错峰：每实例单独忙的时长（应 > IDLE）
CYCLES="${CYCLES:-2}"              # 错峰：循环轮数
QPS="${QPS:-5}"                    # 错峰：发请求速率
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"  # 等待实例就绪的最长秒数

# ---------------- 路径 ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARBITER="$SCRIPT_DIR/multi_model_arbiter.py"
SKEW="$SCRIPT_DIR/skew_workload.py"
LOG_DIR="$SCRIPT_DIR/pool_test_logs"
mkdir -p "$LOG_DIR"

PIDS=()

cleanup() {
    echo ""
    echo "[run] 清理拉起的进程..."
    for pid in "${PIDS[@]:-}"; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # 给进程一点时间优雅退出
    sleep 2
    for pid in "${PIDS[@]:-}"; do
        if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "[run] 完成。日志在 $LOG_DIR/"
}
trap cleanup EXIT INT TERM

# ---------------- 等待实例就绪 ----------------
wait_ready() {
    local url="$1" name="$2" waited=0
    echo "[run] 等待 $name ($url) 就绪..."
    while (( waited < STARTUP_TIMEOUT )); do
        if curl -sf "$url/health" >/dev/null 2>&1; then
            echo "[run] $name 已就绪"
            return 0
        fi
        sleep 3
        waited=$(( waited + 3 ))
    done
    echo "[run] ERROR: $name 在 ${STARTUP_TIMEOUT}s 内未就绪，见 $LOG_DIR/server_${name}.log"
    return 1
}

echo "========================================================"
echo "[run] 优化 9 多模型资源池化测试"
echo "[run]   MODEL=$MODEL  A=:$PORT_A  B=:$PORT_B  GPU_UTIL=$GPU_UTIL"
echo "[run]   编排器: IDLE=${IDLE}s POLL=${POLL}s DURATION=${DURATION}s"
echo "[run]   错峰:   BUSY=${BUSY}s CYCLES=$CYCLES QPS=$QPS"
echo "========================================================"

# ---------------- 进程1：实例 A ----------------
echo "[run] 启动实例 A (:$PORT_A) ..."
VLLM_SERVER_DEV_MODE=1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --enable-sleep-mode \
    --gpu-memory-utilization "$GPU_UTIL" --port "$PORT_A" \
    > "$LOG_DIR/server_A.log" 2>&1 &
PIDS+=($!)

# ---------------- 进程2：实例 B ----------------
echo "[run] 启动实例 B (:$PORT_B) ..."
VLLM_SERVER_DEV_MODE=1 python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --enable-sleep-mode \
    --gpu-memory-utilization "$GPU_UTIL" --port "$PORT_B" \
    > "$LOG_DIR/server_B.log" 2>&1 &
PIDS+=($!)

# ---------------- 等两个实例就绪 ----------------
wait_ready "http://127.0.0.1:$PORT_A" "A"
wait_ready "http://127.0.0.1:$PORT_B" "B"

# ---------------- 进程3：编排器 ----------------
echo "[run] 启动编排器 ..."
python "$ARBITER" \
    --backend "name=modelA,url=http://127.0.0.1:$PORT_A" \
    --backend "name=modelB,url=http://127.0.0.1:$PORT_B" \
    --idle-sleep-seconds "$IDLE" --poll-interval "$POLL" --duration "$DURATION" \
    --result-file "$LOG_DIR/arbiter_result.json" \
    > "$LOG_DIR/arbiter.log" 2>&1 &
PIDS+=($!)
echo "[run] 编排器日志: $LOG_DIR/arbiter.log"

# 给编排器一点时间开始轮询
sleep 3

# ---------------- 进程4：错峰压测（前台，跑完即结束）----------------
echo "[run] 启动错峰压测（前台，跑完自动收尾）..."
python "$SKEW" \
    --backend "name=modelA,url=http://127.0.0.1:$PORT_A" \
    --backend "name=modelB,url=http://127.0.0.1:$PORT_B" \
    --model "$MODEL" \
    --busy-seconds "$BUSY" --cycles "$CYCLES" --qps "$QPS" \
    --result-file "$LOG_DIR/skew_result.json" \
    2>&1 | tee "$LOG_DIR/skew.log"

echo ""
echo "[run] 错峰压测结束。结果文件："
echo "[run]   编排统计: $LOG_DIR/arbiter_result.json"
echo "[run]   压测统计: $LOG_DIR/skew_result.json"
echo "[run] 可用 nvidia-smi / Grafana 交叉查看 sleep 前后显存与利用率。"
# trap cleanup 会在退出时关闭 A/B/编排器
