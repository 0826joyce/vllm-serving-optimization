#!/usr/bin/env python3
"""多模型资源池化编排器（优化 9 · 方案 B：外部编排）

对应 benchmarks/basic_optimization/scheduling-resource-management-optimization.md 的
「优化 9：多模型实例的资源池化调度」。

设计背景（可行性结论）：
    vLLM V1 的一个引擎实例只绑定一个模型，`sleep/wake_up` 是对「整个实例」而非
    「实例内某个模型」的操作。因此「单实例内多模型仲裁」在 V1 架构下不可行。
    真正可落地的形态是「多个独立 vLLM 实例 + 外部编排层」——本脚本即该外部编排器。

它做什么：
    - 周期性轮询每个后端实例的负载（vLLM /metrics 里的 num_requests_running/waiting）；
    - 对「持续空闲超过阈值」的实例调用 POST /sleep 释放显存（弹性缩容 / scale-to-zero）；
    - 对「重新收到请求（waiting>0）」的休眠实例调用 POST /wake_up 唤醒；
    - 记录每次 sleep/wake 的时间戳与唤醒延迟，输出到 JSON，供测试归因。

它不做什么（诚实边界）：
    - 不做底层显存虚拟化（CUDA VMM / Global XTensor）——那是 xLLM PR #861 级别的
      基础设施改造，超出「调度/编排层」范围；
    - sleep/wake 是粗粒度整实例切换，唤醒有秒级延迟（用时间换空间）。

前置要求（每个被编排的 vLLM 实例启动时必须满足）：
    VLLM_SERVER_DEV_MODE=1 \
    python -m vllm.entrypoints.openai.api_server \
        --model <model> --enable-sleep-mode \
        --gpu-memory-utilization <单卡多实例时按比例分> --port <port>
    - `--enable-sleep-mode`：否则 sleep 的显存池机制不生效；
    - `VLLM_SERVER_DEV_MODE=1`：否则 /sleep、/wake_up、/is_sleeping 路由不注册（404）。

用法：
    python multi_model_arbiter.py \
        --backend name=modelA,url=http://127.0.0.1:8001 \
        --backend name=modelB,url=http://127.0.0.1:8002 \
        --idle-sleep-seconds 60 \
        --poll-interval 5 \
        --duration 600 \
        --result-file arbiter_result.json

    # 只观测不实际 sleep/wake（先看负载曲线，dry-run）：
    python multi_model_arbiter.py --backend ... --dry-run
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict


# ============================================================
# 1. 后端实例抽象
# ============================================================


@dataclass
class Backend:
    """一个被编排的 vLLM 实例。"""

    name: str
    url: str  # e.g. http://127.0.0.1:8001

    # ---- 运行时状态 ----
    sleeping: bool = False
    last_active_ts: float = field(default_factory=time.monotonic)  # 最近一次有负载的时刻
    # ---- 统计（供测试归因）----
    sleep_count: int = 0
    wake_count: int = 0
    total_sleep_seconds: float = 0.0
    wake_latencies: list[float] = field(default_factory=list)  # 每次唤醒耗时（秒）
    _sleep_started_ts: float = 0.0


# ============================================================
# 2. HTTP 工具（标准库，零额外依赖）
# ============================================================


def _http_post(url: str, timeout: float = 60.0) -> int:
    """POST 请求，返回状态码；异常返回 -1。"""
    try:
        req = urllib.request.Request(url, method="POST", data=b"")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return -1


def _http_get_text(url: str, timeout: float = 10.0) -> str | None:
    """GET 请求，返回文本；异常返回 None。"""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def _parse_metric(metrics_text: str, metric_name: str) -> float:
    """从 Prometheus /metrics 文本里解析某个 gauge 的当前值（取该指标所有 label 的和）。

    vLLM 指标形如：
        vllm:num_requests_running{model_name="..."} 8.0
    """
    total = 0.0
    found = False
    for line in metrics_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(metric_name):
            # metric_name 后面必须是 '{' 或空格，避免前缀误匹配
            rest = line[len(metric_name):]
            if rest and rest[0] not in ("{", " "):
                continue
            try:
                value = float(line.rsplit(" ", 1)[1])
                total += value
                found = True
            except (ValueError, IndexError):
                continue
    return total if found else 0.0


# ============================================================
# 3. 负载探测
# ============================================================


def probe_load(backend: Backend) -> dict:
    """探测一个实例的当前负载。

    返回 {running, waiting, reachable}。
    休眠中的实例 /metrics 可能仍可达但负载为 0。
    """
    text = _http_get_text(f"{backend.url}/metrics")
    if text is None:
        return {"running": 0.0, "waiting": 0.0, "reachable": False}
    running = _parse_metric(text, "vllm:num_requests_running")
    waiting = _parse_metric(text, "vllm:num_requests_waiting")
    return {"running": running, "waiting": waiting, "reachable": True}


def is_sleeping_remote(backend: Backend) -> bool | None:
    """查询实例的真实休眠状态（/is_sleeping）。异常返回 None。"""
    text = _http_get_text(f"{backend.url}/is_sleeping")
    if text is None:
        return None
    try:
        return bool(json.loads(text).get("is_sleeping", False))
    except Exception:
        return None


# ============================================================
# 4. sleep / wake 动作
# ============================================================


def do_sleep(backend: Backend, level: int, dry_run: bool) -> None:
    if backend.sleeping:
        return
    print(f"[arbiter] SLEEP  {backend.name} ({backend.url}) level={level}"
          f"{' [dry-run]' if dry_run else ''}")
    if not dry_run:
        code = _http_post(f"{backend.url}/sleep?level={level}")
        if code != 200:
            print(f"[arbiter] WARN: sleep {backend.name} 返回 {code}，跳过")
            return
    backend.sleeping = True
    backend.sleep_count += 1
    backend._sleep_started_ts = time.monotonic()


def do_wake(backend: Backend, dry_run: bool) -> None:
    if not backend.sleeping:
        return
    print(f"[arbiter] WAKE   {backend.name} ({backend.url})"
          f"{' [dry-run]' if dry_run else ''}")
    t0 = time.monotonic()
    if not dry_run:
        code = _http_post(f"{backend.url}/wake_up")
        if code != 200:
            print(f"[arbiter] WARN: wake_up {backend.name} 返回 {code}，跳过")
            return
    latency = time.monotonic() - t0  # 唤醒延迟（HTTP 同步返回视为唤醒完成）
    backend.sleeping = False
    backend.wake_count += 1
    backend.wake_latencies.append(latency)
    backend.total_sleep_seconds += time.monotonic() - backend._sleep_started_ts
    backend.last_active_ts = time.monotonic()
    print(f"[arbiter]        唤醒延迟 = {latency:.3f}s")


# ============================================================
# 5. 编排主循环
# ============================================================


def arbitrate_once(backends: list[Backend], idle_sleep_seconds: float,
                   sleep_level: int, dry_run: bool) -> list[dict]:
    """一轮仲裁：探测负载 → 空闲的睡 / 有请求的醒。返回本轮快照。"""
    now = time.monotonic()
    snapshot = []
    for b in backends:
        load = probe_load(b)
        has_work = load["running"] > 0 or load["waiting"] > 0

        if has_work:
            b.last_active_ts = now
            if b.sleeping:
                # 有新请求进来 → 唤醒
                do_wake(b, dry_run)
        else:
            idle_for = now - b.last_active_ts
            if not b.sleeping and idle_for >= idle_sleep_seconds:
                # 持续空闲超阈值 → 休眠释放显存
                do_sleep(b, sleep_level, dry_run)

        snapshot.append({
            "name": b.name,
            "running": load["running"],
            "waiting": load["waiting"],
            "reachable": load["reachable"],
            "sleeping": b.sleeping,
            "idle_for": round(now - b.last_active_ts, 1),
        })
    return snapshot


def run(backends: list[Backend], poll_interval: float, idle_sleep_seconds: float,
        duration: float, sleep_level: int, dry_run: bool, result_file: str | None):
    print(f"[arbiter] 启动：{len(backends)} 个后端，poll={poll_interval}s，"
          f"idle_sleep={idle_sleep_seconds}s，level={sleep_level}，"
          f"duration={duration}s{'，dry-run' if dry_run else ''}")
    for b in backends:
        print(f"[arbiter]   - {b.name}: {b.url}")

    start = time.monotonic()
    timeline: list[dict] = []
    try:
        while time.monotonic() - start < duration:
            snap = arbitrate_once(backends, idle_sleep_seconds, sleep_level, dry_run)
            timeline.append({"t": round(time.monotonic() - start, 1), "backends": snap})
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[arbiter] 收到中断，收尾统计…")

    # ---- 汇总统计 ----
    summary = {
        "config": {
            "poll_interval": poll_interval,
            "idle_sleep_seconds": idle_sleep_seconds,
            "sleep_level": sleep_level,
            "duration": duration,
            "dry_run": dry_run,
        },
        "backends": [],
    }
    print("\n========== 编排统计 ==========")
    for b in backends:
        avg_wake = (sum(b.wake_latencies) / len(b.wake_latencies)
                    if b.wake_latencies else 0.0)
        stat = {
            "name": b.name,
            "sleep_count": b.sleep_count,
            "wake_count": b.wake_count,
            "total_sleep_seconds": round(b.total_sleep_seconds, 1),
            "avg_wake_latency_s": round(avg_wake, 3),
            "max_wake_latency_s": round(max(b.wake_latencies), 3) if b.wake_latencies else 0.0,
            "wake_latencies": [round(x, 3) for x in b.wake_latencies],
        }
        summary["backends"].append(stat)
        print(f"  {b.name}: sleep×{b.sleep_count}, wake×{b.wake_count}, "
              f"累计休眠 {stat['total_sleep_seconds']}s, "
              f"平均唤醒延迟 {stat['avg_wake_latency_s']}s")

    if result_file:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "timeline": timeline}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n[arbiter] 结果已写入 {result_file}")


# ============================================================
# 6. CLI
# ============================================================


def _parse_backend(spec: str) -> Backend:
    """解析 --backend name=modelA,url=http://127.0.0.1:8001"""
    kv = {}
    for part in spec.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"非法 backend 片段: {part}")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    if "name" not in kv or "url" not in kv:
        raise argparse.ArgumentTypeError(f"backend 必须含 name 和 url: {spec}")
    return Backend(name=kv["name"], url=kv["url"].rstrip("/"))


def main():
    ap = argparse.ArgumentParser(
        description="多模型资源池化编排器（优化 9 · 方案 B）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--backend", action="append", required=True, type=_parse_backend,
                    metavar="name=<name>,url=<url>",
                    help="被编排的 vLLM 实例，可重复指定多个")
    ap.add_argument("--poll-interval", type=float, default=5.0,
                    help="负载轮询间隔（秒），默认 5")
    ap.add_argument("--idle-sleep-seconds", type=float, default=60.0,
                    help="持续空闲多久后 sleep（秒），默认 60")
    ap.add_argument("--sleep-level", type=int, default=1, choices=[1, 2],
                    help="sleep 级别：1=offload 权重+丢 KV，2=丢全部显存，默认 1")
    ap.add_argument("--duration", type=float, default=600.0,
                    help="编排器运行总时长（秒），默认 600")
    ap.add_argument("--dry-run", action="store_true",
                    help="只观测负载、不实际调用 sleep/wake（先看曲线）")
    ap.add_argument("--result-file", type=str, default=None,
                    help="统计结果 JSON 输出路径")
    args = ap.parse_args()

    run(
        backends=args.backend,
        poll_interval=args.poll_interval,
        idle_sleep_seconds=args.idle_sleep_seconds,
        duration=args.duration,
        sleep_level=args.sleep_level,
        dry_run=args.dry_run,
        result_file=args.result_file,
    )


if __name__ == "__main__":
    main()
