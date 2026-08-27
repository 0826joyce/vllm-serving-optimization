#!/usr/bin/env python3
"""错峰负载压测脚本（配合优化 9 多模型资源池化编排器）

用途：
    为 `multi_model_arbiter.py`（优化 9 · 方案 B）制造「一个实例忙、另一个空闲」的
    错峰负载，从而触发编排器对空闲实例的 sleep、以及重新有请求时的 wake_up。

错峰时间线（默认，可用参数调）：
    Phase 1  [0 ~ busy_seconds)              : 只打实例 A（B 空闲 → 应被编排器 sleep）
    Phase 2  [busy_seconds ~ 2*busy_seconds) : 只打实例 B（B 被唤醒；A 空闲 → 应被 sleep）
    （可用 --cycles 重复以上两阶段，观察多次 sleep/wake）

配合使用（4 个进程，无需 Docker）：
    # 1) 起实例 A / B（略，见 SCHEDULING_BENCHMARK_GUIDE.md 附录 B）
    # 2) 起编排器 multi_model_arbiter.py
    # 3) 起本脚本制造错峰：
    python skew_workload.py \
        --backend name=modelA,url=http://127.0.0.1:8001 \
        --backend name=modelB,url=http://127.0.0.1:8002 \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --busy-seconds 90 --cycles 2 --qps 5

说明：
    - busy_seconds 建议 > 编排器的 idle_sleep_seconds，才能让空闲实例真正进入 sleep；
    - 本脚本只负责「发请求」，sleep/wake 的判定与执行由编排器完成；
    - 用标准库 + 线程并发发请求，零额外依赖。
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class Target:
    name: str
    url: str  # e.g. http://127.0.0.1:8001


def _parse_backend(spec: str) -> Target:
    """解析 name=modelA,url=http://127.0.0.1:8001"""
    kv = {}
    for part in spec.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"非法 backend 片段: {part}")
        k, v = part.split("=", 1)
        kv[k.strip()] = v.strip()
    if "name" not in kv or "url" not in kv:
        raise argparse.ArgumentTypeError(f"backend 必须含 name 和 url: {spec}")
    return Target(name=kv["name"], url=kv["url"].rstrip("/"))


def send_one(target: Target, model: str, prompt: str, max_tokens: int,
             timeout: float) -> tuple[bool, float]:
    """向某实例发一个 completion 请求。返回 (成功, 耗时秒)。

    休眠中的实例会拒绝/超时——这是预期的（编排器会在下一轮唤醒它），
    这里把失败也计入统计，用于观察「唤醒窗口内的失败」。
    """
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{target.url}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
        return True, time.monotonic() - t0
    except Exception:
        return False, time.monotonic() - t0


def drive_phase(target: Target, model: str, duration: float, qps: float,
                max_tokens: int, stats: dict) -> None:
    """在 duration 秒内，以约 qps 的速率持续向 target 发请求（多线程）。"""
    print(f"[skew] >>> 打流量到 {target.name} ({target.url})，{duration}s @ {qps} QPS")
    end = time.monotonic() + duration
    interval = 1.0 / qps if qps > 0 else 0.2
    prompt = "Explain the concept of speculative decoding in large language models."
    threads: list[threading.Thread] = []

    def _worker():
        ok, cost = send_one(target, model, prompt, max_tokens, timeout=120.0)
        with stats["lock"]:
            key = target.name
            stats[key]["sent"] += 1
            if ok:
                stats[key]["ok"] += 1
                stats[key]["latencies"].append(cost)
            else:
                stats[key]["fail"] += 1

    while time.monotonic() < end:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(interval)

    # 等待收尾（给最后一批请求留时间，含可能的唤醒延迟）
    for t in threads:
        t.join(timeout=120.0)
    print(f"[skew] <<< {target.name} 阶段结束")


def main():
    ap = argparse.ArgumentParser(
        description="错峰负载压测（配合优化 9 编排器）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--backend", action="append", required=True, type=_parse_backend,
                    metavar="name=<name>,url=<url>",
                    help="被打流量的 vLLM 实例，需指定 2 个（A、B）")
    ap.add_argument("--model", required=True, help="模型名（各实例可相同）")
    ap.add_argument("--busy-seconds", type=float, default=90.0,
                    help="每个实例被单独打流量的时长（秒），建议 > 编排器 idle_sleep_seconds")
    ap.add_argument("--cycles", type=int, default=1,
                    help="错峰循环次数（A忙→B忙 记一轮），默认 1")
    ap.add_argument("--qps", type=float, default=5.0, help="每阶段发请求速率，默认 5")
    ap.add_argument("--max-tokens", type=int, default=64, help="每请求生成 token 数")
    ap.add_argument("--result-file", type=str, default=None, help="统计 JSON 输出路径")
    args = ap.parse_args()

    if len(args.backend) < 2:
        ap.error("需要指定 2 个 --backend（A 和 B）以制造错峰")

    a, b = args.backend[0], args.backend[1]
    stats = {
        "lock": threading.Lock(),
        a.name: {"sent": 0, "ok": 0, "fail": 0, "latencies": []},
        b.name: {"sent": 0, "ok": 0, "fail": 0, "latencies": []},
    }

    print(f"[skew] 错峰压测开始：A={a.name}, B={b.name}, "
          f"busy={args.busy_seconds}s, cycles={args.cycles}, qps={args.qps}")
    start = time.monotonic()
    for c in range(args.cycles):
        print(f"\n[skew] ===== 第 {c + 1}/{args.cycles} 轮 =====")
        drive_phase(a, args.model, args.busy_seconds, args.qps, args.max_tokens, stats)
        drive_phase(b, args.model, args.busy_seconds, args.qps, args.max_tokens, stats)
    total = time.monotonic() - start

    # ---- 汇总 ----
    print("\n========== 错峰压测统计 ==========")
    summary = {"total_seconds": round(total, 1), "backends": []}
    for name in (a.name, b.name):
        s = stats[name]
        lat = s["latencies"]
        avg_lat = sum(lat) / len(lat) if lat else 0.0
        rec = {
            "name": name,
            "sent": s["sent"],
            "ok": s["ok"],
            "fail": s["fail"],
            "avg_latency_s": round(avg_lat, 3),
            "max_latency_s": round(max(lat), 3) if lat else 0.0,
        }
        summary["backends"].append(rec)
        print(f"  {name}: sent={s['sent']}, ok={s['ok']}, fail={s['fail']}, "
              f"avg_lat={rec['avg_latency_s']}s, max_lat={rec['max_latency_s']}s")
    print("  提示：某实例刚被唤醒时的首个请求 max_latency 会明显偏高——"
          "那部分就包含了 wake_up 的唤醒延迟。")

    if args.result_file:
        with open(args.result_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[skew] 结果已写入 {args.result_file}")


if __name__ == "__main__":
    main()
