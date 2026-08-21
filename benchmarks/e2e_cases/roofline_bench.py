"""Roofline 模型基准测试脚本

用于配合 nsys/ncu 验证 GPU 各运算阶段是否达到对应的理论峰值（roof）。

Roofline 模型回顾：
    算术强度 AI = FLOPs / Bytes（每个字节访存对应多少浮点运算）
    - AI 低 → 受显存带宽限制（Memory-bound），可达性能 = AI × 带宽
    - AI 高 → 受算力限制（Compute-bound），可达性能 = 峰值算力
    拐点 = 峰值算力 / 带宽（ridge point）

本脚本针对 RTX 5070 Ti（Blackwell, sm_120）：
    - FP32 峰值算力   ≈ 55.4 TFLOPS（8960 cores × 2 × 3.09 GHz）
    - FP16 峰值算力   ≈ 110  TFLOPS（Tensor Core，非稀疏）
    - 显存带宽        ≈ 896  GB/s（GDDR7）
    - 拐点（FP32）    ≈ 55.4e12 / 896e9 ≈ 61.8 FLOP/Byte

设计目标：用不同算术强度的算子，分别命中「访存屋顶」和「算力屋顶」，
并打印实测值 + 与理论峰值的百分比，判断是否达到对应 roof。

用法（Windows 原生环境）：
    python roofline_bench.py            # 全跑，打印汇总表
    python roofline_bench.py --kind all # 同上
    # 配合 nsys/ncu 单跑某一项（见 SCHEDULING_BENCHMARK_GUIDE / TUTORIAL）
    python roofline_bench.py --kind memcpy
    python roofline_bench.py --kind fp32
    python roofline_bench.py --kind fp16
"""

import argparse
import time

import torch

# ============================================================
# RTX 5070 Ti 理论峰值（可用 nvidia-smi 实测的 boost clock 修正）
# ============================================================
THEORETICAL = {
    "fp32_tflops": 55.4,   # TFLOPS，8960 CUDA cores × 2 FLOP × 3.09 GHz
    "fp16_tflops": 110.0,  # TFLOPS，Tensor Core（非稀疏），约为 FP32 的 2 倍
    "mem_bw_gbps": 896.0,  # GB/s，GDDR7 16GB
}

# 拐点（ridge point）：算力 / 带宽，单位 FLOP/Byte
RIDGE_FP32 = THEORETICAL["fp32_tflops"] * 1e12 / (THEORETICAL["mem_bw_gbps"] * 1e9)


def _gflops(flops, seconds):
    return flops / seconds / 1e9  # GFLOP/s


def _bandwidth(bytes_moved, seconds):
    return bytes_moved / seconds / 1e9  # GB/s


def bench_memcpy(n_bytes=2**30):
    """访存型：纯数据搬运，算术强度≈0，应命中「显存带宽屋顶」。

    AI ≈ 0 → 完全 memory-bound，实测带宽应接近 896 GB/s 的理论带宽。
    """
    n = n_bytes // 4  # float32 元素个数
    a = torch.randn(n, device="cuda")
    b = torch.empty_like(a)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    b.copy_(a)  # 读 a + 写 b，共 2×n_bytes 字节
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    bw = _bandwidth(2 * n_bytes, dt)
    return {
        "kind": "memcpy",
        "ai": 0.0,
        "bw_gbps": bw,
        "roof": THEORETICAL["mem_bw_gbps"],
        "pct_of_roof": bw / THEORETICAL["mem_bw_gbps"] * 100,
        "note": "memory-bound: 应命中显存带宽屋顶",
    }


def bench_fp32_matmul(n=4096):
    """算力型（FP32）：大矩阵 GEMM，算术强度高，应命中「FP32 算力屋顶」。

    GEMM(M,K,N)：FLOPs = 2×M×K×N；访存 ≈ (M×K + K×N + M×N)×4 字节。
    n=4096 时 AI ≈ 2×4096³ / (3×4096²×4) ≈ 682 FLOP/Byte，远超拐点 61.8。
    """
    a = torch.randn(n, n, device="cuda")
    b = torch.randn(n, n, device="cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b  # FP32 GEMM
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    flops = 2 * n * n * n
    bytes_moved = 3 * n * n * 4  # 近似：A + B 读，C 写
    ai = flops / bytes_moved
    tflops = flops / dt / 1e12

    return {
        "kind": "fp32_matmul",
        "ai": ai,
        "tf": tflops,
        "roof": THEORETICAL["fp32_tflops"],
        "pct_of_roof": tflops / THEORETICAL["fp32_tflops"] * 100,
        "note": "compute-bound: 应命中 FP32 算力屋顶",
    }


def bench_fp16_matmul(n=4096):
    """算力型（FP16 Tensor Core）：应命中「FP16 算力屋顶」（约 110 TFLOPS）。

    注意：需要 torch 用 Tensor Core，矩阵尺寸应为 16 的倍数（Blackwell 为 16）。
    """
    a = torch.randn(n, n, device="cuda", dtype=torch.float16)
    b = torch.randn(n, n, device="cuda", dtype=torch.float16)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b  # FP16 GEMM，走 Tensor Core
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    flops = 2 * n * n * n
    bytes_moved = 3 * n * n * 2  # fp16 每个元素 2 字节
    ai = flops / bytes_moved
    tflops = flops / dt / 1e12

    return {
        "kind": "fp16_matmul",
        "ai": ai,
        "tf": tflops,
        "roof": THEORETICAL["fp16_tflops"],
        "pct_of_roof": tflops / THEORETICAL["fp16_tflops"] * 100,
        "note": "compute-bound: 应命中 FP16 Tensor Core 屋顶",
    }


def bench_fp32_reduce(n=2**26):
    """中等算术强度：逐元素运算（如 sum/relu 类），介于两屋顶之间。

    sum：读 n 个 float32（4n 字节），做 n 次加法（≈n FLOP），AI ≈ 0.25 FLOP/Byte，
    仍是 memory-bound，但比纯 memcpy 多一点算力。
    """
    a = torch.randn(n, device="cuda")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    s = a.sum()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    flops = n
    bytes_moved = n * 4
    ai = flops / bytes_moved
    bw = _bandwidth(bytes_moved, dt)

    return {
        "kind": "fp32_reduce",
        "ai": ai,
        "bw_gbps": bw,
        "roof": THEORETICAL["mem_bw_gbps"],
        "pct_of_roof": bw / THEORETICAL["mem_bw_gbps"] * 100,
        "note": "memory-bound: 低算术强度，应命中带宽屋顶",
    }


BENCHES = {
    "memcpy": bench_memcpy,
    "fp32": bench_fp32_matmul,
    "fp16": bench_fp16_matmul,
    "reduce": bench_fp32_reduce,
}


def print_roofline_context():
    print("=" * 72)
    print("Roofline 模型上下文 — RTX 5070 Ti (Blackwell, sm_120)")
    print("=" * 72)
    print(f"  理论 FP32 峰值算力 : {THEORETICAL['fp32_tflops']:.1f} TFLOPS")
    print(f"  理论 FP16 峰值算力 : {THEORETICAL['fp16_tflops']:.1f} TFLOPS (Tensor Core)")
    print(f"  理论显存带宽       : {THEORETICAL['mem_bw_gbps']:.1f} GB/s")
    print(f"  Ridge Point (FP32) : {RIDGE_FP32:.1f} FLOP/Byte")
    print("  AI < ridge → memory-bound；AI > ridge → compute-bound")
    print()


def run(kind):
    print_roofline_context()
    print(f"{'算子':<14}{'算术强度':>12}{'实测值':>16}{'理论峰值':>16}{'达成率':>10}  判定")
    print("-" * 72)

    for name in kind:
        r = BENCHES[name]()
        if "tf" in r:
            val_str = f"{r['tf']:.2f} TFLOPS"
            roof_str = f"{r['roof']:.2f} TFLOPS"
        else:
            val_str = f"{r['bw_gbps']:.1f} GB/s"
            roof_str = f"{r['roof']:.1f} GB/s"
        print(
            f"{r['kind']:<14}{r['ai']:>10.1f}{val_str:>16}{roof_str:>16}"
            f"{r['pct_of_roof']:>9.1f}%  {r['note']}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(description="Roofline 模型基准测试")
    parser.add_argument(
        "--kind",
        type=str,
        default="all",
        help="all / memcpy / fp32 / fp16 / reduce",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请确认装的是 CUDA 版 PyTorch")

    dev = torch.cuda.get_device_name(0)
    print(f"检测到设备: {dev}\n")

    if args.kind == "all":
        run(list(BENCHES.keys()))
    elif args.kind in BENCHES:
        run([args.kind])
    else:
        raise ValueError(f"未知 kind: {args.kind}，可选 {list(BENCHES.keys())}")


if __name__ == "__main__":
    main()
