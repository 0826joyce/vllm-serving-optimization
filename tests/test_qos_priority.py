"""
测试优化1：QoS 优先级调度
=========================
用法：先启动 vllm server（加 --scheduling-policy priority），然后运行本脚本。

测试场景：
1. 基础测试：不同 priority 的请求是否按优先级顺序完成
2. 饥饿防护测试：低优先级请求不会被永远饿死
3. 默认行为测试：不传 priority 时行为正常（默认 priority=0）
"""

import asyncio
import time
import aiohttp
import json

BASE_URL = "http://localhost:8000"
MODEL = "facebook/opt-125m"


async def send_completion(session, prompt, priority, request_id):
    """发送一个 completion 请求，返回 (request_id, priority, 完成时间)"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 50,
        "priority": priority,
    }
    start = time.time()
    try:
        async with session.post(
            f"{BASE_URL}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            result = await resp.json()
            elapsed = time.time() - start
            if resp.status == 200:
                return {
                    "id": request_id,
                    "priority": priority,
                    "elapsed": elapsed,
                    "status": "ok",
                    "text": result["choices"][0]["text"][:50],
                }
            else:
                return {
                    "id": request_id,
                    "priority": priority,
                    "elapsed": elapsed,
                    "status": "error",
                    "detail": result.get("detail", str(result)),
                }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "id": request_id,
            "priority": priority,
            "elapsed": elapsed,
            "status": "exception",
            "detail": str(e),
        }


async def test_basic_priority():
    """
    测试1：基础优先级排序
    同时发送多个请求，priority 值越小应越先被处理。
    """
    print("=" * 60)
    print("测试1：基础优先级排序")
    print("  发送 6 个请求，priority 分别为 10, 5, 0, -5, -10, 20")
    print("  预期：priority 小的先完成（或完成时间更短）")
    print("=" * 60)

    # 用较长的 prompt 增加调度差异的可观测性
    long_prompt = "Once upon a time in a land far away, " * 20

    requests_config = [
        (long_prompt, 10, "req-p10"),
        (long_prompt, 5, "req-p5"),
        (long_prompt, 0, "req-p0"),
        (long_prompt, -5, "req-p-5"),
        (long_prompt, -10, "req-p-10"),
        (long_prompt, 20, "req-p20"),
    ]

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_completion(session, prompt, pri, rid)
            for prompt, pri, rid in requests_config
        ]
        results = await asyncio.gather(*tasks)

    # 按完成时间排序
    results.sort(key=lambda r: r["elapsed"])
    print("\n结果（按完成时间排序）：")
    print(f"{'请求ID':<12} {'priority':<10} {'耗时(s)':<10} {'状态':<8}")
    print("-" * 45)
    for r in results:
        print(f"{r['id']:<12} {r['priority']:<10} {r['elapsed']:<10.3f} {r['status']:<8}")

    print()
    return results


async def test_starvation_protection():
    """
    测试2：饥饿防护
    先发送一个低优先级请求（priority=100），然后持续发送高优先级请求。
    低优先级请求应该在一段时间后仍然能完成（不会被永远饿死）。
    """
    print("=" * 60)
    print("测试2：饥饿防护")
    print("  先发低优先级(100)请求，再持续发高优先级(-10)请求")
    print("  预期：低优先级请求最终仍然能完成")
    print("=" * 60)

    long_prompt = "Tell me a story about " * 30

    async with aiohttp.ClientSession() as session:
        # 先发低优先级
        low_pri_task = asyncio.create_task(
            send_completion(session, long_prompt, 100, "low-pri-100")
        )

        # 稍等一下确保低优先级请求先到达
        await asyncio.sleep(0.1)

        # 发送一批高优先级请求
        high_pri_tasks = [
            asyncio.create_task(
                send_completion(session, long_prompt, -10, f"high-pri-{i}")
            )
            for i in range(5)
        ]

        # 等待所有完成
        all_results = await asyncio.gather(low_pri_task, *high_pri_tasks)

    print("\n结果：")
    print(f"{'请求ID':<16} {'priority':<10} {'耗时(s)':<10} {'状态':<8}")
    print("-" * 50)
    for r in all_results:
        print(f"{r['id']:<16} {r['priority']:<10} {r['elapsed']:<10.3f} {r['status']:<8}")

    low_result = all_results[0]
    if low_result["status"] == "ok":
        print("\n✅ 低优先级请求成功完成！饥饿防护生效。")
    else:
        print(f"\n⚠️ 低优先级请求状态: {low_result['status']}")

    print()
    return all_results


async def test_default_priority():
    """
    测试3：默认行为（不传 priority）
    不传 priority 参数时应该默认为 0，行为正常。
    """
    print("=" * 60)
    print("测试3：默认行为（不传 priority）")
    print("=" * 60)

    async with aiohttp.ClientSession() as session:
        # 不传 priority
        payload = {
            "model": MODEL,
            "prompt": "Hello, how are you?",
            "max_tokens": 20,
        }
        start = time.time()
        async with session.post(
            f"{BASE_URL}/v1/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            result = await resp.json()
            elapsed = time.time() - start

        if resp.status == 200:
            print(f"✅ 默认 priority 请求成功，耗时: {elapsed:.3f}s")
            print(f"   生成文本: {result['choices'][0]['text'][:80]}")
        else:
            print(f"❌ 请求失败: {result}")

    print()


async def test_prompt_length_adjustment():
    """
    测试4：Prompt 长度调整
    同 priority 下，短 prompt 应该比长 prompt 更快完成。
    """
    print("=" * 60)
    print("测试4：Prompt 长度对优先级的影响")
    print("  同 priority=0，短/中/长 prompt 的完成时间对比")
    print("=" * 60)

    short_prompt = "Hi"
    medium_prompt = "Tell me about " * 100  # ~200 tokens
    long_prompt = "Explain in detail " * 500  # ~1000 tokens

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_completion(session, short_prompt, 0, "short-prompt"),
            send_completion(session, medium_prompt, 0, "medium-prompt"),
            send_completion(session, long_prompt, 0, "long-prompt"),
        ]
        results = await asyncio.gather(*tasks)

    results.sort(key=lambda r: r["elapsed"])
    print("\n结果（按完成时间排序）：")
    print(f"{'请求ID':<16} {'priority':<10} {'耗时(s)':<10} {'状态':<8}")
    print("-" * 50)
    for r in results:
        print(f"{r['id']:<16} {r['priority']:<10} {r['elapsed']:<10.3f} {r['status']:<8}")

    print()
    return results


async def main():
    print(f"\n{'#' * 60}")
    print(f"# QoS 优先级调度（优化1）测试")
    print(f"# 服务地址: {BASE_URL}")
    print(f"# 模型: {MODEL}")
    print(f"{'#' * 60}\n")

    # 先检查服务是否可用
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{BASE_URL}/v1/models",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    print("✅ 服务已就绪\n")
                else:
                    print(f"❌ 服务返回 {resp.status}，请检查服务是否正常启动")
                    return
    except Exception as e:
        print(f"❌ 无法连接到服务: {e}")
        print("请先启动 vllm server：")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
        print("  python -m vllm.entrypoints.openai.api_server \\")
        print("    --model facebook/opt-125m \\")
        print("    --gpu-memory-utilization 0.5 \\")
        print("    --enforce-eager \\")
        print("    --scheduling-policy priority")
        return

    await test_default_priority()
    await test_basic_priority()
    await test_prompt_length_adjustment()
    await test_starvation_protection()

    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
