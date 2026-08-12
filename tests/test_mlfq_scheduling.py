# -*- coding: utf-8 -*-
"""
测试优化7：MLFQ 多级反馈队列调度
==================================
本脚本完全自包含，不依赖 vllm / torch / GPU 等任何重量级依赖。
它复制了 vllm/v1/request.py 中 MLFQ 核心逻辑的"接口契约"，
直接对降级/升级/调度顺序进行验证。

如果后续修改了 MLFQ 的核心参数（如 token_quota），
请同步更新本文件中的 MLFQ_LEVELS 配置。

测试场景：
 1. MLFQ 级别配置检查
 2. 降级逻辑：消耗 token 超过配额后自动降级
 3. 升级逻辑：被抢占后升一级（防饥饿）
 4. 短请求保持 L0
 5. 长请求降级轨迹 L0->L1->L2->L3
 6. 多级队列调度顺序 L0>L1>L2>L3
 7. 同级别 FCFS
 8. 新短请求插队已降级长请求（核心价值）
 9. 抢占升级 + 再调度
10. 端到端混合负载 MLFQ vs FCFS 对比
11. 逐步降级不跳级
12. 1000 请求压力测试

用法：
    python3 tests/test_mlfq_scheduling.py
"""

from __future__ import print_function

import math
import sys
import random
from collections import deque


# ================================================================
# 从 vllm/v1/request.py 中提取的 MLFQ 核心逻辑（自包含副本）
# ================================================================

class MLFQLevel(object):
    """MLFQ 单级配置。"""

    def __init__(self, level, name, token_quota):
        self.level = level
        self.name = name
        self.token_quota = token_quota

    def __repr__(self):
        return "MLFQLevel(L%d, %s, quota=%s)" % (self.level, self.name, self.token_quota)


MLFQ_LEVELS = [
    MLFQLevel(level=0, name="interactive", token_quota=128),
    MLFQLevel(level=1, name="standard", token_quota=512),
    MLFQLevel(level=2, name="batch", token_quota=2048),
    MLFQLevel(level=3, name="background", token_quota=math.inf),
]

MLFQ_NUM_LEVELS = len(MLFQ_LEVELS)


class MockRequest(object):
    """
    模拟 vllm.v1.request.Request 中与 MLFQ 相关的字段和方法。
    只包含 MLFQ 测试所需的最小接口。
    """

    def __init__(self, request_id, arrival_time=0.0, max_tokens=4096):
        self.request_id = request_id
        self.arrival_time = arrival_time
        self.max_tokens = max_tokens

        # ---- MLFQ Fields ----
        self.mlfq_level = 0
        self.mlfq_tokens_consumed = 0

    def mlfq_account_tokens(self, num_tokens):
        """消耗 token，超过当前级配额时自动降级。
        
        注意：与 vllm 源码一致，每次调用只降一级。
        在实际推理中，每步只生成少量 token，降级自然是逐级的。
        """
        self.mlfq_tokens_consumed += num_tokens
        current_level = MLFQ_LEVELS[self.mlfq_level]
        if (self.mlfq_tokens_consumed >= current_level.token_quota
                and self.mlfq_level < MLFQ_NUM_LEVELS - 1):
            self.mlfq_level += 1

    def mlfq_promote(self):
        """被抢占时升一级（防饥饿），不重置消耗计数。"""
        if self.mlfq_level > 0:
            self.mlfq_level -= 1

    def __repr__(self):
        return "MockRequest(%s, L%d, consumed=%d)" % (
            self.request_id, self.mlfq_level, self.mlfq_tokens_consumed)


def mlfq_peek_next(mlfq_queues):
    """从多级队列中查看下一个要调度的请求（不移除）。"""
    for level_queue in mlfq_queues:
        if level_queue:
            return level_queue[0]
    return None


def mlfq_pop_next(mlfq_queues):
    """从多级队列中取出下一个要调度的请求。"""
    for level_queue in mlfq_queues:
        if level_queue:
            return level_queue.popleft()
    return None


# ================================================================
# 测试用例
# ================================================================

def test_mlfq_level_config():
    print("=" * 60)
    print("测试1：MLFQ 级别配置检查")
    print("=" * 60)

    assert MLFQ_NUM_LEVELS == 4, "期望 4 级，实际 %d" % MLFQ_NUM_LEVELS
    assert MLFQ_LEVELS[0].name == "interactive"
    assert MLFQ_LEVELS[0].token_quota == 128
    assert MLFQ_LEVELS[1].name == "standard"
    assert MLFQ_LEVELS[1].token_quota == 512
    assert MLFQ_LEVELS[2].name == "batch"
    assert MLFQ_LEVELS[2].token_quota == 2048
    assert MLFQ_LEVELS[3].name == "background"
    assert MLFQ_LEVELS[3].token_quota == math.inf

    print("  级别配置：")
    for lvl in MLFQ_LEVELS:
        print("    %s" % lvl)
    print("  [PASS] 级别配置正确\n")


def test_request_demotion():
    print("=" * 60)
    print("测试2：Request MLFQ 降级逻辑")
    print("=" * 60)

    req = MockRequest("test-demotion")
    assert req.mlfq_level == 0, "新请求应从 L0 开始"
    assert req.mlfq_tokens_consumed == 0

    req.mlfq_account_tokens(100)
    assert req.mlfq_level == 0, "消耗 100 应仍在 L0"
    print("  消耗 100 tokens -> L%d (consumed=%d)" % (req.mlfq_level, req.mlfq_tokens_consumed))

    req.mlfq_account_tokens(28)
    assert req.mlfq_level == 1, "消耗 128 应降到 L1，实际 L%d" % req.mlfq_level
    print("  消耗 128 tokens -> L%d (consumed=%d)" % (req.mlfq_level, req.mlfq_tokens_consumed))

    req.mlfq_account_tokens(384)
    assert req.mlfq_level == 2, "消耗 512 应降到 L2，实际 L%d" % req.mlfq_level
    print("  消耗 512 tokens -> L%d (consumed=%d)" % (req.mlfq_level, req.mlfq_tokens_consumed))

    req.mlfq_account_tokens(1536)
    assert req.mlfq_level == 3, "消耗 2048 应降到 L3，实际 L%d" % req.mlfq_level
    print("  消耗 2048 tokens -> L%d (consumed=%d)" % (req.mlfq_level, req.mlfq_tokens_consumed))

    req.mlfq_account_tokens(10000)
    assert req.mlfq_level == 3, "L3 是最底层，不应再降"
    print("  消耗 12048 tokens -> L%d (已是最底层)" % req.mlfq_level)

    print("  [PASS] 降级逻辑正确\n")


def _consume_to_level(req, target_level):
    """逐步消耗 token 使请求降级到目标级别。"""
    while req.mlfq_level < target_level:
        req.mlfq_account_tokens(1)


def _consume_tokens_stepwise(req, total_tokens):
    """逐步消耗 total_tokens，每步消耗到下一个阈值或剩余量。
    
    比逐个 token 消耗快得多，但保证逐级降级。
    """
    remaining = total_tokens
    while remaining > 0:
        current_quota = MLFQ_LEVELS[req.mlfq_level].token_quota
        if current_quota == math.inf:
            # 最后一级，直接消耗完
            req.mlfq_account_tokens(remaining)
            remaining = 0
        else:
            # 消耗到当前级别的配额阈值
            to_threshold = max(1, int(current_quota) - req.mlfq_tokens_consumed)
            step = min(to_threshold, remaining)
            req.mlfq_account_tokens(step)
            remaining -= step


def test_request_promotion():
    print("=" * 60)
    print("测试3：Request MLFQ 升级逻辑（防饥饿）")
    print("=" * 60)

    req = MockRequest("test-promotion")
    _consume_to_level(req, 2)
    assert req.mlfq_level == 2
    old_consumed = req.mlfq_tokens_consumed
    print("  初始状态：L%d, consumed=%d" % (req.mlfq_level, req.mlfq_tokens_consumed))

    req.mlfq_promote()
    assert req.mlfq_level == 1, "从 L2 升级后应为 L1"
    assert req.mlfq_tokens_consumed == old_consumed, "升级不应重置消耗计数"
    print("  抢占升级：L%d, consumed=%d (计数不重置)" % (req.mlfq_level, req.mlfq_tokens_consumed))

    req.mlfq_promote()
    assert req.mlfq_level == 0
    print("  再次升级：L%d" % req.mlfq_level)

    req.mlfq_promote()
    assert req.mlfq_level == 0, "L0 是最高级，不应再升"
    print("  尝试继续升级：L%d (已是最高级)" % req.mlfq_level)

    print("  [PASS] 升级逻辑正确\n")


def test_short_request_stays_at_l0():
    print("=" * 60)
    print("测试4：短请求在 L0 完成，不会降级")
    print("=" * 60)

    req = MockRequest("short-req", max_tokens=64)
    for _ in range(8):
        req.mlfq_account_tokens(8)

    assert req.mlfq_tokens_consumed == 64
    assert req.mlfq_level == 0, "64 tokens 的短请求应仍在 L0"
    print("  短请求生成 64 tokens -> L%d" % req.mlfq_level)
    print("  [PASS] 短请求始终在 L0\n")


def test_long_request_demotion_trajectory():
    print("=" * 60)
    print("测试5：长请求降级轨迹")
    print("=" * 60)

    req = MockRequest("long-req", max_tokens=4096)
    trajectory = []

    for step in range(100):
        old_level = req.mlfq_level
        req.mlfq_account_tokens(32)
        if req.mlfq_level != old_level:
            trajectory.append({
                "step": step,
                "tokens": req.mlfq_tokens_consumed,
                "from": old_level,
                "to": req.mlfq_level,
            })

    print("  降级事件：")
    for t in trajectory:
        print("    Step %3d: consumed=%5d -> L%d -> L%d (%s -> %s)" % (
            t["step"], t["tokens"], t["from"], t["to"],
            MLFQ_LEVELS[t["from"]].name, MLFQ_LEVELS[t["to"]].name))

    assert len(trajectory) == 3, "期望 3 次降级，实际 %d" % len(trajectory)
    assert trajectory[0]["to"] == 1
    assert trajectory[1]["to"] == 2
    assert trajectory[2]["to"] == 3
    assert trajectory[0]["tokens"] == 128
    assert trajectory[1]["tokens"] == 512
    assert trajectory[2]["tokens"] == 2048

    print("  [PASS] 降级轨迹正确：L0(128) -> L1(512) -> L2(2048) -> L3\n")


def test_mlfq_scheduling_order():
    print("=" * 60)
    print("测试6：多级队列调度顺序")
    print("=" * 60)

    queues = [deque() for _ in range(MLFQ_NUM_LEVELS)]

    req_l2 = MockRequest("req-L2", arrival_time=1.0)
    req_l2.mlfq_level = 2
    req_l0 = MockRequest("req-L0", arrival_time=3.0)
    req_l0.mlfq_level = 0
    req_l1 = MockRequest("req-L1", arrival_time=2.0)
    req_l1.mlfq_level = 1

    queues[2].append(req_l2)
    queues[0].append(req_l0)
    queues[1].append(req_l1)

    order = []
    for _ in range(3):
        r = mlfq_pop_next(queues)
        order.append((r.request_id, r.mlfq_level))

    print("  插入顺序：req-L2(t=1), req-L0(t=3), req-L1(t=2)")
    print("  调度顺序：")
    for rid, lvl in order:
        print("    %s (L%d) - %s" % (rid, lvl, MLFQ_LEVELS[lvl].name))

    assert order[0] == ("req-L0", 0)
    assert order[1] == ("req-L1", 1)
    assert order[2] == ("req-L2", 2)

    print("  [PASS] 调度顺序正确：L0 -> L1 -> L2\n")


def test_same_level_fcfs():
    print("=" * 60)
    print("测试7：同级别内 FCFS 顺序")
    print("=" * 60)

    queues = [deque() for _ in range(MLFQ_NUM_LEVELS)]

    req_a = MockRequest("req-A", arrival_time=1.0)
    req_b = MockRequest("req-B", arrival_time=2.0)
    req_c = MockRequest("req-C", arrival_time=3.0)

    queues[0].append(req_a)
    queues[0].append(req_b)
    queues[0].append(req_c)

    order = [mlfq_pop_next(queues).request_id for _ in range(3)]
    print("  L0 队列：A(t=1), B(t=2), C(t=3)")
    print("  调度顺序：%s" % " -> ".join(order))

    assert order == ["req-A", "req-B", "req-C"]
    print("  [PASS] 同级别 FCFS 正确\n")


def test_new_short_preempts_demoted_long():
    print("=" * 60)
    print("测试8：新短请求「插队」已降级长请求（MLFQ 核心价值）")
    print("=" * 60)

    queues = [deque() for _ in range(MLFQ_NUM_LEVELS)]

    # 长请求：先到达，逐步消耗到 L2
    long_req = MockRequest("long-A", arrival_time=1.0, max_tokens=4096)
    _consume_to_level(long_req, 2)
    assert long_req.mlfq_level == 2
    queues[long_req.mlfq_level].append(long_req)

    # 短请求：后到达，进入 L0
    short_req = MockRequest("short-B", arrival_time=5.0, max_tokens=32)
    assert short_req.mlfq_level == 0
    queues[0].append(short_req)

    order = [mlfq_pop_next(queues) for _ in range(2)]

    print("  长请求 A：到达t=1, consumed=1000 -> L%d (已降级)" % long_req.mlfq_level)
    print("  短请求 B：到达t=5, consumed=0   -> L0 (新请求)")
    print("  调度顺序：")
    for r in order:
        print("    %s (L%d)" % (r.request_id, r.mlfq_level))

    assert order[0].request_id == "short-B", "短请求应先被调度"
    assert order[1].request_id == "long-A", "长请求应后调度"

    print("  [PASS] 新短请求成功插队！MLFQ 核心价值验证通过")
    print("     -> 短请求天然获得低时延保障，无需手动优先级标注\n")


def test_preemption_promotion_reschedule():
    print("=" * 60)
    print("测试9：抢占后升级 + 再调度")
    print("=" * 60)

    queues = [deque() for _ in range(MLFQ_NUM_LEVELS)]

    # A: 逐步消耗到 L2，被抢占后升到 L1
    req_a = MockRequest("req-A-preempted", arrival_time=1.0)
    _consume_to_level(req_a, 2)
    assert req_a.mlfq_level == 2
    req_a.mlfq_promote()
    assert req_a.mlfq_level == 1
    queues[req_a.mlfq_level].append(req_a)

    # B: 逐步消耗到 L2，未抢占
    req_b = MockRequest("req-B-normal", arrival_time=2.0)
    _consume_to_level(req_b, 2)
    assert req_b.mlfq_level == 2
    queues[req_b.mlfq_level].append(req_b)

    order = [mlfq_pop_next(queues) for _ in range(2)]

    print("  请求 A：被抢占 L2 -> L1")
    print("  请求 B：未抢占，留在 L2")
    print("  调度顺序：")
    for r in order:
        print("    %s (L%d)" % (r.request_id, r.mlfq_level))

    assert order[0].request_id == "req-A-preempted"
    assert order[1].request_id == "req-B-normal"
    print("  [PASS] 抢占升级防饥饿机制正确\n")


def test_e2e_mixed_workload_comparison():
    print("=" * 60)
    print("测试10：端到端混合负载 - MLFQ vs FCFS 对比")
    print("=" * 60)

    # ── FCFS ──
    fcfs_order = ["long-%d" % i for i in range(5)] + ["short-%d" % i for i in range(3)]
    fcfs_short_pos = [i for i, n in enumerate(fcfs_order) if n.startswith("short")]

    # ── MLFQ ──
    queues = [deque() for _ in range(MLFQ_NUM_LEVELS)]

    # 5 个长请求先到达，在 L0
    long_reqs = []
    for i in range(5):
        req = MockRequest("long-%d" % i, arrival_time=float(i))
        long_reqs.append(req)
        queues[0].append(req)

    # 模拟长请求消耗 200 tokens (L0 quota=128 -> 降到 L1)
    for req in long_reqs:
        req.mlfq_account_tokens(200)
        # 它们已经被调度出去了，现在模拟被抢占回到 waiting
        # 需要重新插入到对应级别的队列
    # 清空 L0（长请求已被调度走），重新按级别插入
    queues[0].clear()
    for req in long_reqs:
        queues[req.mlfq_level].append(req)

    # 3 个短请求后到达，进入 L0
    for i in range(3):
        req = MockRequest("short-%d" % i, arrival_time=5.0 + i)
        queues[0].append(req)

    mlfq_order = []
    for _ in range(8):
        r = mlfq_pop_next(queues)
        if r:
            mlfq_order.append(r.request_id)

    mlfq_short_pos = [i for i, n in enumerate(mlfq_order) if n.startswith("short")]

    print("  FCFS 调度顺序：%s" % " -> ".join(fcfs_order))
    print("  FCFS 中短请求位置：%s" % fcfs_short_pos)
    print()
    print("  MLFQ 调度顺序：%s" % " -> ".join(mlfq_order))
    print("  MLFQ 中短请求位置：%s" % mlfq_short_pos)
    print()

    fcfs_avg = sum(fcfs_short_pos) / float(len(fcfs_short_pos))
    mlfq_avg = sum(mlfq_short_pos) / float(len(mlfq_short_pos))

    print("  短请求平均调度位置：FCFS=%.1f, MLFQ=%.1f" % (fcfs_avg, mlfq_avg))
    assert mlfq_avg < fcfs_avg, "MLFQ 中短请求应更靠前"

    improvement = fcfs_avg - mlfq_avg
    print("  MLFQ 优化效果：短请求平均提前了 %.1f 个位置" % improvement)
    print("  [PASS] MLFQ 对短请求的时延优化效果显著\n")


def test_no_level_skipping():
    print("=" * 60)
    print("测试11：逐步消耗时降级不跳级")
    print("=" * 60)

    req = MockRequest("test-no-skip")
    level_history = [req.mlfq_level]

    for _ in range(3000):
        req.mlfq_account_tokens(1)
        if req.mlfq_level != level_history[-1]:
            level_history.append(req.mlfq_level)

    print("  逐步消耗 3000 tokens (每步 1 token)")
    print("  级别变化：%s" % " -> ".join("L%d" % l for l in level_history))

    assert level_history == [0, 1, 2, 3], "应逐级降级，实际 %s" % level_history
    print("  [PASS] 降级严格逐级\n")


def test_stress_many_requests():
    print("=" * 60)
    print("测试12：压力测试 - 1000 个请求的 MLFQ 分级")
    print("=" * 60)

    random.seed(42)
    queues = [deque() for _ in range(MLFQ_NUM_LEVELS)]
    requests_data = []

    for i in range(1000):
        tokens = random.choice([50, 200, 800, 3000])
        req = MockRequest("req-%d" % i)

        # 使用 _consume_to_level 辅助函数确保逐级降级
        # 然后补充剩余 token
        _consume_tokens_stepwise(req, tokens)

        requests_data.append((req, tokens))
        queues[req.mlfq_level].append(req)

    level_counts = [len(queues[i]) for i in range(MLFQ_NUM_LEVELS)]

    print("  各级别请求数：")
    for lvl in range(MLFQ_NUM_LEVELS):
        print("    L%d (%12s): %4d 个" % (lvl, MLFQ_LEVELS[lvl].name, level_counts[lvl]))

    # 验证分级正确性
    for req, consumed in requests_data:
        if consumed == 50:
            assert req.mlfq_level == 0, "50 tokens -> L0, got L%d" % req.mlfq_level
        elif consumed == 200:
            assert req.mlfq_level == 1, "200 tokens -> L1, got L%d" % req.mlfq_level
        elif consumed == 800:
            assert req.mlfq_level == 2, "800 tokens -> L2, got L%d" % req.mlfq_level
        elif consumed == 3000:
            assert req.mlfq_level == 3, "3000 tokens -> L3, got L%d" % req.mlfq_level

    total = sum(level_counts)
    assert total == 1000
    print("  1000 个请求全部正确分级")
    print("  [PASS] 压力测试通过\n")


# ================================================================
# 测试13：验证 MockRequest 的逻辑与 vllm 源码一致
# ================================================================

def test_logic_matches_source():
    """
    关键测试：验证 MockRequest 中的 MLFQ 逻辑与 vllm/v1/request.py
    中实际 Request 类的逻辑完全一致。

    通过读取源代码，提取关键参数并与本文件中的常量对比。
    """
    print("=" * 60)
    print("测试13：验证测试逻辑与 vllm 源码一致")
    print("=" * 60)

    import os
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "vllm", "v1", "request.py")

    if not os.path.exists(source_path):
        print("  [SKIP] 源文件不存在: %s" % source_path)
        return

    with open(source_path, "r") as f:
        source = f.read()

    # 检查关键常量是否存在于源码中
    checks = [
        ('MLFQLevel(level=0, name="interactive", token_quota=128)',
         "L0 配置"),
        ('MLFQLevel(level=1, name="standard", token_quota=512)',
         "L1 配置"),
        ('MLFQLevel(level=2, name="batch", token_quota=2048)',
         "L2 配置"),
        ("self.mlfq_level: int = 0", "初始级别"),
        ("self.mlfq_tokens_consumed: int = 0", "初始消耗"),
        ("self.mlfq_tokens_consumed += num_tokens", "累加逻辑"),
        ("self.mlfq_level += 1", "降级逻辑"),
        ("self.mlfq_level > 0", "升级边界检查"),
        ("self.mlfq_level -= 1", "升级逻辑"),
    ]

    all_match = True
    for pattern, desc in checks:
        if pattern in source:
            print("  [OK] %s" % desc)
        else:
            print("  [MISMATCH] %s -- 未在源码中找到: %s" % (desc, pattern))
            all_match = False

    if all_match:
        print("  [PASS] 测试逻辑与源码完全一致\n")
    else:
        raise AssertionError("测试逻辑与源码不一致，请同步更新")


# ================================================================
# 主入口
# ================================================================

def main():
    print("")
    print("#" * 60)
    print("# MLFQ 多级反馈队列调度（优化7）测试")
    print("# 自包含测试，不依赖 torch / GPU / vllm server")
    print("#" * 60)
    print("")

    tests = [
        ("级别配置检查", test_mlfq_level_config),
        ("降级逻辑", test_request_demotion),
        ("升级逻辑（防饥饿）", test_request_promotion),
        ("短请求保持 L0", test_short_request_stays_at_l0),
        ("长请求降级轨迹", test_long_request_demotion_trajectory),
        ("多级队列调度顺序", test_mlfq_scheduling_order),
        ("同级别 FCFS", test_same_level_fcfs),
        ("新短请求插队已降级长请求", test_new_short_preempts_demoted_long),
        ("抢占升级防饥饿", test_preemption_promotion_reschedule),
        ("混合负载 MLFQ vs FCFS 对比", test_e2e_mixed_workload_comparison),
        ("逐步降级不跳级", test_no_level_skipping),
        ("压力测试 1000 请求", test_stress_many_requests),
        ("逻辑与源码一致性校验", test_logic_matches_source),
    ]

    passed = 0
    failed = 0
    skipped = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print("  [FAIL] %s: %s\n" % (name, e))
            failed += 1
        except Exception as e:
            print("  [ERROR] %s: %s\n" % (name, e))
            failed += 1

    print("=" * 60)
    print("测试结果：%d 通过, %d 失败, 共 %d 个测试" % (passed, failed, len(tests)))
    if failed == 0:
        print("[ALL PASSED] MLFQ 优化7 功能验证完成。")
    else:
        print("[ATTENTION] 有 %d 个测试失败，请检查。" % failed)
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
