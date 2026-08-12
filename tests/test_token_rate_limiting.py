#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Token Rate Limiting (Token Bucket) Unit Tests
==============================================

Tests for Optimization 4: Per-request token generation rate control.

This test script is completely self-contained — it does NOT import any vllm
modules (to avoid torch/CUDA dependencies).  Instead, it reproduces the core
TokenRateLimiter logic and the scheduler's rate-limiting integration as
"contract tests" that verify the design invariants.

Run:
    python3 tests/test_token_rate_limiting.py
"""

from __future__ import print_function

import math
import random
import sys
import time


# ======================================================================
# Reproduce core classes (mirrors vllm/v1/request.py TokenRateLimiter)
# ======================================================================

class TokenRateLimiter(object):
    """Per-request token bucket rate limiter."""

    DEFAULT_RATE_HIGH = math.inf
    DEFAULT_RATE_NORMAL = 64.0
    DEFAULT_RATE_LOW = 16.0
    DEFAULT_BURST = 128

    LOAD_THRESHOLD_MODERATE = 0.5
    LOAD_THRESHOLD_HIGH = 0.8

    def __init__(self, rate=math.inf, burst=128):
        self.rate = rate
        self.burst = burst
        self.tokens = float(burst)

    def refill(self):
        if self.rate == math.inf:
            self.tokens = float(self.burst)
        else:
            self.tokens = min(self.tokens + self.rate, float(self.burst))

    def consume(self, requested):
        if self.rate == math.inf:
            return requested
        allowed = min(requested, max(0, int(self.tokens)))
        self.tokens -= allowed
        return allowed

    def available(self):
        if self.rate == math.inf:
            return 2**31
        return max(0, int(self.tokens))

    def set_rate(self, rate, burst=None):
        self.rate = rate
        if burst is not None:
            self.burst = burst

    def is_limited(self):
        return self.rate != math.inf


# ---- MLFQ Level Config ----

class MLFQLevel(object):
    def __init__(self, level, name, token_quota):
        self.level = level
        self.name = name
        self.token_quota = token_quota

MLFQ_LEVELS = [
    MLFQLevel(level=0, name="interactive", token_quota=128),
    MLFQLevel(level=1, name="standard", token_quota=512),
    MLFQLevel(level=2, name="batch", token_quota=2048),
    MLFQLevel(level=3, name="background", token_quota=math.inf),
]
MLFQ_NUM_LEVELS = len(MLFQ_LEVELS)


# ---- Mock Request ----

class MockRequest(object):
    def __init__(self, request_id, mlfq_level=0, effective_priority=0.0):
        self.request_id = request_id
        self.mlfq_level = mlfq_level
        self.mlfq_tokens_consumed = 0
        self._effective_priority = effective_priority
        self.rate_limiter = TokenRateLimiter()

    @property
    def effective_priority(self):
        return self._effective_priority

    def mlfq_account_tokens(self, num_tokens):
        self.mlfq_tokens_consumed += num_tokens
        current_level = MLFQ_LEVELS[self.mlfq_level]
        if (self.mlfq_tokens_consumed >= current_level.token_quota
                and self.mlfq_level < MLFQ_NUM_LEVELS - 1):
            self.mlfq_level += 1


# ---- Scheduler helpers (mirrors scheduler.py logic) ----

def get_priority_tier(request, enable_mlfq=True, enable_qos=False):
    if enable_mlfq:
        if request.mlfq_level <= 1:
            return "HIGH"
        elif request.mlfq_level == 2:
            return "NORMAL"
        else:
            return "LOW"
    elif enable_qos:
        ep = request.effective_priority
        if ep < 0:
            return "HIGH"
        elif ep < 5:
            return "NORMAL"
        else:
            return "LOW"
    else:
        return "HIGH"


def update_rate_limiters(running, max_num_running):
    """Simulate scheduler's _update_rate_limiters."""
    if not running:
        return
    load = len(running) / max(1, max_num_running)
    for request in running:
        rl = request.rate_limiter
        rl.refill()
        tier = get_priority_tier(request)
        if load < TokenRateLimiter.LOAD_THRESHOLD_MODERATE:
            rl.set_rate(math.inf)
        elif load < TokenRateLimiter.LOAD_THRESHOLD_HIGH:
            if tier == "HIGH":
                rl.set_rate(math.inf)
            elif tier == "NORMAL":
                rl.set_rate(TokenRateLimiter.DEFAULT_RATE_NORMAL)
            else:
                rl.set_rate(TokenRateLimiter.DEFAULT_RATE_LOW)
        else:
            if tier == "HIGH":
                rl.set_rate(math.inf)
            elif tier == "NORMAL":
                rl.set_rate(TokenRateLimiter.DEFAULT_RATE_LOW * 2)
            else:
                rl.set_rate(max(1.0, TokenRateLimiter.DEFAULT_RATE_LOW / 2))


# ======================================================================
# Tests
# ======================================================================

passed = 0
failed = 0


def check(condition, msg=""):
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print("  [FAIL] %s" % msg)


def test_token_bucket_basic():
    print("=" * 60)
    print("Test 1: Token bucket basic operations")
    print("=" * 60)

    rl = TokenRateLimiter(rate=10.0, burst=50)
    print("  Initial: rate=10, burst=50, tokens=%.1f" % rl.tokens)

    # Initial bucket is full (burst)
    check(rl.tokens == 50.0, "Initial tokens should be burst (50)")
    check(rl.is_limited(), "rate=10 should be limited")

    # Consume 30 tokens
    allowed = rl.consume(30)
    check(allowed == 30, "Should allow 30 from a bucket of 50")
    check(abs(rl.tokens - 20.0) < 0.01, "Should have 20 left")
    print("  After consuming 30: tokens=%.1f" % rl.tokens)

    # Consume more than available
    allowed = rl.consume(100)
    check(allowed == 20, "Should only allow 20 (remaining)")
    check(rl.tokens <= 0, "Bucket should be empty")
    print("  After consuming 100 (only 20 available): allowed=%d, tokens=%.1f"
          % (allowed, rl.tokens))

    # Refill
    rl.refill()
    check(abs(rl.tokens - 10.0) < 0.01, "After refill should have 10 (rate)")
    print("  After refill: tokens=%.1f" % rl.tokens)

    # Multiple refills should cap at burst
    for _ in range(10):
        rl.refill()
    check(abs(rl.tokens - 50.0) < 0.01, "Should cap at burst (50)")
    print("  After 10 refills: tokens=%.1f (capped at burst)" % rl.tokens)
    print("  PASSED\n")


def test_unlimited_rate():
    print("=" * 60)
    print("Test 2: Unlimited rate (high-priority)")
    print("=" * 60)

    rl = TokenRateLimiter(rate=math.inf, burst=128)
    check(not rl.is_limited(), "rate=inf should not be limited")

    # Should always allow any amount
    for amount in [1, 100, 10000, 1000000]:
        allowed = rl.consume(amount)
        check(allowed == amount,
              "Unlimited should allow %d, got %d" % (amount, allowed))
    print("  Unlimited rate allows any amount: PASSED")

    # available() should return huge number
    check(rl.available() >= 2**30, "available() should be effectively unlimited")
    print("  PASSED\n")


def test_dynamic_rate_adjustment():
    print("=" * 60)
    print("Test 3: Dynamic rate adjustment")
    print("=" * 60)

    rl = TokenRateLimiter(rate=10.0, burst=50)
    check(rl.rate == 10.0, "Initial rate should be 10")

    rl.set_rate(100.0)
    check(rl.rate == 100.0, "Rate should be updated to 100")

    rl.set_rate(math.inf)
    check(not rl.is_limited(), "After set_rate(inf), should not be limited")

    rl.set_rate(5.0, burst=20)
    check(rl.rate == 5.0, "Rate should be 5")
    check(rl.burst == 20, "Burst should be updated to 20")

    print("  Dynamic rate adjustment: PASSED\n")


def test_refill_does_not_exceed_burst():
    print("=" * 60)
    print("Test 4: Refill never exceeds burst capacity")
    print("=" * 60)

    rl = TokenRateLimiter(rate=1000.0, burst=50)
    # Even with very high rate, refill should not exceed burst
    rl.tokens = 0
    rl.refill()
    check(rl.tokens <= 50.0,
          "Refill should not exceed burst: got %.1f" % rl.tokens)
    print("  After refill with rate=1000, burst=50: tokens=%.1f" % rl.tokens)

    rl.refill()
    check(rl.tokens <= 50.0,
          "Second refill should still not exceed burst: got %.1f" % rl.tokens)
    print("  PASSED\n")


def test_priority_tier_classification():
    print("=" * 60)
    print("Test 5: Priority tier classification (MLFQ-based)")
    print("=" * 60)

    # MLFQ L0 -> HIGH
    req0 = MockRequest("r0", mlfq_level=0)
    check(get_priority_tier(req0) == "HIGH", "L0 should be HIGH")
    print("  L0 -> %s" % get_priority_tier(req0))

    # MLFQ L1 -> HIGH
    req1 = MockRequest("r1", mlfq_level=1)
    check(get_priority_tier(req1) == "HIGH", "L1 should be HIGH")
    print("  L1 -> %s" % get_priority_tier(req1))

    # MLFQ L2 -> NORMAL
    req2 = MockRequest("r2", mlfq_level=2)
    check(get_priority_tier(req2) == "NORMAL", "L2 should be NORMAL")
    print("  L2 -> %s" % get_priority_tier(req2))

    # MLFQ L3 -> LOW
    req3 = MockRequest("r3", mlfq_level=3)
    check(get_priority_tier(req3) == "LOW", "L3 should be LOW")
    print("  L3 -> %s" % get_priority_tier(req3))

    print("  PASSED\n")


def test_priority_tier_qos_fallback():
    print("=" * 60)
    print("Test 6: Priority tier classification (QoS fallback)")
    print("=" * 60)

    # effective_priority < 0 -> HIGH
    req_h = MockRequest("rh", effective_priority=-2.0)
    check(get_priority_tier(req_h, enable_mlfq=False, enable_qos=True) == "HIGH",
          "ep=-2 should be HIGH")

    # 0 <= effective_priority < 5 -> NORMAL
    req_n = MockRequest("rn", effective_priority=3.0)
    check(get_priority_tier(req_n, enable_mlfq=False, enable_qos=True) == "NORMAL",
          "ep=3 should be NORMAL")

    # effective_priority >= 5 -> LOW
    req_l = MockRequest("rl", effective_priority=10.0)
    check(get_priority_tier(req_l, enable_mlfq=False, enable_qos=True) == "LOW",
          "ep=10 should be LOW")

    # No priority info -> all HIGH (no limiting)
    req_x = MockRequest("rx")
    check(get_priority_tier(req_x, enable_mlfq=False, enable_qos=False) == "HIGH",
          "No priority -> HIGH")

    print("  QoS fallback: PASSED\n")


def test_load_based_rate_assignment_idle():
    print("=" * 60)
    print("Test 7: Load-based rate assignment — idle system (<50%%)")
    print("=" * 60)

    max_running = 100
    # 20 requests = 20% load (idle)
    running = []
    for i in range(20):
        level = i % 4
        running.append(MockRequest("r%d" % i, mlfq_level=level))

    update_rate_limiters(running, max_running)

    all_unlimited = all(not r.rate_limiter.is_limited() for r in running)
    check(all_unlimited, "All requests should be unlimited when idle")
    print("  Load=20%%: all unlimited = %s" % all_unlimited)
    print("  PASSED\n")


def test_load_based_rate_assignment_moderate():
    print("=" * 60)
    print("Test 8: Load-based rate assignment — moderate load (50-80%%)")
    print("=" * 60)

    max_running = 100
    # 60 requests = 60% load
    running = []
    # 15 each: L0, L1, L2, L3
    for i in range(15):
        running.append(MockRequest("h1-%d" % i, mlfq_level=0))
        running.append(MockRequest("h2-%d" % i, mlfq_level=1))
        running.append(MockRequest("n-%d" % i, mlfq_level=2))
        running.append(MockRequest("l-%d" % i, mlfq_level=3))

    update_rate_limiters(running, max_running)

    for r in running:
        tier = get_priority_tier(r)
        if tier == "HIGH":
            check(not r.rate_limiter.is_limited(),
                  "HIGH tier should be unlimited at moderate load")
        elif tier == "NORMAL":
            check(r.rate_limiter.is_limited(),
                  "NORMAL tier should be limited at moderate load")
            check(r.rate_limiter.rate == TokenRateLimiter.DEFAULT_RATE_NORMAL,
                  "NORMAL rate should be %s" % TokenRateLimiter.DEFAULT_RATE_NORMAL)
        else:
            check(r.rate_limiter.is_limited(),
                  "LOW tier should be limited at moderate load")
            check(r.rate_limiter.rate == TokenRateLimiter.DEFAULT_RATE_LOW,
                  "LOW rate should be %s" % TokenRateLimiter.DEFAULT_RATE_LOW)

    print("  Load=60%%: HIGH=unlimited, NORMAL=64, LOW=16")
    print("  PASSED\n")


def test_load_based_rate_assignment_high():
    print("=" * 60)
    print("Test 9: Load-based rate assignment — high load (>80%%)")
    print("=" * 60)

    max_running = 100
    # 85 requests = 85% load
    running = []
    for i in range(20):
        running.append(MockRequest("h-%d" % i, mlfq_level=0))
    for i in range(30):
        running.append(MockRequest("n-%d" % i, mlfq_level=2))
    for i in range(35):
        running.append(MockRequest("l-%d" % i, mlfq_level=3))

    update_rate_limiters(running, max_running)

    for r in running:
        tier = get_priority_tier(r)
        if tier == "HIGH":
            check(not r.rate_limiter.is_limited(),
                  "HIGH tier should be unlimited even at high load")
        elif tier == "NORMAL":
            expected = TokenRateLimiter.DEFAULT_RATE_LOW * 2
            check(r.rate_limiter.rate == expected,
                  "NORMAL rate at high load should be %s, got %s"
                  % (expected, r.rate_limiter.rate))
        else:
            expected = max(1.0, TokenRateLimiter.DEFAULT_RATE_LOW / 2)
            check(r.rate_limiter.rate == expected,
                  "LOW rate at high load should be %s, got %s"
                  % (expected, r.rate_limiter.rate))

    print("  Load=85%%: HIGH=unlimited, NORMAL=32, LOW=8")
    print("  PASSED\n")


def test_rate_limiting_effect_on_token_budget():
    """Core value test: demonstrate that rate limiting preserves budget
    for high-priority requests."""
    print("=" * 60)
    print("Test 10: Rate limiting preserves token budget for high-prio")
    print("=" * 60)

    max_running = 10
    total_budget = 2048

    # Scenario: 3 HIGH + 7 LOW, all want 200 tokens each step
    high_reqs = [MockRequest("high-%d" % i, mlfq_level=0) for i in range(3)]
    low_reqs = [MockRequest("low-%d" % i, mlfq_level=3) for i in range(7)]
    running = high_reqs + low_reqs  # 10/10 = 100% load

    # === Without rate limiting ===
    budget_no_limit = total_budget
    tokens_high_no_limit = 0
    tokens_low_no_limit = 0
    for r in running:
        want = 200
        give = min(want, budget_no_limit)
        budget_no_limit -= give
        if r in high_reqs:
            tokens_high_no_limit += give
        else:
            tokens_low_no_limit += give

    print("  Without rate limiting:")
    print("    HIGH got: %d tokens" % tokens_high_no_limit)
    print("    LOW  got: %d tokens" % tokens_low_no_limit)
    print("    LOW share: %.1f%%" % (100.0 * tokens_low_no_limit / total_budget))

    # === With rate limiting ===
    update_rate_limiters(running, max_running)

    budget_with_limit = total_budget
    tokens_high_with_limit = 0
    tokens_low_with_limit = 0
    for r in running:
        want = 200
        rl = r.rate_limiter
        if rl.is_limited():
            allowed = rl.consume(want)
        else:
            allowed = want
        give = min(allowed, budget_with_limit)
        budget_with_limit -= give
        if r in high_reqs:
            tokens_high_with_limit += give
        else:
            tokens_low_with_limit += give

    print("  With rate limiting:")
    print("    HIGH got: %d tokens" % tokens_high_with_limit)
    print("    LOW  got: %d tokens" % tokens_low_with_limit)
    print("    LOW share: %.1f%%" % (100.0 * tokens_low_with_limit / total_budget))

    # Core assertions:
    # 1. HIGH requests should get the same or more tokens
    check(tokens_high_with_limit >= tokens_high_no_limit,
          "HIGH should get >= tokens with rate limiting")
    # 2. LOW requests should get significantly fewer tokens
    check(tokens_low_with_limit < tokens_low_no_limit,
          "LOW should get fewer tokens with rate limiting")
    # 3. More budget left for potential new HIGH requests
    check(budget_with_limit > budget_no_limit,
          "More budget should remain with rate limiting")

    print("  Budget saved: %d tokens (for new high-prio requests)"
          % (budget_with_limit - budget_no_limit))
    print("  PASSED\n")


def test_multi_step_simulation():
    """Simulate multiple scheduling steps to verify rate limiting over time."""
    print("=" * 60)
    print("Test 11: Multi-step simulation (10 steps)")
    print("=" * 60)

    max_running = 10
    high_req = MockRequest("high-sim", mlfq_level=0)
    low_req = MockRequest("low-sim", mlfq_level=3)
    running = [high_req, low_req]

    # Pad to 80% load to trigger limiting
    for i in range(6):
        running.append(MockRequest("filler-%d" % i, mlfq_level=0))

    high_total = 0
    low_total = 0
    num_steps = 10

    for step in range(num_steps):
        update_rate_limiters(running, max_running)
        # Each request wants 200 tokens per step
        want = 200
        h_rl = high_req.rate_limiter
        l_rl = low_req.rate_limiter

        if h_rl.is_limited():
            h_allowed = h_rl.consume(want)
        else:
            h_allowed = want
        high_total += h_allowed

        if l_rl.is_limited():
            l_allowed = l_rl.consume(want)
        else:
            l_allowed = want
        low_total += l_allowed

    print("  Over %d steps (load=80%%):" % num_steps)
    print("    HIGH total tokens: %d (avg %.1f/step)"
          % (high_total, high_total / num_steps))
    print("    LOW  total tokens: %d (avg %.1f/step)"
          % (low_total, low_total / num_steps))

    # HIGH should get substantially more
    check(high_total > low_total,
          "HIGH should get more tokens over time than LOW")
    # HIGH should be at full speed (200/step * 10 = 2000)
    check(high_total == num_steps * 200,
          "HIGH should run at full speed: expected %d, got %d"
          % (num_steps * 200, high_total))
    print("  PASSED\n")


def test_idle_system_no_penalty():
    """When system is idle, even low-priority requests should run at full speed."""
    print("=" * 60)
    print("Test 12: Idle system — no penalty for any request")
    print("=" * 60)

    max_running = 100
    # Only 10 running = 10% load
    running = [MockRequest("low-%d" % i, mlfq_level=3) for i in range(10)]

    update_rate_limiters(running, max_running)

    for r in running:
        check(not r.rate_limiter.is_limited(),
              "All should be unlimited at 10%% load")

    print("  Load=10%%: all LOW requests unlimited = %s"
          % all(not r.rate_limiter.is_limited() for r in running))
    print("  PASSED\n")


def test_demotion_triggers_limiting():
    """As a request gets demoted in MLFQ, its rate limiting should change."""
    print("=" * 60)
    print("Test 13: MLFQ demotion changes rate limiting tier")
    print("=" * 60)

    max_running = 10
    req = MockRequest("demo-req", mlfq_level=0)
    # Fill to 80% load
    running = [req] + [MockRequest("f-%d" % i, mlfq_level=0) for i in range(7)]

    # Step 1: L0 (HIGH) — should be unlimited
    update_rate_limiters(running, max_running)
    check(not req.rate_limiter.is_limited(), "L0 should be unlimited")
    print("  L0: limited=%s" % req.rate_limiter.is_limited())

    # Demote to L1 (still HIGH)
    req.mlfq_level = 1
    update_rate_limiters(running, max_running)
    check(not req.rate_limiter.is_limited(), "L1 should still be unlimited")
    print("  L1: limited=%s" % req.rate_limiter.is_limited())

    # Demote to L2 (NORMAL)
    req.mlfq_level = 2
    update_rate_limiters(running, max_running)
    check(req.rate_limiter.is_limited(), "L2 should be limited")
    print("  L2: limited=%s, rate=%.1f"
          % (req.rate_limiter.is_limited(), req.rate_limiter.rate))

    # Demote to L3 (LOW)
    req.mlfq_level = 3
    update_rate_limiters(running, max_running)
    check(req.rate_limiter.is_limited(), "L3 should be limited")
    check(req.rate_limiter.rate < TokenRateLimiter.DEFAULT_RATE_NORMAL,
          "L3 rate should be lower than NORMAL rate")
    print("  L3: limited=%s, rate=%.1f"
          % (req.rate_limiter.is_limited(), req.rate_limiter.rate))
    print("  PASSED\n")


def test_consume_zero_when_empty():
    """When bucket is empty, consume should return 0."""
    print("=" * 60)
    print("Test 14: Consume returns 0 when bucket is empty")
    print("=" * 60)

    rl = TokenRateLimiter(rate=10.0, burst=50)
    rl.tokens = 0  # Empty bucket

    allowed = rl.consume(100)
    check(allowed == 0, "Should return 0 when bucket is empty")
    print("  Empty bucket: consume(100) = %d" % allowed)

    # After refill, should have tokens again
    rl.refill()
    allowed = rl.consume(5)
    check(allowed == 5, "After refill, should allow 5")
    print("  After refill: consume(5) = %d" % allowed)
    print("  PASSED\n")


def test_source_code_consistency():
    """Verify that the source code contains expected patterns."""
    print("=" * 60)
    print("Test 15: Source code consistency check")
    print("=" * 60)

    import os
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Check request.py has TokenRateLimiter
    request_path = os.path.join(base, "vllm", "v1", "request.py")
    if os.path.exists(request_path):
        with open(request_path, "r") as f:
            src = f.read()
        check("class TokenRateLimiter" in src,
              "request.py should have TokenRateLimiter class")
        check("def refill(self)" in src,
              "TokenRateLimiter should have refill()")
        check("def consume(self" in src,
              "TokenRateLimiter should have consume()")
        check("self.rate_limiter" in src,
              "Request should have rate_limiter field")
        check("LOAD_THRESHOLD_MODERATE" in src,
              "Should have LOAD_THRESHOLD_MODERATE constant")
        check("LOAD_THRESHOLD_HIGH" in src,
              "Should have LOAD_THRESHOLD_HIGH constant")
        print("  request.py: TokenRateLimiter class present")
    else:
        check(False, "request.py not found at %s" % request_path)

    # Check scheduler.py has rate limiting integration
    scheduler_path = os.path.join(base, "vllm", "v1", "core", "scheduler.py")
    if os.path.exists(scheduler_path):
        with open(scheduler_path, "r") as f:
            src = f.read()
        check("enable_token_rate_limiting" in src,
              "scheduler.py should have enable_token_rate_limiting flag")
        check("_update_rate_limiters" in src,
              "scheduler.py should have _update_rate_limiters method")
        check("_get_priority_tier" in src,
              "scheduler.py should have _get_priority_tier method")
        check("rate_limiter.consume" in src,
              "scheduler.py should call rate_limiter.consume()")
        check("rate_limiter.is_limited()" in src,
              "scheduler.py should check rate_limiter.is_limited()")
        print("  scheduler.py: rate limiting integration present")
    else:
        check(False, "scheduler.py not found at %s" % scheduler_path)

    print("  PASSED\n")


def test_stress_random_workload():
    """Stress test with random workload to verify invariants."""
    print("=" * 60)
    print("Test 16: Stress test — random workload (1000 iterations)")
    print("=" * 60)

    random.seed(42)
    max_running = 50

    for iteration in range(1000):
        num_running = random.randint(1, 80)
        running = []
        for i in range(num_running):
            level = random.choice([0, 1, 2, 3])
            running.append(MockRequest("r%d-%d" % (iteration, i),
                                       mlfq_level=level))

        update_rate_limiters(running, max_running)

        load = num_running / max_running
        for r in running:
            rl = r.rate_limiter
            tier = get_priority_tier(r)

            # Invariant 1: HIGH tier is never limited
            if tier == "HIGH":
                check(not rl.is_limited(),
                      "iter=%d: HIGH should never be limited" % iteration)

            # Invariant 2: If system is idle (<50% load), nobody is limited
            if load < TokenRateLimiter.LOAD_THRESHOLD_MODERATE:
                check(not rl.is_limited(),
                      "iter=%d: idle system should not limit anyone" % iteration)

            # Invariant 3: rate is always positive
            if rl.is_limited():
                check(rl.rate > 0,
                      "iter=%d: rate should be positive" % iteration)

            # Invariant 4: consume never returns more than requested
            want = random.randint(1, 500)
            got = rl.consume(want)
            check(got <= want,
                  "iter=%d: consume(%d) returned %d" % (iteration, want, got))

    print("  1000 iterations with random workloads: all invariants held")
    print("  PASSED\n")


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Token Rate Limiting (Token Bucket) Test Suite")
    print("Optimization 4: Per-request token generation rate control")
    print("=" * 60 + "\n")

    test_token_bucket_basic()
    test_unlimited_rate()
    test_dynamic_rate_adjustment()
    test_refill_does_not_exceed_burst()
    test_priority_tier_classification()
    test_priority_tier_qos_fallback()
    test_load_based_rate_assignment_idle()
    test_load_based_rate_assignment_moderate()
    test_load_based_rate_assignment_high()
    test_rate_limiting_effect_on_token_budget()
    test_multi_step_simulation()
    test_idle_system_no_penalty()
    test_demotion_triggers_limiting()
    test_consume_zero_when_empty()
    test_source_code_consistency()
    test_stress_random_workload()

    print("=" * 60)
    print("Results: %d passed, %d failed" % (passed, failed))
    print("=" * 60)

    if failed > 0:
        print("\nSOME TESTS FAILED!")
        sys.exit(1)
    else:
        print("\nALL TESTS PASSED!")
        sys.exit(0)
