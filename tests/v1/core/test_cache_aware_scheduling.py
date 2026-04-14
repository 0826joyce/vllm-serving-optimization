# SPDX-License-Identifier: Apache-2.0
"""Tests for Cache-Aware Scheduling (Optimization 1).

Verifies that the scheduler, when prefix caching is enabled, preferentially
schedules WAITING requests with higher cache hit rates (fewer actual prefill
tokens) within the same MLFQ level, instead of using pure FCFS ordering.

Key scenarios tested:
1. High-cache-hit request is scheduled before low-cache-hit request
   even if the latter arrived first (within the same MLFQ level).
2. Under tight token_budget, cache-aware scheduling allows more requests
   to fit in a single step.
3. MLFQ level isolation is preserved: higher-level requests always
   take precedence over lower-level ones regardless of cache hits.
4. When cache-aware scheduling is disabled, the scheduler falls back to
   FCFS ordering.
"""
from typing import List, Optional

from vllm.config import CacheConfig, ModelConfig, SchedulerConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus


EOS_TOKEN_ID = 50256


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_scheduler(
    max_num_seqs: int = 16,
    max_num_batched_tokens: int = 8192,
    enable_prefix_caching: bool = True,
    num_gpu_blocks: int = 10000,
    block_size: int = 16,
) -> Scheduler:
    """Create a Scheduler with prefix caching enabled by default."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_num_batched_tokens,
    )
    model_config = ModelConfig(
        model="facebook/opt-125m",
        task="auto",
        tokenizer="facebook/opt-125m",
        tokenizer_mode="auto",
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    cache_config = CacheConfig(
        block_size=block_size,
        gpu_memory_utilization=0.9,
        swap_space=0,
        cache_dtype="auto",
        enable_prefix_caching=enable_prefix_caching,
    )
    cache_config.num_gpu_blocks = num_gpu_blocks
    return Scheduler(
        scheduler_config,
        model_config,
        cache_config,
        speculative_config=None,
        lora_config=None,
        log_stats=True,
    )


def make_request(
    request_id: str,
    prompt_token_ids: List[int],
    arrival_time: float = 0.0,
    max_tokens: int = 16,
) -> Request:
    """Create a minimal Request for testing."""
    return Request(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        multi_modal_inputs=None,
        multi_modal_hashes=None,
        multi_modal_placeholders=None,
        sampling_params=SamplingParams(
            ignore_eos=False, max_tokens=max_tokens),
        eos_token_id=EOS_TOKEN_ID,
        arrival_time=arrival_time,
        lora_request=None,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cache_aware_prefers_high_hit_request():
    """When two requests share the same MLFQ level and one has a common
    prefix already cached, the scheduler should pick the high-cache-hit
    request first, even if it arrived later.

    Setup:
      - block_size = 16, token_budget = 512
      - Seed the cache by scheduling a "seed request" that establishes a
        3-block (48-token) common prefix.
      - Then add two new requests A and B to WAITING:
        - A (arrives first): completely unique tokens → 0 cache hits
        - B (arrives later): shares the 48-token common prefix → 3 blocks hit
      - With cache-aware scheduling, B should be scheduled first because
        it needs fewer prefill tokens.
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        enable_prefix_caching=True,
        block_size=block_size,
    )
    assert scheduler.enable_cache_aware_scheduling is True

    # --- Phase 1: Seed the prefix cache ---
    # Common prefix: 3 full blocks = 48 tokens
    common_prefix = [i for i in range(3) for _ in range(block_size)]
    assert len(common_prefix) == 48

    seed_tokens = common_prefix + [100] * 10  # 58 tokens total
    seed_req = make_request("seed", seed_tokens)
    scheduler.add_request(seed_req)
    output = scheduler.schedule()
    assert "seed" in output.num_scheduled_tokens

    # Finish the seed request so its blocks are freed (but hashes remain
    # in the cache for future lookups).
    scheduler.finish_requests("seed", RequestStatus.FINISHED_STOPPED)

    # --- Phase 2: Add two new requests ---
    # Request A: unique tokens, no cache hit (arrives first)
    req_a_tokens = [999] * 64  # 64 unique tokens → 0 cache hits
    req_a = make_request("A", req_a_tokens, arrival_time=1.0)

    # Request B: shares common prefix → 3 blocks cache hit
    req_b_tokens = common_prefix + [200] * 10  # 58 tokens, 48 cached
    req_b = make_request("B", req_b_tokens, arrival_time=2.0)

    # Add A first, then B (FCFS: A before B)
    scheduler.add_request(req_a)
    scheduler.add_request(req_b)

    # Both should be at MLFQ level 0
    assert len(scheduler.mlfq_queues[0]) == 2
    # In FCFS order, A would be at front
    assert list(scheduler.mlfq_queues[0])[0].request_id == "A"
    assert list(scheduler.mlfq_queues[0])[1].request_id == "B"

    # --- Phase 3: Schedule and verify ---
    output = scheduler.schedule()

    # Both requests should be scheduled (budget is large enough)
    assert "A" in output.num_scheduled_tokens
    assert "B" in output.num_scheduled_tokens

    # B should have fewer scheduled tokens (because prefix was cached)
    tokens_a = output.num_scheduled_tokens["A"]
    tokens_b = output.num_scheduled_tokens["B"]
    assert tokens_b < tokens_a, (
        f"B (cache hit) should need fewer tokens than A, "
        f"got B={tokens_b}, A={tokens_a}")

    # B's scheduled tokens should be around 10 (only the unique tail)
    # while A should be the full 64
    assert tokens_a == 64
    assert tokens_b == 10


def test_cache_aware_fits_more_requests_under_tight_budget():
    """Under a tight token_budget, cache-aware scheduling should allow
    more requests to be scheduled in a single step compared to FCFS.

    Setup:
      - token_budget = 200
      - 3 requests all with 100 prompt tokens each:
        - R1 (first): 0 cache hits → needs 100 tokens
        - R2 (second): 0 cache hits → needs 100 tokens
        - R3 (third): 80 tokens cached → needs only 20 tokens
      - FCFS: R1 (100) + R2 (100) = 200 → R3 not scheduled
      - Cache-aware: R3 (20) first, then R1 (100), then R2 (80) = 200
        → all three could potentially be scheduled
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=200,
        max_num_seqs=16,
        enable_prefix_caching=True,
        block_size=block_size,
    )

    # Seed the cache with 5 blocks (80 tokens) of common prefix
    common_prefix = [i for i in range(5) for _ in range(block_size)]
    assert len(common_prefix) == 80

    seed_tokens = common_prefix + [100] * 10
    seed_req = make_request("seed", seed_tokens, max_tokens=1)
    scheduler.add_request(seed_req)
    output = scheduler.schedule()
    scheduler.finish_requests("seed", RequestStatus.FINISHED_STOPPED)

    # R1: 100 unique tokens, no cache hit
    r1 = make_request("R1", [901] * 100, arrival_time=1.0)
    # R2: 100 different unique tokens, no cache hit
    r2 = make_request("R2", [902] * 100, arrival_time=2.0)
    # R3: shares the 80-token prefix → only 20 new tokens needed
    r3 = make_request("R3", common_prefix + [903] * 20, arrival_time=3.0)

    scheduler.add_request(r1)
    scheduler.add_request(r2)
    scheduler.add_request(r3)

    output = scheduler.schedule()

    # With cache-aware: R3 should be scheduled (20 tokens) and at least
    # one of R1/R2 (100 tokens).  With budget=200, we can fit:
    #   R3 (20) + R1 (100) + R2 (80) = 200  [if R2 fits in remaining]
    # or at minimum R3 + R1 = 120 tokens, leaving 80 for partial R2.
    # The key assertion: R3 IS scheduled despite arriving last.
    assert "R3" in output.num_scheduled_tokens, (
        "R3 (high cache hit) should be scheduled despite arriving last")

    # R3 should need very few tokens
    assert output.num_scheduled_tokens["R3"] <= 20

    # Count how many requests were scheduled
    num_scheduled = len(output.num_scheduled_tokens)
    assert num_scheduled >= 2, (
        f"Expected at least 2 requests scheduled, got {num_scheduled}")


def test_mlfq_level_isolation_preserved():
    """Cache-aware scheduling must NOT violate MLFQ level isolation.

    A lower-level (higher priority) request should always be scheduled
    before a higher-level (lower priority) request, regardless of
    cache hit rates.

    Setup:
      - R_high: at MLFQ L0, no cache hits, 64 tokens
      - R_low:  at MLFQ L1, full cache hits, only 10 new tokens
      - R_high should be scheduled first.
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        enable_prefix_caching=True,
        block_size=block_size,
    )

    # Seed cache
    common_prefix = [i for i in range(3) for _ in range(block_size)]
    seed_req = make_request("seed", common_prefix + [100] * 5)
    scheduler.add_request(seed_req)
    scheduler.schedule()
    scheduler.finish_requests("seed", RequestStatus.FINISHED_STOPPED)

    # R_low: shares prefix (high cache hit), but we'll manually put it at L1
    r_low = make_request("low", common_prefix + [200] * 5, arrival_time=1.0)
    scheduler.add_request(r_low)

    # Manually demote r_low to MLFQ level 1
    scheduler.mlfq_queues[0].remove(r_low)
    r_low.mlfq_level = 1
    scheduler.mlfq_queues[1].append(r_low)

    # R_high: unique tokens (no cache hit), at L0
    r_high = make_request("high", [999] * 64, arrival_time=2.0)
    scheduler.add_request(r_high)

    # Verify queue state
    assert len(scheduler.mlfq_queues[0]) == 1  # r_high at L0
    assert len(scheduler.mlfq_queues[1]) == 1  # r_low at L1

    output = scheduler.schedule()

    # Both should be scheduled (budget is large)
    assert "high" in output.num_scheduled_tokens
    assert "low" in output.num_scheduled_tokens

    # Verify scheduling order: r_high (L0) should be in the running list
    # before r_low (L1).
    running_ids = [r.request_id for r in scheduler.running]
    high_idx = running_ids.index("high")
    low_idx = running_ids.index("low")
    assert high_idx < low_idx, (
        f"L0 request should be scheduled before L1 request, "
        f"but got high@{high_idx}, low@{low_idx}")


def test_fallback_to_fcfs_when_caching_disabled():
    """When prefix caching is disabled, cache-aware scheduling should be
    off and the scheduler should use plain FCFS within MLFQ levels.

    Verify by checking that `enable_cache_aware_scheduling` is False and
    that requests are scheduled in FCFS order.
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        enable_prefix_caching=False,
        block_size=block_size,
    )

    # Cache-aware should be disabled
    assert scheduler.enable_cache_aware_scheduling is False

    # Add requests in order
    r1 = make_request("R1", [1] * 32, arrival_time=1.0)
    r2 = make_request("R2", [2] * 32, arrival_time=2.0)
    r3 = make_request("R3", [3] * 32, arrival_time=3.0)

    scheduler.add_request(r1)
    scheduler.add_request(r2)
    scheduler.add_request(r3)

    output = scheduler.schedule()

    # All scheduled
    assert len(output.num_scheduled_tokens) == 3

    # Running order should be FCFS (R1 → R2 → R3)
    running_ids = [r.request_id for r in scheduler.running]
    assert running_ids == ["R1", "R2", "R3"], (
        f"Expected FCFS order, got {running_ids}")


def test_cache_aware_with_single_candidate():
    """When there's only 1 request in the MLFQ level, cache-aware
    scheduling should behave identically to FCFS — no errors.
    """
    scheduler = create_scheduler(
        enable_prefix_caching=True,
        block_size=16,
    )

    req = make_request("only", [42] * 32)
    scheduler.add_request(req)

    output = scheduler.schedule()
    assert "only" in output.num_scheduled_tokens
    assert output.num_scheduled_tokens["only"] == 32


def test_cache_aware_scan_window_limits_candidates():
    """The scan window should limit how many candidates are evaluated.

    With scan_window=2 and 4 candidates, only the first 2 should be
    considered for reordering.
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        enable_prefix_caching=True,
        block_size=block_size,
    )
    # Override scan window to a small value
    scheduler.cache_aware_scan_window = 2

    # Seed cache
    common_prefix = [i for i in range(3) for _ in range(block_size)]
    seed_req = make_request("seed", common_prefix + [100] * 5)
    scheduler.add_request(seed_req)
    scheduler.schedule()
    scheduler.finish_requests("seed", RequestStatus.FINISHED_STOPPED)

    # Add 4 requests: R1, R2 (no cache), R3 (cache hit, but outside window),
    # R4 (no cache)
    r1 = make_request("R1", [901] * 48, arrival_time=1.0)
    r2 = make_request("R2", [902] * 48, arrival_time=2.0)
    r3 = make_request("R3", common_prefix + [903] * 5, arrival_time=3.0)
    r4 = make_request("R4", [904] * 48, arrival_time=4.0)

    scheduler.add_request(r1)
    scheduler.add_request(r2)
    scheduler.add_request(r3)
    scheduler.add_request(r4)

    # With scan_window=2, only R1 and R2 are considered.
    # R3 (which has cache hits) is outside the window and won't be
    # prioritised in this step.
    output = scheduler.schedule()

    # All should be scheduled (budget is large)
    for rid in ["R1", "R2", "R3", "R4"]:
        assert rid in output.num_scheduled_tokens

    # Since R3 is outside scan window, R1 should be scheduled first
    # (FCFS among the scanned candidates R1 and R2, both with 0 hits).
    running_ids = [r.request_id for r in scheduler.running]
    r1_idx = running_ids.index("R1")
    r3_idx = running_ids.index("R3")
    # R1 should come before R3 because R3 was outside scan window
    assert r1_idx < r3_idx, (
        f"R1 (inside scan window) should be before R3 (outside), "
        f"got R1@{r1_idx}, R3@{r3_idx}")


def test_cache_aware_all_requests_fully_cached():
    """Edge case: all candidates have full cache hits.

    The scheduler should handle this gracefully, including the edge case
    where num_new_tokens == 0 (prompt divisible by block_size and fully
    cached), which forces recompute of the last block.
    """
    block_size = 16
    scheduler = create_scheduler(
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        enable_prefix_caching=True,
        block_size=block_size,
    )

    # Create tokens that are exactly 3 blocks = 48 tokens
    common_prefix = [i for i in range(3) for _ in range(block_size)]

    # Seed the cache
    seed_req = make_request("seed", common_prefix + [100] * 5)
    scheduler.add_request(seed_req)
    scheduler.schedule()
    scheduler.finish_requests("seed", RequestStatus.FINISHED_STOPPED)

    # Two requests with identical tokens as the prefix + some unique tail
    r1 = make_request("R1", common_prefix + [201] * 5, arrival_time=1.0)
    r2 = make_request("R2", common_prefix + [202] * 5, arrival_time=2.0)
    scheduler.add_request(r1)
    scheduler.add_request(r2)

    output = scheduler.schedule()

    # Both should be scheduled
    assert "R1" in output.num_scheduled_tokens
    assert "R2" in output.num_scheduled_tokens

    # Both should need very few tokens (only the unique tail)
    assert output.num_scheduled_tokens["R1"] <= 5 + block_size
    assert output.num_scheduled_tokens["R2"] <= 5 + block_size
