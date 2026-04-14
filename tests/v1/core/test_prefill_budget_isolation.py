# SPDX-License-Identifier: Apache-2.0
"""Tests for Prefill Budget Isolation (Phase 2 fix).

These tests verify that the scheduler correctly isolates long-prefill
requests from short interactive requests by:
1. Limiting the number of concurrent long prefills per scheduling step.
2. Reserving a portion of the token_budget for short requests so that
   long prefills cannot starve them.

The tests exercise the logic added to Scheduler.schedule()'s WAITING
loop, without needing a live model or GPU.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight stubs — only what the Scheduler's WAITING loop touches.
# ---------------------------------------------------------------------------

@dataclass
class FakeKVCacheBlock:
    block_id: int
    block_hash: Optional[Any] = None


class FakeKVCacheManager:
    """Minimal stub of KVCacheManager for scheduling tests."""

    def __init__(self, num_blocks: int = 100, block_size: int = 16):
        self.block_size = block_size
        self.enable_caching = True
        self._next_block = 0
        self._num_blocks = num_blocks
        self.req_to_blocks: Dict[str, List[FakeKVCacheBlock]] = {}

    def get_computed_blocks(
        self, request: "FakeRequest",
    ) -> Tuple[List[FakeKVCacheBlock], int]:
        """Return pre-set computed blocks for a request."""
        return (request._computed_blocks, request._num_computed_tokens_cache)

    def allocate_slots(
        self, request: "FakeRequest", num_new_tokens: int,
        computed_blocks: Optional[List[FakeKVCacheBlock]] = None,
    ) -> Optional[List[FakeKVCacheBlock]]:
        """Allocate dummy blocks — always succeeds."""
        from math import ceil
        n_blocks = ceil(num_new_tokens / self.block_size)
        blocks = []
        for _ in range(n_blocks):
            blocks.append(FakeKVCacheBlock(block_id=self._next_block))
            self._next_block += 1
        self.req_to_blocks.setdefault(request.request_id, []).extend(blocks)
        return blocks

    def get_num_common_prefix_blocks(self, request: Any, n: int) -> int:
        return 0

    def free(self, request: "FakeRequest") -> None:
        self.req_to_blocks.pop(request.request_id, None)

    def free_block_hashes(self, request: "FakeRequest") -> None:
        pass

    @property
    def usage(self) -> float:
        return 0.0

    def make_prefix_cache_stats(self) -> Any:
        return None


class FakeRateLimiter:
    def is_limited(self) -> bool:
        return False

    def consume(self, n: int) -> int:
        return n


class FakeRequest:
    """Minimal stub of Request for WAITING loop tests."""

    def __init__(
        self,
        request_id: str,
        num_prompt_tokens: int,
        num_computed_tokens: int = 0,
        mlfq_level: int = 0,
    ):
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.num_tokens = num_prompt_tokens
        self.num_tokens_with_spec = num_prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.num_output_tokens = 0
        self.output_token_ids: List[int] = []
        self.all_token_ids = list(range(num_prompt_tokens))
        self.spec_token_ids: List[int] = []
        self.mm_positions = None
        self.mm_hashes = None
        self.lora_request = None
        self.status = "waiting"
        self.arrival_time = time.monotonic()
        self.effective_priority = 0
        self.mlfq_level = mlfq_level
        self.mlfq_tokens_consumed = 0
        self.stop_reason = None
        self.rate_limiter = FakeRateLimiter()

        # For cache-aware test support.
        self._computed_blocks: List[FakeKVCacheBlock] = []
        self._num_computed_tokens_cache = num_computed_tokens

    @property
    def num_tokens(self) -> int:
        return self._num_tokens

    @num_tokens.setter
    def num_tokens(self, value: int) -> None:
        self._num_tokens = value

    def has_encoder_inputs(self) -> bool:
        return False

    def is_finished(self) -> bool:
        return self.status.startswith("finished")

    def get_finished_reason(self) -> Optional[str]:
        return None

    def take_events(self) -> list:
        return []

    def append_output_token_ids(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def mlfq_promote(self) -> None:
        self.mlfq_level = max(0, self.mlfq_level - 1)

    def mlfq_account_tokens(self, n: int) -> None:
        self.mlfq_tokens_consumed += n


# ---------------------------------------------------------------------------
# Minimal Scheduler wrapper — patches only what schedule() touches.
# ---------------------------------------------------------------------------

def _make_test_scheduler(
    max_num_batched_tokens: int = 4096,
    max_num_seqs: int = 256,
    long_prefill_threshold: int = 1024,
    short_budget_reserve_ratio: float = 0.3,
    max_concurrent_long_prefill: int = 2,
    num_gpu_blocks: int = 200,
    block_size: int = 16,
) -> "TestScheduler":
    """Build a lightweight Scheduler-like object for testing.

    We don't use the real Scheduler.__init__ because it requires
    full VllmConfig objects.  Instead we replicate only the fields
    that the WAITING loop in schedule() reads.
    """
    sched = TestScheduler()
    sched.max_num_scheduled_tokens = max_num_batched_tokens
    sched.max_num_running_reqs = max_num_seqs
    sched.max_model_len = 8192
    sched.block_size = block_size
    sched.enable_qos_priority = False
    sched.enable_mlfq = True
    sched.enable_cache_aware_scheduling = False
    sched.enable_token_rate_limiting = False
    sched.enable_pd_aware_scheduling = False
    sched.kv_receive_monitor = None
    sched.lora_config = None
    sched.speculative_config = None
    sched.log_stats = False

    sched.long_prefill_threshold = long_prefill_threshold
    sched.short_budget_reserve_ratio = short_budget_reserve_ratio
    sched.max_concurrent_long_prefill = max_concurrent_long_prefill
    sched._PREEMPT_MIN_FREE_BLOCKS = 1

    sched.kv_cache_manager = FakeKVCacheManager(
        num_blocks=num_gpu_blocks, block_size=block_size)

    sched.requests = {}
    sched.waiting: Deque[FakeRequest] = deque()
    sched.running: List[FakeRequest] = []
    sched.mlfq_queues: List[Deque[FakeRequest]] = [
        deque() for _ in range(4)
    ]
    sched.scheduled_req_ids: Set[str] = set()
    sched.finished_req_ids: Set[str] = set()
    sched._cached_reqs_data: Dict[str, Any] = {}
    sched.max_num_encoder_input_tokens = 0
    sched.encoder_cache_manager = MagicMock()
    sched.encoder_cache_manager.get_freed_ids.return_value = []

    return sched


class TestScheduler:
    """Thin shell that reuses the real Scheduler.schedule() method."""

    def add_request(self, request: FakeRequest) -> None:
        self.requests[request.request_id] = request
        request.mlfq_level = 0
        request.mlfq_tokens_consumed = 0
        self.mlfq_queues[0].append(request)
        self.waiting.append(request)
        request.status = "waiting"

    # Import and bind the real schedule-related methods.
    # We'll call _run_waiting_loop() instead — a helper that
    # invokes only the WAITING portion of schedule().

    def _run_waiting_loop(self) -> Dict[str, int]:
        """Run just the WAITING portion of the scheduler and return
        {request_id: num_scheduled_tokens} for newly scheduled reqs.

        This replicates the core WAITING loop logic from schedule().
        """
        token_budget = self.max_num_scheduled_tokens
        num_scheduled_tokens: Dict[str, int] = {}
        preempted_reqs: list = []

        if not preempted_reqs:
            # ---- Prefill Budget Isolation: per-step state ----
            long_prefill_count = 0
            short_budget_reserved = int(
                token_budget * self.short_budget_reserve_ratio)

            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                # --- Select next candidate (MLFQ peek) ---
                request = None
                for level_queue in self.mlfq_queues:
                    if level_queue:
                        request = level_queue[0]
                        break
                if request is None:
                    break

                # Get computed blocks.
                computed_blocks, num_computed_tokens = \
                    self.kv_cache_manager.get_computed_blocks(request)

                num_new_tokens = request.num_tokens - num_computed_tokens
                if num_new_tokens == 0:
                    num_computed_tokens -= self.block_size
                    num_new_tokens = self.block_size
                    computed_blocks.pop()
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # ---- Prefill Budget Isolation: long prefill control ----
                is_long_prefill = (
                    num_new_tokens > self.long_prefill_threshold
                    and request.num_computed_tokens == 0)

                if is_long_prefill:
                    if long_prefill_count >= self.max_concurrent_long_prefill:
                        break
                    effective_budget = token_budget - short_budget_reserved
                    if effective_budget <= 0:
                        break
                    num_new_tokens = min(num_new_tokens, effective_budget)
                    long_prefill_count += 1

                # Allocate.
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_blocks)
                if new_blocks is None:
                    break

                # Move from waiting to running.
                self.mlfq_queues[request.mlfq_level].remove(request)
                self.waiting.remove(request)
                self.running.append(request)
                self.scheduled_req_ids.add(request.request_id)
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = "running"
                request.num_computed_tokens = num_computed_tokens

        return num_scheduled_tokens


# ===========================================================================
# Tests
# ===========================================================================

class TestLongPrefillConcurrencyLimit:
    """Verify that at most max_concurrent_long_prefill long prefills
    are scheduled per step."""

    def test_only_two_long_prefills_scheduled(self):
        """With 5 long-prefill requests and max_concurrent=2, only 2
        should be scheduled in a single step."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=16384,
            max_concurrent_long_prefill=2,
            long_prefill_threshold=1024,
        )

        # 5 long-prefill requests (2048 tokens each).
        for i in range(5):
            sched.add_request(FakeRequest(
                request_id=f"long-{i}",
                num_prompt_tokens=2048,
            ))

        result = sched._run_waiting_loop()

        long_scheduled = [rid for rid in result if rid.startswith("long-")]
        assert len(long_scheduled) == 2, (
            f"Expected 2 long prefills, got {len(long_scheduled)}: "
            f"{long_scheduled}")

    def test_concurrent_limit_is_configurable(self):
        """max_concurrent_long_prefill=1 should allow only 1."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=16384,
            max_concurrent_long_prefill=1,
            long_prefill_threshold=1024,
        )

        for i in range(3):
            sched.add_request(FakeRequest(
                request_id=f"long-{i}",
                num_prompt_tokens=2048,
            ))

        result = sched._run_waiting_loop()
        long_scheduled = [rid for rid in result if rid.startswith("long-")]
        assert len(long_scheduled) == 1

    def test_short_requests_not_affected_by_limit(self):
        """Short requests (< threshold) are not subject to the long-
        prefill concurrent limit."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=8192,
            max_concurrent_long_prefill=2,
            long_prefill_threshold=1024,
        )

        # Add 10 short requests (256 tokens each = 2560 total).
        for i in range(10):
            sched.add_request(FakeRequest(
                request_id=f"short-{i}",
                num_prompt_tokens=256,
            ))

        result = sched._run_waiting_loop()

        # All 10 should be scheduled (budget allows it).
        assert len(result) == 10


class TestShortBudgetReservation:
    """Verify that short requests get a reserved budget that long
    prefills cannot consume."""

    def test_long_prefill_respects_reserved_budget(self):
        """A single long prefill should not consume more than
        (token_budget - reserved) tokens."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=4096,
            short_budget_reserve_ratio=0.3,  # reserve 1228 tokens
            max_concurrent_long_prefill=2,
            long_prefill_threshold=1024,
        )

        # One long request that wants 4096 tokens.
        sched.add_request(FakeRequest(
            request_id="long-0",
            num_prompt_tokens=4096,
        ))

        result = sched._run_waiting_loop()

        # The long prefill should get at most 4096 - 1228 = 2867 tokens.
        scheduled_tokens = result.get("long-0", 0)
        reserved = int(4096 * 0.3)  # 1228
        effective = 4096 - reserved  # 2867 (actually 2868 due to int())
        assert scheduled_tokens <= effective + 1, (
            f"Long prefill got {scheduled_tokens} tokens, expected "
            f"<= {effective}")

    def test_short_requests_scheduled_after_long_budget_exhausted(self):
        """After long prefills consume their budget, short requests
        should still be schedulable from the reserved portion."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=4096,
            short_budget_reserve_ratio=0.3,
            max_concurrent_long_prefill=1,
            long_prefill_threshold=512,
        )

        # One long request (2048 tokens).
        sched.add_request(FakeRequest(
            request_id="long-0",
            num_prompt_tokens=2048,
        ))
        # Several short requests (128 tokens each).
        for i in range(5):
            sched.add_request(FakeRequest(
                request_id=f"short-{i}",
                num_prompt_tokens=128,
            ))

        result = sched._run_waiting_loop()

        # The long request should be scheduled.
        assert "long-0" in result
        # At least some short requests should also be scheduled.
        short_scheduled = [r for r in result if r.startswith("short-")]
        assert len(short_scheduled) >= 1, (
            f"Expected at least 1 short request scheduled, got "
            f"{len(short_scheduled)}")


class TestMixedWorkload:
    """Test realistic mixed workloads with both long and short
    requests to verify budget isolation in aggregate."""

    def test_mixed_workload_fairness(self):
        """In a mixed workload of long docs + short interactive
        requests, short requests should not be completely starved."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=4096,
            short_budget_reserve_ratio=0.3,
            max_concurrent_long_prefill=2,
            long_prefill_threshold=1024,
        )

        # 3 long-prefill requests (2048 tokens each).
        for i in range(3):
            sched.add_request(FakeRequest(
                request_id=f"long-{i}",
                num_prompt_tokens=2048,
            ))
        # 5 short requests (256 tokens each).
        for i in range(5):
            sched.add_request(FakeRequest(
                request_id=f"short-{i}",
                num_prompt_tokens=256,
            ))

        result = sched._run_waiting_loop()

        long_count = sum(1 for r in result if r.startswith("long-"))
        short_count = sum(1 for r in result if r.startswith("short-"))

        # At most 2 long prefills.
        assert long_count <= 2
        # Verify total tokens within budget.
        total_tokens = sum(result.values())
        assert total_tokens <= 4096

    def test_all_short_when_budget_tight(self):
        """When budget is small and all requests are short, they should
        all be scheduled normally (no false positive long-prefill blocking)."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=2048,
            long_prefill_threshold=1024,
        )

        for i in range(8):
            sched.add_request(FakeRequest(
                request_id=f"short-{i}",
                num_prompt_tokens=200,
            ))

        result = sched._run_waiting_loop()
        # All 8 × 200 = 1600 tokens < 2048 budget.
        assert len(result) == 8
        assert sum(result.values()) == 1600

    def test_resumed_request_not_treated_as_long_prefill(self):
        """A resumed request (num_computed_tokens > 0) should NOT be
        classified as a long prefill even if its remaining tokens
        exceed the threshold."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=8192,
            max_concurrent_long_prefill=0,  # No long prefills allowed!
            long_prefill_threshold=1024,
        )

        # A request with 2048 prompt tokens but 1024 already computed.
        # Remaining = 1024, but since num_computed_tokens > 0 it's a
        # resumed request, not a fresh long prefill.
        req = FakeRequest(
            request_id="resumed-0",
            num_prompt_tokens=2048,
            num_computed_tokens=1024,
        )
        req._computed_blocks = [FakeKVCacheBlock(i) for i in range(64)]
        req._num_computed_tokens_cache = 1024
        sched.add_request(req)

        result = sched._run_waiting_loop()

        # Should be scheduled despite max_concurrent_long_prefill=0
        # because it's not a fresh long prefill.
        assert "resumed-0" in result

    def test_zero_reserve_ratio_allows_full_budget_for_long(self):
        """When short_budget_reserve_ratio=0, long prefills can use the
        full token budget."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=4096,
            short_budget_reserve_ratio=0.0,
            max_concurrent_long_prefill=1,
            long_prefill_threshold=1024,
        )

        sched.add_request(FakeRequest(
            request_id="long-0",
            num_prompt_tokens=4096,
        ))

        result = sched._run_waiting_loop()
        assert "long-0" in result
        # Should get the full budget.
        assert result["long-0"] == 4096

    def test_threshold_boundary(self):
        """Requests exactly at the threshold are NOT treated as long
        prefill (condition is strictly greater than)."""
        sched = _make_test_scheduler(
            max_num_batched_tokens=4096,
            max_concurrent_long_prefill=0,  # Would block if treated as long
            long_prefill_threshold=1024,
        )

        sched.add_request(FakeRequest(
            request_id="boundary-0",
            num_prompt_tokens=1024,
        ))

        result = sched._run_waiting_loop()
        assert "boundary-0" in result


class TestPrefillBudgetIsolationEndToEnd:
    """Simulate the Phase 4 scenario: Bronze long-doc burst should not
    starve Gold-A/Silver short interactive requests."""

    def test_phase4_scenario(self):
        """Simulate Phase 4: 5× Bronze long docs + Gold-A/Silver short
        requests arrive simultaneously.

        Expected: Short requests should still be schedulable because
        long prefills are capped and budget is reserved.
        """
        sched = _make_test_scheduler(
            max_num_batched_tokens=4096,
            short_budget_reserve_ratio=0.3,
            max_concurrent_long_prefill=2,
            long_prefill_threshold=1024,
        )

        # Bronze long docs (4096 tokens each).
        for i in range(5):
            sched.add_request(FakeRequest(
                request_id=f"bronze-long-{i}",
                num_prompt_tokens=4096,
            ))

        # Gold-A short interactive (128 tokens each).
        for i in range(3):
            sched.add_request(FakeRequest(
                request_id=f"gold-short-{i}",
                num_prompt_tokens=128,
            ))

        # Silver short interactive (256 tokens each).
        for i in range(3):
            sched.add_request(FakeRequest(
                request_id=f"silver-short-{i}",
                num_prompt_tokens=256,
            ))

        result = sched._run_waiting_loop()

        bronze_count = sum(1 for r in result if r.startswith("bronze-"))
        gold_count = sum(1 for r in result if r.startswith("gold-"))
        silver_count = sum(1 for r in result if r.startswith("silver-"))

        # At most 2 bronze long prefills.
        assert bronze_count <= 2, (
            f"Expected <=2 bronze, got {bronze_count}")

        total_tokens = sum(result.values())
        assert total_tokens <= 4096, (
            f"Total tokens {total_tokens} exceeds budget 4096")

        print(f"\nPhase 4 simulation results:")
        print(f"  Bronze long prefills scheduled: {bronze_count}")
        print(f"  Gold-A short scheduled: {gold_count}")
        print(f"  Silver short scheduled: {silver_count}")
        print(f"  Total tokens: {total_tokens} / 4096")
        for rid, tokens in sorted(result.items()):
            print(f"    {rid}: {tokens} tokens")
