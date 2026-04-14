# SPDX-License-Identifier: Apache-2.0
"""Tests for Optimization 3: Preemption Cache Shield.

These tests verify that:
1. KVCacheManager.free_partial() correctly retains prefix blocks and
   releases tail blocks.
2. The retained blocks keep ref_cnt > 0 and stay out of the free queue.
3. num_cached_block and req_to_blocks are updated consistently.
4. Scheduler integration: preemption uses partial free when caching is
   enabled and prefix blocks have hashes.
"""

import pytest
from collections import defaultdict
from typing import List, Optional
from unittest.mock import MagicMock

from vllm.v1.core.kv_cache_utils import (
    BlockHashType,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    hash_block_tokens,
)
from vllm.v1.core.kv_cache_manager import KVCacheManager


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_manager(num_blocks: int = 20, block_size: int = 16) -> KVCacheManager:
    """Create a minimal KVCacheManager for testing."""
    return KVCacheManager(
        block_size=block_size,
        num_gpu_blocks=num_blocks,
        max_model_len=2048,
        sliding_window=None,
        enable_caching=True,
        num_preallocate_tokens=0,  # Disable preallocation for predictable tests
    )


def _make_request(request_id: str = "req-1",
                  num_tokens: int = 100) -> MagicMock:
    """Create a mock Request with required attributes."""
    req = MagicMock()
    req.request_id = request_id
    req.num_tokens = num_tokens
    req.num_computed_tokens = 0
    req.status = "RUNNING"
    req.all_token_ids = list(range(num_tokens))
    req.spec_token_ids = []
    return req


def _assign_blocks_to_request(
    manager: KVCacheManager,
    request_id: str,
    num_blocks: int,
    num_hashed: int = 0,
) -> List[KVCacheBlock]:
    """Manually assign blocks to a request for testing.

    Args:
        manager: The KVCacheManager.
        request_id: The request ID.
        num_blocks: Total blocks to assign.
        num_hashed: Number of leading blocks to give a fake block_hash.

    Returns:
        The list of assigned blocks.
    """
    blocks = []
    for i in range(num_blocks):
        block = manager.free_block_queue.popleft()
        block.incr_ref()
        blocks.append(block)

    manager.req_to_blocks[request_id] = blocks
    manager.num_cached_block[request_id] = num_hashed

    # Assign fake hashes to the first num_hashed blocks.
    for idx in range(num_hashed):
        blk = blocks[idx]
        prev_hash = blocks[idx - 1].block_hash.hash_value if idx > 0 else None
        fake_tokens = tuple(range(idx * manager.block_size,
                                  (idx + 1) * manager.block_size))
        block_hash = hash_block_tokens(prev_hash, fake_tokens)
        # Use object.__setattr__ to bypass the "already has hash" assertion
        # since we're setting it for the first time on fresh blocks.
        blk._block_hash = block_hash
        manager.cached_block_hash_to_block[block_hash][blk.block_id] = blk

    return blocks


# ---------------------------------------------------------------------------
# Tests for free_partial()
# ---------------------------------------------------------------------------

class TestFreePartialBasic:
    """Basic tests for KVCacheManager.free_partial()."""

    def test_free_partial_keeps_prefix(self):
        """Retained prefix blocks keep ref_cnt > 0."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 10, num_hashed=6)

        initial_free = mgr.free_block_queue.num_free_blocks
        freed = mgr.free_partial(req, keep_prefix_blocks=6)

        assert freed == 4  # 10 - 6 tail blocks freed
        assert mgr.free_block_queue.num_free_blocks == initial_free + 4

        # Retained blocks still have ref_cnt > 0.
        kept = mgr.req_to_blocks["r1"]
        assert len(kept) == 6
        for blk in kept:
            assert blk.ref_cnt > 0

    def test_free_partial_updates_req_to_blocks(self):
        """req_to_blocks should only contain kept blocks."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 8, num_hashed=3)

        original_ids = [b.block_id for b in blocks]
        mgr.free_partial(req, keep_prefix_blocks=3)

        kept = mgr.req_to_blocks["r1"]
        assert [b.block_id for b in kept] == original_ids[:3]

    def test_free_partial_updates_num_cached_block(self):
        """num_cached_block should be min(prev, keep_count)."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        _assign_blocks_to_request(mgr, "r1", 10, num_hashed=8)
        assert mgr.num_cached_block["r1"] == 8

        mgr.free_partial(req, keep_prefix_blocks=5)
        assert mgr.num_cached_block["r1"] == 5  # min(8, 5)

    def test_free_partial_keep_zero_is_full_free(self):
        """keep_prefix_blocks=0 should free everything."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        _assign_blocks_to_request(mgr, "r1", 5, num_hashed=3)

        freed = mgr.free_partial(req, keep_prefix_blocks=0)
        assert freed == 5
        assert "r1" not in mgr.req_to_blocks
        assert "r1" not in mgr.num_cached_block

    def test_free_partial_keep_all(self):
        """keep_prefix_blocks >= total should free nothing."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        _assign_blocks_to_request(mgr, "r1", 5, num_hashed=5)

        initial_free = mgr.free_block_queue.num_free_blocks
        freed = mgr.free_partial(req, keep_prefix_blocks=10)

        assert freed == 0
        assert mgr.free_block_queue.num_free_blocks == initial_free
        assert len(mgr.req_to_blocks["r1"]) == 5

    def test_free_partial_returns_count(self):
        """Return value should be the number of freed blocks."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        _assign_blocks_to_request(mgr, "r1", 7, num_hashed=2)

        freed = mgr.free_partial(req, keep_prefix_blocks=2)
        assert freed == 5


class TestFreePartialRefCounting:
    """Tests for ref_cnt correctness during partial free."""

    def test_freed_blocks_have_zero_refcnt(self):
        """Freed blocks should have ref_cnt == 0 (when not shared)."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 6, num_hashed=2)

        tail_blocks = blocks[2:]  # these will be freed
        mgr.free_partial(req, keep_prefix_blocks=2)

        for blk in tail_blocks:
            assert blk.ref_cnt == 0

    def test_shared_blocks_not_freed_to_queue(self):
        """If a tail block is shared (ref_cnt > 1), it should not enter
        the free queue after partial free (ref_cnt drops to 1, not 0)."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 6, num_hashed=2)

        # Simulate sharing: bump ref_cnt on a tail block.
        blocks[4].incr_ref()  # ref_cnt = 2

        initial_free = mgr.free_block_queue.num_free_blocks
        mgr.free_partial(req, keep_prefix_blocks=2)

        # 4 tail blocks, but one had ref_cnt=2 → only 3 enter free queue.
        assert mgr.free_block_queue.num_free_blocks == initial_free + 3
        assert blocks[4].ref_cnt == 1  # shared, not fully freed


class TestFreePartialSegmentedLRU:
    """Tests for Segmented LRU integration with free_partial()."""

    def test_promoted_block_enters_protected_zone(self):
        """A freed tail block with _promoted=True should enter protected."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 6, num_hashed=2)

        # Mark a tail block as promoted (simulating a prior cache hit).
        blocks[3]._promoted = True

        mgr.free_partial(req, keep_prefix_blocks=2)

        # blocks[3] should be in protected zone.
        assert blocks[3].free_zone == "protected"
        assert blocks[3]._promoted is False  # reset after entering protected

    def test_non_promoted_block_enters_probation(self):
        """A freed tail block without _promoted should enter probation."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 6, num_hashed=2)

        mgr.free_partial(req, keep_prefix_blocks=2)

        for blk in blocks[2:]:
            assert blk.free_zone == "probation"


class TestFreePartialHashPreservation:
    """Tests for hash and cache preservation during partial free."""

    def test_kept_blocks_retain_hash(self):
        """Retained prefix blocks should still have their block_hash."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 8, num_hashed=5)

        hashes_before = [b.block_hash for b in blocks[:5]]
        mgr.free_partial(req, keep_prefix_blocks=5)

        kept = mgr.req_to_blocks["r1"]
        for i, blk in enumerate(kept):
            assert blk.block_hash == hashes_before[i]

    def test_kept_blocks_remain_in_cache(self):
        """Retained blocks should still be findable in cached_block_hash_to_block."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 8, num_hashed=5)

        mgr.free_partial(req, keep_prefix_blocks=5)

        for blk in mgr.req_to_blocks["r1"]:
            bh = blk.block_hash
            assert bh is not None
            assert blk.block_id in mgr.cached_block_hash_to_block.get(bh, {})


class TestFreePartialEdgeCases:
    """Edge case tests for free_partial()."""

    def test_nonexistent_request(self):
        """Partial free on a non-existent request should return 0."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("ghost")
        freed = mgr.free_partial(req, keep_prefix_blocks=5)
        assert freed == 0

    def test_empty_blocks_list(self):
        """Partial free when req has empty block list should return 0."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        mgr.req_to_blocks["r1"] = []
        freed = mgr.free_partial(req, keep_prefix_blocks=3)
        assert freed == 0

    def test_keep_more_than_total(self):
        """keep_prefix_blocks > total blocks should keep all."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        _assign_blocks_to_request(mgr, "r1", 3, num_hashed=3)

        freed = mgr.free_partial(req, keep_prefix_blocks=100)
        assert freed == 0
        assert len(mgr.req_to_blocks["r1"]) == 3

    def test_negative_keep(self):
        """Negative keep_prefix_blocks should be clamped to 0."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        _assign_blocks_to_request(mgr, "r1", 4, num_hashed=2)

        freed = mgr.free_partial(req, keep_prefix_blocks=-5)
        assert freed == 4
        assert "r1" not in mgr.req_to_blocks


class TestPreemptionKeepCountLogic:
    """Tests for the keep_count calculation logic used in the scheduler.

    The scheduler determines keep_count by iterating blocks and counting
    consecutive blocks that have a block_hash. This tests that logic
    in isolation.
    """

    def _compute_keep_count(self, blocks: List[KVCacheBlock]) -> int:
        """Replicate the scheduler's keep_count logic."""
        keep_count = 0
        for blk in blocks:
            if blk.block_hash is not None:
                keep_count += 1
            else:
                break
        return keep_count

    def test_all_hashed(self):
        """All blocks have hashes → keep all."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 5, num_hashed=5)
        assert self._compute_keep_count(blocks) == 5

    def test_none_hashed(self):
        """No blocks have hashes → keep none."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 5, num_hashed=0)
        assert self._compute_keep_count(blocks) == 0

    def test_partial_hashed(self):
        """First 3 of 8 have hashes → keep 3."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 8, num_hashed=3)
        assert self._compute_keep_count(blocks) == 3

    def test_gap_in_hashes(self):
        """Hash chain breaks at block 2 (blocks 0,1 have hash, 2 doesn't,
        3 has hash) → keep 2 (stop at first None)."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 5, num_hashed=2)

        # Manually add hash to block 3 (simulating a gap).
        blocks[3]._block_hash = hash_block_tokens(None, tuple(range(16)))

        assert self._compute_keep_count(blocks) == 2  # stops at block 2


class TestFreePartialThenResume:
    """Tests simulating partial free followed by resume.

    Verifies that retained blocks can be discovered by
    get_computed_blocks() on resume.
    """

    def test_resume_finds_retained_blocks(self):
        """After partial free, retained blocks should be discoverable."""
        mgr = _make_manager(num_blocks=20, block_size=16)
        req = _make_request("r1", num_tokens=160)  # 10 blocks worth

        # Allocate 10 blocks, hash the first 6.
        blocks = _assign_blocks_to_request(mgr, "r1", 10, num_hashed=6)

        # Pre-populate req_to_block_hashes so get_computed_blocks works.
        hashes = []
        for blk in blocks[:6]:
            hashes.append(blk.block_hash)
        mgr.req_to_block_hashes["r1"] = hashes

        # Partial free: keep 6, free 4.
        mgr.free_partial(req, keep_prefix_blocks=6)
        assert len(mgr.req_to_blocks["r1"]) == 6

        # Simulate resume: get_computed_blocks should find the 6 cached blocks.
        # Note: get_computed_blocks uses req_to_block_hashes (which we kept)
        # and cached_block_hash_to_block (which still has our hashes).
        computed_blocks, num_computed_tokens = mgr.get_computed_blocks(req)

        assert len(computed_blocks) == 6
        assert num_computed_tokens == 6 * 16


class TestPreemptionFallbackToFullFree:
    """Tests for the fallback logic: when partial free would release too
    few blocks, fall back to full free to guarantee forward progress.

    This tests the scheduler-side keep_count + would_free logic.
    """

    _PREEMPT_MIN_FREE_BLOCKS = 1  # mirror the scheduler constant

    def _should_partial_free(
        self, blocks: List[KVCacheBlock], enable_caching: bool = True,
    ) -> tuple:
        """Replicate the scheduler's keep_count + fallback logic.

        Returns:
            (use_partial: bool, keep_count: int, would_free: int)
        """
        keep_count = 0
        if enable_caching:
            for blk in blocks:
                if blk.block_hash is not None:
                    keep_count += 1
                else:
                    break
        would_free = len(blocks) - keep_count
        use_partial = (keep_count > 0
                       and would_free >= self._PREEMPT_MIN_FREE_BLOCKS)
        return use_partial, keep_count, would_free

    def test_all_hashed_falls_back_to_full_free(self):
        """If all blocks have hashes, would_free=0 < threshold → full free."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 5, num_hashed=5)

        use_partial, keep_count, would_free = self._should_partial_free(blocks)
        assert not use_partial
        assert keep_count == 5
        assert would_free == 0

    def test_one_unhashed_uses_partial(self):
        """5 hashed + 1 unhashed → would_free=1 >= threshold → partial."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 6, num_hashed=5)

        use_partial, keep_count, would_free = self._should_partial_free(blocks)
        assert use_partial
        assert keep_count == 5
        assert would_free == 1

    def test_no_hashed_falls_back_to_full_free(self):
        """No hashed blocks → keep_count=0 → full free."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 5, num_hashed=0)

        use_partial, keep_count, would_free = self._should_partial_free(blocks)
        assert not use_partial
        assert keep_count == 0

    def test_caching_disabled_falls_back_to_full_free(self):
        """When caching is disabled, always full free."""
        mgr = _make_manager(num_blocks=20)
        blocks = _assign_blocks_to_request(mgr, "r1", 5, num_hashed=3)

        use_partial, _, _ = self._should_partial_free(
            blocks, enable_caching=False)
        assert not use_partial

    def test_mostly_hashed_full_free_releases_all(self):
        """End-to-end: all hashed → fallback → full free releases all."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 5, num_hashed=5)

        use_partial, keep_count, would_free = self._should_partial_free(blocks)
        assert not use_partial

        # Full free (the fallback path).
        initial_free = mgr.free_block_queue.num_free_blocks
        mgr.free(req)
        assert mgr.free_block_queue.num_free_blocks == initial_free + 5
        assert "r1" not in mgr.req_to_blocks

    def test_partial_free_sufficient_blocks(self):
        """End-to-end: partial free releases enough blocks."""
        mgr = _make_manager(num_blocks=20)
        req = _make_request("r1")
        blocks = _assign_blocks_to_request(mgr, "r1", 10, num_hashed=6)

        use_partial, keep_count, would_free = self._should_partial_free(blocks)
        assert use_partial
        assert would_free == 4

        initial_free = mgr.free_block_queue.num_free_blocks
        freed = mgr.free_partial(req, keep_prefix_blocks=keep_count)
        assert freed == 4
        assert mgr.free_block_queue.num_free_blocks == initial_free + 4
        assert len(mgr.req_to_blocks["r1"]) == 6
