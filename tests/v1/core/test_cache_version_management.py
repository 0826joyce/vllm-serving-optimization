# SPDX-License-Identifier: Apache-2.0
"""Tests for Cache Version Management (Phase 1 fix).

These tests verify:
1. FreeKVCacheBlockQueue.resize_protected() correctly demotes excess
   protected blocks to probation when the ratio is shrunk, and correctly
   restores the cap when the ratio is enlarged.
2. KVCacheManager._check_cache_health() automatically shrinks the
   protected zone on hit-rate drop and restores it on recovery.
"""

import pytest
from typing import List

from vllm.v1.core.kv_cache_utils import (
    FreeKVCacheBlockQueue,
    KVCacheBlock,
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _make_blocks(n: int) -> List[KVCacheBlock]:
    """Create *n* KVCacheBlock instances with ids 0..n-1."""
    return [KVCacheBlock(block_id=i) for i in range(n)]


def _probation_ids(queue: FreeKVCacheBlockQueue) -> List[int]:
    """Get block ids in probation zone (head→tail order)."""
    ids = []
    curr = queue._probation_head
    while curr is not None:
        ids.append(curr.block_id)
        curr = curr.next_free_block
    return ids


def _protected_ids(queue: FreeKVCacheBlockQueue) -> List[int]:
    """Get block ids in protected zone (head→tail order)."""
    ids = []
    curr = queue._protected_head
    while curr is not None:
        ids.append(curr.block_id)
        curr = curr.next_free_block
    return ids


# ===========================================================================
# FreeKVCacheBlockQueue — resize_protected() tests
# ===========================================================================

class TestResizeProtected:
    """Tests for the resize_protected() method."""

    def test_shrink_demotes_excess_to_probation(self):
        """Shrinking protected zone demotes excess blocks to probation head."""
        blocks = _make_blocks(10)
        q = FreeKVCacheBlockQueue(blocks, protected_ratio=0.5)
        # max_protected = 5

        # Promote 5 blocks to fill protected zone.
        for b in blocks[:5]:
            q.promote(b)
        # Probation: [5,6,7,8,9], Protected: [0,1,2,3,4]
        assert q.num_protected_blocks == 5
        assert q.num_probation_blocks == 5

        # Shrink to 10% → new max_protected = 1
        q.resize_protected(0.1)

        assert q.max_protected_blocks == 1
        assert q.num_protected_blocks == 1
        # 4 blocks demoted from protected → probation head
        assert q.num_probation_blocks == 9

        # The protected zone should keep only the newest (tail) block.
        # Demotion removes from protected head, so blocks[0..3] are demoted.
        # The remaining protected block should be blocks[4] (the last promoted).
        prot = _protected_ids(q)
        assert len(prot) == 1
        assert prot[0] == 4

        # Demoted blocks should appear at probation head (LIFO demotion).
        prob = _probation_ids(q)
        # Blocks demoted in order: 0, 1, 2, 3 — each prepended to probation
        # head, so order is [3, 2, 1, 0, 5, 6, 7, 8, 9].
        assert prob[:4] == [3, 2, 1, 0]
        assert prob[4:] == [5, 6, 7, 8, 9]

    def test_expand_allows_more_protected(self):
        """Expanding protected zone allows more blocks to be promoted."""
        blocks = _make_blocks(10)
        q = FreeKVCacheBlockQueue(blocks, protected_ratio=0.1)
        # max_protected = 1

        q.promote(blocks[0])
        assert q.num_protected_blocks == 1

        # Try to promote another — should demote blocks[0].
        q.promote(blocks[1])
        assert q.num_protected_blocks == 1
        assert _protected_ids(q) == [1]

        # Now expand to 50%.
        q.resize_protected(0.5)
        assert q.max_protected_blocks == 5

        # Now we can promote more without demotion.
        q.promote(blocks[2])
        q.promote(blocks[3])
        assert q.num_protected_blocks == 3  # 1(existing) + 2(new)

    def test_resize_to_zero_demotes_all(self):
        """Resizing to 0% demotes all protected blocks."""
        blocks = _make_blocks(6)
        q = FreeKVCacheBlockQueue(blocks, protected_ratio=0.5)
        # max_protected = 3

        q.promote(blocks[0])
        q.promote(blocks[1])
        q.promote(blocks[2])
        assert q.num_protected_blocks == 3

        q.resize_protected(0.0)
        assert q.max_protected_blocks == 0
        assert q.num_protected_blocks == 0
        assert q.num_probation_blocks == 6

    def test_resize_with_empty_free_queue(self):
        """Resizing when free queue is empty should not crash."""
        blocks = _make_blocks(4)
        q = FreeKVCacheBlockQueue(blocks, protected_ratio=0.5)

        # Pop all blocks (simulate all allocated).
        for _ in range(4):
            q.popleft()
        assert q.num_free_blocks == 0

        # Should not crash.
        q.resize_protected(0.1)
        assert q.max_protected_blocks == 0

    def test_resize_noop_when_under_limit(self):
        """Resizing when protected count is already under new limit is a noop."""
        blocks = _make_blocks(10)
        q = FreeKVCacheBlockQueue(blocks, protected_ratio=0.5)
        # max_protected = 5

        q.promote(blocks[0])
        q.promote(blocks[1])
        assert q.num_protected_blocks == 2

        # Shrink to 30% → new max_protected = 3, but we only have 2.
        q.resize_protected(0.3)
        assert q.max_protected_blocks == 3
        assert q.num_protected_blocks == 2  # No demotion needed.

    def test_linked_list_integrity_after_resize(self):
        """Verify linked list integrity after resize."""
        blocks = _make_blocks(10)
        q = FreeKVCacheBlockQueue(blocks, protected_ratio=0.5)

        # Fill protected zone.
        for b in blocks[:5]:
            q.promote(b)

        # Shrink.
        q.resize_protected(0.2)

        # Verify counts match traversal.
        prob = _probation_ids(q)
        prot = _protected_ids(q)
        assert len(prob) == q.num_probation_blocks
        assert len(prot) == q.num_protected_blocks
        assert q.num_free_blocks == len(prob) + len(prot)

        # No duplicates.
        all_ids = prob + prot
        assert len(all_ids) == len(set(all_ids))

        # Total should still be 10.
        assert len(all_ids) == 10


# ===========================================================================
# KVCacheManager — _check_cache_health() tests
# ===========================================================================

class TestCacheHealthCheck:
    """Test the cache health monitoring and adaptive protected zone resizing."""

    def _make_manager(self, num_blocks: int = 20, block_size: int = 16):
        """Create a KVCacheManager with caching enabled."""
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        return KVCacheManager(
            block_size=block_size,
            num_gpu_blocks=num_blocks,
            max_model_len=8192,
            sliding_window=None,
            enable_caching=True,
            num_preallocate_tokens=0,
        )

    def test_shrink_on_hit_rate_drop(self):
        """Protected zone shrinks when hit rate drops sharply."""
        mgr = self._make_manager(num_blocks=20)

        # Promote some blocks into protected zone first.
        for b in mgr.block_pool[:5]:
            mgr.free_block_queue.promote(b)
        assert mgr.free_block_queue.num_protected_blocks == 5

        initial_max = mgr.free_block_queue.max_protected_blocks

        # Simulate 10 high hit rates (older window).
        for _ in range(10):
            mgr._hit_rate_window.append(0.8)

        # Simulate 10 low hit rates (recent window).
        for _ in range(10):
            mgr._hit_rate_window.append(0.1)

        # Trigger health check.
        mgr._check_cache_health()

        # Protected zone should have been shrunk.
        assert mgr._protected_ratio_shrunk is True
        assert mgr.free_block_queue.max_protected_blocks < initial_max
        # Most protected blocks should have been demoted.
        assert mgr.free_block_queue.num_protected_blocks <= \
            mgr.free_block_queue.max_protected_blocks

    def test_restore_on_hit_rate_recovery(self):
        """Protected zone restores when hit rate recovers."""
        mgr = self._make_manager(num_blocks=20)

        # First: trigger a shrink.
        for _ in range(10):
            mgr._hit_rate_window.append(0.8)
        for _ in range(10):
            mgr._hit_rate_window.append(0.1)
        mgr._check_cache_health()
        assert mgr._protected_ratio_shrunk is True
        shrunk_max = mgr.free_block_queue.max_protected_blocks

        # Now: simulate recovery (high hit rates in the recent window).
        for _ in range(10):
            mgr._hit_rate_window.append(0.7)
        mgr._check_cache_health()

        assert mgr._protected_ratio_shrunk is False
        assert mgr.free_block_queue.max_protected_blocks > shrunk_max

    def test_no_shrink_when_stable_high(self):
        """No action when hit rate is stably high."""
        mgr = self._make_manager(num_blocks=20)

        for b in mgr.block_pool[:5]:
            mgr.free_block_queue.promote(b)

        initial_max = mgr.free_block_queue.max_protected_blocks

        # All high hit rates.
        for _ in range(20):
            mgr._hit_rate_window.append(0.8)

        mgr._check_cache_health()

        assert mgr._protected_ratio_shrunk is False
        assert mgr.free_block_queue.max_protected_blocks == initial_max

    def test_no_shrink_when_stable_low(self):
        """No action when hit rate is stably low (no sudden drop)."""
        mgr = self._make_manager(num_blocks=20)

        for b in mgr.block_pool[:5]:
            mgr.free_block_queue.promote(b)

        initial_max = mgr.free_block_queue.max_protected_blocks

        # All low hit rates — no sudden drop from high.
        for _ in range(20):
            mgr._hit_rate_window.append(0.2)

        mgr._check_cache_health()

        assert mgr._protected_ratio_shrunk is False
        assert mgr.free_block_queue.max_protected_blocks == initial_max

    def test_no_action_with_insufficient_data(self):
        """No action when hit rate window has < 20 samples."""
        mgr = self._make_manager(num_blocks=20)

        # Only 15 samples.
        for _ in range(15):
            mgr._hit_rate_window.append(0.1)

        mgr._check_cache_health()

        assert mgr._protected_ratio_shrunk is False

    def test_double_shrink_is_idempotent(self):
        """Calling _check_cache_health() twice with low rate doesn't
        shrink more (idempotent due to _protected_ratio_shrunk flag)."""
        mgr = self._make_manager(num_blocks=20)

        for _ in range(10):
            mgr._hit_rate_window.append(0.8)
        for _ in range(10):
            mgr._hit_rate_window.append(0.1)

        mgr._check_cache_health()
        max_after_first = mgr.free_block_queue.max_protected_blocks

        # Call again with same window.
        mgr._check_cache_health()
        max_after_second = mgr.free_block_queue.max_protected_blocks

        assert max_after_first == max_after_second

    def test_health_check_interval_triggers_correctly(self):
        """Verify that _check_cache_health is triggered every N calls."""
        mgr = self._make_manager(num_blocks=20)

        # Override interval to 5 for easier testing.
        mgr._cache_health_check_interval = 5

        # Pre-fill the window so _check_cache_health has data.
        for _ in range(10):
            mgr._hit_rate_window.append(0.8)
        for _ in range(10):
            mgr._hit_rate_window.append(0.1)

        # Manually simulate the counter logic from get_computed_blocks.
        for i in range(1, 6):
            mgr._hit_rate_window.append(0.1)
            mgr._cache_health_counter += 1
            if mgr._cache_health_counter >= mgr._cache_health_check_interval:
                mgr._check_cache_health()
                mgr._cache_health_counter = 0

        # After 5 iterations, health check should have been triggered.
        assert mgr._protected_ratio_shrunk is True


# ===========================================================================
# Integration scenario: prompt version switch
# ===========================================================================

class TestPromptVersionSwitchScenario:
    """Simulate the full prompt v1→v2 switch scenario.

    1. Fill protected zone with v1 prompt blocks (high hit rate).
    2. Switch to v2 prompt → hit rate drops.
    3. Health check triggers → protected zone shrinks.
    4. v1 blocks are demoted and evicted.
    5. v2 blocks gradually fill the cache → hit rate recovers.
    6. Protected zone is restored.
    """

    def _make_manager(self, num_blocks: int = 20, block_size: int = 16):
        from vllm.v1.core.kv_cache_manager import KVCacheManager
        return KVCacheManager(
            block_size=block_size,
            num_gpu_blocks=num_blocks,
            max_model_len=8192,
            sliding_window=None,
            enable_caching=True,
            num_preallocate_tokens=0,
        )

    def test_full_prompt_switch_flow(self):
        """End-to-end prompt version switch flow."""
        mgr = self._make_manager(num_blocks=20)

        # Phase 1: v1 prompt blocks are cached and promoted (high freq).
        v1_blocks = mgr.block_pool[:5]
        for b in v1_blocks:
            mgr.free_block_queue.promote(b)
        assert mgr.free_block_queue.num_protected_blocks == 5
        assert all(b.free_zone == "protected" for b in v1_blocks)

        initial_max = mgr.free_block_queue.max_protected_blocks

        # Phase 2: Simulate high hit rate period (v1 prompt serving).
        for _ in range(10):
            mgr._hit_rate_window.append(0.85)

        # Phase 3: Prompt switches to v2 → hit rate drops.
        for _ in range(10):
            mgr._hit_rate_window.append(0.05)

        mgr._check_cache_health()

        # Protected zone should be shrunk.
        assert mgr._protected_ratio_shrunk is True
        new_max = mgr.free_block_queue.max_protected_blocks
        assert new_max < initial_max

        # v1 blocks should be mostly demoted to probation.
        demoted_count = sum(
            1 for b in v1_blocks if b.free_zone == "probation")
        assert demoted_count >= 3  # At least 3 out of 5 demoted

        # Phase 4: Evict the demoted v1 blocks (simulating new allocations).
        evicted_v1 = []
        while mgr.free_block_queue.num_probation_blocks > 0:
            evicted = mgr.free_block_queue.popleft()
            evicted_v1.append(evicted.block_id)

        # Some v1 blocks should have been evicted from probation.
        v1_ids = {b.block_id for b in v1_blocks}
        evicted_v1_ids = set(evicted_v1) & v1_ids
        assert len(evicted_v1_ids) >= 3

        # Phase 5: v2 prompt fills the cache → hit rate recovers.
        for _ in range(10):
            mgr._hit_rate_window.append(0.75)

        mgr._check_cache_health()

        # Phase 6: Protected zone should be restored.
        assert mgr._protected_ratio_shrunk is False
        restored_max = mgr.free_block_queue.max_protected_blocks
        assert restored_max > new_max
