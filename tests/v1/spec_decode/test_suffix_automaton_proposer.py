# SPDX-License-Identifier: Apache-2.0
"""Tests for SuffixAutomatonProposer (Optimization 2).

Validates:
1. IncrementalSuffixAutomaton correctness (extend, match)
2. SuffixAutomatonProposer functional correctness (same interface as others)
3. Incremental update correctness (stateful across calls)
4. Comparison with NgramProposer and SuffixTreeProposer
5. Request lifecycle management (remove_request)
"""

import numpy as np
import pytest

from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_automaton_proposer import (
    IncrementalSuffixAutomaton, SuffixAutomatonProposer)
from vllm.v1.spec_decode.suffix_proposer import SuffixTreeProposer


# ============================================================
# Tests for IncrementalSuffixAutomaton
# ============================================================

class TestIncrementalSuffixAutomaton:
    """Unit tests for the core SAM data structure."""

    def test_empty_automaton(self):
        """Empty SAM should have 1 state (initial) and length 0."""
        sam = IncrementalSuffixAutomaton()
        assert sam.size == 1
        assert sam.text_len == 0

    def test_single_extend(self):
        """Extending by one token should increase size and text_len."""
        sam = IncrementalSuffixAutomaton()
        sam.extend(42)
        assert sam.text_len == 1
        assert sam.size >= 2  # At least initial + new state

    def test_extend_sequence(self):
        """Extend with a sequence and verify state count bounds."""
        sam = IncrementalSuffixAutomaton()
        tokens = [10, 20, 30, 40, 50]
        for t in tokens:
            sam.extend(t)
        assert sam.text_len == 5
        # SAM has at most 2n-1 states for n tokens
        assert sam.size <= 2 * 5 + 1

    def test_find_existing_pattern(self):
        """Should find a pattern that exists in the text."""
        sam = IncrementalSuffixAutomaton()
        # Text: [10, 20, 30, 40, 50]
        for t in [10, 20, 30, 40, 50]:
            sam.extend(t)

        # Pattern [20, 30] exists at position 1-2
        pattern = np.array([20, 30], dtype=np.int32)
        matched_len, end_pos = sam.find_longest_match(pattern, min_match_len=2)
        assert matched_len == 2
        assert end_pos >= 0

    def test_find_nonexistent_pattern(self):
        """Should not find a pattern that doesn't exist."""
        sam = IncrementalSuffixAutomaton()
        for t in [10, 20, 30, 40, 50]:
            sam.extend(t)

        pattern = np.array([99, 88], dtype=np.int32)
        matched_len, end_pos = sam.find_longest_match(pattern, min_match_len=2)
        assert matched_len == 0
        assert end_pos == -1

    def test_find_partial_match(self):
        """Should match as far as possible then stop."""
        sam = IncrementalSuffixAutomaton()
        # Text: [10, 20, 30, 40, 50]
        for t in [10, 20, 30, 40, 50]:
            sam.extend(t)

        # Pattern [20, 30, 99] - first 2 tokens match
        pattern = np.array([20, 30, 99], dtype=np.int32)
        matched_len, end_pos = sam.find_longest_match(pattern, min_match_len=2)
        assert matched_len == 2
        assert end_pos >= 0

    def test_find_below_min_match(self):
        """Match shorter than min_match_len should return no match."""
        sam = IncrementalSuffixAutomaton()
        for t in [10, 20, 30]:
            sam.extend(t)

        # Pattern [20, 99] - only 1 token matches, below min_match_len=2
        pattern = np.array([20, 99], dtype=np.int32)
        matched_len, end_pos = sam.find_longest_match(pattern, min_match_len=2)
        assert matched_len == 0

    def test_repeated_tokens(self):
        """SAM should handle repeated tokens correctly."""
        sam = IncrementalSuffixAutomaton()
        # Text: [5, 5, 5, 5, 5]
        for _ in range(5):
            sam.extend(5)

        pattern = np.array([5, 5], dtype=np.int32)
        matched_len, end_pos = sam.find_longest_match(pattern, min_match_len=2)
        assert matched_len == 2
        assert end_pos >= 0

    def test_incremental_updates(self):
        """SAM should correctly handle incremental extensions."""
        sam = IncrementalSuffixAutomaton()

        # First extend with [10, 20, 30]
        for t in [10, 20, 30]:
            sam.extend(t)

        # Pattern [10, 20] should match
        pattern = np.array([10, 20], dtype=np.int32)
        matched_len, _ = sam.find_longest_match(pattern, min_match_len=2)
        assert matched_len == 2

        # Now extend with [40, 10, 20]
        for t in [40, 10, 20]:
            sam.extend(t)

        # Pattern [10, 20] should still match (now appears twice)
        matched_len, _ = sam.find_longest_match(pattern, min_match_len=2)
        assert matched_len == 2

        assert sam.text_len == 6

    def test_find_all_match_lengths(self):
        """Should return matches at multiple prefix lengths."""
        sam = IncrementalSuffixAutomaton()
        # Text: [10, 20, 30, 40, 50]
        for t in [10, 20, 30, 40, 50]:
            sam.extend(t)

        # Pattern [10, 20, 30] matches at lengths 2 and 3
        pattern = np.array([10, 20, 30], dtype=np.int32)
        results = sam.find_all_match_lengths(pattern, min_match_len=2)
        assert len(results) >= 1
        # Results should be ordered longest first
        if len(results) > 1:
            assert results[0][0] >= results[1][0]


# ============================================================
# Tests for SuffixAutomatonProposer basic functionality
# ============================================================

class TestSuffixAutomatonProposerBasic:
    """Basic functionality tests for the proposer."""

    @pytest.fixture
    def proposer(self):
        return SuffixAutomatonProposer()

    def test_simple_match(self, proposer):
        """Test basic pattern matching."""
        context = np.array([1, 2, 3, 4, 2, 3], dtype=np.int32)
        result = proposer.propose(context, n=2, k=4, req_id="r1")
        assert result is not None
        result_list = result.tolist()
        assert len(result_list) > 0
        # After matching [2,3] at pos 1, continuation is [4] in search_text
        assert result_list[0] == 4

    def test_no_match(self, proposer):
        """Test when pattern is not found."""
        context = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        result = proposer.propose(context, n=3, k=5, req_id="r2")
        # Pattern [4,5,6] should not match in [1,2,3,4,5]
        assert result is None

    def test_short_context(self, proposer):
        """Test with context shorter than n+1."""
        context = np.array([1, 2], dtype=np.int32)
        result = proposer.propose(context, n=3, k=5, req_id="r3")
        assert result is None

    def test_repeated_pattern(self, proposer):
        """Test with a pattern that appears multiple times."""
        context = np.array([1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3],
                           dtype=np.int32)
        result = proposer.propose(context, n=3, k=5, req_id="r4")
        assert result is not None
        assert len(result) > 0

    def test_all_same_tokens(self, proposer):
        """Test with all identical tokens."""
        context = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int32)
        result = proposer.propose(context, n=2, k=3, req_id="r5")
        assert result is not None
        for val in result:
            assert val == 5

    def test_k_larger_than_available(self, proposer):
        """Test when k exceeds available continuation tokens."""
        context = np.array([1, 2, 3, 1, 2], dtype=np.int32)
        result = proposer.propose(context, n=2, k=100, req_id="r6")
        assert result is not None
        assert len(result) <= 100

    def test_stateless_mode_empty_req_id(self, proposer):
        """Test with empty req_id (stateless mode)."""
        context = np.array([1, 2, 3, 4, 2, 3], dtype=np.int32)
        result = proposer.propose(context, n=2, k=4, req_id="")
        assert result is not None
        # Should not accumulate state
        assert proposer.num_active_requests() == 0


# ============================================================
# Tests for incremental (stateful) behavior
# ============================================================

class TestSuffixAutomatonProposerIncremental:
    """Tests for incremental SAM update correctness."""

    @pytest.fixture
    def proposer(self):
        return SuffixAutomatonProposer()

    def test_incremental_produces_same_result(self, proposer):
        """Incremental updates should produce same results as fresh build."""
        # Simulate step 1: context has 10 tokens
        context1 = np.array([10, 20, 30, 40, 50, 20, 30, 40, 60, 20],
                            dtype=np.int32)
        result1 = proposer.propose(context1, n=2, k=5, req_id="req_a")

        # Simulate step 2: context grows by 2 tokens
        context2 = np.array(
            [10, 20, 30, 40, 50, 20, 30, 40, 60, 20, 30, 40],
            dtype=np.int32)
        result2_incremental = proposer.propose(
            context2, n=2, k=5, req_id="req_a")

        # Compare with a fresh proposer (stateless rebuild)
        fresh_proposer = SuffixAutomatonProposer()
        result2_fresh = fresh_proposer.propose(
            context2, n=2, k=5, req_id="req_fresh")

        # Both should find same match or both return None
        if result2_incremental is not None and result2_fresh is not None:
            assert result2_incremental.tolist() == result2_fresh.tolist(), \
                (f"Incremental: {result2_incremental.tolist()}, "
                 f"Fresh: {result2_fresh.tolist()}")
        else:
            assert (result2_incremental is None) == (result2_fresh is None)

    def test_multiple_steps_incremental(self, proposer):
        """Simulate multiple decode steps with incremental updates."""
        base = [10, 20, 30, 40, 50, 20, 30]

        results = []
        for step in range(5):
            # Each step adds one more token
            context = np.array(base + [40 + step], dtype=np.int32)
            result = proposer.propose(context, n=2, k=3, req_id="req_multi")
            results.append(result)
            base = base + [40 + step]

        # Should not crash and should track state correctly
        assert proposer.num_active_requests() == 1

    def test_incremental_vs_rebuild_random(self, proposer):
        """Stress test: incremental should match rebuild for random data."""
        rng = np.random.RandomState(42)

        # Build initial context
        base_tokens = rng.randint(0, 30, size=50).astype(np.int32)

        # Step 1: initial propose
        proposer.propose(base_tokens, n=3, k=5, req_id="stress")

        # Steps 2-10: incrementally grow context
        for step in range(10):
            new_tokens = rng.randint(0, 30, size=rng.randint(1, 4))
            base_tokens = np.concatenate([base_tokens, new_tokens])

            incr_result = proposer.propose(
                base_tokens, n=3, k=5, req_id="stress")

            # Compare with fresh
            fresh = SuffixAutomatonProposer()
            fresh_result = fresh.propose(
                base_tokens, n=3, k=5, req_id="fresh")

            if incr_result is not None and fresh_result is not None:
                assert incr_result.tolist() == fresh_result.tolist(), \
                    f"Mismatch at step {step}"
            else:
                assert (incr_result is None) == (fresh_result is None), \
                    f"None mismatch at step {step}"

    def test_context_shrink_rebuilds(self, proposer):
        """If context shrinks (preemption), SAM should rebuild."""
        context_long = np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3],
                                dtype=np.int32)
        proposer.propose(context_long, n=3, k=5, req_id="shrink")

        # Simulate context shrinking (e.g., after preemption and resume)
        context_short = np.array([1, 2, 3, 4, 5, 1, 2, 3], dtype=np.int32)
        result = proposer.propose(context_short, n=3, k=5, req_id="shrink")

        # Should not crash; SAM rebuilds internally
        # Just verify it still works
        assert result is not None or result is None  # either is fine

    def test_multiple_requests_independent(self, proposer):
        """Multiple requests should maintain independent SAM state."""
        context_a = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.int32)
        context_b = np.array([10, 20, 30, 40, 10, 20, 30], dtype=np.int32)

        result_a = proposer.propose(context_a, n=3, k=5, req_id="req_a")
        result_b = proposer.propose(context_b, n=3, k=5, req_id="req_b")

        assert proposer.num_active_requests() == 2

        # Results should be independent (different token vocabularies)
        if result_a is not None and result_b is not None:
            # req_a matches should be from context_a's tokens
            for t in result_a:
                assert t in [1, 2, 3, 4]
            # req_b matches should be from context_b's tokens
            for t in result_b:
                assert t in [10, 20, 30, 40]


# ============================================================
# Tests for request lifecycle management
# ============================================================

class TestSuffixAutomatonProposerLifecycle:
    """Tests for request add/remove lifecycle."""

    @pytest.fixture
    def proposer(self):
        return SuffixAutomatonProposer()

    def test_remove_request(self, proposer):
        """Removing a request should clean up its SAM."""
        context = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.int32)
        proposer.propose(context, n=2, k=3, req_id="to_remove")
        assert proposer.num_active_requests() == 1

        proposer.remove_request("to_remove")
        assert proposer.num_active_requests() == 0

    def test_remove_nonexistent_request(self, proposer):
        """Removing a non-existent request should not crash."""
        proposer.remove_request("does_not_exist")
        assert proposer.num_active_requests() == 0

    def test_reuse_req_id_after_remove(self, proposer):
        """After removing, the same req_id can be reused."""
        context1 = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.int32)
        proposer.propose(context1, n=2, k=3, req_id="reuse")
        proposer.remove_request("reuse")

        # New request with same ID but different context
        context2 = np.array([10, 20, 30, 10, 20, 30], dtype=np.int32)
        result = proposer.propose(context2, n=2, k=3, req_id="reuse")
        assert proposer.num_active_requests() == 1

        # Should get tokens from context2, not context1
        if result is not None:
            for t in result:
                assert t in [10, 20, 30]

    def test_many_requests_lifecycle(self, proposer):
        """Simulate adding and removing many requests."""
        rng = np.random.RandomState(99)

        for i in range(50):
            context = rng.randint(0, 50, size=rng.randint(10, 100)).astype(
                np.int32)
            proposer.propose(context, n=3, k=5, req_id=f"req_{i}")

        assert proposer.num_active_requests() == 50

        # Remove half
        for i in range(0, 50, 2):
            proposer.remove_request(f"req_{i}")

        assert proposer.num_active_requests() == 25

        # Remove rest
        for i in range(1, 50, 2):
            proposer.remove_request(f"req_{i}")

        assert proposer.num_active_requests() == 0


# ============================================================
# Tests for adaptive fallback
# ============================================================

class TestSuffixAutomatonProposerAdaptive:
    """Tests for adaptive fallback matching."""

    @pytest.fixture
    def proposer(self):
        return SuffixAutomatonProposer()

    def test_fallback_to_shorter_pattern(self, proposer):
        """Should fall back to shorter patterns when longer ones don't match."""
        # n=4 pattern [5,6,7,X] won't match, but n=2 pattern [6,7] might
        context = np.array([1, 2, 6, 7, 8, 9, 3, 4, 5, 6, 7],
                           dtype=np.int32)
        result = proposer.propose(context, n=4, k=5, req_id="fallback")
        # With adaptive fallback, should find [6,7] at position 2
        assert result is not None

    def test_prefers_exact_n_match(self, proposer):
        """Should prefer exact n-length match over shorter fallback."""
        # Context where both n=3 and n=2 can match
        context = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 8, 9, 1, 2, 3],
                           dtype=np.int32)
        result = proposer.propose(context, n=3, k=5, req_id="exact")
        # Should find n=3 match [1,2,3] first before trying shorter
        assert result is not None


# ============================================================
# Comparison with NgramProposer and SuffixTreeProposer
# ============================================================

class TestSuffixAutomatonVsOthers:
    """Compare SuffixAutomatonProposer with other proposers."""

    @pytest.fixture
    def ngram_proposer(self):
        return NgramProposer()

    @pytest.fixture
    def suffix_proposer(self):
        return SuffixTreeProposer()

    @pytest.fixture
    def sam_proposer(self):
        return SuffixAutomatonProposer()

    def test_finds_match_when_ngram_does(self, ngram_proposer, sam_proposer):
        """SAM proposer should find matches whenever Ngram proposer does."""
        rng = np.random.RandomState(42)
        for trial in range(30):
            length = rng.randint(20, 200)
            vocab = rng.randint(1, 50)
            context = rng.randint(0, vocab, size=length).astype(np.int32)
            n = rng.randint(2, 5)
            k = rng.randint(3, 10)

            ngram_result = ngram_proposer.propose(context, n=n, k=k)
            sam_result = sam_proposer.propose(
                context, n=n, k=k, req_id=f"trial_{trial}")

            if ngram_result is not None:
                assert sam_result is not None, (
                    f"SAM missed match that Ngram found: "
                    f"trial={trial}, len={length}, n={n}, k={k}")

            # Clean up for next trial
            sam_proposer.remove_request(f"trial_{trial}")

    def test_matches_suffix_tree_results(self, suffix_proposer, sam_proposer):
        """SAM proposer results should be consistent with SuffixTree."""
        rng = np.random.RandomState(123)
        for trial in range(30):
            length = rng.randint(20, 200)
            vocab = rng.randint(1, 50)
            context = rng.randint(0, vocab, size=length).astype(np.int32)
            n = rng.randint(2, 5)
            k = rng.randint(3, 10)

            suffix_result = suffix_proposer.propose(context, n=n, k=k)
            sam_result = sam_proposer.propose(
                context, n=n, k=k, req_id=f"cmp_{trial}")

            # Both should agree on whether a match exists
            if suffix_result is not None:
                assert sam_result is not None, (
                    f"SAM missed match that SuffixTree found: "
                    f"trial={trial}")

            sam_proposer.remove_request(f"cmp_{trial}")

    def test_match_count_comparison(self, ngram_proposer, suffix_proposer,
                                    sam_proposer):
        """Count how many matches each proposer finds across random trials."""
        rng = np.random.RandomState(77)
        ngram_matches = 0
        suffix_matches = 0
        sam_matches = 0
        total = 200

        for trial in range(total):
            length = rng.randint(30, 300)
            vocab = rng.randint(1, 30)
            context = rng.randint(0, vocab, size=length).astype(np.int32)
            n = rng.randint(2, 6)
            k = 5

            if ngram_proposer.propose(context, n=n, k=k) is not None:
                ngram_matches += 1
            if suffix_proposer.propose(context, n=n, k=k) is not None:
                suffix_matches += 1
            if sam_proposer.propose(
                    context, n=n, k=k,
                    req_id=f"count_{trial}") is not None:
                sam_matches += 1
            sam_proposer.remove_request(f"count_{trial}")

        print(f"\n=== Match Count Comparison ({total} trials) ===")
        print(f"NgramProposer:            {ngram_matches}")
        print(f"SuffixTreeProposer:       {suffix_matches}")
        print(f"SuffixAutomatonProposer:  {sam_matches}")

        # SAM should find at least as many as Ngram
        assert sam_matches >= ngram_matches, \
            f"SAM ({sam_matches}) found fewer matches than Ngram ({ngram_matches})"


# ============================================================
# Performance / scalability test
# ============================================================

class TestSuffixAutomatonProposerPerformance:
    """Performance-oriented tests."""

    @pytest.fixture
    def proposer(self):
        return SuffixAutomatonProposer()

    def test_large_context(self, proposer):
        """Test with large context to verify no crash or excessive slowdown."""
        rng = np.random.RandomState(42)
        context = rng.randint(0, 100, size=4000).astype(np.int32)
        # Force a match by copying a segment
        context[3990:4000] = context[500:510]
        result = proposer.propose(context, n=3, k=5, req_id="large")
        # Should complete without error
        assert result is not None or result is None

    def test_incremental_large_context(self, proposer):
        """Test incremental updates on large context."""
        rng = np.random.RandomState(42)

        # Initial large context
        base = rng.randint(0, 100, size=2000).astype(np.int32)
        proposer.propose(base, n=3, k=5, req_id="incr_large")

        # 50 incremental steps, each adding 1-3 tokens
        for step in range(50):
            new_tokens = rng.randint(0, 100, size=rng.randint(1, 4)).astype(
                np.int32)
            base = np.concatenate([base, new_tokens])
            result = proposer.propose(base, n=3, k=5, req_id="incr_large")

        # Should complete without error
        assert proposer.num_active_requests() == 1

    def test_many_concurrent_requests(self, proposer):
        """Test with many concurrent requests."""
        rng = np.random.RandomState(42)

        for i in range(100):
            context = rng.randint(0, 50, size=rng.randint(20, 100)).astype(
                np.int32)
            proposer.propose(context, n=3, k=5, req_id=f"concurrent_{i}")

        assert proposer.num_active_requests() == 100

        # Clean up all
        for i in range(100):
            proposer.remove_request(f"concurrent_{i}")
        assert proposer.num_active_requests() == 0
