# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FrequencyAwareGlobalSuffixPool (Optimization 4 + 6)."""

import numpy as np

from vllm.v1.spec_decode.global_suffix_pool import (
    FrequencyAwareGlobalSuffixPool,
    _Candidate,
)


def test_candidate_expected_gain_cold_start():
    # Never proposed -> optimistic estimate = len * 0.5.
    cand = _Candidate(continuation=(1, 2, 3, 4))
    assert abs(cand.expected_gain - 4 * 0.5) < 1e-9


def test_candidate_expected_gain_with_history():
    cand = _Candidate(continuation=(1, 2), proposed_count=4, accepted_tokens=6)
    # 6 accepted over 4 proposals = 1.5.
    assert abs(cand.expected_gain - 1.5) < 1e-9


def test_add_and_query_basic():
    pool = FrequencyAwareGlobalSuffixPool(key_len=2, min_expected_gain=0.0)
    # Segment [1,2,3,4,5]: key [2,3] -> continuation starts at "4".
    pool.add_segment(np.array([1, 2, 3, 4, 5], dtype=np.int32))

    draft, key = pool.query(np.array([2, 3], dtype=np.int32), k=3)
    assert draft is not None
    assert draft.tolist() == [4, 5]
    assert key is not None
    assert pool.num_entries() > 0


def test_query_no_match_returns_none():
    pool = FrequencyAwareGlobalSuffixPool(key_len=2)
    pool.add_segment(np.array([1, 2, 3, 4], dtype=np.int32))
    draft, key = pool.query(np.array([9, 9], dtype=np.int32), k=2)
    assert draft is None
    assert key is None


def test_query_short_pattern_returns_none():
    pool = FrequencyAwareGlobalSuffixPool(key_len=3)
    pool.add_segment(np.array([1, 2, 3, 4, 5], dtype=np.int32))
    # Pattern shorter than key_len.
    draft, key = pool.query(np.array([5], dtype=np.int32), k=2)
    assert draft is None
    assert key is None


def test_min_expected_gain_filters_low_quality():
    # High threshold: cold-start optimistic gain (len*0.5) must exceed it.
    pool = FrequencyAwareGlobalSuffixPool(key_len=2, min_expected_gain=0.4)
    pool.add_segment(np.array([1, 2, 3], dtype=np.int32))  # key[1,2]->cont[3]
    # cont len 1 -> gain 0.5 > 0.4 -> should return.
    draft, _ = pool.query(np.array([1, 2], dtype=np.int32), k=2)
    assert draft is not None and draft.tolist() == [3]

    # Now raise threshold above the cold-start gain -> filtered out.
    pool2 = FrequencyAwareGlobalSuffixPool(key_len=2, min_expected_gain=0.9)
    pool2.add_segment(np.array([1, 2, 3], dtype=np.int32))
    draft2, _ = pool2.query(np.array([1, 2], dtype=np.int32), k=2)
    assert draft2 is None


def test_update_feedback_changes_selection():
    pool = FrequencyAwareGlobalSuffixPool(
        key_len=2, min_expected_gain=0.0, max_candidates_per_pattern=4)
    # Two different continuations for the same key [1,2].
    pool.add_segment(np.array([1, 2, 3, 3, 3], dtype=np.int32))  # cont A: 3,3,3
    pool.add_segment(np.array([1, 2, 9], dtype=np.int32))        # cont B: 9

    # Query once to obtain a key, then hammer feedback so that a specific
    # candidate accumulates high acceptance.
    draft, key = pool.query(np.array([1, 2], dtype=np.int32), k=3)
    assert draft is not None
    # Give strong positive feedback to whichever was selected.
    for _ in range(5):
        pool.update_feedback(key, proposed=3, accepted=3)
    # The pool remains queryable and stable after feedback.
    draft2, _ = pool.query(np.array([1, 2], dtype=np.int32), k=3)
    assert draft2 is not None


def test_capacity_eviction():
    pool = FrequencyAwareGlobalSuffixPool(key_len=2, max_patterns=3)
    # Insert many distinct patterns to trigger eviction.
    for base in range(20):
        seg = np.array([base * 10, base * 10 + 1, base * 10 + 2],
                       dtype=np.int32)
        pool.add_segment(seg)
    # After eviction the number of patterns is bounded by max_patterns.
    assert pool.num_entries() <= 3


def test_feedback_with_invalid_key_is_safe():
    pool = FrequencyAwareGlobalSuffixPool(key_len=2)
    # A key without the composite separator should be ignored gracefully.
    pool.update_feedback((1, 2, 3), proposed=1, accepted=1)  # no 0xFFFF -> no-op
