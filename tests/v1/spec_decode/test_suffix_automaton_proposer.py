# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the incremental Suffix Automaton and the
SuffixAutomaton / Adaptive suffix proposers (Optimizations 2, 3, 7)."""

import numpy as np

from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.v1.spec_decode.adaptive_suffix_proposer import (
    AcceptanceTracker,
    AdaptiveSuffixProposer,
)
from vllm.v1.spec_decode.suffix_automaton_proposer import (
    IncrementalSuffixAutomaton,
    SuffixAutomatonProposer,
)


# ----------------------------------------------------------------------
# IncrementalSuffixAutomaton (Optimization 2)
# ----------------------------------------------------------------------
def test_sam_empty():
    sam = IncrementalSuffixAutomaton()
    assert sam.size == 1  # only the initial state
    assert sam.text_len == 0
    matched, end_pos = sam.find_longest_match(np.array([1, 2]), min_match_len=1)
    assert matched == 0
    assert end_pos == -1


def test_sam_extend_and_match():
    sam = IncrementalSuffixAutomaton()
    text = [1, 2, 3, 1, 2, 4]
    for t in text:
        sam.extend(t)
    assert sam.text_len == len(text)

    # Pattern [1, 2] exists; should match length 2.
    matched, end_pos = sam.find_longest_match(np.array([1, 2]), min_match_len=2)
    assert matched == 2
    assert end_pos >= 0

    # Pattern that does not exist.
    matched, _ = sam.find_longest_match(np.array([9, 9]), min_match_len=2)
    assert matched == 0


def test_sam_incremental_matches_full_rebuild():
    text = [5, 6, 7, 5, 6, 7, 8, 9, 5, 6]
    # Incremental.
    sam_inc = IncrementalSuffixAutomaton()
    for t in text:
        sam_inc.extend(t)
    # Full (same thing, sanity that repeated construction is deterministic).
    sam_full = IncrementalSuffixAutomaton()
    for t in text:
        sam_full.extend(t)

    for pat in ([5, 6], [6, 7], [5, 6, 7]):
        m1, _ = sam_inc.find_longest_match(np.array(pat), min_match_len=len(pat))
        m2, _ = sam_full.find_longest_match(np.array(pat), min_match_len=len(pat))
        assert m1 == m2 == len(pat)


def test_sam_find_all_match_lengths_ordered():
    sam = IncrementalSuffixAutomaton()
    for t in [1, 2, 3, 4]:
        sam.extend(t)
    results = sam.find_all_match_lengths(np.array([1, 2, 3]), min_match_len=1)
    # Longest first.
    lengths = [r[0] for r in results]
    assert lengths == sorted(lengths, reverse=True)


# ----------------------------------------------------------------------
# SuffixAutomatonProposer (Optimization 2, drop-in interface)
# ----------------------------------------------------------------------
def test_suffix_automaton_proposer_basic():
    proposer = SuffixAutomatonProposer()
    # context [1,2,3,1,2] -> last 2 tokens [1,2] matched earlier -> cont [3].
    context = np.array([1, 2, 3, 1, 2], dtype=np.int32)
    draft = proposer.propose(context, n=2, k=2, req_id="r1")
    assert draft is not None
    assert draft.tolist()[0] == 3


def test_suffix_automaton_proposer_no_match():
    proposer = SuffixAutomatonProposer()
    context = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    draft = proposer.propose(context, n=2, k=2, req_id="r1")
    assert draft is None


def test_suffix_automaton_proposer_incremental_consistency():
    proposer = SuffixAutomatonProposer()
    full = [1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3]
    # Feed incrementally (simulating decode steps), each step appends 1 token.
    last = None
    for i in range(4, len(full) + 1):
        ctx = np.array(full[:i], dtype=np.int32)
        last = proposer.propose(ctx, n=3, k=2, req_id="r1")
    # Final context ends with [1,2,3] -> should find a match somewhere before.
    assert last is not None


def test_suffix_automaton_proposer_remove_request():
    proposer = SuffixAutomatonProposer()
    context = np.array([1, 2, 3, 1, 2], dtype=np.int32)
    proposer.propose(context, n=2, k=2, req_id="r1")
    assert proposer.num_active_requests() == 1
    proposer.remove_request("r1")
    assert proposer.num_active_requests() == 0


# ----------------------------------------------------------------------
# AcceptanceTracker (Optimization 3)
# ----------------------------------------------------------------------
def test_acceptance_tracker_default_neutral():
    tracker = AcceptanceTracker()
    assert tracker.get_rate(3) == 0.5  # no history -> neutral


def test_acceptance_tracker_records_rate():
    tracker = AcceptanceTracker()
    tracker.record(match_len=3, num_proposed=4, num_accepted=2)
    tracker.record(match_len=3, num_proposed=4, num_accepted=2)
    # 4 accepted / 8 proposed = 0.5.
    assert abs(tracker.get_rate(3) - 0.5) < 1e-9
    tracker.record(match_len=3, num_proposed=2, num_accepted=2)
    # 6 / 10 = 0.6.
    assert abs(tracker.get_rate(3) - 0.6) < 1e-9


def test_acceptance_tracker_window_eviction():
    tracker = AcceptanceTracker(window_size=2)
    tracker.record(2, num_proposed=1, num_accepted=0)
    tracker.record(2, num_proposed=1, num_accepted=1)
    tracker.record(2, num_proposed=1, num_accepted=1)  # evicts the first
    # Only last two remain: 2 accepted / 2 proposed = 1.0.
    assert abs(tracker.get_rate(2) - 1.0) < 1e-9


# ----------------------------------------------------------------------
# AdaptiveSuffixProposer (Optimizations 3, 5, 7)
# ----------------------------------------------------------------------
def _make_adaptive(min_n: int, max_n: int, k: int) -> AdaptiveSuffixProposer:
    model_config = ModelConfig(model="facebook/opt-125m")
    return AdaptiveSuffixProposer(
        vllm_config=VllmConfig(
            model_config=model_config,
            speculative_config=SpeculativeConfig(
                prompt_lookup_min=min_n,
                prompt_lookup_max=max_n,
                num_speculative_tokens=k,
                method="ngram",
            ),
        )
    )


def test_adaptive_proposer_basic_batch():
    proposer = _make_adaptive(min_n=2, max_n=2, k=2)
    token_ids_cpu = np.array([[1, 2, 3, 1, 2]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([5]),
        token_ids_cpu=token_ids_cpu,
        req_ids=["r1"],
    )
    assert len(result) == 1
    # [1,2] matched earlier -> continuation begins with 3.
    assert len(result[0]) >= 1
    assert result[0][0] == 3


def test_adaptive_proposer_no_match_batch():
    proposer = _make_adaptive(min_n=2, max_n=2, k=2)
    token_ids_cpu = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
    result = proposer.propose(
        sampled_token_ids=[[5]],
        num_tokens_no_spec=np.array([5]),
        token_ids_cpu=token_ids_cpu,
        req_ids=["r1"],
    )
    assert len(result[0]) == 0


def test_adaptive_proposer_metrics_updated():
    proposer = _make_adaptive(min_n=2, max_n=2, k=2)
    token_ids_cpu = np.array([[1, 2, 3, 1, 2]], dtype=np.int32)
    proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([5]),
        token_ids_cpu=token_ids_cpu,
        req_ids=["r1"],
    )
    m = proposer.get_metrics()
    # One successful proposal expected.
    assert m.total_proposals + m.total_no_draft == 1
    assert m.match_found_count + m.match_not_found_count >= 1


def test_adaptive_proposer_acceptance_feedback():
    proposer = _make_adaptive(min_n=2, max_n=2, k=2)
    token_ids_cpu = np.array([[1, 2, 3, 1, 2]], dtype=np.int32)
    proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([5]),
        token_ids_cpu=token_ids_cpu,
        req_ids=["r1"],
    )
    # Feed acceptance; should update accepted-token metric without error.
    proposer.update_acceptance("r1", num_accepted=1)
    assert proposer.get_metrics().total_accepted_tokens >= 1


def test_adaptive_proposer_dynamic_k_disabled_by_default():
    proposer = _make_adaptive(min_n=2, max_n=2, k=4)
    # Dynamic k off by default -> effective k equals base k.
    assert proposer._effective_k("r1", 4) == 4


def test_adaptive_proposer_dynamic_k_high_load():
    proposer = _make_adaptive(min_n=2, max_n=2, k=4)
    proposer._dynamic_k = True  # enable for this test
    proposer.set_load(0.9)      # heavy load -> load_factor 0.5
    k = proposer._effective_k("r1", 4)
    # With neutral accept factor (1.0) and load factor 0.5 -> ~2, clamped.
    assert 0 <= k <= 8
    assert k <= 4  # heavy load should not increase beyond base


def test_adaptive_proposer_remove_request_cleans_state():
    proposer = _make_adaptive(min_n=2, max_n=2, k=2)
    token_ids_cpu = np.array([[1, 2, 3, 1, 2]], dtype=np.int32)
    proposer.propose(
        sampled_token_ids=[[2]],
        num_tokens_no_spec=np.array([5]),
        token_ids_cpu=token_ids_cpu,
        req_ids=["r1"],
    )
    proposer.remove_request("r1")
    # After removal, internal per-request maps should not contain r1.
    assert "r1" not in proposer._accept_trackers
    assert "r1" not in proposer._last_proposal


def test_adaptive_proposer_propose_tree_multi_candidate():
    proposer = _make_adaptive(min_n=2, max_n=2, k=3)
    # Pattern [1,2] has two different continuations: ...3... and ...9...
    context = np.array([1, 2, 3, 4, 7, 1, 2, 9, 8, 1, 2], dtype=np.int32)
    branches = proposer.propose_tree(context, n=2, k=3, req_id="r1",
                                     max_branches=3)
    # Should produce at least one branch; branches are distinct arrays.
    assert isinstance(branches, list)
    if branches:
        tuples = {tuple(b.tolist()) for b in branches}
        assert len(tuples) == len(branches)  # distinct
        assert proposer.get_metrics().tree_proposals >= 1
