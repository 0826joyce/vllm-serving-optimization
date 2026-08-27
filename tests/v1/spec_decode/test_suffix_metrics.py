# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SuffixDecodeMetrics (Optimization 5)."""

from vllm.v1.spec_decode.suffix_metrics import SuffixDecodeMetrics


def test_default_rates_no_division_by_zero():
    m = SuffixDecodeMetrics()
    # All derived rates must be safe (0.0) with empty counters.
    assert m.acceptance_rate == 0.0
    assert m.avg_accepted_length == 0.0
    assert m.match_rate == 0.0
    assert m.avg_match_length == 0.0
    assert m.global_pool_hit_rate == 0.0
    assert m.avg_dyn_len == 0.0
    assert m.avg_tree_branches == 0.0


def test_acceptance_rate():
    m = SuffixDecodeMetrics()
    m.total_draft_tokens = 10
    m.total_accepted_tokens = 7
    assert abs(m.acceptance_rate - 0.7) < 1e-9


def test_avg_accepted_length():
    m = SuffixDecodeMetrics()
    m.total_proposals = 4
    m.total_accepted_tokens = 10
    assert abs(m.avg_accepted_length - 2.5) < 1e-9


def test_match_rate_and_avg_match_length():
    m = SuffixDecodeMetrics()
    m.match_found_count = 3
    m.match_not_found_count = 1
    m.match_lengths_sum = 9
    assert abs(m.match_rate - 0.75) < 1e-9
    assert abs(m.avg_match_length - 3.0) < 1e-9


def test_global_pool_hit_rate():
    m = SuffixDecodeMetrics()
    m.global_pool_queries = 5
    m.global_pool_hits = 2
    assert abs(m.global_pool_hit_rate - 0.4) < 1e-9


def test_avg_dyn_len_and_tree_branches():
    m = SuffixDecodeMetrics()
    m.dyn_len_adjust_count = 3
    m.dyn_len_sum = 12
    assert abs(m.avg_dyn_len - 4.0) < 1e-9

    m.tree_proposals = 2
    m.tree_branches_sum = 6
    assert abs(m.avg_tree_branches - 3.0) < 1e-9


def test_report_and_to_dict():
    m = SuffixDecodeMetrics()
    m.total_proposals = 2
    m.total_draft_tokens = 8
    m.total_accepted_tokens = 5
    m.match_found_count = 2
    m.match_not_found_count = 0
    m.match_lengths_sum = 6

    report = m.report()
    assert "Suffix Decode Metrics" in report
    assert "Acceptance Rate" in report

    d = m.to_dict()
    # Raw counters and derived rates both present.
    assert d["total_proposals"] == 2
    assert abs(d["acceptance_rate"] - 5 / 8) < 1e-9
    assert abs(d["match_rate"] - 1.0) < 1e-9
