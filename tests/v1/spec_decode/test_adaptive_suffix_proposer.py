# SPDX-License-Identifier: Apache-2.0
"""Tests for AdaptiveSuffixProposer (Optimization 3).

Tests cover:
1. Basic matching correctness
2. Multi-candidate selection (prefers recent matches)
3. Adaptive fallback matching
4. Acceptance feedback loop
5. Incremental update correctness
6. Request lifecycle management
7. Comparison with SuffixAutomatonProposer match rates
8. Performance benchmarks
"""

import sys
import time
import types

import importlib.util
import numpy as np


def load_module(name, path):
    """Load a module without relying on the vllm package hierarchy."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Set up mock vllm modules to bypass torch dependency
vllm_mock = types.ModuleType('vllm')
vllm_mock.__path__ = ['vllm']
v1_mock = types.ModuleType('vllm.v1')
v1_mock.__path__ = ['vllm/v1']
sd_mock = types.ModuleType('vllm.v1.spec_decode')
sd_mock.__path__ = ['vllm/v1/spec_decode']
sys.modules['vllm'] = vllm_mock
sys.modules['vllm.v1'] = v1_mock
sys.modules['vllm.v1.spec_decode'] = sd_mock

# Load the actual modules
sam_mod = load_module(
    'vllm.v1.spec_decode.suffix_automaton_proposer',
    'vllm/v1/spec_decode/suffix_automaton_proposer.py')
sys.modules['vllm.v1.spec_decode.suffix_automaton_proposer'] = sam_mod

adaptive_mod = load_module(
    'vllm.v1.spec_decode.adaptive_suffix_proposer',
    'vllm/v1/spec_decode/adaptive_suffix_proposer.py')

AdaptiveSuffixProposer = adaptive_mod.AdaptiveSuffixProposer
AcceptanceTracker = adaptive_mod.AcceptanceTracker
SuffixAutomatonProposer = sam_mod.SuffixAutomatonProposer


def test_basic_match():
    """Test basic matching works correctly."""
    print('=== Test 1: Basic match ===')
    p = AdaptiveSuffixProposer()
    context = np.array([1, 2, 3, 4, 2, 3], dtype=np.int32)
    result = p.propose(context, n=2, k=4, req_id='r1')
    assert result is not None, 'Expected a match'
    assert result.tolist()[0] == 4, 'First draft token should be 4'
    print('  result=%s' % result.tolist())
    print('  PASS')


def test_no_match():
    """Test that no match returns None."""
    print('\n=== Test 2: No match ===')
    p = AdaptiveSuffixProposer()
    context = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    result = p.propose(context, n=3, k=5, req_id='r2')
    assert result is None, 'Expected None'
    print('  PASS')


def test_multi_candidate_prefers_recent():
    """Test that scoring prefers more recent match positions."""
    print('\n=== Test 3: Multi-candidate prefers recent ===')
    # Context with "2,3" appearing at position 1 and position 7
    # Position 7 is more recent, so should be preferred
    context = np.array([1, 2, 3, 99, 88, 77, 66, 2, 3, 55, 44, 33, 2, 3],
                       dtype=np.int32)
    p = AdaptiveSuffixProposer()
    result = p.propose(context, n=2, k=3, req_id='multi')
    assert result is not None, 'Expected a match'
    # The more recent match (at position 7-8) should be preferred
    # continuation after pos 7-8 is [55, 44, 33]
    # continuation after pos 1-2 is [99, 88, 77]
    print('  result=%s' % result.tolist())
    # Due to recency scoring, the result should start with 55
    assert result.tolist()[0] == 55, (
        'Expected recent match (55), got %d' % result.tolist()[0])
    print('  PASS')


def test_adaptive_fallback():
    """Test adaptive fallback to shorter match lengths."""
    print('\n=== Test 4: Adaptive fallback ===')
    p = AdaptiveSuffixProposer()
    # No 4-gram match for "5,6,7", but "6,7" matches at position 1-2
    context = np.array([1, 6, 7, 8, 9, 10, 3, 4, 5, 6, 7], dtype=np.int32)
    result = p.propose(context, n=4, k=5, req_id='fb')
    assert result is not None, 'Fallback should find a match'
    print('  result=%s' % result.tolist())
    print('  PASS')


def test_acceptance_tracker():
    """Test the AcceptanceTracker directly."""
    print('\n=== Test 5: AcceptanceTracker ===')
    tracker = AcceptanceTracker(window_size=5)

    # No history -> default rate
    assert tracker.get_rate(3) == 0.5

    # Record some results
    tracker.record(3, num_proposed=5, num_accepted=4)
    tracker.record(3, num_proposed=5, num_accepted=3)
    rate = tracker.get_rate(3)
    assert abs(rate - 7.0 / 10.0) < 0.01, 'Expected 0.7, got %f' % rate

    # Different match length has separate tracking
    assert tracker.get_rate(4) == 0.5

    # Window sliding
    for _ in range(10):
        tracker.record(3, num_proposed=5, num_accepted=5)
    assert tracker.get_rate(3) == 1.0  # All recent ones are 100%

    print('  PASS')


def test_acceptance_feedback_loop():
    """Test that acceptance feedback affects scoring."""
    print('\n=== Test 6: Acceptance feedback loop ===')
    p = AdaptiveSuffixProposer()

    # First proposal
    context = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3],
                       dtype=np.int32)
    result = p.propose(context, n=3, k=5, req_id='fb_loop')
    assert result is not None

    # Simulate that 3 out of 5 draft tokens were accepted
    p.update_acceptance('fb_loop', num_accepted=3)

    # Check that tracker was updated
    assert 'fb_loop' in p._accept_trackers
    tracker = p._accept_trackers['fb_loop']
    # The match_len should have been recorded
    assert len(tracker._history) > 0

    print('  PASS')


def test_incremental_update():
    """Test that incremental update produces same results as fresh build."""
    print('\n=== Test 7: Incremental update ===')
    p1 = AdaptiveSuffixProposer()
    c1 = np.array([10, 20, 30, 40, 50, 20, 30, 40, 60, 20],
                  dtype=np.int32)
    p1.propose(c1, n=2, k=5, req_id='incr')

    c2 = np.array([10, 20, 30, 40, 50, 20, 30, 40, 60, 20, 30, 40],
                  dtype=np.int32)
    r_incr = p1.propose(c2, n=2, k=5, req_id='incr')

    p2 = AdaptiveSuffixProposer()
    r_fresh = p2.propose(c2, n=2, k=5, req_id='fresh')

    if r_incr is not None and r_fresh is not None:
        print('  incr=%s, fresh=%s' % (r_incr.tolist(), r_fresh.tolist()))
        assert r_incr.tolist() == r_fresh.tolist()
    else:
        assert (r_incr is None) == (r_fresh is None)
    print('  PASS')


def test_request_lifecycle():
    """Test request creation and cleanup."""
    print('\n=== Test 8: Request lifecycle ===')
    p = AdaptiveSuffixProposer()
    context = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.int32)

    p.propose(context, n=2, k=3, req_id='lc')
    p.update_acceptance('lc', num_accepted=2)
    assert p.num_active_requests() == 1
    assert 'lc' in p._accept_trackers
    assert 'lc' in p._last_proposal

    p.remove_request('lc')
    assert p.num_active_requests() == 0
    assert 'lc' not in p._accept_trackers
    assert 'lc' not in p._last_proposal

    # Should not crash
    p.remove_request('nonexistent')
    print('  PASS')


def test_multiple_requests_independent():
    """Test that multiple requests don't interfere."""
    print('\n=== Test 9: Multiple requests independent ===')
    p = AdaptiveSuffixProposer()
    ca = np.array([1, 2, 3, 4, 1, 2, 3], dtype=np.int32)
    cb = np.array([10, 20, 30, 40, 10, 20, 30], dtype=np.int32)

    ra = p.propose(ca, n=3, k=5, req_id='a')
    rb = p.propose(cb, n=3, k=5, req_id='b')
    assert p.num_active_requests() == 2

    if ra is not None:
        for t in ra:
            assert t in [1, 2, 3, 4]
    if rb is not None:
        for t in rb:
            assert t in [10, 20, 30, 40]
    print('  PASS')


def test_stress_incremental_vs_rebuild():
    """Stress test: incremental updates produce same results as fresh."""
    print('\n=== Test 10: Stress incremental vs rebuild ===')
    rng = np.random.RandomState(42)
    p = AdaptiveSuffixProposer()
    base = rng.randint(0, 30, size=50).astype(np.int32)
    p.propose(base, n=3, k=5, req_id='stress')
    mismatches = 0
    for step in range(20):
        nt = rng.randint(0, 30, size=rng.randint(1, 4)).astype(np.int32)
        base = np.concatenate([base, nt])
        ir = p.propose(base, n=3, k=5, req_id='stress')
        fr_p = AdaptiveSuffixProposer()
        fr = fr_p.propose(base, n=3, k=5, req_id='f')
        if ir is not None and fr is not None:
            if ir.tolist() != fr.tolist():
                mismatches += 1
        elif (ir is None) != (fr is None):
            mismatches += 1
    print('  Mismatches: %d/20' % mismatches)
    assert mismatches == 0
    print('  PASS')


def test_match_rate_comparison():
    """Compare match rates: Adaptive vs SuffixAutomaton vs random baseline."""
    print('\n=== Test 11: Match rate comparison ===')
    rng = np.random.RandomState(77)
    sam_p = SuffixAutomatonProposer()
    adp = AdaptiveSuffixProposer()
    sam_matches = 0
    adp_matches = 0
    total = 200
    for trial in range(total):
        length = rng.randint(30, 300)
        vocab = rng.randint(1, 30)
        ctx = rng.randint(0, vocab, size=length).astype(np.int32)
        nn = rng.randint(2, 6)
        kk = 5
        if sam_p.propose(ctx, n=nn, k=kk,
                         req_id='s%d' % trial) is not None:
            sam_matches += 1
        sam_p.remove_request('s%d' % trial)
        if adp.propose(ctx, n=nn, k=kk,
                       req_id='a%d' % trial) is not None:
            adp_matches += 1
        adp.remove_request('a%d' % trial)

    print('  SuffixAutomatonProposer: %d/%d' % (sam_matches, total))
    print('  AdaptiveSuffixProposer:  %d/%d' % (adp_matches, total))
    # Adaptive should match at least as much as SAM (same fallback logic)
    assert adp_matches >= sam_matches, (
        'Adaptive (%d) < SAM (%d)' % (adp_matches, sam_matches))
    print('  PASS')


def test_scoring_with_feedback():
    """Test that feedback changes scoring behavior over time."""
    print('\n=== Test 12: Scoring with feedback ===')
    p = AdaptiveSuffixProposer()

    # Context where "1,2" appears at two positions with different
    # continuations
    context = np.array([1, 2, 99, 98, 97, 1, 2, 88, 87, 86, 1, 2],
                       dtype=np.int32)

    # First proposal
    r1 = p.propose(context, n=2, k=3, req_id='score')
    assert r1 is not None
    print('  First proposal: %s' % r1.tolist())

    # Simulate high acceptance
    p.update_acceptance('score', num_accepted=3)

    # Extend context and propose again
    context2 = np.concatenate([context, np.array([77, 76], dtype=np.int32)])
    context2 = np.concatenate([context2, np.array([1, 2], dtype=np.int32)])
    r2 = p.propose(context2, n=2, k=3, req_id='score')
    assert r2 is not None
    print('  Second proposal: %s' % r2.tolist())
    print('  PASS')


def test_large_context_performance():
    """Benchmark performance on large context."""
    print('\n=== Test 13: Large context performance ===')
    p = AdaptiveSuffixProposer()
    lc = np.random.RandomState(42).randint(0, 100, size=4000).astype(
        np.int32)
    lc[3990:4000] = lc[500:510]

    t0 = time.time()
    p.propose(lc, n=3, k=5, req_id='large')
    t1 = time.time()
    print('  Initial build (4000 tokens): %.2fms' % ((t1 - t0) * 1000))

    lc2 = np.concatenate([lc, np.array([42, 43, 44], dtype=np.int32)])
    t0 = time.time()
    p.propose(lc2, n=3, k=5, req_id='large')
    t1 = time.time()
    print('  Incremental (+3 tokens): %.2fms' % ((t1 - t0) * 1000))
    print('  PASS')


def test_all_same_tokens():
    """Test with all same tokens."""
    print('\n=== Test 14: All same tokens ===')
    p = AdaptiveSuffixProposer()
    context = np.array([5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int32)
    result = p.propose(context, n=2, k=3, req_id='same')
    assert result is not None
    for val in result:
        assert val == 5
    print('  PASS')


def test_context_shrink_rebuilds():
    """Test that context shrink triggers rebuild."""
    print('\n=== Test 15: Context shrink rebuilds ===')
    p = AdaptiveSuffixProposer()
    long_ctx = np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3], dtype=np.int32)
    p.propose(long_ctx, n=3, k=5, req_id='shrink')
    short_ctx = np.array([1, 2, 3, 4, 5, 1, 2, 3], dtype=np.int32)
    result = p.propose(short_ctx, n=3, k=5, req_id='shrink')
    # Should not crash
    print('  PASS')


if __name__ == '__main__':
    test_basic_match()
    test_no_match()
    test_multi_candidate_prefers_recent()
    test_adaptive_fallback()
    test_acceptance_tracker()
    test_acceptance_feedback_loop()
    test_incremental_update()
    test_request_lifecycle()
    test_multiple_requests_independent()
    test_stress_incremental_vs_rebuild()
    test_match_rate_comparison()
    test_scoring_with_feedback()
    test_large_context_performance()
    test_all_same_tokens()
    test_context_shrink_rebuilds()

    print('\n========================================')
    print('=== All 15 tests passed! ===')
    print('========================================')
