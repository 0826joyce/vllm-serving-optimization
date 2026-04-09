# SPDX-License-Identifier: Apache-2.0
"""Suffix-based speculative decoding proposer.

Replaces the N-gram KMP search with a suffix array + binary search approach,
providing:
  1. O(pattern_len * log(context_len)) query instead of O(context_len)
  2. Adaptive match length (fallback from n down to n//2)
  3. Best-match selection (longest continuation) instead of first-match
"""

from typing import List, Optional, Tuple

import numpy as np
from numba import jit


class SuffixTreeProposer:
    """Suffix-array-based speculative decoding proposer.

    Core advantages over NgramProposer:
    1. Query time O(m * log(n)) vs O(n), where m = pattern length, n = context
    2. Automatically finds longest continuation match
    3. Returns best match (longest continuation) instead of first match
    4. Adaptive fallback: tries shorter patterns when longer ones don't match

    Maintains the same external interface as NgramProposer for drop-in
    replacement.
    """

    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
    ) -> Optional[np.ndarray]:
        """Generate draft token proposals (same interface as NgramProposer).

        Algorithm:
        1. Build suffix array on context[:-1] (exclude last position to avoid
           self-match)
        2. Try matching last n tokens, falling back to shorter patterns
        3. Among all matches, pick the one with the longest continuation
        4. Return up to k tokens following the best match

        Args:
            context_token_ids: Numpy array of token IDs representing the
                               context sequence (prompt + output + sampled).
            n: Minimum n-gram match length (used as starting match length).
            k: Number of draft tokens to return.

        Returns:
            np.ndarray: The draft token sequence following the best match.
            None: If no matching pattern is found.
        """
        return _suffix_propose(context_token_ids, n, k)


@jit(nopython=True)
def _suffix_propose(
    context_token_ids: np.ndarray,
    n: int,
    k: int,
) -> Optional[np.ndarray]:
    """Core suffix-based proposal logic, JIT-compiled with Numba.

    Uses suffix array with binary search for efficient pattern matching,
    with adaptive fallback on match length.
    """
    context_len = context_token_ids.shape[0]
    if context_len < n + 1:
        return None

    # Search text: context[:-1] to avoid self-matching the tail
    search_len = context_len - 1

    # Build suffix array on search_text
    sa = _build_suffix_array_simple(context_token_ids, search_len)

    # Adaptive matching: try from n down to max(2, n//2)
    min_match = max(2, n // 2)

    best_draft_start = -1
    best_draft_len = 0

    for match_len in range(n, min_match - 1, -1):
        if context_len < match_len:
            continue

        # Pattern is the last match_len tokens of context
        pattern_start = context_len - match_len

        # Binary search for all occurrences of pattern in suffix array
        lo = _sa_lower_bound(context_token_ids, sa, search_len,
                             context_token_ids, pattern_start, match_len)
        hi = _sa_upper_bound(context_token_ids, sa, search_len,
                             context_token_ids, pattern_start, match_len)

        if lo >= hi:
            continue

        # Found matches, select the one with longest continuation
        for idx in range(lo, hi):
            pos = sa[idx]
            # Position after the matched pattern
            cont_start = pos + match_len
            if cont_start >= search_len:
                continue
            # Skip if this match IS the tail pattern itself
            if pos == pattern_start:
                continue
            cont_len = search_len - cont_start
            if cont_len > best_draft_len:
                best_draft_len = cont_len
                best_draft_start = cont_start

        if best_draft_start >= 0:
            # Found a good match at this match_len, no need to try shorter
            break

    if best_draft_start < 0:
        return None

    actual_k = min(k, best_draft_len)
    return context_token_ids[best_draft_start:best_draft_start + actual_k]


@jit(nopython=True)
def _build_suffix_array_simple(text: np.ndarray, length: int) -> np.ndarray:
    """Build suffix array using simple O(n * log^2(n)) algorithm.

    Uses iterative doubling with rank arrays, JIT-compatible.
    For typical context lengths (< 8K tokens), this is efficient enough.

    Args:
        text: The full token array.
        length: Number of tokens to consider (text[:length]).

    Returns:
        Suffix array as np.ndarray of int32.
    """
    n = length
    if n == 0:
        return np.empty(0, dtype=np.int32)

    # Initialize rank from token values
    sa = np.arange(n, dtype=np.int32)
    rank = np.zeros(n, dtype=np.int32)
    tmp_rank = np.zeros(n, dtype=np.int32)

    # Initial ranking based on single character
    for i in range(n):
        rank[i] = text[i]

    # Iterative doubling
    gap = 1
    while gap < n:
        # Sort by (rank[i], rank[i+gap]) using a simple insertion sort
        # for small n, or radix-like approach
        # For JIT compatibility, use a simple comparison-based sort
        _sort_by_rank_pair(sa, rank, gap, n)

        # Update ranks
        tmp_rank[sa[0]] = 0
        for i in range(1, n):
            prev = sa[i - 1]
            curr = sa[i]
            # Compare (rank[prev], rank[prev+gap]) vs (rank[curr],
            # rank[curr+gap])
            r1_prev = rank[prev]
            r2_prev = rank[prev + gap] if prev + gap < n else -1
            r1_curr = rank[curr]
            r2_curr = rank[curr + gap] if curr + gap < n else -1

            if r1_curr == r1_prev and r2_curr == r2_prev:
                tmp_rank[curr] = tmp_rank[prev]
            else:
                tmp_rank[curr] = tmp_rank[prev] + 1

        for i in range(n):
            rank[i] = tmp_rank[i]

        # Early termination: all ranks are unique
        if rank[sa[n - 1]] == n - 1:
            break

        gap *= 2

    return sa


@jit(nopython=True)
def _sort_by_rank_pair(
    sa: np.ndarray,
    rank: np.ndarray,
    gap: int,
    n: int,
) -> None:
    """Sort suffix array by (rank[i], rank[i+gap]) pairs.

    Uses shell sort for JIT compatibility and reasonable performance.
    Shell sort is O(n^(4/3)) worst case, good enough for our use case.
    """
    # Shell sort with Ciura gap sequence
    gaps = np.array([701, 301, 132, 57, 23, 10, 4, 1], dtype=np.int32)

    for g in gaps:
        if g >= n:
            continue
        for i in range(g, n):
            temp = sa[i]
            r1_temp = rank[temp]
            r2_temp = rank[temp + gap] if temp + gap < n else -1

            j = i
            while j >= g:
                cand = sa[j - g]
                r1_cand = rank[cand]
                r2_cand = rank[cand + gap] if cand + gap < n else -1

                # Compare (r1_cand, r2_cand) > (r1_temp, r2_temp)
                if r1_cand > r1_temp or (r1_cand == r1_temp
                                         and r2_cand > r2_temp):
                    sa[j] = sa[j - g]
                    j -= g
                else:
                    break
            sa[j] = temp


@jit(nopython=True)
def _compare_with_pattern(
    text: np.ndarray,
    text_pos: int,
    text_len: int,
    pattern: np.ndarray,
    pattern_start: int,
    pattern_len: int,
) -> int:
    """Compare text[text_pos:text_pos+pattern_len] with
    pattern[pattern_start:pattern_start+pattern_len].

    Returns:
        -1 if text suffix < pattern
         0 if text suffix == pattern (prefix match)
         1 if text suffix > pattern
    """
    avail = text_len - text_pos
    cmp_len = min(avail, pattern_len)

    for i in range(cmp_len):
        a = text[text_pos + i]
        b = pattern[pattern_start + i]
        if a < b:
            return -1
        elif a > b:
            return 1

    # If we compared all of pattern_len, it's a match
    if avail >= pattern_len:
        return 0
    # text suffix is shorter than pattern, treated as less
    return -1


@jit(nopython=True)
def _sa_lower_bound(
    text: np.ndarray,
    sa: np.ndarray,
    text_len: int,
    pattern: np.ndarray,
    pattern_start: int,
    pattern_len: int,
) -> int:
    """Binary search for lower bound of pattern in suffix array."""
    lo = 0
    hi = len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        cmp = _compare_with_pattern(text, sa[mid], text_len, pattern,
                                    pattern_start, pattern_len)
        if cmp < 0:
            lo = mid + 1
        else:
            hi = mid
    return lo


@jit(nopython=True)
def _sa_upper_bound(
    text: np.ndarray,
    sa: np.ndarray,
    text_len: int,
    pattern: np.ndarray,
    pattern_start: int,
    pattern_len: int,
) -> int:
    """Binary search for upper bound of pattern in suffix array."""
    lo = 0
    hi = len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        cmp = _compare_with_pattern(text, sa[mid], text_len, pattern,
                                    pattern_start, pattern_len)
        if cmp <= 0:
            lo = mid + 1
        else:
            hi = mid
    return lo
