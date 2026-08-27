# SPDX-License-Identifier: Apache-2.0
"""Cross-request global suffix pool with frequency-aware filtering.

Combines Optimization 4 (cross-request sharing) and Optimization 6
(frequency-aware draft quality filtering) into one component:

  - Collects "accepted continuations" from all requests (per Worker), so a
    request can reuse patterns that other requests have already produced.
  - Records, per (pattern -> continuation), how often the continuation was
    proposed and how many of its tokens were accepted, and selects the
    candidate with the highest EXPECTED GAIN (accept_rate x length) instead
    of blindly choosing the longest / most recent one.
  - Bounded capacity with LRU + low-value eviction.

Design choices (kept simple & dependency-free):
  - The index maps a fixed-length pattern key (the last ``key_len`` tokens of
    an accepted segment prefix) to a small list of candidate continuations.
  - This is O(1) lookup and avoids rebuilding a global suffix automaton on
    every insert (which the original sketch required). It captures the same
    value: "given this recent pattern, what tends to come next".
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class _Candidate:
    """A candidate continuation for a pattern, with frequency stats."""

    continuation: Tuple[int, ...]
    proposed_count: int = 0
    accepted_tokens: int = 0
    last_access: float = field(default_factory=time.monotonic)

    @property
    def expected_gain(self) -> float:
        """Average number of tokens accepted per proposal.

        Cold start (never proposed) uses an optimistic estimate so the
        candidate gets at least one chance to be tried.
        """
        if self.proposed_count == 0:
            return len(self.continuation) * 0.5
        return self.accepted_tokens / self.proposed_count


class FrequencyAwareGlobalSuffixPool:
    """Global, frequency-aware suffix pool shared across requests.

    Thread-safety: intended to be used from a single Worker process's
    proposer (same thread as the model runner step loop), so no locking.
    """

    def __init__(
        self,
        key_len: int = 3,
        max_patterns: int = 20000,
        max_candidates_per_pattern: int = 4,
        min_expected_gain: float = 0.3,
        max_continuation_len: int = 16,
    ) -> None:
        self.key_len = key_len
        self.max_patterns = max_patterns
        self.max_candidates_per_pattern = max_candidates_per_pattern
        self.min_expected_gain = min_expected_gain
        self.max_continuation_len = max_continuation_len

        # pattern key (tuple of last key_len tokens) -> list of candidates.
        self._index: Dict[Tuple[int, ...], List[_Candidate]] = {}

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def add_segment(self, tokens: np.ndarray) -> None:
        """Register an accepted token segment into the pool.

        The segment is split into (pattern_key -> continuation) entries using
        a sliding window: for each position i with enough left context, the
        preceding ``key_len`` tokens form the key and the following tokens
        (up to ``max_continuation_len``) form the continuation.
        """
        seg = [int(t) for t in tokens]
        n = len(seg)
        if n < self.key_len + 1:
            return

        # Use a few anchor positions to avoid quadratic blow-up on long segs.
        for i in range(self.key_len, n):
            key = tuple(seg[i - self.key_len:i])
            cont = tuple(seg[i:i + self.max_continuation_len])
            if not cont:
                continue
            self._insert(key, cont)

        if len(self._index) > self.max_patterns:
            self._evict()

    def _insert(self, key: Tuple[int, ...], cont: Tuple[int, ...]) -> None:
        bucket = self._index.setdefault(key, [])
        for cand in bucket:
            if cand.continuation == cont:
                cand.last_access = time.monotonic()
                return
        bucket.append(_Candidate(continuation=cont))
        # Cap candidates per pattern: drop the lowest expected-gain one.
        if len(bucket) > self.max_candidates_per_pattern:
            bucket.sort(key=lambda c: c.expected_gain, reverse=True)
            del bucket[self.max_candidates_per_pattern:]

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(
        self,
        pattern: np.ndarray,
        k: int,
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, ...]]]:
        """Look up the best continuation for the pattern's suffix.

        Returns (draft_tokens, feedback_key) where feedback_key identifies
        the chosen (pattern, continuation) pair for later update_feedback().
        Returns (None, None) when no candidate clears the gain threshold.
        """
        if len(pattern) < self.key_len:
            return None, None

        key = tuple(int(t) for t in pattern[-self.key_len:])
        bucket = self._index.get(key)
        if not bucket:
            return None, None

        best: Optional[_Candidate] = None
        best_gain = self.min_expected_gain    # below threshold -> skip
        for cand in bucket:
            gain = cand.expected_gain
            if gain > best_gain:
                best_gain = gain
                best = cand

        if best is None:
            return None, None

        best.last_access = time.monotonic()
        draft = np.array(best.continuation[:k], dtype=np.int32)
        feedback_key = key + (0xFFFF,) + best.continuation  # composite id
        return draft, feedback_key

    def update_feedback(
        self,
        feedback_key: Tuple[int, ...],
        proposed: int,
        accepted: int,
    ) -> None:
        """Update frequency stats for a previously queried candidate."""
        # Split composite id back into (key, continuation).
        try:
            sep = feedback_key.index(0xFFFF)
        except ValueError:
            return
        key = feedback_key[:sep]
        cont = feedback_key[sep + 1:]
        bucket = self._index.get(key)
        if not bucket:
            return
        for cand in bucket:
            if cand.continuation == cont:
                cand.proposed_count += 1
                cand.accepted_tokens += max(0, accepted)
                cand.last_access = time.monotonic()
                return

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def _evict(self) -> None:
        """Evict low-value patterns (LRU + low expected gain)."""
        # Score each pattern by best candidate gain and recency.
        def pattern_score(item):
            _, bucket = item
            if not bucket:
                return (0.0, 0.0)
            best_gain = max(c.expected_gain for c in bucket)
            recent = max(c.last_access for c in bucket)
            return (best_gain, recent)

        items = sorted(self._index.items(), key=pattern_score, reverse=True)
        keep = items[:self.max_patterns]
        self._index = dict(keep)

    def num_entries(self) -> int:
        """Number of distinct pattern keys currently stored."""
        return len(self._index)
