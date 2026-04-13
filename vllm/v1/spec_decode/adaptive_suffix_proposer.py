# SPDX-License-Identifier: Apache-2.0
"""Adaptive Suffix Proposer with multi-candidate scoring.

Optimization 3: Builds on the incremental SAM from Optimization 2 with:
  1. Multi-candidate evaluation: finds ALL match positions, scores them
  2. Acceptance-rate-aware scoring: uses historical acceptance feedback
  3. Improved adaptive fallback: evaluates candidates across all fallback
     levels before choosing the best one

The key insight is that simply returning the first match (like NgramProposer)
or the longest continuation (like SuffixAutomatonProposer) is suboptimal.
A match at a more recent position in the context is more likely to be
contextually relevant, even if it offers fewer continuation tokens. This
proposer balances match length, continuation length, recency, and historical
acceptance rate through a weighted scoring function.

Acceptance feedback loop:
  - After each step, generate_draft_token_ids() receives valid_sampled_token_ids
    which tells us how many tokens were accepted from the previous draft
  - The proposer records (num_proposed, num_accepted) per (req_id, match_len)
  - This feedback adjusts the scoring weights over time
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from vllm.v1.spec_decode.suffix_automaton_proposer import (
    IncrementalSuffixAutomaton, SuffixAutomatonProposer)


class AcceptanceTracker:
    """Sliding-window acceptance rate tracker per match length.

    Tracks the historical acceptance rate for different match lengths
    within a single request. This is used by the scoring function to
    prefer match lengths that historically lead to higher acceptance.

    Attributes:
        _window_size: Maximum number of recent records to keep per length.
        _history: Map from match_len to deque of (num_proposed, num_accepted).
    """

    def __init__(self, window_size: int = 20) -> None:
        self._window_size = window_size
        self._history: Dict[int, Deque[Tuple[int, int]]] = {}

    def record(
        self,
        match_len: int,
        num_proposed: int,
        num_accepted: int,
    ) -> None:
        """Record an acceptance result for a given match length.

        Args:
            match_len: The match length that produced the draft.
            num_proposed: Number of draft tokens proposed.
            num_accepted: Number of tokens accepted by rejection sampler.
        """
        if match_len not in self._history:
            self._history[match_len] = deque(maxlen=self._window_size)
        self._history[match_len].append((num_proposed, num_accepted))

    def get_rate(self, match_len: int) -> float:
        """Get the historical acceptance rate for a match length.

        Returns 0.5 (neutral) if no history is available.
        """
        history = self._history.get(match_len)
        if not history:
            return 0.5
        total_proposed = sum(p for p, _ in history)
        total_accepted = sum(a for _, a in history)
        if total_proposed == 0:
            return 0.5
        return total_accepted / total_proposed


class AdaptiveSuffixProposer(SuffixAutomatonProposer):
    """Adaptive Suffix Proposer with multi-candidate scoring.

    Extends SuffixAutomatonProposer (Optimization 2) with:
    1. Multi-candidate search: finds all match positions at each length
    2. Weighted scoring: balances match_len, cont_len, recency, accept_rate
    3. Acceptance feedback: tracks per-request acceptance history
    4. Cross-level best selection: compares candidates across all fallback
       levels to find the globally best candidate

    The scoring function uses four factors:
    - match_score (w=0.25): Longer matches are more likely correct
    - cont_score  (w=0.20): More continuation tokens = more draft
    - recency     (w=0.25): More recent matches are more contextually relevant
    - accept_rate (w=0.30): Historical feedback from rejection sampler

    Usage:
        Set VLLM_SPEC_PROPOSER=adaptive to use this proposer.
    """

    # Scoring weights (sum to 1.0)
    W_MATCH: float = 0.25
    W_CONT: float = 0.20
    W_RECENCY: float = 0.25
    W_ACCEPT: float = 0.30

    # Normalization caps
    MAX_MATCH_LEN_CAP: int = 8     # Match lengths beyond this don't help
    MAX_CONT_LEN_CAP: int = 10     # Continuation lengths beyond this cap

    def __init__(self) -> None:
        super().__init__()
        # Per-request acceptance tracking
        self._accept_trackers: Dict[str, AcceptanceTracker] = {}
        # Per-request: last proposed (match_len, num_proposed) for feedback
        self._last_proposal: Dict[str, Tuple[int, int]] = {}

    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
        req_id: str = "",
    ) -> Optional[np.ndarray]:
        """Generate draft token proposals with multi-candidate scoring.

        Algorithm:
        1. Incrementally update (or create) the SAM for this request
        2. For each fallback length from n down to max(2, n//2):
           a. Find ALL match positions in the SAM
           b. Score each candidate using the weighted scoring function
           c. Track the globally best candidate across all lengths
        3. Return the best candidate's continuation tokens

        Args:
            context_token_ids: Full context token array.
            n: Starting match length (will fall back to shorter).
            k: Maximum number of draft tokens to return.
            req_id: Request identifier for stateful tracking.

        Returns:
            np.ndarray of draft tokens, or None if no match found.
        """
        context_len = len(context_token_ids)
        if context_len < n + 1:
            return None

        # The search text is context[:-1] to avoid self-matching
        search_len = context_len - 1

        # Update or create SAM (inherited from SuffixAutomatonProposer)
        sam = self._get_or_create_sam(req_id, context_token_ids, search_len)

        # Get acceptance tracker for scoring
        tracker = self._accept_trackers.get(req_id)

        # Adaptive matching: try all fallback lengths, keep global best
        min_match = max(2, n // 2)
        best_draft = None
        best_score = -1.0
        best_match_len = 0

        for match_len in range(n, min_match - 1, -1):
            if context_len < match_len:
                continue

            pattern = context_token_ids[context_len - match_len:]

            # Find all match candidates at this length
            candidates = self._find_all_candidates(
                sam, pattern, match_len, search_len, context_token_ids)

            for cont_start, cont_available in candidates:
                actual_k = min(k, cont_available)
                if actual_k <= 0:
                    continue

                score = self._score_candidate(
                    match_len=match_len,
                    cont_len=actual_k,
                    match_pos=cont_start,
                    context_len=search_len,
                    tracker=tracker,
                )

                if score > best_score:
                    best_score = score
                    best_draft = context_token_ids[
                        cont_start:cont_start + actual_k]
                    best_match_len = match_len

        if best_draft is not None and req_id:
            # Record what we proposed for acceptance feedback
            self._last_proposal[req_id] = (best_match_len, len(best_draft))

        return best_draft

    def _find_all_candidates(
        self,
        sam: IncrementalSuffixAutomaton,
        pattern: np.ndarray,
        match_len: int,
        search_len: int,
        context_token_ids: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """Find all match positions for a pattern in the indexed text.

        First verifies via the SAM that the pattern exists (O(m) check).
        Then performs a linear scan of the indexed text to find ALL
        occurrences of the pattern, collecting each as a candidate.

        We use linear scan instead of suffix link traversal because SAM
        suffix links represent different equivalence classes (shorter
        suffixes), not different occurrences of the same pattern. Finding
        all endpos would require building the full endpos sets via
        topological sort, which adds complexity. Linear scan is simpler
        and still efficient for typical context lengths (< 100K tokens).

        Args:
            sam: The suffix automaton to verify pattern existence.
            pattern: Token pattern to match.
            match_len: Minimum match length required.
            search_len: Length of the indexed text.
            context_token_ids: Full context array (for reading continuations).

        Returns:
            List of (cont_start, cont_available) tuples, where cont_start
            is the start position of continuation tokens and cont_available
            is how many continuation tokens are available.
        """
        candidates: List[Tuple[int, int]] = []

        # Quick existence check via SAM traversal (O(m))
        node_idx = 0
        matched = 0
        nodes = sam._nodes
        for token in pattern:
            token_int = int(token)
            if token_int in nodes[node_idx].transitions:
                node_idx = nodes[node_idx].transitions[token_int]
                matched += 1
            else:
                break

        if matched < match_len:
            return candidates

        # Linear scan to find ALL occurrences in indexed text
        # The indexed text is context_token_ids[0:search_len]
        pat_len = len(pattern)
        for i in range(search_len - pat_len + 1):
            # Check if pattern matches at position i
            if self._pattern_matches_at(
                    context_token_ids, i, pattern, pat_len):
                cont_start = i + pat_len
                if cont_start < search_len:
                    candidates.append(
                        (cont_start, search_len - cont_start))

        return candidates

    @staticmethod
    def _pattern_matches_at(
        text: np.ndarray,
        pos: int,
        pattern: np.ndarray,
        pat_len: int,
    ) -> bool:
        """Check if pattern matches text starting at position pos."""
        for j in range(pat_len):
            if text[pos + j] != pattern[j]:
                return False
        return True

    def _score_candidate(
        self,
        match_len: int,
        cont_len: int,
        match_pos: int,
        context_len: int,
        tracker: Optional[AcceptanceTracker],
    ) -> float:
        """Score a match candidate for selection.

        Combines four factors:
        1. match_score: Longer matches indicate stronger pattern evidence
        2. cont_score: More continuation tokens allow more speculation
        3. recency: More recent positions are more contextually relevant
        4. accept_rate: Historical acceptance rate for this match length

        All factors are normalized to [0, 1] before weighted combination.

        Args:
            match_len: Number of tokens matched.
            cont_len: Number of continuation tokens available.
            match_pos: Position in the context where continuation starts.
            context_len: Total length of the search text.
            tracker: Acceptance tracker for this request (may be None).

        Returns:
            Score in [0, 1], higher is better.
        """
        # Factor 1: Match length (longer = better, capped)
        match_score = min(match_len / self.MAX_MATCH_LEN_CAP, 1.0)

        # Factor 2: Continuation length (more = better, capped)
        cont_score = min(cont_len / self.MAX_CONT_LEN_CAP, 1.0)

        # Factor 3: Recency (position closer to end = better)
        recency = match_pos / max(1, context_len)

        # Factor 4: Historical acceptance rate
        if tracker is not None:
            accept_rate = tracker.get_rate(match_len)
        else:
            accept_rate = 0.5  # Neutral default

        return (self.W_MATCH * match_score +
                self.W_CONT * cont_score +
                self.W_RECENCY * recency +
                self.W_ACCEPT * accept_rate)

    def update_acceptance(
        self,
        req_id: str,
        num_accepted: int,
    ) -> None:
        """Update acceptance statistics from rejection sampler feedback.

        Called by generate_draft_token_ids() with the number of accepted
        tokens from the previous step's draft.

        The acceptance information is derived from valid_sampled_token_ids:
          num_accepted = len(valid_sampled_token_ids[i]) - 1
          (the -1 accounts for the bonus token)

        Args:
            req_id: Request identifier.
            num_accepted: Number of draft tokens that were accepted.
        """
        if req_id not in self._last_proposal:
            return

        match_len, num_proposed = self._last_proposal[req_id]
        if num_proposed <= 0:
            return

        if req_id not in self._accept_trackers:
            self._accept_trackers[req_id] = AcceptanceTracker()

        self._accept_trackers[req_id].record(
            match_len, num_proposed, num_accepted)

    def remove_request(self, req_id: str) -> None:
        """Clean up all state when a request finishes."""
        super().remove_request(req_id)
        self._accept_trackers.pop(req_id, None)
        self._last_proposal.pop(req_id, None)

    def num_active_requests(self) -> int:
        """Return the number of requests with active SAM state."""
        return len(self._automata)
