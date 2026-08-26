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

This is a port of the original single-request proposer (from the v0.7.x
scheduling-optimization work) to the v0.20.x batch interface. The core
matching/scoring algorithm is unchanged; only the outer ``propose`` interface
was adapted from per-request to per-batch, matching ``NgramProposer``.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from vllm.config import VllmConfig
from vllm.v1.spec_decode.suffix_automaton_proposer import (
    IncrementalSuffixAutomaton,
    SuffixAutomatonProposer,
)


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

    The public ``propose`` interface matches ``NgramProposer`` (batch-based),
    so this proposer is a drop-in replacement for ``--speculative-config
    method="ngram"`` when wired into the model runner.
    """

    # Scoring weights (sum to 1.0)
    W_MATCH: float = 0.25
    W_CONT: float = 0.20
    W_RECENCY: float = 0.25
    W_ACCEPT: float = 0.30

    # Normalization caps
    MAX_MATCH_LEN_CAP: int = 8     # Match lengths beyond this don't help
    MAX_CONT_LEN_CAP: int = 10     # Continuation lengths beyond this cap

    def __init__(self, vllm_config: VllmConfig) -> None:
        super().__init__()

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        assert spec_config.prompt_lookup_min is not None

        # Minimum match length (starting n for adaptive fallback).
        self.min_n = spec_config.prompt_lookup_min
        # Number of draft tokens to propose (k).
        self.k = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # Per-request acceptance tracking, keyed by request id (str).
        self._accept_trackers: Dict[str, AcceptanceTracker] = {}
        # Per-request: last proposed (match_len, num_proposed) for feedback.
        self._last_proposal: Dict[str, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Batch interface (v0.20.x compatible, mirrors NgramProposer.propose)
    # ------------------------------------------------------------------
    def propose(
        self,
        sampled_token_ids: List[List[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        slot_mappings=None,
        req_ids: Optional[List[str]] = None,
    ) -> List[List[int]]:
        """Batch propose: generate draft tokens for each request.

        Mirrors ``NgramProposer.propose`` signature. For each request, the
        context is ``token_ids_cpu[i, :num_tokens_no_spec[i]]``; the result
        is a list (length == batch size) of per-request draft token lists
        (empty list when no match is found).

        ``req_ids`` is optional: when provided (from ``input_batch.req_ids``),
        per-request SAM/acceptance state is keyed by the stable request id,
        surviving batch condense/reorder. When omitted, batch index is used
        as a best-effort key (state may be lost on condense, but correctness
        is unaffected).
        """
        batch_size = len(sampled_token_ids)
        draft_token_ids: List[List[int]] = []

        for i in range(batch_size):
            num_tokens = int(num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len:
                # Reached max model length; skip speculative decoding.
                draft_token_ids.append([])
                continue

            context = token_ids_cpu[i, :num_tokens]
            req_id = req_ids[i] if req_ids is not None else f"idx_{i}"

            draft = self._propose_single(context, self.min_n, self.k, req_id)
            draft_token_ids.append(draft.tolist() if draft is not None else [])

        return draft_token_ids

    def _propose_single(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
        req_id: str,
    ) -> Optional[np.ndarray]:
        """Generate draft token proposals with multi-candidate scoring.

        Algorithm:
        1. Incrementally update (or create) the SAM for this request
        2. For each fallback length from n down to max(2, n//2):
           a. Find ALL match positions in the SAM
           b. Score each candidate using the weighted scoring function
           c. Track the globally best candidate across all lengths
        3. Return the best candidate's continuation tokens
        """
        context_len = len(context_token_ids)
        if context_len < n + 1:
            return None

        # The search text is context[:-1] to avoid self-matching.
        search_len = context_len - 1

        # Update or create SAM (inherited from SuffixAutomatonProposer).
        sam = self._get_or_create_sam(req_id, context_token_ids, search_len)

        tracker = self._accept_trackers.get(req_id)

        min_match = max(2, n // 2)
        best_draft = None
        best_score = -1.0
        best_match_len = 0

        for match_len in range(n, min_match - 1, -1):
            if context_len < match_len:
                continue

            pattern = context_token_ids[context_len - match_len:]

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

        if best_draft is not None:
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
        suffixes), not different occurrences of the same pattern.
        """
        candidates: List[Tuple[int, int]] = []

        # Quick existence check via SAM traversal (O(m)).
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

        # Linear scan to find ALL occurrences in indexed text.
        pat_len = len(pattern)
        for i in range(search_len - pat_len + 1):
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

        Combines four factors (all normalized to [0, 1]):
        1. match_score: Longer matches indicate stronger pattern evidence
        2. cont_score: More continuation tokens allow more speculation
        3. recency: More recent positions are more contextually relevant
        4. accept_rate: Historical acceptance rate for this match length
        """
        match_score = min(match_len / self.MAX_MATCH_LEN_CAP, 1.0)
        cont_score = min(cont_len / self.MAX_CONT_LEN_CAP, 1.0)
        recency = match_pos / max(1, context_len)

        if tracker is not None:
            accept_rate = tracker.get_rate(match_len)
        else:
            accept_rate = 0.5

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

        Called by the model runner with the number of accepted tokens from
        the previous step's draft (derived from valid_sampled_token_ids).
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

    def load_model(self, *args, **kwargs) -> None:
        """No draft model to load (algorithmic proposer, like NgramProposer)."""
        pass
