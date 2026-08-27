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

import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from vllm.config import VllmConfig
from vllm.v1.spec_decode.suffix_automaton_proposer import (
    IncrementalSuffixAutomaton,
    SuffixAutomatonProposer,
)
from vllm.v1.spec_decode.suffix_metrics import SuffixDecodeMetrics


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

        # Optimization 5: in-process observability metrics.
        self.metrics = SuffixDecodeMetrics()

        # Optimization 4/6: cross-request global suffix pool (opt-in via
        # VLLM_SUFFIX_GLOBAL_POOL=1). Lazily imported to avoid import cycles.
        self._global_pool = None
        if os.environ.get("VLLM_SUFFIX_GLOBAL_POOL") == "1":
            from vllm.v1.spec_decode.global_suffix_pool import (
                FrequencyAwareGlobalSuffixPool,
            )

            self._global_pool = FrequencyAwareGlobalSuffixPool()
        # Per-request: last global-pool key (for feedback), if pool was used.
        self._last_global_key: Dict[str, Optional[Tuple[int, ...]]] = {}

        # Optimization 7: dynamic speculation length (opt-in via
        # VLLM_SUFFIX_DYNAMIC_K=1). Load signal is set by the model runner
        # each step through set_load().
        self._dynamic_k = os.environ.get("VLLM_SUFFIX_DYNAMIC_K") == "1"
        self._load: float = 0.0

        # Optimization 8: tree-style multi-candidate drafts (opt-in via
        # VLLM_SUFFIX_TREE=1). When enabled, propose_tree() can be used by
        # the model runner to build a small draft tree.
        self._tree_enabled = os.environ.get("VLLM_SUFFIX_TREE") == "1"
        self.tree_max_branches = int(
            os.environ.get("VLLM_SUFFIX_TREE_BRANCHES", "3")
        )

    # ------------------------------------------------------------------
    # Optimization 7: load signal setter (called by model runner per step)
    # ------------------------------------------------------------------
    def set_load(self, load: float) -> None:
        """Update the current system load (running / max_running) in [0, 1].

        Used by the dynamic speculation-length logic when enabled.
        """
        self._load = max(0.0, min(1.0, load))

    def _effective_k(self, req_id: str, base_k: int) -> int:
        """Compute the per-request speculation length for this step.

        Combines the system load and the request's recent acceptance rate.
        Returns base_k unchanged when dynamic mode is disabled.
        """
        if not self._dynamic_k:
            return base_k

        # Factor 1: system load.
        if self._load < 0.5:
            load_factor = 1.5      # light load: speculate aggressively
        elif self._load < 0.8:
            load_factor = 1.0      # medium load: normal
        else:
            load_factor = 0.5      # heavy load: conservative

        # Factor 2: recent acceptance rate for this request.
        tracker = self._accept_trackers.get(req_id)
        if tracker is not None:
            # Use the acceptance rate at the base match length as a proxy.
            accept_rate = tracker.get_rate(self.min_n)
        else:
            accept_rate = 0.5
        accept_factor = 0.5 + accept_rate   # 0.5 ~ 1.5

        k = int(round(base_k * load_factor * accept_factor))
        k = max(0, min(k, base_k * 2))

        self.metrics.dyn_len_adjust_count += 1
        self.metrics.dyn_len_sum += k
        if k == 0:
            self.metrics.dyn_len_zero_count += 1
        return k

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

            # Optimization 7: dynamic speculation length (no-op if disabled).
            eff_k = self._effective_k(req_id, self.k)
            if eff_k <= 0:
                # Dynamic policy decided to skip speculation this step.
                draft_token_ids.append([])
                continue

            draft = self._propose_single(context, self.min_n, eff_k, req_id)

            if draft is None or len(draft) == 0:
                self.metrics.total_no_draft += 1
                draft_token_ids.append([])
            else:
                self.metrics.total_proposals += 1
                self.metrics.total_draft_tokens += len(draft)
                draft_token_ids.append(draft.tolist())

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
        best_is_recent = False
        num_candidates_seen = 0
        matched_at_top = False

        for match_len in range(n, min_match - 1, -1):
            if context_len < match_len:
                continue

            pattern = context_token_ids[context_len - match_len:]

            candidates = self._find_all_candidates(
                sam, pattern, match_len, search_len, context_token_ids)

            if candidates:
                if match_len == n:
                    matched_at_top = True
                num_candidates_seen += len(candidates)
                # The last candidate in scan order is the most recent match.
                recent_pos = candidates[-1][0]

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
                    best_is_recent = (cont_start == recent_pos)

        # ---- Optimization 5: record match/scoring metrics ----
        if best_draft is not None:
            self._last_proposal[req_id] = (best_match_len, len(best_draft))
            self.metrics.match_found_count += 1
            self.metrics.match_lengths_sum += best_match_len
            if not matched_at_top:
                self.metrics.fallback_count += 1
            if num_candidates_seen > 1:
                self.metrics.multi_candidate_count += 1
                if best_is_recent:
                    self.metrics.best_from_recent_count += 1
            self._last_global_key[req_id] = None
            return best_draft

        # ---- Optimization 4/6: fall back to the cross-request global pool ----
        if self._global_pool is not None:
            pattern = context_token_ids[context_len - n:]
            self.metrics.global_pool_queries += 1
            g_draft, g_key = self._global_pool.query(pattern, k)
            self.metrics.global_pool_segments = self._global_pool.num_entries()
            if g_draft is not None and len(g_draft) > 0:
                self.metrics.global_pool_hits += 1
                self.metrics.match_found_count += 1
                self._last_proposal[req_id] = (n, len(g_draft))
                self._last_global_key[req_id] = g_key
                return g_draft
            self._last_global_key[req_id] = None

        self.metrics.match_not_found_count += 1
        return None

    # ------------------------------------------------------------------
    # Optimization 8: tree-style multi-candidate draft generation
    # ------------------------------------------------------------------
    def propose_tree(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
        req_id: str,
        max_branches: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Produce multiple candidate draft branches for one request.

        Unlike ``_propose_single`` (which picks the single best-scoring
        continuation), this returns up to ``max_branches`` distinct
        continuations for the current suffix pattern — "bet on several
        paths" so the target can accept whichever matches.

        NOTE on verification: vLLM's official tree-attention verification
        (``TreeAttentionMetadataBuilder`` + model ``propose_tree``) is built
        for MODEL-based drafters (EAGLE-style) that run their own GPU forward
        passes. The algorithmic (ngram-path) proposer used here does not own
        ``draft_attn_groups`` and cannot drive that GPU tree verification
        directly. So this method provides the *multi-candidate generation*
        capability; wiring it into a genuine one-shot tree verification would
        require the ngram spec-decode path (RejectionSampler) to support tree
        drafts, which is a larger, separate change. Branches are returned
        longest-first so a caller can also use them as ranked alternatives.

        Returns an empty list when no match is found.
        """
        if max_branches is None:
            max_branches = self.tree_max_branches

        context_len = len(context_token_ids)
        if context_len < n + 1:
            return []

        search_len = context_len - 1
        sam = self._get_or_create_sam(req_id, context_token_ids, search_len)
        tracker = self._accept_trackers.get(req_id)

        min_match = max(2, n // 2)
        scored: List[Tuple[float, Tuple[int, ...]]] = []
        seen: set = set()

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
                cont = tuple(
                    int(t) for t in
                    context_token_ids[cont_start:cont_start + actual_k])
                if cont in seen:
                    continue
                seen.add(cont)
                score = self._score_candidate(
                    match_len=match_len,
                    cont_len=actual_k,
                    match_pos=cont_start,
                    context_len=search_len,
                    tracker=tracker,
                )
                scored.append((score, cont))

        if not scored:
            return []

        # Keep the top-scoring, distinct branches.
        scored.sort(key=lambda x: x[0], reverse=True)
        branches = [
            np.array(cont, dtype=np.int32)
            for _, cont in scored[:max_branches]
        ]

        # Optimization 5: tree metrics.
        self.metrics.tree_proposals += 1
        self.metrics.tree_branches_sum += len(branches)
        return branches

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

        # Optimization 5: accepted-token metric.
        self.metrics.total_accepted_tokens += max(0, num_accepted)

        if req_id not in self._accept_trackers:
            self._accept_trackers[req_id] = AcceptanceTracker()

        self._accept_trackers[req_id].record(
            match_len, num_proposed, num_accepted)

        # Optimization 4/6: feed acceptance back to the global pool.
        if self._global_pool is not None:
            g_key = self._last_global_key.get(req_id)
            if g_key is not None:
                self._global_pool.update_feedback(
                    g_key, proposed=num_proposed, accepted=num_accepted)

    def record_accepted_segment(self, tokens: np.ndarray) -> None:
        """Optimization 4/6: contribute an accepted continuation to the pool.

        Called by the model runner when a request accepts a run of tokens,
        so other requests can reuse the pattern via the global pool.
        No-op when the global pool is disabled.
        """
        if self._global_pool is not None and len(tokens) >= 3:
            self._global_pool.add_segment(tokens)

    def get_metrics(self) -> SuffixDecodeMetrics:
        """Return the in-process metrics object (Optimization 5)."""
        return self.metrics

    def remove_request(self, req_id: str) -> None:
        """Clean up all state when a request finishes."""
        super().remove_request(req_id)
        self._accept_trackers.pop(req_id, None)
        self._last_proposal.pop(req_id, None)
        self._last_global_key.pop(req_id, None)

    def load_model(self, *args, **kwargs) -> None:
        """No draft model to load (algorithmic proposer, like NgramProposer)."""
        pass
