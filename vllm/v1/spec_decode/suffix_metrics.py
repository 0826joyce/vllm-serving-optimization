# SPDX-License-Identifier: Apache-2.0
"""Suffix decoding metrics (Optimization 5: observability).

Provides a lightweight, dependency-free metrics container for the
suffix-decoding proposers. It is updated in-process (per Worker) by the
proposer during propose()/acceptance feedback, and can be rendered to a
human-readable report or a plain dict for logging / Prometheus export.

Design notes:
  - Pure CPU counters; updates are O(1) and off the GPU critical path.
  - No external deps; safe to import anywhere in the spec_decode package.
  - Rates are computed lazily via properties (never divide-by-zero).
"""

from dataclasses import dataclass, asdict


@dataclass
class SuffixDecodeMetrics:
    """In-process metrics for suffix-decoding proposers.

    All fields are cumulative counters unless noted otherwise. Derived
    quantities (acceptance rate, match rate, ...) are exposed as properties.
    """

    # ---- Proposal / acceptance basics ----
    total_proposals: int = 0          # propose() calls that produced a draft
    total_no_draft: int = 0           # propose() calls that produced nothing
    total_draft_tokens: int = 0       # total draft tokens proposed
    total_accepted_tokens: int = 0    # total draft tokens accepted by sampler

    # ---- Match behavior ----
    match_found_count: int = 0        # times a match was found
    match_not_found_count: int = 0    # times no match was found
    match_lengths_sum: int = 0        # sum of matched pattern lengths
    fallback_count: int = 0           # times adaptive fallback (n -> shorter)

    # ---- Multi-candidate scoring ----
    multi_candidate_count: int = 0    # proposals that had >1 candidate
    best_from_recent_count: int = 0   # best candidate came from most-recent pos

    # ---- Global pool (Optimization 4/6) ----
    global_pool_queries: int = 0      # times the global pool was queried
    global_pool_hits: int = 0         # times the global pool produced a draft
    global_pool_segments: int = 0     # current segment/entry count (snapshot)

    # ---- Dynamic speculation length (Optimization 7) ----
    dyn_len_adjust_count: int = 0     # times k was adjusted from base
    dyn_len_sum: int = 0              # sum of effective k values (for average)
    dyn_len_zero_count: int = 0       # times k was clamped to 0 (skip specul.)

    # ---- Tree verification (Optimization 8) ----
    tree_proposals: int = 0           # times a tree draft was proposed
    tree_branches_sum: int = 0        # sum of branch counts across tree drafts

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    @property
    def acceptance_rate(self) -> float:
        """Accepted / proposed draft tokens."""
        return self.total_accepted_tokens / max(1, self.total_draft_tokens)

    @property
    def avg_accepted_length(self) -> float:
        """Average accepted tokens per successful proposal."""
        return self.total_accepted_tokens / max(1, self.total_proposals)

    @property
    def match_rate(self) -> float:
        """Fraction of propose() calls that found a match."""
        total = self.match_found_count + self.match_not_found_count
        return self.match_found_count / max(1, total)

    @property
    def avg_match_length(self) -> float:
        """Average matched pattern length over successful matches."""
        return self.match_lengths_sum / max(1, self.match_found_count)

    @property
    def global_pool_hit_rate(self) -> float:
        """Global pool hit rate."""
        return self.global_pool_hits / max(1, self.global_pool_queries)

    @property
    def avg_dyn_len(self) -> float:
        """Average effective speculation length (k)."""
        total = self.dyn_len_adjust_count
        return self.dyn_len_sum / max(1, total)

    @property
    def avg_tree_branches(self) -> float:
        """Average branch count per tree draft."""
        return self.tree_branches_sum / max(1, self.tree_proposals)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Return raw counters + derived rates as a flat dict."""
        d = asdict(self)
        d.update(
            acceptance_rate=self.acceptance_rate,
            avg_accepted_length=self.avg_accepted_length,
            match_rate=self.match_rate,
            avg_match_length=self.avg_match_length,
            global_pool_hit_rate=self.global_pool_hit_rate,
            avg_dyn_len=self.avg_dyn_len,
            avg_tree_branches=self.avg_tree_branches,
        )
        return d

    def report(self) -> str:
        """Render a human-readable multi-line report."""
        return (
            "=== Suffix Decode Metrics ===\n"
            f"Proposals:        {self.total_proposals} "
            f"(no-draft: {self.total_no_draft})\n"
            f"Acceptance Rate:  {self.acceptance_rate:.1%} "
            f"({self.total_accepted_tokens}/{self.total_draft_tokens})\n"
            f"Avg Accept Len:   {self.avg_accepted_length:.2f}\n"
            f"Match Rate:       {self.match_rate:.1%}\n"
            f"Avg Match Len:    {self.avg_match_length:.2f}\n"
            f"Fallback Count:   {self.fallback_count}\n"
            f"Multi-Candidate:  {self.multi_candidate_count} "
            f"(best-from-recent: {self.best_from_recent_count})\n"
            f"Global Pool Hit:  {self.global_pool_hit_rate:.1%} "
            f"({self.global_pool_hits}/{self.global_pool_queries}), "
            f"segments={self.global_pool_segments}\n"
            f"Dyn Spec Len:     avg={self.avg_dyn_len:.2f}, "
            f"adjusts={self.dyn_len_adjust_count}, "
            f"skips(k=0)={self.dyn_len_zero_count}\n"
            f"Tree Drafts:      {self.tree_proposals}, "
            f"avg branches={self.avg_tree_branches:.2f}\n"
        )
