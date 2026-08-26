# SPDX-License-Identifier: Apache-2.0
"""Incremental Suffix Automaton (SAM) based speculative decoding proposer.

Optimization 2: Replaces the stateless suffix array approach (Optimization 1)
with an incremental Suffix Automaton that supports:
  1. O(1) amortized online append per new token (no full rebuild)
  2. O(m) query time where m = pattern length
  3. Stateful per-request: maintains SAM across decode steps
  4. Adaptive fallback matching (inherited from Optimization 1)

The Suffix Automaton (SAM) is the minimal DFA that recognizes all suffixes
of a string. Key properties:
  - At most 2n-1 states and 3n-4 transitions for a string of length n
  - Each state represents an equivalence class of substrings
  - Online construction: extend() adds one token in O(1) amortized time
  - Query: traverse transitions for O(m) pattern matching

Reference: Blumer et al., "The smallest automaton recognizing the subwords
of a text" (1985)
"""

from typing import Dict, List, Optional, Tuple

import numpy as np


class _SAMNode:
    """Suffix Automaton node.

    Attributes:
        len: Length of the longest substring in this equivalence class.
        link: Suffix link to the parent equivalence class (-1 for initial).
        transitions: Map from token_id to target node index.
        first_end_pos: The end position (exclusive) where this state's
            longest substring first occurs in the text. Used to locate
            continuation tokens after a match.
    """
    __slots__ = ['len', 'link', 'transitions', 'first_end_pos']

    def __init__(self, length: int = 0):
        self.len: int = length
        self.link: int = -1
        self.transitions: Dict[int, int] = {}
        self.first_end_pos: int = -1


class IncrementalSuffixAutomaton:
    """Incremental Suffix Automaton supporting O(1) amortized online append.

    This is a faithful implementation of the online SAM construction
    algorithm. Each call to extend() appends one token and updates the
    automaton structure.

    The automaton can then be queried to find the longest matching suffix
    of a given pattern, and return the position in the original text where
    that match occurs.
    """

    def __init__(self) -> None:
        # Node 0 is the initial state (empty string)
        self._nodes: List[_SAMNode] = [_SAMNode(0)]
        self._last: int = 0  # Index of the state representing the full text
        self._size: int = 1
        self._text_len: int = 0  # Number of tokens appended so far

    @property
    def size(self) -> int:
        """Number of states in the automaton."""
        return self._size

    @property
    def text_len(self) -> int:
        """Number of tokens that have been appended."""
        return self._text_len

    def extend(self, token_id: int) -> None:
        """Append one token to the automaton. O(1) amortized.

        This is the standard online SAM construction:
        1. Create a new state 'cur' for the extended text
        2. Walk up suffix links from 'last', adding transitions to 'cur'
        3. Handle the case where we need to clone a state

        Args:
            token_id: The integer token ID to append.
        """
        cur = self._size
        new_node = _SAMNode(self._nodes[self._last].len + 1)
        new_node.first_end_pos = new_node.len - 1
        self._nodes.append(new_node)
        self._size += 1
        self._text_len += 1

        p = self._last
        while p != -1 and token_id not in self._nodes[p].transitions:
            self._nodes[p].transitions[token_id] = cur
            p = self._nodes[p].link

        if p == -1:
            # No existing state has this transition; link to initial
            self._nodes[cur].link = 0
        else:
            q = self._nodes[p].transitions[token_id]
            if self._nodes[p].len + 1 == self._nodes[q].len:
                # q is the correct suffix link target
                self._nodes[cur].link = q
            else:
                # Need to clone q
                clone = self._size
                clone_node = _SAMNode(self._nodes[p].len + 1)
                clone_node.link = self._nodes[q].link
                clone_node.transitions = dict(self._nodes[q].transitions)
                clone_node.first_end_pos = self._nodes[q].first_end_pos
                self._nodes.append(clone_node)
                self._size += 1

                # Redirect transitions from p's suffix chain
                while (p != -1
                       and self._nodes[p].transitions.get(token_id) == q):
                    self._nodes[p].transitions[token_id] = clone
                    p = self._nodes[p].link

                self._nodes[q].link = clone
                self._nodes[cur].link = clone

        self._last = cur

    def find_longest_match(
        self,
        pattern: np.ndarray,
        min_match_len: int = 2,
    ) -> Tuple[int, int]:
        """Find the longest prefix of pattern that exists in the automaton.

        Traverses the automaton following the pattern tokens. Stops at the
        first token that has no transition.

        Args:
            pattern: Token sequence to match (typically the last n tokens
                of the context).
            min_match_len: Minimum number of tokens that must match for
                the result to be considered valid.

        Returns:
            (matched_length, first_end_pos) where:
                - matched_length: number of pattern tokens matched (0 if none)
                - first_end_pos: the first_end_pos of the final matched state,
                  which indicates where in the original text the matched
                  substring ends (exclusive). Returns -1 if no valid match.
        """
        node_idx = 0
        matched_len = 0

        for token in pattern:
            token_int = int(token)
            if token_int in self._nodes[node_idx].transitions:
                node_idx = self._nodes[node_idx].transitions[token_int]
                matched_len += 1
            else:
                break

        if matched_len < min_match_len:
            return (0, -1)

        return (matched_len, self._nodes[node_idx].first_end_pos)

    def find_all_match_lengths(
        self,
        pattern: np.ndarray,
        min_match_len: int = 2,
    ) -> List[Tuple[int, int]]:
        """Find matches at multiple prefix lengths of the pattern.

        This traverses the automaton following the pattern, recording the
        state at each step. Returns matches at all valid prefix lengths,
        enabling the caller to choose the best one.

        Args:
            pattern: Token sequence to match.
            min_match_len: Minimum match length to include.

        Returns:
            List of (matched_length, first_end_pos) for each valid prefix
            length, ordered from longest to shortest match.
        """
        results: List[Tuple[int, int]] = []
        node_idx = 0
        matched_len = 0

        for token in pattern:
            token_int = int(token)
            if token_int in self._nodes[node_idx].transitions:
                node_idx = self._nodes[node_idx].transitions[token_int]
                matched_len += 1
                if matched_len >= min_match_len:
                    end_pos = self._nodes[node_idx].first_end_pos
                    if end_pos >= 0:
                        results.append((matched_len, end_pos))
            else:
                break

        # Return longest match first
        results.reverse()
        return results


class SuffixAutomatonProposer:
    """Stateful Suffix Automaton-based speculative decoding proposer.

    Unlike the stateless SuffixTreeProposer (Optimization 1) which rebuilds
    the suffix array on every propose() call, this proposer maintains an
    incremental Suffix Automaton per request. On each call:
      - First call for a request: builds SAM from all context tokens O(n)
      - Subsequent calls: only appends new tokens O(new_tokens)

    This reduces per-step overhead from O(n * log^2(n)) to O(1) amortized
    for the index building, while keeping O(m) query time.

    The proposer uses req_id (string) to track per-request state, avoiding
    issues with req_index changes during batch condense operations.

    External interface is compatible with NgramProposer and SuffixTreeProposer
    for drop-in replacement, with an additional req_id parameter.
    """

    def __init__(self) -> None:
        # Per-request SAM state, keyed by req_id (str)
        self._automata: Dict[str, IncrementalSuffixAutomaton] = {}
        # Track how many tokens have been fed into each request's SAM
        self._fed_len: Dict[str, int] = {}

    def propose(
        self,
        context_token_ids: np.ndarray,
        n: int,
        k: int,
        req_id: str = "",
    ) -> Optional[np.ndarray]:
        """Generate draft token proposals using incremental SAM.

        Algorithm:
        1. Incrementally update (or create) the SAM for this request
           with any new tokens since last call
        2. Try matching the last n tokens of context against the SAM
        3. If match found, return up to k tokens following the match
        4. If no match at length n, try adaptive fallback to shorter lengths

        Args:
            context_token_ids: Numpy array of token IDs (prompt + output +
                newly sampled). The SAM is built on context[:-1] to avoid
                self-matching the tail.
            n: Starting match length (will fall back to shorter if needed).
            k: Maximum number of draft tokens to return.
            req_id: Request identifier for stateful tracking. If empty,
                falls back to stateless mode (rebuilds each time).

        Returns:
            np.ndarray: Draft token sequence, or None if no match found.
        """
        context_len = len(context_token_ids)
        if context_len < n + 1:
            return None

        # The search text is context[:-1] to avoid self-matching
        search_len = context_len - 1

        # Update or create SAM
        sam = self._get_or_create_sam(
            req_id, context_token_ids, search_len)

        # Adaptive matching: try from n down to max(2, n//2)
        min_match = max(2, n // 2)

        for match_len in range(n, min_match - 1, -1):
            if context_len < match_len:
                continue

            pattern = context_token_ids[context_len - match_len:]

            matched, end_pos = sam.find_longest_match(pattern, match_len)

            if matched >= match_len and end_pos >= 0:
                # end_pos is the end position (exclusive) of the matched
                # substring in the text fed to the SAM.
                # The continuation starts right after the match.
                cont_start = end_pos + 1
                if cont_start >= search_len:
                    continue

                available = search_len - cont_start
                actual_k = min(k, available)
                if actual_k <= 0:
                    continue

                return context_token_ids[cont_start:cont_start + actual_k]

        return None

    def _get_or_create_sam(
        self,
        req_id: str,
        context_token_ids: np.ndarray,
        search_len: int,
    ) -> IncrementalSuffixAutomaton:
        """Get existing SAM and incrementally update, or create a new one.

        Args:
            req_id: Request identifier.
            context_token_ids: Full context token array.
            search_len: Number of tokens to index (context_len - 1).

        Returns:
            The up-to-date IncrementalSuffixAutomaton for this request.
        """
        if not req_id or req_id not in self._automata:
            # First time: build SAM from scratch
            sam = IncrementalSuffixAutomaton()
            for i in range(search_len):
                sam.extend(int(context_token_ids[i]))
            if req_id:
                self._automata[req_id] = sam
                self._fed_len[req_id] = search_len
            return sam

        # Incremental update: only append new tokens
        sam = self._automata[req_id]
        prev_len = self._fed_len[req_id]

        if search_len > prev_len:
            for i in range(prev_len, search_len):
                sam.extend(int(context_token_ids[i]))
            self._fed_len[req_id] = search_len
        elif search_len < prev_len:
            # Context shrank (e.g., request was preempted and resumed).
            # Rebuild from scratch since SAM doesn't support deletion.
            sam = IncrementalSuffixAutomaton()
            for i in range(search_len):
                sam.extend(int(context_token_ids[i]))
            self._automata[req_id] = sam
            self._fed_len[req_id] = search_len

        return sam

    def remove_request(self, req_id: str) -> None:
        """Clean up SAM state when a request finishes.

        Must be called when requests are removed from the batch to prevent
        memory leaks.

        Args:
            req_id: The request ID to clean up.
        """
        self._automata.pop(req_id, None)
        self._fed_len.pop(req_id, None)

    def num_active_requests(self) -> int:
        """Return the number of requests with active SAM state."""
        return len(self._automata)
