# SPDX-License-Identifier: Apache-2.0

import enum
import math
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.inputs import PlaceholderRange


# ---- Token Rate Limiter (Token Bucket) ----
# Per-request token generation rate control.  High-priority requests are
# not rate-limited; low-priority requests are throttled so that compute
# resources (token_budget) are freed for high-priority requests.
# This is analogous to IO QoS token-bucket / leaky-bucket rate limiting
# in cloud storage systems.

class TokenRateLimiter:
    """Per-request token bucket rate limiter.

    Each request carries its own token bucket.  The bucket is refilled at
    ``rate`` tokens per scheduling step.  ``burst`` controls the maximum
    number of tokens that can accumulate (for bursty generation).

    A rate of ``math.inf`` means *no limit* (high-priority / idle system).
    """

    # Default rates per priority tier.  These are overridden by the
    # scheduler based on system load.
    #   HIGH  -> unlimited (no throttle)
    #   NORMAL -> moderate limit under load
    #   LOW   -> aggressive limit under load
    DEFAULT_RATE_HIGH: float = math.inf
    DEFAULT_RATE_NORMAL: float = 64.0   # tokens per step
    DEFAULT_RATE_LOW: float = 16.0      # tokens per step
    DEFAULT_BURST: int = 128            # max burst capacity

    # Load thresholds: when the fraction of running requests relative to
    # max_num_running_reqs exceeds these thresholds, rate limiting kicks in.
    LOAD_THRESHOLD_MODERATE: float = 0.5   # 50% → start limiting low-prio
    LOAD_THRESHOLD_HIGH: float = 0.8       # 80% → aggressive limiting

    def __init__(
        self,
        rate: float = math.inf,
        burst: int = 128,
    ) -> None:
        self.rate = rate      # tokens replenished per step
        self.burst = burst    # maximum bucket capacity
        self.tokens = float(burst)  # current available tokens

    def refill(self) -> None:
        """Refill the bucket (called once per scheduling step)."""
        if self.rate == math.inf:
            self.tokens = float(self.burst)
        else:
            self.tokens = min(self.tokens + self.rate, float(self.burst))

    def consume(self, requested: int) -> int:
        """Try to consume ``requested`` tokens.

        Returns the number of tokens actually allowed (may be less than
        ``requested`` if the bucket is drained).
        """
        if self.rate == math.inf:
            return requested
        allowed = min(requested, max(0, int(self.tokens)))
        self.tokens -= allowed
        return allowed

    def available(self) -> int:
        """Return the number of tokens currently available."""
        if self.rate == math.inf:
            return 2**31  # effectively unlimited
        return max(0, int(self.tokens))

    def set_rate(self, rate: float, burst: Optional[int] = None) -> None:
        """Dynamically adjust the rate and optionally the burst size."""
        self.rate = rate
        if burst is not None:
            self.burst = burst

    def is_limited(self) -> bool:
        """Return True if this limiter imposes any restriction."""
        return self.rate != math.inf


# ---- MLFQ Level Configuration ----
# Multi-Level Feedback Queue: requests start at the highest priority level
# (L0) and are demoted to lower levels as they consume more token budget.
# Short requests naturally finish at high levels; long requests gradually
# sink to lower levels — no explicit priority annotation needed.

class MLFQLevel:
    """Configuration for a single MLFQ level."""

    def __init__(self, level: int, name: str, token_quota: float):
        self.level = level
        self.name = name
        # Max cumulative tokens a request can consume at this level
        # before being demoted to the next level.
        self.token_quota = token_quota

    def __repr__(self) -> str:
        return f"MLFQLevel(L{self.level}, {self.name}, quota={self.token_quota})"


# Default MLFQ level definitions.
# token_quota is the cumulative token budget threshold for demotion.
MLFQ_LEVELS: List[MLFQLevel] = [
    MLFQLevel(level=0, name="interactive", token_quota=128),
    MLFQLevel(level=1, name="standard", token_quota=512),
    MLFQLevel(level=2, name="batch", token_quota=2048),
    MLFQLevel(level=3, name="background", token_quota=math.inf),
]

# Number of MLFQ levels.
MLFQ_NUM_LEVELS: int = len(MLFQ_LEVELS)


class Request:

    # ---- QoS Priority Configuration ----
    # Prompt length thresholds for automatic priority boosting.
    # Requests shorter than SHORT_PROMPT_THRESHOLD get a priority boost
    # (lower effective_priority value = higher scheduling priority).
    SHORT_PROMPT_THRESHOLD: int = 512      # tokens
    MEDIUM_PROMPT_THRESHOLD: int = 2048    # tokens

    # Priority boost values applied based on prompt length.
    # These are *subtracted* from the base priority, so a positive boost
    # means higher scheduling priority (smaller effective_priority).
    SHORT_PROMPT_BOOST: int = 2    # short requests get significant boost
    MEDIUM_PROMPT_BOOST: int = 1   # medium requests get moderate boost
    LONG_PROMPT_PENALTY: int = 1   # long requests get slight penalty

    # Anti-starvation: how fast waiting time decays priority.
    # Every STARVATION_DECAY_INTERVAL seconds of waiting reduces
    # effective_priority by 1 (i.e., raises scheduling priority).
    STARVATION_DECAY_INTERVAL: float = 5.0   # seconds
    MAX_STARVATION_BOOST: int = 10           # cap on starvation boost

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        multi_modal_inputs: Optional[List["MultiModalKwargs"]],
        multi_modal_hashes: Optional[List[str]],
        multi_modal_placeholders: Optional[List["PlaceholderRange"]],
        sampling_params: SamplingParams,
        eos_token_id: Optional[int],
        arrival_time: float,
        lora_request: Optional[LoRARequest] = None,
        priority: int = 0,
    ) -> None:
        self.request_id = request_id
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.arrival_time = arrival_time
        self.lora_request = lora_request

        self.status = RequestStatus.WAITING
        self.events: List[EngineCoreEvent] = []
        self.stop_reason: Union[int, str, None] = None
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens

        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._output_token_ids: List[int] = []
        self._all_token_ids: List[int] = self.prompt_token_ids.copy()
        self.spec_token_ids: List[int] = []
        self.num_computed_tokens = 0

        # Multi-modal related
        self.mm_positions = multi_modal_placeholders or []
        self.mm_inputs = multi_modal_inputs or []
        self.mm_hashes: List[str] = multi_modal_hashes or []

        # Sanity check
        assert len(self.mm_inputs) == len(self.mm_positions)
        if self.mm_hashes:
            assert len(self.mm_inputs) == len(self.mm_hashes)

        # Read-only views
        # Prevent directly appending to the these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)

        # ---- QoS Priority Fields ----
        # Base priority from API caller (lower value = higher priority).
        self.priority = priority
        # Effective priority computed by multi-dimensional formula.
        # Updated dynamically each scheduling step.
        self._effective_priority: float = float(priority)

        # ---- Token Rate Limiter Fields ----
        # Per-request token bucket for rate-limiting token generation.
        # High-priority requests get unlimited rate; low-priority requests
        # are throttled under load to free token_budget for high-priority ones.
        self.rate_limiter: TokenRateLimiter = TokenRateLimiter()

        # ---- MLFQ (Multi-Level Feedback Queue) Fields ----
        # Current MLFQ level (0 = highest priority, MLFQ_NUM_LEVELS-1 = lowest).
        self.mlfq_level: int = 0
        # Cumulative output tokens generated so far for MLFQ accounting.
        # This tracks the "CPU time" consumed by the request, used to
        # decide when to demote to a lower level.
        self.mlfq_tokens_consumed: int = 0

    # ---- QoS Priority Methods ----

    @property
    def effective_priority(self) -> float:
        """Return the last computed effective priority.

        Lower value = higher scheduling priority.
        """
        return self._effective_priority

    def compute_effective_priority(self,
                                   now: Optional[float] = None) -> float:
        """Compute multi-dimensional effective priority.

        Formula:
            effective_priority = base_priority
                                 + length_adjustment
                                 - starvation_boost

        Where:
        - base_priority: API-provided static priority (default 0)
        - length_adjustment: automatic boost based on prompt token count
          (short prompts get negative adjustment = higher priority)
        - starvation_boost: increases over time to prevent starvation
          (subtracted, so longer wait = higher priority)
        """
        if now is None:
            now = time.time()

        base = self.priority

        # Length-based adjustment
        if self.num_prompt_tokens < self.SHORT_PROMPT_THRESHOLD:
            length_adjustment = -self.SHORT_PROMPT_BOOST
        elif self.num_prompt_tokens < self.MEDIUM_PROMPT_THRESHOLD:
            length_adjustment = -self.MEDIUM_PROMPT_BOOST
        else:
            length_adjustment = self.LONG_PROMPT_PENALTY

        # Anti-starvation boost based on waiting time
        waiting_time = max(0.0, now - self.arrival_time)
        starvation_boost = min(
            int(waiting_time / self.STARVATION_DECAY_INTERVAL),
            self.MAX_STARVATION_BOOST,
        )

        # Combine: lower value = higher priority
        self._effective_priority = base + length_adjustment - starvation_boost
        return self._effective_priority

    # ---- MLFQ Methods ----

    def mlfq_account_tokens(self, num_tokens: int) -> None:
        """Account for tokens consumed by this request in the MLFQ.

        Called after each scheduling step with the number of *output* tokens
        generated.  When the cumulative consumption exceeds the current
        level's quota the request is automatically demoted.
        """
        self.mlfq_tokens_consumed += num_tokens
        # Check if demotion is needed.
        current_level = MLFQ_LEVELS[self.mlfq_level]
        if (self.mlfq_tokens_consumed >= current_level.token_quota
                and self.mlfq_level < MLFQ_NUM_LEVELS - 1):
            self.mlfq_level += 1

    def mlfq_promote(self) -> None:
        """Promote the request by one MLFQ level (anti-starvation).

        Called when a request is preempted — it gets promoted one level
        (but not beyond L0) so that it receives slightly better treatment
        when re-scheduled.  The token consumption counter is **not** reset
        so that the request cannot game the system by being repeatedly
        preempted.
        """
        if self.mlfq_level > 0:
            self.mlfq_level -= 1

    def __lt__(self, other: "Request") -> bool:
        """Compare two requests for priority scheduling.

        Uses effective_priority (multi-dimensional) if computed,
        otherwise falls back to arrival_time (FCFS).
        Lower effective_priority = higher scheduling priority.
        """
        self_prio = self.effective_priority
        other_prio = other.effective_priority
        if self_prio != other_prio:
            return self_prio < other_prio
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        return self.request_id < other.request_id

    @classmethod
    def from_engine_core_request(cls, request: EngineCoreRequest) -> "Request":
        return cls(
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            multi_modal_inputs=request.mm_inputs,
            multi_modal_hashes=request.mm_hashes,
            multi_modal_placeholders=request.mm_placeholders,
            sampling_params=request.sampling_params,
            eos_token_id=request.eos_token_id,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
        )

    def queued(self, timestamp: Optional[float] = None) -> None:
        self.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.QUEUED, timestamp))

    def scheduled(self, timestamp: Optional[float] = None) -> None:
        self.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED,
                                      timestamp))

    def take_events(self) -> Optional[List[EngineCoreEvent]]:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    def append_output_token_ids(
        self,
        token_ids: Union[int, List[int]],
    ) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> Union[FinishReason, None]:
        return RequestStatus.get_finished_reason(self.status)

    def has_encoder_inputs(self) -> bool:
        return len(self.mm_inputs) > 0

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_positions)

    def get_num_encoder_tokens(self, input_id: int) -> int:
        assert input_id < len(self.mm_positions)
        num_tokens = self.mm_positions[input_id]["length"]
        return num_tokens


class RequestStatus(enum.IntEnum):
    """Status of a request."""
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    # Note: anything after PREEMPTED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(
            status: "RequestStatus") -> Union[FinishReason, None]:
        return _FINISHED_REASON_MAP.get(status)


# Mapping of finished statuses to their finish reasons.
# NOTE: The ignored requests are the requests whose prompt lengths
# are longer than the model's length cap. Therefore, the stop
# reason should also be "length" as in OpenAI API.
_FINISHED_REASON_MAP = {
    RequestStatus.FINISHED_STOPPED: FinishReason.STOP,
    RequestStatus.FINISHED_LENGTH_CAPPED: FinishReason.LENGTH,
    RequestStatus.FINISHED_ABORTED: FinishReason.ABORT,
    RequestStatus.FINISHED_IGNORED: FinishReason.LENGTH,
}
