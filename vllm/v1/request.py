# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
import math
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import (
    EngineCoreEvent,
    EngineCoreEventType,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class StreamingUpdate:
    """Lightweight data for streaming session continuation.

    Contains only the fields needed to update an existing streaming session
    with new input data.
    """

    mm_features: list[MultiModalFeatureSpec] | None
    prompt_token_ids: list[int] | None
    max_tokens: int
    arrival_time: float
    sampling_params: SamplingParams | None

    @classmethod
    def from_request(cls, request: "Request") -> "StreamingUpdate | None":
        if not request.resumable:
            return None
        return cls(
            mm_features=request.mm_features,
            prompt_token_ids=request.prompt_token_ids,
            max_tokens=request.max_tokens,
            arrival_time=request.arrival_time,
            sampling_params=request.sampling_params,
        )


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

    # ---- SLA / Deadline Configuration (Phase 4) ----
    # Default SLA TTFT targets per QoS tier (milliseconds).
    # These are used when no explicit sla_ttft_ms is provided.
    DEFAULT_SLA_TTFT_MS: Dict[str, float] = {
        "HIGH": 500.0,       # Gold-A/B: 500ms TTFT target
        "NORMAL": 1000.0,    # Silver: 1s TTFT target
        "LOW": 5000.0,       # Bronze: 5s TTFT target
    }

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
        tenant_id: str = "default",
        sla_ttft_ms: float = float('inf'),
    ) -> None:
        self.request_id = request_id
        self.client_index = client_index
        self.priority = priority
        # QoS: cached effective priority (updated by scheduler each step)
        self._effective_priority: int | None = None
        self.sampling_params = sampling_params
        # Because of LoRA, the eos token id can be different for each request.
        self.eos_token_id = eos_token_id
        self.arrival_time = arrival_time
        self.lora_request = lora_request
        # ---- Tenant isolation (Phase 3) ----
        self.tenant_id = tenant_id

        # ---- SLA / Deadline tracking (Phase 4) ----
        # sla_ttft_ms: maximum acceptable time-to-first-token in ms.
        # deadline: absolute monotonic time by which TTFT must occur.
        self.sla_ttft_ms = sla_ttft_ms
        if sla_ttft_ms < float('inf'):
            self.deadline: float = arrival_time + sla_ttft_ms / 1000.0
        else:
            self.deadline = float('inf')

        self.status = RequestStatus.WAITING
        self.events: list[EngineCoreEvent] = []
        self.stop_reason: int | str | None = None

        # P/D: Connector-specific KV transfer parameters.
        self.kv_transfer_params: dict[str, Any] | None = None

        if pooling_params is not None:
            # Pooling models.
            self.max_tokens = 1
        elif sampling_params is not None:
            # Generative models.
            assert sampling_params.max_tokens is not None
            self.max_tokens = sampling_params.max_tokens
            if self.structured_output_request is not None:
                self.status = RequestStatus.WAITING_FOR_FSM

            if sampling_params.extra_args is not None:
                self.kv_transfer_params = sampling_params.extra_args.get(
                    "kv_transfer_params"
                )
        else:
            raise ValueError("sampling_params and pooling_params can't both be unset")

        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        # Cache per-block prompt-embed hashes to avoid rehashing the same
        # tensor slices when generating extra keys.
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
        self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            prompt_token_ids, prompt_embeds
        )
        self._output_token_ids: list[int] = []
        self._all_token_ids: list[int] = (
            self.prompt_token_ids.copy()
            if self.prompt_token_ids is not None
            else [0] * self.num_prompt_tokens
        )

        # Used in async scheduling.
        self.num_output_placeholders = 0
        # Used in forced preemption (reset_prefix_cache) with async scheduling.
        self.discard_latest_async_tokens = False

        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt

        # Multi-modal related
        self.mm_features = mm_features or []

        # Read-only views
        # Prevent directly appending to these lists since
        # they should also be updated simultaneously.
        self.output_token_ids = ConstantList(self._output_token_ids)
        self.all_token_ids = ConstantList(self._all_token_ids)
        # trace_headers
        self.trace_headers = trace_headers
        # State
        # The number of tokens with prefix cache hits.
        self.num_cached_tokens = -1

        # True if this request is scheduled as a non-final prefill chunk.
        self.is_prefill_chunk = False

        # The number of NaNs in logits. A value greater than 0
        # indicates that the output is corrupted
        self.num_nans_in_logits = 0

        # The number of times this request has been preempted by the scheduler.
        self.num_preemptions = 0

        # The number of tokens that have been computed remotely.
        self.num_external_computed_tokens = 0

        self.block_hashes: list[BlockHash] = []
        # Store the block hasher without binding self to avoid creating a
        # reference cycle (Request -> partial -> Request) that prevents
        # immediate garbage collection via reference counting.
        self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
        self.update_block_hashes()

        self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

        # Used for streaming
        self.resumable = resumable
        # None entry in the queue means finished.
        self.streaming_queue: deque[StreamingUpdate | None] | None = None

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

    # ---- SLA / Deadline Methods (Phase 4) ----

    @property
    def slack_time(self) -> float:
        """Remaining time before SLA deadline (seconds).

        Positive = still within SLA; negative = already violated.
        Returns ``float('inf')`` if no SLA is configured.
        """
        if self.deadline == float('inf'):
            return float('inf')
        return self.deadline - time.monotonic()

    def is_sla_violated(self) -> bool:
        """Return True if the request has exceeded its SLA deadline."""
        return self.slack_time <= 0

    @property
    def sla_urgency(self) -> float:
        """Urgency score for deadline-aware scheduling.

        Lower value = more urgent (closer to or past deadline).
        Requests with no SLA get ``float('inf')`` (least urgent).
        """
        return self.slack_time

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
    def from_engine_core_request(
        cls,
        request: EngineCoreRequest,
        block_hasher: Callable[["Request"], list["BlockHash"]] | None,
    ) -> "Request":
        return cls(
            request_id=request.request_id,
            client_index=request.client_index,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            mm_features=request.mm_features,
            sampling_params=request.sampling_params,
            pooling_params=request.pooling_params,
            arrival_time=request.arrival_time,
            lora_request=request.lora_request,
            tenant_id=getattr(request, 'tenant_id', 'default'),
            sla_ttft_ms=getattr(request, 'sla_ttft_ms', float('inf')),
        )

    def append_output_token_ids(
        self,
        token_ids: int | list[int],
    ) -> None:
        if isinstance(token_ids, int):
            self._output_token_ids.append(token_ids)
            self._all_token_ids.append(token_ids)
        else:
            self._output_token_ids.extend(token_ids)
            self._all_token_ids.extend(token_ids)

        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        """Compute block hashes for any new full blocks and append them."""
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))

    @property
    def use_structured_output(self) -> bool:
        return self.structured_output_request is not None

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_tokens_with_spec(self) -> int:
        return len(self._all_token_ids) + len(self.spec_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    @property
    def num_encoder_inputs(self) -> int:
        return len(self.mm_features)

    @property
    def has_encoder_inputs(self) -> bool:
        return self.num_encoder_inputs > 0

    def get_skip_reading_prefix_cache(self) -> bool:
        if (
            self.sampling_params is not None
            and self.sampling_params.skip_reading_prefix_cache is not None
        ):
            return self.sampling_params.skip_reading_prefix_cache
        elif (
            self.pooling_params is not None
            and self.pooling_params.skip_reading_prefix_cache is not None
        ):
            return self.pooling_params.skip_reading_prefix_cache
        return False

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)

    def get_finished_reason(self) -> FinishReason | None:
        return RequestStatus.get_finished_reason(self.status)

    def get_num_encoder_embeds(self, input_id: int) -> int:
        assert input_id < len(self.mm_features)
        return self.mm_features[input_id].mm_position.get_num_embeds()

    def record_event(
        self,
        event_type: EngineCoreEventType,
        timestamp: float | None = None,
    ) -> None:
        self.events.append(EngineCoreEvent.new_event(event_type, timestamp))

    def take_events(self) -> list[EngineCoreEvent] | None:
        if not self.events:
            return None
        events, self.events = self.events, []
        return events

    @property
    def effective_priority(self) -> int:
        """Get the effective priority for scheduling decisions.

        This is the multi-dimensional priority that combines:
        1. API-provided base priority
        2. Prompt length-based boost (short requests get higher priority)
        3. Waiting time anti-starvation decay

        Lower value = higher scheduling priority.
        If not yet computed, falls back to static priority.
        """
        if self._effective_priority is not None:
            return self._effective_priority
        return self.priority

    def compute_effective_priority(self, now: float | None = None) -> int:
        """Compute and cache the multi-dimensional effective priority.

        This method should be called by the scheduler at the beginning of
        each scheduling step for all waiting requests.

        Args:
            now: Current time (monotonic). If None, uses time.time().

        Returns:
            The computed effective priority value.
            Lower value = higher scheduling priority.
        """
        if now is None:
            now = time.time()

        # Start with API-provided base priority.
        # Lower priority value = higher scheduling priority.
        base = self.priority

        # 1. Prompt length-based boost:
        #    Short prompts (<512 tokens) get priority boost (subtracted).
        #    Long prompts (>2048 tokens) get slight penalty (added).
        num_prompt = self.num_prompt_tokens
        if num_prompt < self.SHORT_PROMPT_THRESHOLD:
            length_adjustment = -self.SHORT_PROMPT_BOOST
        elif num_prompt < self.MEDIUM_PROMPT_THRESHOLD:
            length_adjustment = -self.MEDIUM_PROMPT_BOOST
        else:
            length_adjustment = self.LONG_PROMPT_PENALTY

        # 2. Anti-starvation waiting time decay:
        #    Every STARVATION_DECAY_INTERVAL seconds of waiting reduces
        #    effective priority by 1 (capped at MAX_STARVATION_BOOST).
        waiting_time = now - self.arrival_time
        starvation_boost = min(
            int(waiting_time / self.STARVATION_DECAY_INTERVAL),
            self.MAX_STARVATION_BOOST,
        )

        # Combine: lower value = higher priority
        self._effective_priority = base + length_adjustment - starvation_boost
        return self._effective_priority

    def __lt__(self, other: "Request") -> bool:
        """
        Compare two requests based on effective priority, arrival time,
        and request ID. Used in priority scheduling.

        Uses effective_priority (multi-dimensional) if computed,
        otherwise falls back to static priority.
        """
        self_prio = self.effective_priority
        other_prio = other.effective_priority
        if self_prio != other_prio:
            return self_prio < other_prio
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)


class RequestStatus(enum.IntEnum):
    """Status of a request."""

    WAITING = enum.auto()
    WAITING_FOR_FSM = enum.auto()
    WAITING_FOR_REMOTE_KVS = enum.auto()
    WAITING_FOR_STREAMING_REQ = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    # Note: anything after PREEMPTED will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6
    FINISHED_REJECTED = 7  # Phase 4: Admission control rejection

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> FinishReason | None:
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
    RequestStatus.FINISHED_REJECTED: FinishReason.ABORT,
}
