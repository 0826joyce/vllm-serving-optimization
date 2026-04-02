# SPDX-License-Identifier: Apache-2.0

import enum
import time
from typing import TYPE_CHECKING, List, Optional, Union

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreRequest, FinishReason)
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.multimodal import MultiModalKwargs
    from vllm.multimodal.inputs import PlaceholderRange


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
