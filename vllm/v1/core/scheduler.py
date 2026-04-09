# SPDX-License-Identifier: Apache-2.0

import heapq
import itertools
import math
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import (CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.logger import init_logger
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.scheduler_output import (CachedRequestData, NewRequestData,
                                           SchedulerOutput)
from vllm.v1.engine import (EngineCoreEvent, EngineCoreEventType,
                            EngineCoreOutput, EngineCoreOutputs)
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import (MLFQ_LEVELS, MLFQ_NUM_LEVELS, Request,
                             RequestStatus, TokenRateLimiter)

logger = init_logger(__name__)


class Scheduler:

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        log_stats: bool,
    ) -> None:
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.lora_config = lora_config
        self.speculative_config = speculative_config
        self.log_stats = log_stats

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len

        num_gpu_blocks = cache_config.num_gpu_blocks
        assert isinstance(num_gpu_blocks, int) and num_gpu_blocks > 0
        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_model_len=self.max_model_len,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching,
            log_stats=self.log_stats)
        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: Dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: Deque[Request] = deque()
        self.running: List[Request] = []

        # ---- QoS Priority Scheduling ----
        # When enabled, the scheduler uses effective_priority to:
        # 1. Sort waiting queue (priority heap) each scheduling step
        # 2. Choose preemption victims (lowest priority = highest value)
        # Enable via scheduling_policy="priority" in SchedulerConfig,
        # or default to True when available.
        self.enable_qos_priority: bool = getattr(
            scheduler_config, 'scheduling_policy', 'fcfs') == 'priority'

        # ---- MLFQ (Multi-Level Feedback Queue) Scheduling ----
        # When enabled, the waiting queue is replaced by N level-queues.
        # Requests start at L0 (highest priority) and are demoted as they
        # consume more tokens.  Scheduling always picks from the highest
        # non-empty level first; within a level, FCFS ordering is used.
        self.enable_mlfq: bool = True
        # Multi-level waiting queues: mlfq_queues[i] holds waiting requests
        # at MLFQ level i.  Level 0 = highest priority (interactive).
        self.mlfq_queues: List[Deque[Request]] = [
            deque() for _ in range(MLFQ_NUM_LEVELS)
        ]

        # ---- Cache-Aware Scheduling ----
        # When enabled (and prefix caching is on), the scheduler scans the
        # top-K candidates within the same MLFQ level and re-orders them
        # by actual prefill tokens (ascending), so requests with higher
        # cache hit rates are scheduled first, maximising token_budget
        # utilisation and reducing TTFT.
        self.enable_cache_aware_scheduling: bool = (
            self.cache_config.enable_prefix_caching)
        # Number of candidates to scan within each MLFQ level for
        # cache-aware reordering.  Larger values give better ordering but
        # increase per-step overhead (one get_computed_blocks call per
        # candidate).
        self.cache_aware_scan_window: int = 8

        # ---- Token Rate Limiting (Token Bucket) ----
        # When enabled, running requests are rate-limited based on their
        # priority.  High-priority requests are not limited; low-priority
        # requests are throttled under load so that token_budget is freed
        # for high-priority requests.
        self.enable_token_rate_limiting: bool = True

        # ---- Preemption Cache Shield ----
        # Minimum number of blocks that a partial free must release.
        # If partial free would release fewer blocks than this threshold,
        # fall back to full free to guarantee forward progress in the
        # preemption loop.  Setting to 1 means "partial free must release
        # at least 1 block to be worthwhile".
        self._PREEMPT_MIN_FREE_BLOCKS: int = 1

        # The requests that have been scheduled and are being executed
        # by the executor.
        self.scheduled_req_ids: Set[str] = set()

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: Set[str] = set()

        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        # Request id -> CachedRequestData
        self._cached_reqs_data: Dict[str, CachedRequestData] = {}

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

    def schedule(self) -> "SchedulerOutput":
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: List[Request] = []
        scheduled_resumed_reqs: List[Request] = []
        scheduled_running_reqs: List[Request] = []
        preempted_reqs: List[Request] = []

        req_to_new_block_ids: Dict[str, List[int]] = {}
        num_scheduled_tokens: Dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: Dict[str, List[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: Dict[str, List[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        # ---- QoS: Update effective priorities ----
        # Before scheduling, recompute multi-dimensional priorities for
        # all requests so that waiting queue ordering and preemption
        # decisions reflect the latest state (waiting time, etc.).
        if self.enable_qos_priority:
            self._update_effective_priorities()

        # ---- Token Rate Limiting: Refill buckets & adjust rates ----
        if self.enable_token_rate_limiting:
            self._update_rate_limiters()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            if request.request_id in self.scheduled_req_ids:
                # This request has already been scheduled.
                req_index += 1
                continue

            num_new_tokens = (request.num_tokens_with_spec -
                              request.num_computed_tokens)
            num_new_tokens = min(num_new_tokens, token_budget)
            assert num_new_tokens > 0

            # ---- Token Rate Limiting ----
            # Apply per-request rate limiting for running requests.
            # High-priority requests are not limited; low-priority requests
            # may have their num_new_tokens reduced.
            if (self.enable_token_rate_limiting
                    and request.rate_limiter.is_limited()):
                allowed = request.rate_limiter.consume(num_new_tokens)
                if allowed < num_new_tokens and allowed > 0:
                    num_new_tokens = allowed
                elif allowed == 0:
                    # Bucket is empty — skip this request this step.
                    # It will be scheduled next step when the bucket refills.
                    req_index += 1
                    continue

            # Schedule encoder inputs.
            encoder_inputs_to_schedule, num_new_tokens, new_encoder_budget = (
                self._try_schedule_encoder_inputs(request,
                                                  request.num_computed_tokens,
                                                  num_new_tokens,
                                                  encoder_budget))
            if num_new_tokens == 0:
                # The request cannot be scheduled because the encoder budget
                # or the encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.enable_qos_priority and len(self.running) > 1:
                        # QoS: Preempt the request with highest
                        # effective_priority value (= lowest priority).
                        preempted_req = max(
                            self.running,
                            key=lambda r: (
                                r.effective_priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                    else:
                        # FCFS fallback: preempt last added (LIFO).
                        preempted_req = self.running.pop()
                    # --- Preemption Cache Shield (Optimization 3) ---
                    # Instead of fully freeing all blocks, retain the
                    # prefix blocks that have a block_hash (cacheable).
                    # This protects high-value prefix blocks from eviction
                    # and allows faster resume via get_computed_blocks().
                    #
                    # Fallback: if partial free would release too few
                    # blocks (fewer than _PREEMPT_MIN_FREE_BLOCKS),
                    # fall back to full free to guarantee forward
                    # progress in the preemption loop.
                    preempt_blocks = self.kv_cache_manager.req_to_blocks.get(
                        preempted_req.request_id, [])
                    keep_count = 0
                    if self.kv_cache_manager.enable_caching:
                        for blk in preempt_blocks:
                            if blk.block_hash is not None:
                                keep_count += 1
                            else:
                                break
                    would_free = len(preempt_blocks) - keep_count
                    if (keep_count > 0
                            and would_free
                            >= self._PREEMPT_MIN_FREE_BLOCKS):
                        self.kv_cache_manager.free_partial(
                            preempted_req, keep_count)
                        preempted_req.num_computed_tokens = (
                            keep_count * self.block_size)
                    else:
                        # Either no cacheable prefix, or partial free
                        # would release too few blocks — full free to
                        # ensure the preemption loop makes progress.
                        self.kv_cache_manager.free(preempted_req)
                        preempted_req.num_computed_tokens = 0
                    preempted_req.status = RequestStatus.PREEMPTED

                    if self.enable_mlfq:
                        # MLFQ: promote preempted request by one level
                        # (anti-starvation) and insert into the correct
                        # level queue.
                        preempted_req.mlfq_promote()
                        self.mlfq_queues[preempted_req.mlfq_level].appendleft(
                            preempted_req)
                    self.waiting.appendleft(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            self.scheduled_req_ids.add(request.request_id)
            req_to_new_block_ids[request.request_id] = [
                b.block_id for b in new_blocks
            ]
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

        # Record the LoRAs in scheduled_running_reqs
        requested_loras: Set[int] = set()
        if self.lora_config:
            requested_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(requested_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        # ---- Cache-Aware Scheduling (Optimization 1) ----
        # When enabled, instead of pure FCFS within each MLFQ level, we
        # scan the top-K candidates and reorder them by actual prefill
        # tokens (ascending).  Requests with more cache hits need fewer
        # new tokens and are therefore scheduled first, maximising
        # token_budget utilisation and reducing TTFT.
        if not preempted_reqs:
            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                # --- Select the next candidate (cache-aware or FCFS) ---
                if (self.enable_cache_aware_scheduling
                        and self.enable_mlfq):
                    # Cache-aware selection: scan top-K candidates in the
                    # highest non-empty MLFQ level, pick the one with the
                    # fewest actual prefill tokens.
                    selection = self._cache_aware_select_next()
                    if selection is None:
                        break
                    (request, computed_blocks,
                     num_computed_tokens) = selection
                    cache_info_precomputed = True
                elif self.enable_mlfq:
                    request = self._mlfq_peek_next()
                    if request is None:
                        break
                    cache_info_precomputed = False
                else:
                    request = self.waiting[0]
                    cache_info_precomputed = False

                # Check that adding the request still respects the max_loras
                # constraint.
                if self.lora_config and request.lora_request:
                    req_lora_id = request.lora_request.lora_int_id
                    if len(requested_loras) == self.lora_config.max_loras and (
                            req_lora_id not in requested_loras):
                        # Cannot schedule.
                        # TODO (varun): This means all the other requests in
                        # the WAITING queue will be blocked by this request,
                        # even if,
                        # 1. these other requests do not use LoRA, or,
                        # 2. these other requests use the already requested
                        # LoRAs.
                        # This is too conservative and could be optimized.
                        break

                if not cache_info_precomputed:
                    # Get already-cached tokens (original path).
                    computed_blocks, num_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(request)
                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed requests,
                # which have output tokens.
                num_new_tokens = request.num_tokens - num_computed_tokens
                if num_new_tokens == 0:
                    # This happens when prompt length is divisible by the block
                    # size and all blocks are cached. Now we force to recompute
                    # the last block. Note that we have to re-compute an entire
                    # block because allocate_slots() assumes num_computed_tokens
                    # is always a multiple of the block size. This limitation
                    # can potentially be removed in the future to slightly
                    # improve the performance.
                    num_computed_tokens -= self.block_size
                    num_new_tokens = self.block_size
                    computed_blocks.pop()
                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                # Schedule encoder inputs.
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, num_computed_tokens, num_new_tokens,
                     encoder_budget)
                if num_new_tokens == 0:
                    # The request cannot be scheduled.
                    break

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request, num_new_tokens, computed_blocks)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                if self.enable_mlfq:
                    self.mlfq_queues[request.mlfq_level].remove(request)
                    self.waiting.remove(request)
                else:
                    self.waiting.popleft()
                self.running.append(request)
                self.scheduled_req_ids.add(request.request_id)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                    self.request_scheduled(request, scheduled_timestamp)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    requested_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = [
                    b.block_id for b in computed_blocks + new_blocks
                ]
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens

                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = 0
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        resumed_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=True,
            ) for req in scheduled_resumed_reqs
        ]
        running_reqs_data = [
            self._make_cached_request_data(
                req,
                num_scheduled_tokens[req.request_id],
                len(scheduled_spec_decode_tokens.get(req.request_id, ())),
                req_to_new_block_ids[req.request_id],
                resumed_from_preemption=False,
            ) for req in scheduled_running_reqs
        ]
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=resumed_reqs_data + running_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
        )

        self.finished_req_ids = set()
        return scheduler_output

    def _make_cached_request_data(
        self,
        request: Request,
        num_scheduled_tokens: int,
        num_scheduled_spec_tokens: int,
        new_block_ids: List[int],
        resumed_from_preemption: bool,
    ) -> "CachedRequestData":
        # OPTIMIZATION: Cache the CachedRequestData objects to avoid creating
        # them at each scheduling step.
        num_computed_tokens = request.num_computed_tokens
        num_regular_tokens = num_scheduled_tokens - num_scheduled_spec_tokens
        new_token_ids = request.all_token_ids[
            num_computed_tokens:num_computed_tokens + num_regular_tokens]
        req_data = self._cached_reqs_data.get(request.request_id)
        if req_data is not None:
            req_data.resumed_from_preemption = resumed_from_preemption
            req_data.new_token_ids = new_token_ids
            req_data.new_block_ids = new_block_ids
            req_data.num_computed_tokens = num_computed_tokens
        else:
            req_data = CachedRequestData.from_request(request,
                                                      resumed_from_preemption,
                                                      new_token_ids,
                                                      new_block_ids)
            self._cached_reqs_data[request.request_id] = req_data
        return req_data

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> Tuple[List[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.
        """
        if not request.has_encoder_inputs():
            return [], num_new_tokens, encoder_budget

        encoder_inputs_to_schedule: List[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info["offset"]
            num_encoder_tokens = pos_info["length"]

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue
            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> EngineCoreOutputs:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens

        new_running: List[Request] = []
        outputs: List[EngineCoreOutput] = []

        # NOTE(woosuk): As len(self.running) can be up to 1K or more, the below
        # loop can be a performance bottleneck. We should do our best to avoid
        # expensive operations inside the loop.
        for request in self.running:
            req_id = request.request_id
            num_tokens_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_tokens_scheduled == 0:
                # The request was not scheduled in this step.
                new_running.append(request)
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[req_index]
            if req_id not in scheduler_output.scheduled_spec_decode_tokens:
                # When the request's num_computed_tokens catches up
                # its num_tokens, the request generates output tokens.
                # Otherwise, we ignore the sampler output for the request.
                request.num_computed_tokens += num_tokens_scheduled
                assert request.num_computed_tokens <= request.num_tokens
            else:
                # num_computed_tokens_step represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections.
                # It is calculated as:
                # num_computed_tokens_step = num_scheduled_tokens -
                #                            num_tokens_rejected,
                # where num_tokens_rejected is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                scheduled_spec_token_ids = (
                    scheduler_output.scheduled_spec_decode_tokens[req_id])

                num_computed_tokens_step = num_scheduled_tokens[req_id] - (
                    len(scheduled_spec_token_ids) + 1 -
                    len(generated_token_ids))
                request.num_computed_tokens += num_computed_tokens_step

            cached_encoder_input_ids = (
                self.encoder_cache_manager.get_cached_input_ids(request))
            # OPTIMIZATION: Avoid list(set) if the set is empty.
            if cached_encoder_input_ids:
                for input_id in list(cached_encoder_input_ids):
                    start_pos = request.mm_positions[input_id]["offset"]
                    num_tokens = request.mm_positions[input_id]["length"]
                    if start_pos + num_tokens <= request.num_computed_tokens:
                        # The encoder output is already processed and stored
                        # in the decoder's KV cache.
                        self.encoder_cache_manager.free_encoder_input(
                            request, input_id)

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)

            stopped = False
            new_logprobs = None
            new_token_ids: List[int] = []

            if request.num_computed_tokens >= request.num_tokens:
                for output_token_id in generated_token_ids:
                    request.append_output_token_ids(output_token_id)
                    new_token_ids.append(output_token_id)

                    # Check for stop and update request state.
                    # This must be called before we make the EngineCoreOutput.
                    stopped = self._check_stop(request)
                    if stopped:
                        self._free_request(request)
                        break

                # ---- MLFQ: Account consumed tokens for demotion ----
                if self.enable_mlfq and new_token_ids:
                    request.mlfq_account_tokens(len(new_token_ids))

                # Extract sample logprobs if needed.
                if request.sampling_params.logprobs is not None:
                    assert logprobs is not None
                    # NOTE: once we support N tokens per step (spec decode),
                    # the outer lists can be of length > 1.
                    new_logprobs = logprobs.slice(req_index, req_index + 1)

            # Transmit partial if chunked prefill & prompt logprobs is enabled
            if new_token_ids or prompt_logprobs_tensors is not None:
                # Add EngineCoreOutput for this Request.
                outputs.append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        stop_reason=request.stop_reason,
                        events=request.take_events()))

            self.scheduled_req_ids.remove(request.request_id)
            if not stopped:
                new_running.append(request)

        self.running = new_running
        return EngineCoreOutputs(
            outputs=outputs,
            scheduler_stats=self.make_stats(),
        )

    def _check_stop(self, request: Request) -> bool:
        if (request.num_tokens >= self.max_model_len
                or request.num_output_tokens >= request.max_tokens):
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True

        sampling_params = request.sampling_params
        last_token_id = request.output_token_ids[-1]
        if (not sampling_params.ignore_eos
                and last_token_id == request.eos_token_id):
            request.status = RequestStatus.FINISHED_STOPPED
            return True

        if last_token_id in (sampling_params.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            return True
        return False

    def add_request(self, request: Request) -> None:
        self.requests[request.request_id] = request
        if self.enable_mlfq:
            # New requests always enter at MLFQ level 0 (highest priority).
            request.mlfq_level = 0
            request.mlfq_tokens_consumed = 0
            self.mlfq_queues[0].append(request)
            # Also keep in self.waiting for compatibility (stats, etc.).
            self.waiting.append(request)
        else:
            self.waiting.append(request)
        self.request_queued(request)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        request_ids = set(request_ids)

        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            if request.status == RequestStatus.RUNNING:
                self.running.remove(request)
                if request.request_id in self.scheduled_req_ids:
                    self.scheduled_req_ids.remove(request.request_id)
            else:
                self.waiting.remove(request)
                if self.enable_mlfq:
                    self._mlfq_remove_from_level(request)
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> None:
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        self.encoder_cache_manager.free(request)
        self._cached_reqs_data.pop(request.request_id, None)
        del self.requests[request.request_id]
        self.finished_req_ids.add(request.request_id)

    def _update_effective_priorities(self) -> None:
        """QoS: Update effective priorities for all requests.

        This method implements multi-dimensional priority computation by
        calling compute_effective_priority() on each request.
        The effective priority combines:
        1. API-provided base priority
        2. Prompt length-based boost (short requests get higher priority)
        3. Waiting time anti-starvation decay (prevents starvation)

        After updating priorities, the waiting queue is re-sorted so that
        the highest-priority request is at the front (index 0).
        Running request priorities are also updated for preemption decisions.
        """
        now = time.time()

        # Update waiting queue priorities and re-sort.
        for request in self.waiting:
            request.compute_effective_priority(now)
        # Re-sort waiting queue by effective priority (lowest value first).
        sorted_waiting = sorted(self.waiting)
        self.waiting = deque(sorted_waiting)

        # Update running request priorities for preemption decisions.
        for request in self.running:
            request.compute_effective_priority(now)

    # ---- MLFQ Helper Methods ----

    def _mlfq_peek_next(self) -> Optional[Request]:
        """Return the next request to schedule from the MLFQ queues.

        Scans levels from L0 (highest priority) to L_{N-1} (lowest).
        Returns the front request of the first non-empty level, or None
        if all levels are empty.
        """
        for level_queue in self.mlfq_queues:
            if level_queue:
                return level_queue[0]
        return None

    def _mlfq_pop_next(self) -> Optional[Request]:
        """Pop and return the next request from the MLFQ queues.

        Same scanning order as _mlfq_peek_next but actually removes
        the request from its level queue.
        """
        for level_queue in self.mlfq_queues:
            if level_queue:
                return level_queue.popleft()
        return None

    def _mlfq_remove_from_level(self, request: Request) -> None:
        """Remove a request from its MLFQ level queue.

        Used when a request is finished or aborted while still in the
        waiting state.
        """
        level = request.mlfq_level
        try:
            self.mlfq_queues[level].remove(request)
        except ValueError:
            # The request may not be in the expected level queue if it was
            # already removed (e.g., race between finish and schedule).
            for q in self.mlfq_queues:
                try:
                    q.remove(request)
                    break
                except ValueError:
                    continue

    # ---- Cache-Aware Scheduling Helper Methods ----

    def _cache_aware_select_next(
        self,
    ) -> Optional[Tuple[Request, List, int]]:
        """Select the next WAITING request using cache-aware ordering.

        Within the highest non-empty MLFQ level, scan up to
        ``cache_aware_scan_window`` candidates and return the one that
        requires the fewest new prefill tokens (i.e. highest cache hit
        rate).  This maximises the number of requests that fit within the
        remaining token_budget.

        The method calls ``get_computed_blocks()`` for each scanned
        candidate.  ``get_computed_blocks()`` is safe to call multiple
        times — it only computes and caches block hashes, it does NOT
        modify reference counts.

        Returns:
            A tuple of (request, computed_blocks, num_computed_tokens)
            for the best candidate, or None if all levels are empty.
        """
        for level_queue in self.mlfq_queues:
            if not level_queue:
                continue

            # Take up to K candidates from this level (without removing
            # them from the queue).
            k = min(self.cache_aware_scan_window, len(level_queue))
            candidates = list(itertools.islice(level_queue, k))

            best: Optional[Tuple[int, Request, List, int]] = None
            for req in candidates:
                computed_blocks, num_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(req))
                actual_prefill = req.num_tokens - num_computed_tokens
                if best is None or actual_prefill < best[0]:
                    best = (actual_prefill, req, computed_blocks,
                            num_computed_tokens)

            if best is not None:
                _, req, comp_blocks, n_computed = best
                return (req, comp_blocks, n_computed)
            # Should not reach here since level_queue is non-empty,
            # but fall through to the next level just in case.

        return None

    # ---- Token Rate Limiting Helper Methods ----

    def _update_rate_limiters(self) -> None:
        """Update rate limiters for all running requests based on system load.

        The rate limiting strategy is:
        - Compute system load as len(running) / max_num_running_reqs.
        - If load < LOAD_THRESHOLD_MODERATE: no limiting (all requests run
          at full speed, maximising throughput when system is idle).
        - If load >= LOAD_THRESHOLD_MODERATE: start limiting low-priority
          requests to free token_budget for high-priority ones.
        - If load >= LOAD_THRESHOLD_HIGH: aggressively limit low-priority
          requests.

        Priority classification uses MLFQ levels (if enabled) or
        effective_priority to determine HIGH / NORMAL / LOW tiers:
        - HIGH:   MLFQ L0-L1, or effective_priority < 0
        - NORMAL: MLFQ L2, or 0 <= effective_priority < 5
        - LOW:    MLFQ L3+, or effective_priority >= 5
        """
        if not self.running:
            return

        load = len(self.running) / max(1, self.max_num_running_reqs)

        for request in self.running:
            rl = request.rate_limiter

            # Always refill the bucket at the start of each step.
            rl.refill()

            # Determine the priority tier for this request.
            tier = self._get_priority_tier(request)

            if load < TokenRateLimiter.LOAD_THRESHOLD_MODERATE:
                # System is idle — no rate limiting for anyone.
                rl.set_rate(math.inf)
            elif load < TokenRateLimiter.LOAD_THRESHOLD_HIGH:
                # Moderate load — limit low-priority requests.
                if tier == "HIGH":
                    rl.set_rate(math.inf)
                elif tier == "NORMAL":
                    rl.set_rate(TokenRateLimiter.DEFAULT_RATE_NORMAL)
                else:  # LOW
                    rl.set_rate(TokenRateLimiter.DEFAULT_RATE_LOW)
            else:
                # High load — aggressive limiting.
                if tier == "HIGH":
                    rl.set_rate(math.inf)
                elif tier == "NORMAL":
                    # Under high load, normal requests also get limited,
                    # but less aggressively than low-priority ones.
                    rl.set_rate(TokenRateLimiter.DEFAULT_RATE_LOW * 2)
                else:  # LOW
                    # Very aggressive: half of default low rate.
                    rl.set_rate(max(1.0, TokenRateLimiter.DEFAULT_RATE_LOW / 2))

    def _get_priority_tier(self, request: Request) -> str:
        """Classify a request into HIGH / NORMAL / LOW priority tier.

        Uses MLFQ level (if enabled) as the primary signal, with
        effective_priority as fallback.
        """
        if self.enable_mlfq:
            if request.mlfq_level <= 1:
                return "HIGH"
            elif request.mlfq_level == 2:
                return "NORMAL"
            else:
                return "LOW"
        elif self.enable_qos_priority:
            ep = request.effective_priority
            if ep < 0:
                return "HIGH"
            elif ep < 5:
                return "NORMAL"
            else:
                return "LOW"
        else:
            # No priority information — treat all as HIGH (no limiting).
            return "HIGH"

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_unfinished_requests(self) -> bool:
        return self.get_num_unfinished_requests() > 0

    def get_num_unscheduled_requests(self) -> int:
        """Number of requests that are not being processed by the executor."""
        return self.get_num_unfinished_requests() - len(self.scheduled_req_ids)

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def request_queued(self, request: Request):
        if not self.log_stats:
            return
        request.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.QUEUED))

    def request_scheduled(self, request: Request, timestamp: float):
        if not self.log_stats:
            return
        request.events.append(
            EngineCoreEvent.new_event(EngineCoreEventType.SCHEDULED,
                                      timestamp))

    def make_stats(self) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            gpu_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=self.kv_cache_manager.make_prefix_cache_stats(),
        )
