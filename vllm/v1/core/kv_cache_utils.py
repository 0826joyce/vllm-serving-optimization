# SPDX-License-Identifier: Apache-2.0
"""KV-Cache Utilities."""
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Tuple

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        KVCacheTensor)
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


class BlockHashType(NamedTuple):
    """Hash value of a block (int), the token IDs in the block, and extra keys.
    We keep a tuple of token IDs and extra keys to reduce the likelihood of
    hash collisions when the hash value is the same. But please note that 
    hash collisions can still theoretically occur, albeit with an extremely 
    low probability.
    """
    # Hash value of the block in an integer.
    hash_value: int
    # Token IDs in the block.
    token_ids: Tuple[int, ...]
    # Extra keys for the block.
    extra_keys: Optional[Any] = None


class PrefixCachingMetrics:
    """Metrics for prefix caching with a hit rate of the most recent N requests.

    Args:
        interval: The number of the most recent requests to aggregate.
            Defaults to 1000.
    """

    def __init__(self, interval: int = 1000):
        self.interval = interval
        # The current aggregated values.
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        # A deque of (requests, queries, hits) for the most recent requests.
        self.query_queue: deque[Tuple[int, int, int]] = deque()

    def observe(self, stats: PrefixCacheStats):
        """Observe the prefix caching for a set of requests.

        This function is called with information gathered when new requests
        are being scheduled and are looking for computed blocks.

        When there are more than `interval` requests, the oldest set of
        requestsare removed from the metrics.

        Args:
            stats: The prefix cache stats.
        """
        # reset_prefix_cache was invoked before the current update.
        # Reset the metrics before aggregating the current stats.
        if stats.reset:
            self.reset()

        # Update the metrics.
        self.query_queue.append((stats.requests, stats.queries, stats.hits))
        self.aggregated_requests += stats.requests
        self.aggregated_query_total += stats.queries
        self.aggregated_query_hit += stats.hits

        # Remove the oldest stats if the number of requests exceeds.
        if self.aggregated_requests > self.interval:
            old_requests, old_queries, old_hits = self.query_queue.popleft()
            self.aggregated_requests -= old_requests
            self.aggregated_query_total -= old_queries
            self.aggregated_query_hit -= old_hits

    def reset(self):
        """Reset the metrics."""
        self.aggregated_requests = 0
        self.aggregated_query_total = 0
        self.aggregated_query_hit = 0
        self.query_queue.clear()

    @property
    def hit_rate(self) -> float:
        """Calculate the hit rate for the past N requests."""
        if self.aggregated_query_total == 0:
            return 0.0
        return self.aggregated_query_hit / self.aggregated_query_total


@dataclass
class KVCacheBlock:
    """KV-cache block metadata."""
    # Block ID, ranging from 0 to num_gpu_blocks - 1.
    block_id: int
    # Reference count.
    ref_cnt: int = 0
    # The hash of the block composed of (block hash, tuple of token IDs).
    # It is only available when the block is full.
    _block_hash: Optional[BlockHashType] = None

    # Used to construct a doubly linked list for free blocks.
    # These two attributes should only be manipulated by FreeKVCacheBlockQueue.
    prev_free_block: Optional["KVCacheBlock"] = None
    next_free_block: Optional["KVCacheBlock"] = None

    # ---- Segmented LRU zone tracking ----
    # Which zone of the free queue the block belongs to when it is free.
    # None = not in free queue, "probation" or "protected".
    free_zone: Optional[str] = None
    # Set to True when the block is cache-hit via _touch() while in the
    # free queue.  When the block is later freed again (ref_cnt → 0),
    # it should enter the protected zone instead of probation.
    _promoted: bool = False

    def incr_ref(self):
        self.ref_cnt += 1

    def decr_ref(self):
        self.ref_cnt -= 1

    @property
    def block_hash(self) -> Optional[BlockHashType]:
        return self._block_hash

    @block_hash.setter
    def block_hash(self, block_hash: BlockHashType):
        assert self.block_hash is None, (
            "The block already has a hash. This should not happen.")
        self._block_hash = block_hash

    def reset_hash(self):
        """Reset the block hash when the block is evicted."""
        self._block_hash = None


class FreeKVCacheBlockQueue:
    """Segmented LRU free block queue for frequency-aware eviction.

    This class organizes free KVCacheBlock objects into two zones:

    - **Probation Zone**: Blocks that have been freed but not yet re-accessed.
      These are evicted first (LRU within the zone).
    - **Protected Zone**: Blocks that were re-accessed (cache hit via _touch)
      while in the probation zone.  These are evicted only when the probation
      zone is empty.

    The two-zone design prevents high-frequency prefix blocks (e.g. System
    Prompt) from being evicted by low-frequency long-context blocks.

    All operations remain O(1) using intrusive doubly-linked lists via
    ``prev_free_block`` / ``next_free_block`` pointers on KVCacheBlock.

    When the protected zone exceeds ``max_protected_blocks``, the oldest
    protected block is demoted to the *head* of the probation zone so it
    becomes the next eviction candidate among probation blocks.

    Args:
        blocks: A list of KVCacheBlock objects.
        protected_ratio: Fraction of total blocks reserved for the protected
            zone.  Default 0.5.
    """

    def __init__(self, blocks: List[KVCacheBlock],
                 protected_ratio: float = 0.5) -> None:
        # --- Probation zone (doubly-linked list) ---
        self._probation_head: Optional[KVCacheBlock] = None
        self._probation_tail: Optional[KVCacheBlock] = None
        self._num_probation: int = 0

        # --- Protected zone (doubly-linked list) ---
        self._protected_head: Optional[KVCacheBlock] = None
        self._protected_tail: Optional[KVCacheBlock] = None
        self._num_protected: int = 0

        self._max_protected: int = max(int(len(blocks) * protected_ratio), 0)

        # All initial blocks go into the probation zone.
        for block in blocks:
            self._append_to_zone(block, "probation")

    # ------------------------------------------------------------------
    # Public interface (compatible with the original FreeKVCacheBlockQueue)
    # ------------------------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return self._num_probation + self._num_protected

    def popleft(self) -> KVCacheBlock:
        """Pop the next block to evict.

        Eviction priority: probation head first, then protected head.

        Returns:
            The evicted block.
        """
        if self._num_probation > 0:
            block = self._probation_head
            assert block is not None
            self._remove_from_zone(block, "probation")
            return block
        elif self._num_protected > 0:
            block = self._protected_head
            assert block is not None
            self._remove_from_zone(block, "protected")
            return block
        else:
            raise ValueError("No free blocks available")

    def remove(self, block: KVCacheBlock) -> None:
        """Remove a specific block from whichever zone it belongs to.

        Called by ``_touch()`` when a free block (ref_cnt == 0) is re-used.

        Args:
            block: The block to remove.
        """
        zone = block.free_zone
        if zone is None:
            raise ValueError(
                f"Block {block.block_id} is not in the free queue")
        self._remove_from_zone(block, zone)

    def append(self, block: KVCacheBlock) -> None:
        """Release a block back into the free queue (probation zone tail).

        Called when a block's ref_cnt drops to 0.

        Args:
            block: The block to append.
        """
        self._append_to_zone(block, "probation")

    def append_protected(self, block: KVCacheBlock) -> None:
        """Release a previously-promoted block into the protected zone tail.

        Called when a block that was cache-hit (``_promoted == True``) has its
        ref_cnt drop back to 0.  If the protected zone is full, the oldest
        protected block is demoted to the head of the probation zone.

        Args:
            block: The block to append to protected zone.
        """
        if self._num_protected >= self._max_protected and self._max_protected > 0:
            demoted = self._protected_head
            assert demoted is not None
            self._remove_from_zone(demoted, "protected")
            self._prepend_to_zone(demoted, "probation")
        self._append_to_zone(block, "protected")

    def promote(self, block: KVCacheBlock) -> None:
        """Promote a block from probation to protected zone.

        Called when a probation-zone block is cache-hit again via ``_touch()``.
        If the protected zone is full, the oldest protected block is demoted
        to the head of the probation zone.

        Args:
            block: The block to promote (must be in probation zone).
        """
        if block.free_zone != "probation":
            # Block is already protected or not in free queue — nothing to do.
            return

        # Remove from probation.
        self._remove_from_zone(block, "probation")

        # If protected zone is full, demote the oldest protected block.
        if self._num_protected >= self._max_protected and self._max_protected > 0:
            demoted = self._protected_head
            assert demoted is not None
            self._remove_from_zone(demoted, "protected")
            # Insert demoted block at the *head* of probation (most likely
            # to be evicted next among probation blocks).
            self._prepend_to_zone(demoted, "probation")

        # Add the promoted block to the tail of protected zone.
        self._append_to_zone(block, "protected")

    def get_all_free_blocks(self) -> List[KVCacheBlock]:
        """Get all free blocks (probation first, then protected).

        Mainly used for testing.

        Returns:
            A list of free blocks in eviction order.
        """
        ret: List[KVCacheBlock] = []
        curr = self._probation_head
        while curr is not None:
            ret.append(curr)
            curr = curr.next_free_block
        curr = self._protected_head
        while curr is not None:
            ret.append(curr)
            curr = curr.next_free_block
        return ret

    # ------------------------------------------------------------------
    # Segmented LRU specific query methods (for testing / observability)
    # ------------------------------------------------------------------

    @property
    def num_probation_blocks(self) -> int:
        return self._num_probation

    @property
    def num_protected_blocks(self) -> int:
        return self._num_protected

    @property
    def max_protected_blocks(self) -> int:
        return self._max_protected

    # ------------------------------------------------------------------
    # Internal linked-list helpers
    # ------------------------------------------------------------------

    def _append_to_zone(self, block: KVCacheBlock, zone: str) -> None:
        """Append *block* to the **tail** of *zone*."""
        head, tail = self._get_zone_endpoints(zone)

        block.prev_free_block = tail
        block.next_free_block = None
        if tail is not None:
            tail.next_free_block = block
        else:
            head = block
        tail = block

        block.free_zone = zone
        self._set_zone_endpoints(zone, head, tail)
        self._incr_zone_count(zone)

    def _prepend_to_zone(self, block: KVCacheBlock, zone: str) -> None:
        """Prepend *block* to the **head** of *zone*."""
        head, tail = self._get_zone_endpoints(zone)

        block.next_free_block = head
        block.prev_free_block = None
        if head is not None:
            head.prev_free_block = block
        else:
            tail = block
        head = block

        block.free_zone = zone
        self._set_zone_endpoints(zone, head, tail)
        self._incr_zone_count(zone)

    def _remove_from_zone(self, block: KVCacheBlock, zone: str) -> None:
        """Remove *block* from *zone*."""
        head, tail = self._get_zone_endpoints(zone)

        prev_blk = block.prev_free_block
        next_blk = block.next_free_block

        if prev_blk is not None:
            prev_blk.next_free_block = next_blk
        if next_blk is not None:
            next_blk.prev_free_block = prev_blk
        if block is head:
            head = next_blk
        if block is tail:
            tail = prev_blk

        block.prev_free_block = None
        block.next_free_block = None
        block.free_zone = None

        self._set_zone_endpoints(zone, head, tail)
        self._decr_zone_count(zone)

    # --- Accessor helpers for zone head/tail ---

    def _get_zone_endpoints(
        self, zone: str
    ) -> Tuple[Optional[KVCacheBlock], Optional[KVCacheBlock]]:
        if zone == "probation":
            return self._probation_head, self._probation_tail
        else:
            return self._protected_head, self._protected_tail

    def _set_zone_endpoints(self, zone: str,
                            head: Optional[KVCacheBlock],
                            tail: Optional[KVCacheBlock]) -> None:
        if zone == "probation":
            self._probation_head = head
            self._probation_tail = tail
        else:
            self._protected_head = head
            self._protected_tail = tail

    def _incr_zone_count(self, zone: str) -> None:
        if zone == "probation":
            self._num_probation += 1
        else:
            self._num_protected += 1

    def _decr_zone_count(self, zone: str) -> None:
        if zone == "probation":
            self._num_probation -= 1
        else:
            self._num_protected -= 1


def need_extra_keys(request: Request) -> bool:
    """Check whether the blocks allocated to this request need extra hash keys.

    Args:
        request (Request): The request. 

    Returns:
        bool: Whether blocks allocated to this request need extra hash keys. 
    """

    # Multimodal requests need to include the MM hash.
    # LoRA requests need to include the LoRA ID.
    return bool(request.mm_positions) or (request.lora_request is not None)


def _gen_mm_extra_hash_keys(request: Request, start_token_idx: int,
                            end_token_idx: int,
                            start_mm_idx: int) -> Tuple[List[Any], int]:
    """Generate extra keys related to MultiModal request for block hash
    computation. For multi-modal inputs, the extra keys are
    (mm_hash, start_offset) that indicate a mm input contained in the
    block and its starting offset in the block tokens.
    
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    extra_keys: List[Any] = []

    mm_positions, mm_hashes = request.mm_positions, request.mm_hashes
    if not mm_positions:
        return extra_keys, start_mm_idx

    if mm_positions and len(mm_positions) != len(mm_hashes):
        raise ValueError(
            "The number of multi-modal positions and hashes must match. This "
            "is likely because you do not enable MM preprocessor hashing. "
            "Please set disable_mm_preprocessor_cache=False.")

    # Note that we assume mm_positions is sorted by offset.
    # We do not need to check all mm inputs if the start token index is out of
    # range. This usually happens in the late prefill phase and decoding phase.
    if mm_positions[-1]["offset"] + mm_positions[-1][
            "length"] < start_token_idx:
        return extra_keys, start_mm_idx

    # Support start_mm_idx == -1 to indicate the last mm input.
    if start_mm_idx < 0:
        assert -start_mm_idx <= len(mm_positions)
        start_mm_idx = len(mm_positions) + start_mm_idx

    curr_mm_idx = start_mm_idx
    while mm_positions and curr_mm_idx < len(mm_positions):
        assert mm_hashes[curr_mm_idx] is not None
        offset = mm_positions[curr_mm_idx]["offset"]
        length = mm_positions[curr_mm_idx]["length"]
        if end_token_idx > offset:
            if start_token_idx > offset + length:
                # This block has passed the current mm input.
                curr_mm_idx += 1
                continue

            # The block contains the current mm input.
            extra_keys.append(mm_hashes[curr_mm_idx])

            if end_token_idx >= offset + length:
                # If this block contains the end of the current mm input,
                # move to the next mm input as this block may also contain
                # the next mm input.
                curr_mm_idx += 1
            else:
                # Otherwise this block is done with mm inputs.
                break
        else:
            # This block has not reached the current mm input.
            break
    return extra_keys, curr_mm_idx


def _gen_lora_extra_hash_keys(request: Request) -> List[int]:
    """Generate extra keys related to LoRA for block hash computation.
    
    Args:
        request: The request object.
    
    Returns:
        Return LoRA id of the request if it is a LoRA request. Return empty
        list otherwise.
    """
    if not request.lora_request:
        return []
    return [request.lora_request.lora_int_id]


def generate_block_hash_extra_keys(
        request: Request, start_token_idx: int, end_token_idx: int,
        start_mm_idx: int) -> Tuple[Optional[Tuple[Any, ...]], int]:
    """Generate extra keys for the block hash. The extra keys can come from
    the multi-modal inputs and request specific metadata (e.g., LoRA ID).
    
    Args:
        request: The request object.
        start_token_idx: The start token index of the block.
        end_token_idx: The end token index of the block.
        start_mm_idx: The start multi-modal index of the block.
    
    Returns:
        A tuple of extra keys and the next multi-modal index.
    """
    mm_extra_keys: List[Any]
    mm_extra_keys, new_start_mm_idx = _gen_mm_extra_hash_keys(
        request, start_token_idx, end_token_idx, start_mm_idx)
    lora_extra_keys: List[int] = _gen_lora_extra_hash_keys(request)

    extra_keys: List[Any] = lora_extra_keys + mm_extra_keys

    if not extra_keys:
        return None, new_start_mm_idx

    return tuple(extra_keys), new_start_mm_idx


def hash_block_tokens(
        parent_block_hash: Optional[int],
        curr_block_token_ids: Sequence[int],
        extra_keys: Optional[Tuple[Any, ...]] = None) -> BlockHashType:
    """Computes a hash value corresponding to the contents of a block and
    the contents of the preceding block(s). The hash value is used for
    prefix caching. We use LRU cache for this function to avoid recomputing
    hash values for the same block contents.

    Args:
        parent_block_hash: The hash of the parent block. None
            if this is the first block.
        curr_block_token_ids: A list of token ids in the current
            block. The current block is assumed to be full.
        extra_keys: Extra keys for the block.

    Returns:
        The hash value of the block and the token ids in the block.
        The entire tuple is used as the hash key of the block.
    """
    if not parent_block_hash:
        # Note that we use 'None' as a string here instead of None because
        # as of Python 3.12, hash(None) returns a constant predictable value.
        # This could possibly make it easier to find and exploit hash
        # collisions. 'None' as a string will be hashed differently per process,
        # but consistently within the same process. This is the same as the
        # behavior of None prior to Python 3.12.
        parent_block_hash = hash('None')

    curr_block_token_ids_tuple = tuple(curr_block_token_ids)
    return BlockHashType(
        hash((parent_block_hash, curr_block_token_ids_tuple, extra_keys)),
        curr_block_token_ids_tuple, extra_keys)


def hash_request_tokens(block_size: int,
                        request: Request) -> List[BlockHashType]:
    """Computes hash values of a chain of blocks given a sequence of
    token IDs. The hash value is used for prefix caching.

    Args:
        block_size: The size of each block.
        request: The request object.

    Returns:
        The list of computed hash values.
    """
    token_ids = request.all_token_ids

    req_need_extra_keys = need_extra_keys(request)
    req_extra_keys = None
    curr_mm_idx = 0

    ret = []
    parent_block_hash_value = None
    for start in range(0, len(token_ids), block_size):
        end = start + block_size
        block_token_ids = token_ids[start:end]
        # Do not hash the block if it is not full.
        if len(block_token_ids) < block_size:
            break

        if req_need_extra_keys:
            # MM and LoRA requests need extra keys for block-hash computation.
            req_extra_keys, curr_mm_idx = generate_block_hash_extra_keys(
                request, start, end, curr_mm_idx)

        block_hash = hash_block_tokens(parent_block_hash_value,
                                       block_token_ids, req_extra_keys)
        ret.append(block_hash)
        parent_block_hash_value = block_hash.hash_value
    return ret


def check_enough_kv_cache_memory(vllm_config: VllmConfig,
                                 kv_cache_spec: KVCacheSpec,
                                 available_memory: int):
    """
    Checks whether `available_memory` is enough for the KV cache to hold at 
    least one request with the model's max_model_len.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.

    Raises:
        ValueError: If there is not enough memory available for the KV cache.
    """

    if available_memory <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")

    max_model_len = vllm_config.model_config.max_model_len
    needed_memory = 0
    for layer_spec in kv_cache_spec.values():
        needed_memory += layer_spec.bytes_for_tokens(max_model_len)

    if needed_memory > available_memory:
        raise ValueError(
            f"To serve at least one request with the models's max seq len "
            f"({max_model_len}), ({needed_memory/1024/1024/1024:.2f} GB KV "
            f"cache is needed, which is larger than the available KV cache "
            f"memory ({available_memory/1024/1024/1024:.2f} GB). Try "
            f"increasing `gpu_memory_utilization` or decreasing "
            f"`max_model_len` when initializing the engine.")


def is_kv_cache_type_uniform(kv_cache_spec: KVCacheSpec) -> bool:
    """
    Whether all layers in the given KVCacheSpec have the same type of KV cache.

    Args:
        kv_cache_spec: The KVCacheSpec of the model

    Returns:
        True if all layers have the same type, False otherwise.
    """

    layer_keys = set(layer.type_id for layer in kv_cache_spec.values())
    return len(layer_keys) == 1


def _get_kv_cache_config_uniform_type(vllm_config: VllmConfig,
                                      kv_cache_spec: KVCacheSpec,
                                      available_memory: int,
                                      num_layers: int) -> KVCacheConfig:
    """
    Generates the KV cache configuration for a model with one type of KV cache.
    Divide the available memory equally among all layers.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_spec: The kv cache spec of the model
        available_memory: Memory available for KV cache in bytes.
        num_layers: The number of layers in the model.

    Returns:
        The generated KVCacheConfig
    """

    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    assert len(page_sizes) == 1
    page_size = page_sizes.pop()

    num_blocks = int(available_memory // page_size // num_layers)
    num_blocks = max(num_blocks, 0)

    if vllm_config.cache_config.num_gpu_blocks_override is not None:
        num_gpu_blocks_override = \
            vllm_config.cache_config.num_gpu_blocks_override
        logger.info(
            "Overriding num_gpu_blocks=%d with "
            "num_gpu_blocks_override=%d", num_blocks, num_gpu_blocks_override)
        num_blocks = num_gpu_blocks_override

    logger.info("# GPU blocks: %d", num_blocks)
    max_concurrency = (num_blocks * vllm_config.cache_config.block_size /
                       vllm_config.model_config.max_model_len)
    logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                vllm_config.model_config.max_model_len, max_concurrency)

    per_layer_size = page_size * num_blocks

    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        tensors={
            layer_name: KVCacheTensor(size=per_layer_size)
            for layer_name in kv_cache_spec
        },
        groups=[[layer_name for layer_name in kv_cache_spec]],
        kv_cache_spec=kv_cache_spec)
    return kv_cache_config


def get_kv_cache_configs(vllm_config: VllmConfig,
                         kv_cache_specs: List[KVCacheSpec],
                         available_memory: int) -> List[KVCacheConfig]:
    """
    Generates the KV cache configuration for a model
    TODO: support hybrid models with more than one type of KV cache.

    Args:
        vllm_config: The global VllmConfig
        kv_cache_specs: The kv cache specs of the model
        available_memory: Memory available for KV cache in bytes.

    Returns:
        The generated KVCacheConfigs
    """
    # Use the max number of layers to conservatively determine
    # the number of blocks.
    num_layers = max(len(kv_cache_spec) for kv_cache_spec in kv_cache_specs)
    kv_cache_configs = []
    for kv_cache_spec in kv_cache_specs:
        check_enough_kv_cache_memory(vllm_config, kv_cache_spec,
                                     available_memory)
        if is_kv_cache_type_uniform(kv_cache_spec):
            # KV cache of all layers are the same, which is true for
            # most models. Allocate the same amount of memory for
            # each layer.
            kv_cache_configs.append(
                _get_kv_cache_config_uniform_type(vllm_config, kv_cache_spec,
                                                  available_memory,
                                                  num_layers))
        else:
            raise NotImplementedError
    return kv_cache_configs
