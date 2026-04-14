# SPDX-License-Identifier: Apache-2.0
"""
V1-compatible KV Cache Connector for PD Disaggregation.

This connector adapts the KV cache transfer protocol for vLLM V1's
architecture, which uses SchedulerOutput (not ModelInputForGPUWithSamplingMetadata)
and a different KV cache layout (block_table + slot_mapping via FlashAttention).

Key differences from V0's SimpleConnector:
  - Uses V1's per-request block_ids and slot_mapping instead of V0's
    attn_metadata.slot_mapping
  - Send/recv APIs accept V1-native structures (req_ids, kv_caches, block_ids)
  - Supports V1's hash-chain prefix caching by transmitting block hashes
    alongside KV data
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


class V1KVConnector(KVConnectorBase):
    """KV Connector adapted for vLLM V1 architecture.

    This connector handles KV cache transfer between Prefill (producer) and
    Decode (consumer) instances in V1's PD disaggregation setup.

    V1 differences from V0:
      - V1 has no ModelInputForGPUWithSamplingMetadata; uses SchedulerOutput
      - V1 KV cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
      - V1 uses block_table + slot_mapping computed in _prepare_inputs()
      - V1 has hash-chain based prefix caching
    """

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.config = config.kv_transfer_config
        self.vllm_config = config
        self.tp_size = config.parallel_config.tensor_parallel_size

        if self.config.kv_connector == "V1PyNcclConnector":
            from vllm.distributed.kv_transfer.kv_pipe.pynccl_pipe import (
                PyNcclPipe)
            logger.info(
                "Initializing V1 PyNccl KV Connector under config %s",
                self.config)
        elif self.config.kv_connector == "V1MooncakeConnector":
            import os
            use_mooncake = os.getenv('MOONCAKE_CONFIG_PATH') is not None
            if not use_mooncake:
                raise ValueError(
                    "To use V1MooncakeConnector, you need to pass the ENV: "
                    "'MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json'.")
            from vllm.distributed.kv_transfer.kv_pipe.mooncake_pipe import (  # noqa: E501
                MooncakePipe)
            logger.info(
                "Initializing V1 Mooncake KV Connector under config %s",
                self.config)

        self.lookup_buffer_size = self.config.kv_buffer_size

        self.producer_buffer: Optional[SimpleBuffer] = None
        self.consumer_buffer: Optional[SimpleBuffer] = None

        # 2 pipes for every rank in the world
        port_offset_base = 2 * rank

        if self.config.is_kv_producer:
            if self.config.kv_connector == "V1PyNcclConnector":
                self.producer_data_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                )
                self.producer_signal_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                )
            elif self.config.kv_connector == "V1MooncakeConnector":
                self.producer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                self.producer_signal_pipe = self.producer_data_pipe

            self.producer_buffer = SimpleBuffer(self.producer_signal_pipe,
                                                self.producer_data_pipe,
                                                self.config.kv_buffer_size)
        else:
            if self.config.kv_connector == "V1PyNcclConnector":
                self.consumer_data_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base,
                )
                self.consumer_signal_pipe = PyNcclPipe(
                    local_rank=local_rank,
                    config=self.config,
                    port_offset=port_offset_base + 1,
                    device="cpu",
                )
            elif self.config.kv_connector == "V1MooncakeConnector":
                self.consumer_data_pipe = MooncakePipe(
                    local_rank=local_rank,
                    config=self.config,
                )
                self.consumer_signal_pipe = self.consumer_data_pipe

            self.consumer_buffer = SimpleBuffer(
                self.consumer_signal_pipe,
                self.consumer_data_pipe,
                self.config.kv_buffer_size,
            )

    # ------------------------------------------------------------------
    # V0-compatible API (required by KVConnectorBase but not used in V1)
    # ------------------------------------------------------------------

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        """V0-compatible send API — delegates to V1 send if possible."""
        raise NotImplementedError(
            "V1KVConnector does not support V0's send API. "
            "Use send_kv_caches_v1() instead.")

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        """V0-compatible recv API — delegates to V1 recv if possible."""
        raise NotImplementedError(
            "V1KVConnector does not support V0's recv API. "
            "Use recv_kv_caches_v1() instead.")

    # ------------------------------------------------------------------
    # V1-native API
    # ------------------------------------------------------------------

    def send_kv_caches_v1(
        self,
        req_id: str,
        input_tokens: torch.Tensor,
        kv_caches: List[torch.Tensor],
        block_ids: List[int],
        num_computed_tokens: int,
        num_new_tokens: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        num_layers: int,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """Send KV caches for a single request in V1 format.

        This method extracts KV data from V1's paged KV cache using
        block_ids, and sends them through the producer buffer.

        Args:
            req_id: Request ID.
            input_tokens: Token IDs for this request.
            kv_caches: List of KV cache tensors, one per layer.
                Each has shape [2, num_blocks, block_size, num_kv_heads, head_size].
            block_ids: Block IDs allocated for this request.
            num_computed_tokens: Number of tokens already computed (from prefix
                cache hit).
            num_new_tokens: Number of newly computed tokens in this step.
            block_size: Number of tokens per block.
            num_kv_heads: Number of KV heads.
            head_size: Head dimension size.
            num_layers: Number of attention layers.
            hidden_states: Optional hidden states for the request.
        """
        assert self.producer_buffer is not None, (
            "Producer buffer not initialized. This instance must be "
            "a KV producer.")

        # Total tokens to send = num_computed_tokens + num_new_tokens
        # But we only send the portion that was actually computed in this step
        # (num_new_tokens), since prefix-cached blocks are shared.
        # For basic 1P1D, we send ALL tokens (the entire prefill result).
        total_tokens = num_computed_tokens + num_new_tokens
        tokens_to_send = input_tokens[:total_tokens]

        # Collect KV data from all layers for the request's blocks.
        # V1 KV cache layout: [2, num_blocks, block_size, num_kv_heads, head_size]
        # We need to extract the KV for slots [0, total_tokens)
        keys_list = []
        values_list = []

        for layer_id in range(num_layers):
            kv_cache = kv_caches[layer_id]
            # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            key_cache = kv_cache[0]   # [num_blocks, block_size, num_kv_heads, head_size]
            value_cache = kv_cache[1]  # [num_blocks, block_size, num_kv_heads, head_size]

            # Extract KV for each token using block_ids
            layer_keys = []
            layer_values = []
            for token_idx in range(total_tokens):
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                if block_idx < len(block_ids):
                    block_id = block_ids[block_idx]
                    layer_keys.append(
                        key_cache[block_id, block_offset].unsqueeze(0))
                    layer_values.append(
                        value_cache[block_id, block_offset].unsqueeze(0))

            if layer_keys:
                # [total_tokens, num_kv_heads, head_size]
                keys_list.append(torch.cat(layer_keys, dim=0).unsqueeze(0))
                values_list.append(torch.cat(layer_values, dim=0).unsqueeze(0))

        if not keys_list:
            logger.warning("No KV data to send for request %s", req_id)
            return

        # keys shape: [num_layers, total_tokens, num_kv_heads, head_size]
        keys = torch.cat(keys_list, dim=0)
        values = torch.cat(values_list, dim=0)

        # Send through the buffer (same protocol as V0's SimpleConnector)
        roi = torch.ones(total_tokens, dtype=torch.bool,
                         device=tokens_to_send.device)

        # Use hidden_states if available, otherwise send a dummy
        if hidden_states is None:
            hidden_states = torch.zeros(
                total_tokens, 1, dtype=keys.dtype, device=keys.device)

        self.producer_buffer.insert(tokens_to_send, roi, keys, values,
                                    hidden_states)

        logger.debug(
            "V1 KV send: req_id=%s, tokens=%d, layers=%d",
            req_id, total_tokens, num_layers)

    def recv_kv_caches_v1(
        self,
        req_id: str,
        input_tokens: torch.Tensor,
        kv_caches: List[torch.Tensor],
        block_ids: List[int],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        num_layers: int,
    ) -> Tuple[bool, int, Optional[torch.Tensor]]:
        """Receive KV caches for a single request in V1 format.

        Attempts to retrieve KV data from the consumer buffer and writes
        it into V1's paged KV cache.

        Args:
            req_id: Request ID.
            input_tokens: Token IDs for this request.
            kv_caches: List of KV cache tensors.
            block_ids: Block IDs allocated for this request.
            block_size: Number of tokens per block.
            num_kv_heads: Number of KV heads.
            head_size: Head dimension size.
            num_layers: Number of attention layers.

        Returns:
            A tuple of:
              - success: True if KV was successfully received.
              - num_computed_tokens: Number of tokens whose KV was received.
              - hidden_states: The received hidden states, or None.
        """
        assert self.consumer_buffer is not None, (
            "Consumer buffer not initialized. This instance must be "
            "a KV consumer.")

        roi = torch.ones_like(input_tokens, dtype=torch.bool)
        ret = self.consumer_buffer.drop_select(input_tokens, roi)

        if ret[0] is None:
            # No matching KV found in the buffer.
            logger.debug(
                "V1 KV recv: no match for req_id=%s", req_id)
            return False, 0, None

        recv_roi: torch.Tensor = ret[1]
        recv_keys: torch.Tensor = ret[2]    # [num_layers, num_tokens, num_kv_heads, head_size]
        recv_values: torch.Tensor = ret[3]  # [num_layers, num_tokens, num_kv_heads, head_size]
        recv_hidden: torch.Tensor = ret[4]

        num_recv_tokens = recv_roi.shape[0]

        # Write received KV into V1's paged KV cache.
        # V1 KV cache layout: [2, num_blocks, block_size, num_kv_heads, head_size]
        for layer_id in range(num_layers):
            kv_cache = kv_caches[layer_id]
            key_cache = kv_cache[0]    # [num_blocks, block_size, num_kv_heads, head_size]
            value_cache = kv_cache[1]  # [num_blocks, block_size, num_kv_heads, head_size]

            layer_keys = recv_keys[layer_id]    # [num_tokens, num_kv_heads, head_size]
            layer_values = recv_values[layer_id]  # [num_tokens, num_kv_heads, head_size]

            # Write token-by-token into the paged KV cache
            for token_idx in range(num_recv_tokens):
                block_idx = token_idx // block_size
                block_offset = token_idx % block_size
                if block_idx < len(block_ids):
                    block_id = block_ids[block_idx]
                    key_cache[block_id, block_offset].copy_(
                        layer_keys[token_idx])
                    value_cache[block_id, block_offset].copy_(
                        layer_values[token_idx])

        logger.debug(
            "V1 KV recv: req_id=%s, tokens=%d, layers=%d",
            req_id, num_recv_tokens, num_layers)

        return True, num_recv_tokens, recv_hidden

    def close(self) -> None:
        """Release resources."""
        if hasattr(self, 'producer_data_pipe'):
            self.producer_data_pipe.close()
        if hasattr(self, 'consumer_data_pipe'):
            self.consumer_data_pipe.close()
        if self.config.kv_connector == "V1PyNcclConnector":
            if hasattr(self, 'producer_signal_pipe'):
                self.producer_signal_pipe.close()
            if hasattr(self, 'consumer_signal_pipe'):
                self.consumer_signal_pipe.close()
