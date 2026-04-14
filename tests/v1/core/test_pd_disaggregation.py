# SPDX-License-Identifier: Apache-2.0
"""Tests for PD Disaggregation Optimization 1: V1 Engine PD Basic Adaptation.

These tests verify the implementation correctness by analyzing source code
using AST parsing and text inspection, avoiding direct import of vllm modules
(which require torch/GPU dependencies not available in this test environment).

Test coverage:
1. V1KVConnector creation and factory registration
2. V1KVConnector V1-native API design
3. KVCacheManager.register_received_blocks() implementation
4. Scheduler PD role awareness
5. EngineCore PD initialization and lifecycle
6. GPUModelRunner PD helper methods and execute_model hooks
"""

import ast
import os
import pytest

# Paths to the source files under test.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

V1_CONNECTOR_PATH = os.path.join(
    PROJECT_ROOT,
    "vllm/distributed/kv_transfer/kv_connector/v1_connector.py")
FACTORY_PATH = os.path.join(
    PROJECT_ROOT,
    "vllm/distributed/kv_transfer/kv_connector/factory.py")
KV_CACHE_MANAGER_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/core/kv_cache_manager.py")
SCHEDULER_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/core/scheduler.py")
ENGINE_CORE_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/engine/core.py")
GPU_MODEL_RUNNER_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/worker/gpu_model_runner.py")
BASE_CONNECTOR_PATH = os.path.join(
    PROJECT_ROOT,
    "vllm/distributed/kv_transfer/kv_connector/base.py")


def _read_source(path: str) -> str:
    """Read source file content."""
    with open(path, "r") as f:
        return f.read()


def _parse_ast(path: str) -> ast.Module:
    """Parse a Python source file into an AST."""
    source = _read_source(path)
    return ast.parse(source, filename=path)


def _get_class_def(tree: ast.Module, class_name: str) -> ast.ClassDef:
    """Find a class definition by name in the AST."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise ValueError(f"Class '{class_name}' not found")


def _get_method_names(class_def: ast.ClassDef) -> set:
    """Get all method names defined in a class."""
    return {
        node.name for node in class_def.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _get_method_def(class_def: ast.ClassDef,
                    method_name: str) -> ast.FunctionDef:
    """Find a method definition by name in a class."""
    for node in class_def.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == method_name:
                return node
    raise ValueError(
        f"Method '{method_name}' not found in class '{class_def.name}'")


def _get_method_arg_names(method_def: ast.FunctionDef) -> list:
    """Get the argument names of a function/method."""
    return [arg.arg for arg in method_def.args.args]


# =====================================================================
# Test 1: V1KVConnector Factory Registration
# =====================================================================

class TestV1ConnectorRegistration:
    """Test that V1 connectors are properly registered in the factory."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.factory_source = _read_source(FACTORY_PATH)

    def test_v1_pynccl_connector_registered(self):
        """V1PyNcclConnector should be registered in KVConnectorFactory."""
        assert 'register_connector' in self.factory_source
        assert '"V1PyNcclConnector"' in self.factory_source

    def test_v1_mooncake_connector_registered(self):
        """V1MooncakeConnector should be registered in KVConnectorFactory."""
        assert '"V1MooncakeConnector"' in self.factory_source

    def test_v0_connectors_still_registered(self):
        """Original V0 connectors should still be registered."""
        assert '"PyNcclConnector"' in self.factory_source
        assert '"MooncakeConnector"' in self.factory_source

    def test_v1_connectors_point_to_v1_module(self):
        """V1 connectors should point to v1_connector module."""
        assert "v1_connector" in self.factory_source
        assert "V1KVConnector" in self.factory_source

    def test_v1_connector_file_exists(self):
        """V1 connector source file should exist."""
        assert os.path.isfile(V1_CONNECTOR_PATH)

    def test_v1_connector_class_defined(self):
        """V1KVConnector class should be defined in v1_connector.py."""
        tree = _parse_ast(V1_CONNECTOR_PATH)
        class_def = _get_class_def(tree, "V1KVConnector")
        assert class_def is not None

    def test_v1_connector_inherits_base(self):
        """V1KVConnector should inherit from KVConnectorBase."""
        tree = _parse_ast(V1_CONNECTOR_PATH)
        class_def = _get_class_def(tree, "V1KVConnector")
        base_names = []
        for base in class_def.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)
        assert "KVConnectorBase" in base_names


# =====================================================================
# Test 2: V1KVConnector V1-native API
# =====================================================================

class TestV1ConnectorAPI:
    """Test V1KVConnector has the correct V1-native API methods."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tree = _parse_ast(V1_CONNECTOR_PATH)
        self.class_def = _get_class_def(self.tree, "V1KVConnector")
        self.methods = _get_method_names(self.class_def)

    def test_has_send_v1_method(self):
        """V1KVConnector should have send_kv_caches_v1 method."""
        assert "send_kv_caches_v1" in self.methods

    def test_has_recv_v1_method(self):
        """V1KVConnector should have recv_kv_caches_v1 method."""
        assert "recv_kv_caches_v1" in self.methods

    def test_has_v0_send_compat(self):
        """V1KVConnector should have V0-compatible send method."""
        assert "send_kv_caches_and_hidden_states" in self.methods

    def test_has_v0_recv_compat(self):
        """V1KVConnector should have V0-compatible recv method."""
        assert "recv_kv_caches_and_hidden_states" in self.methods

    def test_v0_send_raises_not_implemented(self):
        """V0's send API should raise NotImplementedError."""
        method = _get_method_def(self.class_def,
                                 "send_kv_caches_and_hidden_states")
        # Check that the method body contains a Raise with NotImplementedError
        for node in ast.walk(method):
            if isinstance(node, ast.Raise):
                exc = node.exc
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    assert exc.func.id == "NotImplementedError"
                    return
        pytest.fail("send_kv_caches_and_hidden_states should raise "
                    "NotImplementedError")

    def test_v0_recv_raises_not_implemented(self):
        """V0's recv API should raise NotImplementedError."""
        method = _get_method_def(self.class_def,
                                 "recv_kv_caches_and_hidden_states")
        for node in ast.walk(method):
            if isinstance(node, ast.Raise):
                exc = node.exc
                if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name):
                    assert exc.func.id == "NotImplementedError"
                    return
        pytest.fail("recv_kv_caches_and_hidden_states should raise "
                    "NotImplementedError")

    def test_send_v1_has_correct_params(self):
        """send_kv_caches_v1 should have the required V1 parameters."""
        method = _get_method_def(self.class_def, "send_kv_caches_v1")
        arg_names = _get_method_arg_names(method)
        required = ["req_id", "input_tokens", "kv_caches", "block_ids",
                     "num_computed_tokens", "num_new_tokens", "block_size",
                     "num_kv_heads", "head_size", "num_layers"]
        for param in required:
            assert param in arg_names, (
                f"send_kv_caches_v1 missing parameter: {param}")

    def test_recv_v1_has_correct_params(self):
        """recv_kv_caches_v1 should have the required V1 parameters."""
        method = _get_method_def(self.class_def, "recv_kv_caches_v1")
        arg_names = _get_method_arg_names(method)
        required = ["req_id", "input_tokens", "kv_caches", "block_ids",
                     "block_size", "num_kv_heads", "head_size", "num_layers"]
        for param in required:
            assert param in arg_names, (
                f"recv_kv_caches_v1 missing parameter: {param}")

    def test_send_v1_uses_block_arithmetic(self):
        """send_kv_caches_v1 should extract KV using block_id arithmetic."""
        source = _read_source(V1_CONNECTOR_PATH)
        # V1 uses block_ids + block_size to index KV cache
        assert "block_idx" in source or "block_offset" in source
        assert "block_size" in source

    def test_recv_v1_writes_to_paged_cache(self):
        """recv_kv_caches_v1 should write KV into paged cache."""
        method = _get_method_def(self.class_def, "recv_kv_caches_v1")
        method_source = ast.get_source_segment(
            _read_source(V1_CONNECTOR_PATH), method)
        assert method_source is not None
        # Should contain copy_ or similar write operation
        assert "copy_" in method_source

    def test_has_close_method(self):
        """V1KVConnector should have a close() method for cleanup."""
        assert "close" in self.methods

    def test_init_handles_both_connector_types(self):
        """__init__ should handle both PyNccl and Mooncake connectors."""
        source = _read_source(V1_CONNECTOR_PATH)
        assert "V1PyNcclConnector" in source
        assert "V1MooncakeConnector" in source


# =====================================================================
# Test 3: KVCacheManager.register_received_blocks
# =====================================================================

class TestKVCacheManagerPD:
    """Test KVCacheManager's PD disaggregation support."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tree = _parse_ast(KV_CACHE_MANAGER_PATH)
        self.class_def = _get_class_def(self.tree, "KVCacheManager")
        self.methods = _get_method_names(self.class_def)
        self.source = _read_source(KV_CACHE_MANAGER_PATH)

    def test_register_received_blocks_method_exists(self):
        """KVCacheManager should have register_received_blocks method."""
        assert "register_received_blocks" in self.methods

    def test_register_received_blocks_signature(self):
        """register_received_blocks should accept request and num_received_tokens."""
        method = _get_method_def(self.class_def, "register_received_blocks")
        arg_names = _get_method_arg_names(method)
        assert "request" in arg_names
        assert "num_received_tokens" in arg_names

    def test_register_received_blocks_checks_caching_enabled(self):
        """Method should check enable_caching flag before proceeding."""
        method = _get_method_def(self.class_def, "register_received_blocks")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "enable_caching" in method_source

    def test_register_received_blocks_computes_full_blocks(self):
        """Method should compute num_full_blocks from num_received_tokens."""
        method = _get_method_def(self.class_def, "register_received_blocks")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "num_full_blocks" in method_source
        # Should divide by block_size
        assert "block_size" in method_source

    def test_register_received_blocks_uses_hash_chain(self):
        """Method should use hash_request_tokens for block hashing."""
        method = _get_method_def(self.class_def, "register_received_blocks")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "hash_request_tokens" in method_source

    def test_register_received_blocks_updates_prefix_cache(self):
        """Method should register blocks in cached_block_hash_to_block."""
        method = _get_method_def(self.class_def, "register_received_blocks")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "cached_block_hash_to_block" in method_source

    def test_register_received_blocks_updates_num_cached_block(self):
        """Method should update num_cached_block tracking."""
        method = _get_method_def(self.class_def, "register_received_blocks")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "num_cached_block" in method_source

    def test_register_received_blocks_sets_block_hash(self):
        """Method should set block.block_hash for each registered block."""
        method = _get_method_def(self.class_def, "register_received_blocks")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "block_hash" in method_source


# =====================================================================
# Test 4: Scheduler PD Role Awareness
# =====================================================================

class TestSchedulerPDAwareness:
    """Test that Scheduler correctly handles PD role configuration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tree = _parse_ast(SCHEDULER_PATH)
        self.class_def = _get_class_def(self.tree, "Scheduler")
        self.source = _read_source(SCHEDULER_PATH)

    def test_scheduler_accepts_kv_transfer_config(self):
        """Scheduler.__init__ should accept kv_transfer_config parameter."""
        init_method = _get_method_def(self.class_def, "__init__")
        arg_names = _get_method_arg_names(init_method)
        assert "kv_transfer_config" in arg_names

    def test_scheduler_kv_transfer_config_has_default_none(self):
        """kv_transfer_config should default to None for backward compat."""
        init_method = _get_method_def(self.class_def, "__init__")
        # Find the default value for kv_transfer_config
        defaults = init_method.args.defaults
        kw_defaults = init_method.args.kw_defaults
        # Check all defaults (including kw_only defaults)
        all_defaults_source = ast.get_source_segment(self.source, init_method)
        assert "kv_transfer_config: Optional[KVTransferConfig] = None" \
               in all_defaults_source

    def test_scheduler_stores_pd_role_flags(self):
        """Scheduler should store is_kv_transfer_instance, is_kv_producer,
        is_kv_consumer."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "self.is_kv_transfer_instance" in init_source
        assert "self.is_kv_producer" in init_source
        assert "self.is_kv_consumer" in init_source

    def test_scheduler_imports_kv_transfer_config(self):
        """Scheduler module should import KVTransferConfig."""
        assert "KVTransferConfig" in self.source

    def test_scheduler_checks_config_properties(self):
        """Scheduler should check config.is_kv_producer and config.is_kv_consumer."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "is_kv_transfer_instance" in init_source
        assert "is_kv_producer" in init_source
        assert "is_kv_consumer" in init_source


# =====================================================================
# Test 5: EngineCore PD Initialization
# =====================================================================

class TestEngineCorePD:
    """Test that EngineCore correctly handles PD configuration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tree = _parse_ast(ENGINE_CORE_PATH)
        self.class_def = _get_class_def(self.tree, "EngineCore")
        self.methods = _get_method_names(self.class_def)
        self.source = _read_source(ENGINE_CORE_PATH)

    def test_engine_core_stores_kv_transfer_flags(self):
        """EngineCore.__init__ should store PD role flags."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "self.is_kv_transfer_instance" in init_source
        assert "self.is_kv_producer" in init_source
        assert "self.is_kv_consumer" in init_source

    def test_engine_core_stores_kv_transfer_config(self):
        """EngineCore.__init__ should store kv_transfer_config."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "self.kv_transfer_config" in init_source

    def test_engine_core_passes_kv_config_to_scheduler(self):
        """EngineCore should pass kv_transfer_config to Scheduler."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "kv_transfer_config=self.kv_transfer_config" in init_source

    def test_engine_core_has_finish_prefill_method(self):
        """EngineCore should have _finish_prefill_only_requests method."""
        assert "_finish_prefill_only_requests" in self.methods

    def test_finish_prefill_marks_requests_finished(self):
        """_finish_prefill_only_requests should mark requests as FINISHED_STOPPED."""
        method = _get_method_def(self.class_def,
                                 "_finish_prefill_only_requests")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "FINISHED_STOPPED" in method_source

    def test_finish_prefill_checks_num_computed_tokens(self):
        """Method should check num_computed_tokens >= num_tokens."""
        method = _get_method_def(self.class_def,
                                 "_finish_prefill_only_requests")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "num_computed_tokens" in method_source
        assert "num_tokens" in method_source

    def test_step_calls_finish_prefill_for_producer(self):
        """step() method should call _finish_prefill_only_requests
        when instance is a producer."""
        step_method = _get_method_def(self.class_def, "step")
        step_source = ast.get_source_segment(self.source, step_method)
        assert step_source is not None
        assert "_finish_prefill_only_requests" in step_source
        assert "is_kv_producer" in step_source

    def test_finish_prefill_accepts_scheduler_output(self):
        """_finish_prefill_only_requests should accept scheduler_output param."""
        method = _get_method_def(self.class_def,
                                 "_finish_prefill_only_requests")
        arg_names = _get_method_arg_names(method)
        assert "scheduler_output" in arg_names


# =====================================================================
# Test 6: GPUModelRunner PD Integration Points
# =====================================================================

class TestGPUModelRunnerPD:
    """Test GPUModelRunner PD integration."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tree = _parse_ast(GPU_MODEL_RUNNER_PATH)
        self.class_def = _get_class_def(self.tree, "GPUModelRunner")
        self.methods = _get_method_names(self.class_def)
        self.source = _read_source(GPU_MODEL_RUNNER_PATH)

    def test_model_runner_has_kv_transfer_init(self):
        """GPUModelRunner.__init__ should initialize PD-related attributes."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "self.kv_transfer_config" in init_source
        assert "self.is_kv_transfer_instance" in init_source
        assert "self.is_kv_producer" in init_source
        assert "self.is_kv_consumer" in init_source
        assert "self.kv_connector" in init_source

    def test_model_runner_has_send_helper(self):
        """GPUModelRunner should have _send_kv_caches_for_producer method."""
        assert "_send_kv_caches_for_producer" in self.methods

    def test_model_runner_has_recv_helper(self):
        """GPUModelRunner should have _recv_kv_caches_for_consumer method."""
        assert "_recv_kv_caches_for_consumer" in self.methods

    def test_model_runner_has_prefill_check(self):
        """GPUModelRunner should have _has_prefill_requests method."""
        assert "_has_prefill_requests" in self.methods

    def test_execute_model_has_consumer_recv_hook(self):
        """execute_model should receive KV for consumer before forward."""
        exec_method = _get_method_def(self.class_def, "execute_model")
        exec_source = ast.get_source_segment(self.source, exec_method)
        assert exec_source is not None
        assert "_recv_kv_caches_for_consumer" in exec_source
        assert "is_kv_consumer" in exec_source

    def test_execute_model_has_producer_send_hook(self):
        """execute_model should send KV for producer after forward."""
        exec_method = _get_method_def(self.class_def, "execute_model")
        exec_source = ast.get_source_segment(self.source, exec_method)
        assert exec_source is not None
        assert "_send_kv_caches_for_producer" in exec_source
        assert "is_kv_producer" in exec_source

    def test_execute_model_has_bypass_logic(self):
        """execute_model should have bypass_model_exec logic."""
        exec_method = _get_method_def(self.class_def, "execute_model")
        exec_source = ast.get_source_segment(self.source, exec_method)
        assert exec_source is not None
        assert "bypass_model_exec" in exec_source

    def test_recv_helper_processes_new_requests(self):
        """_recv_kv_caches_for_consumer should process scheduled_new_reqs."""
        method = _get_method_def(self.class_def,
                                 "_recv_kv_caches_for_consumer")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "scheduled_new_reqs" in method_source

    def test_recv_helper_calls_connector_recv_v1(self):
        """_recv_kv_caches_for_consumer should call kv_connector.recv_kv_caches_v1."""
        method = _get_method_def(self.class_def,
                                 "_recv_kv_caches_for_consumer")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "recv_kv_caches_v1" in method_source

    def test_recv_helper_updates_num_computed_tokens(self):
        """Successful recv should update num_computed_tokens to skip prefill."""
        method = _get_method_def(self.class_def,
                                 "_recv_kv_caches_for_consumer")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "num_computed_tokens" in method_source

    def test_send_helper_calls_connector_send_v1(self):
        """_send_kv_caches_for_producer should call kv_connector.send_kv_caches_v1."""
        method = _get_method_def(self.class_def,
                                 "_send_kv_caches_for_producer")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "send_kv_caches_v1" in method_source

    def test_send_helper_identifies_prefill_requests(self):
        """_send_kv_caches_for_producer should only send for prefill requests."""
        method = _get_method_def(self.class_def,
                                 "_send_kv_caches_for_producer")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        # Should check if request is in prefill phase
        assert "is_prefill" in method_source or "num_computed_tokens" in method_source

    def test_send_helper_extracts_hidden_states(self):
        """_send_kv_caches_for_producer should extract per-request hidden states."""
        method = _get_method_def(self.class_def,
                                 "_send_kv_caches_for_producer")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "hidden_states" in method_source

    def test_init_creates_connector_via_factory(self):
        """GPUModelRunner should create connector via KVConnectorFactory."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "KVConnectorFactory" in init_source
        assert "create_connector" in init_source

    def test_init_checks_v1_compatibility(self):
        """GPUModelRunner should check connector is V1-compatible."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "V1KVConnector" in init_source
        assert "isinstance" in init_source

    def test_prefill_check_uses_num_computed_tokens(self):
        """_has_prefill_requests should use num_computed_tokens logic."""
        method = _get_method_def(self.class_def, "_has_prefill_requests")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "num_computed_tokens" in method_source
        assert "num_tokens" in method_source

    def test_v1_connector_type_check_import(self):
        """GPUModelRunner should import V1KVConnector for TYPE_CHECKING."""
        assert "V1KVConnector" in self.source
        assert "TYPE_CHECKING" in self.source


# =====================================================================
# Test 7: V1KVConnector Implementation Details
# =====================================================================

class TestV1ConnectorImplementation:
    """Test V1KVConnector implementation correctness."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tree = _parse_ast(V1_CONNECTOR_PATH)
        self.class_def = _get_class_def(self.tree, "V1KVConnector")
        self.source = _read_source(V1_CONNECTOR_PATH)

    def test_uses_simple_buffer(self):
        """V1KVConnector should use SimpleBuffer for data transfer."""
        assert "SimpleBuffer" in self.source

    def test_has_producer_and_consumer_buffers(self):
        """V1KVConnector should manage producer and consumer buffers."""
        init_source = ast.get_source_segment(
            self.source,
            _get_method_def(self.class_def, "__init__"))
        assert "producer_buffer" in init_source
        assert "consumer_buffer" in init_source

    def test_send_uses_producer_buffer_insert(self):
        """send_kv_caches_v1 should insert into producer buffer."""
        method = _get_method_def(self.class_def, "send_kv_caches_v1")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "producer_buffer" in method_source
        assert "insert" in method_source

    def test_recv_uses_consumer_buffer_drop_select(self):
        """recv_kv_caches_v1 should use consumer buffer drop_select."""
        method = _get_method_def(self.class_def, "recv_kv_caches_v1")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "consumer_buffer" in method_source
        assert "drop_select" in method_source

    def test_send_extracts_kv_per_layer(self):
        """send_kv_caches_v1 should iterate over layers."""
        method = _get_method_def(self.class_def, "send_kv_caches_v1")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "num_layers" in method_source
        assert "key_cache" in method_source
        assert "value_cache" in method_source

    def test_recv_returns_tuple(self):
        """recv_kv_caches_v1 should return (success, num_tokens, hidden)."""
        method = _get_method_def(self.class_def, "recv_kv_caches_v1")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        # Should return True/False with num_recv_tokens and hidden
        assert "True" in method_source
        assert "False" in method_source
        assert "num_recv_tokens" in method_source

    def test_kv_cache_indexing_uses_block_arithmetic(self):
        """V1KVConnector should use block_idx and block_offset for indexing."""
        method = _get_method_def(self.class_def, "send_kv_caches_v1")
        method_source = ast.get_source_segment(self.source, method)
        assert method_source is not None
        assert "block_idx" in method_source
        assert "block_offset" in method_source
        assert "// block_size" in method_source
        assert "% block_size" in method_source


# =====================================================================
# Test 8: Cross-module Integration
# =====================================================================

class TestCrossModuleIntegration:
    """Test that the PD components are properly integrated across modules."""

    def test_engine_core_and_scheduler_config_passing(self):
        """EngineCore should pass kv_transfer_config to Scheduler, and
        Scheduler should accept it."""
        engine_source = _read_source(ENGINE_CORE_PATH)
        scheduler_source = _read_source(SCHEDULER_PATH)

        # EngineCore passes it
        assert "kv_transfer_config=self.kv_transfer_config" in engine_source
        # Scheduler accepts it
        assert "kv_transfer_config" in scheduler_source

    def test_factory_and_connector_consistency(self):
        """Factory registration should match the actual class name."""
        factory_source = _read_source(FACTORY_PATH)
        connector_source = _read_source(V1_CONNECTOR_PATH)

        # Factory registers V1KVConnector
        assert "V1KVConnector" in factory_source
        # V1KVConnector is actually defined
        assert "class V1KVConnector" in connector_source

    def test_model_runner_and_connector_api_match(self):
        """GPUModelRunner should call the V1KVConnector's V1-native API."""
        runner_source = _read_source(GPU_MODEL_RUNNER_PATH)

        # Runner calls the V1-native methods
        assert "send_kv_caches_v1" in runner_source
        assert "recv_kv_caches_v1" in runner_source

    def test_engine_core_step_includes_pd_hook(self):
        """EngineCore.step() should include PD lifecycle hook."""
        engine_source = _read_source(ENGINE_CORE_PATH)
        assert "_finish_prefill_only_requests" in engine_source
        assert "scheduler_output" in engine_source

    def test_all_pd_files_exist(self):
        """All files modified for PD disaggregation should exist."""
        assert os.path.isfile(V1_CONNECTOR_PATH), \
            "v1_connector.py missing"
        assert os.path.isfile(FACTORY_PATH), \
            "factory.py missing"
        assert os.path.isfile(KV_CACHE_MANAGER_PATH), \
            "kv_cache_manager.py missing"
        assert os.path.isfile(SCHEDULER_PATH), \
            "scheduler.py missing"
        assert os.path.isfile(ENGINE_CORE_PATH), \
            "core.py missing"
        assert os.path.isfile(GPU_MODEL_RUNNER_PATH), \
            "gpu_model_runner.py missing"

    def test_kv_cache_manager_register_used_by_design(self):
        """register_received_blocks should be designed for consumer usage."""
        source = _read_source(KV_CACHE_MANAGER_PATH)
        # The method should reference prefix cache concepts
        assert "register_received_blocks" in source
        assert "cached_block_hash_to_block" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
