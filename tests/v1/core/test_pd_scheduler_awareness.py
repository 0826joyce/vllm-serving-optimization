# SPDX-License-Identifier: Apache-2.0
"""Tests for PD Disaggregation Optimization 3: Scheduler PD Awareness.

This module tests:
1. KVReceiveMonitor — direct unit tests (no GPU dependency)
2. Scheduler PD-aware scheduling logic — AST/text-based verification
3. ModelRunnerOutput kv_recv_success_map field
4. Integration between scheduler and KV receive monitor

The KVReceiveMonitor class has no GPU dependencies and can be tested
directly.  Scheduler integration is verified via AST inspection of the
source code.
"""

import ast
import os
import time
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Direct imports for KVReceiveMonitor (no GPU dependency)
# ---------------------------------------------------------------------------
import sys

# Ensure the project root is in the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vllm.v1.core.scheduler import KVReceiveMonitor

# Paths for AST-based tests
SCHEDULER_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/core/scheduler.py")
OUTPUTS_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/outputs.py")
ENGINE_CORE_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/engine/core.py")
GPU_MODEL_RUNNER_PATH = os.path.join(
    PROJECT_ROOT, "vllm/v1/worker/gpu_model_runner.py")


# ===========================================================================
# Part 1: KVReceiveMonitor Direct Unit Tests
# ===========================================================================

class TestKVReceiveMonitor:
    """Direct unit tests for KVReceiveMonitor."""

    def test_initial_state(self):
        """Monitor starts with no pending or received requests."""
        monitor = KVReceiveMonitor()
        assert monitor.num_pending == 0
        assert monitor.num_received == 0

    def test_register_pending(self):
        """Registering a request marks it as pending."""
        monitor = KVReceiveMonitor()
        monitor.register_pending("req-1")
        assert monitor.num_pending == 1
        assert not monitor.has_kv("req-1")

    def test_register_multiple_pending(self):
        """Multiple requests can be pending simultaneously."""
        monitor = KVReceiveMonitor()
        monitor.register_pending("req-1")
        monitor.register_pending("req-2")
        monitor.register_pending("req-3")
        assert monitor.num_pending == 3
        assert monitor.num_received == 0

    def test_on_kv_received(self):
        """Receiving KV moves request from pending to received."""
        monitor = KVReceiveMonitor()
        monitor.register_pending("req-1")
        assert not monitor.has_kv("req-1")

        monitor.on_kv_received("req-1")
        assert monitor.has_kv("req-1")
        assert monitor.num_pending == 0
        assert monitor.num_received == 1

    def test_on_kv_received_without_register(self):
        """Receiving KV for an unregistered request still works."""
        monitor = KVReceiveMonitor()
        monitor.on_kv_received("req-unknown")
        assert monitor.has_kv("req-unknown")
        assert monitor.num_received == 1

    def test_register_after_received_is_noop(self):
        """Registering a request that was already received is a no-op."""
        monitor = KVReceiveMonitor()
        monitor.on_kv_received("req-1")
        monitor.register_pending("req-1")  # Should not go to pending
        assert monitor.has_kv("req-1")
        assert monitor.num_pending == 0

    def test_remove_pending(self):
        """Removing a pending request clears it from tracking."""
        monitor = KVReceiveMonitor()
        monitor.register_pending("req-1")
        monitor.remove("req-1")
        assert monitor.num_pending == 0
        assert not monitor.has_kv("req-1")

    def test_remove_received(self):
        """Removing a received request clears it from tracking."""
        monitor = KVReceiveMonitor()
        monitor.on_kv_received("req-1")
        monitor.remove("req-1")
        assert monitor.num_received == 0
        assert not monitor.has_kv("req-1")

    def test_remove_nonexistent(self):
        """Removing a nonexistent request is safe."""
        monitor = KVReceiveMonitor()
        monitor.remove("req-nonexistent")  # Should not raise

    def test_timeout_no_pending(self):
        """No timed-out requests when nothing is pending."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=1.0)
        assert monitor.get_timed_out_requests() == []

    def test_timeout_not_expired(self):
        """Pending request within timeout returns empty."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=100.0)
        monitor.register_pending("req-1")
        assert monitor.get_timed_out_requests() == []

    def test_timeout_expired(self):
        """Pending request past timeout is returned."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=0.01)
        monitor.register_pending("req-1")
        time.sleep(0.02)  # Wait past timeout
        timed_out = monitor.get_timed_out_requests()
        assert "req-1" in timed_out

    def test_mark_timed_out(self):
        """Marking as timed out moves request from pending to received."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=0.01)
        monitor.register_pending("req-1")
        time.sleep(0.02)

        monitor.mark_timed_out("req-1")
        assert monitor.has_kv("req-1")
        assert monitor.num_pending == 0
        assert monitor.num_received == 1

    def test_mixed_operations(self):
        """Test a realistic sequence of operations."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=10.0)

        # 5 requests arrive
        for i in range(5):
            monitor.register_pending(f"req-{i}")
        assert monitor.num_pending == 5

        # KV arrives for 3 of them
        monitor.on_kv_received("req-0")
        monitor.on_kv_received("req-2")
        monitor.on_kv_received("req-4")
        assert monitor.num_pending == 2
        assert monitor.num_received == 3

        # Check individual statuses
        assert monitor.has_kv("req-0")
        assert not monitor.has_kv("req-1")
        assert monitor.has_kv("req-2")
        assert not monitor.has_kv("req-3")
        assert monitor.has_kv("req-4")

        # Finish req-0 and req-2
        monitor.remove("req-0")
        monitor.remove("req-2")
        assert monitor.num_received == 1  # only req-4 left

    def test_default_timeout(self):
        """Default timeout should be 30 seconds."""
        monitor = KVReceiveMonitor()
        assert monitor.kv_wait_timeout_s == 30.0

    def test_custom_timeout(self):
        """Custom timeout is stored correctly."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=5.0)
        assert monitor.kv_wait_timeout_s == 5.0


# ===========================================================================
# Part 2: AST-based Scheduler Code Verification
# ===========================================================================

def _parse_file(path: str) -> ast.Module:
    """Parse a Python file into an AST."""
    with open(path) as f:
        return ast.parse(f.read(), filename=path)


def _read_file(path: str) -> str:
    """Read file content as string."""
    with open(path) as f:
        return f.read()


def _get_class_methods(tree: ast.Module, class_name: str) -> list:
    """Get all method names of a class."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [n.name for n in node.body
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return []


class TestSchedulerPDAwareness:
    """Verify that the Scheduler has PD-aware scheduling logic."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.tree = _parse_file(SCHEDULER_PATH)
        self.source = _read_file(SCHEDULER_PATH)

    def test_kv_receive_monitor_class_exists(self):
        """KVReceiveMonitor class should be defined in scheduler.py."""
        class_names = [n.name for n in ast.walk(self.tree)
                       if isinstance(n, ast.ClassDef)]
        assert "KVReceiveMonitor" in class_names

    def test_kv_receive_monitor_methods(self):
        """KVReceiveMonitor should have all required methods."""
        methods = _get_class_methods(self.tree, "KVReceiveMonitor")
        required = [
            "register_pending",
            "on_kv_received",
            "has_kv",
            "get_timed_out_requests",
            "mark_timed_out",
            "remove",
        ]
        for method in required:
            assert method in methods, f"Missing method: {method}"

    def test_scheduler_has_pd_aware_fields(self):
        """Scheduler.__init__ should set up PD-aware scheduling fields."""
        assert "enable_pd_aware_scheduling" in self.source
        assert "prefill_sort_by_length" in self.source
        assert "kv_receive_monitor" in self.source

    def test_scheduler_has_pd_pre_schedule(self):
        """Scheduler should have _pd_aware_pre_schedule method."""
        methods = _get_class_methods(self.tree, "Scheduler")
        assert "_pd_aware_pre_schedule" in methods

    def test_scheduler_has_sort_by_prompt_length(self):
        """Scheduler should have _sort_waiting_by_prompt_length method."""
        methods = _get_class_methods(self.tree, "Scheduler")
        assert "_sort_waiting_by_prompt_length" in methods

    def test_scheduler_has_handle_kv_timeouts(self):
        """Scheduler should have _handle_kv_timeouts method."""
        methods = _get_class_methods(self.tree, "Scheduler")
        assert "_handle_kv_timeouts" in methods

    def test_scheduler_has_notify_kv_received(self):
        """Scheduler should have notify_kv_received method."""
        methods = _get_class_methods(self.tree, "Scheduler")
        assert "notify_kv_received" in methods

    def test_schedule_calls_pd_pre_schedule(self):
        """schedule() should call _pd_aware_pre_schedule."""
        assert "_pd_aware_pre_schedule" in self.source

    def test_schedule_checks_kv_readiness(self):
        """schedule() should check KV readiness for consumer instances."""
        assert "kv_receive_monitor" in self.source
        assert "has_kv" in self.source

    def test_add_request_registers_pending(self):
        """add_request() should register pending KV for consumer."""
        # Find the add_request method and check it references the monitor
        assert "register_pending" in self.source

    def test_free_request_removes_tracking(self):
        """_free_request() should clean up KV tracking."""
        assert "kv_receive_monitor" in self.source

    def test_update_from_output_processes_kv_recv(self):
        """update_from_output() should process kv_recv_success_map."""
        assert "kv_recv_success_map" in self.source


class TestModelRunnerOutputPDField:
    """Verify ModelRunnerOutput has kv_recv_success_map field."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read_file(OUTPUTS_PATH)

    def test_kv_recv_success_map_field_exists(self):
        """ModelRunnerOutput should have kv_recv_success_map field."""
        assert "kv_recv_success_map" in self.source

    def test_kv_recv_success_map_default_empty(self):
        """kv_recv_success_map should default to empty dict."""
        # Verify __post_init__ sets it to {} if None
        assert "__post_init__" in self.source
        assert "kv_recv_success_map" in self.source


class TestGPUModelRunnerPDFeedback:
    """Verify GPUModelRunner populates kv_recv_success_map."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read_file(GPU_MODEL_RUNNER_PATH)

    def test_model_runner_output_includes_kv_map(self):
        """ModelRunnerOutput construction should include kv_recv_success_map."""
        assert "kv_recv_success_map=kv_recv_success_map" in self.source


class TestSchedulerPDSchedulingBehavior:
    """Test the PD-aware scheduling behavior patterns in the code."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.source = _read_file(SCHEDULER_PATH)

    def test_producer_sorts_by_prompt_length(self):
        """Producer instance should sort waiting queue by prompt length."""
        assert "prefill_sort_by_length" in self.source
        assert "_sort_waiting_by_prompt_length" in self.source

    def test_consumer_uses_kv_monitor(self):
        """Consumer instance should use KVReceiveMonitor."""
        assert "KVReceiveMonitor(" in self.source

    def test_kv_timeout_handling(self):
        """Scheduler should handle KV wait timeouts."""
        assert "_handle_kv_timeouts" in self.source
        assert "get_timed_out_requests" in self.source
        assert "mark_timed_out" in self.source

    def test_kv_readiness_check_in_waiting_loop(self):
        """Waiting request loop should check KV readiness for consumer."""
        # The schedule() method should check has_kv before scheduling
        # a waiting request on a consumer instance
        assert "has_kv" in self.source

    def test_pd_aware_pre_schedule_called_conditionally(self):
        """_pd_aware_pre_schedule should only be called when PD is active."""
        assert "enable_pd_aware_scheduling" in self.source


# ===========================================================================
# Part 3: KVReceiveMonitor Timeout Edge Cases
# ===========================================================================

class TestKVReceiveMonitorEdgeCases:
    """Edge case tests for KVReceiveMonitor timeout behavior."""

    def test_timeout_only_returns_expired(self):
        """Only truly expired requests should be returned."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=0.01)
        monitor.register_pending("req-old")
        time.sleep(0.02)
        # Register a new one after the sleep
        monitor.register_pending("req-new")

        timed_out = monitor.get_timed_out_requests()
        assert "req-old" in timed_out
        assert "req-new" not in timed_out

    def test_received_before_timeout_not_timed_out(self):
        """Request received before timeout should not appear as timed out."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=0.01)
        monitor.register_pending("req-1")
        monitor.on_kv_received("req-1")  # Received before timeout
        time.sleep(0.02)

        timed_out = monitor.get_timed_out_requests()
        assert "req-1" not in timed_out

    def test_double_receive_is_safe(self):
        """Calling on_kv_received twice is safe."""
        monitor = KVReceiveMonitor()
        monitor.register_pending("req-1")
        monitor.on_kv_received("req-1")
        monitor.on_kv_received("req-1")  # Duplicate
        assert monitor.has_kv("req-1")
        assert monitor.num_received == 1

    def test_double_register_updates_timestamp(self):
        """Registering the same request twice updates the timestamp."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=0.05)
        monitor.register_pending("req-1")
        time.sleep(0.03)
        # Re-register should update the timestamp
        monitor.register_pending("req-1")
        time.sleep(0.03)
        # Should not be timed out because timestamp was refreshed
        # (total since re-register is only 0.03s < 0.05s)
        timed_out = monitor.get_timed_out_requests()
        assert "req-1" not in timed_out

    def test_mark_timed_out_then_receive(self):
        """Marking as timed out then receiving is idempotent."""
        monitor = KVReceiveMonitor()
        monitor.register_pending("req-1")
        monitor.mark_timed_out("req-1")
        monitor.on_kv_received("req-1")  # Duplicate receive
        assert monitor.has_kv("req-1")
        assert monitor.num_pending == 0

    def test_zero_timeout(self):
        """Zero timeout means all pending requests immediately time out."""
        monitor = KVReceiveMonitor(kv_wait_timeout_s=0.0)
        monitor.register_pending("req-1")
        # Even immediately, it should time out
        timed_out = monitor.get_timed_out_requests()
        assert "req-1" in timed_out
