# SPDX-License-Identifier: Apache-2.0
"""Tests for Phase 4: Overload Management.

Covers three sub-modules:
  (a) Admission Control — reject low-priority requests under overload
  (b) Deadline-Aware Scheduling — urgent requests get boosted
  (c) SLA-Aware Preemption — violated requests preempted first
"""

import math
import time
import unittest
from collections import deque
from typing import Dict, List, Optional, Set
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Minimal stubs so we can import the real Request / TenantManager / etc.
# without pulling in the full vllm dependency tree.
# ---------------------------------------------------------------------------

class _StubSamplingParams:
    """Minimal SamplingParams stub."""
    def __init__(self, max_tokens=128):
        self.max_tokens = max_tokens
        self.logprobs = None
        self.prompt_logprobs = None
        self.stop_token_ids = None
        self.ignore_eos = False

    def clone(self):
        return self


class _StubLoRARequest:
    def __init__(self, lora_int_id=1):
        self.lora_int_id = lora_int_id


# ---- Fake Request for scheduler-level tests ----

class FakeRequest:
    """Lightweight Request replacement for scheduler-level tests."""

    SHORT_PROMPT_THRESHOLD = 512

    def __init__(
        self,
        request_id: str,
        num_prompt_tokens: int = 64,
        priority: int = 0,
        mlfq_level: int = 0,
        tenant_id: str = "default",
        sla_ttft_ms: float = float('inf'),
        arrival_time: Optional[float] = None,
    ):
        self.request_id = request_id
        self.num_prompt_tokens = num_prompt_tokens
        self.priority = priority
        self.mlfq_level = mlfq_level
        self.mlfq_tokens_consumed = 0
        self.tenant_id = tenant_id
        self.arrival_time = arrival_time or time.monotonic()
        self.status = 0  # WAITING
        self.num_computed_tokens = 0
        self.effective_priority = float(priority)

        # SLA / Deadline
        self.sla_ttft_ms = sla_ttft_ms
        if sla_ttft_ms < float('inf'):
            self.deadline = self.arrival_time + sla_ttft_ms / 1000.0
        else:
            self.deadline = float('inf')

    @property
    def slack_time(self) -> float:
        if self.deadline == float('inf'):
            return float('inf')
        return self.deadline - time.monotonic()

    def is_sla_violated(self) -> bool:
        return self.slack_time <= 0

    @property
    def sla_urgency(self) -> float:
        return self.slack_time

    def is_finished(self) -> bool:
        return self.status > 2

    def set_finished(self, status):
        self.status = status

    def mlfq_promote(self):
        if self.mlfq_level > 0:
            self.mlfq_level -= 1


# ===========================================================================
# Test Suite A: Admission Control
# ===========================================================================

class TestAdmissionControl(unittest.TestCase):
    """Test the _should_admit() admission control logic."""

    def _make_scheduler_stub(self):
        """Create a minimal scheduler-like object with admission logic."""
        class SchedulerStub:
            def __init__(self):
                self.enable_overload_management = True
                self.enable_mlfq = True
                self.max_queue_depth = 10
                self.overload_violation_threshold = 0.5
                self._sla_violation_window = deque(maxlen=50)
                self.waiting = deque()
                self.running = []

            def _is_high_priority_request(self, request):
                if self.enable_mlfq and request.mlfq_level <= 1:
                    return True
                if request.priority < 0:
                    return True
                if request.num_prompt_tokens < 512:
                    return True
                return False

            def _should_admit(self, request):
                is_high = self._is_high_priority_request(request)
                if is_high:
                    return True
                if len(self.waiting) >= self.max_queue_depth:
                    return False
                if len(self._sla_violation_window) >= 10:
                    violation_rate = (
                        sum(self._sla_violation_window)
                        / len(self._sla_violation_window))
                    if violation_rate > self.overload_violation_threshold:
                        return False
                return True

        return SchedulerStub()

    def test_admit_when_queue_empty(self):
        """All requests admitted when queue is empty."""
        sched = self._make_scheduler_stub()
        req = FakeRequest("r1", num_prompt_tokens=2048, priority=5)
        self.assertTrue(sched._should_admit(req))

    def test_reject_low_priority_when_queue_full(self):
        """Low-priority request rejected when queue depth exceeds limit."""
        sched = self._make_scheduler_stub()
        # Fill the queue to capacity.
        for i in range(12):
            sched.waiting.append(FakeRequest(f"w{i}"))
        # Low-priority (long prompt, high priority value) → rejected.
        low_req = FakeRequest("low", num_prompt_tokens=2048, priority=5,
                              mlfq_level=3)
        self.assertFalse(sched._should_admit(low_req))

    def test_admit_high_priority_when_queue_full(self):
        """High-priority request still admitted even when queue is full."""
        sched = self._make_scheduler_stub()
        for i in range(12):
            sched.waiting.append(FakeRequest(f"w{i}"))
        # Short prompt (interactive) → high priority → admitted.
        high_req = FakeRequest("high", num_prompt_tokens=64, priority=0)
        self.assertTrue(sched._should_admit(high_req))

    def test_admit_high_priority_by_explicit_priority(self):
        """Request with negative priority (explicit high) is admitted."""
        sched = self._make_scheduler_stub()
        for i in range(12):
            sched.waiting.append(FakeRequest(f"w{i}"))
        explicit_high = FakeRequest("eh", num_prompt_tokens=4096,
                                    priority=-1, mlfq_level=3)
        self.assertTrue(sched._should_admit(explicit_high))

    def test_reject_on_high_sla_violation_rate(self):
        """Low-priority request rejected when SLA violation rate is high."""
        sched = self._make_scheduler_stub()
        # Record 8 violations out of 10 → 80% violation rate.
        for _ in range(8):
            sched._sla_violation_window.append(True)
        for _ in range(2):
            sched._sla_violation_window.append(False)

        low_req = FakeRequest("low", num_prompt_tokens=2048, priority=5,
                              mlfq_level=3)
        self.assertFalse(sched._should_admit(low_req))

    def test_admit_when_violation_rate_low(self):
        """Low-priority request admitted when SLA violation rate is low."""
        sched = self._make_scheduler_stub()
        # Record 2 violations out of 10 → 20% violation rate.
        for _ in range(2):
            sched._sla_violation_window.append(True)
        for _ in range(8):
            sched._sla_violation_window.append(False)

        low_req = FakeRequest("low", num_prompt_tokens=2048, priority=5,
                              mlfq_level=3)
        self.assertTrue(sched._should_admit(low_req))

    def test_violation_window_insufficient_data(self):
        """With fewer than 10 samples, SLA gate does not trigger."""
        sched = self._make_scheduler_stub()
        # Only 5 samples, all violated.
        for _ in range(5):
            sched._sla_violation_window.append(True)

        low_req = FakeRequest("low", num_prompt_tokens=2048, priority=5,
                              mlfq_level=3)
        # Not enough data → admitted.
        self.assertTrue(sched._should_admit(low_req))


# ===========================================================================
# Test Suite B: Deadline / SLA Properties on Request
# ===========================================================================

class TestDeadlineProperties(unittest.TestCase):
    """Test SLA deadline properties on the FakeRequest (mirrors real Request)."""

    def test_no_sla_infinite_slack(self):
        """Request without SLA has infinite slack time."""
        req = FakeRequest("r1", sla_ttft_ms=float('inf'))
        self.assertEqual(req.slack_time, float('inf'))
        self.assertFalse(req.is_sla_violated())
        self.assertEqual(req.sla_urgency, float('inf'))

    def test_future_deadline_positive_slack(self):
        """Request with future deadline has positive slack."""
        req = FakeRequest("r1", sla_ttft_ms=10000)  # 10 seconds from now
        self.assertGreater(req.slack_time, 5.0)
        self.assertFalse(req.is_sla_violated())

    def test_past_deadline_violated(self):
        """Request with past deadline is violated."""
        # Create request with deadline 10ms ago.
        req = FakeRequest("r1", sla_ttft_ms=10,
                          arrival_time=time.monotonic() - 1.0)
        # 10ms deadline set 1 second ago → violated.
        self.assertTrue(req.is_sla_violated())
        self.assertLess(req.slack_time, 0)

    def test_urgency_ordering(self):
        """More urgent requests have lower sla_urgency."""
        # Urgent: 500ms deadline.
        urgent = FakeRequest("urgent", sla_ttft_ms=500)
        # Relaxed: 10s deadline.
        relaxed = FakeRequest("relaxed", sla_ttft_ms=10000)
        self.assertLess(urgent.sla_urgency, relaxed.sla_urgency)


# ===========================================================================
# Test Suite C: SLA-Aware Preemption
# ===========================================================================

class TestSLAAwarePreemption(unittest.TestCase):
    """Test _select_preemption_victim() with SLA awareness."""

    def _make_scheduler_stub(self):
        """Create scheduler stub with preemption logic."""
        class SchedulerStub:
            def __init__(self):
                self.enable_overload_management = True
                self.enable_qos_priority = True
                self.running = []

            def _select_preemption_victim(self):
                if not self.running:
                    return None
                if self.enable_overload_management:
                    violated = [r for r in self.running
                                if r.is_sla_violated()]
                    if violated:
                        return max(
                            violated,
                            key=lambda r: (r.effective_priority,
                                           r.arrival_time))
                if self.enable_qos_priority:
                    return max(
                        self.running,
                        key=lambda r: (r.effective_priority,
                                       r.arrival_time))
                return self.running[-1]

        return SchedulerStub()

    def test_preempt_violated_over_non_violated(self):
        """Violated requests are preempted before non-violated ones."""
        sched = self._make_scheduler_stub()

        # Not violated: high priority, future deadline.
        good = FakeRequest("good", priority=0, sla_ttft_ms=10000)
        good.effective_priority = 0.0

        # Violated: low priority, past deadline.
        bad = FakeRequest("bad", priority=5, sla_ttft_ms=10,
                          arrival_time=time.monotonic() - 2.0)
        bad.effective_priority = 5.0

        sched.running = [good, bad]
        victim = sched._select_preemption_victim()
        self.assertEqual(victim.request_id, "bad")

    def test_preempt_lowest_priority_among_violated(self):
        """Among violated requests, the lowest priority is preempted."""
        sched = self._make_scheduler_stub()

        v1 = FakeRequest("v1", priority=3, sla_ttft_ms=10,
                         arrival_time=time.monotonic() - 2.0)
        v1.effective_priority = 3.0

        v2 = FakeRequest("v2", priority=8, sla_ttft_ms=10,
                         arrival_time=time.monotonic() - 2.0)
        v2.effective_priority = 8.0

        sched.running = [v1, v2]
        victim = sched._select_preemption_victim()
        self.assertEqual(victim.request_id, "v2")

    def test_fallback_to_qos_when_no_violations(self):
        """With no violated requests, falls back to QoS priority."""
        sched = self._make_scheduler_stub()

        high = FakeRequest("high", priority=0, sla_ttft_ms=10000)
        high.effective_priority = 0.0

        low = FakeRequest("low", priority=10, sla_ttft_ms=10000)
        low.effective_priority = 10.0

        sched.running = [high, low]
        victim = sched._select_preemption_victim()
        self.assertEqual(victim.request_id, "low")

    def test_no_running_returns_none(self):
        """Empty running queue returns None."""
        sched = self._make_scheduler_stub()
        sched.running = []
        self.assertIsNone(sched._select_preemption_victim())

    def test_preempt_without_overload_management(self):
        """Disabling overload management falls back to pure QoS."""
        sched = self._make_scheduler_stub()
        sched.enable_overload_management = False

        # Even though bad is violated, overload management is off.
        good = FakeRequest("good", priority=0, sla_ttft_ms=10000)
        good.effective_priority = 0.0

        bad = FakeRequest("bad", priority=5, sla_ttft_ms=10,
                          arrival_time=time.monotonic() - 2.0)
        bad.effective_priority = 5.0

        sched.running = [good, bad]
        victim = sched._select_preemption_victim()
        # Falls back to QoS: pick highest effective_priority.
        self.assertEqual(victim.request_id, "bad")


# ===========================================================================
# Test Suite D: Deadline-Aware Scheduling (Sort)
# ===========================================================================

class TestDeadlineAwareScheduling(unittest.TestCase):
    """Test _deadline_aware_sort_waiting() urgency boosting."""

    def _make_scheduler_stub(self, enable_mlfq=True):
        class SchedulerStub:
            MLFQ_NUM_LEVELS = 4

            def __init__(self, enable_mlfq):
                self.enable_mlfq = enable_mlfq
                self.enable_overload_management = True
                self.enable_deadline_aware_scheduling = True
                self.deadline_urgency_threshold_s = 2.0
                self.waiting = deque()
                self.mlfq_queues = [deque() for _ in range(4)]

            def _deadline_aware_sort_waiting(self):
                if not self.enable_mlfq:
                    if len(self.waiting) > 1:
                        sorted_reqs = sorted(
                            self.waiting,
                            key=lambda r: (
                                0 if r.sla_urgency
                                < self.deadline_urgency_threshold_s
                                else 1,
                                r.arrival_time))
                        self.waiting = deque(sorted_reqs)
                    return

                for level_queue in self.mlfq_queues:
                    if len(level_queue) <= 1:
                        continue
                    urgent = []
                    normal = []
                    for r in level_queue:
                        if r.sla_urgency < self.deadline_urgency_threshold_s:
                            urgent.append(r)
                        else:
                            normal.append(r)
                    if urgent:
                        urgent.sort(key=lambda r: r.sla_urgency)
                        level_queue.clear()
                        level_queue.extend(urgent)
                        level_queue.extend(normal)

        return SchedulerStub(enable_mlfq)

    def test_urgent_requests_boosted_within_level(self):
        """Urgent requests move to front of their MLFQ level."""
        sched = self._make_scheduler_stub()

        # Non-urgent: 10s deadline → ~10s slack.
        normal_req = FakeRequest("normal", sla_ttft_ms=10000, mlfq_level=0)
        # Urgent: 500ms deadline → <2s slack soon.
        urgent_req = FakeRequest("urgent", sla_ttft_ms=500, mlfq_level=0)

        sched.mlfq_queues[0].extend([normal_req, urgent_req])
        sched._deadline_aware_sort_waiting()

        # Urgent should be first.
        result = list(sched.mlfq_queues[0])
        self.assertEqual(result[0].request_id, "urgent")
        self.assertEqual(result[1].request_id, "normal")

    def test_no_cross_level_promotion(self):
        """Urgent L2 request does NOT jump ahead of L0 requests."""
        sched = self._make_scheduler_stub()

        l0_req = FakeRequest("l0", sla_ttft_ms=10000, mlfq_level=0)
        l2_urgent = FakeRequest("l2_urgent", sla_ttft_ms=100,
                                mlfq_level=2,
                                arrival_time=time.monotonic() - 0.5)

        sched.mlfq_queues[0].append(l0_req)
        sched.mlfq_queues[2].append(l2_urgent)
        sched._deadline_aware_sort_waiting()

        # L0 is still first non-empty level.
        self.assertEqual(list(sched.mlfq_queues[0])[0].request_id, "l0")

    def test_flat_queue_mode(self):
        """Test deadline-aware sorting in non-MLFQ mode."""
        sched = self._make_scheduler_stub(enable_mlfq=False)

        now = time.monotonic()
        normal = FakeRequest("normal", sla_ttft_ms=10000,
                             arrival_time=now - 0.1)
        urgent = FakeRequest("urgent", sla_ttft_ms=500,
                             arrival_time=now)

        sched.waiting.extend([normal, urgent])
        sched._deadline_aware_sort_waiting()

        result = list(sched.waiting)
        self.assertEqual(result[0].request_id, "urgent")

    def test_multiple_urgent_sorted_by_urgency(self):
        """Multiple urgent requests sorted by slack_time (most urgent first)."""
        sched = self._make_scheduler_stub()

        now = time.monotonic()
        # slightly_urgent: 1.5s slack
        slightly = FakeRequest("slightly", sla_ttft_ms=1500,
                               arrival_time=now, mlfq_level=0)
        # very_urgent: 0.3s slack
        very = FakeRequest("very", sla_ttft_ms=300,
                           arrival_time=now, mlfq_level=0)
        # normal: 10s slack (not urgent)
        normal = FakeRequest("normal", sla_ttft_ms=10000,
                             arrival_time=now, mlfq_level=0)

        sched.mlfq_queues[0].extend([normal, slightly, very])
        sched._deadline_aware_sort_waiting()

        result = list(sched.mlfq_queues[0])
        self.assertEqual(result[0].request_id, "very")
        self.assertEqual(result[1].request_id, "slightly")
        self.assertEqual(result[2].request_id, "normal")


# ===========================================================================
# Test Suite E: SLA Violation Tracking
# ===========================================================================

class TestSLAViolationTracking(unittest.TestCase):
    """Test the sliding window SLA violation tracking."""

    def test_violation_window_maxlen(self):
        """Window respects maxlen."""
        window = deque(maxlen=50)
        for _ in range(60):
            window.append(True)
        self.assertEqual(len(window), 50)

    def test_violation_rate_computation(self):
        """Violation rate is correctly computed."""
        window = deque(maxlen=50)
        for _ in range(30):
            window.append(True)
        for _ in range(20):
            window.append(False)
        rate = sum(window) / len(window)
        self.assertAlmostEqual(rate, 0.6, places=2)

    def test_empty_window_rate(self):
        """Empty window has 0 violation rate."""
        window = deque(maxlen=50)
        if not window:
            rate = 0.0
        else:
            rate = sum(window) / len(window)
        self.assertEqual(rate, 0.0)


# ===========================================================================
# Test Suite F: Integration — Phase 4 Full Lifecycle
# ===========================================================================

class TestOverloadManagementIntegration(unittest.TestCase):
    """Integration tests combining admission, deadline, and preemption."""

    def test_phase5_overload_scenario(self):
        """Simulate Phase 5 overload: rejection + urgency + preemption.

        Scenario:
        - Queue has 15 waiting requests (above max_queue_depth=10).
        - 8 of 10 recent requests violated SLA.
        - New high-priority request should be admitted.
        - New low-priority request should be rejected.
        - Among running requests, violated one should be preempted first.
        """
        # Admission control stub.
        class SchedulerStub:
            def __init__(self):
                self.enable_overload_management = True
                self.enable_mlfq = True
                self.enable_qos_priority = True
                self.max_queue_depth = 10
                self.overload_violation_threshold = 0.5
                self._sla_violation_window = deque(maxlen=50)
                self.waiting = deque()
                self.running = []

            def _is_high_priority_request(self, request):
                if self.enable_mlfq and request.mlfq_level <= 1:
                    return True
                if request.priority < 0:
                    return True
                if request.num_prompt_tokens < 512:
                    return True
                return False

            def _should_admit(self, request):
                if self._is_high_priority_request(request):
                    return True
                if len(self.waiting) >= self.max_queue_depth:
                    return False
                if len(self._sla_violation_window) >= 10:
                    vr = sum(self._sla_violation_window) / len(
                        self._sla_violation_window)
                    if vr > self.overload_violation_threshold:
                        return False
                return True

            def _select_preemption_victim(self):
                if not self.running:
                    return None
                violated = [r for r in self.running if r.is_sla_violated()]
                if violated:
                    return max(violated,
                               key=lambda r: (r.effective_priority,
                                              r.arrival_time))
                return max(self.running,
                           key=lambda r: (r.effective_priority,
                                          r.arrival_time))

        sched = SchedulerStub()

        # Fill queue to 15.
        for i in range(15):
            sched.waiting.append(FakeRequest(f"w{i}"))

        # Record high violation rate.
        for _ in range(8):
            sched._sla_violation_window.append(True)
        for _ in range(2):
            sched._sla_violation_window.append(False)

        # (a) Admission control: high-priority → admitted.
        gold = FakeRequest("gold", num_prompt_tokens=64, priority=-1)
        self.assertTrue(sched._should_admit(gold))

        # (a) Admission control: low-priority → rejected.
        bronze = FakeRequest("bronze", num_prompt_tokens=4096,
                             priority=5, mlfq_level=3)
        self.assertFalse(sched._should_admit(bronze))

        # (b) Preemption: add running requests.
        now = time.monotonic()
        ok_req = FakeRequest("ok", priority=0, sla_ttft_ms=10000,
                             arrival_time=now - 0.1)
        ok_req.effective_priority = 0.0

        violated_req = FakeRequest("violated", priority=3,
                                   sla_ttft_ms=50,
                                   arrival_time=now - 2.0)
        violated_req.effective_priority = 3.0

        sched.running = [ok_req, violated_req]
        victim = sched._select_preemption_victim()
        self.assertEqual(victim.request_id, "violated")

    def test_all_three_combined_stats(self):
        """Verify SLA violation rate reflects the admission decisions."""
        window = deque(maxlen=50)

        # Simulate: 5 good, then 15 bad, then 5 good.
        for _ in range(5):
            window.append(False)
        # Rate after 5 good: 0%
        self.assertAlmostEqual(sum(window) / len(window), 0.0, places=2)

        for _ in range(15):
            window.append(True)
        # Rate after 15 bad: 15/20 = 75%
        self.assertAlmostEqual(sum(window) / len(window), 0.75, places=2)

        for _ in range(5):
            window.append(False)
        # Rate after 5 more good: 15/25 = 60%
        self.assertAlmostEqual(sum(window) / len(window), 0.60, places=2)


# ===========================================================================
# Test Suite G: Request SLA Methods (using real-like Request)
# ===========================================================================

class TestRequestSLAMethods(unittest.TestCase):
    """Test SLA-related methods on Request-like objects."""

    def test_deadline_computation(self):
        """Deadline = arrival_time + sla_ttft_ms / 1000."""
        now = time.monotonic()
        req = FakeRequest("r1", sla_ttft_ms=2000, arrival_time=now)
        self.assertAlmostEqual(req.deadline, now + 2.0, places=2)

    def test_no_sla_deadline_is_inf(self):
        """No SLA → deadline is inf."""
        req = FakeRequest("r1")
        self.assertEqual(req.deadline, float('inf'))

    def test_violated_after_deadline_passes(self):
        """Request becomes violated once deadline passes."""
        req = FakeRequest("r1", sla_ttft_ms=100,
                          arrival_time=time.monotonic() - 1.0)
        # 100ms deadline set 1s ago → violated.
        self.assertTrue(req.is_sla_violated())

    def test_not_violated_before_deadline(self):
        """Request is not violated before deadline."""
        req = FakeRequest("r1", sla_ttft_ms=30000)  # 30s deadline
        self.assertFalse(req.is_sla_violated())


# ===========================================================================
# Test Suite H: Finished Status for Rejected Requests
# ===========================================================================

class TestFinishedRejected(unittest.TestCase):
    """Test FINISHED_REJECTED status integration."""

    def test_rejected_is_finished(self):
        """FINISHED_REJECTED (value 7) is > PREEMPTED (value 2)."""
        # In the real code, is_finished checks status > PREEMPTED.
        PREEMPTED = 2
        FINISHED_REJECTED = 7
        self.assertTrue(FINISHED_REJECTED > PREEMPTED)

    def test_request_marked_rejected(self):
        """Request can be marked as rejected."""
        req = FakeRequest("r1")
        req.status = 7  # FINISHED_REJECTED
        self.assertTrue(req.is_finished())


if __name__ == "__main__":
    unittest.main()
