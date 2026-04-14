"""Tests for Phase 3: Tenant-Level Resource Isolation.

Validates that the TenantManager and its integration with the scheduler
correctly enforce per-tenant concurrency caps and weighted fair scheduling,
preventing a single large tenant from starving other tenants.
"""

import pytest


# ---------------------------------------------------------------------------
# TenantManager unit tests (pure Python, no vllm dependencies)
# ---------------------------------------------------------------------------

class TestTenantManagerConcurrencyCap:
    """Test per-tenant concurrency limits."""

    def _make_manager(self, **kwargs):
        """Create a TenantManager without importing vllm at module level."""
        import sys
        import os
        import types

        # Stub vllm.logger
        if 'vllm' not in sys.modules:
            vllm_mod = types.ModuleType('vllm')
            vllm_mod.__path__ = [
                os.path.join(os.getcwd(), 'vllm')]
            sys.modules['vllm'] = vllm_mod
        if 'vllm.logger' not in sys.modules:
            import logging
            logger_mod = types.ModuleType('vllm.logger')
            logger_mod.init_logger = lambda name: logging.getLogger(name)
            sys.modules['vllm.logger'] = logger_mod

        from vllm.v1.core.tenant_manager import TenantManager
        return TenantManager(**kwargs)

    def test_default_tenant_can_schedule(self):
        """Requests with default tenant_id should be schedulable."""
        mgr = self._make_manager(default_max_running=10)
        assert mgr.can_schedule("default") is True

    def test_concurrency_cap_blocks_excess(self):
        """Once a tenant hits max_running, can_schedule returns False."""
        mgr = self._make_manager(default_max_running=3)
        for _ in range(3):
            assert mgr.can_schedule("gold-a") is True
            mgr.on_request_scheduled("gold-a")
        # 4th request should be blocked
        assert mgr.can_schedule("gold-a") is False

    def test_concurrency_cap_releases_on_finish(self):
        """Finishing a request frees capacity."""
        mgr = self._make_manager(default_max_running=2)
        mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("gold-a")
        assert mgr.can_schedule("gold-a") is False

        mgr.on_request_finished("gold-a")
        assert mgr.can_schedule("gold-a") is True

    def test_per_tenant_custom_limit(self):
        """Registered tenant uses custom max_running."""
        mgr = self._make_manager(default_max_running=100)
        mgr.register_tenant("silver", max_running=2)
        mgr.on_request_scheduled("silver")
        mgr.on_request_scheduled("silver")
        assert mgr.can_schedule("silver") is False
        # Default tenant still has capacity
        assert mgr.can_schedule("default") is True

    def test_multiple_tenants_independent(self):
        """Each tenant's cap is independent."""
        mgr = self._make_manager(default_max_running=2)
        mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("gold-a")
        assert mgr.can_schedule("gold-a") is False
        # Silver is unaffected
        assert mgr.can_schedule("silver") is True

    def test_finish_does_not_go_negative(self):
        """Calling on_request_finished on an empty tenant doesn't go below 0."""
        mgr = self._make_manager(default_max_running=5)
        mgr.on_request_finished("nonexistent")
        assert mgr.get_running_count("nonexistent") == 0


class TestTenantManagerWFQ:
    """Test weighted fair queueing logic."""

    def _make_manager(self, **kwargs):
        import sys
        import os
        import types
        if 'vllm' not in sys.modules:
            vllm_mod = types.ModuleType('vllm')
            vllm_mod.__path__ = [
                os.path.join(os.getcwd(), 'vllm')]
            sys.modules['vllm'] = vllm_mod
        if 'vllm.logger' not in sys.modules:
            import logging
            logger_mod = types.ModuleType('vllm.logger')
            logger_mod.init_logger = lambda name: logging.getLogger(name)
            sys.modules['vllm.logger'] = logger_mod
        from vllm.v1.core.tenant_manager import TenantManager
        return TenantManager(**kwargs)

    def test_idle_tenant_has_highest_weight(self):
        """A tenant with 0 running requests has the highest weight."""
        mgr = self._make_manager(default_weight=1.0)
        mgr.register_tenant("gold-a", weight=2.0)
        mgr.register_tenant("silver", weight=1.0)

        # Both idle: weight = base_weight / max(1, 0) = base_weight
        assert mgr.get_scheduling_weight("gold-a") == 2.0
        assert mgr.get_scheduling_weight("silver") == 1.0

    def test_busy_tenant_has_lower_weight(self):
        """As a tenant uses more slots, its effective weight decreases."""
        mgr = self._make_manager(default_weight=1.0)
        mgr.register_tenant("gold-a", weight=2.0)

        mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("gold-a")
        # weight = 2.0 / max(1, 2) = 1.0
        assert mgr.get_scheduling_weight("gold-a") == pytest.approx(1.0)

        mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("gold-a")
        # weight = 2.0 / max(1, 4) = 0.5
        assert mgr.get_scheduling_weight("gold-a") == pytest.approx(0.5)

    def test_wfq_prefers_underserved_tenant(self):
        """The tenant with fewer running requests should have higher weight."""
        mgr = self._make_manager(default_weight=1.0)
        # gold-a has 5 running, silver has 1 running
        for _ in range(5):
            mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("silver")

        # gold-a: 1.0/5 = 0.2, silver: 1.0/1 = 1.0
        assert (mgr.get_scheduling_weight("silver")
                > mgr.get_scheduling_weight("gold-a"))

    def test_default_tenant_weight(self):
        """Unregistered tenant uses default weight."""
        mgr = self._make_manager(default_weight=1.5)
        assert mgr.get_scheduling_weight("unknown") == 1.5


class TestTenantManagerStats:
    """Test observability/stats methods."""

    def _make_manager(self, **kwargs):
        import sys
        import os
        import types
        if 'vllm' not in sys.modules:
            vllm_mod = types.ModuleType('vllm')
            vllm_mod.__path__ = [
                os.path.join(os.getcwd(), 'vllm')]
            sys.modules['vllm'] = vllm_mod
        if 'vllm.logger' not in sys.modules:
            import logging
            logger_mod = types.ModuleType('vllm.logger')
            logger_mod.init_logger = lambda name: logging.getLogger(name)
            sys.modules['vllm.logger'] = logger_mod
        from vllm.v1.core.tenant_manager import TenantManager
        return TenantManager(**kwargs)

    def test_get_tenant_stats(self):
        """Stats snapshot includes all known tenants."""
        mgr = self._make_manager(default_max_running=10, default_weight=1.0)
        mgr.register_tenant("gold-a", max_running=5, weight=2.0)
        mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("silver")  # auto-created

        stats = mgr.get_tenant_stats()
        assert "gold-a" in stats
        assert "silver" in stats
        assert stats["gold-a"]["running"] == 1
        assert stats["gold-a"]["max_running"] == 5
        assert stats["gold-a"]["weight"] == 2.0
        assert stats["silver"]["running"] == 1
        assert stats["silver"]["max_running"] == 10  # default


class TestTenantIsolationIntegration:
    """Integration test: simulate Gold-A burst starving Silver.

    Scenario (mirrors Phase 3 in LANDING_PLAN):
    - Gold-A has 4x burst (8 concurrent requests)
    - Silver has normal load (3 requests)
    - Without tenant isolation: Gold-A takes all slots, Silver starved
    - With tenant isolation (max_running=5 per tenant): Silver gets scheduled
    """

    def test_gold_a_burst_does_not_starve_silver(self):
        """With tenant caps, Silver requests get scheduled even during
        Gold-A burst."""
        import sys
        import os
        import types
        if 'vllm' not in sys.modules:
            vllm_mod = types.ModuleType('vllm')
            vllm_mod.__path__ = [
                os.path.join(os.getcwd(), 'vllm')]
            sys.modules['vllm'] = vllm_mod
        if 'vllm.logger' not in sys.modules:
            import logging
            logger_mod = types.ModuleType('vllm.logger')
            logger_mod.init_logger = lambda name: logging.getLogger(name)
            sys.modules['vllm.logger'] = logger_mod
        from vllm.v1.core.tenant_manager import TenantManager

        total_capacity = 10
        mgr = TenantManager(default_max_running=5, default_weight=1.0)

        # Simulate scheduling: Gold-A requests arrive first
        gold_a_scheduled = 0
        silver_scheduled = 0

        # Gold-A burst: 8 requests want to schedule
        gold_a_requests = 8
        # Silver: 3 requests want to schedule
        silver_requests = 3

        running_total = 0

        # Round-robin simulation (scheduler iterates waiting queue)
        waiting = (
            [("gold-a", f"ga-{i}") for i in range(gold_a_requests)]
            + [("silver", f"sv-{i}") for i in range(silver_requests)]
        )

        for tenant_id, req_id in waiting:
            if running_total >= total_capacity:
                break
            if mgr.can_schedule(tenant_id):
                mgr.on_request_scheduled(tenant_id)
                running_total += 1
                if tenant_id == "gold-a":
                    gold_a_scheduled += 1
                else:
                    silver_scheduled += 1

        # Gold-A should be capped at 5 (not 8)
        assert gold_a_scheduled == 5
        # Silver should get all 3 slots
        assert silver_scheduled == 3
        # Total fits in capacity
        assert running_total == 8

    def test_without_isolation_gold_starves_silver(self):
        """Without tenant caps, Gold-A takes all capacity."""
        import sys
        import os
        import types
        if 'vllm' not in sys.modules:
            vllm_mod = types.ModuleType('vllm')
            vllm_mod.__path__ = [
                os.path.join(os.getcwd(), 'vllm')]
            sys.modules['vllm'] = vllm_mod
        if 'vllm.logger' not in sys.modules:
            import logging
            logger_mod = types.ModuleType('vllm.logger')
            logger_mod.init_logger = lambda name: logging.getLogger(name)
            sys.modules['vllm.logger'] = logger_mod
        from vllm.v1.core.tenant_manager import TenantManager

        total_capacity = 8
        # No isolation: max_running = total_capacity (effectively unlimited)
        mgr = TenantManager(
            default_max_running=total_capacity, default_weight=1.0)

        gold_a_scheduled = 0
        silver_scheduled = 0
        running_total = 0

        waiting = (
            [("gold-a", f"ga-{i}") for i in range(8)]
            + [("silver", f"sv-{i}") for i in range(3)]
        )

        for tenant_id, req_id in waiting:
            if running_total >= total_capacity:
                break
            if mgr.can_schedule(tenant_id):
                mgr.on_request_scheduled(tenant_id)
                running_total += 1
                if tenant_id == "gold-a":
                    gold_a_scheduled += 1
                else:
                    silver_scheduled += 1

        # Gold-A takes all 8 slots, Silver gets none
        assert gold_a_scheduled == 8
        assert silver_scheduled == 0

    def test_wfq_ordering_favors_underserved_tenant(self):
        """When reordering by WFQ weight, underserved tenants go first."""
        import sys
        import os
        import types
        if 'vllm' not in sys.modules:
            vllm_mod = types.ModuleType('vllm')
            vllm_mod.__path__ = [
                os.path.join(os.getcwd(), 'vllm')]
            sys.modules['vllm'] = vllm_mod
        if 'vllm.logger' not in sys.modules:
            import logging
            logger_mod = types.ModuleType('vllm.logger')
            logger_mod.init_logger = lambda name: logging.getLogger(name)
            sys.modules['vllm.logger'] = logger_mod
        from vllm.v1.core.tenant_manager import TenantManager

        mgr = TenantManager(default_max_running=10, default_weight=1.0)
        # Simulate: gold-a already has 5 running, silver has 1
        for _ in range(5):
            mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("silver")

        # Candidates from mixed tenants
        candidates = [
            ("gold-a", "ga-6"),
            ("silver", "sv-2"),
            ("gold-a", "ga-7"),
            ("silver", "sv-3"),
        ]

        # Sort by WFQ weight (descending = higher weight first)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: mgr.get_scheduling_weight(c[0]),
            reverse=True,
        )

        # Silver should come first (weight 1.0/1=1.0 > 1.0/5=0.2)
        assert sorted_candidates[0][0] == "silver"
        assert sorted_candidates[1][0] == "silver"


class TestTenantIsolationE2E:
    """End-to-end scenario: Phase 3 Gold-A burst with dynamic adaptation."""

    def test_phase3_scenario(self):
        """Simulate the full Phase 3 lifecycle:
        1. Steady state: Gold-A=2, Silver=2 running
        2. Gold-A bursts 4x: 8 new requests arrive
        3. Tenant cap prevents starvation
        4. Gold-A requests finish, Silver gets scheduled
        5. System returns to steady state
        """
        import sys
        import os
        import types
        if 'vllm' not in sys.modules:
            vllm_mod = types.ModuleType('vllm')
            vllm_mod.__path__ = [
                os.path.join(os.getcwd(), 'vllm')]
            sys.modules['vllm'] = vllm_mod
        if 'vllm.logger' not in sys.modules:
            import logging
            logger_mod = types.ModuleType('vllm.logger')
            logger_mod.init_logger = lambda name: logging.getLogger(name)
            sys.modules['vllm.logger'] = logger_mod
        from vllm.v1.core.tenant_manager import TenantManager

        mgr = TenantManager(default_max_running=5, default_weight=1.0)

        # --- Step 1: Steady state ---
        mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("gold-a")
        mgr.on_request_scheduled("silver")
        mgr.on_request_scheduled("silver")
        assert mgr.get_running_count("gold-a") == 2
        assert mgr.get_running_count("silver") == 2

        # --- Step 2: Gold-A burst ---
        burst_scheduled = 0
        for _ in range(8):
            if mgr.can_schedule("gold-a"):
                mgr.on_request_scheduled("gold-a")
                burst_scheduled += 1
        # Only 3 more can be scheduled (cap=5, already had 2)
        assert burst_scheduled == 3
        assert mgr.get_running_count("gold-a") == 5
        assert mgr.can_schedule("gold-a") is False

        # --- Step 3: Silver still has capacity ---
        assert mgr.can_schedule("silver") is True
        mgr.on_request_scheduled("silver")
        assert mgr.get_running_count("silver") == 3

        # --- Step 4: Some Gold-A requests finish ---
        mgr.on_request_finished("gold-a")
        mgr.on_request_finished("gold-a")
        mgr.on_request_finished("gold-a")
        assert mgr.get_running_count("gold-a") == 2
        assert mgr.can_schedule("gold-a") is True

        # --- Step 5: Back to steady state ---
        mgr.on_request_finished("silver")
        assert mgr.get_running_count("silver") == 2
        stats = mgr.get_tenant_stats()
        assert stats["gold-a"]["running"] == 2
        assert stats["silver"]["running"] == 2
