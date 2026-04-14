# SPDX-License-Identifier: Apache-2.0
"""Tenant-level resource isolation manager.

Provides per-tenant concurrency caps and weighted fair scheduling (simplified
WFQ) to prevent a single large tenant from starving others when traffic surges.

Key capabilities:
1. **Concurrency cap**: Each tenant has a ``max_running`` limit.  Once reached,
   new requests from that tenant are skipped in the current scheduling step
   (they stay in the waiting queue for the next step).
2. **Weighted fair scheduling**: ``get_scheduling_weight()`` returns a dynamic
   weight that decreases as a tenant consumes more running slots.  The
   scheduler can use this to reorder candidates within the same priority
   level so that underserved tenants get preference.

Usage in scheduler:
    - Before scheduling a WAITING request, call ``can_schedule(tenant_id)``
      to check the concurrency cap.
    - After scheduling, call ``on_request_scheduled(tenant_id)``.
    - When a request finishes, call ``on_request_finished(tenant_id)``.
"""

from typing import Dict, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


class TenantManager:
    """Per-tenant resource isolation with concurrency caps and WFQ weights."""

    # Sentinel tenant ID used when no tenant is specified.
    DEFAULT_TENANT_ID: str = "default"

    def __init__(
        self,
        default_max_running: int = 64,
        default_weight: float = 1.0,
    ) -> None:
        self.default_max_running = default_max_running
        self.default_weight = default_weight

        # Per-tenant configuration
        self.tenant_max_running: Dict[str, int] = {}
        self.tenant_weights: Dict[str, float] = {}

        # Per-tenant runtime state
        self.tenant_running: Dict[str, int] = {}  # current running count

    # ---- Configuration API ----

    def register_tenant(
        self,
        tenant_id: str,
        max_running: Optional[int] = None,
        weight: Optional[float] = None,
    ) -> None:
        """Register or update a tenant's configuration.

        Args:
            tenant_id: Unique identifier for the tenant.
            max_running: Maximum number of concurrent running requests for
                this tenant.  ``None`` means use the global default.
            weight: Scheduling weight for WFQ.  Higher weight = more share.
                ``None`` means use the global default (1.0).
        """
        if max_running is not None:
            self.tenant_max_running[tenant_id] = max_running
        if weight is not None:
            self.tenant_weights[tenant_id] = weight
        logger.info(
            "Tenant registered: %s (max_running=%s, weight=%s)",
            tenant_id,
            self.tenant_max_running.get(tenant_id, self.default_max_running),
            self.tenant_weights.get(tenant_id, self.default_weight),
        )

    # ---- Scheduling decision API ----

    def can_schedule(self, tenant_id: str) -> bool:
        """Check whether the tenant still has capacity for another request.

        Returns True if the tenant's current running count is below its
        configured max_running limit.
        """
        current = self.tenant_running.get(tenant_id, 0)
        max_allowed = self.tenant_max_running.get(
            tenant_id, self.default_max_running)
        return current < max_allowed

    def get_scheduling_weight(self, tenant_id: str) -> float:
        """Return the effective scheduling weight for WFQ ordering.

        The effective weight decreases as a tenant uses more running slots:
            effective_weight = base_weight / max(1, running_count)

        This means a tenant that already has many running requests will have
        a lower scheduling weight, giving underserved tenants preference.
        """
        weight = self.tenant_weights.get(tenant_id, self.default_weight)
        running = self.tenant_running.get(tenant_id, 0)
        return weight / max(1, running)

    def get_running_count(self, tenant_id: str) -> int:
        """Return the current number of running requests for a tenant."""
        return self.tenant_running.get(tenant_id, 0)

    # ---- Lifecycle callbacks ----

    def on_request_scheduled(self, tenant_id: str) -> None:
        """Called when a request from this tenant is scheduled to run."""
        self.tenant_running[tenant_id] = (
            self.tenant_running.get(tenant_id, 0) + 1)

    def on_request_finished(self, tenant_id: str) -> None:
        """Called when a request from this tenant finishes (or is aborted)."""
        current = self.tenant_running.get(tenant_id, 0)
        self.tenant_running[tenant_id] = max(0, current - 1)

    # ---- Observability ----

    def get_tenant_stats(self) -> Dict[str, Dict[str, float]]:
        """Return a snapshot of per-tenant stats for monitoring."""
        all_tenants = set(self.tenant_running.keys())
        all_tenants.update(self.tenant_max_running.keys())
        all_tenants.update(self.tenant_weights.keys())
        stats: Dict[str, Dict[str, float]] = {}
        for tid in sorted(all_tenants):
            stats[tid] = {
                "running": self.tenant_running.get(tid, 0),
                "max_running": self.tenant_max_running.get(
                    tid, self.default_max_running),
                "weight": self.tenant_weights.get(
                    tid, self.default_weight),
                "effective_weight": self.get_scheduling_weight(tid),
            }
        return stats
