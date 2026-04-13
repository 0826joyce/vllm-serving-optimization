# SPDX-License-Identifier: Apache-2.0
"""PD Disaggregation - Health Monitor & Load Metrics Collection.

Periodically collects load metrics from Prefill and Decode instances
via their Prometheus /metrics endpoints, and maintains health status
for intelligent routing decisions.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import aiohttp

from vllm.logger import init_logger

logger = init_logger(__name__)

# Default configuration constants
DEFAULT_HEALTH_CHECK_INTERVAL_SEC = 5.0
DEFAULT_METRICS_TIMEOUT_SEC = 3.0
DEFAULT_UNHEALTHY_THRESHOLD = 3  # consecutive failures before marking unhealthy
DEFAULT_HEALTHY_THRESHOLD = 2  # consecutive successes before marking healthy


class EndpointStatus(str, Enum):
    """Health status of an endpoint."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class EndpointMetrics:
    """Metrics collected from a vLLM instance."""
    running_requests: int = 0
    waiting_requests: int = 0
    gpu_cache_usage: float = 0.0
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    timestamp: float = field(default_factory=time.monotonic)

    @property
    def total_pending_requests(self) -> int:
        """Total requests being processed or waiting."""
        return self.running_requests + self.waiting_requests

    @property
    def load_score(self) -> float:
        """Composite load score for routing decisions.

        Lower score = less loaded = preferred for routing.
        Weighting:
          - running_requests: weight 1.0 (active work)
          - waiting_requests: weight 2.0 (indicates backpressure)
          - gpu_cache_usage: weight 10.0 (memory pressure)
        """
        return (self.running_requests * 1.0 + self.waiting_requests * 2.0 +
                self.gpu_cache_usage * 10.0)


@dataclass
class EndpointState:
    """Tracks the state and metrics history of a single endpoint."""
    url: str
    status: EndpointStatus = EndpointStatus.UNKNOWN
    metrics: Optional[EndpointMetrics] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_check_time: float = 0.0
    last_error: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.status == EndpointStatus.HEALTHY


class EndpointPool:
    """Manages a pool of endpoints of the same role (Prefill or Decode)."""

    def __init__(self, endpoints: List[str], role: str = "unknown"):
        self.role = role
        self._endpoints: Dict[str, EndpointState] = {}
        for url in endpoints:
            normalized = url.rstrip("/")
            self._endpoints[normalized] = EndpointState(url=normalized)

    @property
    def all_endpoints(self) -> List[EndpointState]:
        return list(self._endpoints.values())

    def get_healthy_endpoints(self) -> List[EndpointState]:
        """Return all healthy endpoints sorted by load score (ascending)."""
        healthy = [
            ep for ep in self._endpoints.values() if ep.is_healthy
        ]
        # Sort by load_score; endpoints with no metrics yet get a default
        healthy.sort(
            key=lambda ep: ep.metrics.load_score
            if ep.metrics else float("inf"))
        return healthy

    def get_by_priority(self) -> List[EndpointState]:
        """Return endpoints sorted for fallback: healthy first, then others.

        Within each group, sort by load score.
        """
        healthy = []
        others = []
        for ep in self._endpoints.values():
            if ep.is_healthy:
                healthy.append(ep)
            else:
                others.append(ep)
        healthy.sort(
            key=lambda ep: ep.metrics.load_score
            if ep.metrics else float("inf"))
        others.sort(
            key=lambda ep: ep.metrics.load_score
            if ep.metrics else float("inf"))
        return healthy + others

    def get_least_loaded(self) -> Optional[EndpointState]:
        """Return the healthy endpoint with the lowest load score."""
        healthy = self.get_healthy_endpoints()
        return healthy[0] if healthy else None

    def mark_unhealthy(self, url: str, reason: str = "manual"):
        """Explicitly mark an endpoint as unhealthy."""
        normalized = url.rstrip("/")
        if normalized in self._endpoints:
            ep = self._endpoints[normalized]
            ep.status = EndpointStatus.UNHEALTHY
            ep.consecutive_successes = 0
            ep.last_error = reason
            logger.warning("Endpoint %s [%s] marked unhealthy: %s",
                           normalized, self.role, reason)

    def mark_healthy(self, url: str):
        """Explicitly mark an endpoint as healthy."""
        normalized = url.rstrip("/")
        if normalized in self._endpoints:
            ep = self._endpoints[normalized]
            ep.status = EndpointStatus.HEALTHY
            ep.consecutive_failures = 0
            ep.last_error = None

    def has_healthy_endpoint(self) -> bool:
        return any(ep.is_healthy for ep in self._endpoints.values())

    def __len__(self) -> int:
        return len(self._endpoints)

    def __repr__(self) -> str:
        healthy_count = sum(1 for ep in self._endpoints.values()
                           if ep.is_healthy)
        return (f"EndpointPool(role={self.role}, "
                f"total={len(self._endpoints)}, "
                f"healthy={healthy_count})")


class HealthMonitor:
    """Periodically collects metrics from Prefill/Decode instances.

    Parses the Prometheus text format exposed at /metrics by vLLM
    to extract running/waiting request counts and GPU cache usage.
    Uses these to maintain health status with hysteresis (consecutive
    success/failure thresholds).
    """

    def __init__(
        self,
        prefill_pool: EndpointPool,
        decode_pool: EndpointPool,
        check_interval: float = DEFAULT_HEALTH_CHECK_INTERVAL_SEC,
        metrics_timeout: float = DEFAULT_METRICS_TIMEOUT_SEC,
        unhealthy_threshold: int = DEFAULT_UNHEALTHY_THRESHOLD,
        healthy_threshold: int = DEFAULT_HEALTHY_THRESHOLD,
    ):
        self.prefill_pool = prefill_pool
        self.decode_pool = decode_pool
        self.check_interval = check_interval
        self.metrics_timeout = metrics_timeout
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold

        self._session: Optional[aiohttp.ClientSession] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self):
        """Start the background health monitoring loop."""
        if self._running:
            return
        self._running = True
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.metrics_timeout))
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(
            "HealthMonitor started: check_interval=%.1fs, "
            "prefill_endpoints=%d, decode_endpoints=%d",
            self.check_interval, len(self.prefill_pool),
            len(self.decode_pool))

    async def stop(self):
        """Stop the background health monitoring loop."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("HealthMonitor stopped")

    async def _monitor_loop(self):
        """Main monitoring loop that periodically checks all endpoints."""
        while self._running:
            try:
                await self._check_all_endpoints()
            except Exception:
                logger.exception("Error in health monitor loop")
            await asyncio.sleep(self.check_interval)

    async def _check_all_endpoints(self):
        """Check all endpoints in both pools concurrently."""
        all_endpoints = (self.prefill_pool.all_endpoints +
                         self.decode_pool.all_endpoints)
        tasks = [
            self._check_single_endpoint(ep) for ep in all_endpoints
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_single_endpoint(self, endpoint: EndpointState):
        """Check a single endpoint's health by fetching its metrics."""
        try:
            metrics = await self.collect_metrics(endpoint.url)
            endpoint.metrics = metrics
            endpoint.last_check_time = time.monotonic()
            endpoint.consecutive_failures = 0
            endpoint.consecutive_successes += 1

            # Transition to healthy after enough consecutive successes
            if (endpoint.status != EndpointStatus.HEALTHY
                    and endpoint.consecutive_successes
                    >= self.healthy_threshold):
                old_status = endpoint.status
                endpoint.status = EndpointStatus.HEALTHY
                endpoint.last_error = None
                logger.info(
                    "Endpoint %s recovered: %s -> HEALTHY "
                    "(running=%d, waiting=%d, gpu_cache=%.1f%%)",
                    endpoint.url, old_status.value,
                    metrics.running_requests, metrics.waiting_requests,
                    metrics.gpu_cache_usage * 100)
        except Exception as e:
            endpoint.consecutive_successes = 0
            endpoint.consecutive_failures += 1
            endpoint.last_error = str(e)

            # Transition to unhealthy after enough consecutive failures
            if (endpoint.consecutive_failures >= self.unhealthy_threshold
                    and endpoint.status != EndpointStatus.UNHEALTHY):
                old_status = endpoint.status
                endpoint.status = EndpointStatus.UNHEALTHY
                logger.warning(
                    "Endpoint %s marked UNHEALTHY: %s -> UNHEALTHY "
                    "(%d consecutive failures: %s)", endpoint.url,
                    old_status.value, endpoint.consecutive_failures, e)

    async def collect_metrics(self, endpoint_url: str) -> EndpointMetrics:
        """Fetch and parse Prometheus metrics from a vLLM instance.

        Parses the Prometheus text exposition format to extract:
        - vllm:num_requests_running
        - vllm:num_requests_waiting
        - vllm:gpu_cache_usage_perc
        - vllm:prompt_tokens_total
        - vllm:generation_tokens_total
        """
        assert self._session is not None, "HealthMonitor not started"

        metrics_url = f"{endpoint_url}/metrics"
        async with self._session.get(metrics_url) as response:
            if response.status != 200:
                raise ConnectionError(
                    f"Metrics endpoint returned status {response.status}")
            text = await response.text()

        return self._parse_prometheus_metrics(text)

    @staticmethod
    def _parse_prometheus_metrics(text: str) -> EndpointMetrics:
        """Parse Prometheus text format into EndpointMetrics.

        Example line format:
            vllm:num_requests_running{model_name="meta-llama/..."} 3.0
        """
        metrics = EndpointMetrics()

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                # Handle metrics with or without labels
                # Format: metric_name{labels} value  OR  metric_name value
                if "{" in line:
                    metric_name = line.split("{")[0]
                    value_str = line.rsplit("}", 1)[-1].strip()
                else:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    metric_name = parts[0]
                    value_str = parts[1]

                value = float(value_str)

                if metric_name == "vllm:num_requests_running":
                    metrics.running_requests = int(value)
                elif metric_name == "vllm:num_requests_waiting":
                    metrics.waiting_requests = int(value)
                elif metric_name == "vllm:gpu_cache_usage_perc":
                    metrics.gpu_cache_usage = value
                elif metric_name == "vllm:prompt_tokens_total":
                    metrics.prompt_tokens_total = value
                elif metric_name == "vllm:generation_tokens_total":
                    metrics.generation_tokens_total = value
            except (ValueError, IndexError):
                # Skip lines that can't be parsed
                continue

        metrics.timestamp = time.monotonic()
        return metrics

    def get_status_summary(self) -> Dict:
        """Return a summary of all endpoint statuses for debugging."""
        summary = {"prefill": [], "decode": []}
        for ep in self.prefill_pool.all_endpoints:
            summary["prefill"].append({
                "url": ep.url,
                "status": ep.status.value,
                "load_score": ep.metrics.load_score if ep.metrics else None,
                "running": ep.metrics.running_requests if ep.metrics else None,
                "waiting": ep.metrics.waiting_requests if ep.metrics else None,
                "gpu_cache":
                ep.metrics.gpu_cache_usage if ep.metrics else None,
                "last_error": ep.last_error,
            })
        for ep in self.decode_pool.all_endpoints:
            summary["decode"].append({
                "url": ep.url,
                "status": ep.status.value,
                "load_score": ep.metrics.load_score if ep.metrics else None,
                "running": ep.metrics.running_requests if ep.metrics else None,
                "waiting": ep.metrics.waiting_requests if ep.metrics else None,
                "gpu_cache":
                ep.metrics.gpu_cache_usage if ep.metrics else None,
                "last_error": ep.last_error,
            })
        return summary
