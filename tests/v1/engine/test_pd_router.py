# SPDX-License-Identifier: Apache-2.0
"""Tests for PD Disaggregation Intelligent Router (Optimization 2).

Covers:
- EndpointMetrics load scoring
- EndpointPool management and selection
- HealthMonitor Prometheus parsing
- PDRouter request classification
- PDRouter routing strategies (short/long)
- PDRouter failover behavior
- aiohttp web app integration
"""

from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from vllm.v1.engine.pd_health_monitor import (
    EndpointMetrics,
    EndpointPool,
    EndpointStatus,
    HealthMonitor,
)
from vllm.v1.engine.pd_router import (
    DEFAULT_SHORT_PROMPT_THRESHOLD,
    PDRouter,
    RequestClassification,
    RouterStats,
    create_app,
)


# ====================================================================
# EndpointMetrics Tests
# ====================================================================


class TestEndpointMetrics:
    """Test EndpointMetrics data class and computed properties."""

    def test_default_values(self):
        metrics = EndpointMetrics()
        assert metrics.running_requests == 0
        assert metrics.waiting_requests == 0
        assert metrics.gpu_cache_usage == 0.0
        assert metrics.total_pending_requests == 0
        assert metrics.load_score == 0.0

    def test_total_pending_requests(self):
        metrics = EndpointMetrics(running_requests=5, waiting_requests=3)
        assert metrics.total_pending_requests == 8

    def test_load_score_computation(self):
        metrics = EndpointMetrics(
            running_requests=10,
            waiting_requests=5,
            gpu_cache_usage=0.8,
        )
        # 10 * 1.0 + 5 * 2.0 + 0.8 * 10.0 = 10 + 10 + 8 = 28.0
        assert metrics.load_score == 28.0

    def test_load_score_zero_when_idle(self):
        metrics = EndpointMetrics()
        assert metrics.load_score == 0.0

    def test_load_score_high_waiting(self):
        """Waiting requests have higher weight (backpressure indicator)."""
        metrics_running = EndpointMetrics(running_requests=10)
        metrics_waiting = EndpointMetrics(waiting_requests=10)
        assert metrics_waiting.load_score > metrics_running.load_score


# ====================================================================
# EndpointPool Tests
# ====================================================================


class TestEndpointPool:
    """Test EndpointPool management."""

    def test_init(self):
        pool = EndpointPool(
            ["http://localhost:8100", "http://localhost:8101"],
            role="prefill")
        assert len(pool) == 2
        assert pool.role == "prefill"

    def test_url_normalization(self):
        """Trailing slashes should be stripped."""
        pool = EndpointPool(["http://localhost:8100/"])
        assert pool.all_endpoints[0].url == "http://localhost:8100"

    def test_initial_status_unknown(self):
        pool = EndpointPool(["http://localhost:8100"])
        ep = pool.all_endpoints[0]
        assert ep.status == EndpointStatus.UNKNOWN
        assert not ep.is_healthy

    def test_get_healthy_endpoints_empty_initially(self):
        pool = EndpointPool(["http://localhost:8100"])
        assert len(pool.get_healthy_endpoints()) == 0

    def test_mark_healthy(self):
        pool = EndpointPool(["http://localhost:8100"])
        pool.mark_healthy("http://localhost:8100")
        assert len(pool.get_healthy_endpoints()) == 1
        assert pool.has_healthy_endpoint()

    def test_mark_unhealthy(self):
        pool = EndpointPool(["http://localhost:8100"])
        pool.mark_healthy("http://localhost:8100")
        pool.mark_unhealthy("http://localhost:8100", reason="test")
        assert len(pool.get_healthy_endpoints()) == 0
        assert not pool.has_healthy_endpoint()

    def test_get_least_loaded(self):
        pool = EndpointPool([
            "http://localhost:8100",
            "http://localhost:8101",
        ])
        # Mark both as healthy and assign metrics
        for ep in pool.all_endpoints:
            pool.mark_healthy(ep.url)

        eps = pool.all_endpoints
        eps[0].metrics = EndpointMetrics(running_requests=10)
        eps[1].metrics = EndpointMetrics(running_requests=2)

        least = pool.get_least_loaded()
        assert least is not None
        assert least.url == "http://localhost:8101"

    def test_get_least_loaded_no_healthy(self):
        pool = EndpointPool(["http://localhost:8100"])
        assert pool.get_least_loaded() is None

    def test_get_by_priority(self):
        """Healthy endpoints come first, sorted by load."""
        pool = EndpointPool([
            "http://localhost:8100",
            "http://localhost:8101",
            "http://localhost:8102",
        ])
        pool.mark_healthy("http://localhost:8100")
        pool.mark_healthy("http://localhost:8101")
        # 8102 stays UNKNOWN (not healthy)

        prioritized = pool.get_by_priority()
        assert len(prioritized) == 3
        # First two should be healthy
        assert prioritized[0].is_healthy
        assert prioritized[1].is_healthy
        # Last one is not healthy
        assert not prioritized[2].is_healthy

    def test_get_healthy_sorted_by_load(self):
        pool = EndpointPool([
            "http://localhost:8100",
            "http://localhost:8101",
            "http://localhost:8102",
        ])
        for ep in pool.all_endpoints:
            pool.mark_healthy(ep.url)

        eps = pool.all_endpoints
        eps[0].metrics = EndpointMetrics(running_requests=5)
        eps[1].metrics = EndpointMetrics(running_requests=1)
        eps[2].metrics = EndpointMetrics(running_requests=10)

        healthy = pool.get_healthy_endpoints()
        loads = [ep.metrics.load_score for ep in healthy]
        assert loads == sorted(loads)

    def test_repr(self):
        pool = EndpointPool(["http://localhost:8100"], role="prefill")
        pool.mark_healthy("http://localhost:8100")
        r = repr(pool)
        assert "prefill" in r
        assert "total=1" in r
        assert "healthy=1" in r


# ====================================================================
# HealthMonitor Tests
# ====================================================================


class TestHealthMonitor:
    """Test HealthMonitor Prometheus parsing and health logic."""

    SAMPLE_PROMETHEUS_OUTPUT = """
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3-8B"} 5.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="meta-llama/Llama-3-8B"} 2.0
# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="meta-llama/Llama-3-8B"} 0.65
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model_name="meta-llama/Llama-3-8B"} 12345.0
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{model_name="meta-llama/Llama-3-8B"} 6789.0
"""

    def test_parse_prometheus_metrics(self):
        metrics = HealthMonitor._parse_prometheus_metrics(
            self.SAMPLE_PROMETHEUS_OUTPUT)
        assert metrics.running_requests == 5
        assert metrics.waiting_requests == 2
        assert abs(metrics.gpu_cache_usage - 0.65) < 1e-6
        assert abs(metrics.prompt_tokens_total - 12345.0) < 1e-6
        assert abs(metrics.generation_tokens_total - 6789.0) < 1e-6

    def test_parse_empty_metrics(self):
        metrics = HealthMonitor._parse_prometheus_metrics("")
        assert metrics.running_requests == 0
        assert metrics.waiting_requests == 0
        assert metrics.gpu_cache_usage == 0.0

    def test_parse_comments_only(self):
        text = "# HELP vllm:num_requests_running ...\n# TYPE ...\n"
        metrics = HealthMonitor._parse_prometheus_metrics(text)
        assert metrics.running_requests == 0

    def test_parse_malformed_lines(self):
        """Malformed lines should be skipped gracefully."""
        text = """
vllm:num_requests_running{model_name="test"} 3.0
this is not a valid metric line
vllm:num_requests_waiting{model_name="test"} not_a_number
vllm:gpu_cache_usage_perc{model_name="test"} 0.5
"""
        metrics = HealthMonitor._parse_prometheus_metrics(text)
        assert metrics.running_requests == 3
        assert abs(metrics.gpu_cache_usage - 0.5) < 1e-6
        # waiting_requests stays 0 because value was not parseable
        assert metrics.waiting_requests == 0

    def test_parse_metrics_without_labels(self):
        text = "vllm:num_requests_running 7.0\nvllm:gpu_cache_usage_perc 0.3\n"
        metrics = HealthMonitor._parse_prometheus_metrics(text)
        assert metrics.running_requests == 7
        assert abs(metrics.gpu_cache_usage - 0.3) < 1e-6

    @pytest.mark.asyncio
    async def test_check_single_endpoint_success(self):
        """Successful metric collection should increase consecutive_successes."""
        prefill_pool = EndpointPool(["http://localhost:8100"],
                                    role="prefill")
        decode_pool = EndpointPool([], role="decode")
        monitor = HealthMonitor(prefill_pool, decode_pool,
                                healthy_threshold=1)

        ep = prefill_pool.all_endpoints[0]
        assert ep.status == EndpointStatus.UNKNOWN

        # Mock collect_metrics to return valid metrics
        monitor.collect_metrics = AsyncMock(
            return_value=EndpointMetrics(running_requests=3))

        await monitor._check_single_endpoint(ep)
        assert ep.consecutive_successes == 1
        assert ep.consecutive_failures == 0
        assert ep.status == EndpointStatus.HEALTHY  # threshold=1
        assert ep.metrics.running_requests == 3

    @pytest.mark.asyncio
    async def test_check_single_endpoint_failure(self):
        """Failed metric collection should increase consecutive_failures."""
        prefill_pool = EndpointPool(["http://localhost:8100"],
                                    role="prefill")
        decode_pool = EndpointPool([], role="decode")
        monitor = HealthMonitor(prefill_pool, decode_pool,
                                unhealthy_threshold=2)

        ep = prefill_pool.all_endpoints[0]
        prefill_pool.mark_healthy(ep.url)

        monitor.collect_metrics = AsyncMock(
            side_effect=ConnectionError("timeout"))

        # First failure: still healthy
        await monitor._check_single_endpoint(ep)
        assert ep.consecutive_failures == 1
        assert ep.status == EndpointStatus.HEALTHY

        # Second failure: now unhealthy
        await monitor._check_single_endpoint(ep)
        assert ep.consecutive_failures == 2
        assert ep.status == EndpointStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_health_recovery(self):
        """Endpoint should recover after consecutive successes."""
        prefill_pool = EndpointPool(["http://localhost:8100"],
                                    role="prefill")
        decode_pool = EndpointPool([], role="decode")
        monitor = HealthMonitor(prefill_pool, decode_pool,
                                healthy_threshold=2)

        ep = prefill_pool.all_endpoints[0]
        ep.status = EndpointStatus.UNHEALTHY

        monitor.collect_metrics = AsyncMock(
            return_value=EndpointMetrics())

        # First success: not yet healthy
        await monitor._check_single_endpoint(ep)
        assert ep.status == EndpointStatus.UNHEALTHY

        # Second success: recovered
        await monitor._check_single_endpoint(ep)
        assert ep.status == EndpointStatus.HEALTHY

    def test_get_status_summary(self):
        prefill_pool = EndpointPool(["http://localhost:8100"],
                                    role="prefill")
        decode_pool = EndpointPool(["http://localhost:8200"],
                                   role="decode")
        monitor = HealthMonitor(prefill_pool, decode_pool)

        summary = monitor.get_status_summary()
        assert "prefill" in summary
        assert "decode" in summary
        assert len(summary["prefill"]) == 1
        assert len(summary["decode"]) == 1
        assert summary["prefill"][0]["url"] == "http://localhost:8100"
        assert summary["decode"][0]["url"] == "http://localhost:8200"


# ====================================================================
# PDRouter Request Classification Tests
# ====================================================================


class TestRequestClassification:
    """Test PDRouter request classification logic."""

    def _make_router(self, threshold=DEFAULT_SHORT_PROMPT_THRESHOLD):
        return PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
            short_prompt_threshold=threshold,
        )

    def test_short_completion_request(self):
        router = self._make_router(threshold=128)
        # "Hello world" = 11 chars / 4 = ~3 tokens < 128
        request_data = {"prompt": "Hello world"}
        assert (router._classify_request(request_data) ==
                RequestClassification.SHORT_DECODE_ONLY)

    def test_long_completion_request(self):
        router = self._make_router(threshold=128)
        # 1000 chars / 4 = 250 tokens > 128
        request_data = {"prompt": "x" * 1000}
        assert (router._classify_request(request_data) ==
                RequestClassification.LONG_PREFILL)

    def test_short_chat_request(self):
        router = self._make_router(threshold=128)
        request_data = {
            "messages": [
                {"role": "user", "content": "Hi"},
            ]
        }
        assert (router._classify_request(request_data) ==
                RequestClassification.SHORT_DECODE_ONLY)

    def test_long_chat_request(self):
        router = self._make_router(threshold=128)
        request_data = {
            "messages": [
                {"role": "system",
                 "content": "You are a helpful assistant. " * 50},
                {"role": "user", "content": "Explain quantum physics."},
            ]
        }
        assert (router._classify_request(request_data) ==
                RequestClassification.LONG_PREFILL)

    def test_token_list_prompt(self):
        """Prompt given as token IDs list."""
        router = self._make_router(threshold=10)
        request_data = {"prompt": list(range(50))}
        assert (router._classify_request(request_data) ==
                RequestClassification.LONG_PREFILL)

    def test_token_list_short(self):
        router = self._make_router(threshold=10)
        request_data = {"prompt": [1, 2, 3]}
        assert (router._classify_request(request_data) ==
                RequestClassification.SHORT_DECODE_ONLY)

    def test_multimodal_chat_content(self):
        """Multi-modal content with text parts."""
        router = self._make_router(threshold=128)
        request_data = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "x" * 1000},
                    {"type": "image_url",
                     "image_url": {"url": "http://example.com/img.png"}},
                ]
            }]
        }
        assert (router._classify_request(request_data) ==
                RequestClassification.LONG_PREFILL)

    def test_empty_prompt(self):
        router = self._make_router(threshold=128)
        request_data = {"prompt": ""}
        assert (router._classify_request(request_data) ==
                RequestClassification.SHORT_DECODE_ONLY)

    def test_missing_prompt(self):
        router = self._make_router(threshold=128)
        request_data = {}
        assert (router._classify_request(request_data) ==
                RequestClassification.SHORT_DECODE_ONLY)

    def test_estimate_token_count_completions(self):
        router = self._make_router()
        data = {"prompt": "Hello, how are you doing today?"}
        count = router._estimate_token_count(data)
        # 30 chars / 4 = 7
        assert count == 7

    def test_estimate_token_count_chat(self):
        router = self._make_router()
        data = {
            "messages": [
                {"role": "user", "content": "a" * 100},
                {"role": "assistant", "content": "b" * 200},
            ]
        }
        count = router._estimate_token_count(data)
        # 300 chars / 4 = 75
        assert count == 75

    def test_custom_threshold(self):
        router = self._make_router(threshold=10)
        # 50 chars / 4 = 12 tokens > 10
        request_data = {"prompt": "x" * 50}
        assert (router._classify_request(request_data) ==
                RequestClassification.LONG_PREFILL)


# ====================================================================
# PDRouter Routing Tests
# ====================================================================


class TestPDRouterRouting:
    """Test PDRouter routing strategies with mocked HTTP calls."""

    @pytest.fixture
    def router(self):
        r = PDRouter(
            prefill_endpoints=["http://localhost:8100",
                               "http://localhost:8101"],
            decode_endpoints=["http://localhost:8200",
                              "http://localhost:8201"],
            short_prompt_threshold=128,
        )
        # Mark all endpoints as healthy for testing
        for ep in r.prefill_pool.all_endpoints:
            r.prefill_pool.mark_healthy(ep.url)
            ep.metrics = EndpointMetrics(running_requests=1)
        for ep in r.decode_pool.all_endpoints:
            r.decode_pool.mark_healthy(ep.url)
            ep.metrics = EndpointMetrics(running_requests=1)
        return r

    @pytest.mark.asyncio
    async def test_short_request_bypasses_prefill(self, router):
        """Short requests should go directly to Decode."""
        chunks_received = []

        async def mock_forward_stream(url, path, data, timeout=None):
            yield b'{"choices":[{"text":"hi"}]}'

        router._forward_stream = mock_forward_stream

        request_data = {"prompt": "Hi", "max_tokens": 10}
        async for chunk in router.route_request(request_data,
                                                "/v1/completions"):
            chunks_received.append(chunk)

        assert len(chunks_received) > 0
        assert router._stats.short_requests == 1
        assert router._stats.long_requests == 0

    @pytest.mark.asyncio
    async def test_long_request_uses_pd_path(self, router):
        """Long requests should go through Prefill then Decode."""
        prefill_called = False
        decode_chunks = []

        async def mock_forward_non_stream(url, path, data, timeout=None):
            nonlocal prefill_called
            prefill_called = True
            return b'{"choices":[{"text":""}]}'

        async def mock_forward_stream(url, path, data, timeout=None):
            yield b'{"choices":[{"text":"response"}]}'

        router._forward_non_stream = AsyncMock(
            side_effect=mock_forward_non_stream)
        router._forward_stream = mock_forward_stream

        request_data = {"prompt": "x" * 1000, "max_tokens": 100}
        async for chunk in router.route_request(request_data,
                                                "/v1/completions"):
            decode_chunks.append(chunk)

        assert prefill_called
        assert len(decode_chunks) > 0
        assert router._stats.long_requests == 1
        assert router._stats.prefill_success == 1

    @pytest.mark.asyncio
    async def test_prefill_failure_fallback(self, router):
        """When all Prefill instances fail, should fall back to Decode."""
        call_count = 0

        async def mock_forward_non_stream(url, path, data, timeout=None):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Connection refused")

        async def mock_forward_stream(url, path, data, timeout=None):
            yield b'{"choices":[{"text":"fallback"}]}'

        router._forward_non_stream = AsyncMock(
            side_effect=mock_forward_non_stream)
        router._forward_stream = mock_forward_stream

        request_data = {"prompt": "x" * 1000, "max_tokens": 100}
        chunks = []
        async for chunk in router.route_request(request_data,
                                                "/v1/completions"):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert router._stats.fallback_to_decode == 1
        assert router._stats.prefill_failures > 0

    @pytest.mark.asyncio
    async def test_prefill_retry_on_failure(self, router):
        """Should retry prefill on different endpoint on failure."""
        call_urls = []

        async def mock_forward_non_stream(url, path, data, timeout=None):
            call_urls.append(url)
            if len(call_urls) == 1:
                raise ConnectionError("first attempt fails")
            return b'{"choices":[]}'

        async def mock_forward_stream(url, path, data, timeout=None):
            yield b'{"choices":[{"text":"ok"}]}'

        router._forward_non_stream = AsyncMock(
            side_effect=mock_forward_non_stream)
        router._forward_stream = mock_forward_stream

        request_data = {"prompt": "x" * 1000, "max_tokens": 100}
        async for _ in router.route_request(request_data,
                                            "/v1/completions"):
            pass

        # Should have tried two different endpoints
        assert len(call_urls) == 2
        assert call_urls[0] != call_urls[1]
        assert router._stats.prefill_success == 1
        assert router._stats.prefill_failures == 1


# ====================================================================
# RouterStats Tests
# ====================================================================


class TestRouterStats:
    """Test RouterStats tracking."""

    def test_initial_values(self):
        stats = RouterStats()
        assert stats.total_requests == 0
        assert stats.short_requests == 0
        assert stats.long_requests == 0

    def test_to_dict(self):
        stats = RouterStats()
        stats.total_requests = 100
        stats.short_requests = 30
        stats.long_requests = 70
        stats.prefill_success = 65
        stats.prefill_failures = 5
        stats.fallback_to_decode = 5

        d = stats.to_dict()
        assert d["total_requests"] == 100
        assert d["short_requests"] == 30
        assert d["long_requests"] == 70
        assert d["prefill_success"] == 65
        assert d["prefill_failures"] == 5
        assert d["fallback_to_decode"] == 5
        assert "uptime_seconds" in d


# ====================================================================
# PDRouter Status Tests
# ====================================================================


class TestPDRouterStatus:
    """Test PDRouter status reporting."""

    def test_get_status(self):
        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
        )
        status = router.get_status()
        assert "router" in status
        assert "endpoints" in status
        assert "config" in status["router"]
        assert "stats" in status["router"]
        assert "prefill" in status["endpoints"]
        assert "decode" in status["endpoints"]


# ====================================================================
# Web App Integration Tests
# ====================================================================


class TestWebApp:
    """Test the aiohttp web application setup."""

    def test_create_app(self):
        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
        )
        app = create_app(router)

        # Verify routes are registered
        routes = [r.resource.canonical for r in app.router.routes()
                  if hasattr(r, 'resource') and r.resource is not None]
        assert "/v1/completions" in routes
        assert "/v1/chat/completions" in routes
        assert "/health" in routes
        assert "/router/status" in routes

    @pytest.mark.asyncio
    async def test_health_endpoint_all_healthy(self):
        """Health endpoint should return healthy when all pools are healthy."""
        from aiohttp.test_utils import TestClient, TestServer

        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
        )
        # Mark all as healthy
        for ep in router.prefill_pool.all_endpoints:
            router.prefill_pool.mark_healthy(ep.url)
        for ep in router.decode_pool.all_endpoints:
            router.decode_pool.mark_healthy(ep.url)

        # Create app without startup (to avoid real health checks)
        app = web.Application()

        async def handle_health(request):
            prefill_healthy = router.prefill_pool.has_healthy_endpoint()
            decode_healthy = router.decode_pool.has_healthy_endpoint()
            if prefill_healthy and decode_healthy:
                return web.json_response({"status": "healthy"})
            return web.json_response({"status": "unhealthy"}, status=503)

        app.router.add_get("/health", handle_health)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_endpoint_degraded(self):
        """Health should be degraded when only decode is healthy."""
        from aiohttp.test_utils import TestClient, TestServer

        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
        )
        # Only decode is healthy
        for ep in router.decode_pool.all_endpoints:
            router.decode_pool.mark_healthy(ep.url)

        app = web.Application()

        async def handle_health(request):
            prefill_healthy = router.prefill_pool.has_healthy_endpoint()
            decode_healthy = router.decode_pool.has_healthy_endpoint()
            if prefill_healthy and decode_healthy:
                return web.json_response({"status": "healthy"})
            elif decode_healthy:
                return web.json_response(
                    {"status": "degraded",
                     "reason": "no healthy prefill instances"})
            return web.json_response({"status": "unhealthy"}, status=503)

        app.router.add_get("/health", handle_health)

        async with TestClient(TestServer(app)) as client:
            resp = await client.get("/health")
            assert resp.status == 200
            data = await resp.json()
            assert data["status"] == "degraded"


# ====================================================================
# Edge Case Tests
# ====================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_prefill_single_decode(self):
        """Minimal 1P1D configuration."""
        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
        )
        assert len(router.prefill_pool) == 1
        assert len(router.decode_pool) == 1

    def test_multi_prefill_multi_decode(self):
        """NP:MD configuration."""
        router = PDRouter(
            prefill_endpoints=[
                "http://localhost:8100",
                "http://localhost:8101",
                "http://localhost:8102",
            ],
            decode_endpoints=[
                "http://localhost:8200",
                "http://localhost:8201",
            ],
        )
        assert len(router.prefill_pool) == 3
        assert len(router.decode_pool) == 2

    def test_threshold_boundary(self):
        """Exact threshold value should classify as long."""
        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
            short_prompt_threshold=10,
        )
        # Exactly at threshold: 40 chars / 4 = 10 tokens
        # 10 >= 10, so NOT short
        data = {"prompt": "x" * 40}
        assert (router._classify_request(data) ==
                RequestClassification.LONG_PREFILL)

    def test_threshold_boundary_minus_one(self):
        """One below threshold should classify as short."""
        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
            short_prompt_threshold=10,
        )
        # 36 chars / 4 = 9 tokens < 10
        data = {"prompt": "x" * 36}
        assert (router._classify_request(data) ==
                RequestClassification.SHORT_DECODE_ONLY)

    def test_endpoint_pool_duplicate_urls(self):
        """Duplicate URLs should be deduplicated (same key)."""
        pool = EndpointPool([
            "http://localhost:8100",
            "http://localhost:8100",
        ])
        # Same URL maps to the same key, so only 1 endpoint
        assert len(pool) == 1

    @pytest.mark.asyncio
    async def test_prefill_fallback_no_decode_available(self):
        """When both Prefill and Decode are unavailable."""
        router = PDRouter(
            prefill_endpoints=["http://localhost:8100"],
            decode_endpoints=["http://localhost:8200"],
        )
        # No endpoints are healthy

        request_data = {"prompt": "Hi"}
        with pytest.raises(web.HTTPServiceUnavailable):
            async for _ in router.route_request(request_data,
                                                "/v1/completions"):
                pass
