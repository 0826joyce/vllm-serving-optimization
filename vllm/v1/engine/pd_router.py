# SPDX-License-Identifier: Apache-2.0
"""PD Disaggregation - Intelligent Request Router / Proxy.

Replaces the simple HTTP proxy with an intelligent router that supports:
- Load-aware routing: routes to the least loaded instance
- Request classification: short prompts bypass PD separation
- Failover: automatic fallback when instances are unavailable
- NP:MD deployment: multiple Prefill + multiple Decode instances
- Both Chat and Completions API support
"""

import argparse
import asyncio
import os
import time
from typing import AsyncGenerator, List, Optional

import aiohttp
from aiohttp import web

from vllm.logger import init_logger
from vllm.v1.engine.pd_health_monitor import (
    EndpointPool,
    EndpointState,
    HealthMonitor,
)

logger = init_logger(__name__)

# Default configuration
DEFAULT_SHORT_PROMPT_THRESHOLD = 128  # tokens
DEFAULT_PROXY_PORT = 8000
DEFAULT_REQUEST_TIMEOUT_SEC = 6 * 60 * 60  # 6 hours
DEFAULT_PREFILL_TIMEOUT_SEC = 5 * 60  # 5 minutes for prefill phase
DEFAULT_MAX_RETRIES = 2
# Rough estimate: ~4 chars per token for English, ~2 chars for Chinese
DEFAULT_CHARS_PER_TOKEN = 4


class RequestClassification:
    """Classification result for a request."""
    SHORT_DECODE_ONLY = "short_decode_only"
    LONG_PREFILL = "long_prefill"


class PDRouter:
    """Intelligent PD (Prefill-Decode) Router.

    Routes incoming OpenAI-compatible API requests to appropriate
    Prefill and Decode instances based on load, request characteristics,
    and instance health.

    Architecture:
        Client -> PDRouter -> [Prefill Instance(s)] -> KV Transfer -> [Decode Instance(s)] -> Client
                           -> [Decode Instance(s)] (for short requests, bypass PD)

    Supports:
        - /v1/completions (OpenAI Completions API)
        - /v1/chat/completions (OpenAI Chat API)
        - /health (router health check)
        - /router/status (router status & metrics)
    """

    def __init__(
        self,
        prefill_endpoints: List[str],
        decode_endpoints: List[str],
        short_prompt_threshold: int = DEFAULT_SHORT_PROMPT_THRESHOLD,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SEC,
        prefill_timeout: float = DEFAULT_PREFILL_TIMEOUT_SEC,
        max_retries: int = DEFAULT_MAX_RETRIES,
        health_check_interval: float = 5.0,
        chars_per_token: int = DEFAULT_CHARS_PER_TOKEN,
    ):
        # Initialize endpoint pools
        self.prefill_pool = EndpointPool(prefill_endpoints, role="prefill")
        self.decode_pool = EndpointPool(decode_endpoints, role="decode")

        # Initialize health monitor
        self.health_monitor = HealthMonitor(
            self.prefill_pool,
            self.decode_pool,
            check_interval=health_check_interval,
        )

        # Configuration
        self.short_prompt_threshold = short_prompt_threshold
        self.request_timeout = aiohttp.ClientTimeout(total=request_timeout)
        self.prefill_timeout = aiohttp.ClientTimeout(total=prefill_timeout)
        self.max_retries = max_retries
        self.chars_per_token = chars_per_token

        # Session (initialized in start())
        self._session: Optional[aiohttp.ClientSession] = None

        # Statistics
        self._stats = RouterStats()

        logger.info(
            "PDRouter initialized: prefill_endpoints=%s, "
            "decode_endpoints=%s, short_threshold=%d tokens",
            prefill_endpoints, decode_endpoints, short_prompt_threshold)

    async def start(self):
        """Start the router and health monitor."""
        self._session = aiohttp.ClientSession()
        await self.health_monitor.start()

        # Wait a brief moment for initial health checks
        await asyncio.sleep(1.0)

        # If no endpoints are healthy yet after initial check,
        # optimistically mark them all healthy to avoid cold-start block
        if not self.prefill_pool.has_healthy_endpoint():
            logger.warning(
                "No prefill endpoints healthy after initial check, "
                "marking all as healthy for cold start")
            for ep in self.prefill_pool.all_endpoints:
                self.prefill_pool.mark_healthy(ep.url)
        if not self.decode_pool.has_healthy_endpoint():
            logger.warning(
                "No decode endpoints healthy after initial check, "
                "marking all as healthy for cold start")
            for ep in self.decode_pool.all_endpoints:
                self.decode_pool.mark_healthy(ep.url)

        logger.info("PDRouter started")

    async def stop(self):
        """Stop the router and clean up resources."""
        await self.health_monitor.stop()
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("PDRouter stopped")

    # ----------------------------------------------------------------
    # Request Classification
    # ----------------------------------------------------------------

    def _classify_request(self, request_data: dict) -> str:
        """Classify request to determine routing strategy.

        Short requests (< short_prompt_threshold tokens) are sent
        directly to a Decode instance to avoid KV transfer overhead.
        Long requests go through the full PD separation path.
        """
        prompt_tokens = self._estimate_token_count(request_data)
        if prompt_tokens < self.short_prompt_threshold:
            return RequestClassification.SHORT_DECODE_ONLY
        return RequestClassification.LONG_PREFILL

    def _estimate_token_count(self, request_data: dict) -> int:
        """Estimate the number of prompt tokens without a tokenizer.

        For /v1/completions: uses the 'prompt' field.
        For /v1/chat/completions: concatenates all message contents.

        Uses a rough heuristic of chars_per_token characters per token.
        """
        # Chat API format
        messages = request_data.get("messages")
        if messages and isinstance(messages, list):
            total_chars = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    # Multi-modal content (list of parts)
                    for part in content:
                        if isinstance(part, dict):
                            text = part.get("text", "")
                            if text:
                                total_chars += len(text)
            return max(1, total_chars // self.chars_per_token)

        # Completions API format
        prompt = request_data.get("prompt", "")
        if isinstance(prompt, str):
            return max(1, len(prompt) // self.chars_per_token)
        elif isinstance(prompt, list):
            # Token IDs already provided
            return len(prompt)

        return 0

    # ----------------------------------------------------------------
    # Endpoint Selection
    # ----------------------------------------------------------------

    def _select_prefill_endpoint(self) -> Optional[EndpointState]:
        """Select the least loaded healthy Prefill endpoint."""
        return self.prefill_pool.get_least_loaded()

    def _select_decode_endpoint(self) -> Optional[EndpointState]:
        """Select the least loaded healthy Decode endpoint."""
        return self.decode_pool.get_least_loaded()

    # ----------------------------------------------------------------
    # Request Forwarding
    # ----------------------------------------------------------------

    async def _forward_stream(
        self,
        endpoint_url: str,
        path: str,
        request_data: dict,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Forward a request and stream the response back.

        Yields response chunks as they arrive.
        """
        assert self._session is not None, "Router not started"

        url = f"{endpoint_url}{path}"
        headers = {}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        effective_timeout = timeout or self.request_timeout

        async with self._session.post(
                url,
                json=request_data,
                headers=headers,
                timeout=effective_timeout,
        ) as response:
            if response.status != 200:
                error_body = await response.text()
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"Upstream error: {error_body}",
                )
            async for chunk in response.content.iter_any():
                if chunk:
                    yield chunk

    async def _forward_non_stream(
        self,
        endpoint_url: str,
        path: str,
        request_data: dict,
        timeout: Optional[aiohttp.ClientTimeout] = None,
    ) -> bytes:
        """Forward a request and return the complete response."""
        assert self._session is not None, "Router not started"

        url = f"{endpoint_url}{path}"
        headers = {}
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        effective_timeout = timeout or self.request_timeout

        async with self._session.post(
                url,
                json=request_data,
                headers=headers,
                timeout=effective_timeout,
        ) as response:
            if response.status != 200:
                error_body = await response.text()
                raise aiohttp.ClientResponseError(
                    response.request_info,
                    response.history,
                    status=response.status,
                    message=f"Upstream error: {error_body}",
                )
            return await response.read()

    # ----------------------------------------------------------------
    # Routing Strategies
    # ----------------------------------------------------------------

    async def _route_short_request(
        self,
        request_data: dict,
        path: str,
    ) -> AsyncGenerator[bytes, None]:
        """Route a short request directly to a Decode instance.

        Short requests bypass PD separation because the KV transfer
        overhead exceeds the Prefill computation savings.
        """
        decode_ep = self._select_decode_endpoint()
        if decode_ep is None:
            raise web.HTTPServiceUnavailable(
                text="No healthy Decode instances available")

        self._stats.short_requests += 1
        logger.debug("Short request routed directly to decode: %s",
                     decode_ep.url)

        try:
            async for chunk in self._forward_stream(
                    decode_ep.url, path, request_data):
                yield chunk
        except Exception as e:
            self.prefill_pool.mark_unhealthy(decode_ep.url, str(e))
            raise

    async def _route_long_request(
        self,
        request_data: dict,
        path: str,
    ) -> AsyncGenerator[bytes, None]:
        """Route a long request through the PD separation path.

        Phase 1: Send to Prefill instance with max_tokens=1
                 (triggers KV computation and transfer)
        Phase 2: Send original request to Decode instance
                 (uses transferred KV cache for decoding)
        """
        self._stats.long_requests += 1

        # Phase 1: Prefill with fallback
        prefill_success = await self._prefill_with_fallback(
            request_data, path)

        if not prefill_success:
            # All Prefill instances failed, fall back to Decode-only
            logger.warning(
                "All prefill instances unavailable, "
                "falling back to decode-side prefill")
            self._stats.fallback_to_decode += 1
            async for chunk in self._route_short_request(
                    request_data, path):
                yield chunk
            return

        # Phase 2: Decode (stream response back to client)
        decode_ep = self._select_decode_endpoint()
        if decode_ep is None:
            raise web.HTTPServiceUnavailable(
                text="No healthy Decode instances available")

        logger.debug("Decode phase routed to: %s", decode_ep.url)

        try:
            async for chunk in self._forward_stream(
                    decode_ep.url, path, request_data):
                yield chunk
        except Exception as e:
            self.decode_pool.mark_unhealthy(decode_ep.url, str(e))
            raise

    async def _prefill_with_fallback(
        self,
        request_data: dict,
        path: str,
    ) -> bool:
        """Execute Prefill phase with automatic failover.

        Tries each healthy Prefill endpoint in order of load.
        If all fail, returns False so the caller can fall back
        to decode-side prefill.
        """
        prefill_request = request_data.copy()
        # Set max_tokens=1 to trigger Prefill-only computation
        prefill_request["max_tokens"] = 1
        # Disable streaming for prefill phase (we discard the output)
        prefill_request["stream"] = False

        tried_endpoints = set()

        for attempt in range(self.max_retries + 1):
            candidates = self.prefill_pool.get_by_priority()
            prefill_ep = None
            for candidate in candidates:
                if candidate.url not in tried_endpoints:
                    prefill_ep = candidate
                    break

            if prefill_ep is None:
                logger.warning(
                    "No more prefill endpoints to try "
                    "(tried %d)", len(tried_endpoints))
                return False

            tried_endpoints.add(prefill_ep.url)
            logger.debug(
                "Prefill attempt %d/%d -> %s", attempt + 1,
                self.max_retries + 1, prefill_ep.url)

            try:
                start_time = time.monotonic()
                await self._forward_non_stream(
                    prefill_ep.url,
                    path,
                    prefill_request,
                    timeout=self.prefill_timeout,
                )
                elapsed = time.monotonic() - start_time
                logger.debug(
                    "Prefill completed on %s in %.2fs",
                    prefill_ep.url, elapsed)
                self._stats.prefill_success += 1
                return True

            except Exception as e:
                logger.warning(
                    "Prefill failed on %s (attempt %d/%d): %s",
                    prefill_ep.url, attempt + 1,
                    self.max_retries + 1, e)
                self.prefill_pool.mark_unhealthy(prefill_ep.url, str(e))
                self._stats.prefill_failures += 1
                continue

        return False

    # ----------------------------------------------------------------
    # Main Request Handler
    # ----------------------------------------------------------------

    async def route_request(
        self,
        request_data: dict,
        path: str,
    ) -> AsyncGenerator[bytes, None]:
        """Main entry point: classify and route a request.

        Args:
            request_data: The parsed JSON request body.
            path: The API path (e.g., /v1/completions).

        Yields:
            Response chunks (bytes).
        """
        self._stats.total_requests += 1

        request_type = self._classify_request(request_data)
        logger.debug(
            "Request classified as %s (estimated tokens: %d)",
            request_type,
            self._estimate_token_count(request_data))

        if request_type == RequestClassification.SHORT_DECODE_ONLY:
            async for chunk in self._route_short_request(
                    request_data, path):
                yield chunk
        else:
            async for chunk in self._route_long_request(
                    request_data, path):
                yield chunk

    # ----------------------------------------------------------------
    # Status & Health
    # ----------------------------------------------------------------

    def get_status(self) -> dict:
        """Return router status for the /router/status endpoint."""
        return {
            "router": {
                "config": {
                    "short_prompt_threshold":
                    self.short_prompt_threshold,
                    "max_retries": self.max_retries,
                },
                "stats": self._stats.to_dict(),
            },
            "endpoints": self.health_monitor.get_status_summary(),
        }


class RouterStats:
    """Tracks routing statistics."""

    def __init__(self):
        self.total_requests: int = 0
        self.short_requests: int = 0
        self.long_requests: int = 0
        self.prefill_success: int = 0
        self.prefill_failures: int = 0
        self.fallback_to_decode: int = 0
        self.start_time: float = time.monotonic()

    def to_dict(self) -> dict:
        uptime = time.monotonic() - self.start_time
        return {
            "total_requests": self.total_requests,
            "short_requests": self.short_requests,
            "long_requests": self.long_requests,
            "prefill_success": self.prefill_success,
            "prefill_failures": self.prefill_failures,
            "fallback_to_decode": self.fallback_to_decode,
            "uptime_seconds": round(uptime, 1),
        }


# ====================================================================
# aiohttp Web Application
# ====================================================================


def create_app(router: PDRouter) -> web.Application:
    """Create the aiohttp web application with all routes."""

    async def handle_completions(request: web.Request) -> web.StreamResponse:
        """Handle /v1/completions requests."""
        return await _handle_api_request(request, router,
                                         "/v1/completions")

    async def handle_chat_completions(
            request: web.Request) -> web.StreamResponse:
        """Handle /v1/chat/completions requests."""
        return await _handle_api_request(request, router,
                                         "/v1/chat/completions")

    async def handle_health(request: web.Request) -> web.Response:
        """Health check endpoint."""
        prefill_healthy = router.prefill_pool.has_healthy_endpoint()
        decode_healthy = router.decode_pool.has_healthy_endpoint()

        if prefill_healthy and decode_healthy:
            return web.json_response({"status": "healthy"})
        elif decode_healthy:
            # Can still serve in degraded mode (decode-only)
            return web.json_response(
                {"status": "degraded",
                 "reason": "no healthy prefill instances"})
        else:
            return web.json_response(
                {"status": "unhealthy",
                 "reason": "no healthy decode instances"},
                status=503)

    async def handle_router_status(
            request: web.Request) -> web.Response:
        """Router status & metrics endpoint."""
        return web.json_response(router.get_status())

    async def on_startup(app: web.Application):
        await router.start()

    async def on_cleanup(app: web.Application):
        await router.stop()

    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    app.router.add_post("/v1/completions", handle_completions)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/router/status", handle_router_status)

    return app


async def _handle_api_request(
    request: web.Request,
    router: PDRouter,
    path: str,
) -> web.StreamResponse:
    """Common handler for API requests (completions & chat)."""
    try:
        request_data = await request.json()
    except Exception:
        return web.json_response(
            {"error": {"message": "Invalid JSON in request body",
                       "type": "invalid_request_error"}},
            status=400)

    is_stream = request_data.get("stream", False)

    try:
        if is_stream:
            # Streaming response
            resp = web.StreamResponse(
                status=200,
                headers={
                    "Content-Type": "text/event-stream",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                })
            await resp.prepare(request)

            async for chunk in router.route_request(request_data, path):
                await resp.write(chunk)

            await resp.write_eof()
            return resp
        else:
            # Non-streaming: collect all chunks and return as one response
            chunks = []
            async for chunk in router.route_request(request_data, path):
                chunks.append(chunk)

            body = b"".join(chunks)
            return web.Response(
                body=body,
                content_type="application/json",
            )

    except web.HTTPException:
        raise
    except aiohttp.ClientError as e:
        logger.error("Upstream connection error: %s", e)
        return web.json_response(
            {"error": {"message": f"Upstream service error: {e}",
                       "type": "server_error"}},
            status=502)
    except Exception as e:
        logger.exception("Unexpected error routing request")
        return web.json_response(
            {"error": {"message": f"Internal router error: {e}",
                       "type": "server_error"}},
            status=500)


# ====================================================================
# CLI Entry Point
# ====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PD Disaggregation Intelligent Router / Proxy")

    parser.add_argument(
        "--prefill-endpoints",
        type=str,
        nargs="+",
        required=True,
        help="URLs of Prefill instances (e.g., http://localhost:8100)")
    parser.add_argument(
        "--decode-endpoints",
        type=str,
        nargs="+",
        required=True,
        help="URLs of Decode instances (e.g., http://localhost:8200)")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PROXY_PORT,
        help=f"Port for the router proxy (default: {DEFAULT_PROXY_PORT})")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument(
        "--short-prompt-threshold",
        type=int,
        default=DEFAULT_SHORT_PROMPT_THRESHOLD,
        help=("Prompt token threshold for short request classification. "
              f"Default: {DEFAULT_SHORT_PROMPT_THRESHOLD}"))
    parser.add_argument(
        "--health-check-interval",
        type=float,
        default=5.0,
        help="Health check interval in seconds (default: 5.0)")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max prefill retries (default: {DEFAULT_MAX_RETRIES})")

    return parser.parse_args()


def main():
    """CLI entry point for the PD Router."""
    args = parse_args()

    router = PDRouter(
        prefill_endpoints=args.prefill_endpoints,
        decode_endpoints=args.decode_endpoints,
        short_prompt_threshold=args.short_prompt_threshold,
        max_retries=args.max_retries,
        health_check_interval=args.health_check_interval,
    )

    app = create_app(router)

    logger.info(
        "Starting PD Router on %s:%d", args.host, args.port)
    logger.info(
        "  Prefill endpoints: %s", args.prefill_endpoints)
    logger.info(
        "  Decode endpoints:  %s", args.decode_endpoints)
    logger.info(
        "  Short prompt threshold: %d tokens",
        args.short_prompt_threshold)

    web.run_app(app, host=args.host, port=args.port,
                print=lambda msg: logger.info(msg))


if __name__ == "__main__":
    main()
