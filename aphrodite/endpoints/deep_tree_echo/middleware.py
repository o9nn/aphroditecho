"""
Middleware for Deep Tree Echo FastAPI endpoints.

Provides request/response middleware for DTESN processing and performance monitoring.
"""

import time
import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class DTESNMiddleware(BaseHTTPMiddleware):
    """Middleware for DTESN-specific request processing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process DTESN-specific request handling.

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response with DTESN processing metadata
        """
        # Add DTESN request context
        request.state.dtesn_context = {
            "request_id": request.headers.get("X-Request-ID", "unknown"),
            "membrane_depth": 0,
            "processing_mode": "server_side",
        }

        # Process request through middleware chain
        response = await call_next(request)

        # Add DTESN response headers
        response.headers["X-DTESN-Processed"] = "true"
        response.headers["X-Processing-Mode"] = "server-side"

        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and request timing."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Monitor request performance and timing.

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response with performance timing headers
        """
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = time.time() - start_time

        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Server-Timestamp"] = str(int(time.time()))

        # Log performance metrics
        logger.info(
            f"DTESN request processed: {request.url.path} in {process_time:.3f}s"
        )

        return response