"""
Middleware for Deep Tree Echo FastAPI endpoints.

Provides request/response middleware for DTESN processing and performance monitoring
with enhanced async resource management and concurrency control.
"""

import time
import asyncio
import logging
from typing import Callable, Optional

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

from aphrodite.endpoints.deep_tree_echo.async_manager import (
    AsyncConnectionPool,
    ConcurrencyManager
)

logger = logging.getLogger(__name__)


class DTESNMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for DTESN-specific request processing with async resource management."""

    def __init__(self, app, connection_pool: Optional[AsyncConnectionPool] = None):
        """Initialize DTESN middleware with connection pool."""
        super().__init__(app)
        self.connection_pool = connection_pool

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process DTESN-specific request handling with enhanced async resource management.

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response with DTESN processing metadata and resource management headers
        """
        # Add enhanced DTESN request context with resource management
        request.state.dtesn_context = {
            "request_id": request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000000)}"),
            "membrane_depth": 0,
            "processing_mode": "server_side",
            "async_processing": True,
            "resource_managed": self.connection_pool is not None,
            "start_time": time.time()
        }

        try:
            # Use connection pool if available for resource-managed processing
            if self.connection_pool and self._is_dtesn_request(request):
                async with self.connection_pool.get_connection() as connection_id:
                    request.state.dtesn_context["connection_id"] = connection_id
                    response = await call_next(request)
            else:
                response = await call_next(request)

            # Add enhanced DTESN response headers
            response.headers["X-DTESN-Processed"] = "true"
            response.headers["X-Processing-Mode"] = "server-side"
            response.headers["X-Async-Managed"] = "true" if self.connection_pool else "false"
            response.headers["X-Request-ID"] = request.state.dtesn_context["request_id"]

            return response

        except Exception as e:
            logger.error(f"DTESN middleware error for request {request.state.dtesn_context['request_id']}: {e}")
            # Return error response while preserving resource cleanup
            return Response(
                content=f"DTESN processing error: {str(e)}",
                status_code=500,
                headers={
                    "X-DTESN-Error": "true",
                    "X-Request-ID": request.state.dtesn_context["request_id"]
                }
            )

    def _is_dtesn_request(self, request: Request) -> bool:
        """Check if request is for DTESN processing endpoints."""
        return (
            request.url.path.startswith("/deep_tree_echo/") and
            request.url.path not in ["/deep_tree_echo/", "/deep_tree_echo/status"]
        )


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Enhanced middleware for performance monitoring and request timing with concurrency control."""

    def __init__(self, app, concurrency_manager: Optional[ConcurrencyManager] = None):
        """Initialize performance monitoring middleware with concurrency manager."""
        super().__init__(app)
        self.concurrency_manager = concurrency_manager

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Monitor request performance and timing with enhanced concurrency control.

        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler

        Returns:
            Response with enhanced performance timing headers and concurrency metrics
        """
        start_time = time.time()

        try:
            # Apply concurrency control if available and needed
            if self.concurrency_manager and self._needs_throttling(request):
                async with self.concurrency_manager.throttle_request():
                    response = await call_next(request)
            else:
                response = await call_next(request)

            # Calculate processing time
            process_time = time.time() - start_time

            # Add enhanced performance headers
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Server-Timestamp"] = str(int(time.time()))
            response.headers["X-Async-Processing"] = "true"
            
            # Add concurrency metrics if available
            if self.concurrency_manager:
                load_stats = self.concurrency_manager.get_current_load()
                response.headers["X-Concurrency-Load"] = f"{load_stats['concurrency_utilization']:.2f}"
                response.headers["X-Rate-Limit-Load"] = f"{load_stats['rate_limit_utilization']:.2f}"

            # Log performance metrics with enhanced details
            logger.info(
                f"DTESN request processed: {request.url.path} in {process_time:.3f}s "
                f"[concurrent: {'yes' if self.concurrency_manager else 'no'}]"
            )

            return response

        except HTTPException:
            # Re-raise HTTP exceptions without modification
            raise
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Performance monitoring error: {e} (after {process_time:.3f}s)")
            
            # Return error response with timing information
            return Response(
                content=f"Performance monitoring error: {str(e)}",
                status_code=500,
                headers={
                    "X-Process-Time": str(process_time),
                    "X-Performance-Error": "true"
                }
            )

    def _needs_throttling(self, request: Request) -> bool:
        """Check if request needs concurrency throttling."""
        return (
            request.method in ["POST", "PUT"] and
            request.url.path.startswith("/deep_tree_echo/")
        )


class AsyncResourceMiddleware(BaseHTTPMiddleware):
    """
    Middleware for managing async resources and connection pooling.
    
    Provides centralized async resource management across all DTESN endpoints
    with proper cleanup and error handling.
    """
    
    def __init__(
        self,
        app,
        connection_pool: Optional[AsyncConnectionPool] = None,
        enable_resource_monitoring: bool = True
    ):
        """Initialize async resource middleware."""
        super().__init__(app)
        self.connection_pool = connection_pool
        self.enable_resource_monitoring = enable_resource_monitoring
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Manage async resources for request processing.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response with resource management headers and cleanup
        """
        resource_context = {
            "pool_available": self.connection_pool is not None,
            "monitoring_enabled": self.enable_resource_monitoring,
            "request_start": time.time()
        }
        
        # Store resource context in request state
        request.state.resource_context = resource_context
        
        try:
            # Process request with resource management
            response = await call_next(request)
            
            # Add resource management headers
            if self.enable_resource_monitoring and self.connection_pool:
                stats = self.connection_pool.get_stats()
                response.headers["X-Pool-Active"] = str(stats.active_connections)
                response.headers["X-Pool-Utilization"] = f"{stats.pool_utilization:.2f}"
                response.headers["X-Pool-Avg-Time"] = f"{stats.avg_response_time:.3f}"
            
            response.headers["X-Resource-Managed"] = "true"
            
            return response
            
        except asyncio.CancelledError:
            logger.warning(f"Request cancelled: {request.url.path}")
            raise
        except Exception as e:
            logger.error(f"Async resource management error: {e}")
            return Response(
                content=f"Resource management error: {str(e)}",
                status_code=503,
                headers={"X-Resource-Error": "true"}
            )