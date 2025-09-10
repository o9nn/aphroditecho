"""
FastAPI application factory for Deep Tree Echo endpoints.

Implements the application factory pattern for creating FastAPI applications
with Deep Tree Echo System Network (DTESN) integration, following SSR best practices.
"""

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aphrodite.endpoints.deep_tree_echo.config import DTESNConfig
from aphrodite.endpoints.deep_tree_echo.middleware import (
    DTESNMiddleware,
    PerformanceMonitoringMiddleware,
)
from aphrodite.endpoints.deep_tree_echo.routes import router
from aphrodite.engine.async_aphrodite import AsyncAphrodite

logger = logging.getLogger(__name__)


def create_app(
    engine: Optional[AsyncAphrodite] = None,
    config: Optional[DTESNConfig] = None
) -> FastAPI:
    """
    Create FastAPI application with Deep Tree Echo endpoints.

    This factory creates a FastAPI application configured for server-side
    rendering with DTESN processing capabilities.

    Args:
        engine: AsyncAphrodite engine instance for model serving
        config: DTESN configuration object

    Returns:
        Configured FastAPI application instance
    """
    if config is None:
        config = DTESNConfig()

    # Create FastAPI app with SSR-focused configuration
    app = FastAPI(
        title="Deep Tree Echo API",
        description="Server-side rendering API for Deep Tree Echo System Network (DTESN) processing",
        version="1.0.0",
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_docs else None,
    )

    # Configure CORS for server-side requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Add performance monitoring middleware
    app.add_middleware(PerformanceMonitoringMiddleware)

    # Add DTESN-specific middleware
    app.add_middleware(DTESNMiddleware)

    # Store engine and config in app state for route handlers
    app.state.engine = engine
    app.state.config = config

    # Include DTESN route handlers
    app.include_router(router, prefix="/deep_tree_echo")

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Server-side health check endpoint."""
        return {
            "status": "healthy",
            "service": "deep_tree_echo",
            "version": "1.0.0"
        }

    logger.info("Deep Tree Echo FastAPI application created successfully")
    return app