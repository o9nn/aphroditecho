"""
FastAPI application factory for Deep Tree Echo endpoints.

Implements the application factory pattern for creating FastAPI applications
with Deep Tree Echo System Network (DTESN) integration, following SSR best practices
and enhanced async resource management.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

from aphrodite.endpoints.deep_tree_echo.config import DTESNConfig
from aphrodite.endpoints.deep_tree_echo.middleware import (
    DTESNMiddleware,
    PerformanceMonitoringMiddleware,
    AsyncResourceMiddleware,
)
from aphrodite.endpoints.deep_tree_echo.async_manager import (
    AsyncConnectionPool,
    ConcurrencyManager,
    ConnectionPoolConfig
)
from aphrodite.endpoints.deep_tree_echo.routes import router
from aphrodite.endpoints.security import (
    InputValidationMiddleware,
    OutputSanitizationMiddleware,
    SecurityMiddleware,
    RateLimitMiddleware
)
from aphrodite.engine.async_aphrodite import AsyncAphrodite

logger = logging.getLogger(__name__)

# Get the templates directory path
TEMPLATES_DIR = Path(__file__).parent / "templates"


def create_app(
    engine: Optional[AsyncAphrodite] = None,
    config: Optional[DTESNConfig] = None,
    enable_async_resources: bool = True
) -> FastAPI:
    """
    Create FastAPI application with Deep Tree Echo endpoints and enhanced async processing.

    This factory creates a FastAPI application configured for server-side
    rendering with DTESN processing capabilities, Jinja2 template integration,
    and advanced async resource management for high-concurrency scenarios.

    Args:
        engine: AsyncAphrodite engine instance for model serving
        config: DTESN configuration object
        enable_async_resources: Enable async connection pooling and resource management

    Returns:
        Configured FastAPI application instance with enhanced SSR and async capabilities
    """
    if config is None:
        config = DTESNConfig()

    # Create FastAPI app with enhanced SSR-focused configuration
    app = FastAPI(
        title="Deep Tree Echo API",
        description="Enhanced server-side rendering API for Deep Tree Echo System Network (DTESN) processing with async resource management",
        version="1.0.0",
        docs_url="/docs" if config.enable_docs else None,
        redoc_url="/redoc" if config.enable_docs else None,
        openapi_url="/openapi.json" if config.enable_docs else None,
    )

    # Initialize Jinja2 templates for server-side rendering
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    
    # Store templates in app state for route handlers
    app.state.templates = templates

    # Initialize async resource management if enabled
    connection_pool = None
    concurrency_manager = None
    
    if enable_async_resources:
        # Configure connection pool for async resource management
        pool_config = ConnectionPoolConfig(
            max_connections=100,
            min_connections=10,
            connection_timeout=30.0,
            idle_timeout=300.0
        )
        connection_pool = AsyncConnectionPool(pool_config)
        
        # Configure concurrency manager for request throttling
        concurrency_manager = ConcurrencyManager(
            max_concurrent_requests=50,
            max_requests_per_second=100.0,
            burst_limit=20
        )
        
        logger.info("Async resource management enabled with connection pooling and concurrency control")

    # Configure CORS for server-side requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )

    # Add security middleware stack (outermost to innermost)
    app.add_middleware(OutputSanitizationMiddleware)  # Final output sanitization
    app.add_middleware(SecurityMiddleware)  # IP blocking and monitoring  
    app.add_middleware(RateLimitMiddleware)  # Rate limiting
    app.add_middleware(InputValidationMiddleware)  # Input validation

    # Add async resource management middleware first (innermost)
    if enable_async_resources and connection_pool:
        app.add_middleware(AsyncResourceMiddleware, connection_pool=connection_pool)

    # Add performance monitoring middleware with concurrency management
    app.add_middleware(PerformanceMonitoringMiddleware, concurrency_manager=concurrency_manager)

    # Add DTESN-specific middleware with connection pooling
    app.add_middleware(DTESNMiddleware, connection_pool=connection_pool)

    # Store engine, config, and async resources in app state for route handlers
    app.state.engine = engine
    app.state.config = config
    app.state.connection_pool = connection_pool
    app.state.concurrency_manager = concurrency_manager

    # Include DTESN route handlers
    app.include_router(router, prefix="/deep_tree_echo")

    # Enhanced health check endpoint with async resource status
    @app.get("/health")
    async def health_check():
        """Enhanced server-side health check endpoint with async resource status."""
        health_data = {
            "status": "healthy",
            "service": "Deep Tree Echo API",
            "version": "1.0.0",
            "server_rendered": True,
            "templates_available": TEMPLATES_DIR.exists(),
            "async_resources": {
                "connection_pool_enabled": connection_pool is not None,
                "concurrency_management_enabled": concurrency_manager is not None
            }
        }
        
        # Add resource pool statistics if available
        if connection_pool:
            stats = connection_pool.get_stats()
            health_data["resource_stats"] = {
                "active_connections": stats.active_connections,
                "idle_connections": stats.idle_connections,
                "pool_utilization": stats.pool_utilization,
                "avg_response_time": stats.avg_response_time
            }
        
        # Add concurrency statistics if available
        if concurrency_manager:
            load_stats = concurrency_manager.get_current_load()
            health_data["concurrency_stats"] = load_stats
        
        return health_data

    # Add startup and shutdown event handlers for async resource management
    @app.on_event("startup")
    async def startup_event():
        """Initialize async resources on application startup."""
        if connection_pool:
            await connection_pool.start()
            logger.info("Connection pool started successfully")
        
        logger.info("Deep Tree Echo FastAPI application started with enhanced async processing")

    @app.on_event("shutdown") 
    async def shutdown_event():
        """Clean up async resources on application shutdown."""
        if connection_pool:
            await connection_pool.stop()
            logger.info("Connection pool stopped successfully")
        
        logger.info("Deep Tree Echo FastAPI application shutdown complete")

    logger.info(f"Deep Tree Echo FastAPI application created successfully with templates at {TEMPLATES_DIR}")
    return app