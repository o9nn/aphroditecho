"""
Deep Tree Echo FastAPI endpoints for server-side rendering.

This module provides FastAPI-based endpoints for Deep Tree Echo System Network (DTESN)
processing with server-side rendering capabilities integrated with the Aphrodite Engine.

Components:
- FastAPI application factory
- Server-side route handlers for DTESN processing
- Integration with echo.kern components
- Server-side template rendering
- Performance monitoring and caching
"""

from aphrodite.endpoints.deep_tree_echo.app_factory import create_app
from aphrodite.endpoints.deep_tree_echo.routes import router

__all__ = ["create_app", "router"]