"""
Route handlers for Deep Tree Echo endpoints.

Implements server-side route handlers for DTESN processing with FastAPI.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from aphrodite.endpoints.deep_tree_echo.dtesn_processor import DTESNProcessor

logger = logging.getLogger(__name__)

# Create API router for DTESN endpoints
router = APIRouter(tags=["deep_tree_echo"])


class DTESNRequest(BaseModel):
    """Request model for DTESN processing."""
    
    input_data: str
    membrane_depth: Optional[int] = 4
    esn_size: Optional[int] = 512
    processing_mode: str = "server_side"


class DTESNResponse(BaseModel):
    """Response model for DTESN processing results."""
    
    status: str
    result: Dict[str, Any]
    processing_time_ms: float
    membrane_layers: int
    server_rendered: bool = True


def get_dtesn_processor(request: Request) -> DTESNProcessor:
    """Dependency to get DTESN processor from app state."""
    config = getattr(request.app.state, "config", None)
    engine = getattr(request.app.state, "engine", None)
    try:
        return DTESNProcessor(config=config, engine=engine)
    except RuntimeError as e:
        logger.error(f"DTESN processor initialization failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"DTESN service unavailable: {e}"
        )


@router.get("/")
async def dtesn_root():
    """Root endpoint for Deep Tree Echo SSR API."""
    return {
        "service": "Deep Tree Echo API",
        "version": "1.0.0",
        "description": "Server-side rendering API for DTESN processing",
        "endpoints": [
            "/process",
            "/status",
            "/membrane_info",
            "/esn_state"
        ],
        "server_rendered": True
    }


@router.post("/process", response_model=DTESNResponse)
async def process_dtesn(
    request: DTESNRequest,
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> DTESNResponse:
    """
    Process input through Deep Tree Echo System Network.

    Server-side processing of input data through DTESN membrane hierarchy
    with Echo State Network integration.

    Args:
        request: DTESN processing request
        processor: DTESN processor instance

    Returns:
        Server-rendered response with processing results
    """
    try:
        # Process input through DTESN
        result = await processor.process(
            input_data=request.input_data,
            membrane_depth=request.membrane_depth,
            esn_size=request.esn_size
        )

        return DTESNResponse(
            status="success",
            result=result.to_dict(),
            processing_time_ms=result.processing_time_ms,
            membrane_layers=result.membrane_layers,
            server_rendered=True
        )

    except Exception as e:
        logger.error(f"DTESN processing error: {e}")
        raise HTTPException(status_code=500, detail=f"DTESN processing failed: {e}")


@router.get("/status")
async def dtesn_status(
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Dict[str, Any]:
    """
    Get DTESN system status.

    Returns server-rendered status information about the DTESN system.

    Returns:
        System status information
    """
    return {
        "dtesn_system": "operational",
        "membrane_hierarchy": "active",
        "esn_reservoir": "ready",
        "server_side": True,
        "processing_capabilities": {
            "max_membrane_depth": processor.config.max_membrane_depth,
            "max_esn_size": processor.config.esn_reservoir_size,
            "bseries_max_order": processor.config.bseries_max_order
        }
    }


@router.get("/membrane_info")
async def membrane_info(
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Dict[str, Any]:
    """
    Get information about DTESN membrane hierarchy.

    Server-side endpoint providing membrane computing information.

    Returns:
        Membrane hierarchy information
    """
    return {
        "membrane_type": "P-System",
        "hierarchy_type": "rooted_tree",
        "oeis_sequence": "A000081",
        "max_depth": processor.config.max_membrane_depth,
        "supported_operations": [
            "membrane_evolution",
            "cross_membrane_communication",
            "rule_application",
            "tree_enumeration"
        ],
        "server_rendered": True
    }


@router.get("/esn_state")
async def esn_state(
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Dict[str, Any]:
    """
    Get Echo State Network reservoir state information.

    Server-side endpoint providing ESN state details.

    Returns:
        ESN state information
    """
    return {
        "reservoir_type": "echo_state_network",
        "reservoir_size": processor.config.esn_reservoir_size,
        "connectivity": "sparse_random",
        "activation": "tanh",
        "spectral_radius": 0.95,
        "leaky_rate": 0.1,
        "state": "ready",
        "server_rendered": True
    }