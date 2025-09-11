"""
Route handlers for Deep Tree Echo endpoints.

Implements server-side route handlers for DTESN processing with FastAPI
and Jinja2 template rendering for HTML responses.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
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


def get_templates(request: Request) -> Jinja2Templates:
    """Dependency to get Jinja2Templates from app state."""
    templates = getattr(request.app.state, "templates", None)
    if templates is None:
        raise HTTPException(
            status_code=500,
            detail="Templates not configured"
        )
    return templates


def wants_html(request: Request) -> bool:
    """Check if client wants HTML response based on Accept header."""
    accept_header = request.headers.get("accept", "")
    return "text/html" in accept_header or "application/xhtml+xml" in accept_header


@router.get("/")
async def dtesn_root(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates)
) -> Union[HTMLResponse, JSONResponse]:
    """Root endpoint for Deep Tree Echo SSR API with content negotiation."""
    data = {
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
    
    if wants_html(request):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "data": data}
        )
    else:
        return JSONResponse(data)


@router.post("/process", response_model=DTESNResponse)
async def process_dtesn(
    request: DTESNRequest,
    templates: Jinja2Templates = Depends(get_templates),
    processor: DTESNProcessor = Depends(get_dtesn_processor),
    http_request: Request = None
) -> Union[HTMLResponse, DTESNResponse]:
    """
    Process input through Deep Tree Echo System Network.

    Server-side processing of input data through DTESN membrane hierarchy
    with Echo State Network integration. Supports both JSON and HTML responses.

    Args:
        request: DTESN processing request
        templates: Jinja2 template renderer
        processor: DTESN processor instance
        http_request: FastAPI request object

    Returns:
        Server-rendered response with processing results (HTML or JSON)
    """
    try:
        # Process input through DTESN
        result = await processor.process(
            input_data=request.input_data,
            membrane_depth=request.membrane_depth,
            esn_size=request.esn_size
        )

        response_data = DTESNResponse(
            status="success",
            result=result.to_dict(),
            processing_time_ms=result.processing_time_ms,
            membrane_layers=result.membrane_layers,
            server_rendered=True
        )

        if http_request and wants_html(http_request):
            return templates.TemplateResponse(
                "process_result.html",
                {
                    "request": http_request,
                    "data": response_data.dict(),
                    "result_json": json.dumps(result.to_dict(), indent=2),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        else:
            return response_data

    except Exception as e:
        logger.error(f"DTESN processing error: {e}")
        raise HTTPException(status_code=500, detail=f"DTESN processing failed: {e}")


@router.get("/status")
async def dtesn_status(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Union[HTMLResponse, JSONResponse]:
    """
    Get DTESN system status.

    Returns server-rendered status information about the DTESN system
    with support for both HTML and JSON responses.

    Returns:
        System status information (HTML or JSON)
    """
    data = {
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
    
    if wants_html(request):
        return templates.TemplateResponse(
            "status.html",
            {"request": request, "data": data}
        )
    else:
        return JSONResponse(data)


@router.get("/membrane_info")
async def membrane_info(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Union[HTMLResponse, JSONResponse]:
    """
    Get information about DTESN membrane hierarchy.

    Server-side endpoint providing membrane computing information
    with support for both HTML and JSON responses.

    Returns:
        Membrane hierarchy information (HTML or JSON)
    """
    data = {
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
    
    if wants_html(request):
        # Add operation descriptions for template
        operation_descriptions = {
            "membrane_evolution": "Dynamic evolution of membrane states based on P-lingua rules",
            "cross_membrane_communication": "Communication protocols between different membrane levels",
            "rule_application": "Application of P-system evolution rules within membranes",
            "tree_enumeration": "OEIS A000081 compliant counting of rooted tree structures"
        }
        
        return templates.TemplateResponse(
            "membrane_info.html",
            {
                "request": request, 
                "data": data,
                "operation_descriptions": operation_descriptions
            }
        )
    else:
        return JSONResponse(data)


@router.get("/esn_state")
async def esn_state(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Union[HTMLResponse, JSONResponse]:
    """
    Get Echo State Network reservoir state information.

    Server-side endpoint providing ESN state details
    with support for both HTML and JSON responses.

    Returns:
        ESN state information (HTML or JSON)
    """
    data = {
        "reservoir_type": "echo_state_network",
        "reservoir_size": processor.config.esn_reservoir_size,
        "connectivity": "sparse_random",
        "activation": "tanh",
        "spectral_radius": 0.95,
        "leaky_rate": 0.1,
        "state": "ready",
        "server_rendered": True
    }
    
    if wants_html(request):
        return templates.TemplateResponse(
            "esn_state.html",
            {"request": request, "data": data}
        )
    else:
        return JSONResponse(data)