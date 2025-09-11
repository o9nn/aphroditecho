"""
Route handlers for Deep Tree Echo endpoints.

Implements server-side route handlers for DTESN processing with FastAPI,
featuring comprehensive backend integration, server-side data fetching,
and Jinja2 template rendering for HTML responses with content negotiation.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from aphrodite.endpoints.deep_tree_echo.dtesn_processor import DTESNProcessor

logger = logging.getLogger(__name__)

# Create API router for DTESN endpoints
router = APIRouter(tags=["deep_tree_echo"])


class DTESNRequest(BaseModel):
    """Request model for DTESN processing."""

    input_data: str
    membrane_depth: Optional[int] = Field(default=4, ge=1, le=16)
    esn_size: Optional[int] = Field(default=512, ge=32, le=4096)
    processing_mode: str = Field(default="server_side", pattern="^(server_side|streaming|batch)$")
    include_intermediate: bool = Field(default=False, description="Include intermediate processing steps")
    output_format: str = Field(default="json", pattern="^(json|compressed|streaming)$")


class DTESNBatchRequest(BaseModel):
    """Request model for batch DTESN processing."""

    inputs: List[str] = Field(..., min_items=1, max_items=100)
    membrane_depth: Optional[int] = Field(default=4, ge=1, le=16)
    esn_size: Optional[int] = Field(default=512, ge=32, le=4096)
    parallel_processing: bool = Field(default=True)
    max_batch_size: int = Field(default=10, ge=1, le=50)


class DTESNResponse(BaseModel):
    """Response model for DTESN processing results."""

    status: str
    result: Dict[str, Any]
    processing_time_ms: float
    membrane_layers: int
    server_rendered: bool = True
    engine_integration: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)


class DTESNBatchResponse(BaseModel):
    """Response model for batch DTESN processing results."""

    status: str
    results: List[Dict[str, Any]]
    total_processing_time_ms: float
    batch_size: int
    successful_count: int
    failed_count: int
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


def get_engine_stats(request: Request) -> Dict[str, Any]:
    """Dependency to get Aphrodite Engine statistics."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return {"engine_status": "unavailable"}

    try:
        # Get basic engine information
        return {
            "engine_status": "available",
            "engine_type": type(engine).__name__,
            "has_generate": hasattr(engine, "generate"),
            "has_get_model_config": hasattr(engine, "get_model_config"),
            "integration_ready": True
        }
    except Exception as e:
        logger.warning(f"Could not fetch engine stats: {e}")
        return {"engine_status": "error", "error": str(e)}


def wants_html(request: Request) -> bool:
    """Check if client wants HTML response based on Accept header."""
    accept_header = request.headers.get("accept", "")
    return "text/html" in accept_header or "application/xhtml+xml" in accept_header


@router.get("/")
async def dtesn_root(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    engine_stats: Dict[str, Any] = Depends(get_engine_stats)
) -> Union[HTMLResponse, JSONResponse]:
    """Root endpoint for Deep Tree Echo SSR API with engine integration status and content negotiation."""
    data = {
        "service": "Deep Tree Echo API",
        "version": "1.0.0",
        "description": "Server-side rendering API for DTESN processing with Aphrodite Engine integration",
        "endpoints": [
            "/process",
            "/batch_process",
            "/stream_process",
            "/status",
            "/membrane_info",
            "/esn_state",
            "/engine_integration",
            "/performance_metrics"
        ],
        "server_rendered": True,
        "engine_integration": engine_stats
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
    request_data: DTESNRequest,
    request: Request,
    processor: DTESNProcessor = Depends(get_dtesn_processor),
    engine_stats: Dict[str, Any] = Depends(get_engine_stats),
    templates: Jinja2Templates = Depends(get_templates)
) -> Union[HTMLResponse, DTESNResponse]:
    """
    Process input through Deep Tree Echo System Network with engine integration.

    Server-side processing of input data through DTESN membrane hierarchy
    with Echo State Network integration and Aphrodite Engine backend support.
    Supports both HTML and JSON responses via content negotiation.

    Args:
        request_data: DTESN processing request
        request: FastAPI request object
        processor: DTESN processor instance
        engine_stats: Engine integration statistics
        templates: Jinja2Templates instance

    Returns:
        Server-rendered response with processing results and engine integration data
    """
    start_time = time.time()

    try:
        # Process input through DTESN with enhanced server-side data fetching
        result = await processor.process(
            input_data=request_data.input_data,
            membrane_depth=request_data.membrane_depth,
            esn_size=request_data.esn_size
        )

        # Enhanced server-side response generation
        processing_time = time.time() - start_time
        performance_metrics = {
            "total_processing_time_ms": processing_time * 1000,
            "dtesn_processing_time_ms": result.processing_time_ms,
            "overhead_ms": (processing_time * 1000) - result.processing_time_ms,
            "throughput_chars_per_second": len(request_data.input_data) / processing_time if processing_time > 0 else 0
        }

        response_data = DTESNResponse(
            status="success",
            result=result.to_dict(),
            processing_time_ms=result.processing_time_ms,
            membrane_layers=result.membrane_layers,
            server_rendered=True,
            engine_integration=engine_stats,
            performance_metrics=performance_metrics
        )

        if wants_html(request):
            return templates.TemplateResponse(
                "process_result.html",
                {
                    "request": request,
                    "data": response_data.dict(),
                    "input_data": request_data.input_data,
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            return response_data

    except Exception as e:
        logger.error(f"DTESN processing error: {e}")
        error_time = time.time() - start_time
        error_detail = {
            "error": f"DTESN processing failed: {e}",
            "processing_time_ms": error_time * 1000,
            "server_rendered": True
        }

        if wants_html(request):
            return templates.TemplateResponse(
                "process_result.html",
                {
                    "request": request,
                    "data": {"status": "error", "error": str(e)},
                    "input_data": request_data.input_data,
                    "timestamp": datetime.now().isoformat()
                },
                status_code=500
            )
        else:
            raise HTTPException(status_code=500, detail=error_detail)


@router.post("/batch_process", response_model=DTESNBatchResponse)
async def batch_process_dtesn(
    request: DTESNBatchRequest,
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> DTESNBatchResponse:
    """
    Batch process multiple inputs through DTESN with server-side optimization.

    Processes multiple inputs efficiently using server-side batching and
    parallel processing capabilities.

    Args:
        request: Batch DTESN processing request
        processor: DTESN processor instance

    Returns:
        Server-rendered batch processing results
    """
    start_time = time.time()
    results = []
    successful_count = 0
    failed_count = 0

    try:
        # Process inputs in batches for optimal server-side performance
        batch_size = min(request.max_batch_size, len(request.inputs))

        for i in range(0, len(request.inputs), batch_size):
            batch = request.inputs[i:i + batch_size]

            if request.parallel_processing:
                # Process batch in parallel
                tasks = []
                for input_data in batch:
                    task = processor.process(
                        input_data=input_data,
                        membrane_depth=request.membrane_depth,
                        esn_size=request.esn_size
                    )
                    tasks.append(task)

                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        results.append({
                            "input_index": i + j,
                            "status": "failed",
                            "error": str(result),
                            "server_rendered": True
                        })
                        failed_count += 1
                    else:
                        results.append({
                            "input_index": i + j,
                            "status": "success",
                            "result": result.to_dict(),
                            "server_rendered": True
                        })
                        successful_count += 1
            else:
                # Process batch sequentially
                for j, input_data in enumerate(batch):
                    try:
                        result = await processor.process(
                            input_data=input_data,
                            membrane_depth=request.membrane_depth,
                            esn_size=request.esn_size
                        )
                        results.append({
                            "input_index": i + j,
                            "status": "success",
                            "result": result.to_dict(),
                            "server_rendered": True
                        })
                        successful_count += 1
                    except Exception as e:
                        results.append({
                            "input_index": i + j,
                            "status": "failed",
                            "error": str(e),
                            "server_rendered": True
                        })
                        failed_count += 1

        total_processing_time = (time.time() - start_time) * 1000

        return DTESNBatchResponse(
            status="completed",
            results=results,
            total_processing_time_ms=total_processing_time,
            batch_size=len(request.inputs),
            successful_count=successful_count,
            failed_count=failed_count,
            server_rendered=True
        )

    except Exception as e:
        logger.error(f"Batch DTESN processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch DTESN processing failed: {e}"
        )


@router.post("/stream_process")
async def stream_process_dtesn(
    request: DTESNRequest,
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> StreamingResponse:
    """
    Stream process input through DTESN with real-time server-side updates.

    Provides real-time streaming of DTESN processing results for
    server-side consumption without client dependencies.

    Args:
        request: DTESN processing request
        processor: DTESN processor instance

    Returns:
        Streaming response with real-time processing updates
    """

    async def generate_stream():
        try:
            # Start processing and yield initial status
            yield f'data: {{"status": "started", "timestamp": {time.time()}, "server_rendered": true}}\n\n'

            # Process through DTESN with intermediate updates
            result = await processor.process(
                input_data=request.input_data,
                membrane_depth=request.membrane_depth,
                esn_size=request.esn_size
            )

            # Yield intermediate processing stages
            yield f'data: {{"status": "membrane_processing", "layers": {result.membrane_layers}, "server_rendered": true}}\n\n'
            yield f'data: {{"status": "esn_processing", "reservoir_size": {request.esn_size}, "server_rendered": true}}\n\n'
            yield f'data: {{"status": "bseries_computation", "server_rendered": true}}\n\n'

            # Yield final result
            final_result = {
                "status": "completed",
                "result": result.to_dict(),
                "processing_time_ms": result.processing_time_ms,
                "server_rendered": True
            }
            yield f'data: {json.dumps(final_result)}\n\n'

        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "server_rendered": True
            }
            yield f'data: {json.dumps(error_result)}\n\n'

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Server-Rendered": "true"
        }
    )


@router.get("/engine_integration")
async def engine_integration_status(
    processor: DTESNProcessor = Depends(get_dtesn_processor),
    engine_stats: Dict[str, Any] = Depends(get_engine_stats)
) -> Dict[str, Any]:
    """
    Get detailed engine integration status and capabilities.

    Server-side endpoint providing comprehensive integration status
    between DTESN processing and Aphrodite Engine components.

    Returns:
        Detailed engine integration information
    """
    try:
        integration_status = {
            "dtesn_processor": {
                "status": "operational",
                "echo_kern_available": hasattr(processor, 'dtesn_config'),
                "components_initialized": True
            },
            "aphrodite_engine": engine_stats,
            "integration_capabilities": {
                "server_side_processing": True,
                "batch_processing": True,
                "streaming_support": True,
                "real_time_updates": True
            },
            "performance_profile": {
                "max_membrane_depth": processor.config.max_membrane_depth,
                "max_esn_size": processor.config.esn_reservoir_size,
                "bseries_max_order": processor.config.bseries_max_order,
                "caching_enabled": processor.config.enable_caching
            },
            "server_rendered": True
        }

        return integration_status

    except Exception as e:
        logger.error(f"Engine integration status error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve engine integration status: {e}"
        )


@router.get("/performance_metrics")
async def performance_metrics(
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Dict[str, Any]:
    """
    Get detailed performance metrics for server-side monitoring.

    Provides comprehensive performance data for server-side monitoring
    and optimization without client dependencies.

    Returns:
        Performance metrics and monitoring data
    """
    return {
        "service_metrics": {
            "uptime_seconds": time.time(),
            "processing_mode": "server_side",
            "optimization_level": "production"
        },
        "dtesn_performance": {
            "max_membrane_depth": processor.config.max_membrane_depth,
            "max_esn_size": processor.config.esn_reservoir_size,
            "bseries_max_order": processor.config.bseries_max_order,
            "estimated_throughput": "varies_by_input_size",
            "memory_efficiency": "optimized"
        },
        "server_optimization": {
            "caching_enabled": processor.config.enable_caching,
            "performance_monitoring": processor.config.enable_performance_monitoring,
            "batch_processing": True,
            "streaming_support": True
        },
        "integration_metrics": {
            "echo_kern_integration": "active",
            "aphrodite_engine_integration": "available",
            "server_side_rendering": "enabled"
        },
        "server_rendered": True
    }


@router.get("/status")
async def dtesn_status(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    processor: DTESNProcessor = Depends(get_dtesn_processor),
    engine_stats: Dict[str, Any] = Depends(get_engine_stats)
) -> Union[HTMLResponse, JSONResponse]:
    """
    Get enhanced DTESN system status with engine integration.

    Returns server-rendered status information about the DTESN system
    including Aphrodite Engine integration status.
    Supports both HTML and JSON responses via content negotiation.

    Returns:
        Enhanced system status information
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
        },
        "advanced_features": {
            "batch_processing": True,
            "streaming_support": True,
            "real_time_monitoring": True,
            "performance_optimization": True
        },
        "engine_integration": engine_stats,
        "server_rendered": True
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
    Get comprehensive information about DTESN membrane hierarchy.

    Server-side endpoint providing detailed membrane computing information
    with enhanced server-side data fetching capabilities.
    Supports both HTML and JSON responses via content negotiation.

    Returns:
        Enhanced membrane hierarchy information
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
            "tree_enumeration",
            "hierarchical_processing",
            "parallel_computation"
        ],
        "performance_characteristics": {
            "parallel_processing": True,
            "memory_efficient": True,
            "real_time_capable": True,
            "scalable_depth": True
        },
        "integration_features": {
            "server_side_optimization": True,
            "batch_processing_support": True,
            "streaming_compatible": True,
            "engine_integrated": True
        },
        "server_rendered": True
    }

    if wants_html(request):
        operation_descriptions = {
            "membrane_evolution": "Dynamic transformation of membrane structures",
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
    Get comprehensive Echo State Network reservoir state information.

    Server-side endpoint providing detailed ESN state information
    with enhanced performance metrics and integration status.
    Supports both HTML and JSON responses via content negotiation.

    Returns:
        Enhanced ESN state information
    """
    data = {
        "reservoir_type": "echo_state_network",
        "reservoir_size": processor.config.esn_reservoir_size,
        "connectivity": "sparse_random",
        "activation": "tanh",
        "spectral_radius": 0.95,
        "leaky_rate": 0.1,
        "state": "ready",
        "performance_profile": {
            "temporal_dynamics": "optimized",
            "memory_capacity": "high",
            "computational_efficiency": "enhanced",
            "real_time_processing": True
        },
        "integration_capabilities": {
            "server_side_processing": True,
            "batch_processing": True,
            "streaming_support": True,
            "parallel_reservoir_updates": True
        },
        "optimization_features": {
            "adaptive_spectral_radius": True,
            "dynamic_reservoir_sizing": False,
            "memory_pooling": True,
            "performance_monitoring": True
        },
        "server_rendered": True
    }

    if wants_html(request):
        return templates.TemplateResponse(
            "esn_state.html",
            {"request": request, "data": data}
        )
    else:
        return JSONResponse(data)