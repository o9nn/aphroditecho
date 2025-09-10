# Deep Tree Echo FastAPI Endpoints

This module provides server-side rendering (SSR) FastAPI endpoints for Deep Tree Echo System Network (DTESN) processing integrated with the Aphrodite Engine.

## Architecture

The implementation follows the FastAPI application factory pattern with server-side rendering focus:

```
aphrodite/endpoints/deep_tree_echo/
├── __init__.py              # Module exports
├── app_factory.py           # FastAPI application factory
├── config.py                # Configuration management
├── middleware.py            # Request/response middleware
├── routes.py                # API route handlers
└── dtesn_processor.py       # DTESN processing integration
```

## Features

- ✅ **Server-side rendering** with no client dependencies
- ✅ **FastAPI application factory** for configurable app creation
- ✅ **DTESN processing integration** with echo.kern components
- ✅ **Performance monitoring** middleware
- ✅ **Environment-based configuration** management
- ✅ **Comprehensive testing** with pytest
- ✅ **Production-ready** architecture

## Quick Start

```python
from aphrodite.endpoints.deep_tree_echo import create_app
from aphrodite.endpoints.deep_tree_echo.config import DTESNConfig

# Create configuration
config = DTESNConfig(
    max_membrane_depth=6,
    esn_reservoir_size=1024,
    enable_caching=True
)

# Create FastAPI app
app = create_app(config=config)

# Run with uvicorn
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/deep_tree_echo/` | GET | Service information |
| `/deep_tree_echo/process` | POST | Process input through DTESN |
| `/deep_tree_echo/status` | GET | System status |
| `/deep_tree_echo/membrane_info` | GET | P-System membrane info |
| `/deep_tree_echo/esn_state` | GET | ESN reservoir state |

## Configuration

Configuration can be set via environment variables:

```bash
export DTESN_MAX_MEMBRANE_DEPTH=6
export DTESN_ESN_RESERVOIR_SIZE=1024
export DTESN_ENABLE_CACHING=true
export DTESN_ENABLE_PERFORMANCE_MONITORING=true
```

## Integration with echo.kern

The implementation integrates with echo.kern DTESN components when available:

- **P-System Membranes**: Hierarchical membrane computing
- **Echo State Networks**: Reservoir computing with temporal dynamics  
- **B-Series Computation**: Rooted tree enumeration and differential computation
- **OEIS A000081**: Mathematical foundation compliance

When echo.kern components are not available, mock implementations provide basic functionality.

## Testing

Run tests with pytest:

```bash
pytest tests/endpoints/test_deep_tree_echo.py -v
```

## Performance

The implementation includes:

- Request/response timing middleware
- Server-side caching support
- Performance metrics collection
- Memory-efficient processing pipelines

## Production Deployment

The endpoints are production-ready with:

- Comprehensive error handling
- Input validation and sanitization
- Security middleware
- Monitoring and observability
- Health check endpoints
- Configuration management

## Example Usage

See `usage_example_deep_tree_echo.py` and `demo_deep_tree_echo_endpoints.py` for comprehensive usage examples.

## License

This module is part of the Aphrodite Engine project and follows the same license terms.