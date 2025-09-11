# Deep Tree Echo Server-Side Template System

## Overview

The Deep Tree Echo FastAPI endpoints now support server-side HTML rendering using Jinja2 templates, providing a complete server-side rendering (SSR) solution that generates HTML without any client-side dependencies.

## Features

- ✅ **Jinja2 Template Integration**: Full server-side HTML generation
- ✅ **Template Inheritance Structure**: Consistent rendering with base template
- ✅ **Server-Side Data Binding**: Dynamic content rendering with template context
- ✅ **Content Negotiation**: Automatic JSON/HTML response based on Accept headers
- ✅ **SSR-Focused Design**: No client-side JavaScript dependencies
- ✅ **DTESN Integration**: Complete integration with Deep Tree Echo System Network

## Architecture

### Template Structure

```
aphrodite/endpoints/deep_tree_echo/templates/
├── base.html              # Base template with inheritance structure
├── index.html             # Home page template
├── status.html            # System status page
├── membrane_info.html     # P-System membrane information
├── esn_state.html         # Echo State Network status
└── process_result.html    # DTESN processing results
```

### Template Inheritance

All templates extend the `base.html` template which provides:

- **Consistent styling**: Deep Tree Echo visual theme
- **Navigation structure**: Standard navigation links
- **Block system**: Extensible content blocks
- **Server-side rendering indicators**: Clear SSR identification

### Content Negotiation

The system automatically serves appropriate content based on the request:

- **JSON Response**: Default for API requests (`application/json`)
- **HTML Response**: When `Accept: text/html` header is present

## Usage

### Basic Template Rendering

```python
from fastapi.templating import Jinja2Templates

# Template dependency injection
def get_templates(request: Request) -> Jinja2Templates:
    return request.app.state.templates

# Route with template rendering
@router.get("/")
async def index(request: Request, templates: Jinja2Templates = Depends(get_templates)):
    data = {"service": "Deep Tree Echo API", "version": "1.0.0"}
    
    if wants_html(request):
        return templates.TemplateResponse("index.html", {"request": request, "data": data})
    else:
        return JSONResponse(data)
```

### Template Context Data

Each endpoint provides rich context data for server-side rendering:

#### Index Page Context
```python
{
    "service": "Deep Tree Echo API",
    "version": "1.0.0",
    "description": "Server-side rendering API for DTESN processing",
    "endpoints": ["/process", "/status", "/membrane_info", "/esn_state"],
    "server_rendered": True
}
```

#### Status Page Context
```python
{
    "dtesn_system": "operational",
    "membrane_hierarchy": "active", 
    "esn_reservoir": "ready",
    "server_side": True,
    "processing_capabilities": {
        "max_membrane_depth": 4,
        "max_esn_size": 512,
        "bseries_max_order": 8
    }
}
```

#### Membrane Info Context
```python
{
    "membrane_type": "P-System",
    "hierarchy_type": "rooted_tree",
    "oeis_sequence": "A000081", 
    "max_depth": 4,
    "supported_operations": ["membrane_evolution", "cross_membrane_communication"],
    "server_rendered": True
}
```

## Template Blocks

### Base Template Blocks

The `base.html` template provides the following extensible blocks:

- `{% block title %}`: Page title
- `{% block head %}`: Additional head content
- `{% block styles %}`: Custom CSS styles
- `{% block header_title %}`: Header title text
- `{% block nav_links %}`: Additional navigation links  
- `{% block content %}`: Main page content
- `{% block footer %}`: Footer content
- `{% block scripts %}`: JavaScript includes

### Example Template Extension

```html
{% extends "base.html" %}

{% block title %}Custom Page - Deep Tree Echo{% endblock %}

{% block content %}
<div class="card">
    <h2>{{ data.title }}</h2>
    <p>{{ data.description }}</p>
</div>
{% endblock %}
```

## Server-Side Data Binding

### Dynamic Content Rendering

Templates support rich server-side data binding:

```html
<!-- Status badges with conditional styling -->
<div class="status-badge {% if data.dtesn_system == 'operational' %}status-success{% else %}status-error{% endif %}">
    DTESN System: {{ data.dtesn_system|upper }}
</div>

<!-- Configuration display -->
<pre>Max Membrane Depth: {{ data.processing_capabilities.max_membrane_depth }}
ESN Reservoir Size: {{ data.processing_capabilities.max_esn_size }}</pre>

<!-- List iteration -->
{% for endpoint in data.endpoints %}
<li><a href="/deep_tree_echo{{ endpoint }}">{{ endpoint }}</a></li>
{% endfor %}
```

### Template Filters

Jinja2 filters are used for data formatting:

```html
<!-- Number formatting -->
Processing Time: {{ "%.2f"|format(data.processing_time_ms) }} ms

<!-- String manipulation -->
Status: {{ data.status|upper }}

<!-- JSON display -->
<pre>{{ result_json }}</pre>
```

## Testing

### Validation Scripts

The system includes comprehensive validation:

- `validate_template_system.py`: Structure and integration validation
- `demo_template_rendering.py`: Functional demonstration
- `test_server_side_templates.py`: Comprehensive test suite

### Running Tests

```bash
# Structure validation
python validate_template_system.py

# Rendering demonstration  
python demo_template_rendering.py

# Full test suite (requires dependencies)
python -m pytest test_server_side_templates.py -v
```

## Integration with FastAPI

### App Factory Configuration

The `create_app()` function automatically configures Jinja2Templates:

```python
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Templates directory setup
TEMPLATES_DIR = Path(__file__).parent / "templates"

def create_app(engine=None, config=None) -> FastAPI:
    app = FastAPI(...)
    
    # Initialize Jinja2 templates
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates
    
    return app
```

### Route Handler Pattern

All endpoints follow the SSR pattern:

```python
@router.get("/endpoint")
async def endpoint_handler(
    request: Request,
    templates: Jinja2Templates = Depends(get_templates),
    processor: DTESNProcessor = Depends(get_dtesn_processor)
) -> Union[HTMLResponse, JSONResponse]:
    # Generate data
    data = {...}
    
    # Content negotiation
    if wants_html(request):
        return templates.TemplateResponse("template.html", {"request": request, "data": data})
    else:
        return JSONResponse(data)
```

## Performance Considerations

### Server-Side Rendering Benefits

- **Complete HTML Generation**: No client-side rendering dependencies
- **SEO Friendly**: Fully rendered content for search engines  
- **Fast Initial Load**: Immediate content display
- **Template Caching**: Jinja2 automatic template caching
- **Efficient Data Binding**: Server-side context compilation

### Optimization Features

- **Template Inheritance**: Reduced code duplication
- **Conditional Rendering**: Server-side logic for optimal output
- **Content Compression**: FastAPI automatic gzip support
- **Static Asset Optimization**: CSS embedded in templates

## Dependencies

The template system requires the following dependencies (added to `requirements/common.txt`):

```txt
fastapi[standard] >= 0.115.0
jinja2 >= 3.0.0
python-multipart
```

## Roadmap Integration

This implementation fulfills **Phase 5.1.3** requirements:

- ✅ Integrate Jinja2 for server-side HTML generation (where appropriate)
- ✅ Create template inheritance structure for consistent rendering
- ✅ Implement server-side data binding and template context
- ✅ **Acceptance Criteria**: Templates render complete HTML server-side

## Security Considerations

- **Template Security**: Jinja2 sandboxing prevents code execution
- **Data Sanitization**: Server-side data validation before rendering
- **XSS Protection**: Automatic HTML escaping for all variables
- **Content-Type Headers**: Proper MIME type specification

## Future Enhancements

Planned enhancements for Phase 7.2 (Advanced Server-Side Template Engine):

- Dynamic template generation based on DTESN results
- Template caching and optimization mechanisms  
- Responsive template adaptation without client code
- Multi-format response generation (JSON, HTML, XML)
- Progressive rendering for complex DTESN results

---

*This documentation covers the complete server-side template system implementation for the Deep Tree Echo FastAPI endpoints, providing SSR capabilities without client-side dependencies.*