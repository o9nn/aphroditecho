"""
Tests for the Deep Tree Echo server-side template system.

Tests the Jinja2 integration and template rendering functionality.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from aphrodite.endpoints.deep_tree_echo.app_factory import create_app
from aphrodite.endpoints.deep_tree_echo.config import DTESNConfig


class TestServerSideTemplates:
    """Test server-side template rendering functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test config
        self.config = DTESNConfig(
            max_membrane_depth=4,
            esn_reservoir_size=512,
            bseries_max_order=8
        )
        
        # Create mock engine
        self.mock_engine = MagicMock()
        
        # Create app with mocks
        self.app = create_app(engine=self.mock_engine, config=self.config)
        self.client = TestClient(self.app)

    def test_templates_directory_exists(self):
        """Test that templates directory is properly configured."""
        from pathlib import Path
        from aphrodite.endpoints.deep_tree_echo.app_factory import TEMPLATES_DIR
        
        assert TEMPLATES_DIR.exists(), f"Templates directory should exist at {TEMPLATES_DIR}"
        assert TEMPLATES_DIR.is_dir(), "Templates path should be a directory"

    def test_base_template_exists(self):
        """Test that base template exists."""
        from aphrodite.endpoints.deep_tree_echo.app_factory import TEMPLATES_DIR
        
        base_template = TEMPLATES_DIR / "base.html"
        assert base_template.exists(), "Base template should exist"

    def test_app_has_templates(self):
        """Test that the app state contains templates."""
        assert hasattr(self.app.state, 'templates'), "App state should have templates"
        assert self.app.state.templates is not None, "Templates should be initialized"

    def test_health_check_includes_templates_status(self):
        """Test that health check includes template status."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "templates_available" in data
        assert data["templates_available"] is True

    def test_json_response_default(self):
        """Test that JSON response is returned by default."""
        response = self.client.get("/deep_tree_echo/")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert data["service"] == "Deep Tree Echo API"
        assert data["server_rendered"] is True

    def test_html_response_with_accept_header(self):
        """Test that HTML response is returned when Accept header requests it."""
        headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
        response = self.client.get("/deep_tree_echo/", headers=headers)
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"<!DOCTYPE html>" in response.content
        assert b"Deep Tree Echo" in response.content

    def test_status_endpoint_json(self):
        """Test status endpoint returns JSON by default."""
        response = self.client.get("/deep_tree_echo/status")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        data = response.json()
        assert data["dtesn_system"] == "operational"
        assert data["server_side"] is True

    def test_status_endpoint_html(self):
        """Test status endpoint returns HTML with Accept header."""
        headers = {"Accept": "text/html"}
        response = self.client.get("/deep_tree_echo/status", headers=headers)
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"System Status" in response.content
        assert b"DTESN System" in response.content

    def test_membrane_info_endpoint_html(self):
        """Test membrane info endpoint returns HTML with Accept header."""
        headers = {"Accept": "text/html"}
        response = self.client.get("/deep_tree_echo/membrane_info", headers=headers)
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Membrane Information" in response.content
        assert b"P-System" in response.content
        assert b"A000081" in response.content

    def test_esn_state_endpoint_html(self):
        """Test ESN state endpoint returns HTML with Accept header."""
        headers = {"Accept": "text/html"}
        response = self.client.get("/deep_tree_echo/esn_state", headers=headers)
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert b"Echo State Network" in response.content
        assert b"Reservoir Configuration" in response.content

    def test_template_inheritance(self):
        """Test that templates properly inherit from base template."""
        headers = {"Accept": "text/html"}
        response = self.client.get("/deep_tree_echo/", headers=headers)
        
        content = response.content.decode()
        
        # Check for base template elements
        assert "<!DOCTYPE html>" in content
        assert '<title>Deep Tree Echo - Home</title>' in content
        assert "Deep Tree Echo" in content
        assert "Server-side rendered with Jinja2" in content

    def test_server_side_data_binding(self):
        """Test that server-side data binding works correctly."""
        headers = {"Accept": "text/html"}
        response = self.client.get("/deep_tree_echo/status", headers=headers)
        
        content = response.content.decode()
        
        # Check that config values are properly bound
        assert str(self.config.max_membrane_depth) in content
        assert str(self.config.esn_reservoir_size) in content
        assert str(self.config.bseries_max_order) in content

    def test_content_negotiation_consistency(self):
        """Test that content negotiation works consistently across endpoints."""
        endpoints = ["/deep_tree_echo/", "/deep_tree_echo/status", 
                    "/deep_tree_echo/membrane_info", "/deep_tree_echo/esn_state"]
        
        for endpoint in endpoints:
            # Test JSON response (default)
            json_response = self.client.get(endpoint)
            assert json_response.status_code == 200
            assert "application/json" in json_response.headers["content-type"]
            
            # Test HTML response
            html_response = self.client.get(endpoint, headers={"Accept": "text/html"})
            assert html_response.status_code == 200
            assert "text/html" in html_response.headers["content-type"]
            assert b"<!DOCTYPE html>" in html_response.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])