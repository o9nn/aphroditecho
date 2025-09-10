"""
Tests for Deep Tree Echo FastAPI endpoints.

Validates the server-side rendering functionality of DTESN endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from aphrodite.endpoints.deep_tree_echo import create_app
from aphrodite.endpoints.deep_tree_echo.config import DTESNConfig


class TestDeepTreeEchoEndpoints:
    """Test suite for Deep Tree Echo FastAPI endpoints."""

    @pytest.fixture
    def config(self):
        """Test configuration fixture."""
        return DTESNConfig(
            enable_docs=True,
            max_membrane_depth=4,
            esn_reservoir_size=256,
            bseries_max_order=8
        )

    @pytest.fixture
    def app(self, config):
        """FastAPI application fixture."""
        return create_app(config=config)

    @pytest.fixture
    def client(self, app):
        """Test client fixture."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "deep_tree_echo"
        assert data["version"] == "1.0.0"

    def test_dtesn_root_endpoint(self, client):
        """Test DTESN root endpoint."""
        response = client.get("/deep_tree_echo/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Deep Tree Echo API"
        assert data["server_rendered"] is True
        assert "endpoints" in data

    def test_dtesn_status_endpoint(self, client):
        """Test DTESN status endpoint."""
        response = client.get("/deep_tree_echo/status")
        assert response.status_code == 200
        data = response.json()
        assert data["dtesn_system"] == "operational"
        assert data["server_side"] is True
        assert "processing_capabilities" in data

    def test_membrane_info_endpoint(self, client):
        """Test membrane information endpoint."""
        response = client.get("/deep_tree_echo/membrane_info")
        assert response.status_code == 200
        data = response.json()
        assert data["membrane_type"] == "P-System"
        assert data["oeis_sequence"] == "A000081"
        assert data["server_rendered"] is True

    def test_esn_state_endpoint(self, client):
        """Test ESN state endpoint."""
        response = client.get("/deep_tree_echo/esn_state")
        assert response.status_code == 200
        data = response.json()
        assert data["reservoir_type"] == "echo_state_network"
        assert data["state"] == "ready"
        assert data["server_rendered"] is True

    def test_dtesn_process_endpoint(self, client):
        """Test DTESN processing endpoint."""
        request_data = {
            "input_data": "test input",
            "membrane_depth": 3,
            "esn_size": 128,
            "processing_mode": "server_side"
        }
        
        response = client.post("/deep_tree_echo/process", json=request_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "success"
        assert data["server_rendered"] is True
        assert data["membrane_layers"] == 3
        assert "result" in data
        assert "processing_time_ms" in data

    def test_performance_headers(self, client):
        """Test performance monitoring headers are present."""
        response = client.get("/deep_tree_echo/status")
        assert response.status_code == 200
        
        # Check performance headers
        assert "X-Process-Time" in response.headers
        assert "X-Server-Timestamp" in response.headers
        assert "X-DTESN-Processed" in response.headers
        assert response.headers["X-Processing-Mode"] == "server-side"

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        import os
        
        # Set test environment variables
        os.environ["DTESN_MAX_MEMBRANE_DEPTH"] = "6"
        os.environ["DTESN_ESN_RESERVOIR_SIZE"] = "512"
        os.environ["DTESN_ENABLE_CACHING"] = "false"
        
        config = DTESNConfig.from_env()
        assert config.max_membrane_depth == 6
        assert config.esn_reservoir_size == 512
        assert config.enable_caching is False
        
        # Cleanup
        del os.environ["DTESN_MAX_MEMBRANE_DEPTH"]
        del os.environ["DTESN_ESN_RESERVOIR_SIZE"]  
        del os.environ["DTESN_ENABLE_CACHING"]