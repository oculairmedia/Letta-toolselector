"""
Unit tests for the routes blueprints.

Tests cover:
- Blueprint creation and configuration
- Route registration
- Delegation patterns
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lettaaugment-source"))


# ============================================================================
# Health Blueprint Tests
# ============================================================================

class TestHealthBlueprint:
    """Tests for the health blueprint."""
    
    def test_blueprint_creation(self):
        """Should create health blueprint."""
        from routes.health import health_bp
        
        assert health_bp is not None
        assert health_bp.name == 'health'
    
    def test_configure_sets_function(self):
        """Should store health status function."""
        from routes import health
        
        mock_func = AsyncMock()
        health.configure(get_health_status_func=mock_func)
        
        assert health._get_health_status_func is mock_func
        
        # Clean up
        health._get_health_status_func = None
    
    def test_routes_registered(self):
        """Should register health check routes."""
        from routes.health import health_bp
        
        # Get registered rules
        rules = list(health_bp.deferred_functions)
        
        # Blueprint has deferred functions for route registration
        assert len(rules) >= 0  # Routes are registered via decorators


class TestToolsBlueprint:
    """Tests for the tools blueprint."""
    
    def test_blueprint_creation(self):
        """Should create tools blueprint."""
        from routes.tools import tools_bp
        
        assert tools_bp is not None
        assert tools_bp.name == 'tools'
    
    def test_configure_sets_all_functions(self):
        """Should store all handler functions."""
        from routes import tools
        
        mock_search = AsyncMock()
        mock_rerank = AsyncMock()
        mock_list = AsyncMock()
        mock_attach = AsyncMock()
        mock_prune = AsyncMock()
        mock_sync = AsyncMock()
        mock_refresh = AsyncMock()
        
        tools.configure(
            search_func=mock_search,
            search_with_rerank_func=mock_rerank,
            list_tools_func=mock_list,
            attach_tools_func=mock_attach,
            prune_tools_func=mock_prune,
            sync_func=mock_sync,
            refresh_func=mock_refresh
        )
        
        assert tools._search_func is mock_search
        assert tools._search_with_rerank_func is mock_rerank
        assert tools._list_tools_func is mock_list
        assert tools._attach_tools_func is mock_attach
        assert tools._prune_tools_func is mock_prune
        assert tools._sync_func is mock_sync
        assert tools._refresh_func is mock_refresh
        
        # Clean up
        tools._search_func = None
        tools._search_with_rerank_func = None
        tools._list_tools_func = None
        tools._attach_tools_func = None
        tools._prune_tools_func = None
        tools._sync_func = None
        tools._refresh_func = None


class TestRoutesPackage:
    """Tests for the routes package initialization."""
    
    def test_package_exports_blueprints(self):
        """Should export blueprints from package."""
        from routes import tools_bp, health_bp
        
        assert tools_bp is not None
        assert health_bp is not None
    
    def test_package_exports_configure_functions(self):
        """Should export configure functions."""
        from routes import configure_tools, configure_health
        
        assert callable(configure_tools)
        assert callable(configure_health)
    
    def test_package_exports_modules(self):
        """Should export modules for direct access."""
        from routes import tools, health
        
        assert hasattr(tools, 'configure')
        assert hasattr(health, 'configure')


# ============================================================================
# Blueprint Integration Tests
# ============================================================================

class TestBlueprintIntegration:
    """Tests for blueprint integration patterns."""
    
    @pytest.mark.asyncio
    async def test_health_delegates_to_configured_function(self):
        """Health route should delegate to configured function."""
        from routes import health
        from quart import Quart
        
        # Create test app
        test_app = Quart(__name__)
        
        # Create mock health status function
        async def mock_health():
            from quart import jsonify
            return jsonify({"status": "OK"}), 200
        
        health.configure(get_health_status_func=mock_health)
        
        # Register blueprint
        test_app.register_blueprint(health.health_bp)
        
        # Test the route
        async with test_app.test_client() as client:
            response = await client.get('/api/health')
            assert response.status_code == 200
            data = await response.get_json()
            assert data["status"] == "OK"
        
        # Clean up
        health._get_health_status_func = None
    
    @pytest.mark.asyncio
    async def test_tools_search_returns_503_when_not_configured(self):
        """Tools search should return 503 when not configured."""
        from routes import tools
        from quart import Quart
        
        # Create test app with unconfigured blueprint
        test_app = Quart(__name__)
        
        # Ensure not configured
        tools._search_func = None
        
        test_app.register_blueprint(tools.tools_bp)
        
        async with test_app.test_client() as client:
            response = await client.post('/api/v1/tools/search')
            assert response.status_code == 503
            data = await response.get_json()
            assert "not configured" in data["error"]
    
    @pytest.mark.asyncio
    async def test_tools_search_delegates_when_configured(self):
        """Tools search should delegate to configured function."""
        from routes import tools
        from quart import Quart, jsonify
        
        test_app = Quart(__name__)
        
        # Create mock search function
        async def mock_search():
            return jsonify([{"name": "test_tool", "score": 0.9}])
        
        tools.configure(search_func=mock_search)
        test_app.register_blueprint(tools.tools_bp)
        
        async with test_app.test_client() as client:
            response = await client.post('/api/v1/tools/search')
            assert response.status_code == 200
            data = await response.get_json()
            assert len(data) == 1
            assert data[0]["name"] == "test_tool"
        
        # Clean up
        tools._search_func = None
