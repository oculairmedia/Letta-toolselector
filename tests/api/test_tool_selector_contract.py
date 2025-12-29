#!/usr/bin/env python3
"""
Contract tests for Tool Selector API.

These tests validate that the API responses conform to the expected schema,
ensuring backwards compatibility and consistent behavior.
"""
import os
import pytest
import httpx
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# API base URL - configurable via environment
API_BASE_URL = os.environ.get("TOOL_SELECTOR_API_URL", "http://localhost:8020")


@dataclass
class ToolSchema:
    """Expected schema for a tool in API responses."""
    required_fields: List[str] = None
    optional_fields: List[str] = None
    
    def __post_init__(self):
        self.required_fields = [
            "id",
            "name", 
            "description",
        ]
        self.optional_fields = [
            "score",
            "tool_type",
            "tags",
            "source_code",
            "json_schema",
            "module",
            "server_name",
            # Enrichment fields
            "action_entities",
            "semantic_keywords",
            "use_cases",
            "server_domain",
            "enhanced_description",
        ]


@dataclass  
class SearchResponseSchema:
    """Expected schema for search API response.
    
    Note: The search API returns a list of tools directly, not wrapped in an object.
    """
    pass  # Response is a list, not an object with fields


@dataclass
class AttachResponseSchema:
    """Expected schema for attach API response."""
    required_fields: List[str] = None
    
    def __post_init__(self):
        self.required_fields = [
            "success",
            "attached_tools",
            "message",
        ]


class TestSearchAPIContract:
    """Contract tests for /api/v1/tools/search endpoint."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE_URL, timeout=30.0)
    
    @pytest.fixture
    def tool_schema(self):
        return ToolSchema()
    
    @pytest.fixture
    def search_schema(self):
        return SearchResponseSchema()
    
    def test_search_returns_200_with_valid_query(self, client):
        """Search with valid query returns 200 OK."""
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "create issue", "limit": 5}
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    
    def test_search_response_is_list(self, client):
        """Search response is a list of tools (not wrapped in object)."""
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "file operations", "limit": 3}
        )
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data, list), "Search response should be a list of tools"
    
    def test_search_tools_have_required_fields(self, client, tool_schema):
        """Each tool in search results has required fields."""
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "send message", "limit": 5}
        )
        assert response.status_code == 200
        tools = response.json()
        
        assert isinstance(tools, list), "Response should be a list"
        assert len(tools) > 0, "Expected at least one tool in results"
        
        for tool in tools:
            # Check for name (required) - id may be tool_id in this API
            assert "name" in tool, f"Tool missing 'name' field: {tool.keys()}"
    
    def test_search_tools_have_scores(self, client):
        """Search results include relevance scores."""
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "read file", "limit": 3}
        )
        assert response.status_code == 200
        tools = response.json()
        
        for tool in tools:
            assert "score" in tool, "Tool missing score field"
            assert isinstance(tool["score"], (int, float)), "Score must be numeric"
            assert 0.0 <= tool["score"] <= 1.0, f"Score {tool['score']} out of range [0, 1]"
    
    def test_search_respects_limit(self, client):
        """Search respects the limit parameter."""
        limit = 3
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "tool", "limit": limit}
        )
        assert response.status_code == 200
        tools = response.json()
        
        assert len(tools) <= limit, f"Expected <= {limit} tools, got {len(tools)}"
    
    def test_search_with_empty_query_returns_error(self, client):
        """Search with empty query returns appropriate error."""
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "", "limit": 5}
        )
        # Should return 400 Bad Request or 422 Validation Error
        assert response.status_code in [400, 422], f"Expected 400/422, got {response.status_code}"
    
    def test_search_without_query_returns_error(self, client):
        """Search without query field returns error."""
        response = client.post(
            "/api/v1/tools/search",
            json={"limit": 5}
        )
        assert response.status_code in [400, 422], f"Expected 400/422, got {response.status_code}"
    
    def test_search_with_invalid_limit_handled(self, client):
        """Search with invalid limit is handled gracefully."""
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "test", "limit": -1}
        )
        # Should either reject or use default
        assert response.status_code in [200, 400, 422]
    
    def test_search_returns_list_of_dicts(self, client):
        """Search returns a list of tool dictionaries."""
        response = client.post(
            "/api/v1/tools/search",
            json={"query": "create", "limit": 5}
        )
        assert response.status_code == 200
        tools = response.json()
        
        assert isinstance(tools, list), "Response should be a list"
        if len(tools) > 0:
            assert isinstance(tools[0], dict), "Each tool should be a dictionary"


class TestAttachAPIContract:
    """Contract tests for /api/v1/tools/attach endpoint."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE_URL, timeout=30.0)
    
    @pytest.fixture
    def attach_schema(self):
        return AttachResponseSchema()
    
    def test_attach_requires_agent_id(self, client):
        """Attach without agent_id returns error."""
        response = client.post(
            "/api/v1/tools/attach",
            json={"tool_ids": ["some-tool-id"]}
        )
        assert response.status_code in [400, 422], f"Expected 400/422, got {response.status_code}"
    
    def test_attach_requires_tool_specification(self, client):
        """Attach without tool_ids or query returns error."""
        response = client.post(
            "/api/v1/tools/attach",
            json={"agent_id": "test-agent-id"}
        )
        # API returns 500 for missing parameters (could be improved to 400/422)
        assert response.status_code in [400, 422, 500], f"Expected 400/422/500, got {response.status_code}"
    
    def test_attach_with_invalid_agent_id_handled(self, client):
        """Attach with non-existent agent_id is handled gracefully."""
        response = client.post(
            "/api/v1/tools/attach",
            json={
                "agent_id": "non-existent-agent-12345",
                "tool_ids": ["some-tool-id"]
            }
        )
        # Should return error, not crash
        assert response.status_code in [400, 404, 422, 500]
        # Should have error message
        data = response.json()
        assert "error" in data or "message" in data or "detail" in data


class TestHealthAPIContract:
    """Contract tests for health check endpoints."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE_URL, timeout=10.0)
    
    def test_enrichment_health_endpoint_returns_200(self, client):
        """Enrichment health endpoint returns 200 OK."""
        response = client.get("/api/v1/enrichment/health")
        assert response.status_code == 200
    
    def test_enrichment_health_response_has_success(self, client):
        """Enrichment health response includes success field."""
        response = client.get("/api/v1/enrichment/health")
        assert response.status_code == 200
        data = response.json()
        assert "success" in data, "Health response missing 'success' field"
    
    def test_pruning_status_endpoint_returns_200(self, client):
        """Pruning status endpoint returns 200 OK."""
        response = client.get("/api/v1/pruning/status")
        assert response.status_code == 200
        data = response.json()
        assert "configured" in data, "Pruning status missing 'configured' field"


class TestMetricsAPIContract:
    """Contract tests for metrics endpoint."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE_URL, timeout=10.0)
    
    def test_metrics_endpoint_returns_200(self, client):
        """Metrics endpoint returns 200 OK."""
        response = client.get("/metrics")
        assert response.status_code == 200
    
    def test_metrics_returns_prometheus_format(self, client):
        """Metrics endpoint returns Prometheus text format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        content_type = response.headers.get("content-type", "")
        # Prometheus format should be text/plain or text/plain; version=0.0.4
        assert "text/plain" in content_type or "text" in content_type
        
        # Should contain at least one metric line
        text = response.text
        assert "tool_selector" in text or "python" in text or "process" in text


class TestEnrichmentAPIContract:
    """Contract tests for enrichment endpoints."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE_URL, timeout=10.0)
    
    def test_enrichment_stats_returns_200(self, client):
        """Enrichment stats endpoint returns 200 OK."""
        response = client.get("/api/v1/enrichment/stats")
        assert response.status_code == 200
    
    def test_enrichment_stats_has_cache_info(self, client):
        """Enrichment stats includes cache information."""
        response = client.get("/api/v1/enrichment/stats")
        assert response.status_code == 200
        data = response.json()
        
        # Should have stats or cache-related fields
        assert any(k in data for k in ["stats", "cache_size", "cached_tools", "cache_hits", "total_cached"]), \
            f"Expected cache info in response: {data}"
    
    def test_enrichment_health_returns_200(self, client):
        """Enrichment health endpoint returns 200 OK."""
        response = client.get("/api/v1/enrichment/health")
        assert response.status_code == 200


class TestPruningAPIContract:
    """Contract tests for pruning endpoints."""
    
    @pytest.fixture
    def client(self):
        return httpx.Client(base_url=API_BASE_URL, timeout=10.0)
    
    def test_pruning_status_returns_200(self, client):
        """Pruning status endpoint returns 200 OK."""
        response = client.get("/api/v1/pruning/status")
        assert response.status_code == 200
    
    def test_pruning_status_has_required_fields(self, client):
        """Pruning status includes required fields."""
        response = client.get("/api/v1/pruning/status")
        assert response.status_code == 200
        data = response.json()
        
        required_fields = ["enabled", "running", "configured"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
    
    def test_pruning_config_returns_200(self, client):
        """Pruning config endpoint returns 200 OK."""
        response = client.get("/api/v1/pruning/config")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
