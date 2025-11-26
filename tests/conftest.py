"""
Pytest configuration and shared fixtures for Letta Tool Selector tests
"""

import os
import sys
import pytest
import asyncio
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import Mock, AsyncMock, patch

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "lettaaugment-source"))
sys.path.insert(0, str(PROJECT_ROOT / "worker-service"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================================
# Session-scoped fixtures
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return PROJECT_ROOT / "tests" / "data"


# ============================================================================
# Configuration fixtures
# ============================================================================

@pytest.fixture
def mock_env_vars() -> Dict[str, str]:
    """Mock environment variables for testing."""
    return {
        "LETTA_API_URL": "https://test-letta.example.com/v1",
        "LETTA_PASSWORD": "test-password",
        "WEAVIATE_URL": "http://localhost:8080/",
        "OPENAI_API_KEY": "test-openai-key",
        "EMBEDDING_PROVIDER": "ollama",
        "OLLAMA_EMBEDDING_HOST": "localhost",
        "OLLAMA_EMBEDDING_MODEL": "test-model",
        "EMBEDDING_DIMENSION": "2560",
        "USE_OLLAMA_EMBEDDINGS": "true",
        "DEFAULT_DROP_RATE": "0.6",
        "MAX_TOTAL_TOOLS": "30",
        "MAX_MCP_TOOLS": "20",
        "MIN_MCP_TOOLS": "7",
        "EXCLUDE_LETTA_CORE_TOOLS": "true",
        "EXCLUDE_OFFICIAL_TOOLS": "true",
        "MANAGE_ONLY_MCP_TOOLS": "true",
        "DEFAULT_MIN_SCORE": "35.0",
        "NEVER_DETACH_TOOLS": "find_tools",
    }


@pytest.fixture
def set_test_env(mock_env_vars: Dict[str, str]) -> Generator[None, None, None]:
    """Set test environment variables."""
    original_env = os.environ.copy()
    os.environ.update(mock_env_vars)
    yield
    os.environ.clear()
    os.environ.update(original_env)


# ============================================================================
# Mock service fixtures
# ============================================================================

@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client."""
    client = Mock()
    client.is_ready.return_value = True
    client.query = Mock()
    client.data_object = Mock()
    client.schema = Mock()
    return client


@pytest.fixture
def mock_http_session():
    """Mock aiohttp ClientSession."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_letta_api_response():
    """Mock successful Letta API response."""
    return {
        "id": "agent-123",
        "name": "Test Agent",
        "tools": [
            {
                "id": "tool-1",
                "name": "test_tool",
                "description": "A test tool",
                "tool_type": "external_mcp"
            }
        ]
    }


# ============================================================================
# Sample data fixtures
# ============================================================================

@pytest.fixture
def sample_tool_data() -> Dict[str, Any]:
    """Sample tool data for testing."""
    return {
        "id": "tool-123",
        "name": "sample_tool",
        "description": "A sample tool for testing",
        "tool_type": "external_mcp",
        "source": "mcp_server",
        "tags": ["test", "sample"],
        "mcp_server_name": "test-server"
    }


@pytest.fixture
def sample_tools_list(sample_tool_data: Dict[str, Any]) -> list:
    """List of sample tools for testing."""
    return [
        sample_tool_data,
        {
            "id": "tool-456",
            "name": "another_tool",
            "description": "Another test tool",
            "tool_type": "external_mcp",
            "source": "mcp_server",
            "tags": ["test"],
            "mcp_server_name": "test-server"
        },
        {
            "id": "tool-789",
            "name": "letta_core_tool",
            "description": "Letta core tool",
            "tool_type": "letta_core",
            "source": "letta",
            "tags": ["core"]
        }
    ]


@pytest.fixture
def sample_search_results() -> list:
    """Sample search results from Weaviate."""
    return [
        {
            "id": "tool-1",
            "name": "web_search",
            "description": "Search the web",
            "score": 0.95,
            "distance": 0.05
        },
        {
            "id": "tool-2",
            "name": "file_reader",
            "description": "Read files",
            "score": 0.85,
            "distance": 0.15
        }
    ]


# ============================================================================
# Configuration validation fixtures
# ============================================================================

@pytest.fixture
def valid_config() -> Dict[str, Any]:
    """Valid configuration for testing."""
    return {
        "search": {
            "embedding": {
                "provider": "ollama",
                "model": "test-model",
                "dimension": 2560,
                "max_tokens": 8192
            },
            "weaviate": {
                "hybrid": {
                    "alpha": 0.75
                }
            }
        },
        "reranker": {
            "enabled": True,
            "type": "cross-encoder",
            "scoring": {
                "top_k": 10
            }
        },
        "experiments": {
            "cost_controls": {
                "daily_budget_usd": 10.0
            }
        },
        "evaluation": {
            "metrics": ["precision_at_k", "ndcg_at_k"]
        }
    }


@pytest.fixture
def invalid_config() -> Dict[str, Any]:
    """Invalid configuration for testing."""
    return {
        "search": {
            "embedding": {
                "provider": "invalid_provider",
                "dimension": -1,
                "max_tokens": 999999
            },
            "weaviate": {
                "hybrid": {
                    "alpha": 5.0  # Invalid: must be 0-1
                }
            }
        },
        "reranker": {
            "enabled": True,
            "type": "invalid_type",
            "scoring": {
                "top_k": 1000  # Invalid: max is 100
            }
        }
    }


# ============================================================================
# Cost control fixtures
# ============================================================================

@pytest.fixture
def sample_cost_entry():
    """Sample cost entry for testing."""
    from datetime import datetime
    return {
        "timestamp": datetime.now().isoformat(),
        "category": "embedding_api",
        "operation": "generate_embedding",
        "cost": 0.001,
        "currency": "USD",
        "metadata": {
            "model": "test-model",
            "tokens": 100
        }
    }


@pytest.fixture
def sample_budget_limit():
    """Sample budget limit for testing."""
    return {
        "category": "embedding_api",
        "period": "daily",
        "limit": 10.0,
        "currency": "USD",
        "alert_thresholds": [0.5, 0.8, 0.95],
        "hard_limit": False,
        "enabled": True
    }


# ============================================================================
# Pytest hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external services"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring services"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests requiring full system"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Auto-mark tests based on path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Auto-mark based on test name patterns
        if "weaviate" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_weaviate)
        if "api" in item.nodeid.lower() and "server" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_api_server)
        if "letta" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_letta)
        if "ollama" in item.nodeid.lower():
            item.add_marker(pytest.mark.requires_ollama)
