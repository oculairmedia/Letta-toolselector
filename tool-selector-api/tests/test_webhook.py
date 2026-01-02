"""
Webhook Integration Tests

Tests for the /webhook/letta endpoint that receives tool events from Letta.
Run with: pytest tests/test_webhook.py -v
"""

import pytest
import json
import hmac
import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestWebhookSignatureVerification:
    """Test HMAC signature verification."""
    
    def test_verify_signature_valid(self):
        """Valid signature should pass verification."""
        from routes.webhook import verify_signature, _config
        
        secret = "test-secret-123"
        _config['webhook_secret'] = secret
        
        payload = b'{"event_type":"tool.created","data":{}}'
        expected_sig = hmac.new(secret.encode(), payload, hashlib.sha256).hexdigest()
        
        assert verify_signature(payload, f"sha256={expected_sig}") is True
    
    def test_verify_signature_invalid(self):
        """Invalid signature should fail verification."""
        from routes.webhook import verify_signature, _config
        
        _config['webhook_secret'] = "test-secret-123"
        payload = b'{"event_type":"tool.created","data":{}}'
        
        assert verify_signature(payload, "sha256=invalid-signature") is False
    
    def test_verify_signature_no_secret_configured(self):
        """No secret configured should pass (warning logged)."""
        from routes.webhook import verify_signature, _config
        
        _config['webhook_secret'] = None
        payload = b'{"event_type":"tool.created","data":{}}'
        
        # Should pass when no secret is configured
        assert verify_signature(payload, "") is True
    
    def test_verify_signature_missing_header(self):
        """Missing signature header should fail when secret is configured."""
        from routes.webhook import verify_signature, _config
        
        _config['webhook_secret'] = "test-secret-123"
        payload = b'{"event_type":"tool.created","data":{}}'
        
        assert verify_signature(payload, "") is False


class TestWebhookCacheOperations:
    """Test tool cache read/write operations."""
    
    @pytest.mark.asyncio
    async def test_update_cache_tool_created(self, tmp_path):
        """tool.created should add tool to cache."""
        from routes.webhook import update_cache_for_tool, read_tool_cache, write_tool_cache, _config
        
        # Setup temp cache
        cache_file = tmp_path / "tool_cache.json"
        _config['cache_dir'] = str(tmp_path)
        _config['tool_cache_file'] = str(cache_file)
        
        # Start with empty cache
        await write_tool_cache([])
        
        # Add a tool
        tool_data = {
            "tool_id": "tool-123",
            "tool_name": "new_test_tool",
            "description": "A test tool",
            "tool_type": "python",
            "tags": ["test"]
        }
        
        await update_cache_for_tool("tool.created", tool_data)
        
        # Verify tool was added
        tools = await read_tool_cache()
        assert len(tools) == 1
        assert tools[0]["name"] == "new_test_tool"
        assert tools[0]["id"] == "tool-123"
    
    @pytest.mark.asyncio
    async def test_update_cache_tool_deleted(self, tmp_path):
        """tool.deleted should remove tool from cache."""
        from routes.webhook import update_cache_for_tool, read_tool_cache, write_tool_cache, _config
        
        cache_file = tmp_path / "tool_cache.json"
        _config['cache_dir'] = str(tmp_path)
        _config['tool_cache_file'] = str(cache_file)
        
        # Start with one tool
        await write_tool_cache([{
            "id": "tool-123",
            "name": "existing_tool",
            "description": "Existing"
        }])
        
        # Delete the tool
        await update_cache_for_tool("tool.deleted", {
            "tool_id": "tool-123",
            "tool_name": "existing_tool"
        })
        
        # Verify tool was removed
        tools = await read_tool_cache()
        assert len(tools) == 0
    
    @pytest.mark.asyncio
    async def test_update_cache_tool_updated(self, tmp_path):
        """tool.updated should update existing tool in cache."""
        from routes.webhook import update_cache_for_tool, read_tool_cache, write_tool_cache, _config
        
        cache_file = tmp_path / "tool_cache.json"
        _config['cache_dir'] = str(tmp_path)
        _config['tool_cache_file'] = str(cache_file)
        
        # Start with one tool
        await write_tool_cache([{
            "id": "tool-123",
            "name": "my_tool",
            "description": "Old description"
        }])
        
        # Update the tool
        await update_cache_for_tool("tool.updated", {
            "tool_id": "tool-123",
            "tool_name": "my_tool",
            "description": "New description"
        })
        
        # Verify tool was updated
        tools = await read_tool_cache()
        assert len(tools) == 1
        assert tools[0]["description"] == "New description"


class TestWebhookPayloads:
    """Test webhook payload structures match Letta's format."""
    
    def test_tool_created_payload_structure(self):
        """Verify expected tool.created payload structure."""
        payload = {
            "event_type": "tool.created",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "tool_id": "tool-abc123",
                "tool_name": "my_new_tool",
                "tool_type": "python",
                "description": "Does something useful",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"}
                    }
                },
                "tags": ["mcp:my_server"]
            }
        }
        
        # Validate structure
        assert payload["event_type"] == "tool.created"
        assert "tool_id" in payload["data"]
        assert "tool_name" in payload["data"]
    
    def test_tool_deleted_payload_structure(self):
        """Verify expected tool.deleted payload structure."""
        payload = {
            "event_type": "tool.deleted",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "tool_id": "tool-abc123",
                "tool_name": "deleted_tool",
                "tool_type": "python",
                "description": "Was useful",
                "json_schema": {},
                "tags": []
            }
        }
        
        assert payload["event_type"] == "tool.deleted"
        assert "tool_name" in payload["data"]


@pytest.fixture
def mock_weaviate_client():
    """Mock Weaviate client for testing."""
    client = MagicMock()
    collection = MagicMock()
    client.collections.get.return_value = collection
    collection.data.insert = MagicMock()
    collection.data.delete_many = MagicMock(return_value=MagicMock(successful=1))
    return client


class TestWeaviateOperations:
    """Test Weaviate index operations."""
    
    @pytest.mark.asyncio
    async def test_add_tool_to_weaviate(self, mock_weaviate_client):
        """Adding tool should insert into Weaviate collection."""
        from routes.webhook import add_tool_to_weaviate, _config
        
        _config['weaviate_client'] = mock_weaviate_client
        
        tool_data = {
            "tool_id": "tool-123",
            "tool_name": "test_tool",
            "description": "Test description",
            "tool_type": "python",
            "json_schema": {},
            "tags": ["test"]
        }
        
        with patch('asyncio.to_thread', new_callable=lambda: AsyncMock(side_effect=lambda f, *a, **k: f(*a, **k))):
            result = await add_tool_to_weaviate(tool_data)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_delete_tool_from_weaviate(self, mock_weaviate_client):
        """Deleting tool should remove from Weaviate collection."""
        from routes.webhook import delete_tool_from_weaviate, _config
        
        _config['weaviate_client'] = mock_weaviate_client
        
        with patch('asyncio.to_thread', new_callable=lambda: AsyncMock(side_effect=lambda f, *a, **k: f(*a, **k))):
            result = await delete_tool_from_weaviate("test_tool")
        
        assert result is True
