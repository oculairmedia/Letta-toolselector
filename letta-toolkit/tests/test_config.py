"""Tests for config module."""

import os
from unittest.mock import patch

import pytest

from letta_toolkit.config import LettaConfig, get_config, set_config


class TestLettaConfig:
    """Tests for LettaConfig dataclass."""

    def test_default_values(self):
        """Test config has sensible defaults."""
        config = LettaConfig()
        
        assert config.base_url is not None
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.max_total_tools == 20

    def test_api_url_adds_v1_suffix(self):
        """Test api_url property adds /v1 suffix."""
        config = LettaConfig(base_url="https://letta.example.com")
        
        assert config.api_url == "https://letta.example.com/v1"

    def test_api_url_doesnt_double_v1(self):
        """Test api_url doesn't add /v1 if already present."""
        config = LettaConfig(base_url="https://letta.example.com/v1")
        
        assert config.api_url == "https://letta.example.com/v1"

    def test_headers_include_auth_when_api_key_set(self):
        """Test headers include auth when API key is set."""
        config = LettaConfig(api_key="test-key")
        
        assert "Authorization" in config.headers
        assert config.headers["Authorization"] == "Bearer test-key"

    def test_headers_no_auth_when_api_key_empty(self):
        """Test headers don't include auth when API key is empty."""
        config = LettaConfig(api_key="")
        
        assert "Authorization" not in config.headers

    @patch.dict(os.environ, {"LETTA_BASE_URL": "https://custom.letta.com"})
    def test_reads_from_environment(self):
        """Test config reads from environment variables."""
        config = LettaConfig()
        
        assert config.base_url == "https://custom.letta.com"

    @patch.dict(os.environ, {"PROTECTED_TOOLS": "tool_a,tool_b,tool_c"})
    def test_protected_tools_from_env(self):
        """Test protected tools parsed from environment."""
        config = LettaConfig()
        
        assert config.protected_tools == ["tool_a", "tool_b", "tool_c"]


class TestGetSetConfig:
    """Tests for global config functions."""

    def test_get_config_returns_default(self):
        """Test get_config returns a config instance."""
        config = get_config()
        
        assert isinstance(config, LettaConfig)

    def test_set_config_overrides_default(self):
        """Test set_config overrides the global config."""
        custom = LettaConfig(base_url="https://custom.com")
        set_config(custom)
        
        result = get_config()
        
        assert result.base_url == "https://custom.com"
