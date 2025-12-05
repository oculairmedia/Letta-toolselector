"""Configuration management for Letta Toolkit."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class LettaConfig:
    """Configuration for Letta API connections.
    
    Attributes:
        base_url: Base URL for Letta API (e.g., https://letta.example.com)
        api_key: API key or password for authentication
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for failed requests
        protected_tools: List of tool names that should never be detached
    """
    
    base_url: str = field(default_factory=lambda: os.environ.get(
        "LETTA_BASE_URL", "https://letta.oculair.ca"
    ))
    api_key: str = field(default_factory=lambda: os.environ.get(
        "LETTA_PASSWORD", ""
    ))
    timeout: int = field(default_factory=lambda: int(os.environ.get(
        "LETTA_TIMEOUT", "30"
    )))
    max_retries: int = field(default_factory=lambda: int(os.environ.get(
        "LETTA_MAX_RETRIES", "3"
    )))
    protected_tools: list[str] = field(default_factory=lambda: [
        t.strip() for t in os.environ.get(
            "PROTECTED_TOOLS", "find_agents,find_tools,matrix_messaging"
        ).split(",") if t.strip()
    ])
    
    # Tool limits
    max_total_tools: int = field(default_factory=lambda: int(os.environ.get(
        "MAX_TOTAL_TOOLS", "20"
    )))
    
    # Pagination defaults
    default_page_limit: ClassVar[int] = 500
    
    @property
    def api_url(self) -> str:
        """Get the full API URL with /v1 suffix."""
        base = self.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base
    
    @property
    def headers(self) -> dict[str, str]:
        """Get standard API headers."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-BARE-PASSWORD"] = f"password {self.api_key}"
        return headers


# Global default config instance
_default_config: LettaConfig | None = None


def get_config() -> LettaConfig:
    """Get the global config instance, creating if needed."""
    global _default_config
    if _default_config is None:
        _default_config = LettaConfig()
    return _default_config


def set_config(config: LettaConfig) -> None:
    """Set the global config instance."""
    global _default_config
    _default_config = config
