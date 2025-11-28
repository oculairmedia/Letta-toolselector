"""Letta Toolkit - Shared utilities for Letta agent tool management."""

from letta_toolkit.client import LettaClient
from letta_toolkit.tools import (
    attach_tool_to_agent,
    batch_attach_tools,
    batch_detach_tools,
    detach_tool_from_agent,
    list_agent_tools,
)
from letta_toolkit.protected import ensure_protected_tools

__version__ = "0.1.0"

__all__ = [
    "LettaClient",
    "list_agent_tools",
    "attach_tool_to_agent",
    "detach_tool_from_agent",
    "batch_attach_tools",
    "batch_detach_tools",
    "ensure_protected_tools",
]
