"""
Per-Agent Tool Pinning Service

Provides runtime, per-agent tool pinning so that specific tools survive
pruning cycles. Pinned tools can only be detached with explicit unpin.

Storage: JSON files in {cache_dir}/pinned_tools/{agent_id}.json
Format:  { "pinned_tool_ids": ["tool-uuid-1", ...], "updated_at": "ISO-timestamp" }

Usage:
    from services.pin_service import configure, get_pinned_tools, pin_tools, unpin_tools

    # Initialize (once at startup)
    configure(cache_dir="/app/runtime_cache")

    # Pin tools for an agent
    newly_pinned = await pin_tools("agent-123", ["tool-abc", "tool-def"])

    # Check pins
    pinned = await get_pinned_tools("agent-123")

    # Unpin
    removed = await unpin_tools("agent-123", ["tool-abc"])
"""

import os
import json
import logging
import asyncio
from typing import List, Set, Dict, Optional
from datetime import datetime, timezone

import aiofiles

logger = logging.getLogger(__name__)


# ============================================================================
# Module State
# ============================================================================

_cache_dir: Optional[str] = None
# TODO: _locks dict grows unbounded as agents accumulate. Consider LRU eviction or weakref if scaling past ~100 agents. (PM feedback)
_locks: Dict[str, asyncio.Lock] = {}


# ============================================================================
# Configuration
# ============================================================================

def configure(cache_dir: str):
    """
    Configure the pin service with the cache directory.

    Creates the pinned_tools subdirectory if it doesn't exist.

    Args:
        cache_dir: Base cache directory (e.g. /app/runtime_cache)
    """
    global _cache_dir
    _cache_dir = cache_dir
    pin_dir = _get_pin_dir()
    os.makedirs(pin_dir, exist_ok=True)
    logger.info(f"Pin service configured with directory: {pin_dir}")


def _get_pin_dir() -> str:
    """Get the pinned tools directory path."""
    base = _cache_dir or os.getenv('CACHE_DIR', '/app/runtime_cache')
    return os.path.join(base, 'pinned_tools')


def _get_pin_file(agent_id: str) -> str:
    """Get the pin file path for an agent."""
    return os.path.join(_get_pin_dir(), f"{agent_id}.json")


def _get_lock(agent_id: str) -> asyncio.Lock:
    """Get or create a lock for an agent's pin file."""
    if agent_id not in _locks:
        _locks[agent_id] = asyncio.Lock()
    return _locks[agent_id]


# ============================================================================
# Internal I/O
# ============================================================================

async def _read_pins(agent_id: str) -> Set[str]:
    """Read pinned tool IDs for an agent from disk."""
    pin_file = _get_pin_file(agent_id)
    if not os.path.exists(pin_file):
        return set()
    try:
        async with aiofiles.open(pin_file, 'r') as f:
            content = await f.read()
            data = json.loads(content) if content else {}
            return set(data.get('pinned_tool_ids', []))
    except Exception as e:
        logger.error(f"Error reading pins for agent {agent_id}: {e}")
        return set()


async def _write_pins(agent_id: str, tool_ids: Set[str]):
    """Write pinned tool IDs for an agent to disk."""
    pin_file = _get_pin_file(agent_id)
    os.makedirs(os.path.dirname(pin_file), exist_ok=True)
    data = {
        'pinned_tool_ids': sorted(list(tool_ids)),
        'updated_at': datetime.now(timezone.utc).isoformat()
    }
    try:
        async with aiofiles.open(pin_file, 'w') as f:
            await f.write(json.dumps(data, indent=2))
    except Exception as e:
        logger.error(f"Error writing pins for agent {agent_id}: {e}")
        raise


# ============================================================================
# Public API
# ============================================================================

async def get_pinned_tools(agent_id: str) -> List[str]:
    """
    Get list of pinned tool IDs for an agent.

    Args:
        agent_id: The agent ID

    Returns:
        Sorted list of pinned tool IDs
    """
    async with _get_lock(agent_id):
        pins = await _read_pins(agent_id)
        return sorted(list(pins))


async def pin_tools(agent_id: str, tool_ids: List[str]) -> List[str]:
    """
    Add tools to an agent's pinned set.

    Args:
        agent_id: The agent ID
        tool_ids: Tool IDs to pin

    Returns:
        List of newly pinned tool IDs (excludes already-pinned ones)
    """
    async with _get_lock(agent_id):
        current = await _read_pins(agent_id)
        new_pins = set(tool_ids) - current
        if new_pins:
            current.update(new_pins)
            await _write_pins(agent_id, current)
            logger.info(f"Pinned {len(new_pins)} tools for agent {agent_id}: {new_pins}")
        return sorted(list(new_pins))


async def unpin_tools(agent_id: str, tool_ids: List[str]) -> List[str]:
    """
    Remove tools from an agent's pinned set.

    Args:
        agent_id: The agent ID
        tool_ids: Tool IDs to unpin

    Returns:
        List of unpinned tool IDs (excludes those that weren't pinned)
    """
    async with _get_lock(agent_id):
        current = await _read_pins(agent_id)
        removed = current & set(tool_ids)
        if removed:
            current -= removed
            await _write_pins(agent_id, current)
            logger.info(f"Unpinned {len(removed)} tools for agent {agent_id}: {removed}")
        return sorted(list(removed))


async def is_pinned(agent_id: str, tool_id: str) -> bool:
    """
    Check if a specific tool is pinned for an agent.

    Args:
        agent_id: The agent ID
        tool_id: The tool ID to check

    Returns:
        True if the tool is pinned
    """
    async with _get_lock(agent_id):
        pins = await _read_pins(agent_id)
        return tool_id in pins


async def clear_pins(agent_id: str) -> int:
    """
    Clear all pins for an agent.

    Args:
        agent_id: The agent ID

    Returns:
        Number of pins that were cleared
    """
    async with _get_lock(agent_id):
        current = await _read_pins(agent_id)
        count = len(current)
        if count > 0:
            await _write_pins(agent_id, set())
            logger.info(f"Cleared {count} pins for agent {agent_id}")
        return count
