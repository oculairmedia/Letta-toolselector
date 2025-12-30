"""
Scheduled Agent Tool Pruning Job

Periodically prunes MCP tools from agents to keep tool counts manageable.

SAFETY RULES:
1. Only prunes MCP tools (tool_type == 'external_mcp' or custom non-core tools)
2. Never touches Letta core tools (is_letta_core_tool() == True)
3. Respects NEVER_DETACH_TOOLS env var (default: 'find_tools')
4. Respects should_protect_tool() from config
5. Enforces MIN_MCP_TOOLS threshold per agent
6. Dry-run mode available for safety testing
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PruningSchedulerConfig:
    """Configuration for the pruning scheduler."""
    
    # Schedule settings
    enabled: bool = False
    interval_hours: float = 24.0  # How often to run (default: daily)
    
    # Pruning behavior
    drop_rate: float = 0.3  # What fraction of low-relevance MCP tools to drop
    dry_run: bool = True  # If True, log what would be pruned but don't actually prune
    
    # Safety settings
    min_mcp_tools: int = 5  # Minimum MCP tools to keep per agent
    skip_agents: Set[str] = field(default_factory=set)  # Agent IDs to never prune
    
    # Rate limiting
    batch_size: int = 10  # How many agents to process per batch
    batch_delay_seconds: float = 2.0  # Delay between batches
    
    @classmethod
    def from_env(cls) -> "PruningSchedulerConfig":
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("PRUNING_SCHEDULER_ENABLED", "false").lower() == "true",
            interval_hours=float(os.getenv("PRUNING_SCHEDULER_INTERVAL_HOURS", "24")),
            drop_rate=float(os.getenv("PRUNING_SCHEDULER_DROP_RATE", "0.3")),
            dry_run=os.getenv("PRUNING_SCHEDULER_DRY_RUN", "true").lower() == "true",
            min_mcp_tools=int(os.getenv("PRUNING_SCHEDULER_MIN_MCP_TOOLS", "5")),
            skip_agents=set(filter(None, os.getenv("PRUNING_SCHEDULER_SKIP_AGENTS", "").split(","))),
            batch_size=int(os.getenv("PRUNING_SCHEDULER_BATCH_SIZE", "10")),
            batch_delay_seconds=float(os.getenv("PRUNING_SCHEDULER_BATCH_DELAY", "2.0")),
        )


@dataclass
class PruningResult:
    """Result of a single agent pruning operation."""
    agent_id: str
    success: bool
    dry_run: bool
    mcp_tools_before: int = 0
    mcp_tools_after: int = 0
    tools_pruned: int = 0
    tools_protected: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skipped_reason: Optional[str] = None


@dataclass
class SchedulerRunResult:
    """Result of a full scheduler run."""
    started_at: datetime
    completed_at: Optional[datetime] = None
    agents_processed: int = 0
    agents_skipped: int = 0
    agents_failed: int = 0
    total_tools_pruned: int = 0
    dry_run: bool = True
    results: List[PruningResult] = field(default_factory=list)
    error: Optional[str] = None


class PruningScheduler:
    """
    Scheduled job that periodically prunes MCP tools from agents.
    
    Safety guarantees:
    - Only MCP tools are ever pruned
    - Core/Letta tools are always preserved
    - Protected tools (NEVER_DETACH_TOOLS) are never removed
    - MIN_MCP_TOOLS threshold is always respected
    - Dry-run mode for testing
    """
    
    def __init__(
        self,
        config: Optional[PruningSchedulerConfig] = None,
        list_agents_func: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        prune_agent_func: Optional[Callable[..., Dict[str, Any]]] = None,
        get_agent_context_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the scheduler.
        
        Args:
            config: Scheduler configuration (defaults to env-based config)
            list_agents_func: Function to list all agents (returns list of {id, name, ...})
            prune_agent_func: Function to prune an agent's tools (the prune_agent_tools function)
            get_agent_context_func: Function to get agent's current context/prompt for relevance scoring
        """
        self.config = config or PruningSchedulerConfig.from_env()
        self._list_agents = list_agents_func
        self._prune_agent = prune_agent_func
        self._get_agent_context = get_agent_context_func
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_run: Optional[SchedulerRunResult] = None
        self._next_run: Optional[datetime] = None
        
        # Stats
        self._total_runs = 0
        self._total_tools_pruned = 0
    
    def configure(
        self,
        list_agents_func: Optional[Callable[[], List[Dict[str, Any]]]] = None,
        prune_agent_func: Optional[Callable[..., Dict[str, Any]]] = None,
        get_agent_context_func: Optional[Callable[[str], str]] = None,
    ) -> None:
        """Configure the scheduler with required functions."""
        if list_agents_func:
            self._list_agents = list_agents_func
        if prune_agent_func:
            self._prune_agent = prune_agent_func
        if get_agent_context_func:
            self._get_agent_context = get_agent_context_func
    
    def is_configured(self) -> bool:
        """Check if required functions are configured."""
        return self._list_agents is not None and self._prune_agent is not None
    
    # =========================================================================
    # Safety helper methods - explicitly check tool types and protection
    # =========================================================================
    
    def _is_mcp_tool(self, tool: Dict[str, Any]) -> bool:
        """Check if a tool is an MCP tool (safe to prune)."""
        tool_type = tool.get("tool_type", "")
        tags = tool.get("tags", [])
        
        # External MCP tools
        if tool_type == "external_mcp":
            return True
        
        # Tools with mcp: tag prefix
        if any(str(tag).startswith("mcp:") for tag in tags):
            return True
        
        return False
    
    def _is_letta_core_tool(self, tool: Dict[str, Any]) -> bool:
        """Check if a tool is a Letta core tool (NEVER prune)."""
        tool_type = tool.get("tool_type", "")
        core_types = {
            "letta_core",
            "letta_memory_core",
            "letta_multi_agent_core",
            "letta_sleeptime_core",
            "letta_voice_sleeptime_core",
            "letta_files_core",
            "letta_builtin",
        }
        return tool_type in core_types
    
    def _is_protected_tool(self, tool_name: str) -> bool:
        """Check if a tool is protected via NEVER_DETACH_TOOLS."""
        protected = os.getenv("NEVER_DETACH_TOOLS", "find_tools").split(",")
        protected = [t.strip() for t in protected if t.strip()]
        return tool_name in protected
    
    def _get_prunable_mcp_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter tools to only those safe to prune (MCP, not protected, not core)."""
        prunable = []
        for tool in tools:
            name = tool.get("name", "")
            
            # Skip Letta core tools
            if self._is_letta_core_tool(tool):
                continue
            
            # Skip protected tools
            if self._is_protected_tool(name):
                continue
            
            # Only include MCP tools
            if self._is_mcp_tool(tool):
                prunable.append(tool)
        
        return prunable
    
    def _calculate_prune_count(self, total_prunable: int) -> int:
        """Calculate how many tools to prune while respecting min_mcp_tools."""
        if total_prunable <= self.config.min_mcp_tools:
            return 0
        
        # Calculate based on drop rate
        to_prune = int(total_prunable * self.config.drop_rate)
        
        # Ensure we don't go below minimum
        max_prunable = total_prunable - self.config.min_mcp_tools
        return min(to_prune, max_prunable)
    
    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            logger.warning("Pruning scheduler already running")
            return
        
        if not self.config.enabled:
            logger.info("Pruning scheduler is disabled (PRUNING_SCHEDULER_ENABLED=false)")
            return
        
        if not self.is_configured():
            logger.error("Pruning scheduler not configured - missing list_agents or prune_agent functions")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Pruning scheduler started (interval={self.config.interval_hours}h, "
            f"dry_run={self.config.dry_run}, drop_rate={self.config.drop_rate})"
        )
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Pruning scheduler stopped")
    
    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                self._next_run = datetime.now() + timedelta(hours=self.config.interval_hours)
                
                # Wait for next run
                await asyncio.sleep(self.config.interval_hours * 3600)
                
                if self._running:
                    await self.run_now()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in pruning scheduler loop: {e}")
                # Wait a bit before retrying
                await asyncio.sleep(60)
    
    async def run_now(self, dry_run: Optional[bool] = None) -> SchedulerRunResult:
        """
        Run pruning immediately for all agents.
        
        Args:
            dry_run: Override dry_run setting (None = use config)
            
        Returns:
            SchedulerRunResult with details of the run
        """
        if not self.is_configured():
            return SchedulerRunResult(
                started_at=datetime.now(),
                completed_at=datetime.now(),
                error="Scheduler not configured"
            )
        
        use_dry_run = dry_run if dry_run is not None else self.config.dry_run
        
        result = SchedulerRunResult(
            started_at=datetime.now(),
            dry_run=use_dry_run
        )
        
        logger.info(f"Starting scheduled pruning run (dry_run={use_dry_run})")
        
        try:
            # Get all agents
            if self._list_agents is None:
                return SchedulerRunResult(
                    started_at=datetime.now(),
                    completed_at=datetime.now(),
                    error="list_agents function not configured"
                )
            agents = await asyncio.to_thread(self._list_agents)
            logger.info(f"Found {len(agents)} agents to process")
            
            # Process in batches
            for i in range(0, len(agents), self.config.batch_size):
                batch = agents[i:i + self.config.batch_size]
                
                for agent in batch:
                    agent_id = agent.get('id') or agent.get('agent_id')
                    agent_name = agent.get('name', 'Unknown')
                    
                    if not agent_id:
                        logger.warning(f"Agent without ID: {agent_name}")
                        continue
                    
                    # Check if agent should be skipped
                    if agent_id in self.config.skip_agents:
                        result.agents_skipped += 1
                        result.results.append(PruningResult(
                            agent_id=agent_id,
                            success=True,
                            dry_run=use_dry_run,
                            skipped_reason="Agent in skip list"
                        ))
                        continue
                    
                    # Prune this agent
                    agent_result = await self._prune_single_agent(
                        agent_id=agent_id,
                        agent_name=agent_name,
                        dry_run=use_dry_run
                    )
                    
                    result.results.append(agent_result)
                    
                    if agent_result.success:
                        result.agents_processed += 1
                        result.total_tools_pruned += agent_result.tools_pruned
                    elif agent_result.skipped_reason:
                        result.agents_skipped += 1
                    else:
                        result.agents_failed += 1
                
                # Delay between batches
                if i + self.config.batch_size < len(agents):
                    await asyncio.sleep(self.config.batch_delay_seconds)
            
            result.completed_at = datetime.now()
            
        except Exception as e:
            logger.exception(f"Error during scheduled pruning: {e}")
            result.error = str(e)
            result.completed_at = datetime.now()
        
        # Update stats
        self._last_run = result
        self._total_runs += 1
        self._total_tools_pruned += result.total_tools_pruned
        
        logger.info(
            f"Pruning run completed: {result.agents_processed} agents processed, "
            f"{result.agents_skipped} skipped, {result.agents_failed} failed, "
            f"{result.total_tools_pruned} tools pruned (dry_run={use_dry_run})"
        )
        
        return result
    
    async def _prune_single_agent(
        self,
        agent_id: str,
        agent_name: str,
        dry_run: bool
    ) -> PruningResult:
        """Prune MCP tools from a single agent."""
        result = PruningResult(
            agent_id=agent_id,
            success=False,
            dry_run=dry_run
        )
        
        try:
            # Get agent context for relevance scoring
            context = ""
            if self._get_agent_context:
                try:
                    context = await asyncio.to_thread(self._get_agent_context, agent_id)
                except Exception as e:
                    logger.debug(f"Could not get context for agent {agent_id}: {e}")
                    context = f"Agent: {agent_name}"  # Fallback
            else:
                context = f"Agent: {agent_name}"
            
            # Call the prune function
            # Note: prune_agent_tools already handles:
            # - MCP-only pruning
            # - Protected tools
            # - MIN_MCP_TOOLS threshold
            if self._prune_agent is None:
                result.error = "prune_agent function not configured"
                return result
            
            prune_result = self._prune_agent(
                agent_id=agent_id,
                user_prompt=context,
                drop_rate=self.config.drop_rate,
                dry_run=dry_run
            )
            # Handle both sync and async prune functions
            if asyncio.iscoroutine(prune_result):
                prune_result = await prune_result
            
            if prune_result.get('success'):
                details = prune_result.get('details', {})
                result.success = True
                result.mcp_tools_before = details.get('mcp_tools_on_agent_before', 0)
                result.mcp_tools_after = details.get('mcp_tools_on_agent_before', 0) - details.get('mcp_tools_detached_count', 0)
                result.tools_pruned = details.get('mcp_tools_detached_count', 0)
                result.tools_protected = details.get('protected_tool_names', [])
                
                # Check if it was skipped due to minimum tools
                if 'minimum required' in prune_result.get('message', '').lower():
                    result.skipped_reason = prune_result.get('message')
                    result.tools_pruned = 0
                    
            else:
                result.error = prune_result.get('error', 'Unknown error')
                
        except Exception as e:
            logger.exception(f"Error pruning agent {agent_id}: {e}")
            result.error = str(e)
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            "enabled": self.config.enabled,
            "running": self._running,
            "configured": self.is_configured(),
            "dry_run": self.config.dry_run,
            "interval_hours": self.config.interval_hours,
            "drop_rate": self.config.drop_rate,
            "min_mcp_tools": self.config.min_mcp_tools,
            "skip_agents_count": len(self.config.skip_agents),
            "next_run": self._next_run.isoformat() if self._next_run else None,
            "last_run": {
                "started_at": self._last_run.started_at.isoformat() if self._last_run else None,
                "completed_at": self._last_run.completed_at.isoformat() if self._last_run and self._last_run.completed_at else None,
                "agents_processed": self._last_run.agents_processed if self._last_run else 0,
                "agents_skipped": self._last_run.agents_skipped if self._last_run else 0,
                "agents_failed": self._last_run.agents_failed if self._last_run else 0,
                "total_tools_pruned": self._last_run.total_tools_pruned if self._last_run else 0,
                "dry_run": self._last_run.dry_run if self._last_run else None,
                "error": self._last_run.error if self._last_run else None,
            } if self._last_run else None,
            "stats": {
                "total_runs": self._total_runs,
                "total_tools_pruned": self._total_tools_pruned,
            }
        }


# Global scheduler instance
_scheduler: Optional[PruningScheduler] = None


def get_pruning_scheduler() -> PruningScheduler:
    """Get or create the global pruning scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = PruningScheduler()
    return _scheduler


async def start_pruning_scheduler() -> None:
    """Start the global pruning scheduler."""
    scheduler = get_pruning_scheduler()
    await scheduler.start()


async def stop_pruning_scheduler() -> None:
    """Stop the global pruning scheduler."""
    scheduler = get_pruning_scheduler()
    await scheduler.stop()
