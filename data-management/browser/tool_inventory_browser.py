"""
Tool Inventory Browser for Letta Tool Selector
Provides comprehensive browsing, search, and analysis of tool inventory with metadata.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter

import aiohttp
import aiofiles
import numpy as np
from tabulate import tabulate

from ..core.embedding_versioning import EmbeddingVersionManager, EmbeddingVersion
from ..monitoring.vector_quality_monitor import VectorQualityMonitor


@dataclass
class ToolMetadata:
    """Extended metadata for a tool."""
    tool_id: str
    name: str
    description: str
    source: str  # letta, mcp, custom
    category: str
    tags: List[str]
    usage_count: int
    last_used: Optional[str]
    creation_date: str
    update_date: str
    version_count: int
    quality_grade: Optional[str]
    embedding_dim: Optional[int]
    file_size: Optional[int]
    complexity_score: float
    dependencies: List[str]
    agents_using: List[str]


@dataclass
class InventoryStats:
    """Statistics about the tool inventory."""
    total_tools: int
    active_tools: int
    deprecated_tools: int
    source_breakdown: Dict[str, int]
    category_breakdown: Dict[str, int]
    quality_breakdown: Dict[str, int]
    avg_complexity: float
    total_storage_mb: float
    last_updated: str


class ToolInventoryBrowser:
    """Comprehensive tool inventory browser with advanced metadata and analytics."""
    
    def __init__(
        self,
        storage_path: str = "/opt/stacks/lettatoolsselector/data-management/browser",
        letta_api_url: str = "https://letta.oculair.ca/v1",
        letta_password: str = None
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.storage_path / "tool_metadata.json"
        self.stats_file = self.storage_path / "inventory_stats.json"
        self.cache_file = self.storage_path / "tool_cache.json"
        
        self.version_manager = EmbeddingVersionManager()
        self.quality_monitor = VectorQualityMonitor()
        
        self.letta_api_url = letta_api_url
        self.letta_password = letta_password or self._get_letta_password()
        
        self.logger = logging.getLogger(__name__)
        
        # Category mappings
        self.category_keywords = {
            "file_operations": ["file", "read", "write", "edit", "create", "delete", "copy", "move"],
            "web": ["web", "http", "api", "fetch", "request", "url", "browser"],
            "data": ["data", "json", "csv", "database", "query", "search", "filter"],
            "communication": ["message", "email", "chat", "notification", "send"],
            "system": ["system", "command", "bash", "terminal", "execute"],
            "ai": ["ai", "llm", "model", "generate", "predict", "analyze"],
            "utility": ["util", "helper", "convert", "format", "validate"],
            "integration": ["integrate", "connect", "sync", "webhook", "api"]
        }
    
    def _get_letta_password(self) -> str:
        """Get Letta password from environment."""
        import os
        return os.getenv('LETTA_PASSWORD', '')
    
    async def _fetch_letta_tools(self) -> List[Dict[str, Any]]:
        """Fetch tools from Letta API."""
        tools = []
        
        try:
            headers = {"Authorization": f"Bearer {self.letta_password}"}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.letta_api_url}/tools",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        tools = await response.json()
                    else:
                        self.logger.error(f"Failed to fetch Letta tools: {response.status}")
        
        except Exception as e:
            self.logger.error(f"Error fetching Letta tools: {e}")
        
        return tools
    
    async def _fetch_agent_tool_usage(self) -> Dict[str, List[str]]:
        """Fetch which agents are using which tools."""
        usage = defaultdict(list)
        
        try:
            headers = {"Authorization": f"Bearer {self.letta_password}"}
            
            async with aiohttp.ClientSession() as session:
                # Get all agents
                async with session.get(
                    f"{self.letta_api_url}/agents",
                    headers=headers
                ) as response:
                    if response.status == 200:
                        agents = await response.json()
                        
                        for agent in agents:
                            agent_id = agent.get('id', '')
                            agent_name = agent.get('name', f'Agent-{agent_id[:8]}')
                            
                            # Get tools for each agent
                            async with session.get(
                                f"{self.letta_api_url}/agents/{agent_id}/tools",
                                headers=headers
                            ) as tools_response:
                                if tools_response.status == 200:
                                    agent_tools = await tools_response.json()
                                    for tool in agent_tools:
                                        tool_id = tool.get('id', '')
                                        if tool_id:
                                            usage[tool_id].append(agent_name)
        
        except Exception as e:
            self.logger.error(f"Error fetching agent tool usage: {e}")
        
        return dict(usage)
    
    def _categorize_tool(self, tool_name: str, description: str) -> str:
        """Automatically categorize a tool based on name and description."""
        text = f"{tool_name} {description}".lower()
        
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "uncategorized"
    
    def _calculate_complexity_score(self, description: str, source_code: str = "") -> float:
        """Calculate complexity score based on description and source code."""
        # Simple heuristic based on description length and content
        score = 0.0
        
        # Base score from description length
        desc_length = len(description.split())
        if desc_length < 10:
            score += 0.2
        elif desc_length < 30:
            score += 0.5
        else:
            score += 0.8
        
        # Complexity indicators in description
        complexity_indicators = [
            "complex", "advanced", "multiple", "various", "comprehensive",
            "algorithm", "process", "analyze", "compute", "generate"
        ]
        
        desc_lower = description.lower()
        complexity_mentions = sum(1 for indicator in complexity_indicators if indicator in desc_lower)
        score += min(complexity_mentions * 0.2, 0.6)
        
        # Source code complexity (if available)
        if source_code:
            code_lines = len(source_code.split('\n'))
            if code_lines > 100:
                score += 0.3
            elif code_lines > 50:
                score += 0.2
            elif code_lines > 20:
                score += 0.1
        
        return min(score, 1.0)
    
    def _extract_tags(self, tool_name: str, description: str) -> List[str]:
        """Extract relevant tags from tool name and description."""
        tags = set()
        text = f"{tool_name} {description}".lower()
        
        # Common technical tags
        tag_patterns = [
            ("api", ["api", "rest", "http", "endpoint"]),
            ("database", ["db", "database", "sql", "query"]),
            ("file", ["file", "document", "pdf", "csv", "json"]),
            ("search", ["search", "find", "lookup", "filter"]),
            ("async", ["async", "concurrent", "parallel"]),
            ("web", ["web", "browser", "html", "url"]),
            ("auth", ["auth", "login", "password", "token"]),
            ("data", ["data", "information", "content"]),
            ("utility", ["util", "helper", "convert", "format"])
        ]
        
        for tag, keywords in tag_patterns:
            if any(keyword in text for keyword in keywords):
                tags.add(tag)
        
        return list(tags)
    
    async def _load_metadata(self) -> Dict[str, ToolMetadata]:
        """Load tool metadata from storage."""
        if not self.metadata_file.exists():
            return {}
        
        try:
            async with aiofiles.open(self.metadata_file, 'r') as f:
                data = json.loads(await f.read())
                return {
                    tool_id: ToolMetadata(**tool_data)
                    for tool_id, tool_data in data.items()
                }
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return {}
    
    async def _save_metadata(self, metadata: Dict[str, ToolMetadata]) -> None:
        """Save tool metadata to storage."""
        try:
            data = {tool_id: asdict(meta) for tool_id, meta in metadata.items()}
            async with aiofiles.open(self.metadata_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            raise
    
    async def refresh_inventory(self) -> None:
        """Refresh the complete tool inventory with latest data."""
        self.logger.info("Refreshing tool inventory...")
        
        # Fetch data
        letta_tools = await self._fetch_letta_tools()
        agent_usage = await self._fetch_agent_tool_usage()
        versions = await self.version_manager._load_versions()
        quality_metrics = await self.version_manager._load_metrics()
        
        metadata = {}
        
        for tool in letta_tools:
            tool_id = tool.get('id', '')
            if not tool_id:
                continue
            
            tool_name = tool.get('name', 'Unknown')
            description = tool.get('description', '')
            source_code = tool.get('source_code', '')
            
            # Determine source
            if tool.get('source_type') == 'python':
                source = "letta"
            elif 'mcp' in tool_name.lower() or 'mcp' in description.lower():
                source = "mcp"
            else:
                source = "custom"
            
            # Get version info
            tool_versions = [v for v in versions.values() if v.tool_id == tool_id]
            version_count = len(tool_versions)
            
            # Get latest quality grade
            quality_grade = None
            embedding_dim = None
            if tool_versions:
                latest_version = max(tool_versions, key=lambda v: v.created_at)
                embedding_dim = latest_version.embedding_dim
                
                if latest_version.version_id in quality_metrics:
                    quality_grade = quality_metrics[latest_version.version_id].quality_grade
            
            # Calculate file size (estimate from source code)
            file_size = len(source_code.encode('utf-8')) if source_code else 0
            
            # Extract metadata
            category = self._categorize_tool(tool_name, description)
            tags = self._extract_tags(tool_name, description)
            complexity_score = self._calculate_complexity_score(description, source_code)
            
            metadata[tool_id] = ToolMetadata(
                tool_id=tool_id,
                name=tool_name,
                description=description,
                source=source,
                category=category,
                tags=tags,
                usage_count=len(agent_usage.get(tool_id, [])),
                last_used=None,  # Would need separate tracking
                creation_date=tool.get('created_at', datetime.now(timezone.utc).isoformat()),
                update_date=tool.get('updated_at', datetime.now(timezone.utc).isoformat()),
                version_count=version_count,
                quality_grade=quality_grade,
                embedding_dim=embedding_dim,
                file_size=file_size,
                complexity_score=complexity_score,
                dependencies=[],  # Would need code analysis
                agents_using=agent_usage.get(tool_id, [])
            )
        
        # Save metadata
        await self._save_metadata(metadata)
        
        # Generate and save stats
        stats = await self._calculate_stats(metadata)
        await self._save_stats(stats)
        
        self.logger.info(f"Refreshed inventory with {len(metadata)} tools")
    
    async def _calculate_stats(self, metadata: Dict[str, ToolMetadata]) -> InventoryStats:
        """Calculate inventory statistics."""
        tools = list(metadata.values())
        
        if not tools:
            return InventoryStats(
                total_tools=0,
                active_tools=0,
                deprecated_tools=0,
                source_breakdown={},
                category_breakdown={},
                quality_breakdown={},
                avg_complexity=0.0,
                total_storage_mb=0.0,
                last_updated=datetime.now(timezone.utc).isoformat()
            )
        
        # Basic counts
        total_tools = len(tools)
        active_tools = sum(1 for t in tools if t.usage_count > 0)
        deprecated_tools = total_tools - active_tools
        
        # Breakdowns
        source_breakdown = dict(Counter(t.source for t in tools))
        category_breakdown = dict(Counter(t.category for t in tools))
        quality_breakdown = dict(Counter(t.quality_grade for t in tools if t.quality_grade))
        
        # Averages
        avg_complexity = sum(t.complexity_score for t in tools) / total_tools
        total_storage_mb = sum(t.file_size or 0 for t in tools) / (1024 * 1024)
        
        return InventoryStats(
            total_tools=total_tools,
            active_tools=active_tools,
            deprecated_tools=deprecated_tools,
            source_breakdown=source_breakdown,
            category_breakdown=category_breakdown,
            quality_breakdown=quality_breakdown,
            avg_complexity=avg_complexity,
            total_storage_mb=total_storage_mb,
            last_updated=datetime.now(timezone.utc).isoformat()
        )
    
    async def _save_stats(self, stats: InventoryStats) -> None:
        """Save inventory statistics."""
        try:
            async with aiofiles.open(self.stats_file, 'w') as f:
                await f.write(json.dumps(asdict(stats), indent=2))
        except Exception as e:
            self.logger.error(f"Error saving stats: {e}")
            raise
    
    async def search_tools(
        self,
        query: str = "",
        category: str = "",
        source: str = "",
        tags: List[str] = None,
        quality_grade: str = "",
        min_complexity: float = 0.0,
        max_complexity: float = 1.0,
        used_by_agents: bool = None,
        limit: int = 50
    ) -> List[ToolMetadata]:
        """Search tools with various filters."""
        metadata = await self._load_metadata()
        tools = list(metadata.values())
        
        # Apply filters
        if query:
            query_lower = query.lower()
            tools = [
                t for t in tools
                if query_lower in t.name.lower() or query_lower in t.description.lower()
            ]
        
        if category:
            tools = [t for t in tools if t.category == category]
        
        if source:
            tools = [t for t in tools if t.source == source]
        
        if tags:
            tools = [
                t for t in tools
                if any(tag in t.tags for tag in tags)
            ]
        
        if quality_grade:
            tools = [t for t in tools if t.quality_grade == quality_grade]
        
        if min_complexity is not None:
            tools = [t for t in tools if t.complexity_score >= min_complexity]
        
        if max_complexity is not None:
            tools = [t for t in tools if t.complexity_score <= max_complexity]
        
        if used_by_agents is not None:
            if used_by_agents:
                tools = [t for t in tools if t.agents_using]
            else:
                tools = [t for t in tools if not t.agents_using]
        
        # Sort by usage count and quality
        tools.sort(key=lambda t: (-t.usage_count, -(ord(t.quality_grade or 'Z') - ord('A'))))
        
        return tools[:limit]
    
    async def get_tool_details(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive details for a specific tool."""
        metadata = await self._load_metadata()
        tool_meta = metadata.get(tool_id)
        
        if not tool_meta:
            return None
        
        # Get version history
        versions = await self.version_manager.get_tool_versions(tool_id)
        
        # Get quality metrics
        quality_metrics = []
        for version in versions:
            metrics = await self.version_manager.get_version_metrics(version.version_id)
            if metrics:
                quality_metrics.append(asdict(metrics))
        
        return {
            "metadata": asdict(tool_meta),
            "versions": [asdict(v) for v in versions],
            "quality_metrics": quality_metrics,
            "embedding_history": len(versions),
            "quality_trend": [m["quality_grade"] for m in quality_metrics]
        }
    
    async def get_inventory_summary(self) -> Dict[str, Any]:
        """Get high-level inventory summary."""
        if not self.stats_file.exists():
            await self.refresh_inventory()
        
        try:
            async with aiofiles.open(self.stats_file, 'r') as f:
                stats_data = json.loads(await f.read())
                stats = InventoryStats(**stats_data)
            
            metadata = await self._load_metadata()
            
            # Top tools by usage
            tools = list(metadata.values())
            top_tools = sorted(tools, key=lambda t: -t.usage_count)[:10]
            
            # Recent activity
            recent_tools = sorted(
                tools, 
                key=lambda t: t.update_date or t.creation_date, 
                reverse=True
            )[:5]
            
            return {
                "stats": asdict(stats),
                "top_tools_by_usage": [
                    {"name": t.name, "usage_count": t.usage_count}
                    for t in top_tools
                ],
                "recently_updated": [
                    {"name": t.name, "update_date": t.update_date}
                    for t in recent_tools
                ],
                "recommendations": await self._get_recommendations(metadata, stats)
            }
        
        except Exception as e:
            self.logger.error(f"Error getting inventory summary: {e}")
            return {}
    
    async def _get_recommendations(
        self, 
        metadata: Dict[str, ToolMetadata], 
        stats: InventoryStats
    ) -> List[str]:
        """Generate inventory recommendations."""
        recommendations = []
        tools = list(metadata.values())
        
        # Unused tools
        unused_tools = [t for t in tools if t.usage_count == 0]
        if len(unused_tools) > 10:
            recommendations.append(f"Consider reviewing {len(unused_tools)} unused tools for cleanup")
        
        # Quality issues
        poor_quality = [t for t in tools if t.quality_grade in ['D', 'F']]
        if poor_quality:
            recommendations.append(f"Improve quality for {len(poor_quality)} low-grade embeddings")
        
        # Missing embeddings
        no_embeddings = [t for t in tools if t.version_count == 0]
        if no_embeddings:
            recommendations.append(f"Generate embeddings for {len(no_embeddings)} tools")
        
        # Category imbalance
        if stats.category_breakdown:
            max_category = max(stats.category_breakdown.values())
            total = sum(stats.category_breakdown.values())
            if max_category > total * 0.5:
                recommendations.append("Tool categories are imbalanced - consider diversifying")
        
        return recommendations
    
    def print_inventory_table(self, tools: List[ToolMetadata], detailed: bool = False) -> None:
        """Print tools in a formatted table."""
        if not tools:
            print("No tools found.")
            return
        
        if detailed:
            headers = [
                "Name", "Category", "Source", "Quality", "Usage", 
                "Complexity", "Agents", "Last Updated"
            ]
            rows = []
            for tool in tools:
                rows.append([
                    tool.name[:30] + "..." if len(tool.name) > 30 else tool.name,
                    tool.category,
                    tool.source,
                    tool.quality_grade or "N/A",
                    tool.usage_count,
                    f"{tool.complexity_score:.2f}",
                    len(tool.agents_using),
                    tool.update_date[:10] if tool.update_date else "N/A"
                ])
        else:
            headers = ["Name", "Category", "Usage", "Quality"]
            rows = []
            for tool in tools:
                rows.append([
                    tool.name[:40] + "..." if len(tool.name) > 40 else tool.name,
                    tool.category,
                    tool.usage_count,
                    tool.quality_grade or "N/A"
                ])
        
        print(tabulate(rows, headers=headers, tablefmt="grid"))


async def main():
    """Example usage and testing."""
    logging.basicConfig(level=logging.INFO)
    
    browser = ToolInventoryBrowser()
    
    print("Refreshing inventory...")
    await browser.refresh_inventory()
    
    print("\nInventory Summary:")
    summary = await browser.get_inventory_summary()
    stats = summary.get("stats", {})
    print(f"Total tools: {stats.get('total_tools', 0)}")
    print(f"Active tools: {stats.get('active_tools', 0)}")
    print(f"Average complexity: {stats.get('avg_complexity', 0):.2f}")
    
    print("\nTop categories:")
    for category, count in stats.get("category_breakdown", {}).items():
        print(f"  {category}: {count}")
    
    print("\nSearching for web tools...")
    web_tools = await browser.search_tools(category="web", limit=5)
    browser.print_inventory_table(web_tools)
    
    if web_tools:
        print(f"\nDetails for '{web_tools[0].name}':")
        details = await browser.get_tool_details(web_tools[0].tool_id)
        if details:
            print(f"  Versions: {details['embedding_history']}")
            print(f"  Quality trend: {details['quality_trend']}")


if __name__ == "__main__":
    asyncio.run(main())