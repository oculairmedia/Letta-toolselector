#!/usr/bin/env python3
"""
Enhancement Cache System

Persistent storage for LLM-enhanced tool descriptions to avoid regeneration.
Saves valuable processing time and maintains consistency across restarts.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class EnhancementCache:
    """
    Manages persistent cache of enhanced tool descriptions.
    
    Features:
    - Hash-based cache keys for change detection
    - Timestamp tracking for cache invalidation
    - JSON-based storage for easy inspection
    - Fallback mechanisms for cache misses
    """
    
    def __init__(self, cache_dir: str = "enhancement_cache"):
        """
        Initialize enhancement cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.cache_file = self.cache_dir / "enhanced_descriptions.json"
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        
        # Load existing cache
        self._cache = self._load_cache()
        self._metadata = self._load_metadata()
        
        logger.info(f"Enhancement cache initialized: {len(self._cache)} cached descriptions")
    
    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"Loaded {len(cache)} cached enhancements")
                return cache
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache metadata: {e}")
        
        return {
            "cache_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_updated": None,
            "total_enhancements": 0,
            "ollama_model": "gemma3:12b",
            "enhancement_stats": {}
        }
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
            
            # Update metadata
            self._metadata["last_updated"] = datetime.now(timezone.utc).isoformat()
            self._metadata["total_enhancements"] = len(self._cache)
            
            with open(self.metadata_file, 'w') as f:
                json.dump(self._metadata, f, indent=2)
                
            logger.debug(f"Saved cache with {len(self._cache)} entries")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_tool_hash(self, tool: Dict[str, Any]) -> str:
        """
        Generate hash key for tool based on content that affects enhancement.
        
        Args:
            tool: Tool dictionary
            
        Returns:
            SHA256 hash string
        """
        # Include fields that affect enhancement quality
        key_fields = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "tool_type": tool.get("tool_type", ""),
            "mcp_server_name": tool.get("mcp_server_name", ""),
            "category": tool.get("category", ""),
            "parameters": tool.get("json_schema", {}).get("parameters", {}),
            "tags": sorted(tool.get("tags", []))  # Sort for consistent hashing
        }
        
        # Create stable hash
        content = json.dumps(key_fields, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_cached_enhancement(self, tool: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """
        Get cached enhancement for a tool.
        
        Args:
            tool: Tool dictionary
            
        Returns:
            Tuple of (enhanced_description, enhancement_category) or None
        """
        tool_hash = self._get_tool_hash(tool)
        
        if tool_hash in self._cache:
            cached = self._cache[tool_hash]
            logger.debug(f"Cache hit for tool: {tool.get('name', 'unknown')}")
            return cached["enhanced_description"], cached["enhancement_category"]
        
        logger.debug(f"Cache miss for tool: {tool.get('name', 'unknown')}")
        return None
    
    def cache_enhancement(self, 
                         tool: Dict[str, Any], 
                         enhanced_description: str, 
                         enhancement_category: str,
                         processing_time: float = 0.0):
        """
        Cache an enhanced description.
        
        Args:
            tool: Tool dictionary
            enhanced_description: Enhanced description text
            enhancement_category: Category used for enhancement
            processing_time: Time taken to generate enhancement
        """
        tool_hash = self._get_tool_hash(tool)
        
        cache_entry = {
            "tool_name": tool.get("name", "unknown"),
            "tool_id": tool.get("id", ""),
            "enhanced_description": enhanced_description,
            "enhancement_category": enhancement_category,
            "original_description": tool.get("description", ""),
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "processing_time": processing_time,
            "tool_metadata": {
                "tool_type": tool.get("tool_type", ""),
                "mcp_server_name": tool.get("mcp_server_name", ""),
                "category": tool.get("category", "")
            }
        }
        
        self._cache[tool_hash] = cache_entry
        
        # Periodically save cache (every 10 entries)
        if len(self._cache) % 10 == 0:
            self._save_cache()
        
        logger.debug(f"Cached enhancement for: {tool.get('name', 'unknown')}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._cache:
            return {"total_entries": 0, "cache_empty": True}
        
        # Calculate stats
        categories = {}
        processing_times = []
        tool_types = {}
        
        for entry in self._cache.values():
            # Category stats
            cat = entry.get("enhancement_category", "unknown")
            categories[cat] = categories.get(cat, 0) + 1
            
            # Processing time stats
            if entry.get("processing_time", 0) > 0:
                processing_times.append(entry["processing_time"])
            
            # Tool type stats  
            tool_type = entry.get("tool_metadata", {}).get("tool_type", "unknown")
            tool_types[tool_type] = tool_types.get(tool_type, 0) + 1
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_entries": len(self._cache),
            "cache_created": self._metadata.get("created_at"),
            "last_updated": self._metadata.get("last_updated"),
            "categories": categories,
            "tool_types": tool_types,
            "average_processing_time": avg_processing_time,
            "total_processing_time": sum(processing_times),
            "cache_file_size": self.cache_file.stat().st_size if self.cache_file.exists() else 0
        }
    
    def export_cache_summary(self, output_file: str = None) -> str:
        """
        Export cache summary for inspection.
        
        Args:
            output_file: Optional file to write summary to
            
        Returns:
            Path to summary file
        """
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.cache_dir / f"cache_summary_{timestamp}.md"
        
        stats = self.get_cache_stats()
        
        summary = f"""# Enhancement Cache Summary

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Cache Statistics
- **Total Cached Enhancements**: {stats['total_entries']}
- **Cache Created**: {stats.get('cache_created', 'Unknown')}
- **Last Updated**: {stats.get('last_updated', 'Never')}
- **Average Processing Time**: {stats['average_processing_time']:.2f}s
- **Total Processing Time Saved**: {stats['total_processing_time']:.1f}s
- **Cache File Size**: {stats['cache_file_size']:,} bytes

## Enhancement Categories
"""
        
        for category, count in sorted(stats.get('categories', {}).items()):
            percentage = (count / stats['total_entries']) * 100 if stats['total_entries'] > 0 else 0
            summary += f"- **{category}**: {count} tools ({percentage:.1f}%)\n"
        
        summary += "\n## Tool Types\n"
        for tool_type, count in sorted(stats.get('tool_types', {}).items()):
            percentage = (count / stats['total_entries']) * 100 if stats['total_entries'] > 0 else 0
            summary += f"- **{tool_type}**: {count} tools ({percentage:.1f}%)\n"
        
        summary += f"""
## Sample Enhanced Descriptions

"""
        
        # Show a few sample entries
        sample_count = 0
        for entry in list(self._cache.values())[:3]:
            if sample_count >= 3:
                break
            summary += f"""### {entry['tool_name']}
- **Category**: {entry['enhancement_category']}
- **Processing Time**: {entry.get('processing_time', 0):.2f}s
- **Original**: {entry['original_description']}
- **Enhanced**: {entry['enhanced_description'][:200]}{'...' if len(entry['enhanced_description']) > 200 else ''}

---

"""
            sample_count += 1
        
        # Write summary
        with open(output_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Cache summary exported to: {output_file}")
        return str(output_file)
    
    def clear_cache(self):
        """Clear all cached enhancements"""
        self._cache.clear()
        self._save_cache()
        logger.info("Cache cleared")
    
    def invalidate_tool(self, tool: Dict[str, Any]):
        """Invalidate cache entry for a specific tool"""
        tool_hash = self._get_tool_hash(tool)
        if tool_hash in self._cache:
            del self._cache[tool_hash]
            self._save_cache()
            logger.info(f"Invalidated cache for tool: {tool.get('name', 'unknown')}")
    
    def finalize(self):
        """Save cache and cleanup - call when enhancement process completes"""
        self._save_cache()
        logger.info(f"Enhancement cache finalized with {len(self._cache)} entries")


# Convenience functions for easy integration
_global_cache = None

def get_cache() -> EnhancementCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = EnhancementCache()
    return _global_cache

def cache_tool_enhancement(tool: Dict[str, Any], 
                          enhanced_description: str, 
                          enhancement_category: str,
                          processing_time: float = 0.0):
    """Cache a tool enhancement using global cache"""
    cache = get_cache()
    cache.cache_enhancement(tool, enhanced_description, enhancement_category, processing_time)

def get_cached_tool_enhancement(tool: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Get cached enhancement using global cache"""
    cache = get_cache()
    return cache.get_cached_enhancement(tool)


if __name__ == "__main__":
    # Test the cache system
    cache = EnhancementCache()
    
    # Example tool
    test_tool = {
        "id": "test-tool-1",
        "name": "test_tool",
        "description": "Test tool description",
        "tool_type": "external_mcp",
        "mcp_server_name": "test_server"
    }
    
    # Cache an enhancement
    cache.cache_enhancement(
        test_tool,
        "This is an enhanced test tool description with semantic keywords and use cases.",
        "mcp_tool",
        12.5
    )
    
    # Retrieve cached enhancement
    cached = cache.get_cached_enhancement(test_tool)
    if cached:
        print(f"Retrieved cached enhancement: {cached[0][:50]}...")
    
    # Show stats
    stats = cache.get_cache_stats()
    print(f"Cache stats: {stats}")
    
    # Export summary
    summary_file = cache.export_cache_summary()
    print(f"Summary exported to: {summary_file}")