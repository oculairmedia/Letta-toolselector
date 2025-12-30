#!/usr/bin/env python3
"""
Query Expansion Module for Multifunctional Tool Discovery

This module addresses the semantic search gap where queries like "create a book" 
fail to find multifunctional CRUD tools like `bookstack_content_crud`.

The solution:
1. Detect operation intent from queries (CRUD operations, management actions)
2. Expand queries with operation-specific keywords
3. Add tool-type awareness for unified/consolidated tools
4. Include common variations and synonyms
"""

import os
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class OperationType(Enum):
    """Standard CRUD and management operation types"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SEARCH = "search"
    MANAGE = "manage"
    CONFIGURE = "configure"
    UNKNOWN = "unknown"


@dataclass
class ExpandedQuery:
    """Result of query expansion"""
    original_query: str
    expanded_query: str
    detected_operations: List[OperationType]
    detected_domains: List[str]
    added_keywords: List[str]
    confidence: float


# Operation synonyms - maps user intent to operation types
OPERATION_SYNONYMS: Dict[OperationType, Set[str]] = {
    OperationType.CREATE: {
        "create", "make", "new", "add", "generate", "build", "compose",
        "write", "draft", "author", "publish", "post", "insert", "initialize",
        "start", "begin", "setup", "establish", "produce", "craft"
    },
    OperationType.READ: {
        "read", "get", "fetch", "retrieve", "view", "show", "display",
        "access", "open", "load", "obtain", "find", "lookup", "see",
        "check", "inspect", "examine", "review"
    },
    OperationType.UPDATE: {
        "update", "edit", "modify", "change", "revise", "alter", "adjust",
        "patch", "amend", "correct", "fix", "improve", "enhance", "refine",
        "rewrite", "rework", "transform", "convert"
    },
    OperationType.DELETE: {
        "delete", "remove", "erase", "destroy", "drop", "clear", "wipe",
        "purge", "eliminate", "discard", "trash", "archive"
    },
    OperationType.LIST: {
        "list", "enumerate", "show all", "display all", "get all", "fetch all",
        "browse", "catalog", "inventory", "index", "directory"
    },
    OperationType.SEARCH: {
        "search", "find", "query", "lookup", "locate", "discover", "seek",
        "explore", "scan", "filter", "match"
    },
    OperationType.MANAGE: {
        "manage", "administer", "control", "handle", "organize", "maintain",
        "oversee", "supervise", "coordinate", "operate"
    },
    OperationType.CONFIGURE: {
        "configure", "setup", "settings", "preferences", "options", "customize",
        "adjust", "tune", "calibrate"
    }
}

# Domain keywords - maps entity types to related terms
# NOTE: Order matters! Last domain with a keyword "wins" in reverse lookup
# We define more specific domains first, then more general ones
DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    # Content management domains (defined first to avoid conflicts)
    "content": {"content", "text", "material", "information", "data", "media"},
    "document": {"document", "doc", "file", "paper", "record", "report"},
    "article": {"article", "post", "story", "piece", "write-up"},  # article is separate from page
    
    # BookStack domains (defined after content to override common terms)
    "book": {"book", "books", "publication", "documentation", "manual", "guide", "volume"},
    "page": {"page", "pages", "section", "entry"},  # Keep page distinct
    "chapter": {"chapter", "chapters", "part", "division"},
    "shelf": {"shelf", "shelves", "bookshelf", "collection", "library", "category"},
    
    # Project management domains
    "project": {"project", "projects", "workspace", "initiative", "plan"},
    "issue": {"issue", "issues", "ticket", "tickets", "bug", "bugs", "task", "tasks", "item", "problem"},
    "milestone": {"milestone", "milestones", "goal", "target", "deadline", "sprint"},
    "component": {"component", "components", "module", "element"},
    
    # Agent/memory domains
    "agent": {"agent", "agents", "assistant", "bot", "ai", "model"},
    "memory": {"memory", "memories", "context", "history", "recall", "remember"},
    "message": {"message", "messages", "chat", "communication", "conversation"},
    
    # Image domains
    "image": {"image", "images", "picture", "photo", "graphic", "illustration", "media"},
}

# Tool type patterns for identifying multifunctional tools
MULTIFUNCTIONAL_TOOL_PATTERNS: Dict[str, List[str]] = {
    "crud": ["crud", "content_crud", "manage", "operations", "unified"],
    "batch": ["batch", "bulk", "batch_operations", "mass"],
    "hub": ["hub", "manager", "advanced", "unified", "operations_hub"],
    "combined": ["combined", "all-in-one", "multi", "comprehensive"]
}

# Known multifunctional tools and their operation mappings
KNOWN_MULTIFUNCTIONAL_TOOLS: Dict[str, Dict[str, List[str]]] = {
    "bookstack_content_crud": {
        "operations": ["create", "read", "update", "delete"],
        "domains": ["book", "page", "chapter", "shelf"],
        "keywords": ["bookstack", "documentation", "wiki", "knowledge base", "content management"]
    },
    "bookstack_batch_operations": {
        "operations": ["create", "update", "delete"],
        "domains": ["book", "page", "chapter", "shelf"],
        "keywords": ["bulk", "mass", "batch", "multiple", "bookstack"]
    },
    "bookstack_manage_images": {
        "operations": ["create", "read", "update", "delete", "list"],
        "domains": ["image"],
        "keywords": ["gallery", "upload", "picture", "photo", "media", "bookstack"]
    },
    "letta_agent_advanced": {
        "operations": ["list", "create", "get", "update", "delete", "manage"],
        "domains": ["agent"],
        "keywords": ["letta", "ai agent", "assistant", "bot", "agent management"]
    },
    "letta_memory_unified": {
        "operations": ["get", "update", "list", "create", "search"],
        "domains": ["memory", "block"],
        "keywords": ["letta", "memory", "context", "archival", "core memory"]
    },
    "letta_tool_manager": {
        "operations": ["list", "get", "create", "update", "delete", "attach", "detach"],
        "domains": ["tool"],
        "keywords": ["letta", "tool", "function", "capability", "plugin"]
    },
    "letta_source_manager": {
        "operations": ["list", "get", "create", "update", "delete", "attach", "upload"],
        "domains": ["source", "file"],
        "keywords": ["letta", "source", "data", "file", "upload", "knowledge"]
    },
    "huly_bulk_update_issues": {
        "operations": ["update"],
        "domains": ["issue"],
        "keywords": ["huly", "bulk", "batch", "mass update", "project management"]
    },
    "matrix_messaging": {
        "operations": ["send", "read", "create", "list", "manage"],
        "domains": ["message", "room", "agent"],
        "keywords": ["matrix", "chat", "messaging", "communication", "letta"]
    }
}


class QueryExpander:
    """
    Expands search queries to improve discovery of multifunctional tools.
    
    Example:
        Input: "create a book in bookstack"
        Output: "create book bookstack CRUD content_crud make add new documentation 
                 wiki knowledge base publish write manage books"
    """
    
    def __init__(self, enable_aggressive_expansion: bool = True):
        """
        Initialize query expander.
        
        Args:
            enable_aggressive_expansion: If True, adds more keywords for better recall
        """
        self.aggressive = enable_aggressive_expansion
        self._build_reverse_mappings()
    
    def _build_reverse_mappings(self):
        """Build reverse lookup from synonym to operation type"""
        self.synonym_to_operation: Dict[str, OperationType] = {}
        for op_type, synonyms in OPERATION_SYNONYMS.items():
            for synonym in synonyms:
                self.synonym_to_operation[synonym.lower()] = op_type
        
        self.keyword_to_domain: Dict[str, str] = {}
        for domain, keywords in DOMAIN_KEYWORDS.items():
            for keyword in keywords:
                self.keyword_to_domain[keyword.lower()] = domain
    
    def detect_operations(self, query: str) -> List[OperationType]:
        """Detect operation types from query text"""
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        detected = set()
        for word in words:
            if word in self.synonym_to_operation:
                detected.add(self.synonym_to_operation[word])
        
        # Also check for multi-word patterns
        for op_type, synonyms in OPERATION_SYNONYMS.items():
            for synonym in synonyms:
                if ' ' in synonym and synonym in query_lower:
                    detected.add(op_type)
        
        return list(detected) if detected else [OperationType.UNKNOWN]
    
    def detect_domains(self, query: str) -> List[str]:
        """Detect entity domains from query text"""
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        detected = set()
        for word in words:
            if word in self.keyword_to_domain:
                detected.add(self.keyword_to_domain[word])
        
        return list(detected)
    
    def get_operation_keywords(self, operations: List[OperationType]) -> Set[str]:
        """Get expansion keywords for detected operations"""
        keywords = set()
        for op in operations:
            if op in OPERATION_SYNONYMS:
                # Add a subset of synonyms (most common ones)
                synonyms = list(OPERATION_SYNONYMS[op])
                keywords.update(synonyms[:5] if not self.aggressive else synonyms)
        return keywords
    
    def get_domain_keywords(self, domains: List[str]) -> Set[str]:
        """Get expansion keywords for detected domains"""
        keywords = set()
        for domain in domains:
            if domain in DOMAIN_KEYWORDS:
                domain_kws = list(DOMAIN_KEYWORDS[domain])
                keywords.update(domain_kws[:4] if not self.aggressive else domain_kws)
        return keywords
    
    def get_multifunctional_tool_keywords(
        self, 
        operations: List[OperationType],
        domains: List[str]
    ) -> Set[str]:
        """
        Get keywords specific to multifunctional tools that match the query intent.
        
        This is the KEY innovation - when we detect CRUD-like operations,
        we inject keywords that will match unified tool descriptions.
        """
        keywords = set()
        
        # Check if query suggests CRUD operations
        crud_ops = {OperationType.CREATE, OperationType.READ, OperationType.UPDATE, OperationType.DELETE}
        if any(op in crud_ops for op in operations):
            # Add CRUD-related keywords that appear in multifunctional tool names/descriptions
            keywords.update(["crud", "operations", "manage", "unified", "hub"])
        
        # Match against known multifunctional tools
        for tool_name, tool_info in KNOWN_MULTIFUNCTIONAL_TOOLS.items():
            tool_ops = set(tool_info.get("operations", []))
            tool_domains = set(tool_info.get("domains", []))
            
            # Check if operations match
            ops_match = any(
                op.value in tool_ops or op.value.lower() in tool_ops 
                for op in operations if op != OperationType.UNKNOWN
            )
            
            # Check if domains match
            domains_match = bool(set(domains) & tool_domains)
            
            # If both match, add tool-specific keywords
            if ops_match and domains_match:
                keywords.update(tool_info.get("keywords", []))
                # Also add the tool name components (helps with name matching)
                keywords.update(tool_name.split("_"))
        
        return keywords
    
    def expand_query(self, query: str) -> ExpandedQuery:
        """
        Main expansion method - takes a query and returns an expanded version.
        
        Args:
            query: Original search query
            
        Returns:
            ExpandedQuery with expanded text and metadata
        """
        # Detect intent
        operations = self.detect_operations(query)
        domains = self.detect_domains(query)
        
        # Gather expansion keywords
        added_keywords = set()
        
        # Add operation synonyms
        op_keywords = self.get_operation_keywords(operations)
        added_keywords.update(op_keywords)
        
        # Add domain synonyms
        domain_keywords = self.get_domain_keywords(domains)
        added_keywords.update(domain_keywords)
        
        # Add multifunctional tool keywords (KEY IMPROVEMENT)
        multifunc_keywords = self.get_multifunctional_tool_keywords(operations, domains)
        added_keywords.update(multifunc_keywords)
        
        # Remove words already in original query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        added_keywords = added_keywords - query_words
        
        # Build expanded query
        # Put original query first, then add expansion keywords
        expansion_str = " ".join(sorted(added_keywords))
        expanded_query = f"{query} {expansion_str}".strip()
        
        # Calculate confidence based on detection success
        confidence = 0.5
        if operations != [OperationType.UNKNOWN]:
            confidence += 0.25
        if domains:
            confidence += 0.25
        
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_query,
            detected_operations=operations,
            detected_domains=domains,
            added_keywords=list(added_keywords),
            confidence=confidence
        )


# Global instance for easy access
_query_expander: Optional[QueryExpander] = None


def get_query_expander() -> QueryExpander:
    """Get or create the global query expander instance"""
    global _query_expander
    if _query_expander is None:
        aggressive = os.getenv("QUERY_EXPANSION_AGGRESSIVE", "true").lower() == "true"
        _query_expander = QueryExpander(enable_aggressive_expansion=aggressive)
    return _query_expander


def expand_search_query(query: str) -> str:
    """
    Convenience function to expand a query string.
    
    Args:
        query: Original search query
        
    Returns:
        Expanded query string
    """
    expander = get_query_expander()
    result = expander.expand_query(query)
    return result.expanded_query


def expand_search_query_with_metadata(query: str) -> ExpandedQuery:
    """
    Expand query and return full metadata.
    
    Args:
        query: Original search query
        
    Returns:
        ExpandedQuery object with all metadata
    """
    expander = get_query_expander()
    return expander.expand_query(query)


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("Query Expansion Testing - Multifunctional Tool Discovery")
    print("=" * 70)
    
    test_queries = [
        "create a book in bookstack",
        "make a new page",
        "delete book from documentation",
        "update chapter content",
        "list all books",
        "search for articles",
        "manage bookstack content",
        "create an agent",
        "send message to matrix",
        "upload image to gallery",
        "bulk update issues",
        "get agent memory",
    ]
    
    expander = QueryExpander(enable_aggressive_expansion=True)
    
    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Original: {query}")
        result = expander.expand_query(query)
        print(f"Operations: {[op.value for op in result.detected_operations]}")
        print(f"Domains: {result.detected_domains}")
        print(f"Added: {result.added_keywords[:10]}...")  # Show first 10
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Expanded: {result.expanded_query[:150]}...")
