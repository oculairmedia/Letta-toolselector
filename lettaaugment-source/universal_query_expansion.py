#!/usr/bin/env python3
"""
Universal Query Expansion for Intelligent Tool Discovery

This module provides a UNIVERSAL approach to query expansion that doesn't rely
on hardcoded tool mappings. Instead, it:

1. Dynamically analyzes tool schemas to detect multifunctional tools
2. Extracts operation keywords from tool names and descriptions
3. Builds a semantic understanding of what operations tools support
4. Expands queries based on detected intent and available tools

The key insight: Instead of hardcoding "bookstack_content_crud supports create/read/update/delete",
we ANALYZE the tool's JSON schema to discover it has an "operation" parameter with those values.
"""

import os
import re
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# OPERATION VOCABULARY - The universal language of tool operations
# =============================================================================

class OperationIntent(Enum):
    """Universal operation intents that map across all tools"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SEARCH = "search"
    MANAGE = "manage"
    ATTACH = "attach"
    DETACH = "detach"
    SEND = "send"
    CONFIGURE = "configure"
    UNKNOWN = "unknown"


# Maps natural language to operation intents
INTENT_SYNONYMS: Dict[OperationIntent, Set[str]] = {
    OperationIntent.CREATE: {
        "create", "make", "new", "add", "generate", "build", "compose", "write",
        "draft", "author", "publish", "post", "insert", "establish", "produce",
        "setup", "init", "initialize", "start", "begin"
    },
    OperationIntent.READ: {
        "read", "get", "fetch", "retrieve", "view", "show", "display", "access",
        "open", "load", "obtain", "see", "check", "inspect", "examine", "review",
        "lookup", "find"
    },
    OperationIntent.UPDATE: {
        "update", "edit", "modify", "change", "revise", "alter", "adjust", "patch",
        "amend", "correct", "fix", "improve", "enhance", "refine", "rewrite"
    },
    OperationIntent.DELETE: {
        "delete", "remove", "erase", "destroy", "drop", "clear", "wipe", "purge",
        "eliminate", "discard", "trash", "archive"
    },
    OperationIntent.LIST: {
        "list", "enumerate", "browse", "catalog", "inventory", "index", "all"
    },
    OperationIntent.SEARCH: {
        "search", "find", "query", "locate", "discover", "seek", "explore", "scan",
        "filter", "match"
    },
    OperationIntent.MANAGE: {
        "manage", "administer", "control", "handle", "organize", "maintain",
        "oversee", "coordinate", "operate"
    },
    OperationIntent.ATTACH: {
        "attach", "connect", "link", "bind", "associate", "join"
    },
    OperationIntent.DETACH: {
        "detach", "disconnect", "unlink", "unbind", "disassociate", "remove"
    },
    OperationIntent.SEND: {
        "send", "transmit", "deliver", "dispatch", "forward", "message", "notify"
    },
    OperationIntent.CONFIGURE: {
        "configure", "setup", "settings", "preferences", "options", "customize"
    },
}

# Build reverse mapping
_SYNONYM_TO_INTENT: Dict[str, OperationIntent] = {}
for intent, synonyms in INTENT_SYNONYMS.items():
    for synonym in synonyms:
        _SYNONYM_TO_INTENT[synonym.lower()] = intent


# =============================================================================
# TOOL SCHEMA ANALYZER - Extract capabilities from JSON schemas
# =============================================================================

@dataclass
class ToolCapabilities:
    """Discovered capabilities of a tool from its schema and metadata"""
    tool_name: str
    tool_id: str = ""
    
    # Detected from schema
    has_operation_param: bool = False
    operation_values: List[str] = field(default_factory=list)
    
    # Detected from name
    name_indicates_multifunctional: bool = False
    name_pattern_matched: str = ""
    
    # Detected from description
    description_operation_count: int = 0
    description_mentions_operations: List[str] = field(default_factory=list)
    
    # Computed scores
    multifunctional_confidence: float = 0.0
    
    # For query expansion
    expansion_keywords: Set[str] = field(default_factory=set)


class ToolSchemaAnalyzer:
    """
    Analyzes tool JSON schemas to understand their capabilities.
    
    This is the KEY to universal query expansion - we don't hardcode what
    tools can do, we DISCOVER it from their schemas.
    """
    
    # Patterns that indicate multifunctional tools in names
    MULTIFUNCTIONAL_NAME_PATTERNS = [
        (r'_crud$', 'crud'),
        (r'_manager$', 'manager'),
        (r'_unified$', 'unified'),
        (r'_hub$', 'hub'),
        (r'_advanced$', 'advanced'),
        (r'_operations$', 'operations'),
        (r'_ops$', 'ops'),
        (r'batch_', 'batch'),
        (r'bulk_', 'bulk'),
        (r'_batch$', 'batch'),
        (r'_bulk$', 'bulk'),
    ]
    
    # Patterns that indicate operation type from tool name (e.g., create_book, delete_page)
    # Format: (prefix_pattern, OperationIntent)
    OPERATION_PREFIX_PATTERNS = [
        (r'^create[_-]', OperationIntent.CREATE),
        (r'^make[_-]', OperationIntent.CREATE),
        (r'^add[_-]', OperationIntent.CREATE),
        (r'^new[_-]', OperationIntent.CREATE),
        (r'^get[_-]', OperationIntent.READ),
        (r'^read[_-]', OperationIntent.READ),
        (r'^fetch[_-]', OperationIntent.READ),
        (r'^retrieve[_-]', OperationIntent.READ),
        (r'^update[_-]', OperationIntent.UPDATE),
        (r'^edit[_-]', OperationIntent.UPDATE),
        (r'^modify[_-]', OperationIntent.UPDATE),
        (r'^delete[_-]', OperationIntent.DELETE),
        (r'^remove[_-]', OperationIntent.DELETE),
        (r'^list[_-]', OperationIntent.LIST),
        (r'^search[_-]', OperationIntent.SEARCH),
        (r'^find[_-]', OperationIntent.SEARCH),
        (r'^attach[_-]', OperationIntent.ATTACH),
        (r'^detach[_-]', OperationIntent.DETACH),
        (r'^send[_-]', OperationIntent.SEND),
    ]
    
    # Operation-related parameter names in schemas
    OPERATION_PARAM_NAMES = {'operation', 'action', 'method', 'command', 'op', 'type'}
    
    def analyze_tool(self, tool: Dict[str, Any]) -> ToolCapabilities:
        """
        Analyze a tool's schema and metadata to understand its capabilities.
        
        Args:
            tool: Tool dictionary with name, description, json_schema, etc.
            
        Returns:
            ToolCapabilities with discovered information
        """
        name = tool.get('name', '')
        description = tool.get('description', '')
        json_schema = tool.get('json_schema', {})
        if isinstance(json_schema, str):
            try:
                json_schema = json.loads(json_schema)
            except:
                json_schema = {}
        
        caps = ToolCapabilities(
            tool_name=name,
            tool_id=tool.get('id', tool.get('tool_id', ''))
        )
        
        # 1. Analyze JSON schema for operation parameter
        self._analyze_schema(json_schema, caps)
        
        # 2. Analyze name for multifunctional patterns
        self._analyze_name(name, caps)
        
        # 3. Analyze description for operation mentions
        self._analyze_description(description, caps)
        
        # 4. Calculate overall confidence
        self._calculate_confidence(caps)
        
        # 5. Generate expansion keywords
        self._generate_expansion_keywords(caps, description)
        
        return caps
    
    def _analyze_schema(self, schema: Dict[str, Any], caps: ToolCapabilities):
        """Extract operation capabilities from JSON schema"""
        properties = schema.get('properties', {})
        
        # Look for operation-like parameters
        for param_name in self.OPERATION_PARAM_NAMES:
            if param_name in properties:
                param_def = properties[param_name]
                
                # Check for enum values
                if 'enum' in param_def:
                    caps.has_operation_param = True
                    caps.operation_values = [str(v).lower() for v in param_def['enum']]
                    return
                
                # Check for $ref to a definition with enum
                if '$ref' in param_def:
                    ref_name = param_def['$ref'].split('/')[-1]
                    defs = schema.get('$defs', schema.get('definitions', {}))
                    if ref_name in defs and 'enum' in defs[ref_name]:
                        caps.has_operation_param = True
                        caps.operation_values = [str(v).lower() for v in defs[ref_name]['enum']]
                        return
    
    def _analyze_name(self, name: str, caps: ToolCapabilities):
        """Detect multifunctional patterns and operation type from tool name"""
        name_lower = name.lower()
        
        # Check for multifunctional patterns (crud, manager, etc.)
        for pattern, pattern_name in self.MULTIFUNCTIONAL_NAME_PATTERNS:
            if re.search(pattern, name_lower):
                caps.name_indicates_multifunctional = True
                caps.name_pattern_matched = pattern_name
                break
        
        # Check for operation prefix patterns (create_, delete_, etc.)
        # This helps us understand what operation this tool performs
        for pattern, intent in self.OPERATION_PREFIX_PATTERNS:
            if re.search(pattern, name_lower):
                # Add to operation values so it can be indexed
                if intent.value not in caps.operation_values:
                    caps.operation_values.append(intent.value)
                break
        
        # Extract entity from name (e.g., "book" from "create_book")
        # This helps with domain-based expansion
        name_parts = name_lower.replace('-', '_').split('_')
        if len(name_parts) >= 2:
            # The entity is usually the second part (after the action)
            potential_entity = name_parts[1] if len(name_parts) > 1 else ""
            if potential_entity and len(potential_entity) > 2:
                caps.expansion_keywords.add(potential_entity)
    
    def _analyze_description(self, description: str, caps: ToolCapabilities):
        """Count operation mentions in description and extract entity types"""
        desc_lower = description.lower()
        
        mentioned_ops = set()
        for intent, synonyms in INTENT_SYNONYMS.items():
            for synonym in synonyms:
                # Look for word boundaries
                if re.search(rf'\b{re.escape(synonym)}\b', desc_lower):
                    mentioned_ops.add(intent.value)
                    # Also add to operation_values for indexing
                    if intent.value not in caps.operation_values:
                        caps.operation_values.append(intent.value)
                    break
        
        caps.description_operation_count = len(mentioned_ops)
        caps.description_mentions_operations = list(mentioned_ops)
        
        # Check for CRUD indicator patterns in description
        crud_patterns = [
            r'\bcrud\b',
            r'create.*read.*update.*delete',
            r'operations.*(create|read|update|delete)',
            r'(create|read|update|delete).*operations',
        ]
        for pattern in crud_patterns:
            if re.search(pattern, desc_lower):
                caps.name_indicates_multifunctional = True
                caps.name_pattern_matched = 'crud'
                # Add all CRUD operations
                for op in ['create', 'read', 'update', 'delete']:
                    if op not in caps.operation_values:
                        caps.operation_values.append(op)
                break
        
        # Extract entity types mentioned with operations (e.g., "create_book" -> "book")
        # Look for patterns like 'create_book', 'delete_page', etc. in description
        entity_patterns = re.findall(r"'(create|read|update|delete|get|list)_(\w+)'", desc_lower)
        for _, entity in entity_patterns:
            if entity and len(entity) > 2:
                caps.expansion_keywords.add(entity)
    
    def _calculate_confidence(self, caps: ToolCapabilities):
        """Calculate multifunctional confidence score"""
        score = 0.0
        
        # Schema-based detection is most reliable
        if caps.has_operation_param:
            # More operations = higher confidence
            op_count = len(caps.operation_values)
            if op_count >= 4:
                score += 0.5
            elif op_count >= 2:
                score += 0.3
        
        # Name patterns are good indicators
        if caps.name_indicates_multifunctional:
            score += 0.3
        
        # Description mentions help
        if caps.description_operation_count >= 3:
            score += 0.2
        elif caps.description_operation_count >= 2:
            score += 0.1
        
        caps.multifunctional_confidence = min(score, 1.0)
    
    def _generate_expansion_keywords(self, caps: ToolCapabilities, description: str):
        """Generate keywords for query expansion based on capabilities"""
        # Start with any keywords already extracted (e.g., from _analyze_description)
        keywords = set(caps.expansion_keywords)
        
        # Add operation values from schema
        if caps.operation_values:
            keywords.update(caps.operation_values)
            
            # Also add synonyms for detected operations
            for op_value in caps.operation_values:
                intent = _SYNONYM_TO_INTENT.get(op_value)
                if intent and intent in INTENT_SYNONYMS:
                    # Add a few key synonyms
                    keywords.update(list(INTENT_SYNONYMS[intent])[:5])
        
        # Add pattern-based keywords
        if caps.name_pattern_matched:
            keywords.add(caps.name_pattern_matched)
            if caps.name_pattern_matched == 'crud':
                keywords.update(['create', 'read', 'update', 'delete', 'manage'])
            elif caps.name_pattern_matched == 'manager':
                keywords.update(['manage', 'list', 'get', 'configure'])
            elif caps.name_pattern_matched in ('batch', 'bulk'):
                keywords.update(['batch', 'bulk', 'multiple', 'mass'])
        
        # Extract domain words from name (e.g., "bookstack" from "bookstack_content_crud")
        name_parts = caps.tool_name.lower().replace('-', '_').split('_')
        # First part is often the service/domain
        if name_parts:
            keywords.add(name_parts[0])
        
        caps.expansion_keywords = keywords


# =============================================================================
# UNIVERSAL QUERY EXPANDER
# =============================================================================

@dataclass
class ExpandedQuery:
    """Result of universal query expansion"""
    original_query: str
    expanded_query: str
    detected_intents: List[OperationIntent]
    matched_tool_capabilities: List[str]  # Tool names that match
    added_keywords: List[str]
    confidence: float


class UniversalQueryExpander:
    """
    Universal query expander that dynamically understands tool capabilities.
    
    Unlike the hardcoded approach, this:
    1. Loads tool schemas from cache/Weaviate
    2. Analyzes them to understand capabilities
    3. Builds a "tool family" index (tools that operate on the same entity)
    4. Expands queries based on detected intent + available tools
    
    Key insight: Tools like create_book, delete_book, list_books form a "family"
    around the "book" entity. When searching for "create a book", we can add
    keywords that help find this family.
    """
    
    def __init__(self, tool_cache_path: Optional[str] = None):
        """
        Initialize with optional tool cache path.
        
        Args:
            tool_cache_path: Path to tool_cache.json, or None to use default
        """
        self.analyzer = ToolSchemaAnalyzer()
        self._tool_capabilities: Dict[str, ToolCapabilities] = {}
        self._intent_to_tools: Dict[OperationIntent, Set[str]] = {
            intent: set() for intent in OperationIntent
        }
        # NEW: Entity to tools mapping (e.g., "book" -> [create_book, delete_book, ...])
        self._entity_to_tools: Dict[str, Set[str]] = {}
        # NEW: MCP server to tools mapping
        self._server_to_tools: Dict[str, Set[str]] = {}
        # NEW: Entity to MCP server mapping
        self._entity_to_server: Dict[str, str] = {}
        
        # Try to load tool cache
        if tool_cache_path:
            self._load_tool_cache(tool_cache_path)
        else:
            # Try default locations
            default_paths = [
                '/app/runtime_cache/tool_cache.json',
                './runtime_cache/tool_cache.json',
                '../runtime_cache/tool_cache.json',
            ]
            for path in default_paths:
                if os.path.exists(path):
                    self._load_tool_cache(path)
                    break
    
    def _load_tool_cache(self, cache_path: str):
        """Load and analyze tools from cache"""
        try:
            with open(cache_path, 'r') as f:
                tools = json.load(f)
            
            logger.info(f"Loaded {len(tools)} tools from {cache_path}")
            self._analyze_tools(tools)
        except Exception as e:
            logger.warning(f"Failed to load tool cache from {cache_path}: {e}")
    
    def _analyze_tools(self, tools: List[Dict[str, Any]]):
        """Analyze all tools and build capability index"""
        for tool in tools:
            caps = self.analyzer.analyze_tool(tool)
            self._tool_capabilities[caps.tool_name] = caps
            
            # Get MCP server name
            mcp_server = tool.get('mcp_server_name', '')
            if mcp_server:
                if mcp_server not in self._server_to_tools:
                    self._server_to_tools[mcp_server] = set()
                self._server_to_tools[mcp_server].add(caps.tool_name)
            
            # Index by supported operations (from schema or name analysis)
            for op_value in caps.operation_values:
                intent = _SYNONYM_TO_INTENT.get(op_value)
                if intent:
                    self._intent_to_tools[intent].add(caps.tool_name)
            
            # Extract entity from tool name and build entity index
            entity = self._extract_entity_from_name(caps.tool_name)
            if entity:
                if entity not in self._entity_to_tools:
                    self._entity_to_tools[entity] = set()
                self._entity_to_tools[entity].add(caps.tool_name)
                
                # Track which MCP server handles this entity
                if mcp_server and entity not in self._entity_to_server:
                    self._entity_to_server[entity] = mcp_server
            
            # Also index entities from expansion_keywords for CRUD tools
            # This captures entities like "book", "page" from descriptions
            if caps.name_indicates_multifunctional and caps.expansion_keywords:
                # Common entity-like keywords (nouns, not verbs)
                entity_candidates = {
                    'book', 'page', 'chapter', 'shelf', 'document', 'article',
                    'issue', 'project', 'task', 'milestone', 'component',
                    'agent', 'memory', 'tool', 'source', 'block',
                    'message', 'room', 'post', 'image', 'file', 'folder'
                }
                for kw in caps.expansion_keywords:
                    if kw in entity_candidates:
                        if kw not in self._entity_to_tools:
                            self._entity_to_tools[kw] = set()
                        self._entity_to_tools[kw].add(caps.tool_name)
                        if mcp_server and kw not in self._entity_to_server:
                            self._entity_to_server[kw] = mcp_server
        
        # Log discovered patterns
        logger.info(f"Discovered {len(self._entity_to_tools)} entity types")
        logger.info(f"Entity examples: {list(self._entity_to_tools.keys())[:10]}")
        
        multifunc_tools = [
            name for name, caps in self._tool_capabilities.items()
            if caps.multifunctional_confidence > 0.3
        ]
        logger.info(f"Discovered {len(multifunc_tools)} multifunctional tools")
    
    def _extract_entity_from_name(self, tool_name: str) -> Optional[str]:
        """
        Extract the entity from a tool name like 'create_book' -> 'book'.
        
        Also handles MCP-style names like 'bookstack_content_crud' -> 'content'
        
        This is crucial for building tool families.
        """
        name_lower = tool_name.lower().replace('-', '_')
        
        # Common action prefixes to strip
        action_prefixes = [
            'create_', 'make_', 'add_', 'new_',
            'get_', 'read_', 'fetch_', 'retrieve_', 'show_',
            'update_', 'edit_', 'modify_', 'patch_',
            'delete_', 'remove_', 'destroy_',
            'list_', 'search_', 'find_', 'query_',
            'attach_', 'detach_',
            'send_', 'receive_',
        ]
        
        for prefix in action_prefixes:
            if name_lower.startswith(prefix):
                remainder = name_lower[len(prefix):]
                # Handle plurals (books -> book)
                if remainder.endswith('s') and len(remainder) > 3:
                    remainder = remainder[:-1]
                return remainder if remainder else None
        
        # Handle MCP-style names like 'bookstack_content_crud' or 'huly_list_issues'
        # Pattern: {mcp_server}_{entity}_{suffix} or {mcp_server}_{action}_{entity}
        parts = name_lower.split('_')
        if len(parts) >= 2:
            # Check if last part is a suffix like 'crud', 'manager', 'ops'
            suffixes = {'crud', 'manager', 'unified', 'hub', 'ops', 'operations', 'advanced'}
            if parts[-1] in suffixes and len(parts) >= 3:
                # Entity is the middle part(s)
                entity = '_'.join(parts[1:-1])
                if entity.endswith('s') and len(entity) > 3:
                    entity = entity[:-1]
                return entity if entity else None
        
        return None
    
    def add_tool(self, tool: Dict[str, Any]):
        """Add a single tool to the index (for dynamic updates)"""
        caps = self.analyzer.analyze_tool(tool)
        self._tool_capabilities[caps.tool_name] = caps
        
        if caps.has_operation_param:
            for op_value in caps.operation_values:
                intent = _SYNONYM_TO_INTENT.get(op_value)
                if intent:
                    self._intent_to_tools[intent].add(caps.tool_name)
    
    def detect_intents(self, query: str) -> List[OperationIntent]:
        """Detect operation intents from query"""
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        detected = set()
        for word in words:
            if word in _SYNONYM_TO_INTENT:
                detected.add(_SYNONYM_TO_INTENT[word])
        
        return list(detected) if detected else [OperationIntent.UNKNOWN]
    
    def find_matching_tools(self, intents: List[OperationIntent]) -> List[str]:
        """Find tools that support the detected intents"""
        matching = set()
        
        for intent in intents:
            if intent in self._intent_to_tools:
                matching.update(self._intent_to_tools[intent])
        
        return list(matching)
    
    def _detect_entities_in_query(self, query: str) -> List[str]:
        """
        Detect entity mentions in the query.
        
        Looks for known entities from our tool index.
        E.g., "create a book" -> ["book"]
        """
        query_lower = query.lower()
        words = set(re.findall(r'\b\w+\b', query_lower))
        
        detected = []
        for entity in self._entity_to_tools.keys():
            # Check for exact match
            if entity in words:
                detected.append(entity)
            # Check for plural
            elif entity + 's' in words or entity + 'es' in words:
                detected.append(entity)
        
        return detected
    
    def expand_query(self, query: str) -> ExpandedQuery:
        """
        Expand a query using dynamically discovered tool capabilities.
        
        This is UNIVERSAL because it doesn't use hardcoded mappings -
        it uses the analyzed tool schemas to understand what keywords
        will help find relevant tools.
        
        Key strategy:
        1. Detect operation intent (create, read, update, delete, etc.)
        2. Detect entity mentions (book, page, agent, etc.)
        3. Find related tools that operate on the same entity
        4. Add MCP server name and related keywords
        """
        # 1. Detect user intent (what operation they want)
        intents = self.detect_intents(query)
        
        # 2. Detect entities mentioned (what they want to operate on)
        entities = self._detect_entities_in_query(query)
        
        # 3. Find tools that match the intent
        matching_tools = self.find_matching_tools(intents)
        
        # 4. Gather expansion keywords - CONSERVATIVE approach
        # Only add keywords directly related to detected entities and their services
        added_keywords = set()
        
        # 4a. Check if query mentions a known service/platform name
        query_lower = query.lower()
        known_services = {'huly', 'bookstack', 'photoprism', 'komodo', 'letta', 'matrix', 
                         'payloadcms', 'cms', 'ghost', 'graphiti', 'opencode', 'vibekanban'}
        mentioned_service = None
        for service in known_services:
            if service in query_lower:
                mentioned_service = service
                added_keywords.add(service)
                break
        
        # 4b. Add entity names and their plural forms
        for entity in entities:
            added_keywords.add(entity)
            added_keywords.add(entity + 's')
            
            # Only add server name if no service was explicitly mentioned
            if not mentioned_service:
                server = self._entity_to_server.get(entity)
                if server:
                    added_keywords.add(server)
        
        # 4c. Add a few intent synonyms (max 3)
        for intent in intents:
            if intent in INTENT_SYNONYMS and intent != OperationIntent.UNKNOWN:
                added_keywords.update(list(INTENT_SYNONYMS[intent])[:3])
        
        # 4d. Add 'ops' keyword for unified tool discovery
        if mentioned_service:
            added_keywords.add(f'{mentioned_service}_ops')
            added_keywords.add('ops')
        
        # 5. Remove words already in query
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        added_keywords = added_keywords - query_words
        
        # 5b. Limit expansion to avoid diluting the query
        # Prioritize: entity names, server names, then other keywords
        MAX_EXPANSION_KEYWORDS = int(os.getenv("MAX_EXPANSION_KEYWORDS", "20"))
        if len(added_keywords) > MAX_EXPANSION_KEYWORDS:
            # Prioritize keeping entity and server names
            priority_keywords = set()
            for entity in entities:
                priority_keywords.add(entity)
                priority_keywords.add(entity + 's')
                server = self._entity_to_server.get(entity)
                if server:
                    priority_keywords.add(server)
            
            # Keep priority keywords, then fill with others up to limit
            other_keywords = added_keywords - priority_keywords
            added_keywords = priority_keywords | set(list(other_keywords)[:MAX_EXPANSION_KEYWORDS - len(priority_keywords)])
        
        # 6. Build expanded query
        expansion_str = " ".join(sorted(added_keywords))
        expanded_query = f"{query} {expansion_str}".strip()
        
        # 7. Calculate confidence
        confidence = 0.5
        if intents != [OperationIntent.UNKNOWN]:
            confidence += 0.2
        if entities:
            confidence += 0.2
        if matching_tools:
            confidence += 0.1
        
        return ExpandedQuery(
            original_query=query,
            expanded_query=expanded_query,
            detected_intents=intents,
            matched_tool_capabilities=matching_tools[:10],  # Limit for readability
            added_keywords=list(added_keywords),
            confidence=confidence
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_global_expander: Optional[UniversalQueryExpander] = None


def get_universal_expander(tool_cache_path: Optional[str] = None) -> UniversalQueryExpander:
    """Get or create the global universal query expander"""
    global _global_expander
    if _global_expander is None:
        _global_expander = UniversalQueryExpander(tool_cache_path)
    return _global_expander


def expand_query_universally(query: str, tool_cache_path: Optional[str] = None) -> str:
    """
    Convenience function for universal query expansion.
    
    Args:
        query: Original search query
        tool_cache_path: Optional path to tool cache
        
    Returns:
        Expanded query string
    """
    expander = get_universal_expander(tool_cache_path)
    result = expander.expand_query(query)
    return result.expanded_query


def expand_query_with_analysis(query: str, tool_cache_path: Optional[str] = None) -> ExpandedQuery:
    """
    Expand query and return full analysis.
    
    Args:
        query: Original search query
        tool_cache_path: Optional path to tool cache
        
    Returns:
        ExpandedQuery with full metadata
    """
    expander = get_universal_expander(tool_cache_path)
    return expander.expand_query(query)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 70)
    print("Universal Query Expansion - Dynamic Tool Discovery")
    print("=" * 70)
    
    # Create expander with tool cache
    cache_path = None
    for path in ['/app/runtime_cache/tool_cache.json', 
                 './runtime_cache/tool_cache.json',
                 '../runtime_cache/tool_cache.json',
                 '/opt/stacks/lettatoolsselector/lettaaugment-source/runtime_cache/tool_cache.json']:
        if os.path.exists(path):
            cache_path = path
            break
    
    if cache_path:
        print(f"\nUsing tool cache: {cache_path}")
    else:
        print("\nNo tool cache found - using empty index")
        print("Run with tool cache for better results!")
    
    expander = UniversalQueryExpander(cache_path)
    
    # Show discovered multifunctional tools
    print(f"\nDiscovered {len(expander._tool_capabilities)} tools")
    multifunc = [
        (name, caps.multifunctional_confidence, caps.operation_values[:5])
        for name, caps in expander._tool_capabilities.items()
        if caps.multifunctional_confidence > 0.3
    ]
    multifunc.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop multifunctional tools ({len(multifunc)} total):")
    for name, conf, ops in multifunc[:10]:
        print(f"  {name}: {conf:.0%} confidence, ops: {ops}")
    
    # Test queries
    test_queries = [
        "create a book",
        "delete page from documentation",
        "update chapter content",
        "list all agents",
        "send message to matrix",
        "manage bookstack content",
        "bulk update issues",
        "search for tools",
    ]
    
    print("\n" + "=" * 70)
    print("Query Expansion Tests")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n{'─' * 70}")
        print(f"Original: {query}")
        result = expander.expand_query(query)
        print(f"Intents: {[i.value for i in result.detected_intents]}")
        print(f"Matched tools: {result.matched_tool_capabilities[:5]}")
        print(f"Added keywords: {result.added_keywords[:10]}...")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Expanded: {result.expanded_query[:120]}...")
