"""
Query Intent Classifier

Advanced intent classification for tool search queries. Goes beyond simple
keyword matching to understand:

1. Primary intent (what the user wants to DO)
2. Target entities (what they want to do it TO)
3. Compound intents (multiple actions in one query)
4. Workflow patterns (sequences of actions)
5. Constraints and filters (what they DON'T want)

This module produces structured intents that can be used to:
- Boost relevant tools in search
- Return multiple tools for compound queries
- Suggest workflow sequences
"""

import re
import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# INTENT TYPES
# =============================================================================

class ActionType(Enum):
    """Primary action categories"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    SEARCH = "search"
    SEND = "send"
    MANAGE = "manage"
    CONFIGURE = "configure"
    ANALYZE = "analyze"
    TRANSFORM = "transform"
    CONNECT = "connect"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """How complex is the query?"""
    SIMPLE = "simple"           # Single action, single entity
    COMPOUND = "compound"       # Multiple actions or entities
    WORKFLOW = "workflow"       # Sequence of dependent actions
    AMBIGUOUS = "ambiguous"     # Unclear intent, needs clarification


@dataclass
class EntityReference:
    """A reference to an entity in the query"""
    entity_type: str          # e.g., "issue", "book", "message"
    entity_value: Optional[str] = None  # e.g., specific issue ID if mentioned
    qualifiers: List[str] = field(default_factory=list)  # e.g., ["open", "high-priority"]
    
    def __hash__(self):
        return hash((self.entity_type, self.entity_value, tuple(self.qualifiers)))


@dataclass
class ActionIntent:
    """A single action intent extracted from query"""
    action: ActionType
    target_entity: Optional[EntityReference] = None
    confidence: float = 1.0
    source_phrase: str = ""  # The part of query that triggered this
    
    # For compound actions
    depends_on: Optional['ActionIntent'] = None  # This action depends on another
    
    def __repr__(self):
        entity_str = f" on {self.target_entity.entity_type}" if self.target_entity else ""
        return f"ActionIntent({self.action.value}{entity_str}, conf={self.confidence:.2f})"


@dataclass 
class QueryIntent:
    """Complete parsed intent from a query"""
    original_query: str
    complexity: QueryComplexity
    
    # Primary intent (highest confidence)
    primary_action: Optional[ActionIntent] = None
    
    # All detected intents (for compound queries)
    all_actions: List[ActionIntent] = field(default_factory=list)
    
    # Entities mentioned
    entities: List[EntityReference] = field(default_factory=list)
    
    # Constraints/filters detected
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # For workflow patterns
    suggested_sequence: List[ActionIntent] = field(default_factory=list)
    
    # Search hints for the retrieval system
    boost_keywords: List[str] = field(default_factory=list)
    filter_keywords: List[str] = field(default_factory=list)
    
    # Domain hints (e.g., "project management", "documentation")
    detected_domains: List[str] = field(default_factory=list)


# =============================================================================
# ACTION VOCABULARY
# =============================================================================

ACTION_PATTERNS: Dict[ActionType, Dict[str, Any]] = {
    ActionType.CREATE: {
        "verbs": {
            "create", "make", "new", "add", "generate", "build", "compose",
            "write", "draft", "author", "publish", "post", "insert", "start",
            "begin", "setup", "initialize", "open", "spawn", "launch"
        },
        "patterns": [
            r"\b(create|make|add|new)\s+(?:a\s+)?(\w+)",
            r"\b(write|draft|compose)\s+(?:a\s+)?(\w+)",
            r"\b(start|begin|open)\s+(?:a\s+)?(?:new\s+)?(\w+)",
        ]
    },
    ActionType.READ: {
        "verbs": {
            "read", "get", "fetch", "retrieve", "view", "show", "display",
            "access", "open", "load", "obtain", "see", "check", "inspect",
            "examine", "review", "lookup", "look up"
        },
        "patterns": [
            r"\b(get|fetch|retrieve|show|view)\s+(?:the\s+)?(\w+)",
            r"\b(read|check|inspect)\s+(?:the\s+)?(\w+)",
            r"\bwhat\s+is\s+(?:the\s+)?(\w+)",
        ]
    },
    ActionType.UPDATE: {
        "verbs": {
            "update", "edit", "modify", "change", "revise", "alter", "adjust",
            "patch", "amend", "correct", "fix", "improve", "enhance", "refine",
            "rewrite", "rename", "move", "set"
        },
        "patterns": [
            r"\b(update|edit|modify|change)\s+(?:the\s+)?(\w+)",
            r"\b(fix|correct|improve)\s+(?:the\s+)?(\w+)",
            r"\b(rename|move)\s+(?:the\s+)?(\w+)",
            r"\bset\s+(?:the\s+)?(\w+)\s+to\b",
        ]
    },
    ActionType.DELETE: {
        "verbs": {
            "delete", "remove", "erase", "destroy", "drop", "clear", "wipe",
            "purge", "eliminate", "discard", "trash", "close", "archive",
            "cancel", "terminate", "kill"
        },
        "patterns": [
            r"\b(delete|remove|destroy)\s+(?:the\s+)?(\w+)",
            r"\b(close|archive|cancel)\s+(?:the\s+)?(\w+)",
            r"\b(clear|wipe|purge)\s+(?:the\s+)?(\w+)",
        ]
    },
    ActionType.LIST: {
        "verbs": {
            "list", "enumerate", "show all", "display all", "get all",
            "fetch all", "browse", "catalog", "all", "every"
        },
        "patterns": [
            r"\b(list|show|get|fetch)\s+(?:all\s+)?(\w+)s?\b",
            r"\bwhat\s+(\w+)s?\s+(?:are|do)\s+",
            r"\ball\s+(?:the\s+)?(\w+)s?\b",
        ]
    },
    ActionType.SEARCH: {
        "verbs": {
            "search", "find", "query", "lookup", "locate", "discover",
            "seek", "explore", "scan", "filter", "match", "where"
        },
        "patterns": [
            r"\b(search|find|locate)\s+(?:for\s+)?(?:a\s+)?(\w+)",
            r"\bwhere\s+(?:is|are)\s+(?:the\s+)?(\w+)",
            r"\b(\w+)\s+(?:with|that|where)\s+",
        ]
    },
    ActionType.SEND: {
        "verbs": {
            "send", "transmit", "deliver", "dispatch", "forward", "message",
            "notify", "email", "post", "broadcast", "publish", "share"
        },
        "patterns": [
            r"\b(send|post|share)\s+(?:a\s+)?(\w+)",
            r"\b(message|notify|email)\s+(?:the\s+)?(\w+)",
        ]
    },
    ActionType.MANAGE: {
        "verbs": {
            "manage", "administer", "control", "handle", "organize",
            "maintain", "oversee", "coordinate", "operate", "run"
        },
        "patterns": [
            r"\b(manage|handle|organize)\s+(?:the\s+)?(\w+)",
        ]
    },
    ActionType.CONFIGURE: {
        "verbs": {
            "configure", "setup", "settings", "preferences", "options",
            "customize", "enable", "disable", "toggle", "turn on", "turn off"
        },
        "patterns": [
            r"\b(configure|setup|customize)\s+(?:the\s+)?(\w+)",
            r"\b(enable|disable|toggle)\s+(?:the\s+)?(\w+)",
        ]
    },
    ActionType.ANALYZE: {
        "verbs": {
            "analyze", "examine", "evaluate", "assess", "measure",
            "calculate", "compute", "count", "summarize", "report"
        },
        "patterns": [
            r"\b(analyze|examine|evaluate)\s+(?:the\s+)?(\w+)",
            r"\bhow\s+many\s+(\w+)",
        ]
    },
    ActionType.CONNECT: {
        "verbs": {
            "connect", "link", "attach", "bind", "associate", "join",
            "integrate", "sync", "synchronize"
        },
        "patterns": [
            r"\b(connect|link|attach)\s+(?:the\s+)?(\w+)",
            r"\b(sync|integrate)\s+(?:the\s+)?(\w+)",
        ]
    },
}


# =============================================================================
# ENTITY VOCABULARY
# =============================================================================

ENTITY_TYPES: Dict[str, Set[str]] = {
    # Project management
    "issue": {"issue", "ticket", "bug", "task", "item", "problem", "defect"},
    "project": {"project", "workspace", "initiative", "plan", "board"},
    "milestone": {"milestone", "goal", "target", "deadline", "sprint", "release"},
    
    # Documentation
    "document": {"document", "doc", "file", "paper", "record", "report"},
    "page": {"page", "section", "entry", "note"},
    "book": {"book", "publication", "manual", "guide", "documentation"},
    "article": {"article", "post", "story", "blog"},
    
    # Communication
    "message": {"message", "chat", "dm", "notification", "alert"},
    "email": {"email", "mail", "letter"},
    "comment": {"comment", "reply", "response", "feedback"},
    
    # Agents/AI
    "agent": {"agent", "assistant", "bot", "ai", "model"},
    "memory": {"memory", "context", "history", "recall"},
    "tool": {"tool", "function", "capability", "skill"},
    
    # Code/Dev
    "repository": {"repository", "repo", "codebase", "project"},
    "branch": {"branch", "feature", "version"},
    "commit": {"commit", "change", "revision"},
    "pull_request": {"pull request", "pr", "merge request", "mr"},
    
    # Data
    "database": {"database", "db", "table", "collection"},
    "record": {"record", "entry", "row", "item"},
    "field": {"field", "column", "attribute", "property"},
    
    # Media
    "image": {"image", "picture", "photo", "graphic", "illustration"},
    "video": {"video", "clip", "recording"},
    "file": {"file", "attachment", "upload", "download"},
    
    # Calendar/Time
    "event": {"event", "meeting", "appointment", "calendar"},
    "reminder": {"reminder", "alarm", "notification"},
    "schedule": {"schedule", "timeline", "agenda"},
    
    # Users
    "user": {"user", "member", "person", "account", "profile"},
    "team": {"team", "group", "organization", "workspace"},
    "role": {"role", "permission", "access"},
}

# Build reverse lookup
_WORD_TO_ENTITY: Dict[str, str] = {}
for entity_type, words in ENTITY_TYPES.items():
    for word in words:
        _WORD_TO_ENTITY[word.lower()] = entity_type


# =============================================================================
# COMPOUND INTENT PATTERNS  
# =============================================================================

# Phrases that indicate multiple actions
COMPOUND_INDICATORS = [
    r"\band\s+(?:then\s+)?",       # "create and assign"
    r"\bthen\s+",                   # "create then assign"
    r"\bafter\s+(?:that\s+)?",     # "after that, send"
    r"\bbefore\s+",                 # "before sending"
    r"\balso\s+",                   # "also update"
    r",\s*(?:and\s+)?",            # "create, update, and delete"
]

# Workflow patterns (action sequences that commonly go together)
WORKFLOW_PATTERNS: Dict[str, List[Tuple[ActionType, str]]] = {
    "publish_content": [
        (ActionType.CREATE, "draft"),
        (ActionType.UPDATE, "content"),
        (ActionType.SEND, "publish"),
    ],
    "issue_lifecycle": [
        (ActionType.CREATE, "issue"),
        (ActionType.UPDATE, "assign"),
        (ActionType.UPDATE, "status"),
        (ActionType.DELETE, "close"),
    ],
    "document_review": [
        (ActionType.READ, "document"),
        (ActionType.UPDATE, "comment"),
        (ActionType.SEND, "approve"),
    ],
}


# =============================================================================
# CONSTRAINT PATTERNS
# =============================================================================

CONSTRAINT_PATTERNS = {
    "negation": [
        r"\b(?:not|don't|doesn't|without|except|exclude|excluding)\s+(\w+)",
        r"\b(\w+)\s+(?:is\s+)?(?:not|never)\b",
    ],
    "filter": [
        r"\b(?:only|just|specifically)\s+(\w+)",
        r"\bwhere\s+(\w+)\s+(?:is|equals?|=)\s+(\w+)",
        r"\bwith\s+(\w+)\s*[=:]\s*(\w+)",
    ],
    "temporal": [
        r"\b(today|yesterday|this week|last week|recent)\b",
        r"\b(before|after|since|until)\s+(\w+)",
    ],
    "quantity": [
        r"\b(first|last|top|bottom)\s+(\d+)\b",
        r"\b(\d+)\s+(\w+)s?\b",
        r"\ball\b",
    ],
}


# =============================================================================
# DOMAIN DETECTION
# =============================================================================

DOMAIN_KEYWORDS: Dict[str, Set[str]] = {
    "project_management": {
        "issue", "ticket", "task", "project", "sprint", "milestone",
        "kanban", "scrum", "agile", "backlog", "jira", "huly", "trello"
    },
    "documentation": {
        "document", "page", "book", "wiki", "knowledge", "article",
        "manual", "guide", "bookstack", "confluence", "notion"
    },
    "communication": {
        "message", "chat", "email", "notification", "slack", "matrix",
        "discord", "teams", "channel", "room"
    },
    "code_development": {
        "repository", "commit", "branch", "pull request", "merge",
        "github", "gitlab", "code", "deploy", "build"
    },
    "ai_agents": {
        "agent", "letta", "memory", "tool", "assistant", "bot",
        "archival", "conversation", "context"
    },
    "data_management": {
        "database", "record", "table", "query", "graphiti", "weaviate",
        "neo4j", "postgres", "schema"
    },
}


# =============================================================================
# QUERY INTENT CLASSIFIER
# =============================================================================

class QueryIntentClassifier:
    """
    Classifies queries into structured intents for better tool matching.
    
    Usage:
        classifier = QueryIntentClassifier()
        intent = classifier.classify("create an issue and assign it to John")
        
        # intent.primary_action -> ActionIntent(CREATE on issue)
        # intent.all_actions -> [CREATE issue, UPDATE assign]
        # intent.complexity -> COMPOUND
    """
    
    def __init__(self):
        self._action_verbs: Dict[str, ActionType] = {}
        self._build_verb_index()
    
    def _build_verb_index(self):
        """Build reverse index from verbs to actions"""
        for action_type, config in ACTION_PATTERNS.items():
            for verb in config["verbs"]:
                self._action_verbs[verb.lower()] = action_type
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify a query into a structured QueryIntent.
        
        Args:
            query: Natural language query
            
        Returns:
            QueryIntent with parsed actions, entities, and metadata
        """
        query_lower = query.lower().strip()
        
        # Extract all components
        actions = self._extract_actions(query_lower)
        entities = self._extract_entities(query_lower)
        constraints = self._extract_constraints(query_lower)
        domains = self._detect_domains(query_lower)
        
        # Determine complexity
        complexity = self._determine_complexity(query_lower, actions)
        
        # Match actions to entities
        matched_actions = self._match_actions_to_entities(actions, entities, query_lower)
        
        # Determine primary action
        primary = matched_actions[0] if matched_actions else None
        
        # Generate search hints
        boost_keywords = self._generate_boost_keywords(matched_actions, entities, domains)
        
        return QueryIntent(
            original_query=query,
            complexity=complexity,
            primary_action=primary,
            all_actions=matched_actions,
            entities=entities,
            constraints=constraints,
            detected_domains=domains,
            boost_keywords=boost_keywords,
        )
    
    def _extract_actions(self, query: str) -> List[Tuple[ActionType, str, int]]:
        """Extract action verbs from query with positions"""
        actions = []
        words = query.split()
        
        for i, word in enumerate(words):
            # Clean punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self._action_verbs:
                action_type = self._action_verbs[clean_word]
                actions.append((action_type, clean_word, i))
        
        # Also check multi-word patterns
        for action_type, config in ACTION_PATTERNS.items():
            for pattern in config.get("patterns", []):
                for match in re.finditer(pattern, query):
                    # Avoid duplicates
                    already_found = any(
                        a[1] in match.group(0) for a in actions
                    )
                    if not already_found:
                        actions.append((action_type, match.group(0), match.start()))
        
        # Sort by position
        actions.sort(key=lambda x: x[2])
        return actions
    
    def _extract_entities(self, query: str) -> List[EntityReference]:
        """Extract entity references from query"""
        entities = []
        words = query.split()
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            # Handle plurals
            singular = clean_word.rstrip('s') if clean_word.endswith('s') else clean_word
            
            entity_type = _WORD_TO_ENTITY.get(clean_word) or _WORD_TO_ENTITY.get(singular)
            if entity_type:
                # Check for qualifiers before this word
                qualifiers = self._extract_qualifiers_for_entity(query, word)
                entities.append(EntityReference(
                    entity_type=entity_type,
                    qualifiers=qualifiers
                ))
        
        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for e in entities:
            if e.entity_type not in seen:
                seen.add(e.entity_type)
                unique_entities.append(e)
        
        return unique_entities
    
    def _extract_qualifiers_for_entity(self, query: str, entity_word: str) -> List[str]:
        """Extract adjectives/qualifiers before an entity word"""
        qualifiers = []
        
        # Common qualifiers
        qualifier_words = {
            "open", "closed", "active", "inactive", "new", "old",
            "high", "low", "urgent", "critical", "important",
            "my", "all", "recent", "latest", "first", "last",
            "public", "private", "shared", "personal"
        }
        
        # Find position of entity word
        idx = query.lower().find(entity_word.lower())
        if idx > 0:
            # Get words before
            before = query[:idx].split()
            for word in before[-3:]:  # Last 3 words before entity
                clean = re.sub(r'[^\w]', '', word.lower())
                if clean in qualifier_words:
                    qualifiers.append(clean)
        
        return qualifiers
    
    def _extract_constraints(self, query: str) -> Dict[str, Any]:
        """Extract constraints/filters from query"""
        constraints = {}
        
        for constraint_type, patterns in CONSTRAINT_PATTERNS.items():
            matches = []
            for pattern in patterns:
                for match in re.finditer(pattern, query):
                    matches.append(match.groups())
            if matches:
                constraints[constraint_type] = matches
        
        return constraints
    
    def _detect_domains(self, query: str) -> List[str]:
        """Detect which domains the query relates to"""
        domains = []
        query_words = set(query.lower().split())
        
        for domain, keywords in DOMAIN_KEYWORDS.items():
            overlap = query_words & keywords
            if overlap:
                domains.append(domain)
        
        return domains
    
    def _determine_complexity(
        self,
        query: str,
        actions: List[Tuple[ActionType, str, int]]
    ) -> QueryComplexity:
        """Determine query complexity"""
        
        # Check for compound indicators
        has_compound = any(
            re.search(pattern, query) for pattern in COMPOUND_INDICATORS
        )
        
        if len(actions) == 0:
            return QueryComplexity.AMBIGUOUS
        elif len(actions) == 1 and not has_compound:
            return QueryComplexity.SIMPLE
        elif has_compound or len(actions) > 1:
            return QueryComplexity.COMPOUND
        else:
            return QueryComplexity.SIMPLE
    
    def _match_actions_to_entities(
        self,
        actions: List[Tuple[ActionType, str, int]],
        entities: List[EntityReference],
        query: str
    ) -> List[ActionIntent]:
        """Match actions to their target entities based on proximity"""
        matched = []
        
        for action_type, verb, pos in actions:
            # Find nearest entity after this action
            target_entity = None
            min_distance = float('inf')
            
            for entity in entities:
                # Find entity position in query
                entity_pos = query.find(entity.entity_type)
                if entity_pos < 0:
                    # Try plural
                    entity_pos = query.find(entity.entity_type + 's')
                
                if entity_pos >= 0:
                    distance = abs(entity_pos - pos)
                    if distance < min_distance:
                        min_distance = distance
                        target_entity = entity
            
            # Calculate confidence based on pattern match quality
            confidence = 0.9 if target_entity else 0.6
            
            matched.append(ActionIntent(
                action=action_type,
                target_entity=target_entity,
                confidence=confidence,
                source_phrase=verb
            ))
        
        # If no actions found but entities exist, infer READ/LIST intent
        if not matched and entities:
            # Likely a "show me X" or "what are the X" query
            matched.append(ActionIntent(
                action=ActionType.LIST,
                target_entity=entities[0],
                confidence=0.5,
                source_phrase="(inferred)"
            ))
        
        return matched
    
    def _generate_boost_keywords(
        self,
        actions: List[ActionIntent],
        entities: List[EntityReference],
        domains: List[str]
    ) -> List[str]:
        """Generate keywords to boost in search"""
        keywords = []
        
        # Add action synonyms
        for action in actions:
            action_config = ACTION_PATTERNS.get(action.action, {})
            keywords.extend(list(action_config.get("verbs", set()))[:5])
        
        # Add entity types and synonyms
        for entity in entities:
            keywords.append(entity.entity_type)
            entity_words = ENTITY_TYPES.get(entity.entity_type, set())
            keywords.extend(list(entity_words)[:3])
        
        # Add domain keywords
        for domain in domains:
            domain_words = DOMAIN_KEYWORDS.get(domain, set())
            keywords.extend(list(domain_words)[:3])
        
        return list(set(keywords))


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_classifier_instance: Optional[QueryIntentClassifier] = None


def get_classifier() -> QueryIntentClassifier:
    """Get singleton classifier instance"""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = QueryIntentClassifier()
    return _classifier_instance


def classify_query(query: str) -> QueryIntent:
    """Classify a query into structured intent"""
    return get_classifier().classify(query)


def get_search_boost_keywords(query: str) -> List[str]:
    """Get keywords to boost in search for this query"""
    intent = classify_query(query)
    return intent.boost_keywords


def is_compound_query(query: str) -> bool:
    """Check if query has multiple intents"""
    intent = classify_query(query)
    return intent.complexity in (QueryComplexity.COMPOUND, QueryComplexity.WORKFLOW)


def get_primary_action(query: str) -> Optional[ActionType]:
    """Get the primary action from a query"""
    intent = classify_query(query)
    return intent.primary_action.action if intent.primary_action else None
