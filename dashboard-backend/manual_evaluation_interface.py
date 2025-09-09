"""
LDTS-39: Manual evaluation interface with Context7 standards

Comprehensive manual evaluation interface for assessing search and reranking quality
following Context7 evaluation standards for information retrieval systems.
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
import hashlib
from pathlib import Path

logger = logging.getLogger(__name__)

class RelevanceLevel(Enum):
    """Context7-standard relevance levels"""
    PERFECTLY_RELEVANT = 4    # Perfectly answers the query
    HIGHLY_RELEVANT = 3       # Mostly relevant with minor gaps
    SOMEWHAT_RELEVANT = 2     # Partially relevant but incomplete
    MARGINALLY_RELEVANT = 1   # Tangentially related
    NOT_RELEVANT = 0          # Not relevant at all

class EvaluationAspect(Enum):
    """Different aspects to evaluate"""
    RELEVANCE = "relevance"           # Core relevance to query
    COMPLETENESS = "completeness"     # How complete the answer is
    ACCURACY = "accuracy"             # Factual accuracy
    RECENCY = "recency"              # How up-to-date the information is
    CLARITY = "clarity"              # How clear and understandable
    USEFULNESS = "usefulness"        # Practical utility

class EvaluatorRole(Enum):
    """Evaluator role types"""
    EXPERT = "expert"                # Domain expert
    END_USER = "end_user"           # Typical end user
    DEVELOPER = "developer"         # Technical developer
    ADMIN = "admin"                 # System administrator

@dataclass
class EvaluationCriteria:
    """Evaluation criteria definition"""
    name: str
    description: str
    scale_min: int = 0
    scale_max: int = 4
    scale_labels: Dict[int, str] = field(default_factory=dict)
    weight: float = 1.0
    required: bool = True

@dataclass
class DocumentEvaluation:
    """Evaluation for a single document"""
    document_id: str
    document_title: str
    document_content: str
    document_source: str
    original_rank: int
    reranked_rank: Optional[int] = None
    
    # Evaluations by aspect
    relevance_score: Optional[int] = None
    completeness_score: Optional[int] = None
    accuracy_score: Optional[int] = None
    recency_score: Optional[int] = None
    clarity_score: Optional[int] = None
    usefulness_score: Optional[int] = None
    
    # Overall evaluation
    overall_score: Optional[float] = None
    binary_relevant: Optional[bool] = None  # For binary relevance metrics
    
    # Evaluator feedback
    comments: Optional[str] = None
    confidence_level: Optional[int] = None  # 1-5 scale
    time_spent_seconds: Optional[int] = None
    
    # Metadata
    evaluated_at: Optional[datetime] = None
    evaluation_id: Optional[str] = None

@dataclass
class QueryEvaluation:
    """Complete evaluation for a query"""
    evaluation_id: str
    query_text: str
    query_intent: Optional[str] = None
    query_category: Optional[str] = None
    
    # Search configuration
    search_config: Dict[str, Any] = field(default_factory=dict)
    reranker_used: Optional[str] = None
    
    # Document evaluations
    document_evaluations: List[DocumentEvaluation] = field(default_factory=list)
    
    # Query-level metrics
    query_difficulty: Optional[int] = None  # 1-5 scale
    query_clarity: Optional[int] = None     # How clear the query is
    expected_result_type: Optional[str] = None  # What type of results expected
    
    # Evaluator information
    evaluator_id: str = "anonymous"
    evaluator_role: EvaluatorRole = EvaluatorRole.END_USER
    evaluator_expertise: Optional[int] = None  # 1-5 scale for domain expertise
    
    # Evaluation metadata
    evaluation_started_at: Optional[datetime] = None
    evaluation_completed_at: Optional[datetime] = None
    total_evaluation_time_seconds: Optional[int] = None
    
    # Quality control
    attention_checks_passed: int = 0
    consistency_score: Optional[float] = None
    
    # Session info
    session_id: Optional[str] = None
    evaluation_version: str = "1.0"

@dataclass
class EvaluationSession:
    """Evaluation session containing multiple queries"""
    session_id: str
    session_name: str
    evaluator_id: str
    evaluator_role: EvaluatorRole
    
    # Session configuration
    evaluation_criteria: List[EvaluationCriteria] = field(default_factory=list)
    instructions: str = ""
    randomize_order: bool = True
    max_documents_per_query: int = 20
    
    # Session data
    queries: List[QueryEvaluation] = field(default_factory=list)
    
    # Session metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    paused_at: Optional[datetime] = None
    
    # Progress tracking
    current_query_index: int = 0
    completed_queries: int = 0
    total_queries: int = 0
    
    # Session settings
    show_original_ranking: bool = True
    show_reranked_comparison: bool = True
    enable_comments: bool = True
    require_confidence_ratings: bool = False
    
    # Quality control
    include_attention_checks: bool = True
    attention_check_frequency: int = 5  # Every N queries

class ManualEvaluationInterface:
    """Manual evaluation interface following Context7 standards"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else Path(__file__).parent / "evaluations"
        self.storage_path.mkdir(exist_ok=True)
        
        self.active_sessions: Dict[str, EvaluationSession] = {}
        self.evaluation_criteria = self._load_default_criteria()
        self._initialized = False
    
    def _load_default_criteria(self) -> List[EvaluationCriteria]:
        """Load default Context7-standard evaluation criteria"""
        return [
            EvaluationCriteria(
                name="relevance",
                description="How relevant is this result to the query?",
                scale_min=0,
                scale_max=4,
                scale_labels={
                    4: "Perfectly relevant - completely answers the query",
                    3: "Highly relevant - mostly answers with minor gaps", 
                    2: "Somewhat relevant - partially answers but incomplete",
                    1: "Marginally relevant - tangentially related",
                    0: "Not relevant - completely unrelated"
                },
                weight=2.0,  # Most important
                required=True
            ),
            EvaluationCriteria(
                name="completeness",
                description="How complete is the information provided?",
                scale_min=0,
                scale_max=4,
                scale_labels={
                    4: "Complete - provides all necessary information",
                    3: "Mostly complete - minor details missing",
                    2: "Somewhat complete - significant gaps",
                    1: "Incomplete - major information missing",
                    0: "Very incomplete - minimal useful information"
                },
                weight=1.5,
                required=True
            ),
            EvaluationCriteria(
                name="accuracy",
                description="How accurate is the information?",
                scale_min=0,
                scale_max=4,
                scale_labels={
                    4: "Completely accurate",
                    3: "Mostly accurate with minor errors",
                    2: "Somewhat accurate with some errors",
                    1: "Contains significant inaccuracies",
                    0: "Mostly or completely inaccurate"
                },
                weight=1.8,
                required=True
            ),
            EvaluationCriteria(
                name="usefulness",
                description="How useful would this be for the user?",
                scale_min=0,
                scale_max=4,
                scale_labels={
                    4: "Extremely useful - directly actionable",
                    3: "Very useful - provides clear value",
                    2: "Moderately useful - some value",
                    1: "Slightly useful - minimal value", 
                    0: "Not useful at all"
                },
                weight=1.3,
                required=True
            ),
            EvaluationCriteria(
                name="clarity",
                description="How clear and understandable is the information?",
                scale_min=0,
                scale_max=4,
                scale_labels={
                    4: "Very clear and easy to understand",
                    3: "Mostly clear with minor ambiguities",
                    2: "Somewhat clear but requires interpretation",
                    1: "Unclear in several places",
                    0: "Very unclear or confusing"
                },
                weight=1.0,
                required=False
            )
        ]
    
    async def initialize(self) -> bool:
        """Initialize the evaluation interface"""
        try:
            logger.info("Initializing manual evaluation interface...")
            
            # Create storage directories
            (self.storage_path / "sessions").mkdir(exist_ok=True)
            (self.storage_path / "evaluations").mkdir(exist_ok=True)
            (self.storage_path / "exports").mkdir(exist_ok=True)
            
            # Load existing sessions
            await self._load_active_sessions()
            
            self._initialized = True
            logger.info(f"Manual evaluation interface initialized with {len(self.active_sessions)} active sessions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize evaluation interface: {e}")
            return False
    
    async def _load_active_sessions(self):
        """Load active evaluation sessions from storage"""
        try:
            sessions_dir = self.storage_path / "sessions"
            
            for session_file in sessions_dir.glob("*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    session = self._deserialize_session(session_data)
                    if session and not session.completed_at:  # Only load incomplete sessions
                        self.active_sessions[session.session_id] = session
                        
                except Exception as e:
                    logger.error(f"Failed to load session {session_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to load active sessions: {e}")
    
    async def create_evaluation_session(self,
                                       session_name: str,
                                       evaluator_id: str,
                                       evaluator_role: EvaluatorRole,
                                       queries: List[str],
                                       custom_criteria: Optional[List[EvaluationCriteria]] = None,
                                       instructions: str = "",
                                       max_documents_per_query: int = 20) -> str:
        """Create a new evaluation session"""
        try:
            session_id = str(uuid.uuid4())
            
            # Prepare query evaluations
            query_evaluations = []
            for query_text in queries:
                query_eval = QueryEvaluation(
                    evaluation_id=str(uuid.uuid4()),
                    query_text=query_text,
                    evaluator_id=evaluator_id,
                    evaluator_role=evaluator_role,
                    session_id=session_id
                )
                query_evaluations.append(query_eval)
            
            # Create session
            session = EvaluationSession(
                session_id=session_id,
                session_name=session_name,
                evaluator_id=evaluator_id,
                evaluator_role=evaluator_role,
                evaluation_criteria=custom_criteria or self.evaluation_criteria,
                instructions=instructions,
                queries=query_evaluations,
                total_queries=len(queries),
                max_documents_per_query=max_documents_per_query
            )
            
            # Save session
            await self._save_session(session)
            
            # Add to active sessions
            self.active_sessions[session_id] = session
            
            logger.info(f"Created evaluation session: {session_name} ({session_id}) with {len(queries)} queries")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create evaluation session: {e}")
            raise
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = self.active_sessions.get(session_id)
        if not session:
            return None
        
        return {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "evaluator_id": session.evaluator_id,
            "evaluator_role": session.evaluator_role.value,
            "total_queries": session.total_queries,
            "completed_queries": session.completed_queries,
            "current_query_index": session.current_query_index,
            "progress_percentage": (session.completed_queries / session.total_queries * 100) if session.total_queries > 0 else 0,
            "created_at": session.created_at.isoformat(),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "evaluation_criteria": [
                {
                    "name": criteria.name,
                    "description": criteria.description,
                    "scale_min": criteria.scale_min,
                    "scale_max": criteria.scale_max,
                    "scale_labels": criteria.scale_labels,
                    "weight": criteria.weight,
                    "required": criteria.required
                }
                for criteria in session.evaluation_criteria
            ],
            "instructions": session.instructions
        }
    
    async def get_next_query_for_evaluation(self, session_id: str, 
                                           search_config: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get the next query to evaluate with search results"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            if session.current_query_index >= len(session.queries):
                return None  # Session complete
            
            current_query = session.queries[session.current_query_index]
            
            # Mark session as started if not already
            if not session.started_at:
                session.started_at = datetime.now(timezone.utc)
            
            # Get search results for this query
            search_results = await self._get_search_results_for_query(
                current_query.query_text,
                search_config or {},
                session.max_documents_per_query
            )
            
            # Prepare document evaluation templates
            document_templates = []
            for i, result in enumerate(search_results.get("results", [])):
                doc_template = {
                    "document_id": result.get("id", f"doc_{i}"),
                    "document_title": result.get("name", "Untitled"),
                    "document_content": result.get("description", ""),
                    "document_source": result.get("source", "unknown"),
                    "original_rank": i + 1,
                    "reranked_rank": result.get("rerank_position"),
                    "original_score": result.get("original_score", 0.0),
                    "reranked_score": result.get("reranked_score"),
                    "metadata": result.get("metadata", {})
                }
                document_templates.append(doc_template)
            
            # Include attention check if needed
            include_attention_check = (
                session.include_attention_checks and 
                session.current_query_index % session.attention_check_frequency == 0 and
                session.current_query_index > 0
            )
            
            return {
                "session_id": session_id,
                "query_evaluation_id": current_query.evaluation_id,
                "query_index": session.current_query_index,
                "total_queries": session.total_queries,
                "query_text": current_query.query_text,
                "query_intent": current_query.query_intent,
                "documents": document_templates,
                "evaluation_criteria": [
                    {
                        "name": criteria.name,
                        "description": criteria.description,
                        "scale_min": criteria.scale_min,
                        "scale_max": criteria.scale_max,
                        "scale_labels": criteria.scale_labels,
                        "weight": criteria.weight,
                        "required": criteria.required
                    }
                    for criteria in session.evaluation_criteria
                ],
                "search_config": search_results.get("search_config", {}),
                "reranker_used": search_results.get("reranker_used"),
                "include_attention_check": include_attention_check,
                "evaluation_started_at": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get next query for evaluation: {e}")
            raise
    
    async def _get_search_results_for_query(self, query: str, search_config: Dict[str, Any], 
                                          max_results: int) -> Dict[str, Any]:
        """Get search results for a query (integrates with search system)"""
        try:
            # Import search functionality
            from main import ml_resources
            
            weaviate_search = ml_resources.get("weaviate_search")
            if not weaviate_search:
                # Return mock results for development
                return self._generate_mock_search_results(query, max_results)
            
            # Use existing search system
            search_params = {
                "query": query,
                "limit": max_results,
                **search_config
            }
            
            # Perform search
            results = await weaviate_search.search_tools_with_query_expansion(**search_params)
            
            # Check if reranking is enabled
            reranker_used = None
            if search_config.get("enable_reranking"):
                reranked_results = await weaviate_search.search_and_rerank_tools(**search_params)
                results = reranked_results
                reranker_used = search_config.get("reranker", "default")
            
            return {
                "results": results,
                "search_config": search_config,
                "reranker_used": reranker_used,
                "total_results": len(results)
            }
            
        except Exception as e:
            logger.warning(f"Search failed, using mock results: {e}")
            return self._generate_mock_search_results(query, max_results)
    
    def _generate_mock_search_results(self, query: str, max_results: int) -> Dict[str, Any]:
        """Generate mock search results for development/testing"""
        mock_results = []
        
        # Sample tools for testing
        sample_tools = [
            {"name": "Text Analyzer", "description": f"Analyzes text content for {query}", "source": "nlp"},
            {"name": "Data Processor", "description": f"Processes data related to {query}", "source": "data"},
            {"name": "File Manager", "description": f"Manages files for {query} tasks", "source": "filesystem"},
            {"name": "API Client", "description": f"API client for {query} integration", "source": "api"},
            {"name": "Search Tool", "description": f"Advanced search capabilities for {query}", "source": "search"},
        ]
        
        for i in range(min(max_results, len(sample_tools))):
            tool = sample_tools[i % len(sample_tools)]
            mock_results.append({
                "id": f"mock_tool_{i}_{hashlib.md5(query.encode()).hexdigest()[:8]}",
                "name": f"{tool['name']} {i+1}",
                "description": tool["description"],
                "source": tool["source"],
                "category": "testing",
                "original_score": max(0.1, 1.0 - (i * 0.15)),
                "reranked_score": max(0.1, 1.0 - (i * 0.12)) if i < 3 else max(0.1, 1.0 - (i * 0.18)),
                "metadata": {"mock": True, "query": query}
            })
        
        return {
            "results": mock_results,
            "search_config": {"mock": True},
            "reranker_used": None,
            "total_results": len(mock_results)
        }
    
    async def submit_query_evaluation(self, session_id: str, evaluation_data: Dict[str, Any]) -> bool:
        """Submit evaluation for a query"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            if session.current_query_index >= len(session.queries):
                raise ValueError("No more queries to evaluate")
            
            current_query = session.queries[session.current_query_index]
            
            # Process document evaluations
            document_evaluations = []
            for doc_data in evaluation_data.get("document_evaluations", []):
                doc_eval = DocumentEvaluation(
                    document_id=doc_data["document_id"],
                    document_title=doc_data.get("document_title", ""),
                    document_content=doc_data.get("document_content", ""),
                    document_source=doc_data.get("document_source", ""),
                    original_rank=doc_data.get("original_rank", 0),
                    reranked_rank=doc_data.get("reranked_rank"),
                    relevance_score=doc_data.get("relevance_score"),
                    completeness_score=doc_data.get("completeness_score"),
                    accuracy_score=doc_data.get("accuracy_score"),
                    usefulness_score=doc_data.get("usefulness_score"),
                    clarity_score=doc_data.get("clarity_score"),
                    binary_relevant=doc_data.get("binary_relevant"),
                    comments=doc_data.get("comments"),
                    confidence_level=doc_data.get("confidence_level"),
                    time_spent_seconds=doc_data.get("time_spent_seconds"),
                    evaluated_at=datetime.now(timezone.utc),
                    evaluation_id=str(uuid.uuid4())
                )
                
                # Calculate overall score
                scores = [
                    doc_eval.relevance_score or 0,
                    doc_eval.completeness_score or 0, 
                    doc_eval.accuracy_score or 0,
                    doc_eval.usefulness_score or 0
                ]
                doc_eval.overall_score = sum(scores) / len([s for s in scores if s is not None])
                
                document_evaluations.append(doc_eval)
            
            # Update query evaluation
            current_query.document_evaluations = document_evaluations
            current_query.query_difficulty = evaluation_data.get("query_difficulty")
            current_query.query_clarity = evaluation_data.get("query_clarity")
            current_query.expected_result_type = evaluation_data.get("expected_result_type")
            current_query.evaluation_completed_at = datetime.now(timezone.utc)
            
            if current_query.evaluation_started_at:
                start_time = datetime.fromisoformat(evaluation_data.get("evaluation_started_at"))
                current_query.total_evaluation_time_seconds = int(
                    (current_query.evaluation_completed_at - start_time).total_seconds()
                )
            
            # Update session progress
            session.current_query_index += 1
            session.completed_queries += 1
            
            # Check if session is complete
            if session.current_query_index >= len(session.queries):
                session.completed_at = datetime.now(timezone.utc)
                logger.info(f"Evaluation session {session_id} completed")
            
            # Save session
            await self._save_session(session)
            
            logger.info(f"Submitted evaluation for query '{current_query.query_text}' in session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to submit query evaluation: {e}")
            raise
    
    async def _save_session(self, session: EvaluationSession):
        """Save session to storage"""
        try:
            sessions_dir = self.storage_path / "sessions"
            sessions_dir.mkdir(exist_ok=True)
            
            session_file = sessions_dir / f"{session.session_id}.json"
            session_data = self._serialize_session(session)
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save session {session.session_id}: {e}")
            raise
    
    def _serialize_session(self, session: EvaluationSession) -> Dict[str, Any]:
        """Convert session to JSON-serializable format"""
        return {
            "session_id": session.session_id,
            "session_name": session.session_name,
            "evaluator_id": session.evaluator_id,
            "evaluator_role": session.evaluator_role.value,
            "evaluation_criteria": [
                {
                    "name": c.name,
                    "description": c.description,
                    "scale_min": c.scale_min,
                    "scale_max": c.scale_max,
                    "scale_labels": c.scale_labels,
                    "weight": c.weight,
                    "required": c.required
                }
                for c in session.evaluation_criteria
            ],
            "instructions": session.instructions,
            "randomize_order": session.randomize_order,
            "max_documents_per_query": session.max_documents_per_query,
            "queries": [self._serialize_query_evaluation(q) for q in session.queries],
            "created_at": session.created_at.isoformat(),
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
            "paused_at": session.paused_at.isoformat() if session.paused_at else None,
            "current_query_index": session.current_query_index,
            "completed_queries": session.completed_queries,
            "total_queries": session.total_queries,
            "show_original_ranking": session.show_original_ranking,
            "show_reranked_comparison": session.show_reranked_comparison,
            "enable_comments": session.enable_comments,
            "require_confidence_ratings": session.require_confidence_ratings,
            "include_attention_checks": session.include_attention_checks,
            "attention_check_frequency": session.attention_check_frequency
        }
    
    def _serialize_query_evaluation(self, query: QueryEvaluation) -> Dict[str, Any]:
        """Convert query evaluation to JSON-serializable format"""
        return {
            "evaluation_id": query.evaluation_id,
            "query_text": query.query_text,
            "query_intent": query.query_intent,
            "query_category": query.query_category,
            "search_config": query.search_config,
            "reranker_used": query.reranker_used,
            "document_evaluations": [self._serialize_document_evaluation(d) for d in query.document_evaluations],
            "query_difficulty": query.query_difficulty,
            "query_clarity": query.query_clarity,
            "expected_result_type": query.expected_result_type,
            "evaluator_id": query.evaluator_id,
            "evaluator_role": query.evaluator_role.value,
            "evaluator_expertise": query.evaluator_expertise,
            "evaluation_started_at": query.evaluation_started_at.isoformat() if query.evaluation_started_at else None,
            "evaluation_completed_at": query.evaluation_completed_at.isoformat() if query.evaluation_completed_at else None,
            "total_evaluation_time_seconds": query.total_evaluation_time_seconds,
            "attention_checks_passed": query.attention_checks_passed,
            "consistency_score": query.consistency_score,
            "session_id": query.session_id,
            "evaluation_version": query.evaluation_version
        }
    
    def _serialize_document_evaluation(self, doc: DocumentEvaluation) -> Dict[str, Any]:
        """Convert document evaluation to JSON-serializable format"""
        return {
            "document_id": doc.document_id,
            "document_title": doc.document_title,
            "document_content": doc.document_content,
            "document_source": doc.document_source,
            "original_rank": doc.original_rank,
            "reranked_rank": doc.reranked_rank,
            "relevance_score": doc.relevance_score,
            "completeness_score": doc.completeness_score,
            "accuracy_score": doc.accuracy_score,
            "recency_score": doc.recency_score,
            "clarity_score": doc.clarity_score,
            "usefulness_score": doc.usefulness_score,
            "overall_score": doc.overall_score,
            "binary_relevant": doc.binary_relevant,
            "comments": doc.comments,
            "confidence_level": doc.confidence_level,
            "time_spent_seconds": doc.time_spent_seconds,
            "evaluated_at": doc.evaluated_at.isoformat() if doc.evaluated_at else None,
            "evaluation_id": doc.evaluation_id
        }
    
    def _deserialize_session(self, data: Dict[str, Any]) -> Optional[EvaluationSession]:
        """Convert JSON data back to session object"""
        try:
            # Deserialize criteria
            criteria = []
            for c_data in data.get("evaluation_criteria", []):
                criteria.append(EvaluationCriteria(
                    name=c_data["name"],
                    description=c_data["description"],
                    scale_min=c_data["scale_min"],
                    scale_max=c_data["scale_max"],
                    scale_labels=c_data["scale_labels"],
                    weight=c_data["weight"],
                    required=c_data["required"]
                ))
            
            # Deserialize queries
            queries = []
            for q_data in data.get("queries", []):
                query = self._deserialize_query_evaluation(q_data)
                if query:
                    queries.append(query)
            
            # Create session
            session = EvaluationSession(
                session_id=data["session_id"],
                session_name=data["session_name"],
                evaluator_id=data["evaluator_id"],
                evaluator_role=EvaluatorRole(data["evaluator_role"]),
                evaluation_criteria=criteria,
                instructions=data.get("instructions", ""),
                randomize_order=data.get("randomize_order", True),
                max_documents_per_query=data.get("max_documents_per_query", 20),
                queries=queries,
                created_at=datetime.fromisoformat(data["created_at"]),
                started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
                completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
                paused_at=datetime.fromisoformat(data["paused_at"]) if data.get("paused_at") else None,
                current_query_index=data.get("current_query_index", 0),
                completed_queries=data.get("completed_queries", 0),
                total_queries=data.get("total_queries", len(queries)),
                show_original_ranking=data.get("show_original_ranking", True),
                show_reranked_comparison=data.get("show_reranked_comparison", True),
                enable_comments=data.get("enable_comments", True),
                require_confidence_ratings=data.get("require_confidence_ratings", False),
                include_attention_checks=data.get("include_attention_checks", True),
                attention_check_frequency=data.get("attention_check_frequency", 5)
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to deserialize session: {e}")
            return None
    
    def _deserialize_query_evaluation(self, data: Dict[str, Any]) -> Optional[QueryEvaluation]:
        """Convert JSON data back to query evaluation object"""
        try:
            # Deserialize document evaluations
            doc_evaluations = []
            for d_data in data.get("document_evaluations", []):
                doc_eval = self._deserialize_document_evaluation(d_data)
                if doc_eval:
                    doc_evaluations.append(doc_eval)
            
            query = QueryEvaluation(
                evaluation_id=data["evaluation_id"],
                query_text=data["query_text"],
                query_intent=data.get("query_intent"),
                query_category=data.get("query_category"),
                search_config=data.get("search_config", {}),
                reranker_used=data.get("reranker_used"),
                document_evaluations=doc_evaluations,
                query_difficulty=data.get("query_difficulty"),
                query_clarity=data.get("query_clarity"),
                expected_result_type=data.get("expected_result_type"),
                evaluator_id=data["evaluator_id"],
                evaluator_role=EvaluatorRole(data["evaluator_role"]),
                evaluator_expertise=data.get("evaluator_expertise"),
                evaluation_started_at=datetime.fromisoformat(data["evaluation_started_at"]) if data.get("evaluation_started_at") else None,
                evaluation_completed_at=datetime.fromisoformat(data["evaluation_completed_at"]) if data.get("evaluation_completed_at") else None,
                total_evaluation_time_seconds=data.get("total_evaluation_time_seconds"),
                attention_checks_passed=data.get("attention_checks_passed", 0),
                consistency_score=data.get("consistency_score"),
                session_id=data.get("session_id"),
                evaluation_version=data.get("evaluation_version", "1.0")
            )
            
            return query
            
        except Exception as e:
            logger.error(f"Failed to deserialize query evaluation: {e}")
            return None
    
    def _deserialize_document_evaluation(self, data: Dict[str, Any]) -> Optional[DocumentEvaluation]:
        """Convert JSON data back to document evaluation object"""
        try:
            return DocumentEvaluation(
                document_id=data["document_id"],
                document_title=data["document_title"],
                document_content=data["document_content"],
                document_source=data["document_source"],
                original_rank=data["original_rank"],
                reranked_rank=data.get("reranked_rank"),
                relevance_score=data.get("relevance_score"),
                completeness_score=data.get("completeness_score"),
                accuracy_score=data.get("accuracy_score"),
                recency_score=data.get("recency_score"),
                clarity_score=data.get("clarity_score"),
                usefulness_score=data.get("usefulness_score"),
                overall_score=data.get("overall_score"),
                binary_relevant=data.get("binary_relevant"),
                comments=data.get("comments"),
                confidence_level=data.get("confidence_level"),
                time_spent_seconds=data.get("time_spent_seconds"),
                evaluated_at=datetime.fromisoformat(data["evaluated_at"]) if data.get("evaluated_at") else None,
                evaluation_id=data.get("evaluation_id")
            )
            
        except Exception as e:
            logger.error(f"Failed to deserialize document evaluation: {e}")
            return None
    
    async def list_evaluation_sessions(self, evaluator_id: Optional[str] = None,
                                      completed_only: bool = False) -> List[Dict[str, Any]]:
        """List evaluation sessions"""
        sessions = []
        
        for session in self.active_sessions.values():
            if evaluator_id and session.evaluator_id != evaluator_id:
                continue
                
            if completed_only and not session.completed_at:
                continue
            
            session_info = await self.get_session_info(session.session_id)
            if session_info:
                sessions.append(session_info)
        
        # Sort by creation date, newest first
        sessions.sort(key=lambda x: x["created_at"], reverse=True)
        
        return sessions
    
    async def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics"""
        stats = {
            "total_sessions": len(self.active_sessions),
            "completed_sessions": sum(1 for s in self.active_sessions.values() if s.completed_at),
            "total_queries_evaluated": sum(s.completed_queries for s in self.active_sessions.values()),
            "total_documents_evaluated": 0,
            "average_evaluation_time_per_query": 0,
            "evaluator_distribution": {},
            "evaluation_criteria_usage": {}
        }
        
        total_eval_time = 0
        evaluated_queries = 0
        
        for session in self.active_sessions.values():
            # Evaluator distribution
            role = session.evaluator_role.value
            stats["evaluator_distribution"][role] = stats["evaluator_distribution"].get(role, 0) + 1
            
            # Process completed queries
            for query in session.queries:
                if query.evaluation_completed_at:
                    evaluated_queries += 1
                    stats["total_documents_evaluated"] += len(query.document_evaluations)
                    
                    if query.total_evaluation_time_seconds:
                        total_eval_time += query.total_evaluation_time_seconds
        
        if evaluated_queries > 0:
            stats["average_evaluation_time_per_query"] = total_eval_time / evaluated_queries
        
        return stats

# Global evaluation interface instance
evaluation_interface: Optional[ManualEvaluationInterface] = None

async def initialize_evaluation_interface(storage_path: Optional[str] = None) -> bool:
    """Initialize global evaluation interface"""
    global evaluation_interface
    
    evaluation_interface = ManualEvaluationInterface(storage_path)
    return await evaluation_interface.initialize()

def get_evaluation_interface() -> ManualEvaluationInterface:
    """Get global evaluation interface instance"""
    if evaluation_interface is None:
        raise RuntimeError("Evaluation interface not initialized")
    return evaluation_interface