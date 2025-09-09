"""
LDTS-28: Configuration validation service

Comprehensive validation system for all dashboard configuration parameters.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pydantic import BaseModel, Field, validator
from enum import Enum
import re
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Configuration validation levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    BLOCKING = "blocking"

class ValidationResult(BaseModel):
    """Single validation result"""
    field: str
    level: ValidationLevel
    message: str
    suggestion: Optional[str] = None
    valid_range: Optional[Dict[str, Any]] = None

class ConfigValidationResponse(BaseModel):
    """Complete validation response"""
    valid: bool
    errors: List[ValidationResult]
    warnings: List[ValidationResult]
    suggestions: List[ValidationResult]
    cost_estimate: Optional[Dict[str, Any]] = None
    performance_impact: Optional[Dict[str, str]] = None

class EmbeddingConfig(BaseModel):
    """Embedding provider configuration"""
    provider: str = Field(..., description="Provider name (openai, ollama, custom)")
    model: str = Field(..., description="Model identifier")
    dimension: int = Field(..., ge=1, le=4096, description="Embedding dimension")
    pooling: str = Field("mean", description="Pooling strategy")
    normalize_l2: bool = Field(True, description="L2 normalization")
    max_tokens: int = Field(8192, ge=1, le=32768, description="Max tokens")
    precision: str = Field("fp32", description="Precision (fp32, fp16, int8)")
    batch_size: int = Field(32, ge=1, le=64, description="Batch size")

class WeaviateConfig(BaseModel):
    """Weaviate search configuration"""
    distance_metric: str = Field("cosine", description="Distance metric")
    vector_search: Dict[str, Any] = Field(default_factory=dict)
    hybrid: Dict[str, Any] = Field(default_factory=dict)
    bm25: Dict[str, Any] = Field(default_factory=dict)
    operational: Dict[str, Any] = Field(default_factory=dict)

class RerankerConfig(BaseModel):
    """Reranker configuration"""
    enabled: bool = Field(True, description="Enable reranking")
    type: str = Field("cross-encoder", description="Reranker type")
    model: str = Field(..., description="Reranker model")
    scoring: Dict[str, Any] = Field(default_factory=dict)
    processing: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)

class DashboardConfig(BaseModel):
    """Complete dashboard configuration"""
    dashboard: Dict[str, Any] = Field(default_factory=dict)
    search: Dict[str, Any] = Field(default_factory=dict)
    reranker: Dict[str, Any] = Field(default_factory=dict)
    experiments: Dict[str, Any] = Field(default_factory=dict)
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    ui: Dict[str, Any] = Field(default_factory=dict)
    operations: Dict[str, Any] = Field(default_factory=dict)

class ConfigurationValidator:
    """Comprehensive configuration validation system"""
    
    def __init__(self):
        self.validation_rules = self._load_validation_rules()
        self.provider_limits = self._load_provider_limits()
        
    def validate_complete_config(self, config: Dict[str, Any]) -> ConfigValidationResponse:
        """Validate complete dashboard configuration"""
        results = []
        
        # Validate each section
        if "search" in config:
            results.extend(self._validate_search_config(config["search"]))
        
        if "reranker" in config:
            results.extend(self._validate_reranker_config(config["reranker"]))
            
        if "experiments" in config:
            results.extend(self._validate_experiments_config(config["experiments"]))
            
        if "evaluation" in config:
            results.extend(self._validate_evaluation_config(config["evaluation"]))
        
        # Cross-validation between sections
        results.extend(self._validate_cross_section_compatibility(config))
        
        # Calculate overall validation status
        errors = [r for r in results if r.level in [ValidationLevel.ERROR, ValidationLevel.BLOCKING]]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        suggestions = [r for r in results if r.level == ValidationLevel.INFO]
        
        valid = len(errors) == 0
        
        # Estimate costs and performance impact
        cost_estimate = self._estimate_configuration_cost(config) if valid else None
        performance_impact = self._assess_performance_impact(config) if valid else None
        
        return ConfigValidationResponse(
            valid=valid,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            cost_estimate=cost_estimate,
            performance_impact=performance_impact
        )
    
    def _validate_search_config(self, search_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate search configuration section"""
        results = []
        
        # Validate embedding configuration
        if "embedding" in search_config:
            results.extend(self._validate_embedding_config(search_config["embedding"]))
        
        # Validate Weaviate configuration
        if "weaviate" in search_config:
            results.extend(self._validate_weaviate_config(search_config["weaviate"]))
        
        return results
    
    def _validate_embedding_config(self, embedding_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate embedding provider configuration"""
        results = []
        
        provider = embedding_config.get("provider")
        if not provider:
            results.append(ValidationResult(
                field="embedding.provider",
                level=ValidationLevel.ERROR,
                message="Embedding provider is required",
                suggestion="Specify provider: 'openai', 'ollama', or 'custom'"
            ))
            return results
        
        # Provider-specific validation
        if provider == "openai":
            results.extend(self._validate_openai_embedding_config(embedding_config))
        elif provider == "ollama":
            results.extend(self._validate_ollama_embedding_config(embedding_config))
        elif provider == "custom":
            results.extend(self._validate_custom_embedding_config(embedding_config))
        else:
            results.append(ValidationResult(
                field="embedding.provider",
                level=ValidationLevel.ERROR,
                message=f"Unknown embedding provider: {provider}",
                suggestion="Use 'openai', 'ollama', or 'custom'"
            ))
        
        # Validate common parameters
        dimension = embedding_config.get("dimension", 0)
        if dimension < 1 or dimension > 4096:
            results.append(ValidationResult(
                field="embedding.dimension",
                level=ValidationLevel.ERROR,
                message=f"Invalid embedding dimension: {dimension}",
                valid_range={"min": 1, "max": 4096}
            ))
        
        max_tokens = embedding_config.get("max_tokens", 8192)
        if max_tokens < 1 or max_tokens > 32768:
            results.append(ValidationResult(
                field="embedding.max_tokens",
                level=ValidationLevel.WARNING,
                message=f"Unusual max_tokens value: {max_tokens}",
                valid_range={"min": 1, "max": 32768, "recommended": 8192}
            ))
        
        return results
    
    def _validate_openai_embedding_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate OpenAI-specific embedding configuration"""
        results = []
        
        model = config.get("model")
        valid_models = ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
        
        if model not in valid_models:
            results.append(ValidationResult(
                field="embedding.model",
                level=ValidationLevel.WARNING,
                message=f"Unknown OpenAI model: {model}",
                suggestion=f"Recommended models: {', '.join(valid_models)}"
            ))
        
        # Check dimension compatibility
        dimension = config.get("dimension", 1536)
        model_dimensions = {
            "text-embedding-3-small": [512, 1536],
            "text-embedding-3-large": [256, 1024, 3072],
            "text-embedding-ada-002": [1536]
        }
        
        if model in model_dimensions and dimension not in model_dimensions[model]:
            results.append(ValidationResult(
                field="embedding.dimension",
                level=ValidationLevel.ERROR,
                message=f"Invalid dimension {dimension} for model {model}",
                valid_range={"valid_values": model_dimensions[model]}
            ))
        
        return results
    
    def _validate_weaviate_config(self, weaviate_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate Weaviate configuration"""
        results = []
        
        # Validate hybrid search parameters
        if "hybrid" in weaviate_config:
            hybrid = weaviate_config["hybrid"]
            alpha = hybrid.get("alpha")
            
            if alpha is not None:
                if not (0.0 <= alpha <= 1.0):
                    results.append(ValidationResult(
                        field="weaviate.hybrid.alpha",
                        level=ValidationLevel.ERROR,
                        message=f"Alpha must be between 0.0 and 1.0, got {alpha}",
                        valid_range={"min": 0.0, "max": 1.0}
                    ))
                
                # Performance recommendations
                if alpha < 0.1:
                    results.append(ValidationResult(
                        field="weaviate.hybrid.alpha",
                        level=ValidationLevel.WARNING,
                        message="Very low alpha value may reduce search quality",
                        suggestion="Consider alpha >= 0.25 for balanced search"
                    ))
                elif alpha > 0.95:
                    results.append(ValidationResult(
                        field="weaviate.hybrid.alpha",
                        level=ValidationLevel.WARNING,
                        message="Very high alpha value may miss keyword matches",
                        suggestion="Consider alpha <= 0.9 for hybrid benefits"
                    ))
        
        # Validate BM25 parameters
        if "bm25" in weaviate_config:
            bm25 = weaviate_config["bm25"]
            
            k1 = bm25.get("k1")
            if k1 is not None and not (0.5 <= k1 <= 3.0):
                results.append(ValidationResult(
                    field="weaviate.bm25.k1",
                    level=ValidationLevel.WARNING,
                    message=f"Unusual BM25 k1 value: {k1}",
                    valid_range={"min": 0.5, "max": 3.0, "typical": 1.2}
                ))
            
            b = bm25.get("b")
            if b is not None and not (0.0 <= b <= 1.0):
                results.append(ValidationResult(
                    field="weaviate.bm25.b",
                    level=ValidationLevel.ERROR,
                    message=f"BM25 b parameter must be between 0.0 and 1.0, got {b}",
                    valid_range={"min": 0.0, "max": 1.0, "typical": 0.75}
                ))
        
        return results
    
    def _validate_reranker_config(self, reranker_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate reranker configuration"""
        results = []
        
        if not reranker_config.get("enabled", True):
            return results  # Skip validation if disabled
        
        reranker_type = reranker_config.get("type")
        valid_types = ["cross-encoder", "bi-encoder", "lambdamart", "noop", "shuffle"]
        
        if reranker_type not in valid_types:
            results.append(ValidationResult(
                field="reranker.type",
                level=ValidationLevel.ERROR,
                message=f"Invalid reranker type: {reranker_type}",
                suggestion=f"Valid types: {', '.join(valid_types)}"
            ))
        
        # Validate scoring parameters
        if "scoring" in reranker_config:
            scoring = reranker_config["scoring"]
            
            top_k = scoring.get("top_k")
            if top_k is not None and not (1 <= top_k <= 100):
                results.append(ValidationResult(
                    field="reranker.scoring.top_k",
                    level=ValidationLevel.ERROR,
                    message=f"top_k must be between 1 and 100, got {top_k}",
                    valid_range={"min": 1, "max": 100}
                ))
            
            threshold = scoring.get("threshold")
            if threshold is not None and not (0.0 <= threshold <= 1.0):
                results.append(ValidationResult(
                    field="reranker.scoring.threshold",
                    level=ValidationLevel.ERROR,
                    message=f"Threshold must be between 0.0 and 1.0, got {threshold}",
                    valid_range={"min": 0.0, "max": 1.0}
                ))
        
        return results
    
    def _validate_cross_section_compatibility(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate compatibility between different configuration sections"""
        results = []
        
        # Check embedding dimension compatibility with Weaviate
        search_config = config.get("search", {})
        embedding_dim = search_config.get("embedding", {}).get("dimension")
        
        if embedding_dim:
            # Validate against expected Weaviate schema
            expected_dim = 1536  # Default for most configurations
            if embedding_dim != expected_dim:
                results.append(ValidationResult(
                    field="compatibility.embedding_dimension",
                    level=ValidationLevel.WARNING,
                    message=f"Embedding dimension {embedding_dim} may not match Weaviate schema",
                    suggestion=f"Ensure Weaviate schema supports dimension {embedding_dim}"
                ))
        
        # Check reranker compatibility with search results
        reranker_config = config.get("reranker", {})
        if reranker_config.get("enabled", True):
            search_limit = search_config.get("weaviate", {}).get("vector_search", {}).get("limit", 50)
            reranker_top_k = reranker_config.get("scoring", {}).get("top_k", 10)
            
            if reranker_top_k > search_limit:
                results.append(ValidationResult(
                    field="compatibility.reranker_top_k",
                    level=ValidationLevel.ERROR,
                    message=f"Reranker top_k ({reranker_top_k}) exceeds search limit ({search_limit})",
                    suggestion="Increase search limit or reduce reranker top_k"
                ))
        
        return results
    
    def _estimate_configuration_cost(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate costs for the given configuration"""
        cost_estimate = {
            "daily_estimate_usd": 0.0,
            "per_query_estimate_usd": 0.0,
            "breakdown": {}
        }
        
        # Estimate embedding costs
        search_config = config.get("search", {})
        embedding_config = search_config.get("embedding", {})
        
        if embedding_config.get("provider") == "openai":
            model = embedding_config.get("model", "text-embedding-3-small")
            tokens_per_query = embedding_config.get("max_tokens", 8192)
            
            # OpenAI pricing (approximate)
            cost_per_1k_tokens = 0.0001 if "small" in model else 0.00013
            cost_per_query = (tokens_per_query / 1000) * cost_per_1k_tokens
            
            cost_estimate["per_query_estimate_usd"] += cost_per_query
            cost_estimate["breakdown"]["embedding"] = cost_per_query
        
        # Estimate reranker costs
        reranker_config = config.get("reranker", {})
        if reranker_config.get("enabled", True) and reranker_config.get("type") != "noop":
            # Approximate reranker cost
            reranker_cost_per_query = 0.001  # Placeholder
            cost_estimate["per_query_estimate_usd"] += reranker_cost_per_query
            cost_estimate["breakdown"]["reranking"] = reranker_cost_per_query
        
        # Daily estimate (assuming 1000 queries/day)
        queries_per_day = 1000
        cost_estimate["daily_estimate_usd"] = cost_estimate["per_query_estimate_usd"] * queries_per_day
        
        return cost_estimate
    
    def _assess_performance_impact(self, config: Dict[str, Any]) -> Dict[str, str]:
        """Assess performance impact of configuration"""
        impact = {
            "latency": "low",
            "memory": "low", 
            "cpu": "low",
            "network": "low"
        }
        
        # Assess based on configuration choices
        search_config = config.get("search", {})
        embedding_dim = search_config.get("embedding", {}).get("dimension", 1536)
        
        if embedding_dim > 2000:
            impact["memory"] = "high"
            impact["cpu"] = "medium"
        
        reranker_config = config.get("reranker", {})
        if reranker_config.get("enabled", True):
            reranker_type = reranker_config.get("type", "cross-encoder")
            top_k = reranker_config.get("scoring", {}).get("top_k", 10)
            
            if reranker_type in ["cross-encoder", "bi-encoder"] and top_k > 20:
                impact["latency"] = "high"
                impact["cpu"] = "high"
        
        return impact
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules configuration"""
        return {
            "embedding_providers": ["openai", "ollama", "custom"],
            "reranker_types": ["cross-encoder", "bi-encoder", "lambdamart", "noop", "shuffle"],
            "distance_metrics": ["cosine", "dot", "l2", "euclidean"]
        }
    
    def _load_provider_limits(self) -> Dict[str, Any]:
        """Load provider-specific limits and constraints"""
        return {
            "openai": {
                "max_tokens": 8192,
                "rate_limit_rpm": 3000,
                "max_batch_size": 64
            },
            "ollama": {
                "max_tokens": 4096,
                "concurrent_requests": 4
            }
        }
    
    def _validate_ollama_embedding_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate Ollama-specific configuration"""
        results = []
        
        base_url = config.get("base_url")
        if not base_url:
            results.append(ValidationResult(
                field="embedding.base_url",
                level=ValidationLevel.ERROR,
                message="Ollama base_url is required",
                suggestion="Provide Ollama server URL (e.g., http://localhost:11434)"
            ))
        
        return results
    
    def _validate_custom_embedding_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate custom provider configuration"""
        results = []
        
        api_url = config.get("api_url")
        if not api_url:
            results.append(ValidationResult(
                field="embedding.api_url",
                level=ValidationLevel.ERROR,
                message="Custom provider api_url is required",
                suggestion="Provide the embedding API endpoint URL"
            ))
        
        return results
    
    def _validate_experiments_config(self, experiments_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate experiments configuration"""
        results = []
        
        cost_controls = experiments_config.get("cost_controls", {})
        daily_budget = cost_controls.get("daily_budget_usd")
        
        if daily_budget is not None and daily_budget < 1.0:
            results.append(ValidationResult(
                field="experiments.cost_controls.daily_budget_usd",
                level=ValidationLevel.WARNING,
                message=f"Very low daily budget: ${daily_budget}",
                suggestion="Consider budget >= $5 for meaningful experiments"
            ))
        
        return results
    
    def _validate_evaluation_config(self, evaluation_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate evaluation configuration"""
        results = []
        
        metrics = evaluation_config.get("metrics", [])
        if not metrics:
            results.append(ValidationResult(
                field="evaluation.metrics",
                level=ValidationLevel.WARNING,
                message="No evaluation metrics specified",
                suggestion="Add metrics like precision_at_k, ndcg_at_k for comprehensive evaluation"
            ))
        
        return results

# Global validator instance
config_validator = ConfigurationValidator()

def validate_configuration(config: Dict[str, Any]) -> ConfigValidationResponse:
    """Validate a complete configuration"""
    return config_validator.validate_complete_config(config)

def validate_yaml_config(yaml_content: str) -> ConfigValidationResponse:
    """Validate configuration from YAML content"""
    try:
        config = yaml.safe_load(yaml_content)
        return validate_configuration(config)
    except yaml.YAMLError as e:
        return ConfigValidationResponse(
            valid=False,
            errors=[ValidationResult(
                field="yaml.syntax",
                level=ValidationLevel.BLOCKING,
                message=f"YAML syntax error: {e}"
            )],
            warnings=[],
            suggestions=[]
        )