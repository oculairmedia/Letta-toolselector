"""
LDTS-69: Simple configuration validation for API server
Simplified version of config_validation.py for the existing API server
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum
import re

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Configuration validation levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    BLOCKING = "blocking"

class ValidationResult:
    """Single validation result"""
    def __init__(self, field: str, level: ValidationLevel, message: str, 
                 suggestion: Optional[str] = None, valid_range: Optional[Dict[str, Any]] = None):
        self.field = field
        self.level = level
        self.message = message
        self.suggestion = suggestion
        self.valid_range = valid_range
    
    def to_dict(self):
        return {
            "field": self.field,
            "level": self.level.value,
            "message": self.message,
            "suggestion": self.suggestion,
            "valid_range": self.valid_range
        }

class ConfigValidationResponse:
    """Complete validation response"""
    def __init__(self, valid: bool, errors: List[ValidationResult], 
                 warnings: List[ValidationResult], suggestions: List[ValidationResult],
                 cost_estimate: Optional[Dict[str, Any]] = None, 
                 performance_impact: Optional[Dict[str, str]] = None):
        self.valid = valid
        self.errors = errors
        self.warnings = warnings
        self.suggestions = suggestions
        self.cost_estimate = cost_estimate
        self.performance_impact = performance_impact
    
    def to_dict(self):
        return {
            "valid": self.valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [w.to_dict() for w in self.warnings],
            "suggestions": [s.to_dict() for s in self.suggestions],
            "cost_estimate": self.cost_estimate,
            "performance_impact": self.performance_impact
        }

class SimpleConfigurationValidator:
    """Simplified configuration validation system for API server"""
    
    def __init__(self):
        self.validation_rules = {
            "embedding_providers": ["openai", "ollama", "custom"],
            "reranker_types": ["cross-encoder", "bi-encoder", "lambdamart", "noop", "shuffle"],
            "distance_metrics": ["cosine", "dot", "l2", "euclidean"]
        }
        
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
        
        # Provider validation
        if provider not in self.validation_rules["embedding_providers"]:
            results.append(ValidationResult(
                field="embedding.provider",
                level=ValidationLevel.ERROR,
                message=f"Unknown embedding provider: {provider}",
                suggestion="Use 'openai', 'ollama', or 'custom'"
            ))
        
        # Validate dimension
        dimension = embedding_config.get("dimension", 0)
        if dimension < 1 or dimension > 4096:
            results.append(ValidationResult(
                field="embedding.dimension",
                level=ValidationLevel.ERROR,
                message=f"Invalid embedding dimension: {dimension}",
                valid_range={"min": 1, "max": 4096}
            ))
        
        # Validate max_tokens
        max_tokens = embedding_config.get("max_tokens", 8192)
        if max_tokens < 1 or max_tokens > 32768:
            results.append(ValidationResult(
                field="embedding.max_tokens",
                level=ValidationLevel.WARNING,
                message=f"Unusual max_tokens value: {max_tokens}",
                valid_range={"min": 1, "max": 32768, "recommended": 8192}
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
        
        return results
    
    def _validate_reranker_config(self, reranker_config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate reranker configuration"""
        results = []
        
        if not reranker_config.get("enabled", True):
            return results  # Skip validation if disabled
        
        reranker_type = reranker_config.get("type")
        if reranker_type and reranker_type not in self.validation_rules["reranker_types"]:
            results.append(ValidationResult(
                field="reranker.type",
                level=ValidationLevel.ERROR,
                message=f"Invalid reranker type: {reranker_type}",
                suggestion=f"Valid types: {', '.join(self.validation_rules['reranker_types'])}"
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
    
    def _validate_cross_section_compatibility(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate compatibility between different configuration sections"""
        results = []
        
        # Check embedding dimension compatibility
        search_config = config.get("search", {})
        embedding_dim = search_config.get("embedding", {}).get("dimension")
        
        if embedding_dim:
            expected_dim = 1536  # Default for most configurations
            if embedding_dim != expected_dim:
                results.append(ValidationResult(
                    field="compatibility.embedding_dimension",
                    level=ValidationLevel.WARNING,
                    message=f"Embedding dimension {embedding_dim} may not match Weaviate schema",
                    suggestion=f"Ensure Weaviate schema supports dimension {embedding_dim}"
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

# Global validator instance
config_validator = SimpleConfigurationValidator()

def validate_configuration(config: Dict[str, Any]) -> ConfigValidationResponse:
    """Validate a complete configuration"""
    return config_validator.validate_complete_config(config)