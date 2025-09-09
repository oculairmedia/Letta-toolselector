"""
LDTS-37: Advanced Configuration Validation Engine

Comprehensive validation system for all LDTS dashboard configuration components
including embedding providers, rerankers, Weaviate settings, and cross-component
compatibility checks.
"""

import logging
import asyncio
import re
import json
from typing import Dict, Any, List, Optional, Union, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import yaml
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels"""
    ERROR = "error"          # Blocks configuration usage
    WARNING = "warning"      # Suboptimal but usable
    INFO = "info"           # Informational only
    SUGGESTION = "suggestion" # Optimization suggestions

class ValidationCategory(Enum):
    """Categories of validation issues"""
    SYNTAX = "syntax"
    COMPATIBILITY = "compatibility"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COST = "cost"
    RESOURCE = "resource"
    DEPENDENCY = "dependency"

@dataclass
class ValidationIssue:
    """Individual validation issue"""
    level: ValidationLevel
    category: ValidationCategory
    message: str
    path: str                 # Configuration path (e.g., "embedding_providers.openai.api_key")
    suggestion: Optional[str] = None
    impact: Optional[str] = None
    auto_fix: Optional[Dict[str, Any]] = None  # Automatic fix if available

@dataclass
class ValidationResult:
    """Complete validation result"""
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    suggestions: List[ValidationIssue] = field(default_factory=list)
    cost_estimate: Optional[Dict[str, Any]] = None
    performance_impact: Optional[Dict[str, Any]] = None
    compatibility_matrix: Optional[Dict[str, Any]] = None
    validation_time: Optional[float] = None

class AdvancedConfigValidator:
    """Advanced configuration validation engine"""
    
    def __init__(self):
        self.schema_cache = {}
        self.validation_rules = {}
        self._load_validation_rules()
    
    def _load_validation_rules(self):
        """Load validation rules and constraints"""
        self.validation_rules = {
            "embedding_providers": {
                "required_fields": ["default_provider"],
                "valid_providers": ["openai", "ollama", "huggingface"],
                "provider_specific": {
                    "openai": {
                        "required": ["api_key", "model"],
                        "valid_models": ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                        "dimension_constraints": {
                            "text-embedding-3-small": [512, 1536],
                            "text-embedding-3-large": [256, 1024, 3072],
                            "text-embedding-ada-002": [1536]
                        }
                    },
                    "ollama": {
                        "required": ["base_url", "model"],
                        "valid_models": ["nomic-embed-text", "all-minilm", "mxbai-embed-large"],
                        "url_pattern": r"^https?://[\w\.-]+:\d+$"
                    }
                }
            },
            "reranker_models": {
                "required_fields": ["default_reranker"],
                "valid_rerankers": ["ollama_reranker", "cohere_reranker"],
                "reranker_specific": {
                    "ollama_reranker": {
                        "required": ["base_url", "model"],
                        "valid_models": ["qwen3-reranker", "bge-reranker", "ms-marco-reranker"],
                        "max_batch_size_limit": 64,
                        "temperature_range": [0.0, 1.0]
                    }
                }
            },
            "weaviate": {
                "required_fields": ["url"],
                "url_pattern": r"^https?://[\w\.-]+:\d+$",
                "schema_constraints": {
                    "ef": {"min": 16, "max": 512, "recommended": [64, 128]},
                    "ef_construction": {"min": 32, "max": 1024, "recommended": [128, 256]},
                    "max_connections": {"min": 16, "max": 128, "recommended": [32, 64]},
                    "alpha": {"min": 0.0, "max": 1.0, "recommended": [0.5, 0.8]}
                }
            },
            "performance": {
                "rate_limiting": {
                    "max_requests_per_minute": 10000,
                    "max_concurrent_requests": 1000
                },
                "resource_limits": {
                    "max_memory_mb": 16384,
                    "max_cpu_percent": 95.0
                }
            }
        }
    
    async def validate_complete_configuration(self, config: Dict[str, Any]) -> ValidationResult:
        """Perform comprehensive configuration validation"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            issues = []
            warnings = []
            suggestions = []
            
            # 1. Schema validation
            schema_issues = await self._validate_schema_structure(config)
            issues.extend([issue for issue in schema_issues if issue.level == ValidationLevel.ERROR])
            warnings.extend([issue for issue in schema_issues if issue.level == ValidationLevel.WARNING])
            
            # 2. Embedding provider validation
            embedding_issues = await self._validate_embedding_providers(config.get('embedding_providers', {}))
            self._categorize_issues(embedding_issues, issues, warnings, suggestions)
            
            # 3. Reranker validation
            reranker_issues = await self._validate_reranker_models(config.get('reranker_models', {}))
            self._categorize_issues(reranker_issues, issues, warnings, suggestions)
            
            # 4. Weaviate configuration validation
            weaviate_issues = await self._validate_weaviate_config(config.get('weaviate', {}))
            self._categorize_issues(weaviate_issues, issues, warnings, suggestions)
            
            # 5. Search configuration validation
            search_issues = await self._validate_search_config(config.get('search_config', {}))
            self._categorize_issues(search_issues, issues, warnings, suggestions)
            
            # 6. Cross-component compatibility
            compatibility_issues = await self._validate_cross_component_compatibility(config)
            self._categorize_issues(compatibility_issues, issues, warnings, suggestions)
            
            # 7. Performance impact analysis
            performance_impact = await self._analyze_performance_impact(config)
            
            # 8. Cost estimation
            cost_estimate = await self._estimate_configuration_cost(config)
            
            # 9. Security validation
            security_issues = await self._validate_security_settings(config.get('security', {}))
            self._categorize_issues(security_issues, issues, warnings, suggestions)
            
            validation_time = asyncio.get_event_loop().time() - start_time
            
            return ValidationResult(
                valid=len(issues) == 0,
                issues=issues,
                warnings=warnings,
                suggestions=suggestions,
                cost_estimate=cost_estimate,
                performance_impact=performance_impact,
                validation_time=validation_time
            )
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return ValidationResult(
                valid=False,
                issues=[ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message=f"Validation engine error: {str(e)}",
                    path="root"
                )],
                validation_time=asyncio.get_event_loop().time() - start_time
            )
    
    async def _validate_schema_structure(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate basic schema structure"""
        issues = []
        
        # Check required top-level sections
        required_sections = [
            'embedding_providers', 'reranker_models', 'weaviate', 
            'search_config', 'dashboard', 'security'
        ]
        
        for section in required_sections:
            if section not in config:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message=f"Missing required configuration section: {section}",
                    path=section,
                    suggestion=f"Add the '{section}' section to your configuration",
                    auto_fix={section: {}}
                ))
        
        # Validate configuration depth (prevent overly nested configs)
        max_depth = self._get_max_depth(config)
        if max_depth > 10:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SYNTAX,
                message=f"Configuration nesting too deep ({max_depth} levels). Consider flattening.",
                path="root",
                impact="May impact readability and maintenance"
            ))
        
        return issues
    
    async def _validate_embedding_providers(self, providers_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate embedding providers configuration"""
        issues = []
        
        if not providers_config:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DEPENDENCY,
                message="Embedding providers configuration is empty",
                path="embedding_providers"
            ))
            return issues
        
        # Check default provider
        default_provider = providers_config.get('default_provider')
        if not default_provider:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DEPENDENCY,
                message="No default embedding provider specified",
                path="embedding_providers.default_provider",
                suggestion="Set default_provider to one of: openai, ollama, huggingface"
            ))
        elif default_provider not in providers_config:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DEPENDENCY,
                message=f"Default provider '{default_provider}' not configured",
                path=f"embedding_providers.{default_provider}",
                suggestion=f"Add configuration for {default_provider} provider"
            ))
        
        # Validate individual providers
        for provider_name, provider_config in providers_config.items():
            if provider_name in ['default_provider', 'fallback_provider']:
                continue
            
            if not isinstance(provider_config, dict):
                continue
                
            provider_issues = await self._validate_single_embedding_provider(
                provider_name, provider_config
            )
            issues.extend(provider_issues)
        
        # Check for at least one enabled provider
        enabled_providers = [
            name for name, config in providers_config.items() 
            if isinstance(config, dict) and config.get('enabled', False)
        ]
        
        if not enabled_providers:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DEPENDENCY,
                message="No embedding providers are enabled",
                path="embedding_providers",
                suggestion="Enable at least one embedding provider"
            ))
        
        return issues
    
    async def _validate_single_embedding_provider(self, name: str, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate a single embedding provider"""
        issues = []
        path_prefix = f"embedding_providers.{name}"
        
        if not config.get('enabled', False):
            return issues  # Skip validation for disabled providers
        
        # Provider-specific validation
        if name == "openai":
            if not config.get('api_key'):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SECURITY,
                    message="OpenAI API key is required but not provided",
                    path=f"{path_prefix}.api_key",
                    suggestion="Set OPENAI_API_KEY environment variable or provide api_key in config"
                ))
            
            model = config.get('model', '')
            valid_models = self.validation_rules["embedding_providers"]["provider_specific"]["openai"]["valid_models"]
            if model and model not in valid_models:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.COMPATIBILITY,
                    message=f"OpenAI model '{model}' not in recommended list",
                    path=f"{path_prefix}.model",
                    suggestion=f"Consider using one of: {', '.join(valid_models)}"
                ))
            
            # Validate dimensions for specific models
            dimensions = config.get('dimensions')
            if model and dimensions:
                constraints = self.validation_rules["embedding_providers"]["provider_specific"]["openai"]["dimension_constraints"]
                valid_dims = constraints.get(model, [])
                if valid_dims and dimensions not in valid_dims:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category=ValidationCategory.COMPATIBILITY,
                        message=f"Invalid dimensions {dimensions} for model {model}",
                        path=f"{path_prefix}.dimensions",
                        suggestion=f"Valid dimensions for {model}: {valid_dims}"
                    ))
        
        elif name == "ollama":
            base_url = config.get('base_url', '')
            if not base_url:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.DEPENDENCY,
                    message="Ollama base_url is required",
                    path=f"{path_prefix}.base_url",
                    suggestion="Provide Ollama server URL (e.g., http://localhost:11434)"
                ))
            elif not re.match(self.validation_rules["embedding_providers"]["provider_specific"]["ollama"]["url_pattern"], base_url):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.SYNTAX,
                    message=f"Ollama base_url format may be invalid: {base_url}",
                    path=f"{path_prefix}.base_url",
                    suggestion="Use format: http://hostname:port"
                ))
        
        # Common validation for all providers
        batch_size = config.get('batch_size', 0)
        if batch_size > 1000:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message=f"Large batch size ({batch_size}) may cause memory issues",
                path=f"{path_prefix}.batch_size",
                suggestion="Consider reducing batch_size to 100-500 for better stability"
            ))
        
        timeout = config.get('timeout_seconds', 0)
        if timeout > 300:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message=f"Very long timeout ({timeout}s) may block operations",
                path=f"{path_prefix}.timeout_seconds",
                suggestion="Consider using timeout between 30-120 seconds"
            ))
        
        return issues
    
    async def _validate_reranker_models(self, reranker_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate reranker models configuration"""
        issues = []
        
        if not reranker_config:
            issues.append(ValidationIssue(
                level=ValidationLevel.INFO,
                category=ValidationCategory.DEPENDENCY,
                message="No reranker models configured (optional)",
                path="reranker_models"
            ))
            return issues
        
        # Check default reranker
        default_reranker = reranker_config.get('default_reranker')
        if default_reranker and default_reranker not in reranker_config:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DEPENDENCY,
                message=f"Default reranker '{default_reranker}' not configured",
                path=f"reranker_models.{default_reranker}",
                suggestion=f"Add configuration for {default_reranker}"
            ))
        
        # Validate individual rerankers
        for reranker_name, reranker_config_item in reranker_config.items():
            if reranker_name == 'default_reranker':
                continue
                
            if isinstance(reranker_config_item, dict):
                reranker_issues = await self._validate_single_reranker(reranker_name, reranker_config_item)
                issues.extend(reranker_issues)
        
        return issues
    
    async def _validate_single_reranker(self, name: str, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate a single reranker configuration"""
        issues = []
        path_prefix = f"reranker_models.{name}"
        
        if not config.get('enabled', False):
            return issues
        
        if name == "ollama_reranker":
            base_url = config.get('base_url', '')
            if not base_url:
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.DEPENDENCY,
                    message="Ollama reranker base_url is required",
                    path=f"{path_prefix}.base_url",
                    suggestion="Provide Ollama server URL"
                ))
            
            max_batch_size = config.get('max_batch_size', 0)
            if max_batch_size > 64:
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.PERFORMANCE,
                    message=f"Large reranker batch size ({max_batch_size}) may be slow",
                    path=f"{path_prefix}.max_batch_size",
                    suggestion="Consider batch size of 8-32 for better performance"
                ))
            
            temperature = config.get('temperature', 0.0)
            if not (0.0 <= temperature <= 1.0):
                issues.append(ValidationIssue(
                    level=ValidationLevel.ERROR,
                    category=ValidationCategory.SYNTAX,
                    message=f"Invalid temperature {temperature}. Must be between 0.0 and 1.0",
                    path=f"{path_prefix}.temperature",
                    auto_fix={"temperature": 0.1}
                ))
        
        return issues
    
    async def _validate_weaviate_config(self, weaviate_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate Weaviate configuration"""
        issues = []
        
        if not weaviate_config:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DEPENDENCY,
                message="Weaviate configuration is missing",
                path="weaviate"
            ))
            return issues
        
        # Validate URL
        url = weaviate_config.get('url', '')
        if not url:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.DEPENDENCY,
                message="Weaviate URL is required",
                path="weaviate.url",
                suggestion="Provide Weaviate server URL (e.g., http://localhost:8080)"
            ))
        
        # Validate HNSW parameters
        schema_config = weaviate_config.get('schema', {})
        for param, constraints in self.validation_rules["weaviate"]["schema_constraints"].items():
            value = schema_config.get(param)
            if value is not None:
                if value < constraints["min"] or value > constraints["max"]:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.ERROR,
                        category=ValidationCategory.SYNTAX,
                        message=f"Weaviate {param} ({value}) outside valid range [{constraints['min']}, {constraints['max']}]",
                        path=f"weaviate.schema.{param}",
                        suggestion=f"Use value between {constraints['min']} and {constraints['max']}"
                    ))
                elif value not in constraints.get("recommended", [value]):
                    recommended = constraints.get("recommended", [])
                    issues.append(ValidationIssue(
                        level=ValidationLevel.SUGGESTION,
                        category=ValidationCategory.PERFORMANCE,
                        message=f"Weaviate {param} ({value}) not in recommended range",
                        path=f"weaviate.schema.{param}",
                        suggestion=f"Consider using recommended values: {recommended}"
                    ))
        
        # Validate search parameters
        search_config = weaviate_config.get('search', {})
        hybrid_config = search_config.get('hybrid', {})
        alpha = hybrid_config.get('alpha', 0.75)
        
        if not (0.0 <= alpha <= 1.0):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SYNTAX,
                message=f"Invalid hybrid search alpha ({alpha}). Must be between 0.0 and 1.0",
                path="weaviate.search.hybrid.alpha",
                auto_fix={"alpha": 0.75}
            ))
        
        return issues
    
    async def _validate_search_config(self, search_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate search configuration"""
        issues = []
        
        # Validate query processing
        query_processing = search_config.get('query_processing', {})
        max_query_length = query_processing.get('max_query_length', 500)
        
        if max_query_length > 2000:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message=f"Very long max_query_length ({max_query_length}) may impact performance",
                path="search_config.query_processing.max_query_length",
                suggestion="Consider limiting to 1000 characters or less"
            ))
        
        # Validate search strategy
        search_strategy = search_config.get('search_strategy', {})
        final_limit = search_strategy.get('final_results_limit', 20)
        
        if final_limit > 1000:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.PERFORMANCE,
                message=f"Very high final_results_limit ({final_limit}) may be slow",
                path="search_config.search_strategy.final_results_limit",
                suggestion="Consider limiting to 100 or less for better performance"
            ))
        
        return issues
    
    async def _validate_cross_component_compatibility(self, config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate compatibility between different components"""
        issues = []
        
        # Check embedding provider and Weaviate dimension compatibility
        embedding_providers = config.get('embedding_providers', {})
        default_provider = embedding_providers.get('default_provider')
        
        if default_provider and default_provider in embedding_providers:
            provider_config = embedding_providers[default_provider]
            if isinstance(provider_config, dict):
                embedding_dims = provider_config.get('dimensions')
                
                # Future: Add validation when Weaviate schema includes dimension info
                if embedding_dims and embedding_dims not in [384, 512, 768, 1024, 1536, 3072]:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.INFO,
                        category=ValidationCategory.COMPATIBILITY,
                        message=f"Non-standard embedding dimensions ({embedding_dims})",
                        path=f"embedding_providers.{default_provider}.dimensions",
                        suggestion="Standard dimensions are 384, 512, 768, 1024, 1536, 3072"
                    ))
        
        # Check reranker and search strategy compatibility
        reranker_models = config.get('reranker_models', {})
        search_config = config.get('search_config', {})
        
        if reranker_models and search_config:
            search_strategy = search_config.get('search_strategy', {})
            if search_strategy.get('enable_reranking') and not reranker_models.get('default_reranker'):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    category=ValidationCategory.COMPATIBILITY,
                    message="Reranking enabled but no default reranker configured",
                    path="search_config.search_strategy.enable_reranking",
                    suggestion="Either disable reranking or configure a default reranker"
                ))
        
        return issues
    
    async def _analyze_performance_impact(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance impact of configuration"""
        impact = {
            "estimated_latency_ms": 0,
            "memory_usage_estimate_mb": 0,
            "cpu_intensity": "medium",
            "bottlenecks": [],
            "optimizations": []
        }
        
        # Estimate latency based on providers
        embedding_providers = config.get('embedding_providers', {})
        default_provider = embedding_providers.get('default_provider')
        
        if default_provider == "openai":
            impact["estimated_latency_ms"] += 200  # API call latency
        elif default_provider == "ollama":
            impact["estimated_latency_ms"] += 50   # Local model latency
        
        # Add reranker latency
        reranker_models = config.get('reranker_models', {})
        if reranker_models.get('default_reranker'):
            impact["estimated_latency_ms"] += 100  # Reranking overhead
        
        # Estimate memory usage
        weaviate_config = config.get('weaviate', {})
        if weaviate_config:
            # Rough estimate based on index parameters
            schema_config = weaviate_config.get('schema', {})
            max_connections = schema_config.get('max_connections', 64)
            impact["memory_usage_estimate_mb"] = max_connections * 2
        
        # Identify potential bottlenecks
        search_config = config.get('search_config', {})
        if search_config:
            search_strategy = search_config.get('search_strategy', {})
            rerank_top_k = search_strategy.get('rerank_top_k', 50)
            
            if rerank_top_k > 100:
                impact["bottlenecks"].append("Large rerank_top_k may slow reranking")
            
            if search_strategy.get('enable_reranking') and not reranker_models:
                impact["bottlenecks"].append("Reranking enabled but no rerankers configured")
        
        # Suggest optimizations
        performance_config = config.get('performance', {})
        caching = performance_config.get('caching', {})
        if not caching.get('enabled', False):
            impact["optimizations"].append("Enable caching for better performance")
        
        return impact
    
    async def _estimate_configuration_cost(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate cost of configuration"""
        cost_estimate = {
            "monthly_estimate_usd": 0.0,
            "cost_breakdown": {},
            "cost_factors": [],
            "optimization_suggestions": []
        }
        
        # Estimate embedding costs
        embedding_providers = config.get('embedding_providers', {})
        default_provider = embedding_providers.get('default_provider')
        
        if default_provider == "openai" and default_provider in embedding_providers:
            provider_config = embedding_providers[default_provider]
            cost_per_1k = provider_config.get('cost_tracking', {}).get('cost_per_1k_tokens', 0.00002)
            
            # Estimate based on assumed usage (rough estimate)
            estimated_monthly_tokens = 1000000  # 1M tokens per month
            monthly_cost = (estimated_monthly_tokens / 1000) * cost_per_1k
            
            cost_estimate["monthly_estimate_usd"] += monthly_cost
            cost_estimate["cost_breakdown"]["openai_embeddings"] = monthly_cost
            cost_estimate["cost_factors"].append(f"OpenAI embeddings: ${cost_per_1k:.6f} per 1K tokens")
        
        elif default_provider == "ollama":
            cost_estimate["cost_factors"].append("Ollama embeddings: Free (self-hosted)")
        
        # Add reranker costs (most are free/self-hosted)
        reranker_models = config.get('reranker_models', {})
        if 'cohere_reranker' in reranker_models:
            cost_estimate["cost_factors"].append("Cohere reranking: API costs apply")
        
        # Optimization suggestions
        if cost_estimate["monthly_estimate_usd"] > 50:
            cost_estimate["optimization_suggestions"].append("Consider using Ollama for embeddings to reduce costs")
        
        if not config.get('performance', {}).get('caching', {}).get('enabled'):
            cost_estimate["optimization_suggestions"].append("Enable caching to reduce API calls and costs")
        
        return cost_estimate
    
    async def _validate_security_settings(self, security_config: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate security configuration"""
        issues = []
        
        if not security_config:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SECURITY,
                message="No security configuration found",
                path="security",
                suggestion="Add security configuration to ensure safe operation"
            ))
            return issues
        
        # Validate safety mode
        safety_mode = security_config.get('safety_mode', {})
        if not safety_mode.get('enabled', True):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SECURITY,
                message="Safety mode is disabled",
                path="security.safety_mode.enabled",
                suggestion="Enable safety mode for production use",
                impact="May allow unsafe operations"
            ))
        
        if not safety_mode.get('read_only_mode', True):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                category=ValidationCategory.SECURITY,
                message="Read-only mode is disabled",
                path="security.safety_mode.read_only_mode",
                suggestion="Enable read-only mode to prevent data modifications",
                impact="May allow unintended data changes"
            ))
        
        # Validate API security
        api_security = security_config.get('api_security', {})
        if api_security.get('require_api_key', False) and not api_security.get('api_key'):
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                category=ValidationCategory.SECURITY,
                message="API key required but not provided",
                path="security.api_security.api_key",
                suggestion="Provide API key or disable API key requirement"
            ))
        
        return issues
    
    def _categorize_issues(self, new_issues: List[ValidationIssue], 
                          issues: List[ValidationIssue], 
                          warnings: List[ValidationIssue], 
                          suggestions: List[ValidationIssue]):
        """Categorize validation issues by level"""
        for issue in new_issues:
            if issue.level == ValidationLevel.ERROR:
                issues.append(issue)
            elif issue.level == ValidationLevel.WARNING:
                warnings.append(issue)
            elif issue.level in [ValidationLevel.SUGGESTION, ValidationLevel.INFO]:
                suggestions.append(issue)
    
    def _get_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum nesting depth of a configuration object"""
        if not isinstance(obj, dict):
            return current_depth
        
        if not obj:
            return current_depth
        
        max_depth = current_depth
        for value in obj.values():
            depth = self._get_max_depth(value, current_depth + 1)
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    async def auto_fix_issues(self, config: Dict[str, Any], 
                             validation_result: ValidationResult) -> Tuple[Dict[str, Any], List[str]]:
        """Apply automatic fixes to configuration issues"""
        fixed_config = config.copy()
        applied_fixes = []
        
        for issue in validation_result.issues + validation_result.warnings:
            if issue.auto_fix:
                try:
                    # Apply auto-fix
                    path_parts = issue.path.split('.')
                    self._apply_fix_to_config(fixed_config, path_parts, issue.auto_fix)
                    applied_fixes.append(f"Fixed {issue.path}: {issue.message}")
                except Exception as e:
                    logger.error(f"Failed to apply auto-fix for {issue.path}: {e}")
        
        return fixed_config, applied_fixes
    
    def _apply_fix_to_config(self, config: Dict[str, Any], path_parts: List[str], fix: Dict[str, Any]):
        """Apply a fix to a configuration at a specific path"""
        current = config
        
        # Navigate to the parent of the target
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Apply the fix
        if path_parts:
            target_key = path_parts[-1]
            for fix_key, fix_value in fix.items():
                if fix_key == target_key or len(fix) == 1:
                    current[target_key] = fix_value
                else:
                    current[fix_key] = fix_value

# Global validator instance
config_validator = AdvancedConfigValidator()

# Convenience functions
async def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """Validate configuration using global validator"""
    return await config_validator.validate_complete_configuration(config)

async def validate_and_fix_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], ValidationResult, List[str]]:
    """Validate configuration and apply auto-fixes"""
    result = await config_validator.validate_complete_configuration(config)
    fixed_config, applied_fixes = await config_validator.auto_fix_issues(config, result)
    
    # Re-validate after fixes
    if applied_fixes:
        final_result = await config_validator.validate_complete_configuration(fixed_config)
        return fixed_config, final_result, applied_fixes
    
    return config, result, applied_fixes