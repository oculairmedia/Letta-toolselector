"""
LDTS-49: Data Validation Engine
Comprehensive data validation with schema checking and quality metrics
"""

import asyncio
import json
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import logging

class ValidationRule(Enum):
    REQUIRED = "required"
    TYPE_CHECK = "type_check"
    FORMAT_CHECK = "format_check"
    RANGE_CHECK = "range_check"
    CUSTOM = "custom"

@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    quality_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataValidator:
    """Main data validation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.schemas = self._load_builtin_schemas()
    
    def _load_builtin_schemas(self) -> Dict[str, Dict]:
        """Load built-in validation schemas"""
        return {
            "query_document": {
                "required": ["query", "documents"],
                "properties": {
                    "query": {"type": "string", "min_length": 1},
                    "documents": {"type": "list", "min_items": 1},
                    "relevance_scores": {"type": "dict"}
                }
            },
            "evaluation_result": {
                "required": ["query_id", "results"],
                "properties": {
                    "query_id": {"type": "string"},
                    "results": {"type": "list"},
                    "metrics": {"type": "dict"}
                }
            }
        }
    
    async def validate_data(
        self, 
        data: Any, 
        schema_name: str,
        custom_rules: List[Dict] = None
    ) -> ValidationResult:
        """Validate data against schema"""
        
        result = ValidationResult(is_valid=True)
        
        try:
            schema = self.schemas.get(schema_name)
            if not schema:
                result.errors.append(f"Unknown schema: {schema_name}")
                result.is_valid = False
                return result
            
            # Check required fields
            for field in schema.get("required", []):
                if field not in data:
                    result.errors.append(f"Missing required field: {field}")
                    result.is_valid = False
            
            # Check field properties
            for field, rules in schema.get("properties", {}).items():
                if field in data:
                    field_result = self._validate_field(data[field], rules)
                    result.errors.extend(field_result.errors)
                    result.warnings.extend(field_result.warnings)
                    if not field_result.is_valid:
                        result.is_valid = False
            
            # Apply custom rules
            if custom_rules:
                for rule in custom_rules:
                    custom_result = self._apply_custom_rule(data, rule)
                    result.errors.extend(custom_result.errors)
                    result.warnings.extend(custom_result.warnings)
                    if not custom_result.is_valid:
                        result.is_valid = False
            
            # Calculate quality score
            result.quality_score = self._calculate_quality_score(data, result)
            
        except Exception as e:
            result.errors.append(f"Validation error: {str(e)}")
            result.is_valid = False
        
        return result
    
    def _validate_field(self, value: Any, rules: Dict) -> ValidationResult:
        """Validate individual field"""
        result = ValidationResult(is_valid=True)
        
        # Type checking
        expected_type = rules.get("type")
        if expected_type and not self._check_type(value, expected_type):
            result.errors.append(f"Invalid type: expected {expected_type}")
            result.is_valid = False
            return result
        
        # String validations
        if expected_type == "string" and isinstance(value, str):
            min_length = rules.get("min_length", 0)
            max_length = rules.get("max_length", float('inf'))
            
            if len(value) < min_length:
                result.errors.append(f"String too short: minimum {min_length} characters")
                result.is_valid = False
            
            if len(value) > max_length:
                result.errors.append(f"String too long: maximum {max_length} characters")
                result.is_valid = False
            
            # Format checking
            pattern = rules.get("pattern")
            if pattern and not re.match(pattern, value):
                result.errors.append(f"String does not match pattern: {pattern}")
                result.is_valid = False
        
        # List validations
        if expected_type == "list" and isinstance(value, list):
            min_items = rules.get("min_items", 0)
            max_items = rules.get("max_items", float('inf'))
            
            if len(value) < min_items:
                result.errors.append(f"List too short: minimum {min_items} items")
                result.is_valid = False
            
            if len(value) > max_items:
                result.errors.append(f"List too long: maximum {max_items} items")
                result.is_valid = False
        
        # Numeric validations
        if expected_type in ["int", "float"] and isinstance(value, (int, float)):
            min_value = rules.get("min_value")
            max_value = rules.get("max_value")
            
            if min_value is not None and value < min_value:
                result.errors.append(f"Value too small: minimum {min_value}")
                result.is_valid = False
            
            if max_value is not None and value > max_value:
                result.errors.append(f"Value too large: maximum {max_value}")
                result.is_valid = False
        
        return result
    
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type"""
        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "int":
            return isinstance(value, int)
        elif expected_type == "float":
            return isinstance(value, (int, float))
        elif expected_type == "bool":
            return isinstance(value, bool)
        elif expected_type == "list":
            return isinstance(value, list)
        elif expected_type == "dict":
            return isinstance(value, dict)
        else:
            return True
    
    def _apply_custom_rule(self, data: Dict, rule: Dict) -> ValidationResult:
        """Apply custom validation rule"""
        result = ValidationResult(is_valid=True)
        
        try:
            rule_type = rule.get("type")
            
            if rule_type == "query_quality":
                # Check query quality metrics
                query = data.get("query", "")
                if len(query.split()) < 3:
                    result.warnings.append("Query is very short - may not be specific enough")
                
                if not any(c.isalpha() for c in query):
                    result.errors.append("Query contains no alphabetic characters")
                    result.is_valid = False
            
            elif rule_type == "relevance_consistency":
                # Check relevance score consistency
                relevance_scores = data.get("relevance_scores", {})
                if relevance_scores:
                    scores = list(relevance_scores.values())
                    if max(scores) - min(scores) > 3.0:
                        result.warnings.append("Large variance in relevance scores")
            
        except Exception as e:
            result.errors.append(f"Custom rule error: {str(e)}")
            result.is_valid = False
        
        return result
    
    def _calculate_quality_score(self, data: Dict, validation_result: ValidationResult) -> float:
        """Calculate data quality score"""
        
        base_score = 1.0
        
        # Penalize for errors
        error_penalty = len(validation_result.errors) * 0.2
        base_score -= min(error_penalty, 0.8)
        
        # Small penalty for warnings
        warning_penalty = len(validation_result.warnings) * 0.05
        base_score -= min(warning_penalty, 0.2)
        
        # Bonus for completeness
        if isinstance(data, dict):
            expected_fields = ["query", "documents", "relevance_scores"]
            completeness = sum(1 for field in expected_fields if field in data) / len(expected_fields)
            base_score += (completeness - 0.5) * 0.2
        
        return max(0.0, min(1.0, base_score))

# Global instance
data_validator = DataValidator()