"""
Unit tests for configuration validation system
Tests the simple_config_validation module without requiring external services
"""

import pytest
from simple_config_validation import (
    SimpleConfigurationValidator,
    ValidationLevel,
    ValidationResult,
    ConfigValidationResponse,
    validate_configuration
)


@pytest.mark.unit
class TestValidationResult:
    """Test ValidationResult class"""
    
    def test_create_validation_result(self):
        """Test creating a validation result"""
        result = ValidationResult(
            field="test.field",
            level=ValidationLevel.ERROR,
            message="Test error message",
            suggestion="Fix it this way"
        )
        
        assert result.field == "test.field"
        assert result.level == ValidationLevel.ERROR
        assert result.message == "Test error message"
        assert result.suggestion == "Fix it this way"
    
    def test_validation_result_to_dict(self):
        """Test converting validation result to dictionary"""
        result = ValidationResult(
            field="test.field",
            level=ValidationLevel.WARNING,
            message="Test warning",
            valid_range={"min": 0, "max": 100}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict["field"] == "test.field"
        assert result_dict["level"] == "warning"
        assert result_dict["message"] == "Test warning"
        assert result_dict["valid_range"] == {"min": 0, "max": 100}


@pytest.mark.unit
class TestConfigValidationResponse:
    """Test ConfigValidationResponse class"""
    
    def test_create_validation_response(self):
        """Test creating a validation response"""
        errors = [
            ValidationResult("field1", ValidationLevel.ERROR, "Error 1")
        ]
        warnings = [
            ValidationResult("field2", ValidationLevel.WARNING, "Warning 1")
        ]
        suggestions = [
            ValidationResult("field3", ValidationLevel.INFO, "Suggestion 1")
        ]
        
        response = ConfigValidationResponse(
            valid=False,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
        
        assert response.valid is False
        assert len(response.errors) == 1
        assert len(response.warnings) == 1
        assert len(response.suggestions) == 1
    
    def test_validation_response_to_dict(self):
        """Test converting validation response to dictionary"""
        response = ConfigValidationResponse(
            valid=True,
            errors=[],
            warnings=[],
            suggestions=[],
            cost_estimate={"daily_estimate_usd": 5.0},
            performance_impact={"latency": "low"}
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["valid"] is True
        assert isinstance(response_dict["errors"], list)
        assert response_dict["cost_estimate"]["daily_estimate_usd"] == 5.0
        assert response_dict["performance_impact"]["latency"] == "low"


@pytest.mark.unit
class TestSimpleConfigurationValidator:
    """Test SimpleConfigurationValidator class"""
    
    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return SimpleConfigurationValidator()
    
    def test_validator_initialization(self, validator):
        """Test validator initialization"""
        assert validator is not None
        assert "embedding_providers" in validator.validation_rules
        assert "reranker_types" in validator.validation_rules
        assert "distance_metrics" in validator.validation_rules
    
    def test_validate_embedding_config_valid(self, validator):
        """Test validating valid embedding configuration"""
        embedding_config = {
            "provider": "ollama",
            "dimension": 1536,
            "max_tokens": 8192
        }
        
        results = validator._validate_embedding_config(embedding_config)
        
        # Should have no errors
        errors = [r for r in results if r.level in [ValidationLevel.ERROR, ValidationLevel.BLOCKING]]
        assert len(errors) == 0
    
    def test_validate_embedding_config_invalid_provider(self, validator):
        """Test validating embedding config with invalid provider"""
        embedding_config = {
            "provider": "invalid_provider",
            "dimension": 1536
        }
        
        results = validator._validate_embedding_config(embedding_config)
        
        # Should have error about invalid provider
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0
        assert any("provider" in r.field for r in errors)
    
    def test_validate_embedding_config_invalid_dimension(self, validator):
        """Test validating embedding config with invalid dimension"""
        embedding_config = {
            "provider": "openai",
            "dimension": -1  # Invalid
        }
        
        results = validator._validate_embedding_config(embedding_config)
        
        # Should have error about invalid dimension
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0
        assert any("dimension" in r.field for r in errors)
    
    def test_validate_embedding_config_missing_provider(self, validator):
        """Test validating embedding config without provider"""
        embedding_config = {
            "dimension": 1536
        }
        
        results = validator._validate_embedding_config(embedding_config)
        
        # Should have error about missing provider
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0
        assert any("provider" in r.field for r in errors)
    
    def test_validate_weaviate_config_valid(self, validator):
        """Test validating valid Weaviate configuration"""
        weaviate_config = {
            "hybrid": {
                "alpha": 0.75
            }
        }
        
        results = validator._validate_weaviate_config(weaviate_config)
        
        # Should have no errors
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0
    
    def test_validate_weaviate_config_invalid_alpha(self, validator):
        """Test validating Weaviate config with invalid alpha"""
        weaviate_config = {
            "hybrid": {
                "alpha": 5.0  # Invalid: must be 0-1
            }
        }
        
        results = validator._validate_weaviate_config(weaviate_config)
        
        # Should have error about alpha
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0
        assert any("alpha" in r.field for r in errors)
    
    def test_validate_weaviate_config_low_alpha_warning(self, validator):
        """Test validating Weaviate config with very low alpha"""
        weaviate_config = {
            "hybrid": {
                "alpha": 0.05  # Valid but low
            }
        }
        
        results = validator._validate_weaviate_config(weaviate_config)
        
        # Should have warning about low alpha
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        assert len(warnings) > 0
    
    def test_validate_reranker_config_valid(self, validator):
        """Test validating valid reranker configuration"""
        reranker_config = {
            "enabled": True,
            "type": "cross-encoder",
            "scoring": {
                "top_k": 10
            }
        }
        
        results = validator._validate_reranker_config(reranker_config)
        
        # Should have no errors
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) == 0
    
    def test_validate_reranker_config_disabled(self, validator):
        """Test validating disabled reranker configuration"""
        reranker_config = {
            "enabled": False,
            "type": "invalid_type"  # Should be ignored when disabled
        }
        
        results = validator._validate_reranker_config(reranker_config)
        
        # Should have no errors since reranker is disabled
        assert len(results) == 0
    
    def test_validate_reranker_config_invalid_type(self, validator):
        """Test validating reranker config with invalid type"""
        reranker_config = {
            "enabled": True,
            "type": "invalid_type"
        }
        
        results = validator._validate_reranker_config(reranker_config)
        
        # Should have error about invalid type
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0
        assert any("type" in r.field for r in errors)
    
    def test_validate_reranker_config_invalid_top_k(self, validator):
        """Test validating reranker config with invalid top_k"""
        reranker_config = {
            "enabled": True,
            "type": "cross-encoder",
            "scoring": {
                "top_k": 1000  # Invalid: max is 100
            }
        }
        
        results = validator._validate_reranker_config(reranker_config)
        
        # Should have error about top_k
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        assert len(errors) > 0
        assert any("top_k" in r.field for r in errors)
    
    def test_validate_complete_config_valid(self, validator, valid_config):
        """Test validating complete valid configuration"""
        response = validator.validate_complete_config(valid_config)
        
        assert isinstance(response, ConfigValidationResponse)
        assert response.valid is True
        assert len(response.errors) == 0
    
    def test_validate_complete_config_invalid(self, validator, invalid_config):
        """Test validating complete invalid configuration"""
        response = validator.validate_complete_config(invalid_config)
        
        assert isinstance(response, ConfigValidationResponse)
        assert response.valid is False
        assert len(response.errors) > 0
    
    def test_estimate_configuration_cost_openai(self, validator):
        """Test cost estimation for OpenAI embedding provider"""
        config = {
            "search": {
                "embedding": {
                    "provider": "openai",
                    "model": "text-embedding-3-small",
                    "max_tokens": 8192
                }
            }
        }
        
        cost_estimate = validator._estimate_configuration_cost(config)
        
        assert "daily_estimate_usd" in cost_estimate
        assert "per_query_estimate_usd" in cost_estimate
        assert "breakdown" in cost_estimate
        assert cost_estimate["per_query_estimate_usd"] > 0
    
    def test_estimate_configuration_cost_ollama(self, validator):
        """Test cost estimation for Ollama embedding provider"""
        config = {
            "search": {
                "embedding": {
                    "provider": "ollama",
                    "model": "test-model"
                }
            }
        }
        
        cost_estimate = validator._estimate_configuration_cost(config)
        
        # Ollama should have zero cost
        assert cost_estimate["per_query_estimate_usd"] == 0.0
    
    def test_assess_performance_impact_low(self, validator):
        """Test performance impact assessment for low-impact configuration"""
        config = {
            "search": {
                "embedding": {
                    "dimension": 1536
                }
            },
            "reranker": {
                "enabled": False
            }
        }
        
        impact = validator._assess_performance_impact(config)
        
        assert impact["latency"] == "low"
        assert impact["memory"] == "low"
        assert impact["cpu"] == "low"
    
    def test_assess_performance_impact_high(self, validator):
        """Test performance impact assessment for high-impact configuration"""
        config = {
            "search": {
                "embedding": {
                    "dimension": 3000  # High dimension
                }
            },
            "reranker": {
                "enabled": True,
                "type": "cross-encoder",
                "scoring": {
                    "top_k": 50  # High top_k
                }
            }
        }
        
        impact = validator._assess_performance_impact(config)
        
        # Should have high impact in some areas
        assert impact["latency"] == "high" or impact["cpu"] == "high"


@pytest.mark.unit
def test_validate_configuration_function(valid_config):
    """Test the global validate_configuration function"""
    result = validate_configuration(valid_config)
    
    assert isinstance(result, ConfigValidationResponse)
    assert result.valid is True


@pytest.mark.unit
def test_validate_configuration_function_invalid(invalid_config):
    """Test the global validate_configuration function with invalid config"""
    result = validate_configuration(invalid_config)
    
    assert isinstance(result, ConfigValidationResponse)
    assert result.valid is False
    assert len(result.errors) > 0
