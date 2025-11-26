"""
Unit tests for bm25_vector_overrides.py

Tests for the BM25 and vector distance parameter override system including:
- Parameter validation
- Parameter set creation and management
- Override application
- Schema validation
- Statistics tracking
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "lettaaugment-source"))

from bm25_vector_overrides import (
    VectorDistanceMetric,
    BM25ParameterType,
    VectorParameterType,
    BM25Override,
    VectorDistanceOverride,
    SearchParameterSet,
    BM25VectorOverrideService
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def override_service():
    """Create fresh override service instance without default sets"""
    service = BM25VectorOverrideService()
    # Clear default parameter sets for isolated testing
    service.parameter_sets.clear()
    service.statistics["total_parameter_sets"] = 0
    return service


@pytest.fixture
def sample_bm25_params():
    """Sample BM25 parameters for testing"""
    return [
        {
            "parameter_type": "k1",
            "value": 1.5,
            "description": "Custom k1 value"
        },
        {
            "parameter_type": "b",
            "value": 0.8,
            "description": "Custom b value"
        }
    ]


@pytest.fixture
def sample_vector_params():
    """Sample vector parameters for testing"""
    return [
        {
            "parameter_type": "distance_metric",
            "value": "cosine",
            "description": "Cosine similarity"
        },
        {
            "parameter_type": "certainty",
            "value": 0.7,
            "description": "Minimum certainty threshold"
        }
    ]


@pytest.fixture
def base_weaviate_config():
    """Base Weaviate configuration for testing overrides"""
    return {
        "collection": "Tool",
        "limit": 10,
        "offset": 0
    }


# ============================================================================
# Enum Tests
# ============================================================================

class TestEnums:
    """Test enum definitions"""
    
    def test_vector_distance_metrics(self):
        """Test all supported vector distance metrics"""
        metrics = [m.value for m in VectorDistanceMetric]
        
        assert "cosine" in metrics
        assert "dot" in metrics
        assert "l2" in metrics
        assert "euclidean" in metrics
        assert "hamming" in metrics
        assert "manhattan" in metrics
    
    def test_bm25_parameter_types(self):
        """Test BM25 parameter types"""
        params = [p.value for p in BM25ParameterType]
        
        assert "k1" in params
        assert "b" in params
        assert "query_properties" in params
        assert "boost" in params
    
    def test_vector_parameter_types(self):
        """Test vector parameter types"""
        params = [p.value for p in VectorParameterType]
        
        assert "distance_metric" in params
        assert "certainty" in params
        assert "distance_threshold" in params
        assert "autocut" in params


# ============================================================================
# Dataclass Tests
# ============================================================================

class TestBM25Override:
    """Test BM25Override dataclass"""
    
    def test_bm25_override_creation(self):
        """Test creating BM25 override"""
        override = BM25Override(
            parameter_type=BM25ParameterType.K1,
            value=1.5,
            description="Custom k1",
            validation_range={"min": 0.0, "max": 3.0},
            enabled=True
        )
        
        assert override.parameter_type == BM25ParameterType.K1
        assert override.value == 1.5
        assert override.description == "Custom k1"
        assert override.enabled is True
    
    def test_bm25_override_defaults(self):
        """Test BM25 override with defaults"""
        override = BM25Override(
            parameter_type=BM25ParameterType.B,
            value=0.75,
            description="Test"
        )
        
        assert override.validation_range is None
        assert override.enabled is True


class TestVectorDistanceOverride:
    """Test VectorDistanceOverride dataclass"""
    
    def test_vector_override_creation(self):
        """Test creating vector override"""
        override = VectorDistanceOverride(
            parameter_type=VectorParameterType.DISTANCE_METRIC,
            value="cosine",
            description="Cosine metric",
            validation_range={"values": ["cosine", "dot", "l2"]},
            enabled=True
        )
        
        assert override.parameter_type == VectorParameterType.DISTANCE_METRIC
        assert override.value == "cosine"
        assert override.description == "Cosine metric"
        assert override.enabled is True


class TestSearchParameterSet:
    """Test SearchParameterSet dataclass"""
    
    def test_parameter_set_creation(self):
        """Test creating search parameter set"""
        bm25_overrides = [
            BM25Override(BM25ParameterType.K1, 1.5, "Test k1")
        ]
        vector_overrides = [
            VectorDistanceOverride(VectorParameterType.CERTAINTY, 0.7, "Test certainty")
        ]
        
        param_set = SearchParameterSet(
            parameter_set_id="test_001",
            name="Test Set",
            description="Test parameter set",
            bm25_overrides=bm25_overrides,
            vector_overrides=vector_overrides,
            hybrid_alpha=0.75,
            fusion_type="relative_score",
            active=True
        )
        
        assert param_set.parameter_set_id == "test_001"
        assert param_set.name == "Test Set"
        assert len(param_set.bm25_overrides) == 1
        assert len(param_set.vector_overrides) == 1
        assert param_set.hybrid_alpha == 0.75
        assert param_set.active is True


# ============================================================================
# Service Initialization Tests
# ============================================================================

class TestServiceInitialization:
    """Test service initialization"""
    
    def test_service_creation(self, override_service):
        """Test creating override service"""
        assert override_service.parameter_sets == {}
        assert "bm25" in override_service.active_overrides
        assert "vector" in override_service.active_overrides
        assert "hybrid" in override_service.active_overrides
    
    def test_bm25_schemas_initialized(self, override_service):
        """Test BM25 schemas are initialized"""
        assert BM25ParameterType.K1 in override_service.bm25_schemas
        assert BM25ParameterType.B in override_service.bm25_schemas
        assert BM25ParameterType.QUERY_PROPERTIES in override_service.bm25_schemas
        assert BM25ParameterType.BOOST in override_service.bm25_schemas
        
        # Check k1 schema details
        k1_schema = override_service.bm25_schemas[BM25ParameterType.K1]
        assert k1_schema["type"] == "float"
        assert k1_schema["min"] == 0.0
        assert k1_schema["max"] == 3.0
        assert k1_schema["default"] == 1.2
    
    def test_vector_schemas_initialized(self, override_service):
        """Test vector schemas are initialized"""
        assert VectorParameterType.DISTANCE_METRIC in override_service.vector_schemas
        assert VectorParameterType.CERTAINTY in override_service.vector_schemas
        assert VectorParameterType.DISTANCE_THRESHOLD in override_service.vector_schemas
        assert VectorParameterType.AUTOCUT in override_service.vector_schemas
        
        # Check distance metric schema details
        metric_schema = override_service.vector_schemas[VectorParameterType.DISTANCE_METRIC]
        assert metric_schema["type"] == "enum"
        assert "cosine" in metric_schema["values"]
        assert metric_schema["default"] == "cosine"
    
    def test_statistics_initialized(self, override_service):
        """Test statistics are initialized"""
        stats = override_service.statistics
        
        assert stats["total_parameter_sets"] == 0
        assert stats["bm25_overrides_applied"] == 0
        assert stats["vector_overrides_applied"] == 0
        assert stats["failed_applications"] == 0
        assert isinstance(stats["most_used_parameters"], dict)
        assert isinstance(stats["distance_metric_usage"], dict)


# ============================================================================
# Parameter Validation Tests
# ============================================================================

class TestParameterValidation:
    """Test parameter value validation"""
    
    def test_validate_float_success(self, override_service):
        """Test validating valid float value"""
        schema = {"type": "float", "min": 0.0, "max": 3.0}
        errors = override_service._validate_parameter_value(1.5, schema)
        assert errors == []
    
    def test_validate_float_type_error(self, override_service):
        """Test validating invalid float type"""
        schema = {"type": "float", "min": 0.0, "max": 3.0}
        errors = override_service._validate_parameter_value("not_a_float", schema)
        assert len(errors) == 1
        assert "Expected float" in errors[0]
    
    def test_validate_float_min_error(self, override_service):
        """Test validating float below minimum"""
        schema = {"type": "float", "min": 0.0, "max": 3.0}
        errors = override_service._validate_parameter_value(-0.5, schema)
        assert len(errors) == 1
        assert "below minimum" in errors[0]
    
    def test_validate_float_max_error(self, override_service):
        """Test validating float above maximum"""
        schema = {"type": "float", "min": 0.0, "max": 3.0}
        errors = override_service._validate_parameter_value(5.0, schema)
        assert len(errors) == 1
        assert "above maximum" in errors[0]
    
    def test_validate_int_success(self, override_service):
        """Test validating valid int value"""
        schema = {"type": "int", "min": 1, "max": 10}
        errors = override_service._validate_parameter_value(5, schema)
        assert errors == []
    
    def test_validate_int_type_error(self, override_service):
        """Test validating invalid int type"""
        schema = {"type": "int", "min": 1, "max": 10}
        errors = override_service._validate_parameter_value(5.5, schema)
        assert len(errors) == 1
        assert "Expected int" in errors[0]
    
    def test_validate_enum_success(self, override_service):
        """Test validating valid enum value"""
        schema = {"type": "enum", "values": ["cosine", "dot", "l2"]}
        errors = override_service._validate_parameter_value("cosine", schema)
        assert errors == []
    
    def test_validate_enum_error(self, override_service):
        """Test validating invalid enum value"""
        schema = {"type": "enum", "values": ["cosine", "dot", "l2"]}
        errors = override_service._validate_parameter_value("invalid", schema)
        assert len(errors) == 1
        assert "not in allowed values" in errors[0]
    
    def test_validate_list_success(self, override_service):
        """Test validating valid list value"""
        schema = {"type": "list"}
        errors = override_service._validate_parameter_value(["a", "b", "c"], schema)
        assert errors == []
    
    def test_validate_list_error(self, override_service):
        """Test validating invalid list type"""
        schema = {"type": "list"}
        errors = override_service._validate_parameter_value("not_a_list", schema)
        assert len(errors) == 1
        assert "Expected list" in errors[0]
    
    def test_validate_dict_success(self, override_service):
        """Test validating valid dict value"""
        schema = {"type": "dict"}
        errors = override_service._validate_parameter_value({"key": "value"}, schema)
        assert errors == []
    
    def test_validate_dict_error(self, override_service):
        """Test validating invalid dict type"""
        schema = {"type": "dict"}
        errors = override_service._validate_parameter_value(["not", "dict"], schema)
        assert len(errors) == 1
        assert "Expected dict" in errors[0]


# ============================================================================
# BM25 Override Creation Tests
# ============================================================================

class TestBM25OverrideCreation:
    """Test creating BM25 overrides"""
    
    def test_create_valid_bm25_override(self, override_service):
        """Test creating valid BM25 override"""
        param_data = {
            "parameter_type": "k1",
            "value": 1.5,
            "description": "Custom k1"
        }
        
        override = override_service._create_bm25_override(param_data)
        
        assert override.parameter_type == BM25ParameterType.K1
        assert override.value == 1.5
        assert override.description == "Custom k1"
        assert override.enabled is True
    
    def test_create_bm25_override_missing_type(self, override_service):
        """Test creating BM25 override without parameter_type"""
        param_data = {"value": 1.5}
        
        with pytest.raises(ValueError, match="Missing parameter_type"):
            override_service._create_bm25_override(param_data)
    
    def test_create_bm25_override_invalid_type(self, override_service):
        """Test creating BM25 override with invalid parameter_type"""
        param_data = {
            "parameter_type": "invalid_param",
            "value": 1.5
        }
        
        with pytest.raises(ValueError, match="Invalid BM25 parameter type"):
            override_service._create_bm25_override(param_data)
    
    def test_create_bm25_override_missing_value(self, override_service):
        """Test creating BM25 override without value"""
        param_data = {"parameter_type": "k1"}
        
        with pytest.raises(ValueError, match="Missing value"):
            override_service._create_bm25_override(param_data)
    
    def test_create_bm25_override_invalid_value(self, override_service):
        """Test creating BM25 override with invalid value"""
        param_data = {
            "parameter_type": "k1",
            "value": 10.0  # Above max of 3.0
        }
        
        with pytest.raises(ValueError, match="Invalid value"):
            override_service._create_bm25_override(param_data)


# ============================================================================
# Vector Override Creation Tests
# ============================================================================

class TestVectorOverrideCreation:
    """Test creating vector overrides"""
    
    def test_create_valid_vector_override(self, override_service):
        """Test creating valid vector override"""
        param_data = {
            "parameter_type": "distance_metric",
            "value": "cosine",
            "description": "Cosine similarity"
        }
        
        override = override_service._create_vector_override(param_data)
        
        assert override.parameter_type == VectorParameterType.DISTANCE_METRIC
        assert override.value == "cosine"
        assert override.description == "Cosine similarity"
        assert override.enabled is True
    
    def test_create_vector_override_missing_type(self, override_service):
        """Test creating vector override without parameter_type"""
        param_data = {"value": "cosine"}
        
        with pytest.raises(ValueError, match="Missing parameter_type"):
            override_service._create_vector_override(param_data)
    
    def test_create_vector_override_invalid_type(self, override_service):
        """Test creating vector override with invalid parameter_type"""
        param_data = {
            "parameter_type": "invalid_param",
            "value": "cosine"
        }
        
        with pytest.raises(ValueError, match="Invalid vector parameter type"):
            override_service._create_vector_override(param_data)
    
    def test_create_vector_override_invalid_value(self, override_service):
        """Test creating vector override with invalid value"""
        param_data = {
            "parameter_type": "distance_metric",
            "value": "invalid_metric"
        }
        
        with pytest.raises(ValueError, match="Invalid value"):
            override_service._create_vector_override(param_data)


# ============================================================================
# Parameter Set Creation Tests
# ============================================================================

class TestParameterSetCreation:
    """Test creating parameter sets"""
    
    def test_create_parameter_set_basic(self, override_service, sample_bm25_params):
        """Test creating basic parameter set"""
        param_id = override_service.create_parameter_set(
            name="Test Set",
            description="Test parameter set",
            bm25_params=sample_bm25_params
        )
        
        assert param_id in override_service.parameter_sets
        param_set = override_service.parameter_sets[param_id]
        assert param_set.name == "Test Set"
        assert len(param_set.bm25_overrides) == 2
    
    def test_create_parameter_set_with_vector(self, override_service, sample_vector_params):
        """Test creating parameter set with vector params"""
        param_id = override_service.create_parameter_set(
            name="Vector Test",
            description="Test with vector params",
            vector_params=sample_vector_params
        )
        
        param_set = override_service.parameter_sets[param_id]
        assert len(param_set.vector_overrides) == 2
    
    def test_create_parameter_set_with_hybrid(self, override_service, sample_bm25_params, sample_vector_params):
        """Test creating parameter set with hybrid config"""
        param_id = override_service.create_parameter_set(
            name="Hybrid Test",
            description="Test with hybrid params",
            bm25_params=sample_bm25_params,
            vector_params=sample_vector_params,
            hybrid_alpha=0.8,
            fusion_type="ranked_fusion"
        )
        
        param_set = override_service.parameter_sets[param_id]
        assert param_set.hybrid_alpha == 0.8
        assert param_set.fusion_type == "ranked_fusion"
    
    def test_create_parameter_set_invalid_alpha(self, override_service):
        """Test creating parameter set with invalid hybrid alpha"""
        with pytest.raises(ValueError, match="Hybrid alpha must be between"):
            override_service.create_parameter_set(
                name="Invalid Alpha",
                description="Test",
                hybrid_alpha=1.5
            )
    
    def test_create_parameter_set_with_metadata(self, override_service):
        """Test creating parameter set with metadata"""
        metadata = {"author": "test", "version": "1.0"}
        
        param_id = override_service.create_parameter_set(
            name="Meta Test",
            description="Test with metadata",
            metadata=metadata
        )
        
        param_set = override_service.parameter_sets[param_id]
        assert param_set.metadata == metadata
    
    def test_create_parameter_set_updates_statistics(self, override_service):
        """Test that creating parameter sets updates statistics"""
        initial_count = override_service.statistics["total_parameter_sets"]
        
        override_service.create_parameter_set(
            name="Stats Test",
            description="Test statistics update"
        )
        
        assert override_service.statistics["total_parameter_sets"] == initial_count + 1


# ============================================================================
# Parameter Set Application Tests
# ============================================================================

class TestParameterSetApplication:
    """Test applying parameter sets to configurations"""
    
    def test_apply_bm25_overrides(self, override_service, sample_bm25_params, base_weaviate_config):
        """Test applying BM25 overrides to config"""
        param_id = override_service.create_parameter_set(
            name="BM25 Test",
            description="Test BM25 application",
            bm25_params=sample_bm25_params
        )
        
        modified_config = override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        assert "bm25" in modified_config
        assert modified_config["bm25"]["k1"] == 1.5
        assert modified_config["bm25"]["b"] == 0.8
    
    def test_apply_vector_overrides(self, override_service, sample_vector_params, base_weaviate_config):
        """Test applying vector overrides to config"""
        param_id = override_service.create_parameter_set(
            name="Vector Test",
            description="Test vector application",
            vector_params=sample_vector_params
        )
        
        modified_config = override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        assert "vector" in modified_config
        assert modified_config["vector"]["distance_metric"] == "cosine"
        assert modified_config["vector"]["certainty"] == 0.7
    
    def test_apply_hybrid_config(self, override_service, base_weaviate_config):
        """Test applying hybrid configuration"""
        param_id = override_service.create_parameter_set(
            name="Hybrid Test",
            description="Test hybrid application",
            hybrid_alpha=0.8,
            fusion_type="ranked_fusion"
        )
        
        modified_config = override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        assert "hybrid" in modified_config
        assert modified_config["hybrid"]["alpha"] == 0.8
        assert modified_config["hybrid"]["fusion_type"] == "ranked_fusion"
    
    def test_apply_doesnt_modify_original(self, override_service, sample_bm25_params, base_weaviate_config):
        """Test that applying doesn't modify original config"""
        param_id = override_service.create_parameter_set(
            name="Immutable Test",
            description="Test immutability",
            bm25_params=sample_bm25_params
        )
        
        original_config = base_weaviate_config.copy()
        modified_config = override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        assert base_weaviate_config == original_config
        assert "bm25" not in base_weaviate_config
        assert "bm25" in modified_config
    
    def test_apply_invalid_parameter_set(self, override_service, base_weaviate_config):
        """Test applying non-existent parameter set"""
        with pytest.raises(ValueError, match="Parameter set not found"):
            override_service.apply_parameter_set("invalid_id", base_weaviate_config)
    
    def test_apply_inactive_parameter_set(self, override_service, sample_bm25_params, base_weaviate_config):
        """Test applying inactive parameter set"""
        param_id = override_service.create_parameter_set(
            name="Inactive Test",
            description="Test inactive application",
            bm25_params=sample_bm25_params
        )
        
        # Deactivate the parameter set
        override_service.parameter_sets[param_id].active = False
        
        modified_config = override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        # Should return config unchanged
        assert modified_config == base_weaviate_config
    
    def test_apply_updates_statistics(self, override_service, sample_bm25_params, base_weaviate_config):
        """Test that applying updates statistics"""
        param_id = override_service.create_parameter_set(
            name="Stats Test",
            description="Test statistics update",
            bm25_params=sample_bm25_params
        )
        
        initial_bm25_count = override_service.statistics["bm25_overrides_applied"]
        
        override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        assert override_service.statistics["bm25_overrides_applied"] > initial_bm25_count


# ============================================================================
# Parameter Set Management Tests
# ============================================================================

class TestParameterSetManagement:
    """Test managing parameter sets"""
    
    def test_get_parameter_set(self, override_service, sample_bm25_params):
        """Test getting parameter set by ID"""
        param_id = override_service.create_parameter_set(
            name="Get Test",
            description="Test getting",
            bm25_params=sample_bm25_params
        )
        
        param_set = override_service.get_parameter_set(param_id)
        
        assert param_set is not None
        assert param_set.name == "Get Test"
    
    def test_get_nonexistent_parameter_set(self, override_service):
        """Test getting non-existent parameter set"""
        param_set = override_service.get_parameter_set("nonexistent")
        assert param_set is None
    
    def test_list_parameter_sets(self, override_service, sample_bm25_params):
        """Test listing parameter sets"""
        # Create a few parameter sets
        override_service.create_parameter_set(
            name="Set 1",
            description="First set",
            bm25_params=sample_bm25_params
        )
        override_service.create_parameter_set(
            name="Set 2",
            description="Second set",
            bm25_params=sample_bm25_params
        )
        
        sets = override_service.list_parameter_sets()
        
        assert len(sets) >= 2
        assert all("parameter_set_id" in s for s in sets)
        assert all("name" in s for s in sets)
    
    def test_list_parameter_sets_active_only(self, override_service, sample_bm25_params):
        """Test listing only active parameter sets"""
        param_id = override_service.create_parameter_set(
            name="Active Test",
            description="Test active filtering",
            bm25_params=sample_bm25_params
        )
        
        # Deactivate one set
        override_service.parameter_sets[param_id].active = False
        
        active_sets = override_service.list_parameter_sets(active_only=True)
        inactive_param_ids = [s["parameter_set_id"] for s in active_sets]
        
        assert param_id not in inactive_param_ids
    
    def test_delete_parameter_set(self, override_service, sample_bm25_params):
        """Test deleting parameter set"""
        param_id = override_service.create_parameter_set(
            name="Delete Test",
            description="Test deletion",
            bm25_params=sample_bm25_params
        )
        
        result = override_service.delete_parameter_set(param_id)
        
        assert result is True
        assert param_id not in override_service.parameter_sets
    
    def test_delete_nonexistent_parameter_set(self, override_service):
        """Test deleting non-existent parameter set"""
        result = override_service.delete_parameter_set("nonexistent")
        assert result is False


# ============================================================================
# Schema and Info Tests
# ============================================================================

class TestSchemaAndInfo:
    """Test schema and information methods"""
    
    def test_get_supported_distance_metrics(self, override_service):
        """Test getting supported distance metrics"""
        metrics = override_service.get_supported_distance_metrics()
        
        assert "cosine" in metrics
        assert "dot" in metrics
        assert "l2" in metrics
        assert "euclidean" in metrics
        assert len(metrics) == 6  # All VectorDistanceMetric values
    
    def test_get_bm25_parameter_schema(self, override_service):
        """Test getting BM25 parameter schema"""
        schema = override_service.get_bm25_parameter_schema()
        
        assert "k1" in schema
        assert "b" in schema
        assert "query_properties" in schema
        assert "boost" in schema
        
        # Check schema structure
        assert schema["k1"]["type"] == "float"
        assert schema["k1"]["min"] == 0.0
        assert schema["k1"]["max"] == 3.0
    
    def test_get_vector_parameter_schema(self, override_service):
        """Test getting vector parameter schema"""
        schema = override_service.get_vector_parameter_schema()
        
        assert "distance_metric" in schema
        assert "certainty" in schema
        assert "distance_threshold" in schema
        assert "autocut" in schema
        
        # Check schema structure
        assert schema["distance_metric"]["type"] == "enum"
        assert "cosine" in schema["distance_metric"]["values"]
    
    def test_get_statistics(self, override_service, sample_bm25_params, base_weaviate_config):
        """Test getting service statistics"""
        # Create and apply a parameter set
        param_id = override_service.create_parameter_set(
            name="Stats Test",
            description="Test statistics",
            bm25_params=sample_bm25_params
        )
        override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        stats = override_service.get_statistics()
        
        assert "total_parameter_sets" in stats
        assert "bm25_overrides_applied" in stats
        assert "vector_overrides_applied" in stats
        assert "supported_distance_metrics" in stats
        assert "active_parameter_sets" in stats
        
        assert stats["total_parameter_sets"] > 0
        assert stats["bm25_overrides_applied"] > 0


# ============================================================================
# Statistics Tracking Tests
# ============================================================================

class TestStatisticsTracking:
    """Test statistics tracking functionality"""
    
    def test_track_bm25_usage(self, override_service, sample_bm25_params, base_weaviate_config):
        """Test tracking BM25 parameter usage"""
        param_id = override_service.create_parameter_set(
            name="Usage Test",
            description="Test usage tracking",
            bm25_params=sample_bm25_params
        )
        
        # Apply multiple times
        override_service.apply_parameter_set(param_id, base_weaviate_config)
        override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        stats = override_service.statistics["most_used_parameters"]
        
        assert "k1" in stats
        assert "b" in stats
        assert stats["k1"] >= 2
        assert stats["b"] >= 2
    
    def test_track_distance_metric_usage(self, override_service, sample_vector_params, base_weaviate_config):
        """Test tracking distance metric usage"""
        param_id = override_service.create_parameter_set(
            name="Metric Test",
            description="Test metric tracking",
            vector_params=sample_vector_params
        )
        
        override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        metric_usage = override_service.statistics["distance_metric_usage"]
        
        assert "cosine" in metric_usage
        assert metric_usage["cosine"] >= 1


# ============================================================================
# Default Parameter Sets Tests
# ============================================================================

class TestDefaultParameterSets:
    """Test default parameter set creation"""
    
    def test_default_sets_created(self):
        """Test that default parameter sets are created"""
        # Create a fresh service that should have defaults
        service = BM25VectorOverrideService()
        # Create default parameter sets explicitly
        service.create_default_parameter_sets()
        # The service creates default sets
        assert len(service.parameter_sets) >= 5
    
    def test_high_precision_bm25_set(self):
        """Test high precision BM25 set exists"""
        service = BM25VectorOverrideService()
        service.create_default_parameter_sets()
        sets = service.list_parameter_sets()
        set_names = [s["name"] for s in sets]
        
        assert "High Precision BM25" in set_names
    
    def test_cosine_vector_search_set(self):
        """Test cosine vector search set exists"""
        service = BM25VectorOverrideService()
        service.create_default_parameter_sets()
        sets = service.list_parameter_sets()
        set_names = [s["name"] for s in sets]
        
        assert "Cosine Vector Search" in set_names
    
    def test_balanced_hybrid_search_set(self):
        """Test balanced hybrid search set exists"""
        service = BM25VectorOverrideService()
        service.create_default_parameter_sets()
        sets = service.list_parameter_sets()
        set_names = [s["name"] for s in sets]
        
        assert "Balanced Hybrid Search" in set_names


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_parameter_set(self, override_service):
        """Test creating parameter set with no params"""
        param_id = override_service.create_parameter_set(
            name="Empty Set",
            description="Set with no parameters"
        )
        
        param_set = override_service.parameter_sets[param_id]
        assert len(param_set.bm25_overrides) == 0
        assert len(param_set.vector_overrides) == 0
    
    def test_disabled_override_not_applied(self, override_service, base_weaviate_config):
        """Test that disabled overrides are not applied"""
        bm25_params = [
            {
                "parameter_type": "k1",
                "value": 1.5,
                "enabled": False  # Disabled
            }
        ]
        
        param_id = override_service.create_parameter_set(
            name="Disabled Test",
            description="Test disabled overrides",
            bm25_params=bm25_params
        )
        
        modified_config = override_service.apply_parameter_set(param_id, base_weaviate_config)
        
        # k1 should not be in config since it's disabled
        assert "bm25" not in modified_config or "k1" not in modified_config.get("bm25", {})
    
    def test_partial_invalid_params_skipped(self, override_service):
        """Test that invalid params are skipped with warning"""
        mixed_params = [
            {"parameter_type": "k1", "value": 1.5},  # Valid
            {"parameter_type": "invalid", "value": 1.0},  # Invalid
            {"parameter_type": "b", "value": 0.75}  # Valid
        ]
        
        param_id = override_service.create_parameter_set(
            name="Mixed Test",
            description="Test mixed valid/invalid params",
            bm25_params=mixed_params
        )
        
        param_set = override_service.parameter_sets[param_id]
        # Should have 2 valid overrides, 1 skipped
        assert len(param_set.bm25_overrides) == 2
