"""
LDTS-79: BM25 parameter overrides and vector distance selection

Enhanced parameter override system specifically for BM25 search parameters
and vector distance metric selection in Weaviate operations.
"""

import logging
# import json  # Currently unused
import copy
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class VectorDistanceMetric(Enum):
    """Supported vector distance metrics"""
    COSINE = "cosine"
    DOT = "dot"
    L2 = "l2"
    EUCLIDEAN = "euclidean"
    HAMMING = "hamming"
    MANHATTAN = "manhattan"

class BM25ParameterType(Enum):
    """BM25-specific parameter types"""
    K1 = "k1"
    B = "b"
    QUERY_PROPERTIES = "query_properties"
    BOOST = "boost"

class VectorParameterType(Enum):
    """Vector-specific parameter types"""
    DISTANCE_METRIC = "distance_metric"
    CERTAINTY = "certainty"
    DISTANCE_THRESHOLD = "distance_threshold"
    AUTOCUT = "autocut"

@dataclass
class BM25Override:
    """BM25 parameter override specification"""
    parameter_type: BM25ParameterType
    value: Union[float, List[str], Dict[str, float]]
    description: str
    validation_range: Optional[Dict[str, Any]] = None
    enabled: bool = True

@dataclass
class VectorDistanceOverride:
    """Vector distance parameter override specification"""
    parameter_type: VectorParameterType
    value: Any
    description: str
    validation_range: Optional[Dict[str, Any]] = None
    enabled: bool = True

@dataclass
class SearchParameterSet:
    """Complete search parameter configuration with overrides"""
    parameter_set_id: str
    name: str
    description: str
    bm25_overrides: List[BM25Override]
    vector_overrides: List[VectorDistanceOverride]
    hybrid_alpha: Optional[float] = 0.75
    fusion_type: str = "relative_score"
    created_at: datetime = None
    active: bool = True
    metadata: Optional[Dict[str, Any]] = None

class BM25VectorOverrideService:
    """
    Enhanced service for BM25 parameter overrides and vector distance selection.
    Extends the base Weaviate override system with specialized support for 
    BM25 and vector distance parameters.
    """
    
    def __init__(self):
        """Initialize BM25 and vector override service"""
        self.parameter_sets = {}
        self.active_overrides = {
            "bm25": {},
            "vector": {},
            "hybrid": {}
        }
        
        # Initialize parameter validation schemas
        self.bm25_schemas = self._initialize_bm25_schemas()
        self.vector_schemas = self._initialize_vector_schemas()
        
        self.statistics = {
            "total_parameter_sets": 0,
            "bm25_overrides_applied": 0,
            "vector_overrides_applied": 0,
            "failed_applications": 0,
            "most_used_parameters": {},
            "distance_metric_usage": {}
        }
        
        logger.info("Initialized BM25VectorOverrideService with comprehensive parameter support")
    
    def _initialize_bm25_schemas(self) -> Dict[BM25ParameterType, Dict[str, Any]]:
        """Initialize BM25 parameter validation schemas"""
        return {
            BM25ParameterType.K1: {
                "type": "float",
                "min": 0.0,
                "max": 3.0,
                "default": 1.2,
                "description": "Term frequency saturation parameter"
            },
            BM25ParameterType.B: {
                "type": "float", 
                "min": 0.0,
                "max": 1.0,
                "default": 0.75,
                "description": "Length normalization parameter"
            },
            BM25ParameterType.QUERY_PROPERTIES: {
                "type": "list",
                "element_type": "str",
                "description": "Properties to search in BM25"
            },
            BM25ParameterType.BOOST: {
                "type": "dict",
                "key_type": "str",
                "value_type": "float",
                "description": "Per-property boost factors"
            }
        }
    
    def _initialize_vector_schemas(self) -> Dict[VectorParameterType, Dict[str, Any]]:
        """Initialize vector parameter validation schemas"""
        return {
            VectorParameterType.DISTANCE_METRIC: {
                "type": "enum",
                "values": [metric.value for metric in VectorDistanceMetric],
                "default": "cosine",
                "description": "Vector distance calculation method"
            },
            VectorParameterType.CERTAINTY: {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Minimum certainty threshold for results"
            },
            VectorParameterType.DISTANCE_THRESHOLD: {
                "type": "float",
                "min": 0.0,
                "description": "Maximum distance threshold for results"
            },
            VectorParameterType.AUTOCUT: {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Automatic result cutoff factor"
            }
        }
    
    def create_parameter_set(
        self,
        name: str,
        description: str,
        bm25_params: List[Dict[str, Any]] = None,
        vector_params: List[Dict[str, Any]] = None,
        hybrid_alpha: Optional[float] = None,
        fusion_type: str = "relative_score",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new parameter set with BM25 and vector overrides"""
        
        parameter_set_id = f"param_set_{int(time.time())}_{len(self.parameter_sets)}"
        
        # Validate and create BM25 overrides
        bm25_overrides = []
        if bm25_params:
            for param in bm25_params:
                try:
                    override = self._create_bm25_override(param)
                    bm25_overrides.append(override)
                except ValueError as e:
                    logger.warning(f"Skipping invalid BM25 parameter: {e}")
        
        # Validate and create vector overrides
        vector_overrides = []
        if vector_params:
            for param in vector_params:
                try:
                    override = self._create_vector_override(param)
                    vector_overrides.append(override)
                except ValueError as e:
                    logger.warning(f"Skipping invalid vector parameter: {e}")
        
        # Validate hybrid alpha if provided
        if hybrid_alpha is not None:
            if not (0.0 <= hybrid_alpha <= 1.0):
                raise ValueError(f"Hybrid alpha must be between 0.0 and 1.0, got {hybrid_alpha}")
        
        parameter_set = SearchParameterSet(
            parameter_set_id=parameter_set_id,
            name=name,
            description=description,
            bm25_overrides=bm25_overrides,
            vector_overrides=vector_overrides,
            hybrid_alpha=hybrid_alpha,
            fusion_type=fusion_type,
            created_at=datetime.now(timezone.utc),
            active=True,
            metadata=metadata
        )
        
        self.parameter_sets[parameter_set_id] = parameter_set
        self.statistics["total_parameter_sets"] += 1
        
        logger.info(f"Created parameter set: {name} ({parameter_set_id}) with "
                   f"{len(bm25_overrides)} BM25 and {len(vector_overrides)} vector parameters")
        
        return parameter_set_id
    
    def _create_bm25_override(self, param_data: Dict[str, Any]) -> BM25Override:
        """Create and validate a BM25 parameter override"""
        
        if "parameter_type" not in param_data:
            raise ValueError("Missing parameter_type for BM25 override")
        
        try:
            param_type = BM25ParameterType(param_data["parameter_type"])
        except ValueError:
            raise ValueError(f"Invalid BM25 parameter type: {param_data['parameter_type']}")
        
        value = param_data.get("value")
        if value is None:
            raise ValueError(f"Missing value for BM25 parameter: {param_type.value}")
        
        # Validate value against schema
        schema = self.bm25_schemas[param_type]
        validation_errors = self._validate_parameter_value(value, schema)
        if validation_errors:
            raise ValueError(f"Invalid value for {param_type.value}: {', '.join(validation_errors)}")
        
        return BM25Override(
            parameter_type=param_type,
            value=value,
            description=param_data.get("description", f"Override for {param_type.value}"),
            validation_range=schema,
            enabled=param_data.get("enabled", True)
        )
    
    def _create_vector_override(self, param_data: Dict[str, Any]) -> VectorDistanceOverride:
        """Create and validate a vector parameter override"""
        
        if "parameter_type" not in param_data:
            raise ValueError("Missing parameter_type for vector override")
        
        try:
            param_type = VectorParameterType(param_data["parameter_type"])
        except ValueError:
            raise ValueError(f"Invalid vector parameter type: {param_data['parameter_type']}")
        
        value = param_data.get("value")
        if value is None:
            raise ValueError(f"Missing value for vector parameter: {param_type.value}")
        
        # Validate value against schema
        schema = self.vector_schemas[param_type]
        validation_errors = self._validate_parameter_value(value, schema)
        if validation_errors:
            raise ValueError(f"Invalid value for {param_type.value}: {', '.join(validation_errors)}")
        
        return VectorDistanceOverride(
            parameter_type=param_type,
            value=value,
            description=param_data.get("description", f"Override for {param_type.value}"),
            validation_range=schema,
            enabled=param_data.get("enabled", True)
        )
    
    def _validate_parameter_value(self, value: Any, schema: Dict[str, Any]) -> List[str]:
        """Validate parameter value against schema"""
        errors = []
        
        value_type = schema.get("type")
        
        if value_type == "float":
            if not isinstance(value, (int, float)):
                errors.append(f"Expected float, got {type(value).__name__}")
            else:
                if "min" in schema and value < schema["min"]:
                    errors.append(f"Value {value} below minimum {schema['min']}")
                if "max" in schema and value > schema["max"]:
                    errors.append(f"Value {value} above maximum {schema['max']}")
        
        elif value_type == "int":
            if not isinstance(value, int):
                errors.append(f"Expected int, got {type(value).__name__}")
            else:
                if "min" in schema and value < schema["min"]:
                    errors.append(f"Value {value} below minimum {schema['min']}")
                if "max" in schema and value > schema["max"]:
                    errors.append(f"Value {value} above maximum {schema['max']}")
        
        elif value_type == "enum":
            if value not in schema.get("values", []):
                errors.append(f"Value {value} not in allowed values: {schema['values']}")
        
        elif value_type == "list":
            if not isinstance(value, list):
                errors.append(f"Expected list, got {type(value).__name__}")
        
        elif value_type == "dict":
            if not isinstance(value, dict):
                errors.append(f"Expected dict, got {type(value).__name__}")
        
        return errors
    
    def apply_parameter_set(
        self,
        parameter_set_id: str,
        base_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a parameter set to a base Weaviate configuration"""
        
        if parameter_set_id not in self.parameter_sets:
            raise ValueError(f"Parameter set not found: {parameter_set_id}")
        
        param_set = self.parameter_sets[parameter_set_id]
        if not param_set.active:
            logger.warning(f"Parameter set {parameter_set_id} is inactive")
            return base_config
        
        # Create a deep copy to avoid modifying the original
        modified_config = copy.deepcopy(base_config)
        
        # Apply BM25 overrides
        if param_set.bm25_overrides:
            if "bm25" not in modified_config:
                modified_config["bm25"] = {}
            
            for override in param_set.bm25_overrides:
                if override.enabled:
                    self._apply_bm25_override(modified_config["bm25"], override)
                    self.statistics["bm25_overrides_applied"] += 1
        
        # Apply vector overrides
        if param_set.vector_overrides:
            if "vector" not in modified_config:
                modified_config["vector"] = {}
            
            for override in param_set.vector_overrides:
                if override.enabled:
                    self._apply_vector_override(modified_config["vector"], override)
                    self.statistics["vector_overrides_applied"] += 1
        
        # Apply hybrid parameters
        if param_set.hybrid_alpha is not None:
            if "hybrid" not in modified_config:
                modified_config["hybrid"] = {}
            modified_config["hybrid"]["alpha"] = param_set.hybrid_alpha
        
        if param_set.fusion_type:
            if "hybrid" not in modified_config:
                modified_config["hybrid"] = {}
            modified_config["hybrid"]["fusion_type"] = param_set.fusion_type
        
        logger.info(f"Applied parameter set {parameter_set_id} with "
                   f"{len(param_set.bm25_overrides)} BM25 and "
                   f"{len(param_set.vector_overrides)} vector overrides")
        
        return modified_config
    
    def _apply_bm25_override(self, bm25_config: Dict[str, Any], override: BM25Override):
        """Apply a single BM25 parameter override"""
        param_name = override.parameter_type.value
        bm25_config[param_name] = override.value
        
        # Track usage statistics
        if param_name not in self.statistics["most_used_parameters"]:
            self.statistics["most_used_parameters"][param_name] = 0
        self.statistics["most_used_parameters"][param_name] += 1
    
    def _apply_vector_override(self, vector_config: Dict[str, Any], override: VectorDistanceOverride):
        """Apply a single vector parameter override"""
        param_name = override.parameter_type.value
        vector_config[param_name] = override.value
        
        # Track distance metric usage
        if param_name == "distance_metric":
            metric = override.value
            if metric not in self.statistics["distance_metric_usage"]:
                self.statistics["distance_metric_usage"][metric] = 0
            self.statistics["distance_metric_usage"][metric] += 1
        
        # Track usage statistics
        if param_name not in self.statistics["most_used_parameters"]:
            self.statistics["most_used_parameters"][param_name] = 0
        self.statistics["most_used_parameters"][param_name] += 1
    
    def get_parameter_set(self, parameter_set_id: str) -> Optional[SearchParameterSet]:
        """Get a parameter set by ID"""
        return self.parameter_sets.get(parameter_set_id)
    
    def list_parameter_sets(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all parameter sets"""
        sets = []
        for param_set in self.parameter_sets.values():
            if not active_only or param_set.active:
                sets.append({
                    "parameter_set_id": param_set.parameter_set_id,
                    "name": param_set.name,
                    "description": param_set.description,
                    "bm25_override_count": len(param_set.bm25_overrides),
                    "vector_override_count": len(param_set.vector_overrides),
                    "hybrid_alpha": param_set.hybrid_alpha,
                    "fusion_type": param_set.fusion_type,
                    "created_at": param_set.created_at.isoformat(),
                    "active": param_set.active
                })
        
        return sorted(sets, key=lambda x: x["created_at"], reverse=True)
    
    def delete_parameter_set(self, parameter_set_id: str) -> bool:
        """Delete a parameter set"""
        if parameter_set_id in self.parameter_sets:
            del self.parameter_sets[parameter_set_id]
            logger.info(f"Deleted parameter set: {parameter_set_id}")
            return True
        return False
    
    def get_supported_distance_metrics(self) -> List[str]:
        """Get list of supported vector distance metrics"""
        return [metric.value for metric in VectorDistanceMetric]
    
    def get_bm25_parameter_schema(self) -> Dict[str, Any]:
        """Get BM25 parameter validation schema"""
        return {
            param_type.value: schema 
            for param_type, schema in self.bm25_schemas.items()
        }
    
    def get_vector_parameter_schema(self) -> Dict[str, Any]:
        """Get vector parameter validation schema"""
        return {
            param_type.value: schema 
            for param_type, schema in self.vector_schemas.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service usage statistics"""
        return {
            **self.statistics,
            "supported_distance_metrics": self.get_supported_distance_metrics(),
            "active_parameter_sets": len([p for p in self.parameter_sets.values() if p.active])
        }
    
    def create_default_parameter_sets(self):
        """Create some default parameter sets for common scenarios"""
        
        # High precision BM25
        self.create_parameter_set(
            name="High Precision BM25",
            description="BM25 optimized for high precision results",
            bm25_params=[
                {"parameter_type": "k1", "value": 1.8, "description": "Higher term frequency impact"},
                {"parameter_type": "b", "value": 0.9, "description": "Strong length normalization"}
            ]
        )
        
        # High recall BM25
        self.create_parameter_set(
            name="High Recall BM25", 
            description="BM25 optimized for high recall",
            bm25_params=[
                {"parameter_type": "k1", "value": 0.8, "description": "Lower term frequency saturation"},
                {"parameter_type": "b", "value": 0.4, "description": "Reduced length normalization"}
            ]
        )
        
        # Cosine similarity vector search
        self.create_parameter_set(
            name="Cosine Vector Search",
            description="Vector search with cosine similarity",
            vector_params=[
                {"parameter_type": "distance_metric", "value": "cosine", "description": "Cosine similarity metric"},
                {"parameter_type": "certainty", "value": 0.7, "description": "70% minimum certainty"}
            ]
        )
        
        # Euclidean distance vector search
        self.create_parameter_set(
            name="Euclidean Vector Search",
            description="Vector search with Euclidean distance",
            vector_params=[
                {"parameter_type": "distance_metric", "value": "euclidean", "description": "Euclidean distance metric"},
                {"parameter_type": "distance_threshold", "value": 1.5, "description": "Maximum distance threshold"}
            ]
        )
        
        # Balanced hybrid search
        self.create_parameter_set(
            name="Balanced Hybrid Search",
            description="Balanced hybrid search with optimal parameters",
            bm25_params=[
                {"parameter_type": "k1", "value": 1.2, "description": "Standard BM25 k1"},
                {"parameter_type": "b", "value": 0.75, "description": "Standard BM25 b"}
            ],
            vector_params=[
                {"parameter_type": "distance_metric", "value": "cosine", "description": "Cosine similarity"}
            ],
            hybrid_alpha=0.75,
            fusion_type="relative_score"
        )
        
        logger.info("Created 5 default parameter sets for common search scenarios")

# Global service instance
bm25_vector_override_service = BM25VectorOverrideService()

# Create default parameter sets on initialization
bm25_vector_override_service.create_default_parameter_sets()