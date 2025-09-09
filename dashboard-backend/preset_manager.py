"""
LDTS-38: Preset management and configuration templates

Comprehensive preset management system for storing, loading, and managing
configuration templates and presets for the LDTS dashboard.
"""

import json
import yaml
import logging
import asyncio
import hashlib
import shutil
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
import uuid

logger = logging.getLogger(__name__)

class PresetCategory(Enum):
    """Preset categories"""
    QUICK_START = "quick_start"
    ADVANCED = "advanced"
    CUSTOM = "custom"
    BENCHMARK = "benchmark"
    DEVELOPMENT = "development"
    PRODUCTION = "production"

class PresetFormat(Enum):
    """Preset storage formats"""
    JSON = "json"
    YAML = "yaml"

@dataclass
class PresetMetadata:
    """Preset metadata information"""
    name: str
    category: PresetCategory
    description: str
    version: str = "1.0.0"
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    use_count: int = 0
    last_used: Optional[datetime] = None
    checksum: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)  # Required services/providers
    performance_rating: Optional[float] = None  # 1.0-5.0 scale
    cost_rating: Optional[float] = None  # 1.0-5.0 scale (1=cheap, 5=expensive)

@dataclass
class ConfigPreset:
    """Complete configuration preset"""
    metadata: PresetMetadata
    configuration: Dict[str, Any]
    format: PresetFormat = PresetFormat.YAML
    validation_result: Optional[Dict[str, Any]] = None

class PresetManager:
    """Comprehensive preset management system"""
    
    def __init__(self, presets_directory: Optional[str] = None):
        self.presets_dir = Path(presets_directory) if presets_directory else Path(__file__).parent / "presets"
        self.presets_dir.mkdir(exist_ok=True)
        
        # Create category subdirectories
        for category in PresetCategory:
            (self.presets_dir / category.value).mkdir(exist_ok=True)
        
        self.loaded_presets: Dict[str, ConfigPreset] = {}
        self.preset_index: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize preset manager"""
        try:
            logger.info(f"Initializing preset manager with directory: {self.presets_dir}")
            
            # Create built-in presets if they don't exist
            await self._ensure_builtin_presets()
            
            # Load all presets
            await self.load_all_presets()
            
            # Build index
            await self._build_preset_index()
            
            self._initialized = True
            logger.info(f"Preset manager initialized with {len(self.loaded_presets)} presets")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize preset manager: {e}")
            return False
    
    async def _ensure_builtin_presets(self):
        """Create built-in presets if they don't exist"""
        builtin_presets = {
            PresetCategory.QUICK_START: [
                {
                    "name": "fast_local_search",
                    "description": "Fast local search using Ollama without reranking",
                    "config": {
                        "embedding_providers": {
                            "default_provider": "ollama",
                            "ollama": {
                                "enabled": True,
                                "base_url": "${OLLAMA_BASE_URL:-http://192.168.50.80:11434}",
                                "model": "nomic-embed-text",
                                "dimensions": 768,
                                "batch_size": 50,
                                "timeout_seconds": 30
                            }
                        },
                        "reranker_models": {},
                        "weaviate": {
                            "url": "${WEAVIATE_URL:-http://weaviate:8080}",
                            "search": {
                                "hybrid": {"alpha": 0.8},
                                "vector": {"certainty": 0.6}
                            }
                        },
                        "search_config": {
                            "search_strategy": {
                                "use_hybrid_search": True,
                                "enable_reranking": False,
                                "final_results_limit": 10
                            }
                        },
                        "performance": {
                            "caching": {"enabled": True, "ttl_seconds": 1800}
                        }
                    },
                    "tags": ["fast", "local", "development"],
                    "dependencies": ["ollama", "weaviate"],
                    "performance_rating": 5.0,
                    "cost_rating": 1.0
                },
                {
                    "name": "high_quality_search",
                    "description": "High-quality search with OpenAI embeddings and reranking",
                    "config": {
                        "embedding_providers": {
                            "default_provider": "openai",
                            "fallback_provider": "ollama",
                            "openai": {
                                "enabled": True,
                                "api_key": "${OPENAI_API_KEY}",
                                "model": "text-embedding-3-small",
                                "dimensions": 1536,
                                "batch_size": 100,
                                "timeout_seconds": 30
                            },
                            "ollama": {
                                "enabled": True,
                                "base_url": "${OLLAMA_BASE_URL:-http://192.168.50.80:11434}",
                                "model": "nomic-embed-text"
                            }
                        },
                        "reranker_models": {
                            "default_reranker": "ollama_reranker",
                            "ollama_reranker": {
                                "enabled": True,
                                "base_url": "${OLLAMA_BASE_URL:-http://192.168.50.80:11434}",
                                "model": "qwen3-reranker-4b",
                                "max_batch_size": 16,
                                "temperature": 0.1
                            }
                        },
                        "search_config": {
                            "search_strategy": {
                                "use_hybrid_search": True,
                                "enable_reranking": True,
                                "rerank_top_k": 50,
                                "final_results_limit": 20
                            }
                        }
                    },
                    "tags": ["quality", "openai", "reranking"],
                    "dependencies": ["openai", "ollama", "weaviate"],
                    "performance_rating": 3.0,
                    "cost_rating": 4.0
                }
            ],
            PresetCategory.DEVELOPMENT: [
                {
                    "name": "debug_mode",
                    "description": "Development preset with debug logging and fast responses",
                    "config": {
                        "embedding_providers": {
                            "default_provider": "ollama",
                            "ollama": {
                                "enabled": True,
                                "base_url": "http://localhost:11434",
                                "model": "nomic-embed-text",
                                "batch_size": 10,
                                "timeout_seconds": 15
                            }
                        },
                        "search_config": {
                            "search_strategy": {
                                "final_results_limit": 5,
                                "enable_reranking": False
                            }
                        },
                        "performance": {
                            "caching": {"enabled": False},
                            "logging": {"level": "DEBUG"}
                        },
                        "dashboard": {
                            "ui": {"show_debug_info": True}
                        }
                    },
                    "tags": ["debug", "development", "local"],
                    "dependencies": ["ollama"],
                    "performance_rating": 4.0,
                    "cost_rating": 1.0
                }
            ],
            PresetCategory.BENCHMARK: [
                {
                    "name": "evaluation_benchmark",
                    "description": "Comprehensive evaluation preset for benchmarking",
                    "config": {
                        "embedding_providers": {
                            "default_provider": "openai",
                            "openai": {
                                "enabled": True,
                                "api_key": "${OPENAI_API_KEY}",
                                "model": "text-embedding-3-small"
                            }
                        },
                        "reranker_models": {
                            "default_reranker": "ollama_reranker",
                            "ollama_reranker": {
                                "enabled": True,
                                "base_url": "${OLLAMA_BASE_URL}",
                                "model": "qwen3-reranker-4b"
                            }
                        },
                        "evaluation": {
                            "metrics": {
                                "precision_at_k": [1, 3, 5, 10],
                                "recall_at_k": [1, 3, 5, 10],
                                "ndcg_at_k": [1, 3, 5, 10],
                                "mrr": True,
                                "map": True
                            },
                            "automated_evaluation": {
                                "enabled": True,
                                "batch_size": 50
                            }
                        },
                        "search_config": {
                            "search_strategy": {
                                "final_results_limit": 50,
                                "enable_reranking": True
                            }
                        }
                    },
                    "tags": ["benchmark", "evaluation", "comprehensive"],
                    "dependencies": ["openai", "ollama", "weaviate"],
                    "performance_rating": 2.0,
                    "cost_rating": 4.0
                }
            ]
        }
        
        # Create preset files
        for category, presets in builtin_presets.items():
            category_dir = self.presets_dir / category.value
            
            for preset_data in presets:
                preset_path = category_dir / f"{preset_data['name']}.yaml"
                
                if not preset_path.exists():
                    # Create preset metadata
                    metadata = PresetMetadata(
                        name=preset_data["name"],
                        category=category,
                        description=preset_data["description"],
                        author="LDTS System",
                        created_at=datetime.now(timezone.utc),
                        tags=preset_data.get("tags", []),
                        dependencies=preset_data.get("dependencies", []),
                        performance_rating=preset_data.get("performance_rating"),
                        cost_rating=preset_data.get("cost_rating")
                    )
                    
                    # Create full preset structure
                    preset_file_content = {
                        "metadata": {
                            "name": metadata.name,
                            "category": metadata.category.value,
                            "description": metadata.description,
                            "version": metadata.version,
                            "author": metadata.author,
                            "created_at": metadata.created_at.isoformat(),
                            "tags": metadata.tags,
                            "dependencies": metadata.dependencies,
                            "performance_rating": metadata.performance_rating,
                            "cost_rating": metadata.cost_rating
                        },
                        "configuration": preset_data["config"]
                    }
                    
                    # Save preset
                    with open(preset_path, 'w') as f:
                        yaml.dump(preset_file_content, f, default_flow_style=False, indent=2)
                    
                    logger.info(f"Created built-in preset: {preset_data['name']}")
    
    async def load_all_presets(self) -> int:
        """Load all presets from the presets directory"""
        try:
            loaded_count = 0
            
            for category in PresetCategory:
                category_dir = self.presets_dir / category.value
                if not category_dir.exists():
                    continue
                
                # Load YAML presets
                for preset_file in category_dir.glob("*.yaml"):
                    try:
                        preset = await self._load_preset_from_file(preset_file, PresetFormat.YAML)
                        if preset:
                            self.loaded_presets[preset.metadata.name] = preset
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to load preset {preset_file}: {e}")
                
                # Load JSON presets
                for preset_file in category_dir.glob("*.json"):
                    try:
                        preset = await self._load_preset_from_file(preset_file, PresetFormat.JSON)
                        if preset:
                            self.loaded_presets[preset.metadata.name] = preset
                            loaded_count += 1
                    except Exception as e:
                        logger.error(f"Failed to load preset {preset_file}: {e}")
            
            logger.info(f"Loaded {loaded_count} presets")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load presets: {e}")
            return 0
    
    async def _load_preset_from_file(self, file_path: Path, format: PresetFormat) -> Optional[ConfigPreset]:
        """Load a preset from a file"""
        try:
            with open(file_path, 'r') as f:
                if format == PresetFormat.YAML:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Extract metadata
            metadata_dict = data.get("metadata", {})
            metadata = PresetMetadata(
                name=metadata_dict.get("name", file_path.stem),
                category=PresetCategory(metadata_dict.get("category", "custom")),
                description=metadata_dict.get("description", ""),
                version=metadata_dict.get("version", "1.0.0"),
                author=metadata_dict.get("author"),
                created_at=datetime.fromisoformat(metadata_dict["created_at"]) if "created_at" in metadata_dict else None,
                updated_at=datetime.fromisoformat(metadata_dict["updated_at"]) if "updated_at" in metadata_dict else None,
                tags=metadata_dict.get("tags", []),
                use_count=metadata_dict.get("use_count", 0),
                last_used=datetime.fromisoformat(metadata_dict["last_used"]) if "last_used" in metadata_dict else None,
                dependencies=metadata_dict.get("dependencies", []),
                performance_rating=metadata_dict.get("performance_rating"),
                cost_rating=metadata_dict.get("cost_rating")
            )
            
            # Calculate checksum
            config_content = json.dumps(data.get("configuration", {}), sort_keys=True)
            metadata.checksum = hashlib.sha256(config_content.encode()).hexdigest()[:16]
            
            return ConfigPreset(
                metadata=metadata,
                configuration=data.get("configuration", {}),
                format=format
            )
            
        except Exception as e:
            logger.error(f"Failed to load preset from {file_path}: {e}")
            return None
    
    async def _build_preset_index(self):
        """Build search index for presets"""
        self.preset_index = {
            "by_category": {},
            "by_tag": {},
            "by_dependency": {},
            "by_rating": {"performance": {}, "cost": {}},
            "by_name": {}
        }
        
        for preset_name, preset in self.loaded_presets.items():
            # Index by category
            category = preset.metadata.category.value
            if category not in self.preset_index["by_category"]:
                self.preset_index["by_category"][category] = []
            self.preset_index["by_category"][category].append(preset_name)
            
            # Index by tags
            for tag in preset.metadata.tags:
                if tag not in self.preset_index["by_tag"]:
                    self.preset_index["by_tag"][tag] = []
                self.preset_index["by_tag"][tag].append(preset_name)
            
            # Index by dependencies
            for dep in preset.metadata.dependencies:
                if dep not in self.preset_index["by_dependency"]:
                    self.preset_index["by_dependency"][dep] = []
                self.preset_index["by_dependency"][dep].append(preset_name)
            
            # Index by ratings
            if preset.metadata.performance_rating:
                rating_key = f"{int(preset.metadata.performance_rating)}_star"
                if rating_key not in self.preset_index["by_rating"]["performance"]:
                    self.preset_index["by_rating"]["performance"][rating_key] = []
                self.preset_index["by_rating"]["performance"][rating_key].append(preset_name)
            
            if preset.metadata.cost_rating:
                rating_key = f"{int(preset.metadata.cost_rating)}_star"
                if rating_key not in self.preset_index["by_rating"]["cost"]:
                    self.preset_index["by_rating"]["cost"][rating_key] = []
                self.preset_index["by_rating"]["cost"][rating_key].append(preset_name)
            
            # Index by name (for fast lookup)
            self.preset_index["by_name"][preset_name] = {
                "category": category,
                "description": preset.metadata.description,
                "tags": preset.metadata.tags,
                "performance_rating": preset.metadata.performance_rating,
                "cost_rating": preset.metadata.cost_rating
            }
    
    async def get_preset(self, name: str) -> Optional[ConfigPreset]:
        """Get a preset by name"""
        return self.loaded_presets.get(name)
    
    async def list_presets(self, 
                          category: Optional[PresetCategory] = None,
                          tag: Optional[str] = None,
                          dependency: Optional[str] = None,
                          min_performance_rating: Optional[float] = None,
                          max_cost_rating: Optional[float] = None) -> List[Dict[str, Any]]:
        """List presets with optional filtering"""
        if not self._initialized:
            await self.initialize()
        
        preset_list = []
        
        # Get preset names based on filters
        candidate_names = set(self.loaded_presets.keys())
        
        if category:
            category_presets = set(self.preset_index["by_category"].get(category.value, []))
            candidate_names &= category_presets
        
        if tag:
            tag_presets = set(self.preset_index["by_tag"].get(tag, []))
            candidate_names &= tag_presets
        
        if dependency:
            dep_presets = set(self.preset_index["by_dependency"].get(dependency, []))
            candidate_names &= dep_presets
        
        # Apply rating filters
        for name in candidate_names:
            preset = self.loaded_presets[name]
            
            if min_performance_rating and (
                not preset.metadata.performance_rating or 
                preset.metadata.performance_rating < min_performance_rating
            ):
                continue
            
            if max_cost_rating and (
                not preset.metadata.cost_rating or 
                preset.metadata.cost_rating > max_cost_rating
            ):
                continue
            
            preset_list.append({
                "name": preset.metadata.name,
                "category": preset.metadata.category.value,
                "description": preset.metadata.description,
                "version": preset.metadata.version,
                "author": preset.metadata.author,
                "tags": preset.metadata.tags,
                "use_count": preset.metadata.use_count,
                "last_used": preset.metadata.last_used.isoformat() if preset.metadata.last_used else None,
                "dependencies": preset.metadata.dependencies,
                "performance_rating": preset.metadata.performance_rating,
                "cost_rating": preset.metadata.cost_rating,
                "checksum": preset.metadata.checksum
            })
        
        # Sort by use count (most used first) then by name
        preset_list.sort(key=lambda x: (-x["use_count"], x["name"]))
        
        return preset_list
    
    async def create_preset(self, 
                           name: str,
                           configuration: Dict[str, Any],
                           category: PresetCategory = PresetCategory.CUSTOM,
                           description: str = "",
                           tags: List[str] = None,
                           author: str = None,
                           dependencies: List[str] = None,
                           format: PresetFormat = PresetFormat.YAML) -> bool:
        """Create a new preset"""
        try:
            # Validate preset doesn't exist
            if name in self.loaded_presets:
                raise ValueError(f"Preset '{name}' already exists")
            
            # Create metadata
            metadata = PresetMetadata(
                name=name,
                category=category,
                description=description,
                author=author,
                created_at=datetime.now(timezone.utc),
                tags=tags or [],
                dependencies=dependencies or []
            )
            
            # Calculate checksum
            config_content = json.dumps(configuration, sort_keys=True)
            metadata.checksum = hashlib.sha256(config_content.encode()).hexdigest()[:16]
            
            # Create preset
            preset = ConfigPreset(
                metadata=metadata,
                configuration=configuration,
                format=format
            )
            
            # Save to disk
            await self._save_preset_to_disk(preset)
            
            # Add to loaded presets
            self.loaded_presets[name] = preset
            
            # Rebuild index
            await self._build_preset_index()
            
            logger.info(f"Created preset: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create preset {name}: {e}")
            return False
    
    async def _save_preset_to_disk(self, preset: ConfigPreset):
        """Save preset to disk"""
        category_dir = self.presets_dir / preset.metadata.category.value
        category_dir.mkdir(exist_ok=True)
        
        # Prepare file content
        file_content = {
            "metadata": {
                "name": preset.metadata.name,
                "category": preset.metadata.category.value,
                "description": preset.metadata.description,
                "version": preset.metadata.version,
                "author": preset.metadata.author,
                "created_at": preset.metadata.created_at.isoformat() if preset.metadata.created_at else None,
                "updated_at": preset.metadata.updated_at.isoformat() if preset.metadata.updated_at else None,
                "tags": preset.metadata.tags,
                "use_count": preset.metadata.use_count,
                "last_used": preset.metadata.last_used.isoformat() if preset.metadata.last_used else None,
                "dependencies": preset.metadata.dependencies,
                "performance_rating": preset.metadata.performance_rating,
                "cost_rating": preset.metadata.cost_rating
            },
            "configuration": preset.configuration
        }
        
        # Save file
        if preset.format == PresetFormat.YAML:
            file_path = category_dir / f"{preset.metadata.name}.yaml"
            with open(file_path, 'w') as f:
                yaml.dump(file_content, f, default_flow_style=False, indent=2)
        else:
            file_path = category_dir / f"{preset.metadata.name}.json"
            with open(file_path, 'w') as f:
                json.dump(file_content, f, indent=2)
    
    async def update_preset(self, name: str, 
                           configuration: Optional[Dict[str, Any]] = None,
                           metadata_updates: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing preset"""
        try:
            if name not in self.loaded_presets:
                raise ValueError(f"Preset '{name}' not found")
            
            preset = self.loaded_presets[name]
            
            # Update configuration if provided
            if configuration is not None:
                preset.configuration = configuration
                # Recalculate checksum
                config_content = json.dumps(configuration, sort_keys=True)
                preset.metadata.checksum = hashlib.sha256(config_content.encode()).hexdigest()[:16]
            
            # Update metadata if provided
            if metadata_updates:
                for key, value in metadata_updates.items():
                    if hasattr(preset.metadata, key):
                        setattr(preset.metadata, key, value)
            
            # Set updated timestamp
            preset.metadata.updated_at = datetime.now(timezone.utc)
            
            # Save to disk
            await self._save_preset_to_disk(preset)
            
            # Rebuild index
            await self._build_preset_index()
            
            logger.info(f"Updated preset: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update preset {name}: {e}")
            return False
    
    async def delete_preset(self, name: str) -> bool:
        """Delete a preset"""
        try:
            if name not in self.loaded_presets:
                raise ValueError(f"Preset '{name}' not found")
            
            preset = self.loaded_presets[name]
            
            # Remove file from disk
            category_dir = self.presets_dir / preset.metadata.category.value
            
            for extension in ["yaml", "json"]:
                file_path = category_dir / f"{name}.{extension}"
                if file_path.exists():
                    file_path.unlink()
                    break
            
            # Remove from memory
            del self.loaded_presets[name]
            
            # Rebuild index
            await self._build_preset_index()
            
            logger.info(f"Deleted preset: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete preset {name}: {e}")
            return False
    
    async def use_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """Use a preset and update usage statistics"""
        try:
            if name not in self.loaded_presets:
                return None
            
            preset = self.loaded_presets[name]
            
            # Update usage statistics
            preset.metadata.use_count += 1
            preset.metadata.last_used = datetime.now(timezone.utc)
            
            # Save updated metadata to disk
            await self._save_preset_to_disk(preset)
            
            logger.info(f"Used preset: {name} (use count: {preset.metadata.use_count})")
            
            return preset.configuration.copy()
            
        except Exception as e:
            logger.error(f"Failed to use preset {name}: {e}")
            return None
    
    async def export_preset(self, name: str, output_path: str) -> bool:
        """Export a preset to a file"""
        try:
            if name not in self.loaded_presets:
                raise ValueError(f"Preset '{name}' not found")
            
            preset = self.loaded_presets[name]
            output_file = Path(output_path)
            
            # Determine format from extension
            if output_file.suffix.lower() == '.json':
                format = PresetFormat.JSON
            else:
                format = PresetFormat.YAML
            
            # Create export data
            export_data = {
                "metadata": {
                    "name": preset.metadata.name,
                    "category": preset.metadata.category.value,
                    "description": preset.metadata.description,
                    "version": preset.metadata.version,
                    "author": preset.metadata.author,
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "tags": preset.metadata.tags,
                    "dependencies": preset.metadata.dependencies,
                    "performance_rating": preset.metadata.performance_rating,
                    "cost_rating": preset.metadata.cost_rating
                },
                "configuration": preset.configuration
            }
            
            # Save export file
            with open(output_file, 'w') as f:
                if format == PresetFormat.JSON:
                    json.dump(export_data, f, indent=2)
                else:
                    yaml.dump(export_data, f, default_flow_style=False, indent=2)
            
            logger.info(f"Exported preset {name} to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export preset {name}: {e}")
            return False
    
    async def import_preset(self, file_path: str, 
                           new_name: Optional[str] = None,
                           category_override: Optional[PresetCategory] = None) -> bool:
        """Import a preset from a file"""
        try:
            input_file = Path(file_path)
            
            if not input_file.exists():
                raise ValueError(f"Import file not found: {file_path}")
            
            # Determine format
            if input_file.suffix.lower() == '.json':
                format = PresetFormat.JSON
            else:
                format = PresetFormat.YAML
            
            # Load preset from file
            preset = await self._load_preset_from_file(input_file, format)
            if not preset:
                raise ValueError("Failed to load preset from file")
            
            # Apply overrides
            if new_name:
                preset.metadata.name = new_name
            
            if category_override:
                preset.metadata.category = category_override
            
            # Check for name conflicts
            if preset.metadata.name in self.loaded_presets:
                raise ValueError(f"Preset '{preset.metadata.name}' already exists")
            
            # Save preset
            await self._save_preset_to_disk(preset)
            
            # Add to loaded presets
            self.loaded_presets[preset.metadata.name] = preset
            
            # Rebuild index
            await self._build_preset_index()
            
            logger.info(f"Imported preset: {preset.metadata.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import preset from {file_path}: {e}")
            return False
    
    def get_preset_statistics(self) -> Dict[str, Any]:
        """Get preset usage and distribution statistics"""
        if not self._initialized:
            return {}
        
        stats = {
            "total_presets": len(self.loaded_presets),
            "by_category": {},
            "by_author": {},
            "usage_stats": {
                "most_used": [],
                "never_used": [],
                "total_usage": 0
            },
            "tag_distribution": {},
            "dependency_distribution": {}
        }
        
        # Calculate statistics
        total_usage = 0
        usage_data = []
        
        for preset in self.loaded_presets.values():
            # Category distribution
            category = preset.metadata.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # Author distribution
            author = preset.metadata.author or "Unknown"
            stats["by_author"][author] = stats["by_author"].get(author, 0) + 1
            
            # Usage tracking
            total_usage += preset.metadata.use_count
            usage_data.append({
                "name": preset.metadata.name,
                "use_count": preset.metadata.use_count,
                "last_used": preset.metadata.last_used
            })
            
            # Tag distribution
            for tag in preset.metadata.tags:
                stats["tag_distribution"][tag] = stats["tag_distribution"].get(tag, 0) + 1
            
            # Dependency distribution
            for dep in preset.metadata.dependencies:
                stats["dependency_distribution"][dep] = stats["dependency_distribution"].get(dep, 0) + 1
        
        # Sort usage data
        usage_data.sort(key=lambda x: x["use_count"], reverse=True)
        
        stats["usage_stats"]["total_usage"] = total_usage
        stats["usage_stats"]["most_used"] = usage_data[:10]  # Top 10
        stats["usage_stats"]["never_used"] = [
            preset["name"] for preset in usage_data if preset["use_count"] == 0
        ]
        
        return stats

# Global preset manager instance
preset_manager: Optional[PresetManager] = None

async def initialize_preset_manager(presets_directory: Optional[str] = None) -> bool:
    """Initialize global preset manager"""
    global preset_manager
    
    preset_manager = PresetManager(presets_directory)
    return await preset_manager.initialize()

def get_preset_manager() -> PresetManager:
    """Get global preset manager instance"""
    if preset_manager is None:
        raise RuntimeError("Preset manager not initialized")
    return preset_manager