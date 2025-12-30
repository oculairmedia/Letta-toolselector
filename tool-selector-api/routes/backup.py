"""
Backup Routes Blueprint

Handles configuration backup, restore, save, and validation endpoints:
- /api/v1/config/backup - Create, list, get, delete backups
- /api/v1/config/restore - Restore from backups
- /api/v1/config/save - Save current configuration
- /api/v1/config/saves - List and manage saved configurations
- /api/v1/config/reset - Reset configuration to defaults or saved state
- /api/v1/config/validate - Real-time configuration validation
"""

from quart import Blueprint, request, jsonify
import logging
import os
import json
import sys
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
backup_bp = Blueprint('backup', __name__, url_prefix='/api/v1/config')

# Module state - to be configured
_cache_dir = None
_log_config_change = None
_perform_configuration_validation = None
_test_service_connection = None


def configure(cache_dir=None, log_config_change=None, 
              perform_configuration_validation=None, test_service_connection=None):
    """
    Configure the backup routes with required dependencies.
    
    Args:
        cache_dir: Directory for storing backups and saves
        log_config_change: Async function to log configuration changes
        perform_configuration_validation: Async function to validate config values
        test_service_connection: Async function to test service connections
    """
    global _cache_dir, _log_config_change, _perform_configuration_validation, _test_service_connection
    _cache_dir = cache_dir
    _log_config_change = log_config_change
    _perform_configuration_validation = perform_configuration_validation
    _test_service_connection = test_service_connection


# =============================================================================
# Configuration Backup Endpoints
# =============================================================================

@backup_bp.route('/backup', methods=['POST'])
async def create_config_backup():
    """Create a backup of all configuration settings."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        backup_name = data.get('name', f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        include_secrets = data.get('include_secrets', False)

        # Collect all configuration data
        backup_data = {
            "backup_info": {
                "name": backup_name,
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "includes_secrets": include_secrets,
                "system_info": {
                    "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown",
                    "python_version": sys.version,
                    "app_version": "1.0.0"
                }
            },
            "configurations": {}
        }

        # Tool Selector Configuration
        try:
            tool_selector_config = {
                "tool_limits": {
                    "max_total_tools": int(os.getenv('MAX_TOTAL_TOOLS', '30')),
                    "max_mcp_tools": int(os.getenv('MAX_MCP_TOOLS', '20')),
                    "min_mcp_tools": int(os.getenv('MIN_MCP_TOOLS', '7'))
                },
                "behavior": {
                    "default_drop_rate": float(os.getenv('DEFAULT_DROP_RATE', '0.6')),
                    "exclude_letta_core_tools": os.getenv('EXCLUDE_LETTA_CORE_TOOLS', 'true').lower() == 'true',
                    "exclude_official_tools": os.getenv('EXCLUDE_OFFICIAL_TOOLS', 'true').lower() == 'true',
                    "manage_only_mcp_tools": os.getenv('MANAGE_ONLY_MCP_TOOLS', 'true').lower() == 'true'
                }
            }
            backup_data["configurations"]["tool_selector"] = tool_selector_config
        except Exception as e:
            logger.warning(f"Failed to backup tool selector config: {e}")

        # Ollama Configuration
        try:
            ollama_config = {
                "connection": {
                    "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                    "port": int(os.getenv('OLLAMA_PORT', '11434')),
                    "timeout": int(os.getenv('OLLAMA_TIMEOUT', '30'))
                },
                "embedding": {
                    "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
                    "enabled": os.getenv('USE_OLLAMA_EMBEDDINGS', 'false').lower() == 'true'
                },
                "generation": {
                    "default_model": os.getenv('OLLAMA_DEFAULT_MODEL', 'mistral:7b'),
                    "temperature": float(os.getenv('OLLAMA_TEMPERATURE', '0.7')),
                    "context_length": int(os.getenv('OLLAMA_CONTEXT_LENGTH', '4096'))
                }
            }
            backup_data["configurations"]["ollama"] = ollama_config
        except Exception as e:
            logger.warning(f"Failed to backup Ollama config: {e}")

        # Weaviate Configuration
        try:
            weaviate_config = {
                "connection": {
                    "url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/'),
                    "timeout": int(os.getenv('WEAVIATE_TIMEOUT', '30')),
                    "retries": int(os.getenv('WEAVIATE_RETRIES', '3'))
                },
                "schema": {
                    "class_name": os.getenv('WEAVIATE_CLASS_NAME', 'Tool'),
                    "vector_index_type": os.getenv('WEAVIATE_VECTOR_INDEX', 'hnsw'),
                    "distance_metric": os.getenv('WEAVIATE_DISTANCE_METRIC', 'cosine')
                },
                "search": {
                    "alpha": float(os.getenv('WEAVIATE_ALPHA', '0.75')),
                    "limit": int(os.getenv('WEAVIATE_LIMIT', '50'))
                }
            }

            if include_secrets and os.getenv('WEAVIATE_API_KEY'):
                weaviate_config["connection"]["api_key"] = os.getenv('WEAVIATE_API_KEY')

            backup_data["configurations"]["weaviate"] = weaviate_config
        except Exception as e:
            logger.warning(f"Failed to backup Weaviate config: {e}")

        # Embedding Configuration
        try:
            embedding_config = {
                "provider": os.getenv('EMBEDDING_PROVIDER', 'openai'),
                "openai_model": os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                "ollama_model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M')
            }

            if include_secrets and os.getenv('OPENAI_API_KEY'):
                embedding_config["openai_api_key"] = os.getenv('OPENAI_API_KEY')

            backup_data["configurations"]["embedding"] = embedding_config
        except Exception as e:
            logger.warning(f"Failed to backup embedding config: {e}")

        # Letta API Configuration
        try:
            letta_config = {
                "api_url": os.getenv('LETTA_API_URL', 'https://letta.example.com/v1'),
                "timeout": int(os.getenv('LETTA_TIMEOUT', '30'))
            }

            if include_secrets and os.getenv('LETTA_PASSWORD'):
                letta_config["password"] = os.getenv('LETTA_PASSWORD')

            backup_data["configurations"]["letta"] = letta_config
        except Exception as e:
            logger.warning(f"Failed to backup Letta config: {e}")

        # Save backup to file
        backup_dir = os.path.join(_cache_dir, 'backups')
        os.makedirs(backup_dir, exist_ok=True)

        backup_file = os.path.join(backup_dir, f"{backup_name}.json")
        with open(backup_file, 'w') as f:
            json.dump(backup_data, f, indent=2)

        logger.info(f"Configuration backup created: {backup_file}")

        return jsonify({
            "success": True,
            "message": "Configuration backup created successfully",
            "data": {
                "backup_name": backup_name,
                "backup_file": backup_file,
                "size_bytes": os.path.getsize(backup_file),
                "configurations_count": len(backup_data["configurations"]),
                "created_at": backup_data["backup_info"]["created_at"]
            }
        })

    except Exception as e:
        logger.error(f"Error creating configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/backup', methods=['GET'])
async def list_config_backups():
    """List all available configuration backups."""
    try:
        backup_dir = os.path.join(_cache_dir, 'backups')

        if not os.path.exists(backup_dir):
            return jsonify({
                "success": True,
                "data": {
                    "backups": [],
                    "total": 0
                }
            })

        backups = []
        for filename in os.listdir(backup_dir):
            if filename.endswith('.json'):
                backup_path = os.path.join(backup_dir, filename)
                try:
                    with open(backup_path, 'r') as f:
                        backup_data = json.load(f)

                    backup_info = backup_data.get("backup_info", {})
                    backups.append({
                        "name": backup_info.get("name", filename.replace('.json', '')),
                        "filename": filename,
                        "created_at": backup_info.get("created_at"),
                        "size_bytes": os.path.getsize(backup_path),
                        "configurations_count": len(backup_data.get("configurations", {})),
                        "includes_secrets": backup_info.get("includes_secrets", False),
                        "version": backup_info.get("version", "unknown")
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse backup {filename}: {e}")
                    backups.append({
                        "name": filename.replace('.json', ''),
                        "filename": filename,
                        "created_at": None,
                        "size_bytes": os.path.getsize(backup_path),
                        "configurations_count": 0,
                        "includes_secrets": False,
                        "version": "unknown",
                        "error": "Failed to parse backup file"
                    })

        # Sort by creation time, newest first
        backups.sort(key=lambda x: x.get("created_at") or "", reverse=True)

        return jsonify({
            "success": True,
            "data": {
                "backups": backups,
                "total": len(backups)
            }
        })

    except Exception as e:
        logger.error(f"Error listing configuration backups: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/backup/<backup_name>', methods=['GET'])
async def get_config_backup(backup_name):
    """Get details of a specific backup."""
    try:
        backup_dir = os.path.join(_cache_dir, 'backups')
        backup_file = os.path.join(backup_dir, f"{backup_name}.json")

        if not os.path.exists(backup_file):
            return jsonify({"success": False, "error": "Backup not found"}), 404

        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        # Mask secrets if they exist
        masked_data = json.loads(json.dumps(backup_data))  # Deep copy
        if "configurations" in masked_data:
            for config_name, config_data in masked_data["configurations"].items():
                if isinstance(config_data, dict):
                    for key, value in config_data.items():
                        if isinstance(value, dict):
                            for sub_key in list(value.keys()):
                                if 'key' in sub_key.lower() or 'password' in sub_key.lower() or 'secret' in sub_key.lower():
                                    value[sub_key] = "***"
                        elif 'key' in key.lower() or 'password' in key.lower() or 'secret' in key.lower():
                            config_data[key] = "***"

        return jsonify({
            "success": True,
            "data": masked_data
        })

    except Exception as e:
        logger.error(f"Error getting configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/restore/<backup_name>', methods=['POST'])
async def restore_config_backup(backup_name):
    """Restore configuration from a backup."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        dry_run = data.get('dry_run', False)
        selected_configs = data.get('configurations', [])

        backup_dir = os.path.join(_cache_dir, 'backups')
        backup_file = os.path.join(backup_dir, f"{backup_name}.json")

        if not os.path.exists(backup_file):
            return jsonify({"success": False, "error": "Backup not found"}), 404

        with open(backup_file, 'r') as f:
            backup_data = json.load(f)

        configurations = backup_data.get("configurations", {})

        if not configurations:
            return jsonify({"success": False, "error": "No configurations found in backup"}), 400

        # Filter configurations if specific ones were selected
        if selected_configs:
            configurations = {k: v for k, v in configurations.items() if k in selected_configs}

        restoration_plan = {}
        warnings = []

        # Plan restoration for each configuration
        for config_name, config_data in configurations.items():
            if config_name == "tool_selector":
                restoration_plan[config_name] = {
                    "type": "environment_variables",
                    "changes": [
                        {"var": "MAX_TOTAL_TOOLS", "current": os.getenv('MAX_TOTAL_TOOLS', '30'), "new": str(config_data["tool_limits"]["max_total_tools"])},
                        {"var": "MAX_MCP_TOOLS", "current": os.getenv('MAX_MCP_TOOLS', '20'), "new": str(config_data["tool_limits"]["max_mcp_tools"])},
                        {"var": "MIN_MCP_TOOLS", "current": os.getenv('MIN_MCP_TOOLS', '7'), "new": str(config_data["tool_limits"]["min_mcp_tools"])},
                        {"var": "DEFAULT_DROP_RATE", "current": os.getenv('DEFAULT_DROP_RATE', '0.6'), "new": str(config_data["behavior"]["default_drop_rate"])},
                        {"var": "EXCLUDE_LETTA_CORE_TOOLS", "current": os.getenv('EXCLUDE_LETTA_CORE_TOOLS', 'true'), "new": str(config_data["behavior"]["exclude_letta_core_tools"]).lower()},
                    ]
                }

            elif config_name == "ollama":
                restoration_plan[config_name] = {
                    "type": "environment_variables",
                    "changes": [
                        {"var": "OLLAMA_EMBEDDING_HOST", "current": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'), "new": config_data["connection"]["host"]},
                        {"var": "OLLAMA_PORT", "current": os.getenv('OLLAMA_PORT', '11434'), "new": str(config_data["connection"]["port"])},
                        {"var": "OLLAMA_EMBEDDING_MODEL", "current": os.getenv('OLLAMA_EMBEDDING_MODEL', ''), "new": config_data["embedding"]["model"]},
                        {"var": "USE_OLLAMA_EMBEDDINGS", "current": os.getenv('USE_OLLAMA_EMBEDDINGS', 'false'), "new": str(config_data["embedding"]["enabled"]).lower()},
                    ]
                }

            elif config_name == "weaviate":
                restoration_plan[config_name] = {
                    "type": "environment_variables",
                    "changes": [
                        {"var": "WEAVIATE_URL", "current": os.getenv('WEAVIATE_URL', ''), "new": config_data["connection"]["url"]},
                        {"var": "WEAVIATE_TIMEOUT", "current": os.getenv('WEAVIATE_TIMEOUT', '30'), "new": str(config_data["connection"]["timeout"])},
                        {"var": "WEAVIATE_ALPHA", "current": os.getenv('WEAVIATE_ALPHA', '0.75'), "new": str(config_data["search"]["alpha"])},
                    ]
                }

                if "api_key" in config_data["connection"]:
                    restoration_plan[config_name]["changes"].append({
                        "var": "WEAVIATE_API_KEY", "current": "***" if os.getenv('WEAVIATE_API_KEY') else None, "new": "***"
                    })

        if dry_run:
            return jsonify({
                "success": True,
                "message": "Restoration plan generated (dry run)",
                "data": {
                    "backup_name": backup_name,
                    "backup_date": backup_data.get("backup_info", {}).get("created_at"),
                    "configurations_to_restore": len(restoration_plan),
                    "restoration_plan": restoration_plan,
                    "warnings": warnings
                }
            })

        logger.info(f"Configuration restoration requested for backup: {backup_name}")
        logger.info(f"Restoration plan: {restoration_plan}")

        return jsonify({
            "success": True,
            "message": "Configuration restored successfully (simulation)",
            "data": {
                "backup_name": backup_name,
                "restored_configurations": list(restoration_plan.keys()),
                "total_changes": sum(len(plan.get("changes", [])) for plan in restoration_plan.values()),
                "warnings": warnings + ["Restoration is currently simulated - actual environment variable updates not implemented"]
            }
        })

    except Exception as e:
        logger.error(f"Error restoring configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/backup/<backup_name>', methods=['DELETE'])
async def delete_config_backup(backup_name):
    """Delete a configuration backup."""
    try:
        backup_dir = os.path.join(_cache_dir, 'backups')
        backup_file = os.path.join(backup_dir, f"{backup_name}.json")

        if not os.path.exists(backup_file):
            return jsonify({"success": False, "error": "Backup not found"}), 404

        os.remove(backup_file)
        logger.info(f"Configuration backup deleted: {backup_file}")

        return jsonify({
            "success": True,
            "message": "Backup deleted successfully",
            "data": {
                "backup_name": backup_name,
                "deleted_at": datetime.now().isoformat()
            }
        })

    except Exception as e:
        logger.error(f"Error deleting configuration backup: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Configuration Save and Reset Endpoints
# =============================================================================

@backup_bp.route('/save', methods=['POST'])
async def save_current_configuration():
    """Save current configuration state with optional name and description."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        config_name = data.get('name', f"config_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        description = data.get('description', 'Manual configuration save')
        include_secrets = data.get('include_secrets', False)

        # Collect all current configurations
        current_config = {}

        # Collect Tool Selector configuration
        try:
            tool_selector_config = {
                "tool_limits": {
                    "max_total_tools": int(os.getenv('MAX_TOTAL_TOOLS', '30')),
                    "max_mcp_tools": int(os.getenv('MAX_MCP_TOOLS', '20')),
                    "min_mcp_tools": int(os.getenv('MIN_MCP_TOOLS', '7'))
                },
                "behavior": {
                    "default_drop_rate": float(os.getenv('DEFAULT_DROP_RATE', '0.6')),
                    "exclude_letta_core_tools": os.getenv('EXCLUDE_LETTA_CORE_TOOLS', 'true').lower() == 'true',
                    "exclude_official_tools": os.getenv('EXCLUDE_OFFICIAL_TOOLS', 'true').lower() == 'true',
                    "manage_only_mcp_tools": os.getenv('MANAGE_ONLY_MCP_TOOLS', 'true').lower() == 'true'
                }
            }
            current_config["tool_selector"] = tool_selector_config
        except Exception as e:
            logger.warning(f"Failed to collect tool selector config: {e}")

        # Collect Embedding configuration
        try:
            embedding_config = {
                "provider": os.getenv('EMBEDDING_PROVIDER', 'openai'),
                "openai": {
                    "model": os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
                    "api_key": "***MASKED***" if not include_secrets else os.getenv('OPENAI_API_KEY', '')
                },
                "ollama": {
                    "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                    "port": int(os.getenv('OLLAMA_PORT', '11434')),
                    "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text'),
                    "timeout": int(os.getenv('OLLAMA_TIMEOUT', '30'))
                }
            }
            current_config["embedding"] = embedding_config
        except Exception as e:
            logger.warning(f"Failed to collect embedding config: {e}")

        # Collect Weaviate configuration
        try:
            weaviate_config = {
                "url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/'),
                "batch_size": int(os.getenv('WEAVIATE_BATCH_SIZE', '100')),
                "timeout": int(os.getenv('WEAVIATE_TIMEOUT', '60')),
                "hybrid_search_alpha": float(os.getenv('WEAVIATE_HYBRID_ALPHA', '0.75'))
            }
            current_config["weaviate"] = weaviate_config
        except Exception as e:
            logger.warning(f"Failed to collect weaviate config: {e}")

        # Collect Letta API configuration
        try:
            letta_config = {
                "url": os.getenv('LETTA_API_URL', 'https://letta.example.com/v1'),
                "password": "***MASKED***" if not include_secrets else os.getenv('LETTA_PASSWORD', ''),
                "timeout": int(os.getenv('LETTA_TIMEOUT', '30'))
            }
            current_config["letta_api"] = letta_config
        except Exception as e:
            logger.warning(f"Failed to collect letta config: {e}")

        # Create save record
        save_record = {
            "name": config_name,
            "description": description,
            "saved_at": datetime.now().isoformat(),
            "include_secrets": include_secrets,
            "configurations": current_config,
            "version": "1.0",
            "saved_by": "admin"
        }

        # Save to file
        saves_dir = os.path.join(_cache_dir, 'config_saves')
        os.makedirs(saves_dir, exist_ok=True)

        save_file = os.path.join(saves_dir, f"{config_name}.json")
        with open(save_file, 'w') as f:
            json.dump(save_record, f, indent=2)

        # Log the save action
        if _log_config_change:
            await _log_config_change(
                action="configuration_save",
                config_type="system_wide",
                changes={"config_name": config_name},
                metadata={
                    "description": description,
                    "configurations_saved": len(current_config),
                    "include_secrets": include_secrets
                }
            )

        return jsonify({
            "success": True,
            "data": {
                "config_name": config_name,
                "description": description,
                "saved_at": save_record["saved_at"],
                "configurations_count": len(current_config),
                "file_size_bytes": os.path.getsize(save_file)
            }
        })

    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/saves', methods=['GET'])
async def list_saved_configurations():
    """List all saved configurations."""
    try:
        saves_dir = os.path.join(_cache_dir, 'config_saves')

        if not os.path.exists(saves_dir):
            return jsonify({
                "success": True,
                "data": {
                    "saves": [],
                    "total": 0
                }
            })

        saves = []
        for filename in os.listdir(saves_dir):
            if filename.endswith('.json'):
                save_path = os.path.join(saves_dir, filename)
                try:
                    with open(save_path, 'r') as f:
                        save_data = json.load(f)

                    saves.append({
                        "name": save_data.get("name"),
                        "description": save_data.get("description"),
                        "saved_at": save_data.get("saved_at"),
                        "configurations_count": len(save_data.get("configurations", {})),
                        "include_secrets": save_data.get("include_secrets", False),
                        "version": save_data.get("version", "unknown"),
                        "file_size_bytes": os.path.getsize(save_path)
                    })
                except Exception as e:
                    logger.warning(f"Failed to read save file {filename}: {e}")
                    continue

        # Sort by saved_at descending
        saves.sort(key=lambda x: x.get("saved_at", ""), reverse=True)

        return jsonify({
            "success": True,
            "data": {
                "saves": saves,
                "total": len(saves)
            }
        })

    except Exception as e:
        logger.error(f"Error listing saved configurations: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/reset', methods=['POST'])
async def reset_configuration():
    """Reset configuration to defaults or to a specific saved state."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        reset_type = data.get('type', 'defaults')
        config_name = data.get('config_name')
        sections = data.get('sections', [])
        dry_run = data.get('dry_run', False)

        reset_actions = []
        warnings = []

        if reset_type == 'defaults':
            # Reset to default values
            default_configs = {
                "tool_selector": {
                    "MAX_TOTAL_TOOLS": "30",
                    "MAX_MCP_TOOLS": "20",
                    "MIN_MCP_TOOLS": "7",
                    "DEFAULT_DROP_RATE": "0.6",
                    "EXCLUDE_LETTA_CORE_TOOLS": "true",
                    "EXCLUDE_OFFICIAL_TOOLS": "true",
                    "MANAGE_ONLY_MCP_TOOLS": "true"
                },
                "embedding": {
                    "EMBEDDING_PROVIDER": "openai",
                    "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
                    "OLLAMA_EMBEDDING_HOST": "192.168.50.80",
                    "OLLAMA_PORT": "11434",
                    "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
                    "OLLAMA_TIMEOUT": "30"
                },
                "weaviate": {
                    "WEAVIATE_URL": "http://weaviate:8080/",
                    "WEAVIATE_BATCH_SIZE": "100",
                    "WEAVIATE_TIMEOUT": "60",
                    "WEAVIATE_HYBRID_ALPHA": "0.75"
                },
                "letta_api": {
                    "LETTA_API_URL": "https://letta.example.com/v1",
                    "LETTA_TIMEOUT": "30"
                }
            }

            # Filter by sections if specified
            if sections:
                filtered_configs = {}
                for section in sections:
                    if section in default_configs:
                        filtered_configs[section] = default_configs[section]
                default_configs = filtered_configs

            # Apply defaults (simulate)
            for section, config in default_configs.items():
                for key, value in config.items():
                    reset_actions.append({
                        "action": "reset_to_default",
                        "section": section,
                        "key": key,
                        "new_value": value,
                        "previous_value": os.getenv(key, "not_set")
                    })

        elif reset_type == 'saved':
            if not config_name:
                return jsonify({"success": False, "error": "config_name required for saved reset"}), 400

            # Load saved configuration
            saves_dir = os.path.join(_cache_dir, 'config_saves')
            save_file = os.path.join(saves_dir, f"{config_name}.json")

            if not os.path.exists(save_file):
                return jsonify({"success": False, "error": f"Saved configuration '{config_name}' not found"}), 404

            with open(save_file, 'r') as f:
                save_data = json.load(f)

            saved_configs = save_data.get("configurations", {})

            # Filter by sections if specified
            if sections:
                filtered_configs = {}
                for section in sections:
                    if section in saved_configs:
                        filtered_configs[section] = saved_configs[section]
                saved_configs = filtered_configs

            # Apply saved configuration
            for section, config in saved_configs.items():
                reset_actions.append({
                    "action": "reset_to_saved",
                    "section": section,
                    "config_name": config_name,
                    "config": config
                })
        else:
            return jsonify({"success": False, "error": f"Invalid reset type: {reset_type}"}), 400

        if not dry_run and _log_config_change:
            await _log_config_change(
                action="configuration_reset",
                config_type="system_wide",
                changes={
                    "reset_type": reset_type,
                    "config_name": config_name,
                    "sections": sections
                },
                metadata={
                    "actions_count": len(reset_actions),
                    "warnings": warnings
                }
            )

        return jsonify({
            "success": True,
            "data": {
                "reset_type": reset_type,
                "config_name": config_name,
                "sections": sections or "all",
                "actions": reset_actions,
                "warnings": warnings,
                "dry_run": dry_run
            }
        })

    except Exception as e:
        logger.error(f"Error resetting configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/saves/<config_name>', methods=['DELETE'])
async def delete_saved_configuration(config_name):
    """Delete a saved configuration."""
    try:
        saves_dir = os.path.join(_cache_dir, 'config_saves')
        save_file = os.path.join(saves_dir, f"{config_name}.json")

        if not os.path.exists(save_file):
            return jsonify({"success": False, "error": f"Saved configuration '{config_name}' not found"}), 404

        # Get file info before deletion
        file_size = os.path.getsize(save_file)

        # Delete the file
        os.remove(save_file)

        # Log the deletion
        if _log_config_change:
            await _log_config_change(
                action="configuration_save_deleted",
                config_type="system_wide",
                changes={"config_name": config_name},
                metadata={"file_size_bytes": file_size}
            )

        return jsonify({
            "success": True,
            "data": {
                "config_name": config_name,
                "deleted_at": datetime.now().isoformat(),
                "file_size_bytes": file_size
            }
        })

    except Exception as e:
        logger.error(f"Error deleting saved configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Configuration Validation Endpoints
# =============================================================================

@backup_bp.route('/validate', methods=['POST'])
async def validate_configuration():
    """Validate configuration values in real-time."""
    try:
        data = await request.get_json()
        config_type = data.get('config_type')
        field = data.get('field')
        value = data.get('value')
        context = data.get('context', {})

        if _perform_configuration_validation:
            validation_result = await _perform_configuration_validation(config_type, field, value, context)
        else:
            validation_result = {"valid": True, "message": "Validation not configured"}

        return jsonify({
            "success": True,
            "data": validation_result
        })

    except Exception as e:
        logger.error(f"Error validating configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/validate/connection', methods=['POST'])
async def validate_connection():
    """Test connection with provided configuration values."""
    try:
        data = await request.get_json()
        service_type = data.get('service_type')
        config = data.get('config', {})

        if _test_service_connection:
            connection_result = await _test_service_connection(service_type, config)
        else:
            connection_result = {"connected": False, "message": "Connection testing not configured"}

        return jsonify({
            "success": True,
            "data": connection_result
        })

    except Exception as e:
        logger.error(f"Error testing connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/validate/bulk', methods=['POST'])
async def validate_bulk_configuration():
    """Validate multiple configuration values at once."""
    try:
        data = await request.get_json()
        validations = data.get('validations', [])

        results = []
        for validation in validations:
            try:
                if _perform_configuration_validation:
                    result = await _perform_configuration_validation(
                        validation.get('config_type'),
                        validation.get('field'),
                        validation.get('value'),
                        validation.get('context', {})
                    )
                else:
                    result = {"valid": True, "message": "Validation not configured"}
                results.append({
                    "config_type": validation.get('config_type'),
                    "field": validation.get('field'),
                    "result": result
                })
            except Exception as e:
                results.append({
                    "config_type": validation.get('config_type'),
                    "field": validation.get('field'),
                    "result": {"valid": False, "error": str(e)}
                })

        # Summary
        valid_count = sum(1 for r in results if r.get("result", {}).get("valid", False))

        return jsonify({
            "success": True,
            "data": {
                "results": results,
                "summary": {
                    "total": len(results),
                    "valid": valid_count,
                    "invalid": len(results) - valid_count
                }
            }
        })

    except Exception as e:
        logger.error(f"Error validating bulk configuration: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Configuration Audit Endpoints
# =============================================================================

@backup_bp.route('/audit', methods=['GET'])
async def get_config_audit_logs():
    """Get configuration change audit logs."""
    try:
        # Query parameters
        limit = int(request.args.get('limit', 50))
        offset = int(request.args.get('offset', 0))
        config_type = request.args.get('config_type')
        action = request.args.get('action')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')

        audit_file = os.path.join(_cache_dir, 'config_audit.json')

        if not os.path.exists(audit_file):
            return jsonify({
                "success": True,
                "data": {
                    "logs": [],
                    "total": 0,
                    "filtered": 0,
                    "offset": offset,
                    "limit": limit
                }
            })

        with open(audit_file, 'r') as f:
            all_logs = []
            for line in f:
                try:
                    all_logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Apply filters
        filtered_logs = all_logs

        if config_type:
            filtered_logs = [log for log in filtered_logs if log.get('config_type') == config_type]

        if action:
            filtered_logs = [log for log in filtered_logs if log.get('action') == action]

        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                filtered_logs = [log for log in filtered_logs
                               if datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00')) >= start_dt]
            except ValueError:
                pass

        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                filtered_logs = [log for log in filtered_logs
                               if datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00')) <= end_dt]
            except ValueError:
                pass

        # Sort by timestamp, newest first
        filtered_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

        # Apply pagination
        total_filtered = len(filtered_logs)
        paginated_logs = filtered_logs[offset:offset + limit]

        return jsonify({
            "success": True,
            "data": {
                "logs": paginated_logs,
                "total": len(all_logs),
                "filtered": total_filtered,
                "offset": offset,
                "limit": limit
            }
        })

    except Exception as e:
        logger.error(f"Error getting config audit logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/audit/stats', methods=['GET'])
async def get_config_audit_stats():
    """Get audit log statistics."""
    try:
        audit_file = os.path.join(_cache_dir, 'config_audit.json')

        if not os.path.exists(audit_file):
            return jsonify({
                "success": True,
                "data": {
                    "total_entries": 0,
                    "config_types": {},
                    "actions": {},
                    "recent_activity": [],
                    "last_change": None
                }
            })

        with open(audit_file, 'r') as f:
            all_logs = []
            for line in f:
                try:
                    all_logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

        # Calculate statistics
        config_types = {}
        actions = {}

        for log in all_logs:
            ct = log.get('config_type', 'unknown')
            act = log.get('action', 'unknown')

            config_types[ct] = config_types.get(ct, 0) + 1
            actions[act] = actions.get(act, 0) + 1

        # Get recent activity (last 10 entries)
        recent_logs = sorted(all_logs, key=lambda x: x.get('timestamp', ''), reverse=True)[:10]

        last_change = recent_logs[0] if recent_logs else None

        return jsonify({
            "success": True,
            "data": {
                "total_entries": len(all_logs),
                "config_types": config_types,
                "actions": actions,
                "recent_activity": recent_logs,
                "last_change": last_change
            }
        })

    except Exception as e:
        logger.error(f"Error getting config audit stats: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@backup_bp.route('/audit/clear', methods=['POST'])
async def clear_config_audit_logs():
    """Clear all audit logs (admin operation)."""
    try:
        data = await request.get_json() if await request.get_data() else {}
        confirm = data.get('confirm', False)

        if not confirm:
            return jsonify({
                "success": False,
                "error": "Confirmation required. Set 'confirm': true in request body."
            }), 400

        audit_file = os.path.join(_cache_dir, 'config_audit.json')
        backup_file = None

        # Create backup before clearing
        if os.path.exists(audit_file):
            backup_file = f"{audit_file}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(audit_file, backup_file)

            # Log the clearing action
            if _log_config_change:
                await _log_config_change(
                    action="clear_audit_logs",
                    config_type="audit_system",
                    changes={"backup_file": backup_file},
                    user_info={"action": "admin_clear", "timestamp": datetime.now().isoformat()}
                )

        return jsonify({
            "success": True,
            "message": "Audit logs cleared successfully",
            "data": {
                "cleared_at": datetime.now().isoformat(),
                "backup_created": backup_file
            }
        })

    except Exception as e:
        logger.error(f"Error clearing config audit logs: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
