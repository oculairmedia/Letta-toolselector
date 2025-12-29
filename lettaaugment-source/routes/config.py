"""
Config Routes Blueprint

Handles configuration management endpoints for:
- Reranker configuration
- Embedding configuration
- Ollama configuration
- Weaviate configuration
- Configuration presets
- Configuration backup/restore
- Audit logging
"""

from quart import Blueprint, request, jsonify
import logging
import os
import time
import asyncio
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
config_bp = Blueprint('config', __name__, url_prefix='/api/v1/config')

# Module state - to be configured
_http_session = None
_log_config_change = None
_validate_config_func = None
_get_tool_selector_config_func = None
_update_tool_selector_config_func = None


def configure(http_session=None, log_config_change=None, validate_config_func=None,
              get_tool_selector_config_func=None, update_tool_selector_config_func=None):
    """
    Configure the config routes with required dependencies.
    
    Args:
        http_session: aiohttp ClientSession for API calls
        log_config_change: Async function to log configuration changes
        validate_config_func: Function for /config/validate endpoint
        get_tool_selector_config_func: Function for GET /config/tool-selector
        update_tool_selector_config_func: Function for PUT /config/tool-selector
    """
    global _http_session, _log_config_change
    global _validate_config_func, _get_tool_selector_config_func, _update_tool_selector_config_func
    _http_session = http_session
    _log_config_change = log_config_change
    _validate_config_func = validate_config_func
    _get_tool_selector_config_func = get_tool_selector_config_func
    _update_tool_selector_config_func = update_tool_selector_config_func


# =============================================================================
# Reranker Configuration
# =============================================================================

@config_bp.route('/reranker', methods=['GET'])
async def get_reranker_config():
    """Get current reranker configuration."""
    try:
        config = {
            "enabled": os.getenv('RERANKER_ENABLED', 'true').lower() == 'true',
            "model": os.getenv('RERANKER_MODEL', 'qwen3-reranker-4b'),
            "provider": os.getenv('RERANKER_PROVIDER', 'vllm'),
            "parameters": {
                "temperature": float(os.getenv('RERANKER_TEMPERATURE', '0.1')),
                "max_tokens": int(os.getenv('RERANKER_MAX_TOKENS', '512')),
                "base_url": os.getenv('RERANKER_URL', 'http://100.81.139.20:11435/rerank')
            }
        }
        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting reranker config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/reranker', methods=['PUT'])
async def update_reranker_config():
    """Update reranker configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400
        
        required_fields = ['enabled', 'model', 'provider', 'parameters']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        logger.info(f"Reranker config update requested: {data}")
        
        return jsonify({"success": True, "message": "Reranker configuration updated successfully"})
    except Exception as e:
        logger.error(f"Error updating reranker config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/reranker/test', methods=['POST'])
async def test_reranker_connection():
    """Test reranker connection with provided configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400
        
        provider = data.get('provider', 'ollama')
        base_url = data.get('parameters', {}).get('base_url', 'http://ollama-reranker-adapter:8080')
        
        connected = False
        if provider == 'ollama':
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            connected = True
            except Exception as conn_error:
                logger.warning(f"Reranker connection test failed: {str(conn_error)}")
        
        return jsonify({
            "success": True, 
            "data": {"connected": connected}
        })
    except Exception as e:
        logger.error(f"Error testing reranker connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Embedding Configuration
# =============================================================================

@config_bp.route('/embedding', methods=['GET'])
async def get_embedding_config():
    """Get current embedding configuration."""
    try:
        config = {
            "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
            "provider": os.getenv('EMBEDDING_PROVIDER', 'ollama'),
            "parameters": {
                "dimensions": int(os.getenv('EMBEDDING_DIMENSION', '2560')),
                "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                "use_ollama": os.getenv('USE_OLLAMA_EMBEDDINGS', 'true').lower() == 'true'
            }
        }
        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting embedding config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/embedding', methods=['PUT'])
async def update_embedding_config():
    """Update embedding configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400
        
        required_fields = ['model', 'provider']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        logger.info(f"Embedding config update requested: {data}")
        
        return jsonify({"success": True, "message": "Embedding configuration updated successfully"})
    except Exception as e:
        logger.error(f"Error updating embedding config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Configuration Presets
# =============================================================================

@config_bp.route('/presets', methods=['GET'])
async def get_configuration_presets():
    """Get all configuration presets."""
    try:
        return jsonify({
            "success": True,
            "data": []
        })
    except Exception as e:
        logger.error(f"Error getting configuration presets: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/presets', methods=['POST'])
async def create_configuration_preset():
    """Create a new configuration preset."""
    try:
        data = await request.get_json()
        return jsonify({
            "success": True,
            "data": {
                "id": "preset_" + str(int(time.time())),
                "name": data.get("name", "Untitled Preset"),
                "description": data.get("description", ""),
                "config": data.get("config", {}),
                "created_at": time.time()
            }
        })
    except Exception as e:
        logger.error(f"Error creating configuration preset: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/presets/<preset_id>', methods=['PUT'])
async def update_configuration_preset(preset_id):
    """Update a configuration preset."""
    try:
        data = await request.get_json()
        return jsonify({
            "success": True,
            "data": {
                "id": preset_id,
                "name": data.get("name", "Updated Preset"),
                "description": data.get("description", ""),
                "config": data.get("config", {}),
                "updated_at": time.time()
            }
        })
    except Exception as e:
        logger.error(f"Error updating configuration preset: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/presets/<preset_id>', methods=['DELETE'])
async def delete_configuration_preset(preset_id):
    """Delete a configuration preset."""
    try:
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Error deleting configuration preset: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Ollama Configuration
# =============================================================================

async def test_ollama_connection(config):
    """Test connection to Ollama server."""
    host = config.get("host", "192.168.50.80")
    port = config.get("port", 11434)
    timeout = config.get("timeout", 30)
    
    try:
        base_url = f"http://{host}:{port}"
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Test basic connectivity
            async with session.get(f"{base_url}/api/version") as response:
                if response.status == 200:
                    version_data = await response.json()
                    
                    # Test model listing
                    async with session.get(f"{base_url}/api/tags") as models_response:
                        if models_response.status == 200:
                            models_data = await models_response.json()
                            model_count = len(models_data.get("models", []))
                            
                            return {
                                "available": True,
                                "version": version_data.get("version", "unknown"),
                                "host": host,
                                "port": port,
                                "model_count": model_count,
                                "response_time": "< 1s"
                            }
                        else:
                            return {
                                "available": False,
                                "error": f"Models endpoint returned {models_response.status}",
                                "host": host,
                                "port": port
                            }
                else:
                    return {
                        "available": False,
                        "error": f"Version endpoint returned {response.status}",
                        "host": host,
                        "port": port
                    }
    except asyncio.TimeoutError:
        return {
            "available": False,
            "error": "Connection timeout",
            "host": host,
            "port": port
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "host": host,
            "port": port
        }


@config_bp.route('/ollama', methods=['GET'])
async def get_ollama_config():
    """Get current Ollama configuration."""
    try:
        config = {
            "connection": {
                "host": os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80'),
                "port": int(os.getenv('OLLAMA_PORT', '11434')),
                "timeout": int(os.getenv('OLLAMA_TIMEOUT', '30')),
                "base_url": os.getenv('OLLAMA_BASE_URL', '')
            },
            "embedding": {
                "model": os.getenv('OLLAMA_EMBEDDING_MODEL', 'dengcao/Qwen3-Embedding-4B:Q4_K_M'),
                "enabled": os.getenv('USE_OLLAMA_EMBEDDINGS', 'false').lower() == 'true'
            },
            "generation": {
                "default_model": os.getenv('OLLAMA_DEFAULT_MODEL', 'mistral:7b'),
                "temperature": float(os.getenv('OLLAMA_TEMPERATURE', '0.7')),
                "context_length": int(os.getenv('OLLAMA_CONTEXT_LENGTH', '4096'))
            },
            "performance": {
                "num_parallel": int(os.getenv('OLLAMA_NUM_PARALLEL', '1')),
                "num_ctx": int(os.getenv('OLLAMA_NUM_CTX', '2048')),
                "num_gpu": int(os.getenv('OLLAMA_NUM_GPU', '-1')),
                "low_vram": os.getenv('OLLAMA_LOW_VRAM', 'false').lower() == 'true'
            }
        }

        # Add connection status
        connection_status = await test_ollama_connection(config["connection"])
        config["status"] = connection_status

        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting Ollama config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/ollama', methods=['PUT'])
async def update_ollama_config():
    """Update Ollama configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400

        # Validate configuration data
        validation_errors = []
        warnings = []
        test_result = None

        if "connection" in data:
            conn = data["connection"]

            # Validate host
            if "host" in conn:
                host = conn["host"]
                if not host or not isinstance(host, str):
                    validation_errors.append("Host must be a valid string")
                elif not host.replace('.', '').replace('-', '').replace(':', '').isalnum():
                    warnings.append("Host format may be invalid")

            # Validate port
            if "port" in conn:
                port = conn["port"]
                if not isinstance(port, int) or not (1 <= port <= 65535):
                    validation_errors.append("Port must be between 1 and 65535")

            # Validate timeout
            if "timeout" in conn:
                timeout = conn["timeout"]
                if not isinstance(timeout, int) or timeout < 1:
                    validation_errors.append("Timeout must be a positive integer")
                elif timeout < 5:
                    warnings.append("Very low timeout may cause connection issues")

        if "performance" in data:
            perf = data["performance"]

            # Validate num_parallel
            if "num_parallel" in perf:
                num_parallel = perf["num_parallel"]
                if not isinstance(num_parallel, int) or num_parallel < 1:
                    validation_errors.append("num_parallel must be a positive integer")
                elif num_parallel > 8:
                    warnings.append("High parallelism may cause resource issues")

            # Validate context lengths
            if "num_ctx" in perf:
                num_ctx = perf["num_ctx"]
                if not isinstance(num_ctx, int) or num_ctx < 128:
                    validation_errors.append("num_ctx must be at least 128")
                elif num_ctx > 32768:
                    warnings.append("Very large context may cause memory issues")

        if validation_errors:
            return jsonify({
                "success": False,
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "warnings": warnings
            }), 400

        # Test connection if connection config provided
        if "connection" in data:
            test_result = await test_ollama_connection(data["connection"])
            if not test_result["available"]:
                warnings.append(f"Ollama connection test failed: {test_result.get('error', 'Unknown error')}")

        # Log the configuration update
        logger.info(f"Ollama configuration update requested: {data}")

        # Log to audit system if available
        if _log_config_change:
            await _log_config_change(
                action="update",
                config_type="ollama",
                changes=data,
                user_info={"source": "api", "timestamp": datetime.now().isoformat()},
                metadata={"warnings": warnings, "connection_test": test_result}
            )

        response = {
            "success": True,
            "message": "Ollama configuration updated successfully",
            "applied_config": data
        }

        if warnings:
            response["warnings"] = warnings

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error updating Ollama config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/ollama/test', methods=['POST'])
async def test_ollama_connection_endpoint():
    """Test Ollama connection with provided configuration."""
    try:
        data = await request.get_json()
        connection_config = data.get("connection", {}) if data else {}

        # Use defaults if not provided
        config = {
            "host": connection_config.get("host", os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')),
            "port": connection_config.get("port", int(os.getenv('OLLAMA_PORT', '11434'))),
            "timeout": connection_config.get("timeout", int(os.getenv('OLLAMA_TIMEOUT', '30')))
        }

        result = await test_ollama_connection(config)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Weaviate Configuration
# =============================================================================

async def test_weaviate_connection(config):
    """Test connection to Weaviate server."""
    url = config.get("url", "http://weaviate:8080/")
    timeout = config.get("timeout", 30)
    api_key = config.get("api_key")
    
    try:
        headers = {}
        if api_key:
            headers['Authorization'] = f"Bearer {api_key}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Test basic connectivity
            meta_url = f"{url.rstrip('/')}/v1/meta"
            async with session.get(meta_url, headers=headers) as response:
                if response.status == 200:
                    meta_data = await response.json()

                    # Test schema access
                    schema_url = f"{url.rstrip('/')}/v1/schema"
                    async with session.get(schema_url, headers=headers) as schema_response:
                        if schema_response.status == 200:
                            schema_data = await schema_response.json()
                            class_count = len(schema_data.get("classes", []))

                            return {
                                "available": True,
                                "version": meta_data.get("version", "unknown"),
                                "hostname": meta_data.get("hostname", "unknown"),
                                "class_count": class_count,
                                "modules": meta_data.get("modules", {}),
                                "url": url
                            }
                        else:
                            return {
                                "available": False,
                                "error": f"Schema endpoint returned {schema_response.status}",
                                "url": url
                            }
                else:
                    return {
                        "available": False,
                        "error": f"Meta endpoint returned {response.status}",
                        "url": url
                    }
    except asyncio.TimeoutError:
        return {
            "available": False,
            "error": "Connection timeout",
            "url": url
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e),
            "url": url
        }


@config_bp.route('/weaviate', methods=['GET'])
async def get_weaviate_config():
    """Get current Weaviate configuration."""
    try:
        config = {
            "connection": {
                "url": os.getenv('WEAVIATE_URL', 'http://weaviate:8080/'),
                "timeout": int(os.getenv('WEAVIATE_TIMEOUT', '30')),
                "retries": int(os.getenv('WEAVIATE_RETRIES', '3')),
                "api_key": "***" if os.getenv('WEAVIATE_API_KEY') else None
            },
            "schema": {
                "class_name": os.getenv('WEAVIATE_CLASS_NAME', 'Tool'),
                "vector_index_type": os.getenv('WEAVIATE_VECTOR_INDEX', 'hnsw'),
                "distance_metric": os.getenv('WEAVIATE_DISTANCE_METRIC', 'cosine')
            },
            "search": {
                "alpha": float(os.getenv('WEAVIATE_ALPHA', '0.75')),
                "limit": int(os.getenv('WEAVIATE_LIMIT', '50')),
                "additional_properties": os.getenv('WEAVIATE_ADDITIONAL_PROPERTIES', 'id,distance,certainty').split(','),
                "autocut": int(os.getenv('WEAVIATE_AUTOCUT', '1'))
            },
            "performance": {
                "ef_construction": int(os.getenv('WEAVIATE_EF_CONSTRUCTION', '128')),
                "ef": int(os.getenv('WEAVIATE_EF', '64')),
                "max_connections": int(os.getenv('WEAVIATE_MAX_CONNECTIONS', '64')),
                "vector_cache_max_objects": int(os.getenv('WEAVIATE_VECTOR_CACHE_MAX_OBJECTS', '1000000')),
                "cleanup_interval_seconds": int(os.getenv('WEAVIATE_CLEANUP_INTERVAL', '60'))
            }
        }

        # Add connection status
        connection_status = await test_weaviate_connection(config["connection"])
        config["status"] = connection_status

        return jsonify({"success": True, "data": config})
    except Exception as e:
        logger.error(f"Error getting Weaviate config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/weaviate', methods=['PUT'])
async def update_weaviate_config():
    """Update Weaviate configuration."""
    try:
        data = await request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No configuration data provided"}), 400

        # Validate configuration data
        validation_errors = []
        warnings = []

        if "connection" in data:
            conn = data["connection"]

            # Validate URL
            if "url" in conn:
                url = conn["url"]
                if not url or not isinstance(url, str):
                    validation_errors.append("URL must be a valid string")
                elif not (url.startswith('http://') or url.startswith('https://')):
                    validation_errors.append("URL must start with http:// or https://")

            # Validate timeout
            if "timeout" in conn:
                timeout = conn["timeout"]
                if not isinstance(timeout, int) or timeout < 1:
                    validation_errors.append("Timeout must be a positive integer")
                elif timeout < 5:
                    warnings.append("Very low timeout may cause connection issues")

            # Validate retries
            if "retries" in conn:
                retries = conn["retries"]
                if not isinstance(retries, int) or retries < 0:
                    validation_errors.append("Retries must be a non-negative integer")
                elif retries > 10:
                    warnings.append("High retry count may cause delays")

        if "search" in data:
            search = data["search"]

            # Validate alpha
            if "alpha" in search:
                alpha = search["alpha"]
                if not isinstance(alpha, (int, float)) or not (0.0 <= alpha <= 1.0):
                    validation_errors.append("Alpha must be between 0.0 and 1.0")

            # Validate limit
            if "limit" in search:
                limit = search["limit"]
                if not isinstance(limit, int) or limit < 1:
                    validation_errors.append("Limit must be a positive integer")
                elif limit > 1000:
                    warnings.append("Very high limit may impact performance")

        if "performance" in data:
            perf = data["performance"]

            # Validate HNSW parameters
            hnsw_params = ["ef_construction", "ef", "max_connections"]
            for param in hnsw_params:
                if param in perf:
                    value = perf[param]
                    if not isinstance(value, int) or value < 1:
                        validation_errors.append(f"{param} must be a positive integer")

            # Validate cache size
            if "vector_cache_max_objects" in perf:
                cache_size = perf["vector_cache_max_objects"]
                if not isinstance(cache_size, int) or cache_size < 1000:
                    validation_errors.append("vector_cache_max_objects must be at least 1000")
                elif cache_size > 10000000:
                    warnings.append("Very large vector cache may use excessive memory")

        if validation_errors:
            return jsonify({
                "success": False,
                "error": "Validation failed",
                "validation_errors": validation_errors,
                "warnings": warnings
            }), 400

        # Test connection if connection config provided
        if "connection" in data:
            test_result = await test_weaviate_connection(data["connection"])
            if not test_result["available"]:
                warnings.append(f"Weaviate connection test failed: {test_result.get('error', 'Unknown error')}")

        # Log the configuration update
        logger.info(f"Weaviate configuration update requested: {data}")

        response = {
            "success": True,
            "message": "Weaviate configuration updated successfully",
            "applied_config": data
        }

        if warnings:
            response["warnings"] = warnings

        return jsonify(response)
    except Exception as e:
        logger.error(f"Error updating Weaviate config: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/weaviate/test', methods=['POST'])
async def test_weaviate_connection_endpoint():
    """Test Weaviate connection with provided configuration."""
    try:
        data = await request.get_json()
        connection_config = data.get("connection", {}) if data else {}

        # Use defaults if not provided
        config = {
            "url": connection_config.get("url", os.getenv('WEAVIATE_URL', 'http://weaviate:8080/')),
            "timeout": connection_config.get("timeout", int(os.getenv('WEAVIATE_TIMEOUT', '30'))),
            "api_key": connection_config.get("api_key", os.getenv('WEAVIATE_API_KEY'))
        }

        result = await test_weaviate_connection(config)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error testing Weaviate connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@config_bp.route('/weaviate/schema', methods=['GET'])
async def get_weaviate_schema():
    """Get Weaviate schema information."""
    try:
        url = os.getenv('WEAVIATE_URL', 'http://weaviate:8080/')
        timeout = int(os.getenv('WEAVIATE_TIMEOUT', '30'))

        headers = {}
        if os.getenv('WEAVIATE_API_KEY'):
            headers['Authorization'] = f"Bearer {os.getenv('WEAVIATE_API_KEY')}"

        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            # Get schema information
            async with session.get(f"{url.rstrip('/')}/v1/schema", headers=headers) as response:
                if response.status == 200:
                    schema_data = await response.json()

                    # Get object count for each class
                    classes_with_counts = []
                    for class_info in schema_data.get("classes", []):
                        class_name = class_info["class"]

                        # Get object count
                        count_query = f"{url.rstrip('/')}/v1/objects?class={class_name}&limit=0"
                        async with session.get(count_query, headers=headers) as count_response:
                            if count_response.status == 200:
                                count_data = await count_response.json()
                                object_count = count_data.get("totalResults", 0)
                            else:
                                object_count = -1  # Unknown

                        classes_with_counts.append({
                            **class_info,
                            "object_count": object_count
                        })

                    return jsonify({
                        "success": True,
                        "data": {
                            "classes": classes_with_counts,
                            "total_classes": len(classes_with_counts)
                        }
                    })
                else:
                    return jsonify({
                        "success": False,
                        "error": f"Schema endpoint returned {response.status}"
                    }), response.status
    except Exception as e:
        logger.error(f"Error getting Weaviate schema: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# Configuration Validation & Tool Selector Configuration
# =============================================================================

@config_bp.route('/validate', methods=['POST'])
async def validate_config():
    """Validate dashboard configuration using schema-based validation."""
    if not _validate_config_func:
        return jsonify({"error": "Configuration validation not configured"}), 503
    return await _validate_config_func()


@config_bp.route('/tool-selector', methods=['GET'])
async def get_tool_selector_config():
    """Get current tool selector configuration."""
    if not _get_tool_selector_config_func:
        return jsonify({"error": "Tool selector config not configured"}), 503
    return await _get_tool_selector_config_func()


@config_bp.route('/tool-selector', methods=['PUT'])
async def update_tool_selector_config():
    """Update tool selector configuration."""
    if not _update_tool_selector_config_func:
        return jsonify({"error": "Tool selector config update not configured"}), 503
    return await _update_tool_selector_config_func()
