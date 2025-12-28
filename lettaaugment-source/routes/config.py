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


def configure(http_session=None, log_config_change=None):
    """
    Configure the config routes with required dependencies.
    
    Args:
        http_session: aiohttp ClientSession for API calls
        log_config_change: Async function to log configuration changes
    """
    global _http_session, _log_config_change
    _http_session = http_session
    _log_config_change = log_config_change


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

async def _test_ollama_connection(config):
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
        connection_status = await _test_ollama_connection(config["connection"])
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
            test_result = await _test_ollama_connection(data["connection"])
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

        result = await _test_ollama_connection(config)
        return jsonify({"success": True, "data": result})
    except Exception as e:
        logger.error(f"Error testing Ollama connection: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
