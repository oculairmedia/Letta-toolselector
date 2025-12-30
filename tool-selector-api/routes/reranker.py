"""
Reranker Routes Blueprint

Handles reranker model registry operations:
- /api/v1/reranker/models/registry - List, register models
- /api/v1/reranker/models/registry/<id> - Update, delete models
- /api/v1/reranker/models/registry/<id>/test - Test model connectivity
"""

from quart import Blueprint, request, jsonify
import logging
import os
import json
import time
import aiohttp
from datetime import datetime

logger = logging.getLogger(__name__)

# Create blueprint
reranker_bp = Blueprint('reranker', __name__, url_prefix='/api/v1/reranker')

# Module state - to be configured
_cache_dir = None


def configure(cache_dir=None):
    """Configure the reranker routes with required dependencies."""
    global _cache_dir
    _cache_dir = cache_dir


def _get_registry_path():
    """Get path to reranker model registry file."""
    return os.path.join(_cache_dir, 'reranker_model_registry.json') if _cache_dir else None


def _load_registry():
    """Load registry from file."""
    registry_path = _get_registry_path()
    if registry_path and os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            return json.load(f)
    return {"models": [], "last_updated": None}


def _save_registry(registry):
    """Save registry to file."""
    registry_path = _get_registry_path()
    if registry_path:
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)


# =============================================================================
# Model Registry Endpoints
# =============================================================================

@reranker_bp.route('/models/registry', methods=['GET'])
async def list_registered_reranker_models():
    """Get all registered reranker models from the registry."""
    try:
        registry = _load_registry()
            
        # Merge with discovered models from providers
        try:
            models_from_discovery = []
            
            # Ollama reranker models discovery
            try:
                ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80') 
                ollama_url = f"http://{ollama_host}:11434"
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            ollama_data = await response.json()
                            for model in ollama_data.get('models', []):
                                model_name = model.get('name', '')
                                # Filter for reranker models
                                if any(keyword in model_name.lower() for keyword in ['rerank', 'cross-encoder', 'bge-rerank', 'colbert']):
                                    models_from_discovery.append({
                                        "id": model_name,
                                        "name": f"Ollama {model_name}",
                                        "provider": "ollama",
                                        "type": "cross-encoder",
                                        "cost_per_1k": 0.0,
                                        "recommended": True,
                                        "size": model.get('size', 0),
                                        "modified_at": model.get('modified_at')
                                    })
                                # Also include general models that can be used for reranking
                                elif any(keyword in model_name.lower() for keyword in ['mistral', 'llama', 'qwen']):
                                    models_from_discovery.append({
                                        "id": model_name,
                                        "name": f"Ollama {model_name} (General)",
                                        "provider": "ollama",
                                        "type": "generative",
                                        "cost_per_1k": 0.0,
                                        "recommended": False,
                                        "size": model.get('size', 0),
                                        "modified_at": model.get('modified_at')
                                    })
            except Exception as e:
                logger.warning(f"Failed to fetch Ollama reranker models for registry: {str(e)}")
            
            # Add fallback models if nothing discovered
            if not models_from_discovery:
                models_from_discovery = [
                    {
                        "id": "mistral:7b",
                        "name": "Mistral 7B (Fallback)",
                        "provider": "ollama",
                        "type": "generative",
                        "cost_per_1k": 0.0,
                        "recommended": True
                    }
                ]
            
            # Update registry with any new discovered models
            existing_ids = {model['id'] for model in registry['models']}
            new_models = []
            
            for discovered_model in models_from_discovery:
                if discovered_model['id'] not in existing_ids:
                    new_models.append({
                        **discovered_model,
                        "registered": True,
                        "validated": False,
                        "last_tested": None,
                        "test_status": "pending",
                        "registry_notes": "Auto-discovered from provider"
                    })
            
            if new_models:
                registry['models'].extend(new_models)
                registry['last_updated'] = datetime.utcnow().isoformat()
                _save_registry(registry)
                    
        except Exception as discovery_error:
            logger.warning(f"Error during model discovery for registry: {str(discovery_error)}")
        
        return jsonify({
            "success": True,
            "data": {
                "models": registry['models'],
                "total": len(registry['models']),
                "last_updated": registry.get('last_updated'),
                "providers": list(set(m["provider"] for m in registry['models'])),
                "types": list(set(m["type"] for m in registry['models'])),
                "validated_count": len([m for m in registry['models'] if m.get('validated', False)]),
                "pending_count": len([m for m in registry['models'] if m.get('test_status') == 'pending'])
            }
        })
        
    except Exception as e:
        logger.error(f"Error listing registered reranker models: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@reranker_bp.route('/models/registry', methods=['POST'])
async def register_reranker_model():
    """Register a new reranker model in the registry."""
    try:
        data = await request.get_json()
        
        # Validate required fields
        required_fields = ['id', 'name', 'provider', 'type']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"}), 400
        
        registry = _load_registry()
        
        # Check if model already exists
        existing_ids = {model['id'] for model in registry['models']}
        if data['id'] in existing_ids:
            return jsonify({"success": False, "error": f"Model {data['id']} already registered"}), 409
        
        # Create new model entry
        new_model = {
            "id": data['id'],
            "name": data['name'],
            "provider": data['provider'],
            "type": data['type'],
            "cost_per_1k": data.get('cost_per_1k', 0.0),
            "recommended": data.get('recommended', False),
            "registered": True,
            "validated": False,
            "last_tested": None,
            "test_status": "pending",
            "registry_notes": data.get('notes', 'Manually registered'),
            "configuration": data.get('configuration', {}),
            "registered_at": datetime.utcnow().isoformat()
        }
        
        # Add optional fields if provided
        optional_fields = ['size', 'modified_at', 'dimensions', 'max_tokens', 'description']
        for field in optional_fields:
            if field in data:
                new_model[field] = data[field]
        
        # Add to registry
        registry['models'].append(new_model)
        registry['last_updated'] = datetime.utcnow().isoformat()
        _save_registry(registry)
        
        logger.info(f"Registered new reranker model: {data['id']}")
        
        return jsonify({
            "success": True,
            "data": new_model,
            "message": f"Model {data['id']} registered successfully"
        })
        
    except Exception as e:
        logger.error(f"Error registering reranker model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@reranker_bp.route('/models/registry/<model_id>', methods=['PUT'])
async def update_registered_model(model_id):
    """Update a registered reranker model."""
    try:
        data = await request.get_json()
        
        registry = _load_registry()
        
        if not registry.get('models'):
            return jsonify({"success": False, "error": "No registry found"}), 404
        
        # Find model to update
        model_to_update = None
        model_index = None
        for i, model in enumerate(registry['models']):
            if model['id'] == model_id:
                model_to_update = model
                model_index = i
                break
        
        if not model_to_update:
            return jsonify({"success": False, "error": f"Model {model_id} not found in registry"}), 404
        
        # Update allowed fields
        updatable_fields = [
            'name', 'recommended', 'registry_notes', 'configuration', 
            'validated', 'test_status', 'description'
        ]
        
        for field in updatable_fields:
            if field in data:
                model_to_update[field] = data[field]
        
        model_to_update['last_updated'] = datetime.utcnow().isoformat()
        registry['models'][model_index] = model_to_update
        registry['last_updated'] = datetime.utcnow().isoformat()
        
        _save_registry(registry)
        
        logger.info(f"Updated registered reranker model: {model_id}")
        
        return jsonify({
            "success": True,
            "data": model_to_update,
            "message": f"Model {model_id} updated successfully"
        })
        
    except Exception as e:
        logger.error(f"Error updating registered model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@reranker_bp.route('/models/registry/<model_id>/test', methods=['POST'])
async def test_registered_model(model_id):
    """Test connectivity and functionality of a registered reranker model."""
    try:
        registry = _load_registry()
        
        if not registry.get('models'):
            return jsonify({"success": False, "error": "No registry found"}), 404
        
        # Find model to test
        model_to_test = None
        model_index = None
        for i, model in enumerate(registry['models']):
            if model['id'] == model_id:
                model_to_test = model
                model_index = i
                break
        
        if not model_to_test:
            return jsonify({"success": False, "error": f"Model {model_id} not found in registry"}), 404
        
        test_results = {
            "model_id": model_id,
            "test_timestamp": datetime.utcnow().isoformat(),
            "connectivity": False,
            "functionality": False,
            "latency_ms": None,
            "error": None,
            "details": {}
        }
        
        try:
            # Test based on provider
            if model_to_test['provider'] == 'ollama':
                # Test Ollama model connectivity
                ollama_host = os.getenv('OLLAMA_EMBEDDING_HOST', '192.168.50.80')
                ollama_url = f"http://{ollama_host}:11434"
                
                start_time = time.time()
                
                async with aiohttp.ClientSession() as session:
                    # Check if model is available
                    async with session.get(f"{ollama_url}/api/tags", timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            models_data = await response.json()
                            available_models = [m['name'] for m in models_data.get('models', [])]
                            
                            if model_id in available_models:
                                test_results["connectivity"] = True
                                
                                # Test basic functionality with a simple rerank task
                                test_payload = {
                                    "model": model_id,
                                    "prompt": "Query: test search\nDocument: This is a test document for reranking.\nRelevance score (0-1):",
                                    "options": {"temperature": 0.1, "num_predict": 10}
                                }
                                
                                async with session.post(f"{ollama_url}/api/generate", 
                                                       json=test_payload, 
                                                       timeout=aiohttp.ClientTimeout(total=30)) as test_response:
                                    if test_response.status == 200:
                                        test_results["functionality"] = True
                                        test_results["latency_ms"] = int((time.time() - start_time) * 1000)
                            else:
                                test_results["error"] = f"Model {model_id} not found in Ollama"
                        else:
                            test_results["error"] = f"Failed to connect to Ollama: HTTP {response.status}"
                            
        except Exception as test_error:
            test_results["error"] = str(test_error)
        
        # Update model in registry with test results
        model_to_test["last_tested"] = test_results["test_timestamp"]
        if test_results["connectivity"] and test_results["functionality"]:
            model_to_test["test_status"] = "passed"
            model_to_test["validated"] = True
        else:
            model_to_test["test_status"] = "failed"
            model_to_test["validated"] = False
        
        model_to_test["test_results"] = test_results
        registry['models'][model_index] = model_to_test
        registry['last_updated'] = datetime.utcnow().isoformat()
        
        _save_registry(registry)
        
        logger.info(f"Tested registered reranker model {model_id}: {model_to_test['test_status']}")
        
        return jsonify({
            "success": True,
            "data": test_results,
            "message": f"Model {model_id} test completed"
        })
        
    except Exception as e:
        logger.error(f"Error testing registered model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@reranker_bp.route('/models/registry/<model_id>', methods=['DELETE'])
async def unregister_reranker_model(model_id):
    """Remove a reranker model from the registry."""
    try:
        registry = _load_registry()
        
        if not registry.get('models'):
            return jsonify({"success": False, "error": "No registry found"}), 404
        
        # Find and remove model
        original_count = len(registry['models'])
        registry['models'] = [model for model in registry['models'] if model['id'] != model_id]
        
        if len(registry['models']) == original_count:
            return jsonify({"success": False, "error": f"Model {model_id} not found in registry"}), 404
        
        registry['last_updated'] = datetime.utcnow().isoformat()
        _save_registry(registry)
        
        logger.info(f"Unregistered reranker model: {model_id}")
        
        return jsonify({
            "success": True,
            "message": f"Model {model_id} unregistered successfully"
        })
        
    except Exception as e:
        logger.error(f"Error unregistering model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500
