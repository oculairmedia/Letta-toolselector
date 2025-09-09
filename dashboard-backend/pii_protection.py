"""
LDTS-72: Add PII redaction and payload size limits for testing API

PII (Personally Identifiable Information) redaction and payload size limiting
for testing API safety and compliance.
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
from fastapi import HTTPException, Request
import hashlib
import time

logger = logging.getLogger(__name__)

class PIIRedactionConfig(BaseModel):
    """Configuration for PII redaction"""
    enabled: bool = True
    redaction_placeholder: str = "[REDACTED]"
    hash_pii: bool = True  # Hash PII instead of just redacting
    log_pii_detection: bool = True
    strict_mode: bool = True  # Block requests with PII vs just redacting

class PayloadLimitConfig(BaseModel):
    """Configuration for payload size limits"""
    max_request_size_mb: float = 10.0  # 10MB default
    max_json_depth: int = 20
    max_array_length: int = 10000
    max_string_length: int = 100000
    enforce_limits: bool = True

class PIIPattern:
    """PII detection patterns"""
    
    # Email addresses
    EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    # Phone numbers (various formats)
    PHONE = re.compile(r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})')
    
    # Social Security Numbers
    SSN = re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b')
    
    # Credit card numbers (basic pattern)
    CREDIT_CARD = re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b')
    
    # IP addresses (can be PII in some contexts)
    IP_ADDRESS = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    
    # API keys and tokens (common patterns)
    API_KEY = re.compile(r'(?i)(api[_-]?key|token|secret|password)["\s]*[:=]["\s]*([a-zA-Z0-9_-]{20,})')
    
    # Names (simple pattern - first/last name)
    NAME_PATTERN = re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b')
    
    # URLs that might contain PII
    URL_WITH_PARAMS = re.compile(r'https?://[^\s]+[?&]([^=]+=[^&\s]+)')

class PIIProtectionManager:
    """Manager for PII protection and payload validation"""
    
    def __init__(self):
        self.redaction_config = PIIRedactionConfig()
        self.payload_config = PayloadLimitConfig()
        self.pii_detections = []
        self.payload_violations = []
        
        # PII patterns to check
        self.pii_patterns = {
            'email': PIIPattern.EMAIL,
            'phone': PIIPattern.PHONE,
            'ssn': PIIPattern.SSN,
            'credit_card': PIIPattern.CREDIT_CARD,
            'ip_address': PIIPattern.IP_ADDRESS,
            'api_key': PIIPattern.API_KEY,
            'name': PIIPattern.NAME_PATTERN,
            'url_params': PIIPattern.URL_WITH_PARAMS
        }
        
        logger.info("PIIProtectionManager initialized")
    
    async def validate_and_redact_request(self, request: Request) -> Dict[str, Any]:
        """Validate request size and redact PII"""
        # Get request body
        body = await request.body()
        
        # Validate payload size
        size_validation = self._validate_payload_size(body, request)
        if not size_validation['valid']:
            if self.payload_config.enforce_limits:
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload too large: {size_validation['reason']}"
                )
        
        # Parse and validate JSON structure if applicable
        content_type = request.headers.get('content-type', '').lower()
        if 'application/json' in content_type:
            try:
                json_data = json.loads(body.decode('utf-8'))
                structure_validation = self._validate_json_structure(json_data)
                if not structure_validation['valid']:
                    if self.payload_config.enforce_limits:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid JSON structure: {structure_validation['reason']}"
                        )
            except json.JSONDecodeError:
                # Not valid JSON, treat as text
                json_data = None
        else:
            json_data = None
        
        # Detect and handle PII
        body_text = body.decode('utf-8', errors='ignore')
        pii_result = self._detect_and_redact_pii(body_text, str(request.url))
        
        return {
            'size_validation': size_validation,
            'pii_result': pii_result,
            'json_data': json_data,
            'original_size': len(body),
            'redacted_content': pii_result['redacted_content'] if pii_result['pii_detected'] else body_text
        }
    
    def _validate_payload_size(self, body: bytes, request: Request) -> Dict[str, Any]:
        """Validate payload size limits"""
        size_mb = len(body) / (1024 * 1024)
        
        if size_mb > self.payload_config.max_request_size_mb:
            violation = {
                'type': 'SIZE_EXCEEDED',
                'size_mb': size_mb,
                'limit_mb': self.payload_config.max_request_size_mb,
                'url': str(request.url),
                'timestamp': time.time()
            }
            self.payload_violations.append(violation)
            
            return {
                'valid': False,
                'reason': f'Request size {size_mb:.2f}MB exceeds limit of {self.payload_config.max_request_size_mb}MB',
                'size_mb': size_mb,
                'limit_mb': self.payload_config.max_request_size_mb
            }
        
        return {
            'valid': True,
            'size_mb': size_mb,
            'limit_mb': self.payload_config.max_request_size_mb
        }
    
    def _validate_json_structure(self, json_data: Any, depth: int = 0) -> Dict[str, Any]:
        """Validate JSON structure limits"""
        if depth > self.payload_config.max_json_depth:
            return {
                'valid': False,
                'reason': f'JSON depth {depth} exceeds limit of {self.payload_config.max_json_depth}'
            }
        
        if isinstance(json_data, dict):
            for value in json_data.values():
                result = self._validate_json_structure(value, depth + 1)
                if not result['valid']:
                    return result
                    
        elif isinstance(json_data, list):
            if len(json_data) > self.payload_config.max_array_length:
                return {
                    'valid': False,
                    'reason': f'Array length {len(json_data)} exceeds limit of {self.payload_config.max_array_length}'
                }
            for item in json_data:
                result = self._validate_json_structure(item, depth + 1)
                if not result['valid']:
                    return result
                    
        elif isinstance(json_data, str):
            if len(json_data) > self.payload_config.max_string_length:
                return {
                    'valid': False,
                    'reason': f'String length {len(json_data)} exceeds limit of {self.payload_config.max_string_length}'
                }
        
        return {'valid': True}
    
    def _detect_and_redact_pii(self, text: str, url: str) -> Dict[str, Any]:
        """Detect and redact PII from text"""
        if not self.redaction_config.enabled:
            return {
                'pii_detected': False,
                'redacted_content': text,
                'detections': []
            }
        
        detections = []
        redacted_text = text
        
        # Check each PII pattern
        for pattern_name, pattern in self.pii_patterns.items():
            matches = pattern.findall(text)
            if matches:
                for match in matches:
                    match_str = match if isinstance(match, str) else str(match)
                    
                    detection = {
                        'type': pattern_name,
                        'match': match_str[:50] + '...' if len(match_str) > 50 else match_str,
                        'url': url,
                        'timestamp': time.time()
                    }
                    
                    if self.redaction_config.hash_pii:
                        # Hash the PII for audit purposes
                        detection['hash'] = hashlib.sha256(match_str.encode()).hexdigest()[:16]
                    
                    detections.append(detection)
                    
                    # Redact from text
                    replacement = self.redaction_config.redaction_placeholder
                    if self.redaction_config.hash_pii:
                        replacement = f"[REDACTED:{detection['hash']}]"
                    
                    redacted_text = pattern.sub(replacement, redacted_text)
        
        # Log PII detections
        if detections and self.redaction_config.log_pii_detection:
            logger.warning(f"PII detected in request to {url}: {len(detections)} instances")
            self.pii_detections.extend(detections)
        
        # Handle strict mode
        if detections and self.redaction_config.strict_mode:
            raise HTTPException(
                status_code=400,
                detail=f"Request contains PII and strict mode is enabled. Detected: {[d['type'] for d in detections]}"
            )
        
        return {
            'pii_detected': len(detections) > 0,
            'redacted_content': redacted_text,
            'detections': detections,
            'detection_count': len(detections)
        }
    
    def redact_response_data(self, data: Any) -> Any:
        """Redact PII from response data"""
        if not self.redaction_config.enabled:
            return data
        
        if isinstance(data, dict):
            return {key: self.redact_response_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self.redact_response_data(item) for item in data]
        elif isinstance(data, str):
            result = self._detect_and_redact_pii(data, "response")
            return result['redacted_content']
        else:
            return data
    
    def get_protection_statistics(self) -> Dict[str, Any]:
        """Get PII protection statistics"""
        return {
            'configuration': {
                'pii_redaction_enabled': self.redaction_config.enabled,
                'strict_mode': self.redaction_config.strict_mode,
                'hash_pii': self.redaction_config.hash_pii,
                'max_request_size_mb': self.payload_config.max_request_size_mb,
                'max_json_depth': self.payload_config.max_json_depth,
                'enforce_limits': self.payload_config.enforce_limits
            },
            'statistics': {
                'pii_detections_count': len(self.pii_detections),
                'payload_violations_count': len(self.payload_violations),
                'recent_pii_detections': self.pii_detections[-10:],  # Last 10
                'recent_payload_violations': self.payload_violations[-10:]  # Last 10
            },
            'patterns_monitored': list(self.pii_patterns.keys()),
            'protection_status': {
                'pii_protection_active': self.redaction_config.enabled,
                'payload_limits_active': self.payload_config.enforce_limits,
                'strict_mode_active': self.redaction_config.strict_mode
            }
        }
    
    def update_configuration(self, pii_config: Dict[str, Any] = None, payload_config: Dict[str, Any] = None):
        """Update protection configuration"""
        if pii_config:
            for key, value in pii_config.items():
                if hasattr(self.redaction_config, key):
                    setattr(self.redaction_config, key, value)
        
        if payload_config:
            for key, value in payload_config.items():
                if hasattr(self.payload_config, key):
                    setattr(self.payload_config, key, value)
        
        logger.info(f"PII protection configuration updated")
    
    def clear_statistics(self):
        """Clear accumulated statistics (for testing/maintenance)"""
        self.pii_detections.clear()
        self.payload_violations.clear()
        logger.info("PII protection statistics cleared")

# Global PII protection manager
pii_protection_manager = PIIProtectionManager()