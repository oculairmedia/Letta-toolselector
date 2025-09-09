"""
LDTS-65: Implement production isolation and safety validation

Comprehensive production isolation and safety validation system to ensure
testing dashboard operations cannot interfere with production environments.
"""

import logging
import re
import os
import hashlib
import time
from typing import List, Dict, Any, Optional, Set, Pattern, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import ipaddress
from urllib.parse import urlparse
import asyncio
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class EnvironmentType(Enum):
    """Environment classification"""
    PRODUCTION = "production"
    STAGING = "staging" 
    DEVELOPMENT = "development"
    TESTING = "testing"
    SANDBOX = "sandbox"
    UNKNOWN = "unknown"

class IsolationLevel(Enum):
    """Isolation enforcement levels"""
    STRICT = "strict"          # Block all production access
    MODERATE = "moderate"      # Allow with warnings
    PERMISSIVE = "permissive"  # Allow with logging only
    DISABLED = "disabled"      # No isolation checks

class ValidationResult(Enum):
    """Validation result status"""
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    WARNING = "warning"

@dataclass
class ProductionResource:
    """Production resource definition"""
    resource_id: str
    resource_type: str  # "database", "api", "service", "host"
    identifier: str     # URL, host, connection string, etc.
    environment: EnvironmentType
    criticality: str    # "high", "medium", "low"
    description: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class IsolationRule:
    """Production isolation rule"""
    rule_id: str
    rule_name: str
    resource_type: str
    pattern: str        # Regex pattern to match
    action: ValidationResult
    description: str
    enabled: bool = True
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SafetyValidation:
    """Safety validation result"""
    validation_id: str
    resource_identifier: str
    resource_type: str
    environment_detected: EnvironmentType
    validation_result: ValidationResult
    matched_rules: List[str]
    risk_score: float
    timestamp: datetime
    details: Dict[str, Any]
    recommendations: List[str]

class ProductionIsolationService:
    """
    Service for enforcing production isolation and safety validation
    to prevent testing dashboard from interfering with production systems.
    """
    
    def __init__(self, isolation_level: IsolationLevel = IsolationLevel.STRICT):
        """
        Initialize production isolation service
        
        Args:
            isolation_level: Enforcement level for isolation checks
        """
        self.isolation_level = isolation_level
        self.production_resources = {}
        self.isolation_rules = {}
        self.validation_cache = {}
        
        # Statistics tracking
        self.statistics = {
            "total_validations": 0,
            "blocked_requests": 0,
            "warning_requests": 0,
            "allowed_requests": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "last_rule_update": None
        }
        
        # Initialize default rules
        self._initialize_default_rules()
        self._initialize_production_indicators()
        
        logger.info(f"Initialized ProductionIsolationService with isolation level: {isolation_level.value}")
    
    def _initialize_default_rules(self):
        """Initialize default production isolation rules"""
        
        # Production domain patterns
        self._add_rule(IsolationRule(
            rule_id="prod_domains",
            rule_name="Production Domain Detection",
            resource_type="url",
            pattern=r".*\.(com|net|org)$|.*prod.*|.*production.*",
            action=ValidationResult.BLOCKED,
            description="Block access to production domains"
        ))
        
        # Production database patterns
        self._add_rule(IsolationRule(
            rule_id="prod_databases",
            rule_name="Production Database Detection",
            resource_type="database",
            pattern=r".*prod.*|.*production.*|.*live.*|.*master.*",
            action=ValidationResult.BLOCKED,
            description="Block access to production databases"
        ))
        
        # Production API endpoints
        self._add_rule(IsolationRule(
            rule_id="prod_apis",
            rule_name="Production API Detection", 
            resource_type="api",
            pattern=r".*api\..*\.com|.*prod.*api.*|.*production.*api.*",
            action=ValidationResult.BLOCKED,
            description="Block access to production APIs"
        ))
        
        # Staging environment warnings
        self._add_rule(IsolationRule(
            rule_id="staging_warning",
            rule_name="Staging Environment Warning",
            resource_type="all",
            pattern=r".*staging.*|.*stage.*|.*stg.*",
            action=ValidationResult.WARNING,
            description="Warn when accessing staging environments"
        ))
        
        # Local development allowed
        self._add_rule(IsolationRule(
            rule_id="local_allowed",
            rule_name="Local Development Allowed",
            resource_type="all",
            pattern=r"localhost|127\.0\.0\.1|0\.0\.0\.0|.*\.local|.*dev.*|.*test.*",
            action=ValidationResult.ALLOWED,
            description="Allow access to local/development environments"
        ))
    
    def _initialize_production_indicators(self):
        """Initialize production environment indicators"""
        
        # Common production indicators
        self.production_indicators = {
            "domain_patterns": [
                r".*\.com$", r".*\.net$", r".*\.org$",
                r".*prod.*", r".*production.*", r".*live.*"
            ],
            "port_patterns": [
                443,  # HTTPS production
                80,   # HTTP production  
                5432, # PostgreSQL production
                3306, # MySQL production
                6379, # Redis production
            ],
            "path_patterns": [
                r"/api/v\d+/", r"/prod/", r"/production/",
                r"/live/", r"/master/"
            ],
            "env_variables": [
                "PROD", "PRODUCTION", "LIVE", "MASTER"
            ]
        }
    
    def _add_rule(self, rule: IsolationRule):
        """Add isolation rule"""
        self.isolation_rules[rule.rule_id] = rule
        logger.debug(f"Added isolation rule: {rule.rule_name}")
    
    async def validate_resource_access(
        self,
        resource_identifier: str,
        resource_type: str,
        operation: str = "access",
        context: Optional[Dict[str, Any]] = None
    ) -> SafetyValidation:
        """
        Validate if access to a resource is safe and allowed
        
        Args:
            resource_identifier: URL, host, connection string, etc.
            resource_type: Type of resource being accessed
            operation: Operation being performed
            context: Additional context information
            
        Returns:
            SafetyValidation result with recommendations
        """
        validation_id = hashlib.md5(
            f"{resource_identifier}:{resource_type}:{operation}:{time.time()}".encode()
        ).hexdigest()
        
        # Check cache first
        cache_key = f"{resource_identifier}:{resource_type}"
        if cache_key in self.validation_cache:
            self.statistics["cache_hits"] += 1
            cached_result = self.validation_cache[cache_key]
            # Update timestamp and validation_id
            cached_result.validation_id = validation_id
            cached_result.timestamp = datetime.now(timezone.utc)
            return cached_result
        
        self.statistics["cache_misses"] += 1
        self.statistics["total_validations"] += 1
        
        # Detect environment
        environment = await self._detect_environment(resource_identifier, resource_type)
        
        # Apply isolation rules
        matched_rules, validation_result = await self._apply_isolation_rules(
            resource_identifier, resource_type, environment
        )
        
        # Calculate risk score
        risk_score = await self._calculate_risk_score(
            resource_identifier, resource_type, environment, matched_rules
        )
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(
            resource_identifier, resource_type, environment, validation_result, risk_score
        )
        
        # Create validation result
        safety_validation = SafetyValidation(
            validation_id=validation_id,
            resource_identifier=resource_identifier,
            resource_type=resource_type,
            environment_detected=environment,
            validation_result=validation_result,
            matched_rules=matched_rules,
            risk_score=risk_score,
            timestamp=datetime.now(timezone.utc),
            details={
                "operation": operation,
                "context": context or {},
                "isolation_level": self.isolation_level.value,
                "cache_used": False
            },
            recommendations=recommendations
        )
        
        # Cache result
        if len(self.validation_cache) < 1000:  # Limit cache size
            self.validation_cache[cache_key] = safety_validation
        
        # Update statistics
        if validation_result == ValidationResult.BLOCKED:
            self.statistics["blocked_requests"] += 1
        elif validation_result == ValidationResult.WARNING:
            self.statistics["warning_requests"] += 1
        else:
            self.statistics["allowed_requests"] += 1
        
        logger.info(f"Safety validation: {resource_identifier} -> {validation_result.value} (risk: {risk_score})")
        
        return safety_validation
    
    async def _detect_environment(self, identifier: str, resource_type: str) -> EnvironmentType:
        """Detect environment type from resource identifier"""
        
        identifier_lower = identifier.lower()
        
        # Production indicators
        production_patterns = [
            r".*prod.*", r".*production.*", r".*live.*", 
            r".*\.com$", r".*\.net$", r".*\.org$",
            r".*master.*", r".*main.*"
        ]
        
        for pattern in production_patterns:
            if re.match(pattern, identifier_lower):
                return EnvironmentType.PRODUCTION
        
        # Staging indicators
        staging_patterns = [
            r".*staging.*", r".*stage.*", r".*stg.*",
            r".*preview.*", r".*beta.*"
        ]
        
        for pattern in staging_patterns:
            if re.match(pattern, identifier_lower):
                return EnvironmentType.STAGING
        
        # Development indicators
        dev_patterns = [
            r"localhost", r"127\.0\.0\.1", r"0\.0\.0\.0",
            r".*\.local", r".*dev.*", r".*development.*",
            r".*test.*", r".*testing.*"
        ]
        
        for pattern in dev_patterns:
            if re.match(pattern, identifier_lower):
                return EnvironmentType.DEVELOPMENT
        
        # Check for private IP ranges
        try:
            if resource_type == "url":
                parsed = urlparse(identifier)
                hostname = parsed.hostname
            else:
                hostname = identifier
            
            if hostname:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private:
                    return EnvironmentType.DEVELOPMENT
        except (ValueError, ipaddress.AddressValueError):
            pass
        
        return EnvironmentType.UNKNOWN
    
    async def _apply_isolation_rules(
        self, 
        identifier: str, 
        resource_type: str, 
        environment: EnvironmentType
    ) -> Tuple[List[str], ValidationResult]:
        """Apply isolation rules to determine access validation"""
        
        matched_rules = []
        highest_severity = ValidationResult.ALLOWED
        
        for rule in self.isolation_rules.values():
            if not rule.enabled:
                continue
            
            # Check if rule applies to this resource type
            if rule.resource_type != "all" and rule.resource_type != resource_type:
                continue
            
            # Check if pattern matches
            try:
                if re.search(rule.pattern, identifier, re.IGNORECASE):
                    matched_rules.append(rule.rule_id)
                    
                    # Determine highest severity action
                    if rule.action == ValidationResult.BLOCKED:
                        highest_severity = ValidationResult.BLOCKED
                    elif rule.action == ValidationResult.WARNING and highest_severity == ValidationResult.ALLOWED:
                        highest_severity = ValidationResult.WARNING
            except re.error:
                logger.warning(f"Invalid regex pattern in rule {rule.rule_id}: {rule.pattern}")
        
        # Apply isolation level override
        if self.isolation_level == IsolationLevel.DISABLED:
            highest_severity = ValidationResult.ALLOWED
        elif self.isolation_level == IsolationLevel.PERMISSIVE:
            if highest_severity == ValidationResult.BLOCKED:
                highest_severity = ValidationResult.WARNING
        
        return matched_rules, highest_severity
    
    async def _calculate_risk_score(
        self,
        identifier: str,
        resource_type: str,
        environment: EnvironmentType,
        matched_rules: List[str]
    ) -> float:
        """Calculate risk score for the resource access"""
        
        base_score = 0.0
        
        # Environment risk
        env_risk = {
            EnvironmentType.PRODUCTION: 10.0,
            EnvironmentType.STAGING: 6.0,
            EnvironmentType.TESTING: 3.0,
            EnvironmentType.DEVELOPMENT: 1.0,
            EnvironmentType.SANDBOX: 0.5,
            EnvironmentType.UNKNOWN: 8.0
        }
        base_score += env_risk.get(environment, 5.0)
        
        # Resource type risk
        resource_risk = {
            "database": 8.0,
            "api": 6.0,
            "service": 5.0,
            "url": 4.0,
            "host": 3.0
        }
        base_score += resource_risk.get(resource_type, 2.0)
        
        # Rule-based risk
        rule_risk = len(matched_rules) * 2.0
        base_score += rule_risk
        
        # External domain risk
        try:
            if resource_type == "url":
                parsed = urlparse(identifier)
                if parsed.netloc and not parsed.netloc.startswith(("localhost", "127.", "192.168.", "10.")):
                    base_score += 3.0
        except:
            pass
        
        # Normalize to 0-10 scale
        return min(base_score, 10.0)
    
    async def _generate_recommendations(
        self,
        identifier: str,
        resource_type: str,
        environment: EnvironmentType,
        validation_result: ValidationResult,
        risk_score: float
    ) -> List[str]:
        """Generate safety recommendations"""
        
        recommendations = []
        
        if validation_result == ValidationResult.BLOCKED:
            recommendations.extend([
                "Access to this resource is blocked due to production isolation rules",
                "Consider using a development or staging environment for testing",
                "Verify that the resource identifier is correct for testing purposes",
                "Contact system administrators if production access is genuinely required"
            ])
        
        elif validation_result == ValidationResult.WARNING:
            recommendations.extend([
                "Exercise caution when accessing this resource",
                "Ensure any operations are safe for the detected environment",
                "Consider using read-only operations where possible",
                "Monitor the impact of operations on this environment"
            ])
        
        if environment == EnvironmentType.PRODUCTION:
            recommendations.extend([
                "Production environment detected - use extreme caution",
                "All operations should be thoroughly tested in staging first",
                "Ensure proper backup and rollback procedures are in place"
            ])
        
        elif environment == EnvironmentType.UNKNOWN:
            recommendations.extend([
                "Unknown environment detected - verify environment classification",
                "Manually confirm the environment type before proceeding",
                "Add environment-specific indicators to improve detection"
            ])
        
        if risk_score >= 8.0:
            recommendations.append("High risk score detected - review all safety measures")
        elif risk_score >= 5.0:
            recommendations.append("Medium risk score - proceed with documented precautions")
        
        return recommendations
    
    def add_production_resource(self, resource: ProductionResource):
        """Register a known production resource"""
        self.production_resources[resource.resource_id] = resource
        logger.info(f"Added production resource: {resource.identifier}")
    
    def add_isolation_rule(self, rule: IsolationRule):
        """Add custom isolation rule"""
        self._add_rule(rule)
        self.statistics["last_rule_update"] = datetime.now(timezone.utc).isoformat()
    
    def update_isolation_level(self, level: IsolationLevel):
        """Update isolation enforcement level"""
        old_level = self.isolation_level
        self.isolation_level = level
        self.validation_cache.clear()  # Clear cache as results may change
        logger.info(f"Updated isolation level from {old_level.value} to {level.value}")
    
    def get_isolation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get all isolation rules"""
        return {
            rule_id: asdict(rule) for rule_id, rule in self.isolation_rules.items()
        }
    
    def get_production_resources(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered production resources"""
        return {
            resource_id: asdict(resource) 
            for resource_id, resource in self.production_resources.items()
        }
    
    def clear_validation_cache(self):
        """Clear validation cache"""
        cache_size = len(self.validation_cache)
        self.validation_cache.clear()
        logger.info(f"Cleared validation cache ({cache_size} entries)")
    
    async def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for production safety"""
        
        issues = []
        
        # Check for production indicators in configuration
        config_str = str(config).lower()
        
        production_indicators = [
            "prod", "production", "live", "master", "main"
        ]
        
        for indicator in production_indicators:
            if indicator in config_str:
                issues.append(f"Production indicator '{indicator}' found in configuration")
        
        # Check URLs
        for key, value in config.items():
            if isinstance(value, str) and ("http" in value or "://" in value):
                validation = await self.validate_resource_access(value, "url", "config_check")
                if validation.validation_result == ValidationResult.BLOCKED:
                    issues.append(f"Configuration key '{key}' contains blocked resource: {value}")
        
        return issues
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get isolation service statistics"""
        return {
            **self.statistics,
            "isolation_level": self.isolation_level.value,
            "total_rules": len(self.isolation_rules),
            "total_resources": len(self.production_resources),
            "cache_size": len(self.validation_cache),
            "cache_hit_rate": (
                self.statistics["cache_hits"] / 
                (self.statistics["cache_hits"] + self.statistics["cache_misses"])
                if (self.statistics["cache_hits"] + self.statistics["cache_misses"]) > 0
                else 0.0
            )
        }

# Global production isolation service instance
production_isolation_service = ProductionIsolationService()