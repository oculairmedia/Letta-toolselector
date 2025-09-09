"""
LDTS-73: Implement centralized audit logger with configuration hashing

Enhanced centralized audit logging system with configuration hashing,
tampering detection, and comprehensive audit trail management.
"""

import logging
import json
import hashlib
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from pathlib import Path
import asyncio
import aiofiles
from pydantic import BaseModel
import structlog
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import base64
import os

logger = structlog.get_logger(__name__)

class AuditIntegrityLevel(Enum):
    """Audit log integrity levels"""
    BASIC = "basic"              # Basic logging
    HASHED = "hashed"           # Configuration hashing
    ENCRYPTED = "encrypted"      # Encrypted audit logs
    TAMPER_EVIDENT = "tamper_evident"  # Tamper detection

class ConfigurationChange(BaseModel):
    """Configuration change tracking model"""
    change_id: str
    timestamp: str
    component: str
    section: str
    old_config_hash: str
    new_config_hash: str
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    change_diff: Dict[str, Any]
    user_context: Dict[str, Any]
    validation_result: Dict[str, Any]
    risk_level: str

class AuditChain(BaseModel):
    """Audit log chain entry for tamper detection"""
    entry_id: str
    timestamp: str
    previous_hash: Optional[str]
    content_hash: str
    signature: str
    integrity_verified: bool = True

class CentralizedAuditLogger:
    """Enhanced centralized audit logger with configuration hashing"""
    
    def __init__(self, 
                 log_directory: str = "audit_logs",
                 integrity_level: AuditIntegrityLevel = AuditIntegrityLevel.HASHED,
                 encryption_key: Optional[bytes] = None):
        
        self.log_directory = Path(log_directory)
        self.integrity_level = integrity_level
        self.encryption_key = encryption_key or self._generate_key()
        
        # Audit log files
        self.audit_log_file = self.log_directory / "centralized_audit.log"
        self.config_log_file = self.log_directory / "configuration_changes.log"
        self.integrity_log_file = self.log_directory / "audit_integrity.log"
        self.chain_file = self.log_directory / "audit_chain.json"
        
        # In-memory tracking
        self.configuration_hashes: Dict[str, str] = {}
        self.audit_chain: List[AuditChain] = []
        self.pending_entries = []
        self.integrity_violations = []
        
        # Configuration change tracking
        self.config_change_history: List[ConfigurationChange] = []
        self.watched_config_sections: Set[str] = {
            "safety_config", "pii_protection", "rate_limiting", 
            "weaviate_config", "embedding_providers", "reranker_config"
        }
        
        # Initialize audit system
        asyncio.create_task(self._initialize_audit_system())
        
        logger.info(f"CentralizedAuditLogger initialized: integrity_level={integrity_level.value}")
    
    def _generate_key(self) -> bytes:
        """Generate encryption key for audit log protection"""
        return os.urandom(32)  # 256-bit key
    
    async def _initialize_audit_system(self):
        """Initialize the centralized audit system"""
        try:
            # Create log directory
            self.log_directory.mkdir(parents=True, exist_ok=True)
            
            # Load existing audit chain
            await self._load_audit_chain()
            
            # Load configuration hashes
            await self._load_configuration_hashes()
            
            # Initialize integrity monitoring
            await self._start_integrity_monitoring()
            
            logger.info("Centralized audit system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize centralized audit system: {e}")
    
    async def log_configuration_change(self,
                                     component: str,
                                     section: str,
                                     old_config: Dict[str, Any],
                                     new_config: Dict[str, Any],
                                     user_context: Dict[str, Any],
                                     validation_result: Dict[str, Any]) -> str:
        """Log configuration change with hashing and integrity tracking"""
        
        change_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Generate configuration hashes
        old_config_hash = self._hash_configuration(old_config)
        new_config_hash = self._hash_configuration(new_config)
        
        # Calculate configuration diff
        change_diff = self._calculate_config_diff(old_config, new_config)
        
        # Assess risk level
        risk_level = self._assess_configuration_risk(change_diff, section)
        
        # Create configuration change record
        config_change = ConfigurationChange(
            change_id=change_id,
            timestamp=timestamp,
            component=component,
            section=section,
            old_config_hash=old_config_hash,
            new_config_hash=new_config_hash,
            old_config=old_config,
            new_config=new_config,
            change_diff=change_diff,
            user_context=user_context,
            validation_result=validation_result,
            risk_level=risk_level
        )
        
        # Log to configuration change log
        await self._write_configuration_log(config_change)
        
        # Update configuration hash tracking
        self.configuration_hashes[f"{component}.{section}"] = new_config_hash
        
        # Create audit chain entry
        await self._add_to_audit_chain(
            entry_type="configuration_change",
            content=config_change.dict(),
            risk_level=risk_level
        )
        
        # Log to main audit log
        await self._write_audit_entry({
            "event_type": "configuration_change",
            "change_id": change_id,
            "component": component,
            "section": section,
            "risk_level": risk_level,
            "user_context": user_context,
            "timestamp": timestamp,
            "config_hashes": {
                "old": old_config_hash,
                "new": new_config_hash
            },
            "change_summary": self._summarize_config_change(change_diff)
        })
        
        self.config_change_history.append(config_change)
        
        logger.info(f"Configuration change logged: {component}.{section} (risk: {risk_level})")
        
        return change_id
    
    def _hash_configuration(self, config: Dict[str, Any]) -> str:
        """Generate stable hash of configuration"""
        # Normalize config for consistent hashing
        normalized = self._normalize_config(config)
        config_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()
    
    def _normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration for consistent hashing"""
        if isinstance(config, dict):
            # Remove metadata fields that shouldn't affect hash
            filtered = {k: v for k, v in config.items() 
                       if k not in ['timestamp', 'last_modified', 'version']}
            return {k: self._normalize_config(v) for k, v in sorted(filtered.items())}
        elif isinstance(config, list):
            return [self._normalize_config(item) for item in config]
        else:
            return config
    
    def _calculate_config_diff(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate detailed configuration difference"""
        diff = {
            "added": {},
            "removed": {},
            "modified": {},
            "unchanged": {}
        }
        
        old_keys = set(old_config.keys())
        new_keys = set(new_config.keys())
        
        # Added keys
        for key in new_keys - old_keys:
            diff["added"][key] = new_config[key]
        
        # Removed keys
        for key in old_keys - new_keys:
            diff["removed"][key] = old_config[key]
        
        # Modified/unchanged keys
        for key in old_keys & new_keys:
            if old_config[key] != new_config[key]:
                diff["modified"][key] = {
                    "old": old_config[key],
                    "new": new_config[key]
                }
            else:
                diff["unchanged"][key] = old_config[key]
        
        return diff
    
    def _assess_configuration_risk(self, change_diff: Dict[str, Any], section: str) -> str:
        """Assess risk level of configuration change"""
        # High risk indicators
        high_risk_patterns = [
            "safety", "security", "auth", "production", "critical"
        ]
        
        # Critical configuration sections
        critical_sections = {
            "safety_config", "authentication", "security_settings"
        }
        
        # Count changes
        added_count = len(change_diff.get("added", {}))
        removed_count = len(change_diff.get("removed", {}))
        modified_count = len(change_diff.get("modified", {}))
        total_changes = added_count + removed_count + modified_count
        
        # Assess risk
        if section in critical_sections:
            return "CRITICAL"
        elif any(pattern in section.lower() for pattern in high_risk_patterns):
            return "HIGH"
        elif total_changes > 10:
            return "MEDIUM"
        elif removed_count > 0:
            return "MEDIUM"  # Removals are generally riskier
        else:
            return "LOW"
    
    def _summarize_config_change(self, change_diff: Dict[str, Any]) -> str:
        """Generate human-readable summary of configuration change"""
        added = len(change_diff.get("added", {}))
        removed = len(change_diff.get("removed", {}))
        modified = len(change_diff.get("modified", {}))
        
        parts = []
        if added > 0:
            parts.append(f"{added} added")
        if removed > 0:
            parts.append(f"{removed} removed")
        if modified > 0:
            parts.append(f"{modified} modified")
        
        return ", ".join(parts) if parts else "no changes"
    
    async def _add_to_audit_chain(self, entry_type: str, content: Dict[str, Any], risk_level: str = "LOW"):
        """Add entry to tamper-evident audit chain"""
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Get previous hash for chaining
        previous_hash = None
        if self.audit_chain:
            previous_hash = self.audit_chain[-1].content_hash
        
        # Create content hash
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        # Create signature (simplified - would use proper crypto in production)
        signature_data = f"{entry_id}:{timestamp}:{previous_hash or ''}:{content_hash}"
        signature = hashlib.sha256(signature_data.encode()).hexdigest()
        
        # Create chain entry
        chain_entry = AuditChain(
            entry_id=entry_id,
            timestamp=timestamp,
            previous_hash=previous_hash,
            content_hash=content_hash,
            signature=signature
        )
        
        self.audit_chain.append(chain_entry)
        
        # Save chain to disk
        await self._save_audit_chain()
        
        # Log to integrity log
        await self._write_integrity_log({
            "entry_id": entry_id,
            "entry_type": entry_type,
            "risk_level": risk_level,
            "chain_position": len(self.audit_chain),
            "integrity_verified": True,
            "timestamp": timestamp
        })
    
    async def verify_audit_integrity(self) -> Dict[str, Any]:
        """Verify integrity of audit chain"""
        verification_result = {
            "chain_valid": True,
            "total_entries": len(self.audit_chain),
            "verified_entries": 0,
            "integrity_violations": [],
            "verification_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        for i, entry in enumerate(self.audit_chain):
            try:
                # Verify chain linkage
                if i > 0:
                    expected_previous = self.audit_chain[i-1].content_hash
                    if entry.previous_hash != expected_previous:
                        violation = {
                            "entry_id": entry.entry_id,
                            "violation_type": "chain_break",
                            "position": i,
                            "expected_previous": expected_previous,
                            "actual_previous": entry.previous_hash
                        }
                        verification_result["integrity_violations"].append(violation)
                        verification_result["chain_valid"] = False
                        continue
                
                # Verify signature
                signature_data = f"{entry.entry_id}:{entry.timestamp}:{entry.previous_hash or ''}:{entry.content_hash}"
                expected_signature = hashlib.sha256(signature_data.encode()).hexdigest()
                
                if entry.signature != expected_signature:
                    violation = {
                        "entry_id": entry.entry_id,
                        "violation_type": "signature_mismatch",
                        "position": i
                    }
                    verification_result["integrity_violations"].append(violation)
                    verification_result["chain_valid"] = False
                    continue
                
                verification_result["verified_entries"] += 1
                
            except Exception as e:
                violation = {
                    "entry_id": entry.entry_id,
                    "violation_type": "verification_error",
                    "position": i,
                    "error": str(e)
                }
                verification_result["integrity_violations"].append(violation)
                verification_result["chain_valid"] = False
        
        # Log verification result
        await self._write_integrity_log({
            "operation": "integrity_verification",
            "result": verification_result,
            "timestamp": verification_result["verification_timestamp"]
        })
        
        if not verification_result["chain_valid"]:
            logger.critical("AUDIT INTEGRITY VIOLATION DETECTED", violations=verification_result["integrity_violations"])
            self.integrity_violations.extend(verification_result["integrity_violations"])
        
        return verification_result
    
    async def get_configuration_history(self, 
                                       component: Optional[str] = None,
                                       section: Optional[str] = None,
                                       limit: int = 100) -> List[ConfigurationChange]:
        """Get configuration change history"""
        history = self.config_change_history.copy()
        
        # Apply filters
        if component:
            history = [c for c in history if c.component == component]
        if section:
            history = [c for c in history if c.section == section]
        
        # Sort by timestamp (most recent first)
        history.sort(key=lambda x: x.timestamp, reverse=True)
        
        return history[:limit]
    
    async def get_configuration_risk_summary(self) -> Dict[str, Any]:
        """Get configuration risk analysis summary"""
        if not self.config_change_history:
            return {"total_changes": 0, "risk_distribution": {}}
        
        risk_counts = {}
        component_risks = {}
        recent_high_risk = []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        
        for change in self.config_change_history:
            # Count by risk level
            risk_level = change.risk_level
            risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
            
            # Count by component
            if change.component not in component_risks:
                component_risks[change.component] = {}
            component_risks[change.component][risk_level] = component_risks[change.component].get(risk_level, 0) + 1
            
            # Track recent high-risk changes
            change_time = datetime.fromisoformat(change.timestamp.replace('Z', '+00:00'))
            if change_time > cutoff_time and risk_level in ["HIGH", "CRITICAL"]:
                recent_high_risk.append({
                    "change_id": change.change_id,
                    "component": change.component,
                    "section": change.section,
                    "risk_level": risk_level,
                    "timestamp": change.timestamp
                })
        
        return {
            "total_changes": len(self.config_change_history),
            "risk_distribution": risk_counts,
            "component_risks": component_risks,
            "recent_high_risk_changes": recent_high_risk,
            "integrity_violations": len(self.integrity_violations),
            "last_verification": await self._get_last_verification_time()
        }
    
    async def _write_configuration_log(self, config_change: ConfigurationChange):
        """Write configuration change to dedicated log"""
        try:
            async with aiofiles.open(self.config_log_file, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(config_change.dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to write configuration log: {e}")
    
    async def _write_audit_entry(self, entry: Dict[str, Any]):
        """Write entry to main audit log"""
        try:
            async with aiofiles.open(self.audit_log_file, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write audit entry: {e}")
    
    async def _write_integrity_log(self, entry: Dict[str, Any]):
        """Write entry to integrity log"""
        try:
            async with aiofiles.open(self.integrity_log_file, 'a', encoding='utf-8') as f:
                await f.write(json.dumps(entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write integrity log: {e}")
    
    async def _save_audit_chain(self):
        """Save audit chain to disk"""
        try:
            chain_data = [entry.dict() for entry in self.audit_chain]
            async with aiofiles.open(self.chain_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(chain_data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save audit chain: {e}")
    
    async def _load_audit_chain(self):
        """Load audit chain from disk"""
        try:
            if self.chain_file.exists():
                async with aiofiles.open(self.chain_file, 'r', encoding='utf-8') as f:
                    chain_data = json.loads(await f.read())
                self.audit_chain = [AuditChain(**entry) for entry in chain_data]
                logger.info(f"Loaded audit chain with {len(self.audit_chain)} entries")
        except Exception as e:
            logger.error(f"Failed to load audit chain: {e}")
            self.audit_chain = []
    
    async def _load_configuration_hashes(self):
        """Load stored configuration hashes"""
        # This would load from a persistent store in production
        pass
    
    async def _start_integrity_monitoring(self):
        """Start background integrity monitoring"""
        # Schedule periodic integrity verification
        async def monitor():
            while True:
                await asyncio.sleep(3600)  # Check every hour
                await self.verify_audit_integrity()
        
        asyncio.create_task(monitor())
    
    async def _get_last_verification_time(self) -> Optional[str]:
        """Get timestamp of last integrity verification"""
        # This would query the integrity log in production
        return datetime.now(timezone.utc).isoformat()

# Global centralized audit logger instance
centralized_audit_logger = CentralizedAuditLogger()