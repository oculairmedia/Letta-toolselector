#!/usr/bin/env python3
"""
Configuration validation script for Letta Tools Selector.

Validates:
1. Environment variable documentation
2. Tool limit consistency
3. Docker Compose configuration
4. API contract documentation
5. Config schema structure

Usage:
    python scripts/validate_config.py
    
Exit codes:
    0 - All validations passed
    1 - Validation errors found
    2 - Validation warnings (non-critical)
"""

import os
import sys
import yaml
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any


class ConfigValidator:
    """Validates configuration files and documentation."""
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.success: List[str] = []
    
    def validate_env_example(self) -> bool:
        """Validate .env.example has all required variables."""
        env_example_path = self.repo_root / ".env.example"
        
        if not env_example_path.exists():
            self.errors.append("❌ .env.example not found")
            return False
        
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        required_vars = [
            'MAX_TOTAL_TOOLS',
            'MAX_MCP_TOOLS',
            'MIN_MCP_TOOLS',
            'DEFAULT_DROP_RATE',
            'NEVER_DETACH_TOOLS',
            'PROTECTED_TOOLS',
            'LETTA_API_URL',
            'LETTA_PASSWORD',
            'WEAVIATE_URL',
            'OPENAI_API_KEY'
        ]
        
        missing = []
        for var in required_vars:
            if var not in content:
                missing.append(var)
        
        if missing:
            self.errors.append(f"❌ Missing variables in .env.example: {', '.join(missing)}")
            return False
        
        self.success.append("✅ All required variables documented in .env.example")
        return True
    
    def validate_tool_limits_consistency(self) -> bool:
        """Validate tool limits are consistent across configs."""
        # Check compose.yaml
        compose_path = self.repo_root / "compose.yaml"
        
        if not compose_path.exists():
            self.warnings.append("⚠️  compose.yaml not found")
            return True  # Non-critical
        
        with open(compose_path, 'r') as f:
            compose_content = f.read()
        
        limits = ['MAX_TOTAL_TOOLS', 'MAX_MCP_TOOLS', 'MIN_MCP_TOOLS']
        missing_in_compose = []
        
        for limit in limits:
            if limit not in compose_content:
                missing_in_compose.append(limit)
        
        if missing_in_compose:
            self.warnings.append(f"⚠️  Limits not in compose.yaml: {', '.join(missing_in_compose)}")
        else:
            self.success.append("✅ All tool limits defined in compose.yaml")
        
        # Extract default values from .env.example
        env_example_path = self.repo_root / ".env.example"
        if env_example_path.exists():
            with open(env_example_path, 'r') as f:
                env_content = f.read()
            
            # Parse limit values
            limit_values = {}
            for limit in limits:
                match = re.search(rf'{limit}=(\d+)', env_content)
                if match:
                    limit_values[limit] = int(match.group(1))
            
            # Validate relationships
            if all(k in limit_values for k in limits):
                max_total = limit_values['MAX_TOTAL_TOOLS']
                max_mcp = limit_values['MAX_MCP_TOOLS']
                min_mcp = limit_values['MIN_MCP_TOOLS']
                
                if max_mcp > max_total:
                    self.errors.append(f"❌ MAX_MCP_TOOLS ({max_mcp}) > MAX_TOTAL_TOOLS ({max_total})")
                    return False
                
                if min_mcp > max_mcp:
                    self.errors.append(f"❌ MIN_MCP_TOOLS ({min_mcp}) > MAX_MCP_TOOLS ({max_mcp})")
                    return False
                
                if min_mcp < 1:
                    self.warnings.append(f"⚠️  MIN_MCP_TOOLS ({min_mcp}) is less than 1")
                
                self.success.append(f"✅ Tool limit relationships valid (MIN={min_mcp} <= MAX_MCP={max_mcp} <= MAX_TOTAL={max_total})")
        
        return True
    
    def validate_docker_compose(self) -> bool:
        """Validate Docker Compose syntax."""
        compose_path = self.repo_root / "compose.yaml"
        
        if not compose_path.exists():
            self.warnings.append("⚠️  compose.yaml not found")
            return True
        
        try:
            with open(compose_path, 'r') as f:
                compose_config = yaml.safe_load(f)
            
            # Validate basic structure
            if not isinstance(compose_config, dict):
                self.errors.append("❌ compose.yaml is not a valid dict")
                return False
            
            if 'services' not in compose_config:
                self.errors.append("❌ compose.yaml missing 'services' key")
                return False
            
            # Check for key services
            expected_services = ['api-server', 'weaviate', 'sync-service']
            services = compose_config.get('services', {})
            
            missing_services = []
            for svc in expected_services:
                if svc not in services:
                    missing_services.append(svc)
            
            if missing_services:
                self.warnings.append(f"⚠️  Expected services not in compose.yaml: {', '.join(missing_services)}")
            
            self.success.append("✅ Docker Compose configuration is valid YAML")
            return True
            
        except yaml.YAMLError as e:
            self.errors.append(f"❌ Invalid YAML in compose.yaml: {e}")
            return False
    
    def validate_api_contract(self) -> bool:
        """Validate API contract documentation."""
        api_contract_path = self.repo_root / "API_CONTRACT.md"
        
        if not api_contract_path.exists():
            self.errors.append("❌ API_CONTRACT.md not found")
            return False
        
        with open(api_contract_path, 'r') as f:
            content = f.read()
        
        # Check for version information
        if not re.search(r'v\d+\.\d+\.\d+', content):
            self.warnings.append("⚠️  API_CONTRACT.md missing version number (vX.Y.Z)")
        
        # Check for required sections
        required_sections = [
            'Request Schema',
            'Response Schema',
            'Behavior Specification',
            'Protected Tools',
            'Tool Limits'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section.lower() not in content.lower():
                missing_sections.append(section)
        
        if missing_sections:
            self.warnings.append(f"⚠️  API_CONTRACT.md missing sections: {', '.join(missing_sections)}")
        else:
            self.success.append("✅ API contract has all required sections")
        
        self.success.append("✅ API contract documentation exists")
        return True
    
    def validate_config_schema(self) -> bool:
        """Validate config schema file if it exists."""
        schema_path = self.repo_root / "dashboard-backend" / "config_schema.yaml"
        
        if not schema_path.exists():
            self.warnings.append("⚠️  Config schema not found (optional)")
            return True
        
        try:
            with open(schema_path, 'r') as f:
                schema = yaml.safe_load(f)
            
            if not isinstance(schema, dict):
                self.errors.append("❌ Config schema is not a valid dict")
                return False
            
            self.success.append("✅ Config schema is valid YAML")
            return True
            
        except yaml.YAMLError as e:
            self.errors.append(f"❌ Invalid YAML in config_schema.yaml: {e}")
            return False
    
    def validate_protected_tools_config(self) -> bool:
        """Validate protected tools are documented."""
        env_example_path = self.repo_root / ".env.example"
        
        if not env_example_path.exists():
            return True  # Already checked elsewhere
        
        with open(env_example_path, 'r') as f:
            content = f.read()
        
        # Check both NEVER_DETACH_TOOLS and PROTECTED_TOOLS are documented
        has_never_detach = 'NEVER_DETACH_TOOLS' in content
        has_protected = 'PROTECTED_TOOLS' in content
        
        if not (has_never_detach or has_protected):
            self.errors.append("❌ Neither NEVER_DETACH_TOOLS nor PROTECTED_TOOLS documented")
            return False
        
        # Check for documentation explaining the relationship
        if has_never_detach and has_protected:
            self.success.append("✅ Both NEVER_DETACH_TOOLS and PROTECTED_TOOLS documented (dual support)")
        
        return True
    
    def validate_all(self) -> int:
        """Run all validations and return exit code."""
        print("🔍 Running configuration validation...\n")
        
        validators = [
            ("Environment Documentation", self.validate_env_example),
            ("Tool Limits Consistency", self.validate_tool_limits_consistency),
            ("Docker Compose", self.validate_docker_compose),
            ("API Contract", self.validate_api_contract),
            ("Config Schema", self.validate_config_schema),
            ("Protected Tools", self.validate_protected_tools_config),
        ]
        
        for name, validator in validators:
            print(f"Validating {name}...")
            validator()
            print()
        
        # Print results
        print("=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        if self.success:
            print("\n✅ SUCCESS:")
            for msg in self.success:
                print(f"  {msg}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for msg in self.warnings:
                print(f"  {msg}")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for msg in self.errors:
                print(f"  {msg}")
        
        print("\n" + "=" * 60)
        
        # Determine exit code
        if self.errors:
            print("❌ Validation FAILED with errors")
            return 1
        elif self.warnings:
            print("⚠️  Validation passed with warnings")
            return 2
        else:
            print("✅ All validations PASSED")
            return 0


def main():
    """Main entry point."""
    # Find repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    
    validator = ConfigValidator(repo_root)
    exit_code = validator.validate_all()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
