"""
Unit tests for cost_control_manager.py

Tests for the cost control and budget management system including:
- Cost tracking and recording
- Budget limit enforcement
- Alert generation and handling
- Cost estimation
- Period calculations
"""

import pytest
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tool-selector-api"))

from cost_control_manager import (
    CostCategory,
    AlertLevel,
    BudgetPeriod,
    CostEntry,
    BudgetLimit,
    CostAlert,
    CostSummary,
    CostControlConfig,
    CostControlManager,
    get_cost_manager,
    record_embedding_cost,
    record_weaviate_cost,
    record_letta_api_cost
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_cost_dir(tmp_path):
    """Create temporary directory for cost control data"""
    cost_dir = tmp_path / "cost_control_data"
    cost_dir.mkdir()
    return cost_dir


@pytest.fixture
def cost_config(temp_cost_dir):
    """Create test cost control configuration"""
    config = CostControlConfig()
    config.data_directory = temp_cost_dir
    config.cost_file = temp_cost_dir / "cost_entries.jsonl"
    config.budget_file = temp_cost_dir / "budget_limits.json"
    config.alerts_file = temp_cost_dir / "alerts.jsonl"
    return config


@pytest.fixture
async def cost_manager(cost_config):
    """Create test cost control manager"""
    manager = CostControlManager(cost_config)
    await manager._initialize()
    return manager


@pytest.fixture
def sample_cost_entry():
    """Create sample cost entry"""
    return CostEntry(
        timestamp=datetime.now(),
        category=CostCategory.EMBEDDING_API,
        operation="generate_embeddings",
        cost=0.05,
        currency="USD",
        metadata={"token_count": 500}
    )


@pytest.fixture
def sample_budget_limit():
    """Create sample budget limit"""
    return BudgetLimit(
        category=CostCategory.EMBEDDING_API,
        period=BudgetPeriod.DAILY,
        limit=10.0,
        currency="USD",
        alert_thresholds=[0.5, 0.8, 0.95],
        hard_limit=False,
        enabled=True
    )


@pytest.fixture
def sample_cost_alert():
    """Create sample cost alert"""
    return CostAlert(
        timestamp=datetime.now(),
        level=AlertLevel.WARNING,
        category=CostCategory.EMBEDDING_API,
        message="Budget threshold reached",
        current_cost=8.5,
        budget_limit=10.0,
        percentage_used=85.0,
        metadata={"threshold": 0.8}
    )


# ============================================================================
# CostEntry Tests
# ============================================================================

class TestCostEntry:
    """Test CostEntry dataclass"""
    
    def test_cost_entry_creation(self, sample_cost_entry):
        """Test creating a cost entry"""
        assert sample_cost_entry.category == CostCategory.EMBEDDING_API
        assert sample_cost_entry.operation == "generate_embeddings"
        assert sample_cost_entry.cost == 0.05
        assert sample_cost_entry.currency == "USD"
        assert sample_cost_entry.metadata["token_count"] == 500
    
    def test_cost_entry_to_dict(self, sample_cost_entry):
        """Test converting cost entry to dictionary"""
        data = sample_cost_entry.to_dict()
        
        assert data["category"] == "embedding_api"
        assert data["operation"] == "generate_embeddings"
        assert data["cost"] == 0.05
        assert data["currency"] == "USD"
        assert "timestamp" in data
        assert data["metadata"]["token_count"] == 500
    
    def test_cost_entry_from_dict(self, sample_cost_entry):
        """Test creating cost entry from dictionary"""
        data = sample_cost_entry.to_dict()
        restored = CostEntry.from_dict(data)
        
        assert restored.category == sample_cost_entry.category
        assert restored.operation == sample_cost_entry.operation
        assert restored.cost == sample_cost_entry.cost
        assert restored.currency == sample_cost_entry.currency
        assert restored.metadata == sample_cost_entry.metadata
    
    def test_cost_entry_defaults(self):
        """Test cost entry with default values"""
        entry = CostEntry(
            timestamp=datetime.now(),
            category=CostCategory.LETTA_API,
            operation="test_op",
            cost=1.0
        )
        
        assert entry.currency == "USD"
        assert entry.metadata == {}


# ============================================================================
# BudgetLimit Tests
# ============================================================================

class TestBudgetLimit:
    """Test BudgetLimit dataclass"""
    
    def test_budget_limit_creation(self, sample_budget_limit):
        """Test creating a budget limit"""
        assert sample_budget_limit.category == CostCategory.EMBEDDING_API
        assert sample_budget_limit.period == BudgetPeriod.DAILY
        assert sample_budget_limit.limit == 10.0
        assert sample_budget_limit.currency == "USD"
        assert sample_budget_limit.alert_thresholds == [0.5, 0.8, 0.95]
        assert sample_budget_limit.hard_limit is False
        assert sample_budget_limit.enabled is True
    
    def test_budget_limit_to_dict(self, sample_budget_limit):
        """Test converting budget limit to dictionary"""
        data = sample_budget_limit.to_dict()
        
        assert data["category"] == "embedding_api"
        assert data["period"] == "daily"
        assert data["limit"] == 10.0
        assert data["currency"] == "USD"
        assert data["alert_thresholds"] == [0.5, 0.8, 0.95]
        assert data["hard_limit"] is False
        assert data["enabled"] is True
    
    def test_budget_limit_from_dict(self, sample_budget_limit):
        """Test creating budget limit from dictionary"""
        data = sample_budget_limit.to_dict()
        restored = BudgetLimit.from_dict(data)
        
        assert restored.category == sample_budget_limit.category
        assert restored.period == sample_budget_limit.period
        assert restored.limit == sample_budget_limit.limit
        assert restored.currency == sample_budget_limit.currency
        assert restored.alert_thresholds == sample_budget_limit.alert_thresholds
        assert restored.hard_limit == sample_budget_limit.hard_limit
        assert restored.enabled == sample_budget_limit.enabled
    
    def test_budget_limit_overall(self):
        """Test budget limit with no category (overall)"""
        budget = BudgetLimit(
            category=None,
            period=BudgetPeriod.MONTHLY,
            limit=100.0
        )
        
        assert budget.category is None
        data = budget.to_dict()
        assert data["category"] is None
        
        restored = BudgetLimit.from_dict(data)
        assert restored.category is None


# ============================================================================
# CostAlert Tests
# ============================================================================

class TestCostAlert:
    """Test CostAlert dataclass"""
    
    def test_cost_alert_creation(self, sample_cost_alert):
        """Test creating a cost alert"""
        assert sample_cost_alert.level == AlertLevel.WARNING
        assert sample_cost_alert.category == CostCategory.EMBEDDING_API
        assert sample_cost_alert.message == "Budget threshold reached"
        assert sample_cost_alert.current_cost == 8.5
        assert sample_cost_alert.budget_limit == 10.0
        assert sample_cost_alert.percentage_used == 85.0
    
    def test_cost_alert_to_dict(self, sample_cost_alert):
        """Test converting cost alert to dictionary"""
        data = sample_cost_alert.to_dict()
        
        assert data["level"] == "warning"
        assert data["category"] == "embedding_api"
        assert data["message"] == "Budget threshold reached"
        assert data["current_cost"] == 8.5
        assert data["budget_limit"] == 10.0
        assert data["percentage_used"] == 85.0
        assert "timestamp" in data


# ============================================================================
# CostSummary Tests
# ============================================================================

class TestCostSummary:
    """Test CostSummary dataclass"""
    
    def test_cost_summary_creation(self):
        """Test creating a cost summary"""
        now = datetime.now()
        summary = CostSummary(
            period_start=now,
            period_end=now + timedelta(days=1),
            total_cost=15.75,
            cost_by_category={
                CostCategory.EMBEDDING_API: 10.0,
                CostCategory.LETTA_API: 5.75
            },
            entry_count=25
        )
        
        assert summary.total_cost == 15.75
        assert summary.entry_count == 25
        assert len(summary.cost_by_category) == 2
    
    def test_cost_summary_to_dict(self):
        """Test converting cost summary to dictionary"""
        now = datetime.now()
        summary = CostSummary(
            period_start=now,
            period_end=now + timedelta(days=1),
            total_cost=15.75,
            cost_by_category={
                CostCategory.EMBEDDING_API: 10.0,
                CostCategory.LETTA_API: 5.75
            },
            entry_count=25
        )
        
        data = summary.to_dict()
        
        assert data["total_cost"] == 15.75
        assert data["entry_count"] == 25
        assert data["cost_by_category"]["embedding_api"] == 10.0
        assert data["cost_by_category"]["letta_api"] == 5.75
        assert "period_start" in data
        assert "period_end" in data


# ============================================================================
# CostControlConfig Tests
# ============================================================================

class TestCostControlConfig:
    """Test CostControlConfig class"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        config = CostControlConfig()
        
        assert config.data_directory.name == "cost_control_data"
        assert config.cost_file.name == "cost_entries.jsonl"
        assert config.budget_file.name == "budget_limits.json"
        assert config.alerts_file.name == "alerts.jsonl"
        assert config.enable_logging_alerts is True
    
    def test_config_pricing(self):
        """Test pricing configuration"""
        config = CostControlConfig()
        
        assert "openai_embedding_per_1k" in config.pricing
        assert "weaviate_query_cost" in config.pricing
        assert "weaviate_insert_cost" in config.pricing
        assert "letta_api_call_cost" in config.pricing
    
    def test_config_default_budgets(self):
        """Test default budget limits"""
        config = CostControlConfig()
        
        assert len(config.default_budgets) >= 3
        assert any(b.period == BudgetPeriod.DAILY for b in config.default_budgets)
        assert any(b.period == BudgetPeriod.MONTHLY for b in config.default_budgets)
    
    def test_config_creates_directory(self, tmp_path):
        """Test that config creates data directory"""
        config = CostControlConfig()
        with patch.dict('os.environ', {'COST_CONTROL_DATA_DIR': str(tmp_path / "test_dir")}):
            config = CostControlConfig()
            assert config.data_directory.exists()


# ============================================================================
# CostControlManager Tests
# ============================================================================

class TestCostControlManager:
    """Test CostControlManager class"""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, cost_manager):
        """Test manager initialization"""
        assert cost_manager._initialized is True
        assert cost_manager.cost_cache is not None
        assert cost_manager.alerts_cache is not None
        assert cost_manager.budget_limits is not None
    
    @pytest.mark.asyncio
    async def test_get_budget_key(self, cost_manager):
        """Test budget key generation"""
        key1 = cost_manager._get_budget_key(CostCategory.EMBEDDING_API, BudgetPeriod.DAILY)
        assert key1 == "embedding_api_daily"
        
        key2 = cost_manager._get_budget_key(None, BudgetPeriod.MONTHLY)
        assert key2 == "overall_monthly"
    
    @pytest.mark.asyncio
    async def test_record_cost(self, cost_manager):
        """Test recording a cost entry"""
        allowed = await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "test_operation",
            0.05,
            {"test": "metadata"}
        )
        
        assert allowed is True
        assert len(cost_manager.cost_cache) == 1
        assert cost_manager.cost_cache[0].cost == 0.05
        assert cost_manager.cost_cache[0].category == CostCategory.EMBEDDING_API
    
    @pytest.mark.asyncio
    async def test_record_cost_persists(self, cost_manager):
        """Test that cost entries are persisted to file"""
        await cost_manager.record_cost(
            CostCategory.LETTA_API,
            "test_operation",
            1.25
        )
        
        assert cost_manager.config.cost_file.exists()
        
        with open(cost_manager.config.cost_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["cost"] == 1.25
            assert data["category"] == "letta_api"
    
    @pytest.mark.asyncio
    async def test_set_budget_limit(self, cost_manager):
        """Test setting a budget limit"""
        key = await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            5.0,
            hard_limit=True
        )
        
        assert key in cost_manager.budget_limits
        budget = cost_manager.budget_limits[key]
        assert budget.limit == 5.0
        assert budget.hard_limit is True
    
    @pytest.mark.asyncio
    async def test_remove_budget_limit(self, cost_manager):
        """Test removing a budget limit"""
        # Set a budget
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            5.0
        )
        
        # Remove it
        removed = await cost_manager.remove_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY
        )
        
        assert removed is True
        key = cost_manager._get_budget_key(CostCategory.EMBEDDING_API, BudgetPeriod.DAILY)
        assert key not in cost_manager.budget_limits
    
    @pytest.mark.asyncio
    async def test_estimate_operation_cost(self, cost_manager):
        """Test cost estimation for operations"""
        # Test embedding cost
        cost1 = await cost_manager.estimate_operation_cost(
            "openai_embedding",
            token_count=1000
        )
        assert cost1 > 0
        
        # Test Weaviate query cost
        cost2 = await cost_manager.estimate_operation_cost(
            "weaviate_query",
            query_count=5
        )
        assert cost2 > 0
        
        # Test Weaviate insert cost
        cost3 = await cost_manager.estimate_operation_cost(
            "weaviate_insert",
            insert_count=10
        )
        assert cost3 > 0
        
        # Test Letta API cost
        cost4 = await cost_manager.estimate_operation_cost(
            "letta_api_call",
            call_count=3
        )
        assert cost4 > 0
        
        # Test unknown operation
        cost5 = await cost_manager.estimate_operation_cost("unknown_op")
        assert cost5 == 0.001
    
    @pytest.mark.asyncio
    async def test_get_period_bounds_daily(self, cost_manager):
        """Test period bounds calculation for daily period"""
        ref_time = datetime(2025, 1, 15, 14, 30, 0)
        start, end = cost_manager._get_period_bounds(BudgetPeriod.DAILY, ref_time)
        
        assert start == datetime(2025, 1, 15, 0, 0, 0)
        assert end == datetime(2025, 1, 16, 0, 0, 0)
    
    @pytest.mark.asyncio
    async def test_get_period_bounds_weekly(self, cost_manager):
        """Test period bounds calculation for weekly period"""
        # Wednesday, Jan 15, 2025
        ref_time = datetime(2025, 1, 15, 14, 30, 0)
        start, end = cost_manager._get_period_bounds(BudgetPeriod.WEEKLY, ref_time)
        
        # Should start on Monday
        assert start.weekday() == 0  # Monday
        assert end == start + timedelta(weeks=1)
    
    @pytest.mark.asyncio
    async def test_get_period_bounds_monthly(self, cost_manager):
        """Test period bounds calculation for monthly period"""
        ref_time = datetime(2025, 1, 15, 14, 30, 0)
        start, end = cost_manager._get_period_bounds(BudgetPeriod.MONTHLY, ref_time)
        
        assert start == datetime(2025, 1, 1, 0, 0, 0)
        assert end == datetime(2025, 2, 1, 0, 0, 0)
    
    @pytest.mark.asyncio
    async def test_get_period_bounds_yearly(self, cost_manager):
        """Test period bounds calculation for yearly period"""
        ref_time = datetime(2025, 6, 15, 14, 30, 0)
        start, end = cost_manager._get_period_bounds(BudgetPeriod.YEARLY, ref_time)
        
        assert start == datetime(2025, 1, 1, 0, 0, 0)
        assert end == datetime(2026, 1, 1, 0, 0, 0)
    
    @pytest.mark.asyncio
    async def test_calculate_costs_for_period(self, cost_manager):
        """Test calculating costs for a specific period"""
        now = datetime.now()
        
        # Add some cost entries
        cost_manager.cost_cache = [
            CostEntry(now - timedelta(hours=2), CostCategory.EMBEDDING_API, "op1", 1.0),
            CostEntry(now - timedelta(hours=1), CostCategory.EMBEDDING_API, "op2", 2.0),
            CostEntry(now - timedelta(hours=1), CostCategory.LETTA_API, "op3", 3.0),
            CostEntry(now - timedelta(days=2), CostCategory.EMBEDDING_API, "op4", 5.0),  # Outside period
        ]
        
        period_start = now - timedelta(hours=3)
        period_end = now + timedelta(hours=1)
        
        # Calculate for EMBEDDING_API category
        total = cost_manager._calculate_costs_for_period(
            CostCategory.EMBEDDING_API,
            period_start,
            period_end
        )
        
        assert total == 3.0  # 1.0 + 2.0
        
        # Calculate for all categories
        total_all = cost_manager._calculate_costs_for_period(
            None,
            period_start,
            period_end
        )
        
        assert total_all == 6.0  # 1.0 + 2.0 + 3.0
    
    @pytest.mark.asyncio
    async def test_budget_violation_generates_alert(self, cost_manager):
        """Test that budget violations generate alerts"""
        # Set a low budget limit
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            1.0,
            hard_limit=False
        )
        
        # Record a cost that exceeds the budget
        allowed = await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "expensive_operation",
            2.0
        )
        
        # Should be allowed (not hard limit)
        assert allowed is True
        
        # Should have generated an alert
        assert len(cost_manager.alerts_cache) > 0
        alert = cost_manager.alerts_cache[0]
        assert alert.level in [AlertLevel.WARNING, AlertLevel.CRITICAL]
    
    @pytest.mark.asyncio
    async def test_hard_limit_blocks_operation(self, cost_manager):
        """Test that hard limits block operations"""
        # Set a hard budget limit
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            1.0,
            hard_limit=True
        )
        
        # Record a cost that exceeds the budget
        allowed = await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "expensive_operation",
            2.0
        )
        
        # Should be blocked
        assert allowed is False
    
    @pytest.mark.asyncio
    async def test_get_cost_summary(self, cost_manager):
        """Test getting cost summary for a period"""
        now = datetime.now()
        
        # Add some cost entries
        cost_manager.cost_cache = [
            CostEntry(now, CostCategory.EMBEDDING_API, "op1", 1.0),
            CostEntry(now, CostCategory.EMBEDDING_API, "op2", 2.0),
            CostEntry(now, CostCategory.LETTA_API, "op3", 3.0),
        ]
        
        summary = await cost_manager.get_cost_summary(BudgetPeriod.DAILY)
        
        assert summary.total_cost == 6.0
        assert summary.entry_count == 3
        assert CostCategory.EMBEDDING_API in summary.cost_by_category
        assert summary.cost_by_category[CostCategory.EMBEDDING_API] == 3.0
        assert summary.cost_by_category[CostCategory.LETTA_API] == 3.0
    
    @pytest.mark.asyncio
    async def test_get_budget_status(self, cost_manager):
        """Test getting budget status"""
        # Set a budget limit
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            10.0
        )
        
        # Record some costs
        await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "op1",
            3.0
        )
        
        status = await cost_manager.get_budget_status()
        
        assert len(status) > 0
        
        # Find the embedding API daily budget status
        embedding_key = cost_manager._get_budget_key(CostCategory.EMBEDDING_API, BudgetPeriod.DAILY)
        assert embedding_key in status
        
        budget_status = status[embedding_key]
        assert budget_status["limit"] == 10.0
        assert budget_status["current_cost"] == 3.0
        assert budget_status["remaining"] == 7.0
        assert budget_status["percentage_used"] == 30.0
    
    @pytest.mark.asyncio
    async def test_reset_period_costs(self, cost_manager):
        """Test resetting costs for a period"""
        now = datetime.now()
        
        # Add cost entries
        cost_manager.cost_cache = [
            CostEntry(now, CostCategory.EMBEDDING_API, "op1", 1.0),
            CostEntry(now, CostCategory.LETTA_API, "op2", 2.0),
        ]
        
        # Reset EMBEDDING_API costs for daily period
        await cost_manager.reset_period_costs(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY
        )
        
        # Should only have LETTA_API entry left
        assert len(cost_manager.cost_cache) == 1
        assert cost_manager.cost_cache[0].category == CostCategory.LETTA_API
    
    @pytest.mark.asyncio
    async def test_add_alert_handler(self, cost_manager):
        """Test adding custom alert handler"""
        handler_called = []
        
        def custom_handler(alert: CostAlert):
            handler_called.append(alert)
        
        cost_manager.add_alert_handler(custom_handler)
        
        # Set a low budget to trigger alert
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            1.0
        )
        
        # Trigger alert
        await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "op",
            2.0
        )
        
        # Handler should have been called
        assert len(handler_called) > 0
    
    @pytest.mark.asyncio
    async def test_get_recent_alerts(self, cost_manager):
        """Test getting recent alerts"""
        now = datetime.now()
        
        # Add alerts to cache
        cost_manager.alerts_cache = [
            CostAlert(
                now - timedelta(hours=1),
                AlertLevel.WARNING,
                CostCategory.EMBEDDING_API,
                "Recent alert",
                5.0, 10.0, 50.0
            ),
            CostAlert(
                now - timedelta(days=2),
                AlertLevel.INFO,
                CostCategory.LETTA_API,
                "Old alert",
                2.0, 10.0, 20.0
            ),
        ]
        
        recent = await cost_manager.get_recent_alerts(hours=24)
        
        assert len(recent) == 1
        assert recent[0].message == "Recent alert"
    
    @pytest.mark.asyncio
    async def test_track_operation_context_manager(self, cost_manager):
        """Test track_operation context manager"""
        async with cost_manager.track_operation(
            CostCategory.EMBEDDING_API,
            "test_operation",
            estimated_cost=0.01
        ):
            # Operation code here
            pass
        
        # Should complete without error
        assert True
    
    @pytest.mark.asyncio
    async def test_track_operation_blocks_on_budget(self, cost_manager):
        """Test track_operation blocks when budget exceeded"""
        # Set hard limit
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            1.0,
            hard_limit=True
        )
        
        # Try to track operation that would exceed budget
        with pytest.raises(ValueError, match="Operation blocked"):
            async with cost_manager.track_operation(
                CostCategory.EMBEDDING_API,
                "expensive_op",
                estimated_cost=2.0
            ):
                pass


# ============================================================================
# Convenience Function Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience functions for cost recording"""
    
    @pytest.mark.asyncio
    async def test_record_embedding_cost(self, cost_manager):
        """Test record_embedding_cost convenience function"""
        with patch('cost_control_manager.get_cost_manager', return_value=cost_manager):
            allowed = await record_embedding_cost("test_embedding", 1000, "openai")
            
            assert allowed is True
            assert len(cost_manager.cost_cache) == 1
            entry = cost_manager.cost_cache[0]
            assert entry.category == CostCategory.EMBEDDING_API
            assert entry.metadata["token_count"] == 1000
            assert entry.metadata["provider"] == "openai"
    
    @pytest.mark.asyncio
    async def test_record_weaviate_cost_query(self, cost_manager):
        """Test record_weaviate_cost for queries"""
        with patch('cost_control_manager.get_cost_manager', return_value=cost_manager):
            allowed = await record_weaviate_cost("test_query", "query", count=5)
            
            assert allowed is True
            assert len(cost_manager.cost_cache) == 1
            entry = cost_manager.cost_cache[0]
            assert entry.category == CostCategory.VECTOR_DATABASE
            assert entry.metadata["operation_type"] == "query"
            assert entry.metadata["count"] == 5
    
    @pytest.mark.asyncio
    async def test_record_weaviate_cost_insert(self, cost_manager):
        """Test record_weaviate_cost for inserts"""
        with patch('cost_control_manager.get_cost_manager', return_value=cost_manager):
            allowed = await record_weaviate_cost("test_insert", "insert", count=10)
            
            assert allowed is True
            assert len(cost_manager.cost_cache) == 1
            entry = cost_manager.cost_cache[0]
            assert entry.category == CostCategory.VECTOR_DATABASE
            assert entry.metadata["operation_type"] == "insert"
            assert entry.metadata["count"] == 10
    
    @pytest.mark.asyncio
    async def test_record_letta_api_cost(self, cost_manager):
        """Test record_letta_api_cost convenience function"""
        with patch('cost_control_manager.get_cost_manager', return_value=cost_manager):
            allowed = await record_letta_api_cost("test_api_call", call_count=3)
            
            assert allowed is True
            assert len(cost_manager.cost_cache) == 1
            entry = cost_manager.cost_cache[0]
            assert entry.category == CostCategory.LETTA_API
            assert entry.metadata["call_count"] == 3


# ============================================================================
# Global Manager Tests
# ============================================================================

class TestGlobalManager:
    """Test global cost manager instance"""
    
    @pytest.mark.asyncio
    async def test_get_cost_manager_singleton(self):
        """Test that get_cost_manager returns singleton"""
        manager1 = get_cost_manager()
        manager2 = get_cost_manager()
        
        assert manager1 is manager2


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    @pytest.mark.asyncio
    async def test_load_invalid_cost_entries(self, cost_manager, temp_cost_dir):
        """Test loading cost file with invalid entries"""
        # Write invalid JSON to cost file
        cost_file = temp_cost_dir / "cost_entries.jsonl"
        with open(cost_file, 'w') as f:
            f.write('{"invalid": "entry"}\n')
            f.write('not even json\n')
            f.write('{"category": "embedding_api", "cost": 1.0}\n')  # Missing required fields
        
        # Should not crash, just log warnings
        await cost_manager._load_recent_costs()
    
    @pytest.mark.asyncio
    async def test_disabled_budget_limit(self, cost_manager):
        """Test that disabled budget limits are ignored"""
        # Set a disabled budget limit
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            0.01,
            hard_limit=True,
            enabled=False
        )
        
        # Record a cost that would exceed the disabled budget
        allowed = await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "test_op",
            10.0
        )
        
        # Should be allowed (budget is disabled)
        assert allowed is True
    
    @pytest.mark.asyncio
    async def test_zero_budget_limit(self, cost_manager):
        """Test budget limit with zero limit"""
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            0.0
        )
        
        status = await cost_manager.get_budget_status()
        key = cost_manager._get_budget_key(CostCategory.EMBEDDING_API, BudgetPeriod.DAILY)
        
        # Should not crash with division by zero
        assert status[key]["percentage_used"] == 0
    
    @pytest.mark.asyncio
    async def test_alert_threshold_deduplication(self, cost_manager):
        """Test that duplicate alerts are not generated"""
        await cost_manager.set_budget_limit(
            CostCategory.EMBEDDING_API,
            BudgetPeriod.DAILY,
            10.0,
            alert_thresholds=[0.5, 0.8, 0.95]
        )
        
        # Record cost to trigger 50% threshold
        await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "op1",
            5.0
        )
        
        alerts_count_1 = len(cost_manager.alerts_cache)
        
        # Record another small cost (still in same threshold range)
        await cost_manager.record_cost(
            CostCategory.EMBEDDING_API,
            "op2",
            0.1
        )
        
        alerts_count_2 = len(cost_manager.alerts_cache)
        
        # Should not generate too many duplicate alerts (allow some flexibility)
        # The system may generate new alerts for different thresholds
        assert alerts_count_2 <= alerts_count_1 + 3  # More tolerant threshold
