"""
LDTS-58: Cost Controls and Budget Management System

A comprehensive system for tracking, controlling, and managing costs associated with
AI operations including embedding generation, vector database operations, and API calls.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from contextlib import asynccontextmanager
# import aiofiles  # Will use standard file operations for compatibility
import threading
from pathlib import Path


logger = logging.getLogger(__name__)


class CostCategory(Enum):
    """Categories for different types of costs"""
    EMBEDDING_API = "embedding_api"           # OpenAI/Ollama embedding calls
    VECTOR_DATABASE = "vector_database"       # Weaviate operations
    LETTA_API = "letta_api"                  # Letta API calls
    COMPUTE_RESOURCES = "compute_resources"   # CPU/Memory usage
    DATA_STORAGE = "data_storage"            # Storage costs
    EXPERIMENTS = "experiments"               # Experimental operations
    OTHER = "other"                          # Other miscellaneous costs


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class BudgetPeriod(Enum):
    """Budget period types"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class CostEntry:
    """Individual cost entry record"""
    timestamp: datetime
    category: CostCategory
    operation: str
    cost: float
    currency: str = "USD"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
            "operation": self.operation,
            "cost": self.cost,
            "currency": self.currency,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CostEntry':
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            category=CostCategory(data["category"]),
            operation=data["operation"],
            cost=data["cost"],
            currency=data.get("currency", "USD"),
            metadata=data.get("metadata", {})
        )


@dataclass
class BudgetLimit:
    """Budget limit configuration"""
    category: Optional[CostCategory]  # None for overall budget
    period: BudgetPeriod
    limit: float
    currency: str = "USD"
    alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.8, 0.95])  # 50%, 80%, 95%
    hard_limit: bool = False  # If True, operations are blocked when exceeded
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "category": self.category.value if self.category else None,
            "period": self.period.value,
            "limit": self.limit,
            "currency": self.currency,
            "alert_thresholds": self.alert_thresholds,
            "hard_limit": self.hard_limit,
            "enabled": self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BudgetLimit':
        """Create from dictionary"""
        return cls(
            category=CostCategory(data["category"]) if data.get("category") else None,
            period=BudgetPeriod(data["period"]),
            limit=data["limit"],
            currency=data.get("currency", "USD"),
            alert_thresholds=data.get("alert_thresholds", [0.5, 0.8, 0.95]),
            hard_limit=data.get("hard_limit", False),
            enabled=data.get("enabled", True)
        )


@dataclass 
class CostAlert:
    """Cost alert notification"""
    timestamp: datetime
    level: AlertLevel
    category: Optional[CostCategory]
    message: str
    current_cost: float
    budget_limit: float
    percentage_used: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "category": self.category.value if self.category else None,
            "message": self.message,
            "current_cost": self.current_cost,
            "budget_limit": self.budget_limit,
            "percentage_used": self.percentage_used,
            "metadata": self.metadata
        }


@dataclass
class CostSummary:
    """Cost summary for a period"""
    period_start: datetime
    period_end: datetime
    total_cost: float
    cost_by_category: Dict[CostCategory, float]
    currency: str = "USD"
    entry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_cost": self.total_cost,
            "cost_by_category": {k.value: v for k, v in self.cost_by_category.items()},
            "currency": self.currency,
            "entry_count": self.entry_count
        }


class CostControlConfig:
    """Configuration for the cost control system"""
    
    def __init__(self):
        self.data_directory = Path(os.getenv("COST_CONTROL_DATA_DIR", "./cost_control_data"))
        self.cost_file = self.data_directory / "cost_entries.jsonl"
        self.budget_file = self.data_directory / "budget_limits.json"
        self.alerts_file = self.data_directory / "alerts.jsonl"
        
        # Pricing configuration (can be overridden by environment)
        self.pricing = {
            "openai_embedding_per_1k": float(os.getenv("OPENAI_EMBEDDING_COST_PER_1K", "0.0001")),  # $0.0001 per 1K tokens
            "weaviate_query_cost": float(os.getenv("WEAVIATE_QUERY_COST", "0.00001")),  # Estimate per query
            "weaviate_insert_cost": float(os.getenv("WEAVIATE_INSERT_COST", "0.00002")),  # Estimate per insert
            "letta_api_call_cost": float(os.getenv("LETTA_API_CALL_COST", "0.001")),  # Estimate per API call
        }
        
        # Default budget limits
        self.default_budgets = [
            BudgetLimit(None, BudgetPeriod.DAILY, float(os.getenv("DAILY_BUDGET_LIMIT", "10.0"))),
            BudgetLimit(None, BudgetPeriod.MONTHLY, float(os.getenv("MONTHLY_BUDGET_LIMIT", "250.0"))),
            BudgetLimit(CostCategory.EMBEDDING_API, BudgetPeriod.DAILY, float(os.getenv("DAILY_EMBEDDING_LIMIT", "5.0"))),
        ]
        
        # Alert configuration
        self.alert_handlers: List[Callable[[CostAlert], None]] = []
        self.enable_logging_alerts = os.getenv("ENABLE_LOGGING_ALERTS", "true").lower() == "true"
        
        # Ensure data directory exists
        self.data_directory.mkdir(parents=True, exist_ok=True)


class CostControlManager:
    """Main cost control and budget management system"""
    
    def __init__(self, config: Optional[CostControlConfig] = None):
        self.config = config or CostControlConfig()
        self.budget_limits: Dict[str, BudgetLimit] = {}
        self.cost_cache: List[CostEntry] = []
        self.alerts_cache: List[CostAlert] = []
        self._lock = threading.Lock()
        self._initialized = False
        
        # Load existing data
        asyncio.create_task(self._initialize())
    
    async def _initialize(self):
        """Initialize the cost control system"""
        try:
            await self._load_budget_limits()
            await self._load_recent_costs()
            self._initialized = True
            logger.info("Cost Control Manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Cost Control Manager: {e}")
    
    def _get_budget_key(self, category: Optional[CostCategory], period: BudgetPeriod) -> str:
        """Generate unique key for budget limits"""
        cat_key = category.value if category else "overall"
        return f"{cat_key}_{period.value}"
    
    async def _load_budget_limits(self):
        """Load budget limits from file"""
        try:
            if self.config.budget_file.exists():
                with open(self.config.budget_file, 'r') as f:
                    content = f.read()
                    data = json.loads(content)
                    
                    for item in data:
                        budget = BudgetLimit.from_dict(item)
                        key = self._get_budget_key(budget.category, budget.period)
                        self.budget_limits[key] = budget
            else:
                # Load default budgets
                for budget in self.config.default_budgets:
                    key = self._get_budget_key(budget.category, budget.period)
                    self.budget_limits[key] = budget
                await self._save_budget_limits()
                
        except Exception as e:
            logger.error(f"Failed to load budget limits: {e}")
            # Load defaults on error
            for budget in self.config.default_budgets:
                key = self._get_budget_key(budget.category, budget.period)
                self.budget_limits[key] = budget
    
    async def _save_budget_limits(self):
        """Save budget limits to file"""
        try:
            data = [budget.to_dict() for budget in self.budget_limits.values()]
            with open(self.config.budget_file, 'w') as f:
                f.write(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save budget limits: {e}")
    
    async def _load_recent_costs(self, days: int = 7):
        """Load recent cost entries from file"""
        try:
            if not self.config.cost_file.exists():
                return
                
            cutoff_time = datetime.now() - timedelta(days=days)
            
            with open(self.config.cost_file, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            entry = CostEntry.from_dict(data)
                            if entry.timestamp > cutoff_time:
                                self.cost_cache.append(entry)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Failed to parse cost entry: {e}")
                            
        except Exception as e:
            logger.error(f"Failed to load recent costs: {e}")
    
    async def _append_cost_entry(self, entry: CostEntry):
        """Append cost entry to file"""
        try:
            with open(self.config.cost_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to append cost entry: {e}")
    
    async def _append_alert(self, alert: CostAlert):
        """Append alert to file"""
        try:
            with open(self.config.alerts_file, 'a') as f:
                f.write(json.dumps(alert.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Failed to append alert: {e}")
    
    def _get_period_bounds(self, period: BudgetPeriod, reference_time: Optional[datetime] = None) -> tuple[datetime, datetime]:
        """Get start and end times for a budget period"""
        if reference_time is None:
            reference_time = datetime.now()
            
        if period == BudgetPeriod.HOURLY:
            start = reference_time.replace(minute=0, second=0, microsecond=0)
            end = start + timedelta(hours=1)
        elif period == BudgetPeriod.DAILY:
            start = reference_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            days_since_monday = reference_time.weekday()
            start = reference_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            start = reference_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        elif period == BudgetPeriod.QUARTERLY:
            quarter = (reference_time.month - 1) // 3 + 1
            start = reference_time.replace(month=(quarter - 1) * 3 + 1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end_month = quarter * 3
            if end_month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=end_month + 1)
        elif period == BudgetPeriod.YEARLY:
            start = reference_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(year=start.year + 1)
        else:
            raise ValueError(f"Unsupported period: {period}")
            
        return start, end
    
    def _calculate_costs_for_period(self, category: Optional[CostCategory], period_start: datetime, period_end: datetime) -> float:
        """Calculate total costs for a category and period"""
        total = 0.0
        
        for entry in self.cost_cache:
            if period_start <= entry.timestamp < period_end:
                if category is None or entry.category == category:
                    total += entry.cost
                    
        return total
    
    async def _check_budget_violations(self, new_cost: float, category: CostCategory) -> List[CostAlert]:
        """Check for budget violations and generate alerts"""
        alerts = []
        
        for budget_key, budget in self.budget_limits.items():
            if not budget.enabled:
                continue
                
            # Check if this budget applies to the category
            if budget.category is not None and budget.category != category:
                continue
            
            # Get period bounds
            period_start, period_end = self._get_period_bounds(budget.period)
            
            # Calculate current costs for this period
            current_cost = self._calculate_costs_for_period(budget.category, period_start, period_end) + new_cost
            
            # Check against budget limit
            if current_cost > budget.limit:
                # Budget exceeded
                percentage_used = (current_cost / budget.limit) * 100
                alert = CostAlert(
                    timestamp=datetime.now(),
                    level=AlertLevel.CRITICAL if budget.hard_limit else AlertLevel.WARNING,
                    category=budget.category,
                    message=f"Budget limit exceeded: ${current_cost:.4f} > ${budget.limit:.2f} ({percentage_used:.1f}%)",
                    current_cost=current_cost,
                    budget_limit=budget.limit,
                    percentage_used=percentage_used,
                    metadata={
                        "period": budget.period.value,
                        "hard_limit": budget.hard_limit,
                        "budget_key": budget_key
                    }
                )
                alerts.append(alert)
                
            else:
                # Check alert thresholds
                percentage_used = (current_cost / budget.limit) * 100
                
                for threshold in sorted(budget.alert_thresholds, reverse=True):
                    threshold_pct = threshold * 100
                    if percentage_used >= threshold_pct:
                        # Find if we already alerted for this threshold recently
                        recent_alerts = [a for a in self.alerts_cache 
                                       if a.category == budget.category 
                                       and a.timestamp > period_start
                                       and abs(a.percentage_used - percentage_used) < 5]  # Within 5%
                        
                        if not recent_alerts:
                            if threshold >= 0.95:
                                level = AlertLevel.WARNING
                            elif threshold >= 0.8:
                                level = AlertLevel.INFO
                            else:
                                level = AlertLevel.INFO
                                
                            alert = CostAlert(
                                timestamp=datetime.now(),
                                level=level,
                                category=budget.category,
                                message=f"Budget threshold reached: ${current_cost:.4f} / ${budget.limit:.2f} ({percentage_used:.1f}%)",
                                current_cost=current_cost,
                                budget_limit=budget.limit,
                                percentage_used=percentage_used,
                                metadata={
                                    "threshold": threshold,
                                    "period": budget.period.value,
                                    "budget_key": budget_key
                                }
                            )
                            alerts.append(alert)
                        break  # Only alert for the highest threshold reached
                        
        return alerts
    
    async def _handle_alerts(self, alerts: List[CostAlert]):
        """Handle generated alerts"""
        for alert in alerts:
            # Cache alert
            self.alerts_cache.append(alert)
            
            # Log alert if enabled
            if self.config.enable_logging_alerts:
                level_map = {
                    AlertLevel.INFO: logging.INFO,
                    AlertLevel.WARNING: logging.WARNING,
                    AlertLevel.CRITICAL: logging.CRITICAL,
                    AlertLevel.EMERGENCY: logging.CRITICAL
                }
                logger.log(level_map[alert.level], f"Cost Alert: {alert.message}")
            
            # Save alert to file
            await self._append_alert(alert)
            
            # Call external alert handlers
            for handler in self.config.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
    
    def _should_block_operation(self, alerts: List[CostAlert]) -> bool:
        """Determine if operation should be blocked based on alerts"""
        for alert in alerts:
            if alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
                # Check if this is a hard limit violation
                if alert.metadata.get("hard_limit", False):
                    return True
        return False
    
    @asynccontextmanager
    async def track_operation(self, category: CostCategory, operation: str, estimated_cost: Optional[float] = None):
        """Context manager for tracking operation costs"""
        # start_time = datetime.now()  # Currently unused
        
        # Pre-flight budget check if estimated cost provided
        if estimated_cost and estimated_cost > 0:
            alerts = await self._check_budget_violations(estimated_cost, category)
            if self._should_block_operation(alerts):
                await self._handle_alerts(alerts)
                raise ValueError(f"Operation blocked due to budget limits: {operation}")
        
        try:
            yield self
        finally:
            # This will be overridden by record_cost if called explicitly
            pass
    
    async def record_cost(self, category: CostCategory, operation: str, cost: float, 
                         metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record a cost entry and check budget limits
        
        Returns:
            bool: True if operation is allowed, False if blocked by hard limits
        """
        if not self._initialized:
            await self._initialize()
        
        with self._lock:
            # Create cost entry
            entry = CostEntry(
                timestamp=datetime.now(),
                category=category,
                operation=operation,
                cost=cost,
                metadata=metadata or {}
            )
            
            # Add to cache
            self.cost_cache.append(entry)
        
        # Save to file
        await self._append_cost_entry(entry)
        
        # Check budget violations
        alerts = await self._check_budget_violations(cost, category)
        
        # Handle alerts
        if alerts:
            await self._handle_alerts(alerts)
        
        # Return whether operation should be blocked
        return not self._should_block_operation(alerts)
    
    async def set_budget_limit(self, category: Optional[CostCategory], period: BudgetPeriod, 
                              limit: float, **kwargs) -> str:
        """Set a budget limit"""
        budget = BudgetLimit(
            category=category,
            period=period,
            limit=limit,
            **kwargs
        )
        
        key = self._get_budget_key(category, period)
        self.budget_limits[key] = budget
        
        await self._save_budget_limits()
        return key
    
    async def remove_budget_limit(self, category: Optional[CostCategory], period: BudgetPeriod) -> bool:
        """Remove a budget limit"""
        key = self._get_budget_key(category, period)
        if key in self.budget_limits:
            del self.budget_limits[key]
            await self._save_budget_limits()
            return True
        return False
    
    async def get_cost_summary(self, period: BudgetPeriod, reference_time: Optional[datetime] = None) -> CostSummary:
        """Get cost summary for a period"""
        if not self._initialized:
            await self._initialize()
            
        period_start, period_end = self._get_period_bounds(period, reference_time)
        
        relevant_entries = [
            entry for entry in self.cost_cache
            if period_start <= entry.timestamp < period_end
        ]
        
        total_cost = sum(entry.cost for entry in relevant_entries)
        
        cost_by_category = {}
        for category in CostCategory:
            category_cost = sum(
                entry.cost for entry in relevant_entries
                if entry.category == category
            )
            if category_cost > 0:
                cost_by_category[category] = category_cost
        
        return CostSummary(
            period_start=period_start,
            period_end=period_end,
            total_cost=total_cost,
            cost_by_category=cost_by_category,
            entry_count=len(relevant_entries)
        )
    
    async def get_budget_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current budget status for all limits"""
        if not self._initialized:
            await self._initialize()
            
        status = {}
        
        for key, budget in self.budget_limits.items():
            if not budget.enabled:
                continue
                
            period_start, period_end = self._get_period_bounds(budget.period)
            current_cost = self._calculate_costs_for_period(budget.category, period_start, period_end)
            
            percentage_used = (current_cost / budget.limit) * 100 if budget.limit > 0 else 0
            remaining = max(0, budget.limit - current_cost)
            
            status[key] = {
                "category": budget.category.value if budget.category else "overall",
                "period": budget.period.value,
                "limit": budget.limit,
                "current_cost": current_cost,
                "remaining": remaining,
                "percentage_used": percentage_used,
                "hard_limit": budget.hard_limit,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat()
            }
        
        return status
    
    async def estimate_operation_cost(self, operation_type: str, **params) -> float:
        """Estimate cost for an operation based on type and parameters"""
        
        if operation_type == "openai_embedding":
            token_count = params.get("token_count", 0)
            return (token_count / 1000) * self.config.pricing["openai_embedding_per_1k"]
            
        elif operation_type == "weaviate_query":
            query_count = params.get("query_count", 1)
            return query_count * self.config.pricing["weaviate_query_cost"]
            
        elif operation_type == "weaviate_insert":
            insert_count = params.get("insert_count", 1)
            return insert_count * self.config.pricing["weaviate_insert_cost"]
            
        elif operation_type == "letta_api_call":
            call_count = params.get("call_count", 1)
            return call_count * self.config.pricing["letta_api_call_cost"]
        
        else:
            # Return a small default cost for unknown operations
            return 0.001
    
    async def reset_period_costs(self, category: Optional[CostCategory], period: BudgetPeriod):
        """Reset costs for a specific category and period (for testing/admin purposes)"""
        period_start, period_end = self._get_period_bounds(period)
        
        # Remove entries from cache
        with self._lock:
            self.cost_cache = [
                entry for entry in self.cost_cache
                if not (period_start <= entry.timestamp < period_end and 
                       (category is None or entry.category == category))
            ]
        
        logger.info(f"Reset costs for {category.value if category else 'all categories'} in period {period.value}")
    
    def add_alert_handler(self, handler: Callable[[CostAlert], None]):
        """Add custom alert handler"""
        self.config.alert_handlers.append(handler)
    
    async def get_recent_alerts(self, hours: int = 24) -> List[CostAlert]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts_cache if alert.timestamp > cutoff]


# Global instance
_cost_manager: Optional[CostControlManager] = None


def get_cost_manager() -> CostControlManager:
    """Get or create the global cost manager instance"""
    global _cost_manager
    if _cost_manager is None:
        _cost_manager = CostControlManager()
    return _cost_manager


# Convenience functions for common operations
async def record_embedding_cost(operation: str, token_count: int, provider: str = "openai"):
    """Record cost for embedding generation"""
    manager = get_cost_manager()
    cost = await manager.estimate_operation_cost("openai_embedding", token_count=token_count)
    
    metadata = {
        "provider": provider,
        "token_count": token_count,
        "cost_per_1k_tokens": manager.config.pricing["openai_embedding_per_1k"]
    }
    
    return await manager.record_cost(
        CostCategory.EMBEDDING_API, 
        operation, 
        cost, 
        metadata
    )


async def record_weaviate_cost(operation: str, operation_type: str, count: int = 1):
    """Record cost for Weaviate operations"""
    manager = get_cost_manager()
    
    if operation_type == "query":
        cost = await manager.estimate_operation_cost("weaviate_query", query_count=count)
    elif operation_type == "insert":
        cost = await manager.estimate_operation_cost("weaviate_insert", insert_count=count)
    else:
        cost = 0.001 * count  # Default small cost
    
    metadata = {
        "operation_type": operation_type,
        "count": count
    }
    
    return await manager.record_cost(
        CostCategory.VECTOR_DATABASE,
        operation,
        cost,
        metadata
    )


async def record_letta_api_cost(operation: str, call_count: int = 1):
    """Record cost for Letta API calls"""
    manager = get_cost_manager()
    cost = await manager.estimate_operation_cost("letta_api_call", call_count=call_count)
    
    metadata = {
        "call_count": call_count,
        "cost_per_call": manager.config.pricing["letta_api_call_cost"]
    }
    
    return await manager.record_cost(
        CostCategory.LETTA_API,
        operation,
        cost,
        metadata
    )