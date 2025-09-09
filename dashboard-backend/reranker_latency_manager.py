"""
LDTS-76: Enforce reranker latency budget and fallback mode

Latency budget enforcement and fallback mode for reranker operations
to ensure system performance and reliability.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class FallbackMode(Enum):
    """Fallback modes when reranker exceeds latency budget"""
    ORIGINAL_ORDER = "original_order"      # Return documents in original order
    SCORE_BASED = "score_based"           # Sort by original scores if available
    RANDOM_SHUFFLE = "random_shuffle"      # Random shuffle to avoid bias
    TOP_K_ONLY = "top_k_only"            # Return only top-k without reranking
    FAIL_FAST = "fail_fast"              # Raise timeout exception immediately

class LatencyBudgetStatus(Enum):
    """Status of latency budget enforcement"""
    WITHIN_BUDGET = "within_budget"
    APPROACHING_LIMIT = "approaching_limit"
    BUDGET_EXCEEDED = "budget_exceeded"
    FALLBACK_TRIGGERED = "fallback_triggered"

@dataclass
class LatencyBudget:
    """Configuration for latency budget"""
    max_latency_ms: int = 5000           # Maximum allowed latency in milliseconds
    warning_threshold: float = 0.8        # Warning at 80% of budget
    fallback_mode: FallbackMode = FallbackMode.ORIGINAL_ORDER
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5    # Number of consecutive timeouts to trigger breaker
    circuit_breaker_reset_time: int = 30  # Time in seconds to reset circuit breaker
    adaptive_budget: bool = True          # Automatically adjust budget based on performance
    min_budget_ms: int = 1000            # Minimum budget when adapting
    max_budget_ms: int = 10000           # Maximum budget when adapting

@dataclass
class LatencyMetrics:
    """Latency performance metrics"""
    total_requests: int = 0
    timeouts: int = 0
    fallbacks_triggered: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    timeout_rate: float = 0.0
    fallback_rate: float = 0.0
    recent_latencies: List[float] = field(default_factory=lambda: [])
    circuit_breaker_trips: int = 0
    budget_adjustments: int = 0

class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreaker:
    """Circuit breaker for reranker latency protection"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    success_count: int = 0
    threshold: int = 5
    reset_time: int = 30
    test_requests_in_half_open: int = 3

class RerankerLatencyManager:
    """Manager for enforcing reranker latency budgets and fallback behavior"""
    
    def __init__(self, latency_budget: LatencyBudget):
        self.budget = latency_budget
        self.metrics = LatencyMetrics()
        self.circuit_breaker = CircuitBreaker(
            threshold=latency_budget.circuit_breaker_threshold,
            reset_time=latency_budget.circuit_breaker_reset_time
        )
        
        # Performance tracking
        self.latency_history: List[float] = []
        self.max_history_size = 1000
        
        # Adaptive budget tracking
        self.last_budget_adjustment = time.time()
        self.adjustment_interval = 300  # 5 minutes
        
        logger.info(f"RerankerLatencyManager initialized: budget={latency_budget.max_latency_ms}ms")
    
    @asynccontextmanager
    async def enforce_budget(self, operation_name: str = "rerank"):
        """Context manager to enforce latency budget on operations"""
        start_time = time.time()
        budget_status = LatencyBudgetStatus.WITHIN_BUDGET
        fallback_triggered = False
        
        # Check circuit breaker
        if self.budget.enable_circuit_breaker:
            if not await self._check_circuit_breaker():
                fallback_triggered = True
                logger.warning(f"Circuit breaker is OPEN for {operation_name}")
                yield {
                    "status": LatencyBudgetStatus.FALLBACK_TRIGGERED,
                    "fallback_triggered": True,
                    "circuit_breaker_open": True,
                    "latency_ms": 0
                }
                return
        
        try:
            # Create timeout task
            timeout_seconds = self.budget.max_latency_ms / 1000.0
            
            # Yield control to the operation
            async with asyncio.timeout(timeout_seconds):
                yield {
                    "status": budget_status,
                    "fallback_triggered": False,
                    "circuit_breaker_open": False,
                    "budget_ms": self.budget.max_latency_ms
                }
            
            # Operation completed within budget
            elapsed_ms = (time.time() - start_time) * 1000
            await self._record_success(elapsed_ms)
            
        except asyncio.TimeoutError:
            # Budget exceeded
            elapsed_ms = (time.time() - start_time) * 1000
            fallback_triggered = True
            budget_status = LatencyBudgetStatus.BUDGET_EXCEEDED
            
            logger.warning(f"Latency budget exceeded for {operation_name}: {elapsed_ms:.1f}ms > {self.budget.max_latency_ms}ms")
            
            await self._record_timeout(elapsed_ms)
            
            # Don't re-raise the timeout - let the caller handle fallback
            
        except Exception as e:
            # Other errors
            elapsed_ms = (time.time() - start_time) * 1000
            await self._record_failure(elapsed_ms)
            raise
        
        finally:
            # Update metrics regardless of outcome
            self._update_metrics(time.time() - start_time, fallback_triggered)
    
    async def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker allows the request"""
        current_time = time.time()
        
        if self.circuit_breaker.state == CircuitBreakerState.CLOSED:
            return True
        
        elif self.circuit_breaker.state == CircuitBreakerState.OPEN:
            # Check if enough time has passed to try half-open
            if (self.circuit_breaker.last_failure_time and 
                current_time - self.circuit_breaker.last_failure_time >= self.circuit_breaker.reset_time):
                self.circuit_breaker.state = CircuitBreakerState.HALF_OPEN
                self.circuit_breaker.success_count = 0
                logger.info("Circuit breaker transitioning to HALF_OPEN")
                return True
            return False
        
        elif self.circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            # Allow limited test requests
            return self.circuit_breaker.success_count < self.circuit_breaker.test_requests_in_half_open
        
        return False
    
    async def _record_success(self, latency_ms: float):
        """Record successful operation"""
        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.max_history_size:
            self.latency_history.pop(0)
        
        # Update circuit breaker
        if self.circuit_breaker.state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker.success_count += 1
            if self.circuit_breaker.success_count >= self.circuit_breaker.test_requests_in_half_open:
                self.circuit_breaker.state = CircuitBreakerState.CLOSED
                self.circuit_breaker.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")
        
        # Check if adaptive budget adjustment is needed
        if self.budget.adaptive_budget:
            await self._maybe_adjust_budget()
    
    async def _record_timeout(self, latency_ms: float):
        """Record timeout/budget exceeded"""
        self.metrics.timeouts += 1
        self.metrics.fallbacks_triggered += 1
        
        # Update circuit breaker
        if self.budget.enable_circuit_breaker:
            self.circuit_breaker.failure_count += 1
            self.circuit_breaker.last_failure_time = time.time()
            
            if self.circuit_breaker.failure_count >= self.circuit_breaker.threshold:
                self.circuit_breaker.state = CircuitBreakerState.OPEN
                self.metrics.circuit_breaker_trips += 1
                logger.warning(f"Circuit breaker OPENED after {self.circuit_breaker.failure_count} consecutive timeouts")
    
    async def _record_failure(self, latency_ms: float):
        """Record general failure"""
        # Similar to timeout handling for circuit breaker
        if self.budget.enable_circuit_breaker:
            self.circuit_breaker.failure_count += 1
            self.circuit_breaker.last_failure_time = time.time()
    
    def _update_metrics(self, elapsed_seconds: float, fallback_triggered: bool):
        """Update performance metrics"""
        self.metrics.total_requests += 1
        latency_ms = elapsed_seconds * 1000
        
        # Update averages
        if self.metrics.total_requests == 1:
            self.metrics.average_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_latency_ms = (
                alpha * latency_ms + (1 - alpha) * self.metrics.average_latency_ms
            )
        
        # Update rates
        self.metrics.timeout_rate = self.metrics.timeouts / self.metrics.total_requests
        self.metrics.fallback_rate = self.metrics.fallbacks_triggered / self.metrics.total_requests
        
        # Update percentiles from recent history
        if self.latency_history:
            sorted_latencies = sorted(self.latency_history)
            n = len(sorted_latencies)
            self.metrics.p95_latency_ms = sorted_latencies[int(n * 0.95)]
            self.metrics.p99_latency_ms = sorted_latencies[int(n * 0.99)]
    
    async def _maybe_adjust_budget(self):
        """Adaptively adjust latency budget based on performance"""
        current_time = time.time()
        if current_time - self.last_budget_adjustment < self.adjustment_interval:
            return
        
        if not self.latency_history:
            return
        
        # Calculate recent performance
        recent_p95 = sorted(self.latency_history[-100:] if len(self.latency_history) >= 100 else self.latency_history)[int(len(self.latency_history) * 0.95)]
        
        old_budget = self.budget.max_latency_ms
        
        # Adjust budget based on recent performance
        if self.metrics.timeout_rate > 0.1:  # Too many timeouts
            # Increase budget
            new_budget = min(
                self.budget.max_budget_ms,
                int(self.budget.max_latency_ms * 1.2)
            )
        elif self.metrics.timeout_rate < 0.01 and recent_p95 < self.budget.max_latency_ms * 0.5:
            # Performance is good, can reduce budget
            new_budget = max(
                self.budget.min_budget_ms,
                int(self.budget.max_latency_ms * 0.9)
            )
        else:
            new_budget = self.budget.max_latency_ms
        
        if new_budget != old_budget:
            self.budget.max_latency_ms = new_budget
            self.metrics.budget_adjustments += 1
            self.last_budget_adjustment = current_time
            
            logger.info(f"Adaptive budget adjustment: {old_budget}ms -> {new_budget}ms (timeout_rate: {self.metrics.timeout_rate:.3f})")
    
    async def apply_fallback(self, 
                           original_documents: List[str], 
                           original_scores: Optional[List[float]] = None,
                           query: str = "") -> Dict[str, Any]:
        """Apply fallback strategy when reranker times out"""
        
        fallback_start = time.time()
        
        try:
            if self.budget.fallback_mode == FallbackMode.ORIGINAL_ORDER:
                # Return documents in original order
                result = {
                    "reranked_indices": list(range(len(original_documents))),
                    "reranked_scores": original_scores or [0.0] * len(original_documents),
                    "reranked_documents": original_documents,
                    "fallback_mode": "original_order",
                    "fallback_reason": "latency_budget_exceeded"
                }
            
            elif self.budget.fallback_mode == FallbackMode.SCORE_BASED and original_scores:
                # Sort by original scores
                indexed_scores = list(enumerate(original_scores))
                indexed_scores.sort(key=lambda x: x[1], reverse=True)
                
                reranked_indices = [idx for idx, _ in indexed_scores]
                sorted_scores = [score for _, score in indexed_scores]
                reranked_documents = [original_documents[i] for i in reranked_indices]
                
                result = {
                    "reranked_indices": reranked_indices,
                    "reranked_scores": sorted_scores,
                    "reranked_documents": reranked_documents,
                    "fallback_mode": "score_based",
                    "fallback_reason": "latency_budget_exceeded"
                }
            
            elif self.budget.fallback_mode == FallbackMode.TOP_K_ONLY:
                # Return only top 10 documents in original order
                top_k = min(10, len(original_documents))
                result = {
                    "reranked_indices": list(range(top_k)),
                    "reranked_scores": (original_scores or [0.0] * len(original_documents))[:top_k],
                    "reranked_documents": original_documents[:top_k],
                    "fallback_mode": "top_k_only",
                    "fallback_reason": "latency_budget_exceeded",
                    "top_k": top_k
                }
            
            elif self.budget.fallback_mode == FallbackMode.RANDOM_SHUFFLE:
                # Random shuffle to avoid bias
                import random
                indices = list(range(len(original_documents)))
                random.shuffle(indices)
                
                result = {
                    "reranked_indices": indices,
                    "reranked_scores": [original_scores[i] if original_scores else 0.0 for i in indices],
                    "reranked_documents": [original_documents[i] for i in indices],
                    "fallback_mode": "random_shuffle", 
                    "fallback_reason": "latency_budget_exceeded"
                }
            
            elif self.budget.fallback_mode == FallbackMode.FAIL_FAST:
                # Raise exception immediately
                raise TimeoutError("Reranker latency budget exceeded - fail fast mode")
            
            else:
                # Default to original order
                result = {
                    "reranked_indices": list(range(len(original_documents))),
                    "reranked_scores": original_scores or [0.0] * len(original_documents),
                    "reranked_documents": original_documents,
                    "fallback_mode": "original_order",
                    "fallback_reason": "latency_budget_exceeded"
                }
            
            # Add fallback metadata
            fallback_time_ms = (time.time() - fallback_start) * 1000
            result.update({
                "is_fallback": True,
                "fallback_processing_time": fallback_time_ms,
                "original_document_count": len(original_documents),
                "query": query,
                "latency_budget_ms": self.budget.max_latency_ms,
                "circuit_breaker_state": self.circuit_breaker.state.value
            })
            
            logger.info(f"Applied fallback mode '{self.budget.fallback_mode.value}' for {len(original_documents)} documents")
            
            return result
            
        except Exception as e:
            logger.error(f"Fallback strategy failed: {e}")
            # Ultimate fallback - return original order
            return {
                "reranked_indices": list(range(len(original_documents))),
                "reranked_scores": original_scores or [0.0] * len(original_documents),
                "reranked_documents": original_documents,
                "fallback_mode": "emergency_fallback",
                "fallback_reason": f"fallback_failed: {str(e)}",
                "is_fallback": True,
                "original_document_count": len(original_documents)
            }
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status and metrics"""
        return {
            "latency_budget": {
                "max_latency_ms": self.budget.max_latency_ms,
                "warning_threshold": self.budget.warning_threshold,
                "fallback_mode": self.budget.fallback_mode.value,
                "adaptive_budget": self.budget.adaptive_budget,
                "min_budget_ms": self.budget.min_budget_ms,
                "max_budget_ms": self.budget.max_budget_ms
            },
            "circuit_breaker": {
                "enabled": self.budget.enable_circuit_breaker,
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "threshold": self.circuit_breaker.threshold,
                "trips": self.metrics.circuit_breaker_trips
            },
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "timeouts": self.metrics.timeouts,
                "fallbacks_triggered": self.metrics.fallbacks_triggered,
                "timeout_rate": self.metrics.timeout_rate,
                "fallback_rate": self.metrics.fallback_rate,
                "average_latency_ms": self.metrics.average_latency_ms,
                "p95_latency_ms": self.metrics.p95_latency_ms,
                "p99_latency_ms": self.metrics.p99_latency_ms,
                "budget_adjustments": self.metrics.budget_adjustments
            },
            "performance": {
                "recent_latencies_count": len(self.latency_history),
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips
            }
        }
    
    def update_budget_config(self, new_config: Dict[str, Any]):
        """Update latency budget configuration"""
        if "max_latency_ms" in new_config:
            old_budget = self.budget.max_latency_ms
            self.budget.max_latency_ms = new_config["max_latency_ms"]
            logger.info(f"Updated latency budget: {old_budget}ms -> {self.budget.max_latency_ms}ms")
        
        if "fallback_mode" in new_config:
            try:
                self.budget.fallback_mode = FallbackMode(new_config["fallback_mode"])
                logger.info(f"Updated fallback mode: {self.budget.fallback_mode.value}")
            except ValueError:
                logger.warning(f"Invalid fallback mode: {new_config['fallback_mode']}")
        
        if "enable_circuit_breaker" in new_config:
            self.budget.enable_circuit_breaker = new_config["enable_circuit_breaker"]
            logger.info(f"Circuit breaker enabled: {self.budget.enable_circuit_breaker}")
        
        if "adaptive_budget" in new_config:
            self.budget.adaptive_budget = new_config["adaptive_budget"]
            logger.info(f"Adaptive budget enabled: {self.budget.adaptive_budget}")
    
    def reset_circuit_breaker(self):
        """Manually reset circuit breaker"""
        self.circuit_breaker.state = CircuitBreakerState.CLOSED
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.last_failure_time = None
        logger.info("Circuit breaker manually reset to CLOSED")

# Global latency manager instance
latency_manager: Optional[RerankerLatencyManager] = None

def initialize_latency_manager(config: Dict[str, Any]) -> RerankerLatencyManager:
    """Initialize global latency manager"""
    global latency_manager
    
    budget_config = LatencyBudget(
        max_latency_ms=config.get("max_latency_ms", 5000),
        warning_threshold=config.get("warning_threshold", 0.8),
        fallback_mode=FallbackMode(config.get("fallback_mode", "original_order")),
        enable_circuit_breaker=config.get("enable_circuit_breaker", True),
        circuit_breaker_threshold=config.get("circuit_breaker_threshold", 5),
        circuit_breaker_reset_time=config.get("circuit_breaker_reset_time", 30),
        adaptive_budget=config.get("adaptive_budget", True),
        min_budget_ms=config.get("min_budget_ms", 1000),
        max_budget_ms=config.get("max_budget_ms", 10000)
    )
    
    latency_manager = RerankerLatencyManager(budget_config)
    return latency_manager

def get_latency_manager() -> RerankerLatencyManager:
    """Get global latency manager instance"""
    if latency_manager is None:
        raise RuntimeError("Latency manager not initialized")
    return latency_manager