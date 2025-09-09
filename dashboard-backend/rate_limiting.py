"""
LDTS-32: Rate limiting and resource management

Implements request rate limiting, resource usage monitoring, and quota enforcement
for the LDTS Reranker Testing Dashboard to prevent abuse and ensure fair usage.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from fastapi import HTTPException, Request
import psutil
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    concurrent_requests: int = 10
    search_requests_per_minute: int = 30
    rerank_requests_per_minute: int = 15
    max_query_length: int = 500
    max_results_per_request: int = 100

@dataclass
class ResourceLimits:
    """Resource usage limits"""
    max_memory_usage_mb: int = 2048
    max_cpu_usage_percent: float = 80.0
    max_disk_usage_mb: int = 1024
    max_request_duration_seconds: int = 30
    memory_check_interval: int = 60  # seconds

@dataclass
class ClientQuota:
    """Per-client usage tracking"""
    client_id: str
    requests_minute: deque = field(default_factory=deque)
    requests_hour: deque = field(default_factory=deque)  
    requests_day: deque = field(default_factory=deque)
    concurrent_requests: int = 0
    search_requests_minute: deque = field(default_factory=deque)
    rerank_requests_minute: deque = field(default_factory=deque)
    total_requests: int = 0
    last_request_time: float = 0
    blocked_until: Optional[float] = None

class RateLimiter:
    """Rate limiting and resource management system"""
    
    def __init__(self, config: RateLimitConfig = None, resource_limits: ResourceLimits = None):
        self.config = config or RateLimitConfig()
        self.resource_limits = resource_limits or ResourceLimits()
        self.client_quotas: Dict[str, ClientQuota] = defaultdict(lambda: ClientQuota(client_id=""))
        self.global_concurrent_requests = 0
        self._resource_monitor_task = None
        self._current_memory_usage = 0
        self._current_cpu_usage = 0
        self._resource_alerts = []
        logger.info("Rate limiter initialized with request limits and resource monitoring")
    
    async def initialize(self):
        """Initialize rate limiter and start resource monitoring"""
        try:
            logger.info("Initializing rate limiter and resource monitor...")
            
            # Start resource monitoring task
            self._resource_monitor_task = asyncio.create_task(self._monitor_resources())
            
            logger.info("Rate limiter and resource monitor started successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize rate limiter: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown rate limiter and cleanup"""
        logger.info("Shutting down rate limiter...")
        
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
            try:
                await self._resource_monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Rate limiter shutdown complete")
    
    def get_client_id(self, request: Request) -> str:
        """Extract client identifier from request"""
        # Try to get client ID from various sources
        client_id = (
            request.headers.get("X-Client-ID") or 
            request.headers.get("User-Agent", "unknown")[:50] or
            request.client.host if request.client else "unknown"
        )
        return client_id
    
    async def check_rate_limit(self, request: Request, operation_type: str = "general") -> bool:
        """Check if request should be rate limited"""
        try:
            client_id = self.get_client_id(request)
            quota = self.client_quotas[client_id]
            quota.client_id = client_id
            
            current_time = time.time()
            
            # Check if client is temporarily blocked
            if quota.blocked_until and current_time < quota.blocked_until:
                raise HTTPException(
                    status_code=429,
                    detail=f"Client temporarily blocked until {datetime.fromtimestamp(quota.blocked_until).isoformat()}"
                )
            
            # Clear expired blocks
            if quota.blocked_until and current_time >= quota.blocked_until:
                quota.blocked_until = None
            
            # Check global concurrent requests
            if self.global_concurrent_requests >= self.config.concurrent_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Server at maximum concurrent request capacity"
                )
            
            # Check per-client concurrent requests
            if quota.concurrent_requests >= self.config.concurrent_requests // 2:  # Per client limit
                raise HTTPException(
                    status_code=429, 
                    detail="Too many concurrent requests from this client"
                )
            
            # Clean old requests from sliding windows
            self._clean_old_requests(quota, current_time)
            
            # Check rate limits based on operation type
            if operation_type == "search":
                if len(quota.search_requests_minute) >= self.config.search_requests_per_minute:
                    raise HTTPException(
                        status_code=429,
                        detail=f"Search rate limit exceeded: {self.config.search_requests_per_minute} requests per minute"
                    )
                quota.search_requests_minute.append(current_time)
            
            elif operation_type == "rerank":
                if len(quota.rerank_requests_minute) >= self.config.rerank_requests_per_minute:
                    raise HTTPException(
                        status_code=429, 
                        detail=f"Rerank rate limit exceeded: {self.config.rerank_requests_per_minute} requests per minute"
                    )
                quota.rerank_requests_minute.append(current_time)
            
            # Check general rate limits
            if len(quota.requests_minute) >= self.config.requests_per_minute:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {self.config.requests_per_minute} requests per minute"
                )
            
            if len(quota.requests_hour) >= self.config.requests_per_hour:
                raise HTTPException(
                    status_code=429,
                    detail=f"Hourly limit exceeded: {self.config.requests_per_hour} requests per hour"
                )
            
            if len(quota.requests_day) >= self.config.requests_per_day:
                # Block client for remainder of day
                tomorrow = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                quota.blocked_until = tomorrow.timestamp()
                raise HTTPException(
                    status_code=429,
                    detail=f"Daily limit exceeded: {self.config.requests_per_day} requests per day. Blocked until {tomorrow.isoformat()}"
                )
            
            # Check resource usage
            await self._check_resource_limits()
            
            # Record request
            quota.requests_minute.append(current_time)
            quota.requests_hour.append(current_time)
            quota.requests_day.append(current_time)
            quota.total_requests += 1
            quota.last_request_time = current_time
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Allow request in case of rate limiter failure
            return True
    
    def _clean_old_requests(self, quota: ClientQuota, current_time: float):
        """Remove expired requests from sliding windows"""
        minute_ago = current_time - 60
        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        
        # Clean minute window
        while quota.requests_minute and quota.requests_minute[0] < minute_ago:
            quota.requests_minute.popleft()
        while quota.search_requests_minute and quota.search_requests_minute[0] < minute_ago:
            quota.search_requests_minute.popleft()
        while quota.rerank_requests_minute and quota.rerank_requests_minute[0] < minute_ago:
            quota.rerank_requests_minute.popleft()
        
        # Clean hour window
        while quota.requests_hour and quota.requests_hour[0] < hour_ago:
            quota.requests_hour.popleft()
        
        # Clean day window
        while quota.requests_day and quota.requests_day[0] < day_ago:
            quota.requests_day.popleft()
    
    async def _check_resource_limits(self):
        """Check system resource usage"""
        # Check memory usage
        if self._current_memory_usage > self.resource_limits.max_memory_usage_mb:
            raise HTTPException(
                status_code=503,
                detail=f"Server memory usage too high: {self._current_memory_usage}MB > {self.resource_limits.max_memory_usage_mb}MB"
            )
        
        # Check CPU usage
        if self._current_cpu_usage > self.resource_limits.max_cpu_usage_percent:
            raise HTTPException(
                status_code=503,
                detail=f"Server CPU usage too high: {self._current_cpu_usage}% > {self.resource_limits.max_cpu_usage_percent}%"
            )
    
    @asynccontextmanager
    async def request_context(self, request: Request, operation_type: str = "general"):
        """Context manager for tracking request lifecycle"""
        client_id = self.get_client_id(request)
        quota = self.client_quotas[client_id]
        
        # Increment concurrent request counters
        quota.concurrent_requests += 1
        self.global_concurrent_requests += 1
        
        start_time = time.time()
        
        try:
            yield
        finally:
            # Decrement concurrent request counters
            quota.concurrent_requests -= 1
            self.global_concurrent_requests -= 1
            
            # Check request duration
            duration = time.time() - start_time
            if duration > self.resource_limits.max_request_duration_seconds:
                logger.warning(f"Long request duration: {duration:.2f}s for client {client_id}")
    
    async def _monitor_resources(self):
        """Background task to monitor system resources"""
        while True:
            try:
                # Get current resource usage
                memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
                cpu_usage = psutil.cpu_percent(interval=1)
                
                self._current_memory_usage = memory_usage
                self._current_cpu_usage = cpu_usage
                
                # Check for resource alerts
                if memory_usage > self.resource_limits.max_memory_usage_mb * 0.9:
                    alert = f"High memory usage: {memory_usage:.1f}MB"
                    if alert not in self._resource_alerts:
                        logger.warning(alert)
                        self._resource_alerts.append(alert)
                
                if cpu_usage > self.resource_limits.max_cpu_usage_percent * 0.9:
                    alert = f"High CPU usage: {cpu_usage:.1f}%"
                    if alert not in self._resource_alerts:
                        logger.warning(alert)
                        self._resource_alerts.append(alert)
                
                # Clean old alerts (keep only recent ones)
                if len(self._resource_alerts) > 10:
                    self._resource_alerts = self._resource_alerts[-5:]
                
                await asyncio.sleep(self.resource_limits.memory_check_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(30)  # Shorter retry interval on error
    
    def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get statistics for a specific client"""
        quota = self.client_quotas.get(client_id)
        if not quota:
            return {"error": "Client not found"}
        
        current_time = time.time()
        self._clean_old_requests(quota, current_time)
        
        return {
            "client_id": client_id,
            "total_requests": quota.total_requests,
            "concurrent_requests": quota.concurrent_requests,
            "requests_last_minute": len(quota.requests_minute),
            "requests_last_hour": len(quota.requests_hour),
            "requests_last_day": len(quota.requests_day),
            "search_requests_last_minute": len(quota.search_requests_minute),
            "rerank_requests_last_minute": len(quota.rerank_requests_minute),
            "last_request_time": quota.last_request_time,
            "blocked_until": quota.blocked_until,
            "rate_limits": {
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour, 
                "requests_per_day": self.config.requests_per_day,
                "search_requests_per_minute": self.config.search_requests_per_minute,
                "rerank_requests_per_minute": self.config.rerank_requests_per_minute
            }
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get overall system statistics"""
        return {
            "global_concurrent_requests": self.global_concurrent_requests,
            "total_clients": len(self.client_quotas),
            "active_clients": sum(1 for q in self.client_quotas.values() if q.concurrent_requests > 0),
            "resource_usage": {
                "memory_mb": self._current_memory_usage,
                "cpu_percent": self._current_cpu_usage,
                "memory_limit_mb": self.resource_limits.max_memory_usage_mb,
                "cpu_limit_percent": self.resource_limits.max_cpu_usage_percent
            },
            "recent_alerts": self._resource_alerts[-5:],  # Last 5 alerts
            "rate_limits": {
                "concurrent_requests": self.config.concurrent_requests,
                "requests_per_minute": self.config.requests_per_minute,
                "requests_per_hour": self.config.requests_per_hour,
                "requests_per_day": self.config.requests_per_day
            }
        }

# Global rate limiter instance
rate_limiter = RateLimiter()

# Convenience functions for middleware usage
async def check_rate_limit(request: Request, operation_type: str = "general") -> bool:
    """Check rate limit for a request"""
    return await rate_limiter.check_rate_limit(request, operation_type)

def get_request_context(request: Request, operation_type: str = "general"):
    """Get request context manager"""
    return rate_limiter.request_context(request, operation_type)

async def validate_request_size(request: Request):
    """Validate request parameters against limits"""
    # This would be called from request validation middleware
    # For now, we'll implement basic checks
    
    if hasattr(request, 'json'):
        try:
            body = await request.json()
            
            # Check query length
            if 'query' in body and len(str(body['query'])) > rate_limiter.config.max_query_length:
                raise HTTPException(
                    status_code=400,
                    detail=f"Query too long: {len(str(body['query']))} > {rate_limiter.config.max_query_length}"
                )
            
            # Check result limit
            if 'limit' in body and int(body.get('limit', 0)) > rate_limiter.config.max_results_per_request:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Too many results requested: {body['limit']} > {rate_limiter.config.max_results_per_request}"
                )
                
        except Exception as e:
            if isinstance(e, HTTPException):
                raise
            # Ignore JSON parsing errors - not all requests have JSON bodies
            pass