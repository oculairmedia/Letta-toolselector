"""
LDTS-63: Weaviate Client Connection Management

Comprehensive connection management system for Weaviate with:
- Connection pooling and load balancing
- Health monitoring and automatic reconnection
- Circuit breaker pattern for resilience
- Connection metrics and monitoring
- Graceful degradation and fallback handling
"""

import weaviate
import asyncio
import threading
import time
import logging
import os
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states for health monitoring"""
    UNKNOWN = "unknown"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    READY = "ready"
    DEGRADED = "degraded"
    FAILED = "failed"
    CLOSED = "closed"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ConnectionConfig:
    """Configuration for Weaviate connections"""
    # Primary connection settings
    http_host: str = "weaviate"
    http_port: int = 8080
    http_secure: bool = False
    grpc_host: str = "weaviate"
    grpc_port: int = 50051
    grpc_secure: bool = False
    
    # Authentication
    openai_api_key: Optional[str] = None
    auth_config: Optional[Dict[str, Any]] = None
    
    # Connection pool settings
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: float = 30.0
    request_timeout: float = 60.0
    
    # Health monitoring
    health_check_interval: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_factor: float = 2.0
    
    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3
    
    # Additional headers
    headers: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def from_env(cls) -> 'ConnectionConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # First, try to parse WEAVIATE_URL for Docker Compose compatibility
        weaviate_url = os.getenv("WEAVIATE_URL")
        if weaviate_url:
            import urllib.parse
            parsed = urllib.parse.urlparse(weaviate_url)
            if parsed.hostname:
                config.http_host = parsed.hostname
            if parsed.port:
                config.http_port = parsed.port
            config.http_secure = parsed.scheme == "https"
        
        # Allow individual settings to override WEAVIATE_URL
        config.http_host = os.getenv("WEAVIATE_HTTP_HOST", config.http_host)
        config.http_port = int(os.getenv("WEAVIATE_HTTP_PORT", config.http_port))
        config.http_secure = os.getenv("WEAVIATE_HTTP_SECURE", str(config.http_secure)).lower() == "true"
        
        config.grpc_host = os.getenv("WEAVIATE_GRPC_HOST", config.http_host)  # Default to same as HTTP
        config.grpc_port = int(os.getenv("WEAVIATE_GRPC_PORT", config.grpc_port))
        config.grpc_secure = os.getenv("WEAVIATE_GRPC_SECURE", "false").lower() == "true"
        
        config.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Connection pool settings
        config.max_connections = int(os.getenv("WEAVIATE_MAX_CONNECTIONS", config.max_connections))
        config.min_connections = int(os.getenv("WEAVIATE_MIN_CONNECTIONS", config.min_connections))
        config.connection_timeout = float(os.getenv("WEAVIATE_CONNECTION_TIMEOUT", config.connection_timeout))
        config.request_timeout = float(os.getenv("WEAVIATE_REQUEST_TIMEOUT", config.request_timeout))
        
        # Health monitoring
        config.health_check_interval = float(os.getenv("WEAVIATE_HEALTH_CHECK_INTERVAL", config.health_check_interval))
        config.max_retries = int(os.getenv("WEAVIATE_MAX_RETRIES", config.max_retries))
        
        # Set headers
        if config.openai_api_key:
            config.headers["X-OpenAI-Api-Key"] = config.openai_api_key
        
        return config


@dataclass
class ConnectionMetrics:
    """Metrics for connection monitoring"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_health_check: Optional[datetime] = None
    uptime_start: datetime = field(default_factory=datetime.utcnow)
    circuit_breaker_trips: int = 0
    
    def uptime(self) -> timedelta:
        """Calculate uptime"""
        return datetime.utcnow() - self.uptime_start
    
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.successful_requests + self.failed_requests
        return (self.successful_requests / total) if total > 0 else 0.0


class CircuitBreaker:
    """Circuit breaker for connection resilience"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def can_execute(self) -> bool:
        """Check if operation can execute based on circuit breaker state"""
        with self._lock:
            if self.state == CircuitBreakerState.CLOSED:
                return True
            elif self.state == CircuitBreakerState.OPEN:
                # Check if we should move to half-open
                if self.last_failure_time and \
                   time.time() - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    return True
                return False
            elif self.state == CircuitBreakerState.HALF_OPEN:
                return True
            return False
    
    def record_success(self):
        """Record a successful operation"""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self):
        """Record a failed operation"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
            elif self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN


class WeaviateConnection:
    """Individual Weaviate connection wrapper"""
    
    def __init__(self, config: ConnectionConfig, connection_id: str):
        self.config = config
        self.connection_id = connection_id
        self.client: Optional[weaviate.Client] = None
        self.state = ConnectionState.UNKNOWN
        self.last_used = datetime.utcnow()
        self.created_at = datetime.utcnow()
        self.error_count = 0
        self.last_error: Optional[Exception] = None
        self._lock = threading.Lock()
    
    def connect(self) -> bool:
        """Establish connection to Weaviate"""
        with self._lock:
            try:
                self.state = ConnectionState.CONNECTING
                logger.info(f"Connecting to Weaviate: {self.connection_id}")
                
                self.client = weaviate.connect_to_custom(
                    http_host=self.config.http_host,
                    http_port=self.config.http_port,
                    http_secure=self.config.http_secure,
                    grpc_host=self.config.grpc_host,
                    grpc_port=self.config.grpc_port,
                    grpc_secure=self.config.grpc_secure,
                    headers=self.config.headers
                )
                
                # Test connection
                if self.client.is_ready():
                    self.state = ConnectionState.READY
                    self.error_count = 0
                    self.last_error = None
                    logger.info(f"Connection established: {self.connection_id}")
                    return True
                else:
                    self.state = ConnectionState.FAILED
                    logger.warning(f"Connection not ready: {self.connection_id}")
                    return False
                    
            except Exception as e:
                self.state = ConnectionState.FAILED
                self.error_count += 1
                self.last_error = e
                logger.error(f"Connection failed: {self.connection_id} - {e}")
                return False
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        try:
            if not self.client:
                return False
            
            # Quick health check
            return self.client.is_ready() and self.client.is_connected()
        except Exception as e:
            self.last_error = e
            self.error_count += 1
            return False
    
    def close(self):
        """Close the connection"""
        with self._lock:
            try:
                if self.client:
                    self.client.close()
                    logger.info(f"Connection closed: {self.connection_id}")
            except Exception as e:
                logger.warning(f"Error closing connection {self.connection_id}: {e}")
            finally:
                self.client = None
                self.state = ConnectionState.CLOSED
    
    def age(self) -> timedelta:
        """Get connection age"""
        return datetime.utcnow() - self.created_at
    
    def idle_time(self) -> timedelta:
        """Get idle time"""
        return datetime.utcnow() - self.last_used


class WeaviateConnectionPool:
    """Connection pool for Weaviate clients"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connections: List[WeaviateConnection] = []
        self.active_connections: Dict[str, WeaviateConnection] = {}
        self.circuit_breaker = CircuitBreaker(
            config.failure_threshold,
            config.recovery_timeout,
            config.success_threshold
        )
        self.metrics = ConnectionMetrics()
        self._lock = threading.Lock()
        self._health_check_thread: Optional[threading.Thread] = None
        self._shutdown = False
        
        # Initialize minimum connections
        self._initialize_pool()
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def _initialize_pool(self):
        """Initialize connection pool with minimum connections"""
        logger.info(f"Initializing connection pool with {self.config.min_connections} connections")
        
        for i in range(self.config.min_connections):
            connection = self._create_connection(f"pool-{i}")
            if connection.connect():
                self.connections.append(connection)
                self.metrics.total_connections += 1
            else:
                self.metrics.failed_connections += 1
        
        logger.info(f"Connection pool initialized: {len(self.connections)} connections")
    
    def _create_connection(self, connection_id: str) -> WeaviateConnection:
        """Create a new connection"""
        return WeaviateConnection(self.config, connection_id)
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        self._health_check_thread = threading.Thread(
            target=self._health_monitor,
            daemon=True,
            name="WeaviateHealthMonitor"
        )
        self._health_check_thread.start()
    
    def _health_monitor(self):
        """Health monitoring loop"""
        while not self._shutdown:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                time.sleep(5.0)
    
    def _perform_health_check(self):
        """Perform health checks on all connections"""
        with self._lock:
            healthy_connections = []
            
            for connection in self.connections:
                if connection.is_healthy():
                    healthy_connections.append(connection)
                else:
                    logger.warning(f"Unhealthy connection detected: {connection.connection_id}")
                    connection.close()
                    self.metrics.failed_connections += 1
            
            # Update connection list
            self.connections = healthy_connections
            self.metrics.active_connections = len(self.connections)
            self.metrics.last_health_check = datetime.utcnow()
            
            # Add new connections if below minimum
            while len(self.connections) < self.config.min_connections:
                connection_id = f"health-{int(time.time())}-{len(self.connections)}"
                connection = self._create_connection(connection_id)
                if connection.connect():
                    self.connections.append(connection)
                    self.metrics.total_connections += 1
                    logger.info(f"Added new connection during health check: {connection_id}")
                else:
                    break
    
    @contextmanager
    def get_connection(self):
        """Get a connection from the pool"""
        connection = None
        
        if not self.circuit_breaker.can_execute():
            raise Exception("Circuit breaker is open - too many failures")
        
        try:
            connection = self._acquire_connection()
            start_time = time.time()
            
            yield connection.client
            
            # Record success
            self.circuit_breaker.record_success()
            self.metrics.successful_requests += 1
            response_time = time.time() - start_time
            self._update_response_time(response_time)
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.metrics.failed_requests += 1
            logger.error(f"Connection operation failed: {e}")
            raise
        finally:
            if connection:
                self._release_connection(connection)
    
    def _acquire_connection(self) -> WeaviateConnection:
        """Acquire a connection from the pool"""
        with self._lock:
            # Find an available healthy connection
            for connection in self.connections:
                if connection.connection_id not in self.active_connections and connection.is_healthy():
                    self.active_connections[connection.connection_id] = connection
                    connection.last_used = datetime.utcnow()
                    return connection
            
            # Create new connection if under max limit
            if len(self.connections) < self.config.max_connections:
                connection_id = f"demand-{int(time.time())}-{len(self.connections)}"
                connection = self._create_connection(connection_id)
                if connection.connect():
                    self.connections.append(connection)
                    self.active_connections[connection.connection_id] = connection
                    self.metrics.total_connections += 1
                    return connection
            
            # Wait for connection to become available (simplified)
            raise Exception("No connections available in pool")
    
    def _release_connection(self, connection: WeaviateConnection):
        """Release a connection back to the pool"""
        with self._lock:
            if connection.connection_id in self.active_connections:
                del self.active_connections[connection.connection_id]
    
    def _update_response_time(self, response_time: float):
        """Update average response time"""
        if self.metrics.average_response_time == 0:
            self.metrics.average_response_time = response_time
        else:
            # Exponential moving average
            self.metrics.average_response_time = (
                0.9 * self.metrics.average_response_time + 0.1 * response_time
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            "total_connections": self.metrics.total_connections,
            "active_connections": len(self.active_connections),
            "available_connections": len(self.connections) - len(self.active_connections),
            "failed_connections": self.metrics.failed_connections,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": self.metrics.success_rate(),
            "average_response_time": self.metrics.average_response_time,
            "uptime": str(self.metrics.uptime()),
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
            "last_health_check": self.metrics.last_health_check.isoformat() if self.metrics.last_health_check else None
        }
    
    def close(self):
        """Close the connection pool"""
        logger.info("Closing Weaviate connection pool")
        self._shutdown = True
        
        with self._lock:
            for connection in self.connections:
                connection.close()
            self.connections.clear()
            self.active_connections.clear()
        
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)


class WeaviateClientManager:
    """Centralized Weaviate client manager"""
    
    def __init__(self, config: Optional[ConnectionConfig] = None):
        self.config = config or ConnectionConfig.from_env()
        self.pool: Optional[WeaviateConnectionPool] = None
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="WeaviateOp")
        self._initialized = False
        self._lock = threading.Lock()
    
    def initialize(self):
        """Initialize the client manager"""
        with self._lock:
            if self._initialized:
                return
            
            logger.info("Initializing Weaviate Client Manager")
            try:
                self.pool = WeaviateConnectionPool(self.config)
                self._initialized = True
                logger.info("Weaviate Client Manager initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Weaviate Client Manager: {e}")
                raise
    
    def ensure_initialized(self):
        """Ensure the manager is initialized"""
        if not self._initialized:
            self.initialize()
    
    @contextmanager
    def get_client(self):
        """Get a client from the pool"""
        self.ensure_initialized()
        with self.pool.get_connection() as client:
            yield client
    
    async def execute_query(self, query_func, *args, **kwargs):
        """Execute a query asynchronously"""
        loop = asyncio.get_event_loop()
        
        def _execute():
            with self.get_client() as client:
                return query_func(client, *args, **kwargs)
        
        return await loop.run_in_executor(self.executor, _execute)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        try:
            pool_stats = self.pool.get_stats()
            
            # Determine overall health
            if pool_stats["available_connections"] == 0:
                health_status = "critical"
            elif pool_stats["success_rate"] < 0.9:
                health_status = "degraded"
            elif pool_stats["circuit_breaker_state"] != "closed":
                health_status = "warning"
            else:
                health_status = "healthy"
            
            return {
                "status": health_status,
                "initialized": self._initialized,
                "pool_stats": pool_stats,
                "config": {
                    "http_host": self.config.http_host,
                    "http_port": self.config.http_port,
                    "max_connections": self.config.max_connections,
                    "min_connections": self.config.min_connections
                }
            }
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def close(self):
        """Close the client manager"""
        with self._lock:
            if self.pool:
                self.pool.close()
                self.pool = None
            
            self.executor.shutdown(wait=True)
            self._initialized = False
        
        logger.info("Weaviate Client Manager closed")


# Global client manager instance
_client_manager: Optional[WeaviateClientManager] = None


def get_client_manager() -> WeaviateClientManager:
    """Get the global client manager instance"""
    global _client_manager
    if _client_manager is None:
        _client_manager = WeaviateClientManager()
        _client_manager.initialize()
    return _client_manager


def initialize_client_manager(config: Optional[ConnectionConfig] = None):
    """Initialize the global client manager with custom config"""
    global _client_manager
    _client_manager = WeaviateClientManager(config)
    _client_manager.initialize()
    return _client_manager


def close_client_manager():
    """Close the global client manager"""
    global _client_manager
    if _client_manager:
        _client_manager.close()
        _client_manager = None


# Context manager for easy usage
@contextmanager
def weaviate_client():
    """Context manager for getting a Weaviate client"""
    manager = get_client_manager()
    with manager.get_client() as client:
        yield client