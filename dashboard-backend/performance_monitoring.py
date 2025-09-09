"""
LDTS-45: Performance Monitoring and SLA Tracking
Real-time monitoring of system performance with SLA compliance tracking
"""

import asyncio
import json
import uuid
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
import logging
import psutil
import numpy as np
from collections import deque, defaultdict
import aiohttp
from contextlib import asynccontextmanager

class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"
    RESOURCE_USAGE = "resource_usage"
    QUEUE_DEPTH = "queue_depth"

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SLAStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    BREACHED = "breached"
    UNKNOWN = "unknown"

@dataclass
class SLAThreshold:
    """SLA threshold definition"""
    metric_type: MetricType
    threshold_value: float
    comparison: str  # "lt", "le", "gt", "ge", "eq", "ne"
    duration_seconds: int = 300  # How long threshold must be exceeded
    description: str = ""

@dataclass
class SLADefinition:
    """Service Level Agreement definition"""
    id: str
    name: str
    description: str
    service: str
    
    # Thresholds
    thresholds: List[SLAThreshold]
    
    # SLA targets
    availability_target: float = 99.9  # 99.9% uptime
    response_time_p95_ms: float = 1000  # 95th percentile response time
    error_rate_threshold: float = 0.01  # 1% error rate
    
    # Measurement window
    measurement_window_hours: int = 24
    
    # Notification settings
    alert_emails: List[str] = field(default_factory=list)
    alert_webhooks: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    active: bool = True
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricDataPoint:
    """Single metric measurement"""
    timestamp: datetime
    service: str
    metric_type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Performance alert generated when SLA is breached"""
    id: str
    sla_id: str
    service: str
    metric_type: MetricType
    severity: AlertSeverity
    
    # Alert details
    threshold_value: float
    actual_value: float
    breach_duration_seconds: int
    
    # Status
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    
    # Notification
    notifications_sent: List[str] = field(default_factory=list)
    
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SLAReport:
    """SLA compliance report"""
    id: str
    sla_id: str
    service: str
    
    # Reporting period
    period_start: datetime
    period_end: datetime
    
    # Metrics
    availability_percentage: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    
    # Response times
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    
    # SLA compliance
    sla_status: SLAStatus
    breaches: List[str] = field(default_factory=list)  # Alert IDs
    
    # Resource usage
    avg_cpu_usage: Optional[float] = None
    avg_memory_usage: Optional[float] = None
    peak_cpu_usage: Optional[float] = None
    peak_memory_usage: Optional[float] = None
    
    generated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, storage_path: str = "monitoring_data", buffer_size: int = 10000):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.slas_path = self.storage_path / "slas"
        self.alerts_path = self.storage_path / "alerts"
        self.reports_path = self.storage_path / "reports"
        self.metrics_path = self.storage_path / "metrics"
        
        for path in [self.slas_path, self.alerts_path, self.reports_path, self.metrics_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # In-memory metric buffers for real-time monitoring
        self.metric_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_task = None
        self.alert_task = None
        
        # Performance counters
        self.request_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.response_times = defaultdict(list)
        
        # Resource monitoring
        self.system_metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
    
    # SLA Management
    async def create_sla(self, sla: SLADefinition) -> bool:
        """Create a new SLA definition"""
        try:
            sla_path = self.slas_path / f"{sla.id}.json"
            with open(sla_path, 'w') as f:
                json.dump(asdict(sla), f, indent=2, default=str)
            
            self.logger.info(f"Created SLA definition {sla.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create SLA {sla.id}: {e}")
            return False
    
    async def load_sla(self, sla_id: str) -> Optional[SLADefinition]:
        """Load SLA definition"""
        try:
            sla_path = self.slas_path / f"{sla_id}.json"
            if not sla_path.exists():
                return None
            
            with open(sla_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings and enums
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
            
            # Convert thresholds
            thresholds = []
            for threshold_data in data['thresholds']:
                threshold_data['metric_type'] = MetricType(threshold_data['metric_type'])
                thresholds.append(SLAThreshold(**threshold_data))
            data['thresholds'] = thresholds
            
            return SLADefinition(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load SLA {sla_id}: {e}")
            return None
    
    async def list_slas(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all SLA definitions"""
        slas = []
        
        for sla_file in self.slas_path.glob("*.json"):
            try:
                sla = await self.load_sla(sla_file.stem)
                if sla and (not active_only or sla.active):
                    slas.append({
                        'id': sla.id,
                        'name': sla.name,
                        'service': sla.service,
                        'availability_target': sla.availability_target,
                        'response_time_p95_ms': sla.response_time_p95_ms,
                        'active': sla.active,
                        'created_at': sla.created_at.isoformat()
                    })
            except Exception as e:
                self.logger.error(f"Failed to load SLA info from {sla_file}: {e}")
                continue
        
        return sorted(slas, key=lambda x: x['created_at'], reverse=True)
    
    # Metric Collection
    async def record_metric(
        self,
        service: str,
        metric_type: MetricType,
        value: float,
        labels: Dict[str, str] = None,
        timestamp: Optional[datetime] = None
    ):
        """Record a performance metric"""
        
        metric = MetricDataPoint(
            timestamp=timestamp or datetime.utcnow(),
            service=service,
            metric_type=metric_type,
            value=value,
            labels=labels or {}
        )
        
        # Add to buffer
        buffer_key = f"{service}:{metric_type.value}"
        self.metric_buffers[buffer_key].append(metric)
        
        # Update counters based on metric type
        if metric_type == MetricType.THROUGHPUT:
            self.request_counters[service] += int(value)
        elif metric_type == MetricType.ERROR_RATE:
            self.error_counters[service] += int(value)
        elif metric_type == MetricType.LATENCY:
            self.response_times[service].append(value)
            # Keep only recent response times
            if len(self.response_times[service]) > 1000:
                self.response_times[service] = self.response_times[service][-1000:]
        
        # Persist to disk periodically
        await self._maybe_persist_metrics()
    
    async def record_request(
        self,
        service: str,
        response_time_ms: float,
        success: bool = True,
        labels: Dict[str, str] = None
    ):
        """Record a request with response time and success status"""
        
        timestamp = datetime.utcnow()
        
        # Record latency
        await self.record_metric(service, MetricType.LATENCY, response_time_ms, labels, timestamp)
        
        # Record throughput (1 request)
        await self.record_metric(service, MetricType.THROUGHPUT, 1.0, labels, timestamp)
        
        # Record error if failed
        if not success:
            await self.record_metric(service, MetricType.ERROR_RATE, 1.0, labels, timestamp)
    
    @asynccontextmanager
    async def measure_request(self, service: str, labels: Dict[str, str] = None):
        """Context manager to automatically measure request performance"""
        start_time = time.time()
        success = True
        
        try:
            yield
        except Exception as e:
            success = False
            raise
        finally:
            response_time_ms = (time.time() - start_time) * 1000
            await self.record_request(service, response_time_ms, success, labels)
    
    async def _maybe_persist_metrics(self, force: bool = False):
        """Persist metrics to disk if buffer is full or forced"""
        
        # Simple persistence strategy - persist every 100 metrics
        total_buffered = sum(len(buffer) for buffer in self.metric_buffers.values())
        
        if force or total_buffered > 1000:
            await self._persist_metrics()
    
    async def _persist_metrics(self):
        """Persist buffered metrics to disk"""
        
        try:
            # Create daily metric files
            today = datetime.utcnow().strftime("%Y-%m-%d")
            
            for buffer_key, buffer in self.metric_buffers.items():
                if not buffer:
                    continue
                
                service, metric_type = buffer_key.split(":", 1)
                metric_file = self.metrics_path / f"{today}_{service}_{metric_type}.jsonl"
                
                # Append new metrics
                with open(metric_file, 'a') as f:
                    while buffer:
                        metric = buffer.popleft()
                        f.write(json.dumps(asdict(metric), default=str) + '\n')
            
        except Exception as e:
            self.logger.error(f"Failed to persist metrics: {e}")
    
    # System Resource Monitoring
    async def collect_system_metrics(self):
        """Collect system resource metrics"""
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Unix only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                pass  # Windows doesn't have load average
            
            system_metrics = {
                'timestamp': datetime.utcnow(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'disk_percent': disk_percent,
                'disk_free_gb': disk_free_gb,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_count': process_count,
                'load_avg_1min': load_avg[0] if load_avg else None,
                'load_avg_5min': load_avg[1] if load_avg else None,
                'load_avg_15min': load_avg[2] if load_avg else None
            }
            
            self.system_metrics_history.append(system_metrics)
            
            # Record as metrics for SLA monitoring
            await self.record_metric("system", MetricType.RESOURCE_USAGE, cpu_percent, 
                                   {"resource": "cpu"})
            await self.record_metric("system", MetricType.RESOURCE_USAGE, memory_percent,
                                   {"resource": "memory"})
            await self.record_metric("system", MetricType.RESOURCE_USAGE, disk_percent,
                                   {"resource": "disk"})
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    # SLA Monitoring and Alerting
    async def start_monitoring(self, check_interval_seconds: int = 60):
        """Start continuous SLA monitoring"""
        
        if self.monitoring_active:
            self.logger.warning("Monitoring is already active")
            return
        
        self.monitoring_active = True
        
        # Start monitoring tasks
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(check_interval_seconds)
        )
        self.alert_task = asyncio.create_task(
            self._alert_processing_loop()
        )
        
        self.logger.info("Started performance monitoring")
    
    async def stop_monitoring(self):
        """Stop continuous SLA monitoring"""
        
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
        if self.alert_task:
            self.alert_task.cancel()
        
        # Persist any remaining metrics
        await self._persist_metrics()
        
        self.logger.info("Stopped performance monitoring")
    
    async def _monitoring_loop(self, check_interval_seconds: int):
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                # Collect system metrics
                await self.collect_system_metrics()
                
                # Check all active SLAs
                slas = await self.list_slas(active_only=True)
                for sla_info in slas:
                    await self._check_sla_compliance(sla_info['id'])
                
                # Wait for next check
                await asyncio.sleep(check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(check_interval_seconds)
    
    async def _alert_processing_loop(self):
        """Process and send alerts"""
        
        while self.monitoring_active:
            try:
                # Process active alerts
                alerts_to_process = list(self.active_alerts.values())
                
                for alert in alerts_to_process:
                    if alert.resolved_at is None:
                        await self._process_alert(alert)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in alert processing loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_sla_compliance(self, sla_id: str):
        """Check if SLA thresholds are being met"""
        
        try:
            sla = await self.load_sla(sla_id)
            if not sla or not sla.active:
                return
            
            # Check each threshold
            for threshold in sla.thresholds:
                await self._check_threshold(sla, threshold)
                
        except Exception as e:
            self.logger.error(f"Failed to check SLA compliance for {sla_id}: {e}")
    
    async def _check_threshold(self, sla: SLADefinition, threshold: SLAThreshold):
        """Check a specific SLA threshold"""
        
        try:
            # Get recent metrics for this threshold
            buffer_key = f"{sla.service}:{threshold.metric_type.value}"
            buffer = self.metric_buffers.get(buffer_key, deque())
            
            if not buffer:
                return
            
            # Get metrics from the last duration_seconds
            cutoff_time = datetime.utcnow() - timedelta(seconds=threshold.duration_seconds)
            recent_metrics = [m for m in buffer if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return
            
            # Calculate aggregate value (average for now)
            values = [m.value for m in recent_metrics]
            current_value = np.mean(values)
            
            # Check if threshold is breached
            is_breached = self._evaluate_threshold(current_value, threshold.threshold_value, threshold.comparison)
            
            # Generate alert if breached
            if is_breached:
                await self._trigger_alert(sla, threshold, current_value)
            else:
                # Resolve existing alert if threshold is no longer breached
                await self._resolve_alert(sla, threshold)
                
        except Exception as e:
            self.logger.error(f"Failed to check threshold for {sla.id}: {e}")
    
    def _evaluate_threshold(self, current_value: float, threshold_value: float, comparison: str) -> bool:
        """Evaluate if a threshold is breached"""
        
        if comparison == "gt":
            return current_value > threshold_value
        elif comparison == "ge":
            return current_value >= threshold_value
        elif comparison == "lt":
            return current_value < threshold_value
        elif comparison == "le":
            return current_value <= threshold_value
        elif comparison == "eq":
            return abs(current_value - threshold_value) < 0.001
        elif comparison == "ne":
            return abs(current_value - threshold_value) >= 0.001
        else:
            return False
    
    async def _trigger_alert(self, sla: SLADefinition, threshold: SLAThreshold, current_value: float):
        """Trigger a performance alert"""
        
        alert_key = f"{sla.id}:{threshold.metric_type.value}"
        
        # Check if alert already exists
        if alert_key in self.active_alerts:
            # Update existing alert
            existing_alert = self.active_alerts[alert_key]
            existing_alert.actual_value = current_value
            existing_alert.breach_duration_seconds = int(
                (datetime.utcnow() - existing_alert.triggered_at).total_seconds()
            )
        else:
            # Create new alert
            severity = self._determine_alert_severity(current_value, threshold.threshold_value)
            
            alert = PerformanceAlert(
                id=str(uuid.uuid4()),
                sla_id=sla.id,
                service=sla.service,
                metric_type=threshold.metric_type,
                severity=severity,
                threshold_value=threshold.threshold_value,
                actual_value=current_value,
                breach_duration_seconds=threshold.duration_seconds,
                triggered_at=datetime.utcnow(),
                message=f"SLA breach: {threshold.description or threshold.metric_type.value} exceeded threshold"
            )
            
            self.active_alerts[alert_key] = alert
            await self._save_alert(alert)
            
            self.logger.warning(f"SLA alert triggered: {alert.id} for {sla.service}")
    
    async def _resolve_alert(self, sla: SLADefinition, threshold: SLAThreshold):
        """Resolve an active alert if threshold is no longer breached"""
        
        alert_key = f"{sla.id}:{threshold.metric_type.value}"
        
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            alert.resolved_at = datetime.utcnow()
            
            await self._save_alert(alert)
            del self.active_alerts[alert_key]
            
            self.logger.info(f"SLA alert resolved: {alert.id}")
    
    def _determine_alert_severity(self, current_value: float, threshold_value: float) -> AlertSeverity:
        """Determine alert severity based on how much threshold is exceeded"""
        
        if threshold_value == 0:
            return AlertSeverity.WARNING
        
        breach_ratio = abs(current_value - threshold_value) / abs(threshold_value)
        
        if breach_ratio > 1.0:  # 100% over threshold
            return AlertSeverity.EMERGENCY
        elif breach_ratio > 0.5:  # 50% over threshold
            return AlertSeverity.CRITICAL
        elif breach_ratio > 0.2:  # 20% over threshold
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    async def _process_alert(self, alert: PerformanceAlert):
        """Process an alert (send notifications, etc.)"""
        
        try:
            # Load SLA to get notification settings
            sla = await self.load_sla(alert.sla_id)
            if not sla:
                return
            
            # Send email notifications
            for email in sla.alert_emails:
                if email not in alert.notifications_sent:
                    success = await self._send_email_alert(alert, email)
                    if success:
                        alert.notifications_sent.append(email)
            
            # Send webhook notifications
            for webhook_url in sla.alert_webhooks:
                if webhook_url not in alert.notifications_sent:
                    success = await self._send_webhook_alert(alert, webhook_url)
                    if success:
                        alert.notifications_sent.append(webhook_url)
            
            # Update alert
            await self._save_alert(alert)
            
        except Exception as e:
            self.logger.error(f"Failed to process alert {alert.id}: {e}")
    
    async def _send_email_alert(self, alert: PerformanceAlert, email: str) -> bool:
        """Send email alert (mock implementation)"""
        # In real implementation, this would use an email service
        self.logger.info(f"Mock email alert sent to {email}: {alert.message}")
        return True
    
    async def _send_webhook_alert(self, alert: PerformanceAlert, webhook_url: str) -> bool:
        """Send webhook alert"""
        try:
            payload = {
                'alert_id': alert.id,
                'service': alert.service,
                'metric_type': alert.metric_type.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'threshold_value': alert.threshold_value,
                'actual_value': alert.actual_value,
                'triggered_at': alert.triggered_at.isoformat()
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert to {webhook_url}: {e}")
            return False
    
    async def _save_alert(self, alert: PerformanceAlert):
        """Save alert to storage"""
        alert_path = self.alerts_path / f"{alert.id}.json"
        with open(alert_path, 'w') as f:
            json.dump(asdict(alert), f, indent=2, default=str)
    
    # Reporting
    async def generate_sla_report(
        self,
        sla_id: str,
        period_start: datetime,
        period_end: datetime
    ) -> Optional[str]:
        """Generate SLA compliance report"""
        
        try:
            sla = await self.load_sla(sla_id)
            if not sla:
                return None
            
            # Calculate metrics for the period
            total_requests = self.request_counters.get(sla.service, 0)
            failed_requests = self.error_counters.get(sla.service, 0)
            successful_requests = total_requests - failed_requests
            
            # Calculate availability
            availability_percentage = (successful_requests / total_requests * 100) if total_requests > 0 else 100.0
            
            # Calculate response times
            response_times = self.response_times.get(sla.service, [])
            avg_response_time = np.mean(response_times) if response_times else 0.0
            p50_response_time = np.percentile(response_times, 50) if response_times else 0.0
            p95_response_time = np.percentile(response_times, 95) if response_times else 0.0
            p99_response_time = np.percentile(response_times, 99) if response_times else 0.0
            
            # Calculate SLA status
            sla_status = SLAStatus.HEALTHY
            if availability_percentage < sla.availability_target:
                sla_status = SLAStatus.BREACHED
            elif p95_response_time > sla.response_time_p95_ms:
                sla_status = SLAStatus.WARNING
            
            # Get system resource metrics
            avg_cpu = avg_memory = peak_cpu = peak_memory = None
            if self.system_metrics_history:
                cpu_values = [m['cpu_percent'] for m in self.system_metrics_history]
                memory_values = [m['memory_percent'] for m in self.system_metrics_history]
                
                avg_cpu = np.mean(cpu_values)
                avg_memory = np.mean(memory_values)
                peak_cpu = np.max(cpu_values)
                peak_memory = np.max(memory_values)
            
            # Create report
            report_id = str(uuid.uuid4())
            report = SLAReport(
                id=report_id,
                sla_id=sla_id,
                service=sla.service,
                period_start=period_start,
                period_end=period_end,
                availability_percentage=availability_percentage,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                avg_response_time_ms=avg_response_time,
                p50_response_time_ms=p50_response_time,
                p95_response_time_ms=p95_response_time,
                p99_response_time_ms=p99_response_time,
                sla_status=sla_status,
                avg_cpu_usage=avg_cpu,
                avg_memory_usage=avg_memory,
                peak_cpu_usage=peak_cpu,
                peak_memory_usage=peak_memory
            )
            
            # Save report
            report_path = self.reports_path / f"{report_id}.json"
            with open(report_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            
            self.logger.info(f"Generated SLA report {report_id}")
            return report_id
            
        except Exception as e:
            self.logger.error(f"Failed to generate SLA report for {sla_id}: {e}")
            return None
    
    # Query Methods
    async def get_current_metrics(self, service: str, metric_type: MetricType) -> Dict[str, Any]:
        """Get current metric values for a service"""
        
        buffer_key = f"{service}:{metric_type.value}"
        buffer = self.metric_buffers.get(buffer_key, deque())
        
        if not buffer:
            return {'value': None, 'timestamp': None}
        
        # Get most recent metrics
        recent_metrics = list(buffer)[-10:]  # Last 10 measurements
        values = [m.value for m in recent_metrics]
        
        return {
            'current_value': recent_metrics[-1].value,
            'avg_value': np.mean(values),
            'min_value': np.min(values),
            'max_value': np.max(values),
            'timestamp': recent_metrics[-1].timestamp,
            'sample_count': len(values)
        }
    
    async def list_active_alerts(self) -> List[PerformanceAlert]:
        """List all active alerts"""
        return list(self.active_alerts.values())

# Global instance
performance_monitor = PerformanceMonitor()

# Example usage
async def create_example_sla():
    """Create an example SLA definition"""
    
    sla = SLADefinition(
        id=str(uuid.uuid4()),
        name="Search Service SLA",
        description="SLA for search and reranking service",
        service="search_service",
        thresholds=[
            SLAThreshold(
                metric_type=MetricType.LATENCY,
                threshold_value=1000.0,  # 1 second
                comparison="gt",
                duration_seconds=300,  # 5 minutes
                description="Response time should not exceed 1 second"
            ),
            SLAThreshold(
                metric_type=MetricType.ERROR_RATE,
                threshold_value=0.05,  # 5%
                comparison="gt",
                duration_seconds=300,
                description="Error rate should not exceed 5%"
            )
        ],
        availability_target=99.5,
        response_time_p95_ms=800,
        error_rate_threshold=0.01
    )
    
    success = await performance_monitor.create_sla(sla)
    if success:
        print(f"Created SLA: {sla.id}")
        return sla.id
    else:
        print("Failed to create SLA")
        return None

async def test_monitoring():
    """Test the performance monitoring system"""
    
    # Create example SLA
    sla_id = await create_example_sla()
    
    if sla_id:
        print("Starting performance monitoring...")
        
        # Start monitoring
        await performance_monitor.start_monitoring(check_interval_seconds=10)
        
        # Simulate some metrics
        for i in range(5):
            await performance_monitor.record_request("search_service", 500 + i * 100, success=True)
            await asyncio.sleep(1)
        
        # Let monitoring run for a bit
        await asyncio.sleep(15)
        
        # Check current metrics
        metrics = await performance_monitor.get_current_metrics("search_service", MetricType.LATENCY)
        print(f"Current latency metrics: {metrics}")
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        print("Stopped monitoring")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_monitoring())