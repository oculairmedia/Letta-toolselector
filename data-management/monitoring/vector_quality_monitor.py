"""
Vector Quality Monitor for Letta Tool Selector
Monitors embedding quality, detects anomalies, and provides quality insights.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import aiohttp
import aiofiles
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from ..core.embedding_versioning import EmbeddingVersionManager, EmbeddingVersion


@dataclass
class QualityAlert:
    """Represents a quality alert for embeddings."""
    alert_id: str
    severity: str  # low, medium, high, critical
    alert_type: str  # outlier, degradation, corruption, cluster_drift
    tool_id: str
    version_id: str
    message: str
    metrics: Dict[str, float]
    created_at: str
    resolved: bool = False


@dataclass
class QualityReport:
    """Quality monitoring report."""
    report_id: str
    generated_at: str
    period_start: str
    period_end: str
    total_embeddings: int
    quality_distribution: Dict[str, int]
    alerts: List[QualityAlert]
    recommendations: List[str]
    health_score: float
    trends: Dict[str, Any]


class VectorQualityMonitor:
    """Monitors vector quality with anomaly detection and trend analysis."""
    
    def __init__(
        self, 
        storage_path: str = "/opt/stacks/lettatoolsselector/data-management/monitoring",
        weaviate_url: str = "http://weaviate:8080"
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.alerts_file = self.storage_path / "quality_alerts.json"
        self.reports_file = self.storage_path / "quality_reports.json"
        self.trends_file = self.storage_path / "quality_trends.json"
        
        self.version_manager = EmbeddingVersionManager()
        self.weaviate_url = weaviate_url
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.thresholds = {
            "min_coherence": 0.1,
            "max_coherence": 2.0,
            "min_diversity": 0.2,
            "max_similarity": 0.95,
            "outlier_std_threshold": 3.0,
            "cluster_drift_threshold": 0.3
        }
    
    async def _load_alerts(self) -> List[QualityAlert]:
        """Load quality alerts from storage."""
        if not self.alerts_file.exists():
            return []
        
        try:
            async with aiofiles.open(self.alerts_file, 'r') as f:
                data = json.loads(await f.read())
                return [QualityAlert(**alert_data) for alert_data in data]
        except Exception as e:
            self.logger.error(f"Error loading alerts: {e}")
            return []
    
    async def _save_alerts(self, alerts: List[QualityAlert]) -> None:
        """Save quality alerts to storage."""
        try:
            data = [asdict(alert) for alert in alerts]
            async with aiofiles.open(self.alerts_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving alerts: {e}")
            raise
    
    async def _load_reports(self) -> List[QualityReport]:
        """Load quality reports from storage."""
        if not self.reports_file.exists():
            return []
        
        try:
            async with aiofiles.open(self.reports_file, 'r') as f:
                data = json.loads(await f.read())
                return [QualityReport(**report_data) for report_data in data]
        except Exception as e:
            self.logger.error(f"Error loading reports: {e}")
            return []
    
    async def _save_reports(self, reports: List[QualityReport]) -> None:
        """Save quality reports to storage."""
        try:
            data = [asdict(report) for report in reports]
            async with aiofiles.open(self.reports_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except Exception as e:
            self.logger.error(f"Error saving reports: {e}")
            raise
    
    async def _get_weaviate_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Fetch embeddings from Weaviate for comparison."""
        embeddings = []
        
        try:
            async with aiohttp.ClientSession() as session:
                query = {
                    "query": """
                    {
                        Get {
                            Tool {
                                _id
                                name
                                description
                                _additional {
                                    vector
                                }
                            }
                        }
                    }
                    """
                }
                
                async with session.post(
                    f"{self.weaviate_url}/v1/graphql",
                    json=query
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tools = data.get("data", {}).get("Get", {}).get("Tool", [])
                        
                        for tool in tools:
                            if "_additional" in tool and "vector" in tool["_additional"]:
                                tool_id = tool.get("_id", "unknown")
                                vector = np.array(tool["_additional"]["vector"])
                                embeddings.append((tool_id, vector))
        
        except Exception as e:
            self.logger.error(f"Error fetching Weaviate embeddings: {e}")
        
        return embeddings
    
    async def detect_outliers(self) -> List[QualityAlert]:
        """Detect outlier embeddings using statistical methods."""
        alerts = []
        
        try:
            # Get embeddings from version manager
            versions = await self.version_manager._load_versions()
            embeddings_data = []
            
            for version in versions.values():
                embedding = await self.version_manager._load_embedding(version.version_id)
                if embedding is not None:
                    embeddings_data.append((version, embedding))
            
            if len(embeddings_data) < 10:  # Need minimum samples
                return alerts
            
            # Extract vectors and compute statistics
            vectors = np.array([emb for _, emb in embeddings_data])
            norms = np.linalg.norm(vectors, axis=1)
            
            # Z-score based outlier detection
            z_scores = np.abs(stats.zscore(norms))
            outlier_indices = np.where(z_scores > self.thresholds["outlier_std_threshold"])[0]
            
            # DBSCAN clustering for geometric outliers
            scaler = StandardScaler()
            scaled_vectors = scaler.fit_transform(vectors)
            
            # Use PCA for dimensionality reduction if needed
            if scaled_vectors.shape[1] > 100:
                pca = PCA(n_components=100)
                scaled_vectors = pca.fit_transform(scaled_vectors)
            
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            cluster_labels = dbscan.fit_predict(scaled_vectors)
            
            # Identify outliers (label = -1)
            cluster_outlier_indices = np.where(cluster_labels == -1)[0]
            
            # Combine outliers
            all_outlier_indices = set(outlier_indices) | set(cluster_outlier_indices)
            
            for idx in all_outlier_indices:
                version, embedding = embeddings_data[idx]
                
                alert = QualityAlert(
                    alert_id=f"outlier_{version.version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    severity="medium" if idx in outlier_indices else "low",
                    alert_type="outlier",
                    tool_id=version.tool_id,
                    version_id=version.version_id,
                    message=f"Outlier embedding detected for tool {version.tool_name}",
                    metrics={
                        "norm": float(norms[idx]),
                        "z_score": float(z_scores[idx]) if idx in outlier_indices else 0.0,
                        "cluster_label": int(cluster_labels[idx])
                    },
                    created_at=datetime.now(timezone.utc).isoformat()
                )
                alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error in outlier detection: {e}")
        
        return alerts
    
    async def detect_quality_degradation(self, days_back: int = 7) -> List[QualityAlert]:
        """Detect quality degradation over time."""
        alerts = []
        
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_back)
            versions = await self.version_manager._load_versions()
            metrics = await self.version_manager._load_metrics()
            
            # Group by tool and analyze trends
            tool_metrics = {}
            for version in versions.values():
                version_date = datetime.fromisoformat(version.created_at.replace('Z', '+00:00'))
                if version_date < cutoff_date:
                    continue
                
                tool_id = version.tool_id
                version_id = version.version_id
                
                if version_id in metrics:
                    if tool_id not in tool_metrics:
                        tool_metrics[tool_id] = []
                    
                    tool_metrics[tool_id].append({
                        'date': version_date,
                        'metrics': metrics[version_id],
                        'version': version
                    })
            
            # Check for degradation trends
            for tool_id, tool_data in tool_metrics.items():
                if len(tool_data) < 2:
                    continue
                
                # Sort by date
                tool_data.sort(key=lambda x: x['date'])
                
                # Check coherence trend
                coherence_scores = [d['metrics'].similarity_score for d in tool_data]
                diversity_scores = [d['metrics'].diversity_score for d in tool_data]
                
                if len(coherence_scores) >= 3:
                    # Simple trend detection using linear regression slope
                    x = np.arange(len(coherence_scores))
                    coherence_slope = np.polyfit(x, coherence_scores, 1)[0]
                    diversity_slope = np.polyfit(x, diversity_scores, 1)[0]
                    
                    if coherence_slope < -0.1 or diversity_slope < -0.1:
                        latest_version = tool_data[-1]['version']
                        
                        alert = QualityAlert(
                            alert_id=f"degradation_{tool_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                            severity="medium",
                            alert_type="degradation",
                            tool_id=tool_id,
                            version_id=latest_version.version_id,
                            message=f"Quality degradation detected for tool {latest_version.tool_name}",
                            metrics={
                                "coherence_slope": float(coherence_slope),
                                "diversity_slope": float(diversity_slope),
                                "samples": len(coherence_scores)
                            },
                            created_at=datetime.now(timezone.utc).isoformat()
                        )
                        alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error in degradation detection: {e}")
        
        return alerts
    
    async def detect_embedding_corruption(self) -> List[QualityAlert]:
        """Detect potentially corrupted embeddings."""
        alerts = []
        
        try:
            versions = await self.version_manager._load_versions()
            
            for version in versions.values():
                embedding = await self.version_manager._load_embedding(version.version_id)
                if embedding is None:
                    continue
                
                # Check for common corruption patterns
                corruption_detected = False
                corruption_type = ""
                corruption_metrics = {}
                
                # Check for NaN or infinite values
                if np.isnan(embedding).any():
                    corruption_detected = True
                    corruption_type = "nan_values"
                    corruption_metrics["nan_count"] = int(np.isnan(embedding).sum())
                
                if np.isinf(embedding).any():
                    corruption_detected = True
                    corruption_type = "infinite_values"
                    corruption_metrics["inf_count"] = int(np.isinf(embedding).sum())
                
                # Check for zero vectors
                if np.allclose(embedding, 0):
                    corruption_detected = True
                    corruption_type = "zero_vector"
                    corruption_metrics["norm"] = float(np.linalg.norm(embedding))
                
                # Check for abnormal patterns (all values same, etc.)
                if len(set(embedding.round(6))) == 1 and not np.allclose(embedding, 0):
                    corruption_detected = True
                    corruption_type = "uniform_values"
                    corruption_metrics["unique_values"] = len(set(embedding.round(6)))
                
                if corruption_detected:
                    alert = QualityAlert(
                        alert_id=f"corruption_{version.version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        severity="high",
                        alert_type="corruption",
                        tool_id=version.tool_id,
                        version_id=version.version_id,
                        message=f"Embedding corruption detected ({corruption_type}) for tool {version.tool_name}",
                        metrics=corruption_metrics,
                        created_at=datetime.now(timezone.utc).isoformat()
                    )
                    alerts.append(alert)
        
        except Exception as e:
            self.logger.error(f"Error in corruption detection: {e}")
        
        return alerts
    
    async def run_quality_check(self) -> List[QualityAlert]:
        """Run comprehensive quality check."""
        all_alerts = []
        
        # Run all detection methods
        outlier_alerts = await self.detect_outliers()
        degradation_alerts = await self.detect_quality_degradation()
        corruption_alerts = await self.detect_embedding_corruption()
        
        all_alerts.extend(outlier_alerts)
        all_alerts.extend(degradation_alerts)
        all_alerts.extend(corruption_alerts)
        
        # Save alerts
        existing_alerts = await self._load_alerts()
        existing_alerts.extend(all_alerts)
        await self._save_alerts(existing_alerts)
        
        if all_alerts:
            self.logger.info(f"Quality check generated {len(all_alerts)} alerts")
        
        return all_alerts
    
    async def generate_quality_report(self, days_back: int = 7) -> QualityReport:
        """Generate comprehensive quality report."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        # Get data
        versions = await self.version_manager._load_versions()
        metrics = await self.version_manager._load_metrics()
        alerts = await self._load_alerts()
        
        # Filter by date range
        recent_versions = [
            v for v in versions.values()
            if datetime.fromisoformat(v.created_at.replace('Z', '+00:00')) >= start_date
        ]
        
        recent_alerts = [
            a for a in alerts
            if datetime.fromisoformat(a.created_at.replace('Z', '+00:00')) >= start_date
            and not a.resolved
        ]
        
        # Quality distribution
        quality_dist = {"A": 0, "B": 0, "C": 0, "D": 0, "F": 0}
        for version in recent_versions:
            version_metrics = metrics.get(version.version_id)
            if version_metrics:
                quality_dist[version_metrics.quality_grade] += 1
        
        # Calculate health score
        total_embeddings = sum(quality_dist.values())
        if total_embeddings > 0:
            health_score = (
                quality_dist["A"] * 1.0 +
                quality_dist["B"] * 0.8 +
                quality_dist["C"] * 0.6 +
                quality_dist["D"] * 0.4 +
                quality_dist["F"] * 0.0
            ) / total_embeddings
        else:
            health_score = 0.0
        
        # Generate recommendations
        recommendations = []
        if quality_dist["F"] > 0:
            recommendations.append(f"Investigate {quality_dist['F']} failed-quality embeddings")
        if quality_dist["D"] > quality_dist["A"]:
            recommendations.append("Consider recomputing embeddings with updated models")
        if len(recent_alerts) > 0:
            recommendations.append(f"Address {len(recent_alerts)} active quality alerts")
        if health_score < 0.7:
            recommendations.append("Overall embedding quality needs attention")
        
        report = QualityReport(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generated_at=end_date.isoformat(),
            period_start=start_date.isoformat(),
            period_end=end_date.isoformat(),
            total_embeddings=total_embeddings,
            quality_distribution=quality_dist,
            alerts=recent_alerts,
            recommendations=recommendations,
            health_score=health_score,
            trends={
                "embeddings_created": len(recent_versions),
                "alerts_generated": len(recent_alerts),
                "avg_quality": sum(
                    ord(grade) - ord('A') for grade in quality_dist.keys()
                    for _ in range(quality_dist[grade])
                ) / max(total_embeddings, 1)
            }
        )
        
        # Save report
        existing_reports = await self._load_reports()
        existing_reports.append(report)
        # Keep only last 30 reports
        if len(existing_reports) > 30:
            existing_reports = existing_reports[-30:]
        await self._save_reports(existing_reports)
        
        return report
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        alerts = await self._load_alerts()
        active_alerts = [a for a in alerts if not a.resolved]
        
        # Count by severity
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for alert in active_alerts:
            severity_counts[alert.severity] += 1
        
        # Overall status
        if severity_counts["critical"] > 0:
            status = "critical"
        elif severity_counts["high"] > 0:
            status = "unhealthy"
        elif severity_counts["medium"] > 5:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "active_alerts": len(active_alerts),
            "severity_breakdown": severity_counts,
            "last_check": datetime.now(timezone.utc).isoformat(),
            "actions_needed": severity_counts["critical"] + severity_counts["high"]
        }


async def main():
    """Example usage and testing."""
    logging.basicConfig(level=logging.INFO)
    
    monitor = VectorQualityMonitor()
    
    print("Running quality check...")
    alerts = await monitor.run_quality_check()
    print(f"Generated {len(alerts)} alerts")
    
    print("\nGenerating quality report...")
    report = await monitor.generate_quality_report(days_back=30)
    print(f"Health score: {report.health_score:.3f}")
    print(f"Total embeddings analyzed: {report.total_embeddings}")
    
    print("\nHealth status:")
    status = await monitor.get_health_status()
    for key, value in status.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())