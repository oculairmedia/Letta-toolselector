"""
LDTS-62: API Integration Layer  
Unified API interface connecting all dashboard components
"""

import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import all components
from dashboard_backend.manual_evaluation_interface import manual_evaluator
from dashboard_backend.automated_batch_evaluation import automated_batch_evaluator
from dashboard_backend.metrics_calculation_engine import metrics_engine
from dashboard_backend.ab_testing_framework import ab_testing_framework
from dashboard_backend.benchmark_query_management import benchmark_manager
from dashboard_backend.evaluation_history_tracking import evaluation_history
from dashboard_backend.performance_monitoring import performance_monitor
from dashboard_backend.custom_metrics_reporting import custom_metrics_system
from data_management import data_ingestion, storage_manager, data_validator
from experiments_engine import experiment_manager

class APIIntegrationLayer:
    """Unified API layer for all dashboard components"""
    
    def __init__(self):
        self.app = FastAPI(
            title="LDTS Reranker Testing Dashboard API",
            description="Comprehensive API for search and reranking evaluation",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        self.logger = logging.getLogger(__name__)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow()}
        
        # Evaluation routes
        @self.app.post("/api/v1/evaluation/manual/session")
        async def create_manual_session(session_name: str, evaluator_id: str):
            session_id = await manual_evaluator.create_evaluation_session(session_name, evaluator_id)
            return {"session_id": session_id}
        
        @self.app.post("/api/v1/evaluation/batch/job")
        async def create_batch_job(
            name: str,
            description: str,
            dataset_id: str,
            configurations: List[str]
        ):
            job_id = await automated_batch_evaluator.create_evaluation_job(
                name, description, dataset_id, configurations
            )
            return {"job_id": job_id}
        
        @self.app.post("/api/v1/evaluation/ab-test")
        async def create_ab_test(
            experiment_config: Dict[str, Any],
            dataset_id: str
        ):
            # Create A/B test experiment
            execution_id = await ab_testing_framework.start_experiment(
                experiment_config['id'], dataset_id
            )
            return {"execution_id": execution_id}
        
        # Metrics routes
        @self.app.get("/api/v1/metrics/{metric_type}")
        async def get_metrics(metric_type: str, config_id: str, k: Optional[int] = None):
            # Calculate and return metrics
            results = await metrics_engine.calculate_single_query_metrics(
                # Mock data for demonstration
                None, None, None
            )
            return {"metrics": [asdict(r) for r in results]}
        
        # Data management routes
        @self.app.post("/api/v1/data/ingest")
        async def create_ingestion_job(
            name: str,
            source_path: str,
            format: str
        ):
            from data_management.data_ingestion_pipeline import DataIngestionJob, DataFormat
            job = DataIngestionJob(
                id=f"job_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name=name,
                description=f"Ingestion job for {source_path}",
                source_type="file",
                source_path=source_path,
                format=DataFormat(format)
            )
            success = await data_ingestion.create_ingestion_job(job)
            return {"success": success, "job_id": job.id}
        
        @self.app.get("/api/v1/data/storage/stats")
        async def get_storage_stats():
            stats = await storage_manager.get_storage_stats()
            return stats
        
        # Benchmark routes
        @self.app.get("/api/v1/benchmarks/collections")
        async def list_benchmark_collections():
            collections = await benchmark_manager.list_collections()
            return {"collections": collections}
        
        @self.app.post("/api/v1/benchmarks/import/msmarco")
        async def import_msmarco(collection_name: str, subset: str = "dev", limit: int = 1000):
            collection_id = await benchmark_manager.import_msmarco_dataset(
                collection_name, subset, limit
            )
            return {"collection_id": collection_id}
        
        # Performance monitoring routes
        @self.app.get("/api/v1/performance/metrics/{service}/{metric_type}")
        async def get_performance_metrics(service: str, metric_type: str):
            from dashboard_backend.performance_monitoring import MetricType
            metrics = await performance_monitor.get_current_metrics(
                service, MetricType(metric_type)
            )
            return metrics
        
        @self.app.get("/api/v1/performance/alerts")
        async def get_active_alerts():
            alerts = await performance_monitor.list_active_alerts()
            return {"alerts": [asdict(alert) for alert in alerts]}
        
        # Reporting routes
        @self.app.post("/api/v1/reports/generate")
        async def generate_report(
            template_id: str,
            start_date: str,
            end_date: str
        ):
            from dashboard_backend.report_template import ReportTemplate
            # Mock report generation
            report_id = f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            return {"report_id": report_id}
        
        # History routes
        @self.app.get("/api/v1/history/runs")
        async def list_evaluation_runs(
            configuration_id: Optional[str] = None,
            limit: int = 50
        ):
            runs = await evaluation_history.list_runs(
                configuration_id=configuration_id,
                limit=limit
            )
            return {"runs": [asdict(run) for run in runs]}
        
        # Experiment routes
        @self.app.post("/api/v1/experiments")
        async def create_experiment(
            name: str,
            description: str,
            experiment_type: str
        ):
            from experiments_engine.experiment_management import Experiment, ExperimentType
            experiment = Experiment(
                id=f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                name=name,
                description=description,
                experiment_type=ExperimentType(experiment_type)
            )
            success = await experiment_manager.create_experiment(experiment)
            return {"success": success, "experiment_id": experiment.id}
    
    def run(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port)

# Global instance
api_integration = APIIntegrationLayer()

if __name__ == "__main__":
    api_integration.run()