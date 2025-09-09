"""
LDTS-47: Data Ingestion Pipeline
Multi-format data ingestion with validation and preprocessing
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
import logging
import hashlib
import aiofiles
import pandas as pd
from urllib.parse import urlparse

class DataFormat(Enum):
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    PARQUET = "parquet"
    EXCEL = "excel"
    XML = "xml"
    YAML = "yaml"

class DataStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    ARCHIVED = "archived"

@dataclass
class DataIngestionJob:
    """Data ingestion job configuration"""
    id: str
    name: str
    description: str
    
    # Source configuration
    source_type: str  # file, url, database, api
    source_path: str
    format: DataFormat
    
    # Processing options
    validation_schema: Optional[str] = None
    preprocessing_steps: List[str] = field(default_factory=list)
    batch_size: int = 1000
    
    # Status tracking
    status: DataStatus = DataStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    total_records: int = 0
    processed_records: int = 0
    failed_records: int = 0
    error_message: Optional[str] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataRecord:
    """Individual data record"""
    id: str
    source_job_id: str
    raw_data: Dict[str, Any]
    processed_data: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

class DataIngestionPipeline:
    """Main data ingestion pipeline"""
    
    def __init__(self, storage_path: str = "data_ingestion"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.jobs_path = self.storage_path / "jobs"
        self.raw_data_path = self.storage_path / "raw"
        self.processed_data_path = self.storage_path / "processed"
        self.failed_data_path = self.storage_path / "failed"
        
        for path in [self.jobs_path, self.raw_data_path, self.processed_data_path, self.failed_data_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.active_jobs: Dict[str, asyncio.Task] = {}
    
    async def create_ingestion_job(self, job: DataIngestionJob) -> bool:
        """Create a new data ingestion job"""
        try:
            job_path = self.jobs_path / f"{job.id}.json"
            with open(job_path, 'w') as f:
                json.dump(asdict(job), f, indent=2, default=str)
            
            self.logger.info(f"Created ingestion job {job.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create ingestion job {job.id}: {e}")
            return False
    
    async def start_ingestion_job(self, job_id: str) -> bool:
        """Start processing an ingestion job"""
        try:
            job = await self.load_job(job_id)
            if not job:
                return False
            
            # Start processing task
            task = asyncio.create_task(self._process_job(job))
            self.active_jobs[job_id] = task
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start job {job_id}: {e}")
            return False
    
    async def _process_job(self, job: DataIngestionJob):
        """Process an ingestion job"""
        try:
            job.status = DataStatus.PROCESSING
            job.started_at = datetime.utcnow()
            await self._save_job(job)
            
            # Load and process data based on source type
            if job.source_type == "file":
                await self._process_file_source(job)
            elif job.source_type == "url":
                await self._process_url_source(job)
            elif job.source_type == "database":
                await self._process_database_source(job)
            else:
                raise ValueError(f"Unsupported source type: {job.source_type}")
            
            job.status = DataStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
        except Exception as e:
            job.status = DataStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            self.logger.error(f"Job {job.id} failed: {e}")
        
        finally:
            await self._save_job(job)
            if job.id in self.active_jobs:
                del self.active_jobs[job.id]
    
    async def _process_file_source(self, job: DataIngestionJob):
        """Process file-based data source"""
        file_path = Path(job.source_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Source file not found: {job.source_path}")
        
        # Process based on format
        if job.format == DataFormat.JSON:
            await self._process_json_file(job, file_path)
        elif job.format == DataFormat.JSONL:
            await self._process_jsonl_file(job, file_path)
        elif job.format == DataFormat.CSV:
            await self._process_csv_file(job, file_path)
        elif job.format == DataFormat.TSV:
            await self._process_tsv_file(job, file_path)
        else:
            raise ValueError(f"Unsupported file format: {job.format}")
    
    async def _process_json_file(self, job: DataIngestionJob, file_path: Path):
        """Process JSON file"""
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            data = json.loads(content)
            
            if isinstance(data, list):
                job.total_records = len(data)
                for i, record in enumerate(data):
                    await self._process_record(job, record, i)
            else:
                job.total_records = 1
                await self._process_record(job, data, 0)
    
    async def _process_jsonl_file(self, job: DataIngestionJob, file_path: Path):
        """Process JSONL file"""
        record_count = 0
        
        async with aiofiles.open(file_path, 'r') as f:
            async for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        await self._process_record(job, record, record_count)
                        record_count += 1
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Invalid JSON line {record_count}: {e}")
                        job.failed_records += 1
        
        job.total_records = record_count
    
    async def _process_csv_file(self, job: DataIngestionJob, file_path: Path):
        """Process CSV file"""
        df = pd.read_csv(file_path)
        job.total_records = len(df)
        
        for i, row in df.iterrows():
            record = row.to_dict()
            await self._process_record(job, record, i)
    
    async def _process_tsv_file(self, job: DataIngestionJob, file_path: Path):
        """Process TSV file"""
        df = pd.read_csv(file_path, sep='\t')
        job.total_records = len(df)
        
        for i, row in df.iterrows():
            record = row.to_dict()
            await self._process_record(job, record, i)
    
    async def _process_record(self, job: DataIngestionJob, raw_data: Dict[str, Any], index: int):
        """Process a single data record"""
        try:
            record = DataRecord(
                id=f"{job.id}_{index}",
                source_job_id=job.id,
                raw_data=raw_data
            )
            
            # Apply preprocessing steps
            processed_data = await self._preprocess_record(record, job.preprocessing_steps)
            record.processed_data = processed_data
            
            # Validate if schema provided
            if job.validation_schema:
                validation_errors = await self._validate_record(record, job.validation_schema)
                record.validation_errors = validation_errors
            
            # Save record
            if record.validation_errors:
                await self._save_failed_record(record)
                job.failed_records += 1
            else:
                await self._save_processed_record(record)
                job.processed_records += 1
            
        except Exception as e:
            self.logger.error(f"Failed to process record {index}: {e}")
            job.failed_records += 1
    
    async def _preprocess_record(self, record: DataRecord, steps: List[str]) -> Dict[str, Any]:
        """Apply preprocessing steps to a record"""
        processed = record.raw_data.copy()
        
        for step in steps:
            if step == "lowercase_text":
                for key, value in processed.items():
                    if isinstance(value, str):
                        processed[key] = value.lower()
            elif step == "strip_whitespace":
                for key, value in processed.items():
                    if isinstance(value, str):
                        processed[key] = value.strip()
            elif step == "remove_nulls":
                processed = {k: v for k, v in processed.items() if v is not None}
            # Add more preprocessing steps as needed
        
        return processed
    
    async def _validate_record(self, record: DataRecord, schema: str) -> List[str]:
        """Validate record against schema"""
        # Simplified validation - in practice, use jsonschema or similar
        errors = []
        
        if schema == "query_record":
            if "query" not in record.raw_data:
                errors.append("Missing required field: query")
            if "docs" not in record.raw_data:
                errors.append("Missing required field: docs")
        
        return errors
    
    async def _save_processed_record(self, record: DataRecord):
        """Save successfully processed record"""
        output_path = self.processed_data_path / f"{record.source_job_id}" / f"{record.id}.json"
        output_path.parent.mkdir(exist_ok=True)
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(asdict(record), default=str, indent=2))
    
    async def _save_failed_record(self, record: DataRecord):
        """Save failed record for debugging"""
        output_path = self.failed_data_path / f"{record.source_job_id}" / f"{record.id}.json"
        output_path.parent.mkdir(exist_ok=True)
        
        async with aiofiles.open(output_path, 'w') as f:
            await f.write(json.dumps(asdict(record), default=str, indent=2))
    
    async def load_job(self, job_id: str) -> Optional[DataIngestionJob]:
        """Load ingestion job"""
        try:
            job_path = self.jobs_path / f"{job_id}.json"
            if not job_path.exists():
                return None
            
            with open(job_path, 'r') as f:
                data = json.load(f)
            
            # Convert datetime strings and enums
            if data.get('created_at'):
                data['created_at'] = datetime.fromisoformat(data['created_at'])
            if data.get('started_at'):
                data['started_at'] = datetime.fromisoformat(data['started_at'])
            if data.get('completed_at'):
                data['completed_at'] = datetime.fromisoformat(data['completed_at'])
            
            data['format'] = DataFormat(data['format'])
            data['status'] = DataStatus(data['status'])
            
            return DataIngestionJob(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to load job {job_id}: {e}")
            return None
    
    async def _save_job(self, job: DataIngestionJob):
        """Save job state"""
        job_path = self.jobs_path / f"{job.id}.json"
        with open(job_path, 'w') as f:
            json.dump(asdict(job), f, indent=2, default=str)
    
    async def list_jobs(self, status_filter: Optional[DataStatus] = None) -> List[Dict[str, Any]]:
        """List ingestion jobs"""
        jobs = []
        
        for job_file in self.jobs_path.glob("*.json"):
            try:
                job = await self.load_job(job_file.stem)
                if job and (not status_filter or job.status == status_filter):
                    jobs.append({
                        'id': job.id,
                        'name': job.name,
                        'status': job.status.value,
                        'format': job.format.value,
                        'total_records': job.total_records,
                        'processed_records': job.processed_records,
                        'failed_records': job.failed_records,
                        'created_at': job.created_at.isoformat()
                    })
            except Exception as e:
                self.logger.error(f"Failed to load job info from {job_file}: {e}")
                continue
        
        return sorted(jobs, key=lambda x: x['created_at'], reverse=True)

# Global instance
data_ingestion = DataIngestionPipeline()

# Example usage
async def create_sample_ingestion_job():
    """Create sample ingestion job"""
    job = DataIngestionJob(
        id=str(uuid.uuid4()),
        name="Sample Query Dataset Ingestion",
        description="Ingest query dataset from JSON file",
        source_type="file",
        source_path="/path/to/queries.json",
        format=DataFormat.JSON,
        validation_schema="query_record",
        preprocessing_steps=["strip_whitespace", "lowercase_text"]
    )
    
    success = await data_ingestion.create_ingestion_job(job)
    if success:
        print(f"Created ingestion job: {job.id}")
        return job.id
    else:
        print("Failed to create ingestion job")
        return None

if __name__ == "__main__":
    asyncio.run(create_sample_ingestion_job())