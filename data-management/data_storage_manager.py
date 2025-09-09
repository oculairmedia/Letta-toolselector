"""
LDTS-48: Data Storage Manager
Unified storage layer with versioning and backup capabilities
"""

import asyncio
import json
import uuid
import hashlib
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import sqlite3
import aiofiles
import aiosqlite
from contextlib import asynccontextmanager

class StorageType(Enum):
    FILE_SYSTEM = "file_system"
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql" 
    S3 = "s3"
    AZURE_BLOB = "azure_blob"

class DataType(Enum):
    QUERIES = "queries"
    DOCUMENTS = "documents"
    EMBEDDINGS = "embeddings"
    RESULTS = "results"
    CONFIGURATIONS = "configurations"
    METRICS = "metrics"

@dataclass
class StorageEntry:
    """Storage entry metadata"""
    id: str
    data_type: DataType
    key: str  # Unique identifier for the data
    version: int
    
    # Content metadata
    content_hash: str
    size_bytes: int
    format: str
    
    # Timestamps
    created_at: datetime
    updated_at: datetime
    accessed_at: Optional[datetime] = None
    
    # Relationships
    parent_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Storage details
    storage_path: str = ""
    backup_paths: List[str] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BackupConfig:
    """Backup configuration"""
    enabled: bool = True
    retention_days: int = 30
    max_versions: int = 10
    backup_locations: List[str] = field(default_factory=list)
    compression: bool = True
    encryption: bool = False

class DataStorageManager:
    """Unified data storage manager"""
    
    def __init__(self, base_path: str = "data_storage", backup_config: BackupConfig = None):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Storage paths
        self.data_path = self.base_path / "data"
        self.backup_path = self.base_path / "backups"
        self.metadata_path = self.base_path / "metadata"
        self.index_path = self.base_path / "storage.db"
        
        for path in [self.data_path, self.backup_path, self.metadata_path]:
            path.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.backup_config = backup_config or BackupConfig()
        
        # Initialize database
        asyncio.create_task(self._init_database())
    
    async def _init_database(self):
        """Initialize SQLite database for metadata"""
        async with aiosqlite.connect(self.index_path) as db:
            await db.execute('''
                CREATE TABLE IF NOT EXISTS storage_entries (
                    id TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    key TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    content_hash TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    format TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    accessed_at TEXT,
                    parent_id TEXT,
                    tags TEXT,
                    storage_path TEXT NOT NULL,
                    backup_paths TEXT,
                    metadata TEXT,
                    UNIQUE(key, version)
                )
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_key ON storage_entries(key);
            ''')
            
            await db.execute('''
                CREATE INDEX IF NOT EXISTS idx_data_type ON storage_entries(data_type);
            ''')
            
            await db.commit()
    
    async def store_data(
        self,
        key: str,
        data: Any,
        data_type: DataType,
        format: str = "json",
        tags: List[str] = None,
        parent_id: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store data with versioning"""
        
        try:
            # Serialize data
            if format == "json":
                content = json.dumps(data, indent=2, default=str)
            elif format == "text":
                content = str(data)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            
            # Check if data already exists with same hash
            existing_entry = await self._get_entry_by_hash(key, content_hash)
            if existing_entry:
                self.logger.info(f"Data with key {key} already exists with same content")
                return existing_entry.id
            
            # Get next version
            latest_version = await self._get_latest_version(key)
            new_version = latest_version + 1
            
            # Create storage entry
            entry_id = str(uuid.uuid4())
            storage_path = self._get_storage_path(data_type, key, new_version)
            
            entry = StorageEntry(
                id=entry_id,
                data_type=data_type,
                key=key,
                version=new_version,
                content_hash=content_hash,
                size_bytes=len(content.encode()),
                format=format,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                parent_id=parent_id,
                tags=tags or [],
                storage_path=str(storage_path),
                metadata=metadata or {}
            )
            
            # Store content
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(storage_path, 'w') as f:
                await f.write(content)
            
            # Store metadata
            await self._store_entry_metadata(entry)
            
            # Create backup if enabled
            if self.backup_config.enabled:
                backup_path = await self._create_backup(entry, content)
                entry.backup_paths.append(str(backup_path))
                await self._store_entry_metadata(entry)  # Update with backup path
            
            self.logger.info(f"Stored data with key {key}, version {new_version}")
            return entry_id
            
        except Exception as e:
            self.logger.error(f"Failed to store data with key {key}: {e}")
            raise
    
    async def retrieve_data(
        self,
        key: str,
        version: Optional[int] = None,
        format: Optional[str] = None
    ) -> Optional[Any]:
        """Retrieve data by key and version"""
        
        try:
            entry = await self._get_entry(key, version)
            if not entry:
                return None
            
            # Update access time
            entry.accessed_at = datetime.utcnow()
            await self._store_entry_metadata(entry)
            
            # Read content
            storage_path = Path(entry.storage_path)
            if not storage_path.exists():
                # Try to restore from backup
                restored = await self._restore_from_backup(entry)
                if not restored:
                    self.logger.error(f"Storage file not found and backup restoration failed: {storage_path}")
                    return None
            
            async with aiofiles.open(storage_path, 'r') as f:
                content = await f.read()
            
            # Parse based on format
            if entry.format == "json":
                return json.loads(content)
            elif entry.format == "text":
                return content
            else:
                return content
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve data with key {key}: {e}")
            return None
    
    async def list_entries(
        self,
        data_type: Optional[DataType] = None,
        tags: Optional[List[str]] = None,
        key_prefix: Optional[str] = None,
        limit: int = 100
    ) -> List[StorageEntry]:
        """List storage entries with filtering"""
        
        query = "SELECT * FROM storage_entries WHERE 1=1"
        params = []
        
        if data_type:
            query += " AND data_type = ?"
            params.append(data_type.value)
        
        if key_prefix:
            query += " AND key LIKE ?"
            params.append(f"{key_prefix}%")
        
        if tags:
            # Simple tag filtering - in practice, might want a separate tags table
            for tag in tags:
                query += " AND tags LIKE ?"
                params.append(f"%{tag}%")
        
        query += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        
        entries = []
        async with aiosqlite.connect(self.index_path) as db:
            async with db.execute(query, params) as cursor:
                async for row in cursor:
                    entry = await self._row_to_entry(row)
                    entries.append(entry)
        
        return entries
    
    async def delete_data(self, key: str, version: Optional[int] = None) -> bool:
        """Delete data by key and version"""
        
        try:
            if version:
                # Delete specific version
                entry = await self._get_entry(key, version)
                if entry:
                    await self._delete_entry(entry)
                    return True
            else:
                # Delete all versions
                entries = await self._get_all_versions(key)
                for entry in entries:
                    await self._delete_entry(entry)
                return len(entries) > 0
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete data with key {key}: {e}")
            return False
    
    async def _get_entry(self, key: str, version: Optional[int] = None) -> Optional[StorageEntry]:
        """Get storage entry by key and version"""
        
        if version:
            query = "SELECT * FROM storage_entries WHERE key = ? AND version = ?"
            params = [key, version]
        else:
            query = "SELECT * FROM storage_entries WHERE key = ? ORDER BY version DESC LIMIT 1"
            params = [key]
        
        async with aiosqlite.connect(self.index_path) as db:
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                if row:
                    return await self._row_to_entry(row)
        
        return None
    
    async def _get_entry_by_hash(self, key: str, content_hash: str) -> Optional[StorageEntry]:
        """Get entry by key and content hash"""
        
        query = "SELECT * FROM storage_entries WHERE key = ? AND content_hash = ? ORDER BY version DESC LIMIT 1"
        params = [key, content_hash]
        
        async with aiosqlite.connect(self.index_path) as db:
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                if row:
                    return await self._row_to_entry(row)
        
        return None
    
    async def _get_latest_version(self, key: str) -> int:
        """Get latest version number for key"""
        
        query = "SELECT MAX(version) FROM storage_entries WHERE key = ?"
        
        async with aiosqlite.connect(self.index_path) as db:
            async with db.execute(query, [key]) as cursor:
                result = await cursor.fetchone()
                return result[0] if result[0] is not None else 0
    
    async def _get_all_versions(self, key: str) -> List[StorageEntry]:
        """Get all versions for a key"""
        
        query = "SELECT * FROM storage_entries WHERE key = ? ORDER BY version"
        entries = []
        
        async with aiosqlite.connect(self.index_path) as db:
            async with db.execute(query, [key]) as cursor:
                async for row in cursor:
                    entry = await self._row_to_entry(row)
                    entries.append(entry)
        
        return entries
    
    async def _store_entry_metadata(self, entry: StorageEntry):
        """Store entry metadata in database"""
        
        query = '''
            INSERT OR REPLACE INTO storage_entries 
            (id, data_type, key, version, content_hash, size_bytes, format, 
             created_at, updated_at, accessed_at, parent_id, tags, 
             storage_path, backup_paths, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = [
            entry.id,
            entry.data_type.value,
            entry.key,
            entry.version,
            entry.content_hash,
            entry.size_bytes,
            entry.format,
            entry.created_at.isoformat(),
            entry.updated_at.isoformat(),
            entry.accessed_at.isoformat() if entry.accessed_at else None,
            entry.parent_id,
            json.dumps(entry.tags),
            entry.storage_path,
            json.dumps(entry.backup_paths),
            json.dumps(entry.metadata, default=str)
        ]
        
        async with aiosqlite.connect(self.index_path) as db:
            await db.execute(query, params)
            await db.commit()
    
    async def _row_to_entry(self, row) -> StorageEntry:
        """Convert database row to StorageEntry"""
        
        return StorageEntry(
            id=row[0],
            data_type=DataType(row[1]),
            key=row[2],
            version=row[3],
            content_hash=row[4],
            size_bytes=row[5],
            format=row[6],
            created_at=datetime.fromisoformat(row[7]),
            updated_at=datetime.fromisoformat(row[8]),
            accessed_at=datetime.fromisoformat(row[9]) if row[9] else None,
            parent_id=row[10],
            tags=json.loads(row[11]) if row[11] else [],
            storage_path=row[12],
            backup_paths=json.loads(row[13]) if row[13] else [],
            metadata=json.loads(row[14]) if row[14] else {}
        )
    
    def _get_storage_path(self, data_type: DataType, key: str, version: int) -> Path:
        """Generate storage path for entry"""
        
        # Create directory structure: data_type/key_prefix/key_version.format
        key_prefix = key[:2] if len(key) >= 2 else "default"
        filename = f"{key}_v{version}.json"
        
        return self.data_path / data_type.value / key_prefix / filename
    
    async def _create_backup(self, entry: StorageEntry, content: str) -> Path:
        """Create backup of entry"""
        
        backup_dir = self.backup_path / entry.data_type.value / datetime.utcnow().strftime("%Y-%m-%d")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_filename = f"{entry.key}_v{entry.version}_{entry.id[:8]}.json"
        backup_path = backup_dir / backup_filename
        
        async with aiofiles.open(backup_path, 'w') as f:
            await f.write(content)
        
        return backup_path
    
    async def _restore_from_backup(self, entry: StorageEntry) -> bool:
        """Restore entry from backup"""
        
        for backup_path_str in entry.backup_paths:
            backup_path = Path(backup_path_str)
            if backup_path.exists():
                try:
                    # Copy backup to original location
                    storage_path = Path(entry.storage_path)
                    storage_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, storage_path)
                    
                    self.logger.info(f"Restored {entry.key} from backup: {backup_path}")
                    return True
                except Exception as e:
                    self.logger.error(f"Failed to restore from backup {backup_path}: {e}")
                    continue
        
        return False
    
    async def _delete_entry(self, entry: StorageEntry):
        """Delete storage entry and its files"""
        
        # Delete main storage file
        storage_path = Path(entry.storage_path)
        if storage_path.exists():
            storage_path.unlink()
        
        # Delete backup files (based on retention policy)
        for backup_path_str in entry.backup_paths:
            backup_path = Path(backup_path_str)
            if backup_path.exists():
                backup_path.unlink()
        
        # Remove from database
        async with aiosqlite.connect(self.index_path) as db:
            await db.execute("DELETE FROM storage_entries WHERE id = ?", [entry.id])
            await db.commit()
    
    async def cleanup_old_backups(self):
        """Clean up old backup files based on retention policy"""
        
        if not self.backup_config.enabled:
            return
        
        cutoff_date = datetime.utcnow() - timedelta(days=self.backup_config.retention_days)
        
        # Find entries with old backups
        query = "SELECT * FROM storage_entries WHERE created_at < ?"
        
        async with aiosqlite.connect(self.index_path) as db:
            async with db.execute(query, [cutoff_date.isoformat()]) as cursor:
                async for row in cursor:
                    entry = await self._row_to_entry(row)
                    
                    # Keep only the most recent backups
                    if len(entry.backup_paths) > self.backup_config.max_versions:
                        # Sort backup paths by modification time and keep only recent ones
                        backup_paths = []
                        for backup_path_str in entry.backup_paths:
                            backup_path = Path(backup_path_str)
                            if backup_path.exists():
                                backup_paths.append((backup_path.stat().st_mtime, backup_path_str))
                        
                        # Sort by modification time (newest first)
                        backup_paths.sort(reverse=True)
                        
                        # Keep only the most recent backups
                        kept_backups = backup_paths[:self.backup_config.max_versions]
                        removed_backups = backup_paths[self.backup_config.max_versions:]
                        
                        # Delete old backups
                        for _, backup_path_str in removed_backups:
                            backup_path = Path(backup_path_str)
                            if backup_path.exists():
                                backup_path.unlink()
                        
                        # Update entry with remaining backup paths
                        entry.backup_paths = [path for _, path in kept_backups]
                        await self._store_entry_metadata(entry)
        
        self.logger.info("Completed backup cleanup")
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        
        query = '''
            SELECT 
                data_type,
                COUNT(*) as count,
                SUM(size_bytes) as total_size,
                AVG(size_bytes) as avg_size
            FROM storage_entries 
            GROUP BY data_type
        '''
        
        stats = {
            'by_type': {},
            'total_entries': 0,
            'total_size_bytes': 0
        }
        
        async with aiosqlite.connect(self.index_path) as db:
            async with db.execute(query) as cursor:
                async for row in cursor:
                    data_type = row[0]
                    count = row[1]
                    total_size = row[2]
                    avg_size = row[3]
                    
                    stats['by_type'][data_type] = {
                        'count': count,
                        'total_size_bytes': total_size,
                        'avg_size_bytes': avg_size
                    }
                    
                    stats['total_entries'] += count
                    stats['total_size_bytes'] += total_size
        
        return stats

# Global instance
storage_manager = DataStorageManager()

# Example usage
async def test_storage_manager():
    """Test storage manager functionality"""
    
    # Store some test data
    test_data = {
        "query": "What is machine learning?",
        "documents": ["doc1", "doc2", "doc3"],
        "relevance_scores": [0.9, 0.7, 0.5]
    }
    
    entry_id = await storage_manager.store_data(
        key="test_query_1",
        data=test_data,
        data_type=DataType.QUERIES,
        tags=["test", "machine_learning"]
    )
    
    print(f"Stored data with entry ID: {entry_id}")
    
    # Retrieve the data
    retrieved_data = await storage_manager.retrieve_data("test_query_1")
    print(f"Retrieved data: {retrieved_data}")
    
    # Get storage stats
    stats = await storage_manager.get_storage_stats()
    print(f"Storage stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_storage_manager())