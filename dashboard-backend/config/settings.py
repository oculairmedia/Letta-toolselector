from typing import Optional, List
import os

class Settings:
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "LDTS Reranker Testing Dashboard Backend"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Backend API for LDTS Reranker Testing Dashboard"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8030
    WORKERS: int = 1
    RELOAD: bool = False
    
    # Safety Configuration
    READ_ONLY_MODE: bool = True  # Default to read-only for safety
    ENABLE_DANGEROUS_OPERATIONS: bool = False
    
    # LDTS Integration
    LDTS_API_URL: str = os.getenv("LDTS_API_URL", "http://localhost:8020")
    LDTS_MCP_URL: str = os.getenv("LDTS_MCP_URL", "http://localhost:3020")
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    
    # Letta API Configuration  
    LETTA_API_URL: str = os.getenv("LETTA_API_URL", "https://letta.oculair.ca/v1")
    LETTA_PASSWORD: str = os.getenv("LETTA_PASSWORD", "")
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    
    # Rate Limiting
    ENABLE_RATE_LIMITING: bool = True
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Search Configuration
    DEFAULT_SEARCH_LIMIT: int = 10
    MAX_SEARCH_LIMIT: int = 50
    ENABLE_QUERY_EXPANSION: bool = True
    
    # Reranking Configuration
    ENABLE_RERANKING: bool = True
    RERANKER_URL: Optional[str] = "http://localhost:11434"  # Ollama default
    RERANKER_MODEL: str = "bge-reranker-large"
    RERANKER_TIMEOUT: int = 30
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-here")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001", 
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:8080",
        "http://192.168.50.90:8406"
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    ENABLE_ACCESS_LOGS: bool = True
    
    # Database/Cache (if needed later)
    REDIS_URL: Optional[str] = None
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()