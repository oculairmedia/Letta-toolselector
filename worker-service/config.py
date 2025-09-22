from __future__ import annotations

import logging
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for the worker service."""

    log_level: str = os.getenv("WORKER_LOG_LEVEL", "INFO").upper()
    service_name: str = os.getenv("WORKER_SERVICE_NAME", "letta-tools-worker")
    port: int = int(os.getenv("WORKER_SERVICE_PORT", "3021"))
    pool_connections: int = int(os.getenv("WORKER_SESSION_POOL_CONNECTIONS", "10"))
    pool_maxsize: int = int(os.getenv("WORKER_SESSION_POOL_MAXSIZE", "25"))
    pool_max_retries: int = int(os.getenv("WORKER_SESSION_MAX_RETRIES", "3"))


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s [worker-service] %(name)s: %(message)s",
    )


settings = Settings()
configure_logging(settings)
