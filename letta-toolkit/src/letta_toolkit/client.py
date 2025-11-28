"""HTTP client for Letta API with retry logic."""

from __future__ import annotations

import logging
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from letta_toolkit.config import LettaConfig, get_config

logger = logging.getLogger(__name__)


class LettaAPIError(Exception):
    """Exception raised for Letta API errors."""
    
    def __init__(self, message: str, status_code: int | None = None, response: dict | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class LettaClient:
    """HTTP client for Letta API with automatic retry and error handling.
    
    Example:
        >>> client = LettaClient()
        >>> tools = client.get(f"/agents/{agent_id}/tools")
    """
    
    def __init__(self, config: LettaConfig | None = None):
        """Initialize client with optional custom config.
        
        Args:
            config: Custom configuration. Uses global config if not provided.
        """
        self.config = config or get_config()
        self._session: requests.Session | None = None
    
    @property
    def session(self) -> requests.Session:
        """Get or create the requests session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(self.config.headers)
        return self._session
    
    def close(self) -> None:
        """Close the session."""
        if self._session is not None:
            self._session.close()
            self._session = None
    
    def __enter__(self) -> "LettaClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()
    
    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        path = path.lstrip("/")
        return f"{self.config.api_url}/{path}"
    
    @retry(
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> requests.Response:
        """Make HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            path: API path (e.g., /agents/{id}/tools)
            **kwargs: Additional arguments passed to requests
            
        Returns:
            Response object
            
        Raises:
            LettaAPIError: If request fails after retries
        """
        url = self._build_url(path)
        timeout = kwargs.pop("timeout", self.config.timeout)
        
        logger.debug(f"[LettaClient] {method} {url}")
        
        response = self.session.request(method, url, timeout=timeout, **kwargs)
        
        return response
    
    def get(self, path: str, params: dict | None = None, **kwargs: Any) -> Any:
        """Make GET request and return JSON response.
        
        Args:
            path: API path
            params: Query parameters
            **kwargs: Additional request arguments
            
        Returns:
            Parsed JSON response
            
        Raises:
            LettaAPIError: If request fails
        """
        response = self._request("GET", path, params=params, **kwargs)
        return self._handle_response(response)
    
    def post(self, path: str, json: dict | None = None, **kwargs: Any) -> Any:
        """Make POST request and return JSON response."""
        response = self._request("POST", path, json=json, **kwargs)
        return self._handle_response(response)
    
    def patch(self, path: str, json: dict | None = None, **kwargs: Any) -> Any:
        """Make PATCH request and return JSON response."""
        response = self._request("PATCH", path, json=json, **kwargs)
        return self._handle_response(response)
    
    def delete(self, path: str, **kwargs: Any) -> Any:
        """Make DELETE request and return JSON response."""
        response = self._request("DELETE", path, **kwargs)
        return self._handle_response(response)
    
    def _handle_response(self, response: requests.Response) -> Any:
        """Handle response, raising errors for non-2xx status codes.
        
        Args:
            response: Response object
            
        Returns:
            Parsed JSON response
            
        Raises:
            LettaAPIError: If status code indicates error
        """
        try:
            data = response.json() if response.content else {}
        except ValueError:
            data = {"raw_text": response.text[:500]}
        
        if response.status_code >= 400:
            error_msg = data.get("error") or data.get("message") or f"HTTP {response.status_code}"
            logger.error(f"[LettaClient] API error: {error_msg}")
            raise LettaAPIError(error_msg, status_code=response.status_code, response=data)
        
        return data
