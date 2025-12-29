"""
Anthropic Client for Semantic Enrichment

Uses the Anthropic OpenAI-compatible proxy for Claude Sonnet API calls.
"""

import os
import json
import logging
import time
from typing import Dict, Any, Optional, List
import httpx

logger = logging.getLogger(__name__)


class AnthropicClient:
    """
    Client for Claude Sonnet via Anthropic OpenAI-compatible proxy.
    
    Uses the proxy at ANTHROPIC_PROXY_URL with OpenAI-compatible API format.
    """
    
    # Default model for enrichment
    DEFAULT_MODEL = "claude-sonnet-4-20250514"
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3
    ):
        """
        Initialize the Anthropic client.
        
        Args:
            base_url: Anthropic proxy URL (default: ANTHROPIC_PROXY_URL env var)
            api_key: API key (default: ANTHROPIC_API_KEY env var)
            model: Model to use (default: claude-sonnet-4-20250514)
            timeout: Request timeout in seconds
            max_retries: Max retry attempts on failure
        """
        self.base_url = base_url or os.getenv(
            "ANTHROPIC_PROXY_URL", 
            "http://192.168.50.90:4010/v1"
        )
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = model or self.DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Create persistent HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
        
        logger.info(
            f"AnthropicClient initialized: base_url={self.base_url}, "
            f"model={self.model}, timeout={timeout}s"
        )
    
    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, '_client'):
            self._client.close()
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            
        Returns:
            Dict with 'content', 'tokens_used', 'model' keys
        """
        # Build request
        request_messages = []
        if system_prompt:
            request_messages.append({"role": "system", "content": system_prompt})
        request_messages.extend(messages)
        
        payload = {
            "model": self.model,
            "messages": request_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format:
            payload["response_format"] = response_format
        
        # Retry loop
        last_error = None
        for attempt in range(self.max_retries):
            try:
                start_time = time.time()
                response = self._client.post("/chat/completions", json=payload)
                duration_ms = (time.time() - start_time) * 1000
                
                response.raise_for_status()
                data = response.json()
                
                # Extract content
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                usage = data.get("usage", {})
                tokens_used = usage.get("total_tokens", 0)
                
                logger.debug(
                    f"Chat completion success: {tokens_used} tokens, "
                    f"{duration_ms:.0f}ms"
                )
                
                return {
                    "content": content,
                    "tokens_used": tokens_used,
                    "model": self.model,
                    "duration_ms": duration_ms
                }
                
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(
                    f"HTTP error on attempt {attempt + 1}/{self.max_retries}: "
                    f"{e.response.status_code} - {e.response.text[:200]}"
                )
                if e.response.status_code >= 500:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise  # Don't retry client errors
                    
            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}"
                )
                time.sleep(2 ** attempt)
        
        raise last_error or Exception("Max retries exceeded")
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Generate a JSON response from a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            
        Returns:
            Parsed JSON response
        """
        messages = [{"role": "user", "content": prompt}]
        
        # Add JSON instruction to system prompt
        json_system = (system_prompt or "") + (
            "\n\nIMPORTANT: Respond ONLY with valid JSON. "
            "Do not include any text before or after the JSON object."
        )
        
        result = self.chat_completion(
            messages=messages,
            system_prompt=json_system,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        
        content = result["content"].strip()
        
        # Try to parse JSON
        try:
            # Handle potential markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1]) if lines[-1] == "```" else "\n".join(lines[1:])
            
            parsed = json.loads(content)
            parsed["_meta"] = {
                "tokens_used": result["tokens_used"],
                "model": result["model"],
                "duration_ms": result["duration_ms"]
            }
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nContent: {content[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the API is accessible.
        
        Returns:
            Dict with 'healthy', 'model', 'latency_ms' keys
        """
        try:
            start = time.time()
            result = self.chat_completion(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=10
            )
            latency = (time.time() - start) * 1000
            
            return {
                "healthy": True,
                "model": self.model,
                "latency_ms": latency,
                "base_url": self.base_url
            }
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "model": self.model,
                "base_url": self.base_url
            }


# Singleton instance
_client_instance: Optional[AnthropicClient] = None


def get_anthropic_client() -> AnthropicClient:
    """Get or create the singleton AnthropicClient instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = AnthropicClient()
    return _client_instance
