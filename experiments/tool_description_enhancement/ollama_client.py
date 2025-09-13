#!/usr/bin/env python3
"""
Ollama Client for Tool Description Enhancement

This module provides a client wrapper for the Ollama Gemma3:12b endpoint
to handle tool description enhancement requests with proper error handling
and optimization.
"""

import os
import json
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Response from Ollama API call"""
    content: str
    model: str
    processing_time: float
    token_count: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None


class OllamaClient:
    """
    Client for interacting with Ollama Gemma3:12b endpoint for tool enhancement.
    
    Supports batch processing, error handling, and performance tracking.
    """
    
    def __init__(self, 
                 base_url: str = "http://100.81.139.20:11434/v1",
                 model: str = "gemma3:12b",
                 timeout: int = 30,
                 max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.error_count = 0
        
        logger.info(f"Initialized OllamaClient: {base_url}, model: {model}")
    
    async def enhance_description(self, 
                                prompt: str,
                                system_prompt: Optional[str] = None) -> OllamaResponse:
        """
        Send a single enhancement request to Ollama.
        
        Args:
            prompt: The enhancement prompt
            system_prompt: Optional system prompt for context
            
        Returns:
            OllamaResponse with enhanced content or error details
        """
        start_time = time.time()
        self.total_requests += 1
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 1000
            }
        }
        
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        
                        if response.status == 200:
                            result = await response.json()
                            content = result['choices'][0]['message']['content']
                            processing_time = time.time() - start_time
                            self.total_processing_time += processing_time
                            
                            logger.info(f"Enhancement successful in {processing_time:.2f}s")
                            
                            return OllamaResponse(
                                content=content,
                                model=self.model,
                                processing_time=processing_time,
                                token_count=result.get('usage', {}).get('total_tokens'),
                                success=True
                            )
                        else:
                            error_text = await response.text()
                            logger.warning(f"Attempt {attempt + 1} failed: {response.status} - {error_text}")
                            
                            if attempt == self.max_retries - 1:
                                self.error_count += 1
                                return OllamaResponse(
                                    content="",
                                    model=self.model,
                                    processing_time=time.time() - start_time,
                                    success=False,
                                    error_message=f"HTTP {response.status}: {error_text}"
                                )
                            
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    self.error_count += 1
                    return OllamaResponse(
                        content="",
                        model=self.model,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message="Request timeout"
                    )
                await asyncio.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    self.error_count += 1
                    return OllamaResponse(
                        content="",
                        model=self.model,
                        processing_time=time.time() - start_time,
                        success=False,
                        error_message=str(e)
                    )
                await asyncio.sleep(2 ** attempt)
    
    async def batch_enhance(self, 
                          prompts: List[str],
                          system_prompt: Optional[str] = None,
                          batch_size: int = 5,
                          delay_between_batches: float = 1.0) -> List[OllamaResponse]:
        """
        Process multiple enhancement requests in batches.
        
        Args:
            prompts: List of enhancement prompts
            system_prompt: Optional system prompt for all requests
            batch_size: Number of concurrent requests per batch
            delay_between_batches: Delay between batches in seconds
            
        Returns:
            List of OllamaResponse objects
        """
        responses = []
        
        logger.info(f"Processing {len(prompts)} prompts in batches of {batch_size}")
        
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_start = time.time()
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            tasks = [
                self.enhance_description(prompt, system_prompt)
                for prompt in batch
            ]
            
            batch_responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            for j, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    logger.error(f"Batch item {i+j} failed with exception: {response}")
                    responses.append(OllamaResponse(
                        content="",
                        model=self.model,
                        processing_time=0.0,
                        success=False,
                        error_message=str(response)
                    ))
                else:
                    responses.append(response)
            
            batch_time = time.time() - batch_start
            logger.info(f"Batch completed in {batch_time:.2f}s")
            
            # Delay between batches (except for the last one)
            if i + batch_size < len(prompts):
                await asyncio.sleep(delay_between_batches)
        
        return responses
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client performance statistics"""
        avg_time = (self.total_processing_time / self.total_requests 
                   if self.total_requests > 0 else 0)
        success_rate = ((self.total_requests - self.error_count) / self.total_requests 
                       if self.total_requests > 0 else 0)
        
        return {
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "error_count": self.error_count,
            "success_rate": success_rate,
            "model": self.model,
            "base_url": self.base_url
        }
    
    async def test_connection(self) -> bool:
        """Test connection to Ollama endpoint"""
        try:
            response = await self.enhance_description("Test connection", "You are a test assistant.")
            return response.success
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False


# Example usage and testing
async def main():
    """Example usage of OllamaClient"""
    client = OllamaClient()
    
    # Test connection
    logger.info("Testing connection...")
    if await client.test_connection():
        logger.info("✅ Connection successful!")
    else:
        logger.error("❌ Connection failed!")
        return
    
    # Test single enhancement
    test_prompt = '''
    Tool: create_book
    Description: Creates a new book in Bookstack
    MCP Server: bookstack
    Parameters: name (required), description (required), tags (optional)
    
    Please create an enhanced description for this tool that includes:
    1. Detailed use cases and scenarios
    2. Keywords users might search for
    3. Context about when to use this tool
    4. Integration patterns
    '''
    
    logger.info("Testing single enhancement...")
    response = await client.enhance_description(test_prompt)
    
    if response.success:
        print(f"\n=== ENHANCED DESCRIPTION ===")
        print(response.content)
        print(f"\nProcessing time: {response.processing_time:.2f}s")
    else:
        print(f"Enhancement failed: {response.error_message}")
    
    # Print statistics
    stats = client.get_stats()
    print(f"\n=== CLIENT STATISTICS ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(main())