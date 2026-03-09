"""
Lightweight proxy to rewrite OpenAI model names to vLLM model names.
Weaviate text2vec-openai requires specific model names, but vLLM serves custom names.
"""
import os
import httpx
from fastapi import FastAPI, Request, Response

app = FastAPI()

VLLM_URL = os.getenv("VLLM_EMBEDDING_URL", "http://100.81.139.20:11450")
VLLM_MODEL = os.getenv("VLLM_MODEL", "qwen3-embedding")

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy(request: Request, path: str):
    url = f"{VLLM_URL}/{path}"
    
    body = await request.body()
    
    if body and request.method == "POST":
        import json
        try:
            data = json.loads(body)
            if "model" in data:
                data["model"] = VLLM_MODEL
            if "dimensions" in data:
                del data["dimensions"]
            body = json.dumps(data).encode()
        except json.JSONDecodeError:
            pass
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.request(
            method=request.method,
            url=url,
            headers={k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length"]},
            content=body,
        )
    
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers=dict(response.headers),
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8450)
