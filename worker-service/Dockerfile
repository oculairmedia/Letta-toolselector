FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY worker-service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY worker-service/ ./
COPY tool_selector_client.py ./tool_selector_client.py
COPY letta_tool_utils.py ./letta_tool_utils.py

EXPOSE 3021

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3021/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "3021"]
