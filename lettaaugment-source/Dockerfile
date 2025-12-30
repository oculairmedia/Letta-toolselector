# Dockerfile

# ---- Builder Stage ----
FROM python:3.12-slim as builder

WORKDIR /app

# Install build dependencies needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Using virtual environment within the builder stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# ---- Final Stage ----
FROM python:3.12-slim

WORKDIR /app

# Create a non-root user and group
RUN groupadd --system app && \
    useradd --system --gid app --home /app app

# Copy installed dependencies from builder stage venv
COPY --chown=app:app --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=app:app *.py ./

# Create and set permissions for the cache directory before switching user
# This helps if the volume is new or doesn't have specific host permissions overriding it.
RUN mkdir -p /app/cache && chown -R app:app /app/cache
# Also create the runtime_cache directory with correct permissions
RUN mkdir -p /app/runtime_cache && \
   chown -R app:app /app/runtime_cache && \
   chmod -R g+w /app/runtime_cache

# Set default environment variables (will be overridden by docker-compose)
# These are just placeholders - actual values come from docker-compose environment section
ENV OPENAI_API_KEY=""
ENV LETTA_API_URL=""
ENV LETTA_PASSWORD=""
ENV SYNC_INTERVAL="300"

# Make sure scripts in venv are usable
ENV PATH="/opt/venv/bin:$PATH"

# Switch to non-root user
USER app

EXPOSE 3001

# Use hypercorn to run the ASGI app
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:3001"]