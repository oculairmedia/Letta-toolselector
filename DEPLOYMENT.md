# LDTS Deployment Guide

This document provides comprehensive deployment instructions for the Letta Dynamic Tool Selection (LDTS) system.

## Architecture Overview

LDTS consists of 5 containerized services:

1. **Frontend Dashboard** (React TypeScript, port 3000)
   - Material-UI admin interface
   - Search, comparison, analytics, and configuration
   - Nginx-served production build

2. **API Server** (Python/Quart, port 8020) 
   - RESTful tool management API
   - Weaviate integration and tool search
   - Cost control and budget management

3. **MCP Server** (Node.js, port 3020)
   - HTTP-based MCP server
   - `find_tools` function for Letta agents
   - Tool attachment and pruning

4. **Sync Service** (Python background)
   - Synchronizes tools between Letta API and Weaviate
   - Runs every 5 minutes
   - Maintains tool cache integrity

5. **Weaviate** (Vector database, port 8080)
   - Stores tool embeddings for semantic search
   - Hybrid search (75% vector, 25% keyword)
   - OpenAI/Ollama embedding support

## Quick Start

### Production Deployment
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### Development with Hot Reload
```bash
# Start development environment
docker-compose -f compose.dev.yaml up -d

# The frontend will have hot reload enabled
# Changes to dashboard-frontend/ will auto-refresh
```

## Service URLs

| Service | Production | Development | Description |
|---------|------------|-------------|-------------|
| Frontend Dashboard | http://localhost:8406 | http://localhost:8406 | Main admin interface |
| API Server | http://localhost:8020 | http://localhost:8020 | RESTful API endpoints |
| MCP Server | http://localhost:3020 | http://localhost:3020 | MCP protocol server |
| Weaviate Console | http://localhost:8080 | http://localhost:8080 | Vector database admin |

## Environment Configuration

Key environment variables in `.env`:

### API Configuration
```bash
LETTA_API_URL=https://letta2.oculair.ca/v1
LETTA_PASSWORD=your_password
OPENAI_API_KEY=sk-...
```

### Embedding Provider
```bash
EMBEDDING_PROVIDER=ollama          # or 'openai'
OLLAMA_EMBEDDING_HOST=192.168.50.80
OLLAMA_EMBEDDING_MODEL=dengcao/Qwen3-Embedding-4B:Q4_K_M
USE_OLLAMA_EMBEDDINGS=true
```

### Tool Management
```bash
MAX_TOTAL_TOOLS=30                 # Maximum total tools per agent
MAX_MCP_TOOLS=20                   # Maximum MCP tools per agent  
MIN_MCP_TOOLS=7                    # Minimum MCP tools (prevents over-pruning)
DEFAULT_DROP_RATE=0.6              # Pruning aggressiveness (60%)
MANAGE_ONLY_MCP_TOOLS=true         # Only manage external MCP tools
```

### Frontend Build
```bash
VERSION=latest
BUILD_DATE=2025-09-09T23:22:00Z
COMMIT_SHA=local-development
```

## Health Monitoring

All services include health checks:

```bash
# Check individual service health
curl http://localhost:8406/health    # Frontend
curl http://localhost:8020/api/v1/health  # API Server  
curl http://localhost:3020/health    # MCP Server
curl http://localhost:8080/v1/.well-known/ready  # Weaviate
```

## CI/CD Pipeline

The system includes automated CI/CD with GitHub Actions:

### Triggered On
- Push to main/develop branches
- Pull requests to main
- Manual workflow dispatch
- Version tags (v*)

### Pipeline Stages
1. **Docker Build** - Builds all 4 service images (mcp-server, api-server, sync-service, frontend)
2. **Security Scanning** - Trivy vulnerability scanning for all images
3. **Integration Testing** - Docker Compose integration tests
4. **Deploy Staging** - Auto-deploy to staging on develop branch
5. **Deploy Production** - Auto-deploy to production on main branch/tags
6. **Cleanup** - Remove old container images (30+ days, keep 5 latest)

### Container Registry
- Registry: `ghcr.io/oculairmedia/`
- Images:
  - `letta-toolselector-mcp-server:latest`
  - `letta-toolselector-api-server:latest`
  - `letta-toolselector-sync-service:latest`
  - `letta-toolselector-frontend:latest`

## Frontend Features

The React TypeScript dashboard includes:

### Search Interface
- Semantic tool search with query expansion
- Real-time results with relevance scoring
- Filter by tool type, source, and metadata

### Results Comparison  
- A/B testing between different configurations
- Statistical significance testing
- Performance metrics visualization

### Manual Evaluation
- Tool relevance rating interface
- Context7 standards compliance
- Evaluation data export

### Analytics Dashboard
- Tool usage statistics
- Performance metrics over time
- Cost tracking and budget alerts

### Configuration Panel
- Embedding provider management
- Reranking algorithm settings
- Tool limits and pruning configuration

## Troubleshooting

### Common Issues

**Frontend not loading:**
```bash
# Check frontend logs
docker logs letta-toolselector-frontend

# Verify API connectivity
curl http://localhost:8020/api/v1/health
```

**Search returning no results:**
```bash
# Check Weaviate tool count
curl -X POST http://localhost:8080/v1/graphql \
  -H "Content-Type: application/json" \
  -d '{"query": "{ Aggregate { Tool { meta { count } } } }"}'

# Run manual sync
python lettaaugment-source/upload_tools_to_weaviate.py
```

**MCP tools not attaching:**
```bash
# Check MCP server logs
docker logs toolselector-mcp

# Verify Letta API connectivity
curl -H "Authorization: Bearer $LETTA_PASSWORD" \
     "$LETTA_API_URL/agents"
```

### Performance Optimization

**For high tool counts (>500 tools):**
- Increase `MAX_MCP_TOOLS` to 25-30
- Reduce `DEFAULT_DROP_RATE` to 0.4-0.5
- Consider dedicated Weaviate instance

**For cost optimization:**
- Use Ollama embedding provider
- Set conservative budget limits
- Monitor cost control dashboard

## Monitoring and Observability

### Metrics Collection
- Container health status
- API response times
- Tool search performance
- Cost tracking per operation

### Log Aggregation
All services log to stdout/stderr for container log collection:

```bash
# View all logs
docker-compose logs

# Follow specific service
docker-compose logs -f frontend

# Filter by error level
docker-compose logs | grep -i error
```

### Alerts
- Budget threshold alerts
- Tool sync failures
- API connectivity issues
- Container health failures

## Backup and Recovery

### Data Persistence
- Weaviate data: `weaviate_data` volume
- Tool cache: `tool_cache_volume` volume  
- Configuration: `.env` file

### Backup Process
```bash
# Backup volumes
docker run --rm -v weaviate_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/weaviate-backup.tar.gz -C /data .

# Backup configuration
cp .env backup/env-$(date +%Y%m%d).backup
```

### Recovery Process
```bash
# Restore volumes
docker run --rm -v weaviate_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/weaviate-backup.tar.gz -C /data

# Restart services
docker-compose down && docker-compose up -d
```

## Security Considerations

### Network Security
- All services run in isolated Docker network
- No direct external access to backend services
- Frontend proxies API requests through Nginx

### Container Security
- Alpine Linux base images for minimal attack surface
- Non-root user execution where possible
- Regular security scanning with Trivy

### API Security
- Bearer token authentication for Letta API
- CORS policies for frontend access
- Input validation and sanitization

### Secret Management
- Environment variables for sensitive configuration
- No secrets in container images
- Support for external secret management systems

## Scaling and Performance

### Horizontal Scaling
- Frontend: Multiple replicas behind load balancer
- API Server: Stateless, can scale horizontally
- MCP Server: Can run multiple instances
- Weaviate: Cluster mode for high availability

### Vertical Scaling
- Increase memory for embedding operations
- CPU scaling for search performance
- Storage scaling for large tool databases

### Performance Monitoring
- Response time tracking
- Resource utilization monitoring
- Search relevance metrics
- User experience analytics