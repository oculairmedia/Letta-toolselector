FROM node:18-slim

WORKDIR /app

LABEL maintainer="Tool Selector Team"
LABEL description="Tool Selector MCP Server for auto-attaching Letta tools"
LABEL version="1.0.0"

# Install Python for the find_tools.py script and curl for health checks
RUN apt-get update && apt-get install -y python3 python3-requests curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY package*.json ./
RUN npm ci --only=production

COPY . .

EXPOSE 3020

ARG PORT=3020
ARG NODE_ENV=production
ENV PORT=${PORT}
ENV NODE_ENV=${NODE_ENV}

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["node", "./src/simple-server.js"]