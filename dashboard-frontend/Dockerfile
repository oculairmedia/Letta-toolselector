# Multi-stage build for React TypeScript frontend
# Stage 1: Build the React app
FROM node:18-alpine AS builder

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Set build-time environment variables
ARG REACT_APP_API_BASE_URL=http://localhost:8020
ARG REACT_APP_VERSION=latest
ARG REACT_APP_BUILD_DATE
ARG REACT_APP_COMMIT_SHA

ENV REACT_APP_API_BASE_URL=$REACT_APP_API_BASE_URL
ENV REACT_APP_VERSION=$REACT_APP_VERSION
ENV REACT_APP_BUILD_DATE=$REACT_APP_BUILD_DATE
ENV REACT_APP_COMMIT_SHA=$REACT_APP_COMMIT_SHA

# Build the app
RUN npm run build

# Stage 2: Serve with Nginx
FROM nginx:alpine AS production

# Install curl for health checks
RUN apk add --no-cache curl

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Copy built app from builder stage
COPY --from=builder /app/build /usr/share/nginx/html

# Copy environment injection script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Create directory for runtime environment injection
RUN mkdir -p /usr/share/nginx/html/env

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost/health || exit 1

# Use custom entrypoint for environment injection
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["nginx", "-g", "daemon off;"]