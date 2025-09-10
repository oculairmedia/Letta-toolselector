#!/bin/sh

# Docker entrypoint script for React frontend
# Handles runtime environment variable injection

set -e

# Create env.js for runtime environment variables
cat <<EOF > /usr/share/nginx/html/env/env.js
window._env_ = {
  REACT_APP_API_BASE_URL: "${REACT_APP_API_BASE_URL:-http://localhost:8020}",
  REACT_APP_VERSION: "${REACT_APP_VERSION:-latest}",
  REACT_APP_BUILD_DATE: "${REACT_APP_BUILD_DATE:-unknown}",
  REACT_APP_COMMIT_SHA: "${REACT_APP_COMMIT_SHA:-unknown}"
};
EOF

echo "Environment variables injected:"
echo "  REACT_APP_API_BASE_URL: ${REACT_APP_API_BASE_URL:-http://localhost:8020}"
echo "  REACT_APP_VERSION: ${REACT_APP_VERSION:-latest}"
echo "  REACT_APP_BUILD_DATE: ${REACT_APP_BUILD_DATE:-unknown}"
echo "  REACT_APP_COMMIT_SHA: ${REACT_APP_COMMIT_SHA:-unknown}"

# Execute the original command
exec "$@"