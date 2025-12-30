# Weaviate Local Connection Migration

This document outlines the changes made to migrate the API server and sync service from using Weaviate Cloud to a local Weaviate instance.

## Changes Made

1. Updated `sync_service.py` to use `connect_to_local()` instead of `connect_to_weaviate_cloud()`:
   - Changed connection method in the `sync_tools()` function
   - Changed connection method in the `main()` function
   - Removed authentication credentials since local Weaviate uses anonymous access
   - Set host to "weaviate" (the service name in docker-compose)
   - Set ports to 8080 (HTTP) and 50051 (gRPC)

2. Updated `weaviate_tool_search.py` to use "weaviate" as the host instead of "localhost":
   - Changed the host parameter in the `connect_to_local()` function

3. Updated `docker-compose.yml` to remove the `WEAVIATE_URL` and `WEAVIATE_API_KEY` environment variables:
   - Removed these variables from the `api-server` service
   - Removed these variables from the `sync-service` service

4. Updated `Dockerfile` to remove the `WEAVIATE_URL` and `WEAVIATE_API_KEY` environment variables:
   - Removed these variables from the ENV section

5. No changes needed for `api_server.py` as it was already using `init_weaviate_client()` from `weaviate_tool_search.py`.

## Rebuild Process

Since the Docker image is pulled from Docker Hub, we need to build and push the updated image before rebuilding the containers:

1. Build and push the Docker image:
   ```
   cd final tools
   .\build-and-push.bat
   ```
   This script will:
   - Build the Docker image with the updated code
   - Push the image to Docker Hub

2. After the image is pushed, run the rebuild script:
   ```
   cd final tools
   .\rebuild-containers.bat
   ```
   This script will:
   - Stop the api-server and sync-service containers
   - Rebuild the containers with the updated image
   - Start the containers again
   - Show the container status
   - Display the logs to verify proper initialization

3. If you need to force a fresh build without using cached layers, use:
   ```
   cd final tools
   .\rebuild-no-cache.bat
   ```

## Manual Rebuild Steps

If you need to manually rebuild the containers:

1. Stop the containers:
   ```
   docker-compose -f docker-compose.yml stop api-server sync-service
   ```

2. Rebuild the containers:
   ```
   docker-compose -f docker-compose.yml build api-server sync-service
   ```

3. Start the containers:
   ```
   docker-compose -f docker-compose.yml up -d api-server sync-service
   ```

4. Check the container status:
   ```
   docker-compose -f docker-compose.yml ps
   ```

5. View the logs:
   ```
   docker-compose -f docker-compose.yml logs -f api-server sync-service
   ```

## Verification

To verify that the services are working correctly:

1. Check that the API server can query the Weaviate instance:
   ```
   curl http://localhost:8020/api/health
   ```
   This should return a JSON response with status "OK" for the Weaviate connection.

2. Check the logs for any errors:
   ```
   docker-compose -f docker-compose.yml logs -f api-server sync-service
   ```

3. Test the API server's search functionality:
   ```
   curl -X POST -H "Content-Type: application/json" -d '{"query":"search"}' http://localhost:8020/api/v1/tools/search
   ```
   This should return a list of tools related to search.

4. Verify that the sync service is properly synchronizing with Weaviate by checking the logs for successful sync messages.