#!/bin/bash
# Automated deployment script for Letta Tool Selector API
# Usage: ./scripts/deploy.sh [--dry-run] [--skip-tests] [--force]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="${COMPOSE_FILE:-compose.yaml}"
API_URL="${API_URL:-http://localhost:8020}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"
SERVICES="${SERVICES:-api-server sync-service}"

# Parse arguments
DRY_RUN=false
SKIP_TESTS=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run     Show what would be done without executing"
            echo "  --skip-tests  Skip running tests before deployment"
            echo "  --force       Deploy even if tests fail or health checks fail"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $1"
    else
        log_info "Running: $1"
        eval "$1"
    fi
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "docker compose is not available"
        exit 1
    fi
    
    if [ ! -f "$COMPOSE_FILE" ]; then
        log_error "Compose file not found: $COMPOSE_FILE"
        exit 1
    fi
    
    log_info "Prerequisites OK"
}

run_tests() {
    if [ "$SKIP_TESTS" = true ]; then
        log_warn "Skipping tests (--skip-tests)"
        return 0
    fi
    
    log_info "Running unit tests..."
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} python3 -m pytest tests/unit/ -q"
    else
        if ! python3 -m pytest tests/unit/ -q --tb=line; then
            if [ "$FORCE" = true ]; then
                log_warn "Tests failed but continuing (--force)"
            else
                log_error "Tests failed. Use --force to deploy anyway."
                exit 1
            fi
        fi
    fi
}

pull_images() {
    log_info "Pulling latest images..."
    run_cmd "docker compose -f $COMPOSE_FILE pull $SERVICES"
}

backup_current() {
    log_info "Creating backup of current state..."
    BACKUP_TAG="backup-$(date +%Y%m%d-%H%M%S)"
    
    for service in $SERVICES; do
        CURRENT_IMAGE=$(docker compose -f $COMPOSE_FILE images $service -q 2>/dev/null || echo "")
        if [ -n "$CURRENT_IMAGE" ]; then
            run_cmd "docker tag $CURRENT_IMAGE letta-toolselector-$service:$BACKUP_TAG || true"
        fi
    done
}

deploy_services() {
    log_info "Deploying services: $SERVICES"
    run_cmd "docker compose -f $COMPOSE_FILE up -d $SERVICES"
}

wait_for_health() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Would wait for health check at $API_URL"
        return 0
    fi
    
    log_info "Waiting for API to be healthy (timeout: ${HEALTH_TIMEOUT}s)..."
    
    local elapsed=0
    local interval=5
    
    while [ $elapsed -lt $HEALTH_TIMEOUT ]; do
        # Try enrichment health endpoint
        if curl -sf "$API_URL/api/v1/enrichment/health" > /dev/null 2>&1; then
            log_info "API is healthy!"
            return 0
        fi
        
        # Try pruning status as fallback
        if curl -sf "$API_URL/api/v1/pruning/status" > /dev/null 2>&1; then
            log_info "API is healthy!"
            return 0
        fi
        
        sleep $interval
        elapsed=$((elapsed + interval))
        echo -n "."
    done
    
    echo ""
    if [ "$FORCE" = true ]; then
        log_warn "Health check timed out but continuing (--force)"
    else
        log_error "Health check failed after ${HEALTH_TIMEOUT}s"
        exit 1
    fi
}

verify_deployment() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} Would verify deployment"
        return 0
    fi
    
    log_info "Verifying deployment..."
    
    # Test search endpoint
    SEARCH_RESULT=$(curl -sf -X POST "$API_URL/api/v1/tools/search" \
        -H "Content-Type: application/json" \
        -d '{"query": "test", "limit": 1}' 2>/dev/null || echo "FAILED")
    
    if [ "$SEARCH_RESULT" = "FAILED" ]; then
        log_error "Search endpoint verification failed"
        if [ "$FORCE" != true ]; then
            exit 1
        fi
    else
        log_info "Search endpoint OK"
    fi
    
    # Check metrics endpoint
    if curl -sf "$API_URL/metrics" > /dev/null 2>&1; then
        log_info "Metrics endpoint OK"
    else
        log_warn "Metrics endpoint not available (prometheus_client may not be installed)"
    fi
}

show_status() {
    log_info "Deployment complete. Current status:"
    docker compose -f $COMPOSE_FILE ps $SERVICES
}

# Main execution
main() {
    log_info "Starting deployment..."
    [ "$DRY_RUN" = true ] && log_warn "DRY RUN MODE - no changes will be made"
    
    check_prerequisites
    run_tests
    pull_images
    backup_current
    deploy_services
    wait_for_health
    verify_deployment
    show_status
    
    log_info "Deployment successful!"
}

main
