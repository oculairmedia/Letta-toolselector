@echo off
echo Building and pushing Docker image with multi-platform support...

echo Checking buildx setup...
docker buildx inspect multiplatform >nul 2>&1
if %errorlevel% neq 0 (
    echo Creating new buildx builder for multi-platform support...
    docker buildx create --name multiplatform --driver docker-container --bootstrap
    docker buildx use multiplatform
) else (
    echo Using existing multiplatform builder...
    docker buildx use multiplatform
)

echo Building Docker image for multiple platforms...
docker buildx build --platform linux/amd64,linux/arm64 -t oculair/lettaaugment:latest . --push

echo Done! Image pushed to Docker Hub with multi-platform support.