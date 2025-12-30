@echo off
echo Building and pushing Docker image with multi-platform support...

echo Using desktop-linux builder for multi-platform support...
docker buildx use desktop-linux

echo Building Docker image for multiple platforms...
docker buildx build --platform linux/amd64,linux/arm64 -t oculair/lettaaugment:latest . --push

echo Done! Image pushed to Docker Hub with multi-platform support.