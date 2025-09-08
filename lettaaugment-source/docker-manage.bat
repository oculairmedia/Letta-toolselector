@echo off
echo Weaviate Tools API Server - Docker Management
echo =============================================
echo.

REM Check if Docker is installed and running
docker info >nul 2>&1
if %errorlevel% neq 0 (
  echo Docker is not running or not installed. Please start Docker Desktop.
  goto :EOF
)

echo Docker is running.
echo.

:MENU
echo Choose an option:
echo 1. Build and start the API server
echo 2. Stop the API server
echo 3. View logs
echo 4. Check container status
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
  echo.
  echo Building and starting the API server...
  docker-compose up -d --build
  echo.
  echo API server is running at http://localhost:8000
  echo To check server health: http://localhost:8000/api/health
  echo.
  goto MENU
)

if "%choice%"=="2" (
  echo.
  echo Stopping the API server...
  docker-compose down
  echo.
  goto MENU
)

if "%choice%"=="3" (
  echo.
  echo Showing logs (press Ctrl+C to exit logs)...
  docker-compose logs -f
  echo.
  goto MENU
)

if "%choice%"=="4" (
  echo.
  echo Checking container status...
  docker-compose ps
  echo.
  goto MENU
)

if "%choice%"=="5" (
  echo.
  echo Exiting...
  goto :EOF
)

echo.
echo Invalid choice. Please try again.
goto MENU