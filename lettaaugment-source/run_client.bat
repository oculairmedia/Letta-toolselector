@echo off
:: Examples of how to use the API client

echo.
echo To check server health:
echo   python api_client.py health
echo.
echo To search and attach tools:
echo   python api_client.py attach-tools --query "web search tools"
echo.
echo All available options:
python api_client.py --help
echo.
echo Choose one of the examples above or input your query:
set /p QUERY="Enter a search query (or press Enter to exit): "

if "%QUERY%"=="" (
  echo Exiting...
  exit /b
)

echo.
echo Running search with query: %QUERY%
python api_client.py attach-tools --query "%QUERY%"
echo.
echo Done!