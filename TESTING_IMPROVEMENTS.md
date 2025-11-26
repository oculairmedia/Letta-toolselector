# Testing Infrastructure Improvements

**Branch**: `feature/improve-test-coverage`  
**Date**: 2025-01-26  
**Status**: In Progress

## Overview

This document tracks improvements made to the testing infrastructure for the Letta Tool Selector project.

## Completed Work ✅

### 1. Pytest Configuration (`pytest.ini`)
- Created comprehensive pytest configuration with:
  - Test discovery patterns
  - Coverage reporting (HTML, XML, terminal)
  - Custom markers for test categorization:
    - `@pytest.mark.unit` - Unit tests without external dependencies
    - `@pytest.mark.integration` - Integration tests requiring services
    - `@pytest.mark.e2e` - End-to-end tests
    - `@pytest.mark.slow` - Long-running tests
    - Service-specific markers (weaviate, api_server, letta, ollama)
  - Coverage exclusions and reporting settings
  - Asyncio support

### 2. Shared Test Fixtures (`tests/conftest.py`)
Created reusable fixtures for:
- **Environment Configuration**: Mock environment variables
- **Service Mocks**: Weaviate client, HTTP sessions, Letta API
- **Sample Data**: Tools, search results, configurations
- **Config Validation**: Valid and invalid configuration samples
- **Cost Control**: Sample cost entries and budget limits
- **Auto-Markers**: Automatic test categorization based on path/name

### 3. Unit Test Suite (`tests/unit/`)
Started unit test suite with:
- `test_config_validation.py` - 25+ tests for configuration validation system
  - ValidationResult class tests
  - ConfigValidationResponse class tests
  - SimpleConfigurationValidator tests
  - Embedding config validation
  - Weaviate config validation
  - Reranker config validation
  - Cost estimation tests
  - Performance impact assessment

## Test Categories

### Unit Tests (No External Dependencies) 🟢
- Configuration validation
- Data models and classes
- Utility functions
- Cost calculation logic

### Integration Tests (Requires Services) 🟡
- Weaviate database operations
- API endpoint functionality
- Tool management workflows
- Embedding generation

### End-to-End Tests (Full System) 🔴
- Complete tool selection workflow
- Reranking pipelines
- Multi-agent scenarios

## Next Steps 📋

### High Priority
1. **Fix Import Issues**: Update test imports to work with project structure
2. **Add More Unit Tests**:
   - Cost control manager (`cost_control_manager.py`)
   - BM25/vector overrides (`bm25_vector_overrides.py`)
   - Embedding providers (`embedding_providers.py`)
   - Worker service models (`worker-service/models.py`)

3. **Create GitHub Actions Workflow**:
   ```yaml
   name: Tests
   on: [push, pull_request]
   jobs:
     unit-tests:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.11'
         - name: Install dependencies
           run: |
             pip install -r lettaaugment-source/requirements.txt
             pip install pytest pytest-cov pytest-asyncio
         - name: Run unit tests
           run: pytest tests/unit/ -v --cov
   ```

### Medium Priority
4. **Coverage Reporting**:
   - Set up Codecov integration
   - Add coverage badges to README
   - Target: 60% coverage initially, 80% eventually

5. **Test Documentation**:
   - Update `tests/README.md` with new structure
   - Add testing guidelines
   - Document fixture usage
   - Create troubleshooting guide

6. **Integration Test Updates**:
   - Review existing integration tests
   - Add proper markers
   - Improve test isolation
   - Add setup/teardown fixtures

### Low Priority
7. **Performance Tests**:
   - Add benchmark tests
   - Create performance baselines
   - Track regression

8. **Test Data Management**:
   - Create `tests/data/` directory
   - Add sample tool definitions
   - Add sample search queries
   - Version control test data

## Test Execution

### Run All Tests
```bash
pytest tests/ -v
```

### Run Only Unit Tests
```bash
pytest tests/unit/ -v -m unit
```

### Run with Coverage
```bash
pytest tests/ --cov=lettaaugment-source --cov-report=html
```

### Run Specific Test Categories
```bash
# Integration tests only
pytest -m integration

# Exclude slow tests
pytest -m "not slow"

# Tests requiring Weaviate
pytest -m requires_weaviate
```

## Current Coverage Status

### Estimated Coverage
- **Unit-testable code**: ~15% (Just started)
- **Integration test coverage**: ~30% (Existing tests)
- **Overall coverage**: Unknown (Need to run coverage report)

### Target Coverage by Module
- **Configuration Validation**: 80% ✅ (New tests added)
- **Cost Control**: 0% 🔴 (To be added)
- **API Server**: 20% 🟡 (Needs improvement)
- **Worker Service**: 0% 🔴 (To be added)
- **Search/Reranking**: 40% 🟡 (Existing integration tests)

## Known Issues

1. **Import Path Issues**: Some tests need sys.path adjustments
2. **Mock Dependencies**: Need more comprehensive mocks for external services
3. **Async Test Support**: Ensure all async tests use proper fixtures
4. **Test Data**: Need centralized test data management

## Benefits of Improvements

1. **Faster Feedback**: Unit tests run in seconds vs minutes for integration tests
2. **Better CI/CD**: Can run unit tests on every commit
3. **Code Quality**: Higher coverage = fewer bugs
4. **Refactoring Confidence**: Tests catch regressions
5. **Documentation**: Tests serve as usage examples
6. **Debugging**: Isolated tests easier to debug

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Plugin](https://pytest-cov.readthedocs.io/)
- [pytest-asyncio Plugin](https://pytest-asyncio.readthedocs.io/)
- [Project Testing README](tests/README.md)
