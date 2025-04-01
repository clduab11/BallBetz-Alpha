# ML Layer Test Suite

This directory contains comprehensive tests for the ML Layer components of the BallBetz-Alpha Triple-Layer Prediction Engine.

## Test Structure

The test suite is organized into the following categories:

1. **Unit Tests**: Test individual components in isolation
   - Feature Engineering
   - Model Selection
   - Training Pipeline
   - Prediction Module

2. **Integration Tests**: Test interactions between components
   - End-to-end workflows
   - Cross-component interactions

3. **Error Handling Tests**: Test edge cases and error conditions
   - Invalid inputs
   - Missing data
   - Configuration errors

4. **Performance Tests**: Test performance characteristics
   - Training performance with different dataset sizes
   - Prediction generation performance
   - Scaling behavior

## Running Tests

### Prerequisites

Ensure you have all required dependencies installed:

```bash
pip install -r requirements.txt
```

### Running All Tests

To run all ML Layer tests:

```bash
pytest tests/unit/ml_layer/ tests/integration/ml_layer/
```

### Running Specific Test Categories

To run only unit tests:

```bash
pytest tests/unit/ml_layer/
```

To run only integration tests:

```bash
pytest tests/integration/ml_layer/
```

To run performance tests (these may take longer):

```bash
pytest tests/performance/ml_layer/
```

### Running Individual Test Files

To run tests for a specific component:

```bash
# Feature Engineering tests
pytest tests/unit/ml_layer/test_feature_engineering.py
pytest tests/unit/ml_layer/test_feature_preprocessor.py
pytest tests/unit/ml_layer/test_feature_engineering_config.py

# Model Selection tests
pytest tests/unit/ml_layer/test_model_selection.py

# Training Pipeline tests
pytest tests/unit/ml_layer/test_training_pipeline.py

# Prediction tests
pytest tests/unit/ml_layer/test_prediction.py
```

## Test Environment

The tests use environment variables to configure the ML Layer components. These are set automatically during test execution, but you can override them if needed:

```bash
# Example: Set custom configuration for tests
export UFL_NUMERIC_IMPUTATION=median
export UFL_MAX_FEATURES=5
export UFL_TASK_TYPE=regression

# Then run tests
pytest tests/unit/ml_layer/
```

## Adding New Tests

When adding new tests, follow these guidelines:

1. **Use TDD Approach**: Write tests before implementing functionality
2. **Maintain Independence**: Tests should be independent and not rely on each other
3. **Mock External Dependencies**: Use mocks for external systems like scrapers
4. **Avoid Hardcoded Secrets**: Never include API keys or credentials in tests
5. **Keep Tests Fast**: Unit tests should execute quickly
6. **Test Edge Cases**: Include tests for error conditions and edge cases

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=ml_layer tests/unit/ml_layer/ tests/integration/ml_layer/
```

For a detailed HTML coverage report:

```bash
pytest --cov=ml_layer --cov-report=html tests/unit/ml_layer/ tests/integration/ml_layer/
```

The HTML report will be available in the `htmlcov` directory.