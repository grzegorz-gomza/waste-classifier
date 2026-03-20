# WasteClassifier Test Suite

Simple test suite for the WasteClassifier demo application.

## Overview

This test suite focuses on core functionality testing rather than exhaustive coverage. It's designed for a demo app to ensure:

- Components can be imported and initialized
- Configuration management works
- Basic PyTorch/ML functionality is available
- Integration points are functional

## Test Files

### Core Tests
- **`test_core_components.py`** - Tests configuration manager, utility functions, and basic component structure
- **`test_data_ingestion_simple.py`** - Tests data ingestion component import and basic functionality
- **`test_training_simple.py`** - Tests training component import and PyTorch availability
- **`test_evaluation_simple.py`** - Tests evaluation component import and metrics availability

## Running Tests

### Quick Run
```bash
cd tests
python run_tests.py
```

### Individual Tests
```bash
# Run specific test file
python -m pytest test_core_components.py -v

# Run with verbose output
python -m pytest test_core_components.py -v --tb=short
```

### All Tests
```bash
# Run all tests in the directory
python -m pytest . -v

# Run with coverage (if installed)
python -m pytest . -v --cov=src
```

## Test Coverage

This is a **demo-level** test suite focusing on:

✅ **What's Tested:**
- Component imports and basic initialization
- Configuration file loading
- Utility functions (JSON/YAML operations)
- PyTorch availability and basic operations
- Directory creation and file operations
- Integration points between components

❌ **What's Not Tested:**
- Full training pipeline execution
- Model training accuracy
- Data ingestion from real URLs
- Complete evaluation workflows
- Edge cases and error handling

## Requirements

- pytest
- torch
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn
- PyYAML

## Notes

- Tests are designed to be fast and lightweight
- Some tests may be skipped if dependencies are missing
- Tests use temporary directories and mock data
- No external data downloads required
