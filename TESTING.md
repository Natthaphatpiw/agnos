# Testing Guide

## Overview

ระบบนี้มี test suite ครบถ้วนสำหรับทดสอบทั้ง API และ ML Model

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
pytest tests/test_model.py -v
```

## Test Structure

```
tests/
├── __init__.py
├── test_api.py       # API endpoint tests
└── test_model.py     # ML model tests
```

## Test Coverage

### API Tests (`test_api.py`)

#### 1. Health Endpoint Tests
- ✅ Health check returns 200
- ✅ Response includes model status and vocabulary size

#### 2. Symptoms Endpoint Tests
- ✅ Returns list of available symptoms
- ✅ Returns correct count

#### 3. Recommendation Endpoint Tests
- ✅ Valid recommendation request
- ✅ Multiple input symptoms
- ✅ Invalid gender validation
- ✅ Invalid age validation (negative, too high)
- ✅ Empty symptoms list validation
- ✅ Unknown symptom handling
- ✅ Default top_k behavior
- ✅ Custom top_k parameter

#### 4. Input Validation Tests
- ✅ Missing required fields
- ✅ Invalid JSON format

#### 5. Edge Cases
- ✅ Very young age (0 years)
- ✅ Very old age (120 years)
- ✅ Maximum top_k value

### Model Tests (`test_model.py`)

#### 1. Model Initialization Tests
- ✅ Model creation with valid parameters
- ✅ Trainable parameters exist
- ✅ Embedding dimensions correct

#### 2. Forward Pass Tests
- ✅ Output shape correctness
- ✅ Output values in valid range [0, 1]
- ✅ Training mode vs evaluation mode

#### 3. Model Components Tests
- ✅ Fusion weights normalization
- ✅ Dropout application
- ✅ Batch normalization behavior

#### 4. Robustness Tests
- ✅ Zero input handling
- ✅ Extreme values handling
- ✅ No NaN/Inf in outputs

#### 5. Save/Load Tests
- ✅ State dict save and load

## Example Test Output

```bash
$ pytest tests/ -v

tests/test_api.py::TestHealthEndpoint::test_health_check PASSED          [ 5%]
tests/test_api.py::TestSymptomsEndpoint::test_get_symptoms PASSED        [10%]
tests/test_api.py::TestRecommendEndpoint::test_recommend_valid_request PASSED [15%]
...
tests/test_model.py::TestModelInitialization::test_model_creation PASSED [80%]
tests/test_model.py::TestModelForward::test_forward_pass_shape PASSED    [85%]
...

======================== 30 passed in 12.45s ========================
```

## Coverage Report

After running tests with coverage:

```bash
pytest tests/ --cov=. --cov-report=html
```

Open `htmlcov/index.html` in browser to see detailed coverage report.

## Continuous Integration

Tests can be integrated with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          pytest tests/ -v --cov=. --cov-report=xml
```

## Writing New Tests

### API Test Template

```python
def test_new_endpoint(self):
    """Test description"""
    payload = {
        "field": "value"
    }
    response = client.post("/endpoint", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "expected_field" in data
```

### Model Test Template

```python
def test_model_feature(self):
    """Test description"""
    model = SymptomRecommender(num_symptoms=100, num_patients=50)
    # Test logic here
    assert expected_condition
```

## Best Practices

1. **Use descriptive test names** - เขียนชื่อ test ให้อธิบายสิ่งที่ทดสอบ
2. **Test one thing at a time** - แต่ละ test ควรทดสอบสิ่งเดียว
3. **Use fixtures** - ใช้ pytest fixtures สำหรับ setup ที่ซ้ำๆ
4. **Assert meaningful things** - ตรวจสอบสิ่งที่สำคัญ ไม่ใช่แค่ไม่มี error
5. **Test edge cases** - ทดสอบ edge cases และ error conditions
