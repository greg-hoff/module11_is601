Installed requirements
pulled app files from greg's github for module 10
renamed container for module 11
copied app calculation files for models and schemas from instructor github for module 11

set up test files for testing the new schema and models

revise conftest and models so that the foreign keys aligned with the addition of the new calculation model and schema.

added relationship definition in user.py 

added and refined new tests for calculations and calculation schema

## Running Tests Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Run all tests with coverage
pytest tests/ --cov=app --cov-report=term-missing -v

# Run specific test categories
pytest tests/unit/           # Unit tests only
pytest tests/integration/    # Integration tests only
pytest tests/e2e/           # End-to-end tests only

# Run with coverage for specific modules
pytest tests/integration/test_user_auth.py --cov=app.models.user --cov-report=term-missing -v
```

https://hub.docker.com/repository/docker/greghoff/module11_is601/general