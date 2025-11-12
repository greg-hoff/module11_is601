pulled app files from instructor github for module 10
copied new conftest for module 10
pulled test_user for module 10 
    took time to set up dockerfile, server startup, new dependencies and requirements
set up test files for testing the schema, database, authentication, and dependencies

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

https://hub.docker.com/repository/docker/greghoff/module10_is601/general