# CI/CD Pipeline Status

This document describes the CI/CD pipelines configured for the Spark Resource Optimizer project.

## Workflows

### 1. Smoke Tests (`smoke-test.yml`)
**Purpose**: Quick validation that core functionality works across different platforms and Python versions.

**Triggers**:
- Push to `main` or `feature/*` branches
- Pull requests to `main`

**Jobs**:
- **smoke-test**: Tests basic functionality across multiple OS with latest Python
  - Runs the automated smoke test script
  - Tests CLI commands
  - Validates Python imports
  - Matrix: Ubuntu/macOS/Windows × Python 3.13

- **functional-test**: Tests actual functionality
  - Database operations (create, save, query)
  - Recommender engine
  - CLI database commands

- **api-server-test**: Tests the API server
  - Starts server in background
  - Tests health endpoint
  - Validates server starts correctly

**Status**: ✅ Should pass (tests what actually works)

### 2. Unit Tests (`test.yml`)
**Purpose**: Run comprehensive unit and integration tests.

**Triggers**:
- Push to `main`
- Pull requests to `main`

**Jobs**:
- **test**: Pytest suite across OS/Python matrix
- **test-integration**: Integration tests with PostgreSQL
- **test-docker**: Docker build verification

**Status**: ⚠️ Will show warnings (most tests are TODO stubs)

### 3. PR Checks (`pr-checks.yml`)
**Purpose**: Enforce PR quality standards.

**Triggers**: Pull requests

**Jobs**:
- PR title format validation (conventional commits)
- PR size labeling
- Label validation
- Merge conflict detection
- Branch name validation

**Status**: ✅ Active

### 4. Linting (`lint.yml`)
**Purpose**: Code quality checks.

**Jobs**: Likely includes flake8, black, mypy checks

### 5. Docker (`docker.yml`)
**Purpose**: Build and publish Docker images.

### 6. CodeQL (`codeql.yml`)
**Purpose**: Security scanning.

### 7. Release (`release.yml`)
**Purpose**: Automated releases.

### 8. Stale (`stale.yml`)
**Purpose**: Mark stale issues/PRs.

## What Gets Tested

### ✅ Currently Tested & Working
- Python package installation
- CLI command execution
- Database creation and operations
- Recommender basic functionality (fallback mode)
- API server startup
- Multi-platform compatibility (Linux/macOS/Windows)
- Latest Python version (3.13)

### ⚠️ Tested but Incomplete
- Unit tests (most are stubs)
- Integration tests (framework exists)
- Coverage reporting (will be low)

### ❌ Not Yet Tested
- Actual event log parsing with real data
- Similarity-based recommendations
- ML model training/prediction
- Cloud provider integrations
- Web UI (separate project)

## Expected CI Results

When you push this branch:

1. **smoke-test.yml**: Should ✅ PASS
2. **test.yml**: May show ⚠️ warnings but won't fail hard
3. **pr-checks.yml**: Should ✅ PASS (if branch follows naming convention)
4. **lint.yml**: Depends on current code style
5. **docker.yml**: Should ✅ PASS with new Dockerfile

## How to Fix Failing Tests

### If smoke tests fail:
```bash
# Run locally first
python smoke_test.py

# Fix any issues with imports or basic functionality
```

### If unit tests fail (beyond expected TODOs):
```bash
# Run tests locally
pytest tests/ -v

# Implement the failing tests
```

### If Docker build fails:
```bash
# Test Docker build locally
docker build -t spark-optimizer:test .
docker run --rm spark-optimizer:test spark-optimizer --version
```

## Adding New Tests

When implementing new features, update the smoke test:

```python
# In smoke_test.py
def test_new_feature():
    """Test your new feature."""
    # Your test code
    pass
```

And add proper unit tests in `tests/`:

```python
# In tests/test_feature.py
def test_feature_implementation():
    """Actual test implementation."""
    # Replace the TODO
    pass
```

## CI Best Practices

1. **Always run smoke test locally** before pushing
2. **Check that tests pass** in all Python versions you support
3. **Update this document** when adding new workflows
4. **Monitor CI failures** and fix promptly
5. **Don't ignore warnings** - they indicate technical debt

## Monitoring

- Check GitHub Actions tab for pipeline status
- Review failed jobs for error details
- Use workflow badges in README for visibility
