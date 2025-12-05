# CI/CD Documentation

## Overview

The Spark Resource Optimizer uses GitHub Actions for continuous integration and deployment. This document describes the CI/CD pipeline and how to work with it.

## Workflows

### 1. Tests (`test.yml`)

**Triggers**: Push to main, Pull requests

**Jobs**:
- **test**: Runs tests on multiple OS (Ubuntu, Windows, macOS) with Python 3.13
- **test-integration**: Runs integration tests with PostgreSQL
- **test-docker**: Builds and tests Docker image

**What it does**:
- Installs dependencies
- Runs pytest with coverage
- Uploads coverage to Codecov
- Tests Docker image build

**Requirements**:
- All tests must pass
- Coverage threshold: 80%

### 2. Lint and Code Quality (`lint.yml`)

**Triggers**: Push to main, Pull requests

**Jobs**:
- **lint**: Runs flake8 for code linting
- **format**: Checks code formatting with black
- **type-check**: Runs mypy for type checking
- **security**: Runs bandit for security analysis
- **dependency-check**: Checks for vulnerable dependencies with safety
- **code-quality**: SonarCloud analysis

**What it does**:
- Enforces code style (black, flake8)
- Checks type annotations (mypy)
- Scans for security vulnerabilities (bandit)
- Checks dependencies for known vulnerabilities (safety)
- Analyzes code quality (SonarCloud)

**Requirements**:
- No linting errors
- Code must be formatted with black
- No security issues found

### 3. Docker Build and Push (`docker.yml`)

**Triggers**: Push to main, Tags (v*.*.*), Pull requests

**Jobs**:
- **build-and-push**: Builds and pushes Docker images to GHCR
- **build-docker-compose**: Tests docker-compose setup

**What it does**:
- Builds Docker images for multiple platforms (amd64, arm64)
- Pushes to GitHub Container Registry
- Runs Trivy security scan
- Tests docker-compose deployment

**Image Tags**:
- `latest`: Latest main branch
- `v1.0.0`: Version tags
- `main-sha123456`: Branch + commit SHA
- `pr-123`: Pull request number

### 4. Release (`release.yml`)

**Triggers**: Tags (v*.*.*)

**Jobs**:
- **create-release**: Creates GitHub release with changelog
- **build-and-publish**: Builds and publishes to PyPI
- **build-docker-release**: Builds and pushes release Docker images
- **update-documentation**: Updates documentation on GitHub Pages

**What it does**:
- Generates changelog from commits
- Creates GitHub release
- Publishes package to PyPI
- Builds multi-platform Docker images
- Deploys documentation to GitHub Pages

**Requirements**:
- Tag must follow semantic versioning (v1.0.0)
- All tests must pass
- Requires PyPI token secret

### 5. CodeQL Analysis (`codeql.yml`)

**Triggers**: Push to main, Pull requests, Weekly schedule

**What it does**:
- Scans code for security vulnerabilities
- Identifies coding errors
- Uploads results to GitHub Security

### 6. PR Checks (`pr-checks.yml`)

**Triggers**: Pull request events

**Jobs**:
- **pr-title-check**: Validates PR title format
- **pr-size-check**: Adds size labels and warns on large PRs
- **label-check**: Ensures PR has required labels
- **conflict-check**: Checks for merge conflicts
- **branch-check**: Validates branch naming convention

**What it does**:
- Enforces conventional commits format
- Labels PRs by size (XS, S, M, L, XL)
- Warns on large PRs (>1000 lines)
- Checks for required labels
- Validates branch names

### 7. Stale Issues and PRs (`stale.yml`)

**Triggers**: Daily schedule

**What it does**:
- Marks issues/PRs as stale after 60 days of inactivity
- Closes stale issues/PRs after 14 more days
- Exempts pinned, security, and roadmap items

## Required Secrets

Configure these in GitHub repository settings:

### PyPI Publishing
- `PYPI_API_TOKEN`: Token for publishing to PyPI
- `TEST_PYPI_API_TOKEN`: Token for publishing to Test PyPI (optional)

### Docker Hub
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token

### Code Quality
- `SONAR_TOKEN`: SonarCloud authentication token
- `CODECOV_TOKEN`: Codecov upload token (auto-generated)

## Branch Strategy

### Main Branch (`main`)
- Protected branch
- Requires PR reviews
- Requires status checks to pass
- No direct commits allowed
- Auto-deploys to production on merge

### Feature Branches
Format: `feature/description`, `fix/description`, `docs/description`

**Examples**:
- `feature/add-ml-recommender`
- `fix/memory-leak-collector`
- `docs/update-api-reference`

## Pull Request Guidelines

### PR Title Format

Use conventional commits format:

```
<type>(<scope>): <subject>

Types:
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- perf: Performance improvements
- test: Test changes
- build: Build system changes
- ci: CI/CD changes
- chore: Other changes
```

**Examples**:
- `feat(recommender): add ML-based prediction model`
- `fix(collector): handle missing event log fields`
- `docs(api): update endpoint documentation`

### PR Size Guidelines

- **XS**: < 100 lines changed
- **S**: < 300 lines changed
- **M**: < 600 lines changed
- **L**: < 1000 lines changed
- **XL**: > 1000 lines changed (try to split)

### Required Labels

Every PR must have at least one of:
- `bug`
- `enhancement`
- `documentation`
- `dependencies`
- `refactor`

## Local Development

### Run Tests Locally

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=spark_optimizer --cov-report=html

# Run specific test file
pytest tests/test_recommender/test_similarity.py
```

### Run Linting Locally

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/

# Type checking
mypy src/spark_optimizer

# Security scan
bandit -r src/
```

### Build Docker Locally

```bash
# Build image
docker build -t spark-optimizer:local .

# Run container
docker run -p 8080:8080 spark-optimizer:local

# Test with docker-compose
docker-compose up -d
```

## Creating a Release

### 1. Prepare Release

```bash
# Update version in setup.py
vim setup.py

# Update CHANGELOG.md
vim CHANGELOG.md

# Commit changes
git add setup.py CHANGELOG.md
git commit -m "chore: prepare release v1.0.0"
git push origin main
```

### 2. Create Tag

```bash
# Create and push tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

### 3. Automated Release Process

The release workflow will automatically:
1. Create GitHub release
2. Build and publish to PyPI
3. Build and push Docker images
4. Update documentation

### 4. Verify Release

- Check GitHub Releases page
- Verify PyPI package: https://pypi.org/project/spark-resource-optimizer/
- Test Docker image: `docker pull gridatek/spark-resource-optimizer:1.0.0`
- Check documentation: https://gridatek.github.io/spark-resource-optimizer/

## Troubleshooting

### Tests Failing

```bash
# Run tests locally
pytest tests/ -v

# Check specific test
pytest tests/test_file.py::test_name -v

# Debug with print statements
pytest tests/ -s
```

### Linting Errors

```bash
# Auto-format code
black src/ tests/

# Check what black would change
black --check --diff src/

# Fix flake8 issues
flake8 src/ --show-source
```

### Docker Build Failing

```bash
# Build locally to see errors
docker build -t test .

# Check build logs
docker build -t test . 2>&1 | tee build.log

# Test specific stage
docker build --target builder -t test-builder .
```

### Release Failing

**PyPI Token Issues**:
- Verify token in repository secrets
- Check token permissions
- Test with Test PyPI first

**Docker Push Issues**:
- Verify Docker Hub credentials
- Check repository permissions
- Try manual push to debug

## Monitoring

### GitHub Actions
- View workflow runs: Repository → Actions tab
- Check workflow logs for errors
- Re-run failed workflows

### Code Coverage
- View reports: https://codecov.io/gh/gridatek/spark-resource-optimizer
- Coverage must stay above 80%

### Code Quality
- View analysis: https://sonarcloud.io/dashboard?id=gridatek_spark-resource-optimizer
- Address quality gate failures

### Security
- Check Security tab for vulnerabilities
- Review Dependabot PRs
- Address security alerts promptly

## Best Practices

1. **Always run tests locally before pushing**
2. **Keep PRs small and focused**
3. **Write meaningful commit messages**
4. **Add tests for new features**
5. **Update documentation**
6. **Respond to review comments promptly**
7. **Fix CI failures immediately**
8. **Keep dependencies up to date**

## Getting Help

- GitHub Issues: https://github.com/gridatek/spark-resource-optimizer/issues
- CI/CD failures: Check workflow logs in Actions tab
- Questions: Open a discussion or issue
