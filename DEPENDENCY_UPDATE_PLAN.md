# Spark Optimizer - Dependency Update Plan

This plan outlines the steps to update all project dependencies to their latest versions.

---

## Overview

| Category | Files Affected | Priority |
|----------|---------------|----------|
| Python Core Dependencies | `pyproject.toml`, `requirements.txt` | High |
| Python Dev Dependencies | `pyproject.toml`, `requirements-dev.txt` | Medium |
| Frontend Dependencies | `web-ui-dashboard/package.json` | High |
| Docker Base Images | `Dockerfile`, `docker-compose.yml` | High |
| CI/CD Actions | `.github/workflows/*.yml` | Medium |
| Helm Chart Dependencies | `helm/spark-optimizer/Chart.yaml` | Low |

---

## Phase 1: Python Dependencies

### 1.1 Core Dependencies (`pyproject.toml` & `requirements.txt`)

| Package | Current | Latest | Breaking Changes |
|---------|---------|--------|------------------|
| pyspark | >=3.0.0 | 3.5.x | Minor API changes in 3.5 |
| pandas | >=1.3.0 | 2.2.x | Yes - significant changes in 2.0 |
| numpy | >=1.21.0 | 2.1.x | Yes - some dtype changes |
| scikit-learn | >=1.0.0 | 1.6.x | Minor deprecations |
| sqlalchemy | >=1.4.0 | 2.0.x | Yes - major rewrite |
| alembic | >=1.7.0 | 1.14.x | Minor |
| psycopg2-binary | >=2.9.0 | 2.9.x | None |
| flask | >=2.0.0 | 3.1.x | Yes - some breaking changes |
| flask-restful | >=0.3.9 | 0.3.x | None (consider Flask-RESTX) |
| flask-cors | >=3.0.10 | 5.0.x | Minor |
| pyyaml | >=5.4.0 | 6.0.x | Minor |
| python-dotenv | >=0.19.0 | 1.0.x | Minor |
| loguru | >=0.6.0 | 0.7.x | None |
| pydantic | >=1.9.0 | 2.10.x | Yes - major rewrite |
| requests | >=2.27.0 | 2.32.x | None |
| click | >=8.0.0 | 8.1.x | None |
| tabulate | >=0.9.0 | 0.9.x | None |
| python-dateutil | >=2.8.0 | 2.9.x | None |

**Recommended Actions:**
1. Update pandas to `>=2.0.0` - requires code review for deprecated APIs
2. Update numpy to `>=1.26.0` (compatible with pandas 2.x)
3. Update SQLAlchemy to `>=2.0.0` - requires migration to 2.0 style queries
4. Update Flask to `>=3.0.0` - review async support changes
5. Update Pydantic to `>=2.0.0` - significant model syntax changes
6. Update all other packages to latest stable versions

### 1.2 Optional Dependencies

| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| boto3 | >=1.26.0 | 1.35.x | Update for latest AWS API support |
| google-cloud-dataproc | >=5.0.0 | 5.14.x | Update for latest GCP features |
| google-cloud-monitoring | >=2.11.0 | 2.24.x | Minor updates |

### 1.3 Development Dependencies

| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| pytest | >=7.0.0 | 8.3.x | Some fixture changes |
| pytest-cov | >=4.0.0 | 6.0.x | Minor |
| pytest-mock | >=3.10.0 | 3.14.x | Minor |
| pytest-asyncio | >=0.20.0 | 0.25.x | Mode changes |
| black | >=22.0.0 | 24.10.x | Style updates |
| flake8 | >=5.0.0 | 7.1.x | Rule updates |
| isort | >=5.11.0 | 5.13.x | Minor |
| mypy | >=0.990 | 1.13.x | Stricter type checking |
| pylint | >=2.15.0 | 3.3.x | Breaking changes in config |
| pre-commit | >=2.20.0 | 4.0.x | Hook format changes |
| sphinx | >=5.0.0 | 8.1.x | Theme compatibility |
| sphinx-rtd-theme | >=1.1.0 | 3.0.x | Sphinx 8 compatibility |
| ipython | >=8.0.0 | 8.30.x | Minor |
| ipdb | >=0.13.0 | 0.13.x | None |

---

## Phase 2: Frontend Dependencies

### 2.1 Angular & Core Dependencies (`web-ui-dashboard/package.json`)

The frontend is already on Angular 21 which is very recent. Updates needed:

| Package | Current | Latest | Notes |
|---------|---------|--------|-------|
| @angular/* | ^21.0.0 | 21.0.x | Already latest major |
| chart.js | ^4.5.1 | 4.4.x | Current is ahead of stable |
| ng2-charts | ^8.0.0 | 8.0.x | Latest |
| rxjs | ~7.8.0 | 7.8.x | Latest 7.x |
| typescript | ~5.9.2 | 5.7.x | Angular 21 requirement |
| tailwindcss | ^4.1.12 | 4.x | Latest |
| @playwright/test | ^1.57.0 | 1.49.x | Current is ahead |

**Recommended Actions:**
1. Frontend dependencies appear to be already up-to-date or ahead
2. Verify compatibility and run `pnpm update`
3. Regenerate `pnpm-lock.yaml`

---

## Phase 3: Docker & Infrastructure

### 3.1 Docker Base Images

| Image | Current | Latest | Notes |
|-------|---------|--------|-------|
| python | 3.11-slim | 3.13-slim | Python 3.13 is latest stable |
| postgres | 15-alpine | 17-alpine | PostgreSQL 17 released |
| redis | 7-alpine | 7.4-alpine | Latest 7.x |
| bitnami/spark | 3.5.0 | 3.5.4 | Latest 3.5.x |
| prom/prometheus | latest | v3.0.x | Pin to specific version |
| grafana/grafana | latest | 11.4.x | Pin to specific version |
| dpage/pgadmin4 | latest | 8.14 | Pin to specific version |

**Recommended Actions:**
1. Update Python base image to 3.13-slim (or 3.12-slim for stability)
2. Update PostgreSQL to 17-alpine
3. Pin all `latest` tags to specific versions for reproducibility
4. Update Spark to 3.5.4

### 3.2 Dockerfile Multi-stage Build
- Consider using `uv` instead of `pip` for faster dependency installation
- Add health check instructions

---

## Phase 4: CI/CD GitHub Actions

### 4.1 GitHub Actions Versions

| Action | Current | Latest | Notes |
|--------|---------|--------|-------|
| actions/checkout | v4 | v4 | Latest |
| actions/setup-python | v6 | v5 | v6 doesn't exist, verify |
| actions/setup-node | v6 | v4 | v6 doesn't exist, verify |
| docker/setup-buildx-action | v3 | v3 | Latest |
| docker/build-push-action | v6 | v6 | Latest |
| docker/login-action | v3 | v3 | Latest |
| actions/upload-artifact | v6 | v4 | v6 doesn't exist, verify |
| actions/cache | v5 | v4 | v5 doesn't exist, verify |
| pnpm/action-setup | v4 | v4 | Latest |
| codecov/codecov-action | v4 | v5 | Update available |
| aquasecurity/trivy-action | 0.33.1 | 0.29.x | Current ahead |
| github/codeql-action | v4 | v3 | v4 doesn't exist, verify |

**Note:** Some actions show versions that don't exist yet. These may be from recent Dependabot updates or need verification.

**Recommended Actions:**
1. Audit all workflow files for action version accuracy
2. Update codecov/codecov-action to v5
3. Ensure all actions use latest stable versions

### 4.2 CI Python Matrix
- Current: 3.10, 3.11, 3.12, 3.13
- Add Python 3.14 when available (currently in alpha)
- Consider dropping 3.10 if moving minimum to 3.11

---

## Phase 5: Helm Chart Dependencies

### 5.1 Bitnami Charts

| Chart | Current | Latest | Notes |
|-------|---------|--------|-------|
| postgresql | 13.x.x | 16.x.x | Major update available |
| redis | 18.x.x | 20.x.x | Major update available |

**Recommended Actions:**
1. Update Helm chart dependencies
2. Review breaking changes in values.yaml structure
3. Test with `helm dependency update`

---

## Implementation Steps

### Step 1: Create Feature Branch
```bash
git checkout -b feature/dependency-updates
```

### Step 2: Python Updates (Highest Risk)

1. **Update pyproject.toml** with new version constraints
2. **Run tests** after each major package update:
   - SQLAlchemy 2.0 migration (most complex)
   - Pydantic 2.0 migration
   - Pandas 2.0 migration
   - Flask 3.0 migration
3. **Update requirements.txt and requirements-dev.txt**
4. **Run full test suite**: `pytest tests/ -v`

### Step 3: Frontend Updates

1. Update `package.json` dependencies
2. Run `pnpm install`
3. Run `pnpm build` to verify build
4. Run E2E tests: `pnpm test:e2e`

### Step 4: Docker Updates

1. Update base images in `Dockerfile`
2. Update service images in `docker-compose.yml`
3. Build and test: `docker compose build && docker compose up`
4. Run integration tests

### Step 5: CI/CD Updates

1. Update GitHub Actions versions in all workflow files
2. Test workflows on feature branch

### Step 6: Helm Updates

1. Update `Chart.yaml` dependencies
2. Run `helm dependency update`
3. Test deployment in staging environment

---

## Risk Assessment

| Update | Risk Level | Mitigation |
|--------|------------|------------|
| SQLAlchemy 1.4 → 2.0 | **High** | Gradual migration, use compatibility mode |
| Pydantic 1.x → 2.x | **High** | Use `pydantic.v1` compatibility imports first |
| Pandas 1.x → 2.x | **Medium** | Review deprecated API usage |
| Flask 2.x → 3.x | **Medium** | Test all endpoints |
| Python 3.11 → 3.13 | **Low** | Good backward compatibility |
| PostgreSQL 15 → 17 | **Low** | Mostly compatible |

---

## Testing Checklist

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Docker build succeeds
- [ ] Docker Compose stack starts correctly
- [ ] API endpoints respond correctly
- [ ] Frontend builds and runs
- [ ] E2E tests pass
- [ ] CI/CD pipelines succeed
- [ ] Helm chart deploys successfully

---

## Rollback Plan

If issues are discovered after deployment:
1. Revert to previous dependency versions
2. Use git tags for version tracking
3. Keep previous Docker images tagged and available

---

## Estimated Effort

| Phase | Complexity | Description |
|-------|------------|-------------|
| Python Core | High | SQLAlchemy, Pydantic, Pandas migrations |
| Python Dev | Low | Mostly compatible updates |
| Frontend | Low | Already up-to-date |
| Docker | Medium | Image updates and testing |
| CI/CD | Low | Version bumps |
| Helm | Medium | Chart dependency updates |

---

## Next Steps

1. Review this plan and prioritize phases
2. Decide on Python version target (3.12 or 3.13)
3. Create tracking issues for each phase
4. Begin with Phase 1 (Python) as it has the most impact
