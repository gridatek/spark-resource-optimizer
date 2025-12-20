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
| Helm Chart Dependencies | `helm/spark-optimizer/Chart.yaml`, `values.yaml` | **Critical** |

> **Note:** Bitnami charts are being deprecated (August 2025) and require paid subscription ($50K+/year).
> Phase 5 covers migration to free, open-source alternatives (CloudNativePG, OT-ContainerKit Redis).

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

## Phase 5: Helm Chart Dependencies - Migrate from Bitnami

### 5.1 Why Migrate from Bitnami?

Bitnami (now owned by Broadcom) is deprecating free container images as of August 28, 2025.
The new "Bitnami Secure" subscription costs $50,000-$72,000/year. **We must migrate to free,
open-source alternatives.**

### 5.2 PostgreSQL Alternatives

| Solution | Type | Repository | Recommendation |
|----------|------|------------|----------------|
| **CloudNativePG** | Operator | `https://cloudnative-pg.github.io/charts` | ⭐ Recommended for production |
| Crunchy PGO | Operator | `https://artifacthub.io/packages/olm/community-operators/postgresql` | Enterprise-ready |
| Zalando Postgres Operator | Operator | `https://opensource.zalando.com/postgres-operator/` | Feature-rich, multi-DB per cluster |
| Percona PG Operator | Operator | `https://artifacthub.io/packages/olm/community-operators/percona-postgresql-operator` | Percona flavor |

**Recommended: CloudNativePG**
- Kubernetes-native PostgreSQL operator
- Simple architecture (1 DB per cluster)
- Active community, regular updates
- Built-in backup/restore, replication, failover
- Chart: `cloudnative-pg/cloudnative-pg` (operator) + `cloudnative-pg/cluster` (database)

### 5.3 Redis Alternatives

| Solution | Type | Repository | Recommendation |
|----------|------|------------|----------------|
| **OT-ContainerKit Redis Operator** | Operator | `https://ot-container-kit.github.io/helm-charts/` | ⭐ Recommended |
| Spotahome Redis Operator | Operator | `https://artifacthub.io/packages/helm/redis-operator/redis-operator` | Popular historical alternative |
| DandyDeveloper redis-ha | Chart | `https://dandydeveloper.github.io/charts/` | Mature community chart |

**Recommended: OT-ContainerKit Redis Operator**
- Supports standalone, cluster, and sentinel modes
- Data migration support
- Active maintenance
- Charts: `redis-operator` (operator) + `redis` / `redis-cluster` (instances)

### 5.4 Migration Plan

#### Current Chart.yaml Dependencies (to be removed):
```yaml
dependencies:
  - name: postgresql
    version: "13.x.x"
    repository: https://charts.bitnami.com/bitnami  # ❌ Remove
    condition: postgresql.enabled
  - name: redis
    version: "18.x.x"
    repository: https://charts.bitnami.com/bitnami  # ❌ Remove
    condition: redis.enabled
```

#### New Chart.yaml Dependencies:
```yaml
dependencies:
  - name: cloudnative-pg
    version: "0.23.x"
    repository: https://cloudnative-pg.github.io/charts
    condition: postgresql.enabled
  - name: redis-operator
    version: "0.18.x"
    repository: https://ot-container-kit.github.io/helm-charts/
    condition: redis.enabled
```

### 5.5 Values.yaml Migration

#### PostgreSQL - Bitnami to CloudNativePG

**Current (Bitnami):**
```yaml
postgresql:
  enabled: true
  auth:
    username: spark_optimizer
    password: ""
    database: spark_optimizer
  primary:
    persistence:
      enabled: true
      size: 10Gi
```

**New (CloudNativePG):**
```yaml
postgresql:
  enabled: true
  cluster:
    name: spark-optimizer-db
    instances: 2  # HA setup
    storage:
      size: 10Gi
    postgresql:
      parameters:
        max_connections: "200"
    bootstrap:
      initdb:
        database: spark_optimizer
        owner: spark_optimizer
        secret:
          name: spark-optimizer-db-credentials
```

#### Redis - Bitnami to OT-ContainerKit

**Current (Bitnami):**
```yaml
redis:
  enabled: false
  auth:
    enabled: true
    password: ""
  master:
    persistence:
      enabled: true
      size: 1Gi
```

**New (OT-ContainerKit):**
```yaml
redis:
  enabled: false
  mode: standalone  # or "cluster" or "sentinel"
  redisStandalone:
    storage:
      volumeClaimTemplate:
        spec:
          accessModes: ["ReadWriteOnce"]
          resources:
            requests:
              storage: 1Gi
  kubernetesConfig:
    image: redis:7-alpine
    imagePullPolicy: IfNotPresent

# Note: Redis Operator requires deploying the operator first,
# then creating Redis/RedisCluster CRs
```

### 5.6 Implementation Steps for Helm Migration

1. **Deploy Operators First** (cluster-wide, one-time setup):
   ```bash
   # CloudNativePG Operator
   helm repo add cnpg https://cloudnative-pg.github.io/charts
   helm install cnpg cnpg/cloudnative-pg -n cnpg-system --create-namespace

   # Redis Operator (if needed)
   helm repo add ot-helm https://ot-container-kit.github.io/helm-charts/
   helm install redis-operator ot-helm/redis-operator -n redis-operator --create-namespace
   ```

2. **Update Application Chart**:
   - Remove Bitnami dependencies from `Chart.yaml`
   - Add CRD templates for PostgreSQL Cluster and Redis
   - Update `values.yaml` with new structure
   - Update application database connection logic

3. **Data Migration** (if upgrading existing deployment):
   - Backup existing PostgreSQL data: `pg_dump`
   - Deploy new CloudNativePG cluster
   - Restore data: `pg_restore`
   - Update connection strings
   - Verify application connectivity

4. **Testing**:
   - Deploy to staging environment
   - Verify database connectivity
   - Test failover scenarios
   - Validate backup/restore procedures

### 5.7 Alternative: External Database

For production, consider using **managed databases** instead of in-cluster:
- AWS RDS PostgreSQL / ElastiCache Redis
- GCP Cloud SQL / Memorystore
- Azure Database for PostgreSQL / Azure Cache for Redis

This simplifies Kubernetes deployment and provides enterprise features (backups, HA, scaling).

To use external databases, set in values.yaml:
```yaml
database:
  external: true
  host: "your-rds-endpoint.amazonaws.com"
  port: 5432
  name: spark_optimizer
  username: spark_optimizer
  password: ""  # Use secret reference

postgresql:
  enabled: false  # Disable in-cluster PostgreSQL
```

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
| **Bitnami → CloudNativePG/Redis Operator** | **Critical** | Test in staging, backup data, phased rollout |
| SQLAlchemy 1.4 → 2.0 | **High** | Gradual migration, use compatibility mode |
| Pydantic 1.x → 2.x | **High** | Use `pydantic.v1` compatibility imports first |
| Pandas 1.x → 2.x | **Medium** | Review deprecated API usage |
| Flask 2.x → 3.x | **Medium** | Test all endpoints |
| Python 3.11 → 3.13 | **Low** | Good backward compatibility |
| PostgreSQL 15 → 17 | **Low** | Mostly compatible |

> **Critical Note:** The Bitnami migration has a hard deadline of August 28, 2025. After this date,
> Bitnami images will no longer receive updates and may be removed from public registries.

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
| **Helm (Bitnami Migration)** | **High** | Operator deployment, CRD templates, values restructure, data migration |

---

## Next Steps

1. Review this plan and prioritize phases
2. Decide on Python version target (3.12 or 3.13)
3. Create tracking issues for each phase
4. **Prioritize Helm/Bitnami migration** due to August 2025 deadline
5. Begin with Phase 1 (Python) as it has the most impact

---

## Sources & References

### Bitnami Migration Resources
- [helm-unbitnami: Curated alternatives to Bitnami Helm charts](https://github.com/TartanLeGrand/helm-unbitnami)
- [Broadcom Ends Free Bitnami Images](https://thenewstack.io/broadcom-ends-free-bitnami-images-forcing-users-to-find-alternatives/)
- [Bitnami Deprecation: Migration Steps and Alternatives](https://northflank.com/blog/bitnami-deprecates-free-images-migration-steps-and-alternatives)
- [Bitnami Helm Charts Deprecated: Migrate to Secure Alternative](https://www.chainguard.dev/supply-chain-security-101/a-practical-guide-to-migrating-helm-charts-from-bitnami)

### Alternative Chart Repositories
- [CloudNativePG](https://cloudnative-pg.io/) - Kubernetes PostgreSQL Operator
- [OT-ContainerKit Redis Operator](https://ot-container-kit.github.io/redis-operator/)
- [Spotahome Redis Operator](https://github.com/spotahome/redis-operator)
- [DandyDeveloper redis-ha Chart](https://github.com/DandyDeveloper/charts)
