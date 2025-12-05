# Docker Deployment Guide

## Overview

This guide covers deploying the Spark Resource Optimizer using Docker and Docker Compose.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 2GB+ available RAM
- 10GB+ available disk space

## Quick Start

### 1. Basic Deployment (API + Database)

```bash
# Clone the repository
git clone https://github.com/yourusername/spark-resource-optimizer.git
cd spark-resource-optimizer

# Copy environment file
cp .env.example .env

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api
```

The API will be available at `http://localhost:8080`

### 2. Check Health

```bash
curl http://localhost:8080/health
```

Expected response:
```json
{
    "status": "healthy",
    "service": "spark-resource-optimizer"
}
```

## Service Profiles

Docker Compose uses profiles to enable optional services:

### Default Profile (API + Database)

```bash
docker-compose up -d
```

Services started:
- `api`: Main REST API server
- `db`: PostgreSQL database

### With Background Worker

```bash
docker-compose --profile with-worker up -d
```

Additional services:
- `worker`: Celery worker for async tasks
- `redis`: Redis for task queue

### With Spark History Server

```bash
docker-compose --profile with-spark up -d
```

Additional services:
- `spark-history`: Spark History Server for testing

### With Monitoring

```bash
docker-compose --profile with-monitoring up -d
```

Additional services:
- `prometheus`: Metrics collection
- `grafana`: Visualization dashboard

Access Grafana at `http://localhost:3000` (admin/admin)

### With Database Tools

```bash
docker-compose --profile with-tools up -d
```

Additional services:
- `pgadmin`: PostgreSQL management UI

Access pgAdmin at `http://localhost:5050` (admin@sparkoptimizer.com/admin)

### All Services

```bash
docker-compose --profile with-worker --profile with-spark --profile with-monitoring --profile with-tools up -d
```

## Configuration

### Environment Variables

Edit `.env` file to customize configuration:

```bash
# Database
SPARK_OPTIMIZER_DB_URL=postgresql://user:password@host:port/database

# API
SPARK_OPTIMIZER_API_HOST=0.0.0.0
SPARK_OPTIMIZER_API_PORT=8080
SPARK_OPTIMIZER_LOG_LEVEL=INFO

# Application
DEFAULT_RECOMMENDER_METHOD=similarity
MIN_SIMILARITY_THRESHOLD=0.7
```

### Volume Mounts

Map local directories to containers:

```yaml
volumes:
  # Event logs (read-only)
  - ./event_logs:/app/event_logs:ro

  # Data directory (database, cache)
  - ./data:/app/data

  # Spark events (for History Server)
  - ./spark-events:/spark-events:ro
```

## Common Operations

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api

# Last 100 lines
docker-compose logs --tail=100 api
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart api
```

### Stop Services

```bash
# Stop all
docker-compose stop

# Stop specific service
docker-compose stop api
```

### Remove Services

```bash
# Stop and remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v

# Remove everything including images
docker-compose down -v --rmi all
```

### Scale Services

```bash
# Run multiple API instances
docker-compose up -d --scale api=3
```

### Execute Commands in Container

```bash
# Open shell
docker-compose exec api bash

# Run CLI command
docker-compose exec api spark-optimizer list-jobs --limit 10

# Run Python script
docker-compose exec api python scripts/setup_db.py
```

## Database Management

### Initialize Database

```bash
docker-compose exec api python scripts/setup_db.py
```

### Connect to Database

```bash
# Using docker-compose
docker-compose exec db psql -U spark_optimizer -d spark_optimizer

# Using local psql client
psql -h localhost -p 5432 -U spark_optimizer -d spark_optimizer
```

### Backup Database

```bash
# Create backup
docker-compose exec db pg_dump -U spark_optimizer spark_optimizer > backup.sql

# Restore backup
docker-compose exec -T db psql -U spark_optimizer spark_optimizer < backup.sql
```

### Database Migrations

```bash
# Run migrations
docker-compose exec api alembic upgrade head

# Create new migration
docker-compose exec api alembic revision --autogenerate -m "description"
```

## Data Collection

### Collect from Event Logs

```bash
# Place event logs in ./event_logs directory
mkdir -p event_logs
cp /path/to/spark-events/* event_logs/

# Run collection
docker-compose exec api spark-optimizer collect \
  --source event-logs \
  --path /app/event_logs
```

### Collect from History Server

```bash
# Start History Server profile
docker-compose --profile with-spark up -d

# Run collection
docker-compose exec api spark-optimizer collect \
  --source history-server \
  --path http://spark-history:18080
```

## API Usage

### Get Recommendation

```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "input_size_gb": 50.0,
    "job_type": "etl"
  }'
```

### List Jobs

```bash
curl http://localhost:8080/jobs?limit=10
```

### Analyze Job

```bash
curl http://localhost:8080/analyze/app-20240101-000001
```

## Building Custom Image

### Build Image

```bash
# Build with default tag
docker build -t spark-optimizer:latest .

# Build with custom tag
docker build -t spark-optimizer:v0.1.0 .

# Build with no cache
docker build --no-cache -t spark-optimizer:latest .
```

### Push to Registry

```bash
# Tag image
docker tag spark-optimizer:latest your-registry/spark-optimizer:latest

# Push to registry
docker push your-registry/spark-optimizer:latest
```

### Use Custom Image

Update `docker-compose.yml`:

```yaml
services:
  api:
    image: your-registry/spark-optimizer:latest
    # Remove 'build' section
```

## Production Deployment

### Security Considerations

1. **Change Default Passwords**
   ```bash
   # Generate secure password
   openssl rand -base64 32
   ```

2. **Use Secrets Management**
   ```yaml
   services:
     api:
       secrets:
         - db_password

   secrets:
     db_password:
       external: true
   ```

3. **Enable TLS/SSL**
   - Use reverse proxy (nginx/traefik)
   - Configure SSL certificates
   - Force HTTPS

4. **Network Isolation**
   ```yaml
   networks:
     frontend:
       driver: overlay
     backend:
       driver: overlay
       internal: true
   ```

### Resource Limits

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### Health Checks

```yaml
services:
  api:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### Logging

```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### High Availability

```yaml
services:
  api:
    deploy:
      mode: replicated
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
      restart_policy:
        condition: on-failure
```

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics` endpoint:

```bash
curl http://localhost:8080/metrics
```

### Grafana Dashboards

1. Access Grafana: `http://localhost:3000`
2. Login: admin/admin
3. Add Prometheus datasource
4. Import dashboards from `monitoring/grafana/dashboards/`

### Container Metrics

```bash
# Resource usage
docker stats

# Specific container
docker stats spark-optimizer-api
```

## Troubleshooting

### API Not Starting

```bash
# Check logs
docker-compose logs api

# Check database connection
docker-compose exec api python -c "from spark_optimizer.storage.database import Database; db = Database('postgresql://spark_optimizer:spark_password@db:5432/spark_optimizer'); print('Connected')"
```

### Database Connection Issues

```bash
# Check database is running
docker-compose ps db

# Check database logs
docker-compose logs db

# Test connection
docker-compose exec db psql -U spark_optimizer -d spark_optimizer -c "SELECT 1;"
```

### Permission Issues

```bash
# Fix volume permissions
sudo chown -R 1000:1000 data/ logs/

# Or run as root (not recommended for production)
docker-compose exec -u root api bash
```

### Out of Memory

```bash
# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Check memory usage
docker stats

# Clear unused resources
docker system prune -a
```

### Port Already in Use

```bash
# Find process using port
lsof -i :8080  # macOS/Linux
netstat -ano | findstr :8080  # Windows

# Kill process or change port in docker-compose.yml
```

## Maintenance

### Update Application

```bash
# Pull latest code
git pull

# Rebuild image
docker-compose build api

# Restart with new image
docker-compose up -d api
```

### Clean Up

```bash
# Remove stopped containers
docker-compose rm

# Remove unused images
docker image prune

# Remove unused volumes
docker volume prune

# Full cleanup
docker system prune -a --volumes
```

### Backup and Restore

```bash
# Backup volumes
docker run --rm -v spark-resource-optimizer_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data

# Restore volumes
docker run --rm -v spark-resource-optimizer_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /
```

## Development with Docker

### Live Code Reload

```yaml
services:
  api:
    volumes:
      - ./src:/app/src
    environment:
      - FLASK_ENV=development
    command: flask run --host 0.0.0.0 --reload
```

### Run Tests

```bash
# Run all tests
docker-compose exec api pytest

# Run with coverage
docker-compose exec api pytest --cov=spark_optimizer

# Run specific test
docker-compose exec api pytest tests/test_recommender/
```

### Debug Container

```bash
# Start with shell
docker-compose run --rm api bash

# Install ipdb for debugging
docker-compose exec api pip install ipdb
```

## Docker Swarm Deployment

### Initialize Swarm

```bash
docker swarm init
```

### Deploy Stack

```bash
docker stack deploy -c docker-compose.yml spark-optimizer
```

### Scale Services

```bash
docker service scale spark-optimizer_api=5
```

### Monitor Services

```bash
docker service ls
docker service ps spark-optimizer_api
docker service logs spark-optimizer_api
```

## Kubernetes Deployment

For Kubernetes deployment, see `k8s/` directory (future addition):

```bash
kubectl apply -f k8s/
kubectl get pods
kubectl logs -f deployment/spark-optimizer-api
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/yourusername/spark-resource-optimizer/issues
- Documentation: https://spark-optimizer.readthedocs.io
