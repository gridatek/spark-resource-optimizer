# Spark Resource Optimizer - Complete Guide

## Table of Contents
1. [Getting Started](#getting-started)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Usage Examples](#usage-examples)
5. [API Reference](#api-reference)
6. [Configuration](#configuration)
7. [Deployment](#deployment)
8. [Contributing](#contributing)

---

## Getting Started

### What is Spark Resource Optimizer?

Spark Resource Optimizer is an open-source tool that helps you choose the best resource configuration for your Apache Spark jobs by learning from historical runs. It analyzes past job executions and recommends optimal settings for:

- Number of executors
- CPU cores per executor
- Memory per executor
- Shuffle partitions
- And more...

### Key Features

✅ **Multiple Collection Methods** - Parse event logs, query History Server, or integrate with cloud providers
✅ **Smart Recommendations** - Uses similarity matching and ML to suggest optimal configurations
✅ **Cost-Aware** - Balance performance vs. cost based on your priorities
✅ **Easy Integration** - REST API, CLI, and Python SDK
✅ **Cloud-Ready** - Works with AWS EMR, Databricks, GCP Dataproc, and more

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   Data Collection Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Event Logs   │  │ History API  │  │ Cloud APIs   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                    Storage Layer                        │
│  ┌───────────────────────────────────────────────────┐  │
│  │  SQLite/PostgreSQL Database                       │  │
│  │  - Job Metadata                                   │  │
│  │  - Performance Metrics                            │  │
│  │  - Resource Configurations                        │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  Analysis & Recommendation              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Feature      │  │ Similarity   │  │ ML Models    │   │
│  │ Extraction   │  │ Matching     │  │              │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                      API Layer                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ REST API     │  │ CLI          │  │ Python SDK   │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### How It Works

1. **Collect**: Parse Spark event logs or query APIs to gather historical job data
2. **Store**: Save metrics and configurations in a structured database
3. **Analyze**: Extract features and identify patterns in resource usage
4. **Recommend**: Match new jobs with similar historical runs and suggest optimal configs
5. **Optimize**: Apply recommendations and track improvements

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) PostgreSQL for production deployments

### Quick Install

```bash
# Clone the repository
git clone https://github.com/gridatek/spark-resource-optimizer.git
cd spark-resource-optimizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Initialize database
python -c "from spark_optimizer.storage.database import DatabaseManager; DatabaseManager()"
```

### Docker Install

```bash
# Build and run with Docker Compose
docker-compose up -d

# The API will be available at http://localhost:8080
```

---

## Usage Examples

### Example 1: Collect Historical Data

```bash
# Collect from local event logs
spark-optimizer collect --event-log-dir /path/to/spark/eventlogs

# Example output:
# Collecting metrics from: /path/to/spark/eventlogs
# Processing event logs... ████████████████████ 100%
# ✓ Successfully processed 127 jobs
```

### Example 2: Get Recommendations

```bash
# Get recommendation for a 50GB ETL job
spark-optimizer recommend \
  --input-size 50GB \
  --job-type etl \
  --priority balanced

# Output:
# ============================================================
# Recommended Configuration
# ============================================================
# 
# ┌────────────────────────┬────────────┐
# │ Parameter              │ Value      │
# ├────────────────────────┼────────────┤
# │ Executors              │ 15         │
# │ Cores per executor     │ 4          │
# │ Memory per executor    │ 8192 MB    │
# │ Driver memory          │ 4096 MB    │
# │ Driver cores           │ 2          │
# │ Shuffle partitions     │ 200        │
# │ Dynamic allocation     │ Disabled   │
# └────────────────────────┴────────────┘
# 
# Predictions:
# Expected Duration: 12.5 minutes
# Expected Cost: $3.25
# Confidence: 85%
```

### Example 3: Use Python SDK

```python
from spark_optimizer.storage.database import DatabaseManager
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender

# Initialize
db = DatabaseManager()
recommender = SimilarityRecommender(db)

# Get recommendation
rec = recommender.recommend(
    input_size_bytes=50 * 1024**3,  # 50 GB
    job_type="etl",
    sla_minutes=15,
    priority="performance"
)

print(f"Use {rec.num_executors} executors with {rec.executor_memory_mb}MB each")
print(f"Expected to complete in {rec.predicted_duration_ms/60000:.1f} minutes")
```

### Example 4: REST API

```bash
# Get recommendation via API
curl -X POST http://localhost:8080/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "input_size_bytes": 53687091200,
    "job_type": "etl",
    "priority": "balanced",
    "sla_minutes": 15
  }'

# Response:
{
  "num_executors": 15,
  "executor_cores": 4,
  "executor_memory_mb": 8192,
  "predicted_duration_ms": 750000,
  "predicted_cost": 3.25,
  "confidence_score": 0.85,
  "similar_jobs": ["app-20241201-001", "app-20241125-045"],
  "reasoning": "Based on 8 similar jobs..."
}
```

### Example 5: Generate spark-submit Command

```bash
# Get recommendation as spark-submit command
spark-optimizer recommend \
  --input-size 100GB \
  --job-type ml \
  --format spark-submit

# Output:
spark-submit \
  --num-executors 20 \
  --executor-cores 4 \
  --executor-memory 16384m \
  --driver-memory 4096m \
  --driver-cores 2 \
  --conf spark.sql.shuffle.partitions=400 \
  your-application.jar
```

### Example 6: Analyze Existing Job

```bash
# Analyze a specific job for optimization opportunities
spark-optimizer analyze --app-id app-20241201-123456

# Output:
# ============================================================
# Job Analysis: ETL Pipeline
# ============================================================
# 
# Application ID: app-20241201-123456
# Duration: 18.5 minutes
# Input Data: 45.2 GB
# Shuffle Data: 12.8 GB
# 
# Resource Configuration:
#   Executors: 10
#   Cores per executor: 4
#   Memory per executor: 8192 MB
# 
# ============================================================
# Optimization Opportunities:
# ============================================================
# 
# ⚠ Disk spill detected: 2.3 GB
#   → Consider increasing executor memory or reducing partition size
# 
# ⚠ High GC time: 15.2% of execution time
#   → Consider increasing executor memory or reducing memory pressure
```

---

## API Reference

### REST API Endpoints

#### POST /api/v1/recommend
Get resource recommendations for a new job.

**Request:**
```json
{
  "input_size_bytes": 10737418240,
  "job_type": "etl",
  "sla_minutes": 30,
  "budget_dollars": 5.0,
  "priority": "balanced"
}
```

**Response:**
```json
{
  "num_executors": 10,
  "executor_cores": 4,
  "executor_memory_mb": 8192,
  "driver_memory_mb": 4096,
  "driver_cores": 2,
  "shuffle_partitions": 200,
  "dynamic_allocation": false,
  "predicted_duration_ms": 720000,
  "predicted_cost": 2.5,
  "confidence_score": 0.82,
  "similar_jobs": ["app-001", "app-002"],
  "recommendation_method": "similarity",
  "reasoning": "Based on 5 similar jobs..."
}
```

#### GET /api/v1/jobs
List recent Spark jobs.

**Query Parameters:**
- `limit` (int): Number of jobs to return (default: 20)
- `job_type` (string): Filter by job type
- `min_duration` (int): Minimum duration in seconds
- `max_duration` (int): Maximum duration in seconds

#### GET /api/v1/jobs/{app_id}
Get detailed information about a specific job.

#### GET /api/v1/jobs/{app_id}/analyze
Analyze a job and get optimization suggestions.

#### GET /api/v1/stats
Get aggregate statistics about all jobs.

### CLI Commands

```bash
# Collect data
spark-optimizer collect --event-log-dir PATH [--db-url URL] [--batch-size N]

# Get recommendations
spark-optimizer recommend --input-size SIZE [OPTIONS]

# List jobs
spark-optimizer list-jobs [--limit N] [--job-type TYPE]

# Analyze job
spark-optimizer analyze --app-id APP_ID

# Show statistics
spark-optimizer stats

# Start API server
spark-optimizer serve [--port PORT] [--host HOST]
```

---

## Configuration

### Database Configuration

By default, the tool uses SQLite. For production, use PostgreSQL:

```python
# config.yaml
database:
  url: postgresql://user:password@localhost:5432/spark_optimizer
  pool_size: 10
  max_overflow: 20
```

### Recommendation Settings

```python
# config.yaml
recommendations:
  similarity_threshold: 0.7
  min_similar_jobs: 3
  max_similar_jobs: 10
  
  # Weight factors for scoring
  weights:
    performance: 0.5
    cost: 0.5
```

### Cloud Integration

```python
# config.yaml
cloud:
  aws:
    region: us-east-1
    emr_cluster_id: j-ABC123
  
  databricks:
    workspace_url: https://your-workspace.cloud.databricks.com
    token: your-token
```

---

## Deployment

### Production Deployment with Docker

```bash
# 1. Build image
docker build -t spark-optimizer:latest .

# 2. Run with PostgreSQL
docker-compose -f docker-compose.prod.yml up -d

# 3. Verify
curl http://localhost:8080/health
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spark-optimizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: spark-optimizer
  template:
    metadata:
      labels:
        app: spark-optimizer
    spec:
      containers:
      - name: spark-optimizer
        image: spark-optimizer:latest
        ports:
        - containerPort: 8080
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: spark-optimizer-secrets
              key: database-url
```

### Monitoring

The application exposes metrics at `/metrics` endpoint for Prometheus:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'spark-optimizer'
    static_configs:
      - targets: ['localhost:8080']
```

---

## Advanced Features

### Custom ML Models

You can train custom ML models for better predictions:

```python
from spark_optimizer.recommender.ml_recommender import MLRecommender

# Train model
recommender = MLRecommender(db)
recommender.train(
    features=['input_size', 'shuffle_ratio', 'stage_count'],
    target='duration_ms',
    model_type='xgboost'
)

# Use for predictions
rec = recommender.recommend(input_size_bytes=50*1024**3)
```

### Integration with CI/CD

Add to your deployment pipeline:

```yaml
# .github/workflows/optimize-spark.yml
- name: Get Spark Configuration
  run: |
    RECOMMENDATION=$(spark-optimizer recommend \
      --input-size $INPUT_SIZE \
      --format json)
    
    echo "EXECUTORS=$(echo $RECOMMENDATION | jq -r '.num_executors')" >> $GITHUB_ENV
    echo "MEMORY=$(echo $RECOMMENDATION | jq -r '.executor_memory_mb')" >> $GITHUB_ENV

- name: Submit Spark Job
  run: |
    spark-submit \
      --num-executors $EXECUTORS \
      --executor-memory ${MEMORY}m \
      my-job.py
```

---

## Troubleshooting

### Common Issues

**Problem:** No similar jobs found
- **Solution:** Collect more historical data or relax similarity thresholds

**Problem:** Recommendations seem inaccurate
- **Solution:** Review job classifications and ensure similar jobs are properly tagged

**Problem:** Event log parsing fails
- **Solution:** Check Spark version compatibility and log format

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
spark-optimizer recommend --input-size 10GB
```

---

## Roadmap

### v0.2.0 (Q1 2025)
- [ ] ML-based prediction models
- [ ] Real-time monitoring integration
- [ ] Web UI dashboard
- [ ] Databricks integration

### v0.3.0 (Q2 2025)
- [ ] Auto-tuning capabilities
- [ ] Cost optimization for spot instances
- [ ] Multi-cloud support
- [ ] Advanced analytics

### v1.0.0 (Q3 2025)
- [ ] Production-ready stability
- [ ] Enterprise features
- [ ] Performance benchmarks
- [ ] Comprehensive documentation

---

## Community & Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Slack**: Join our community chat
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Built with ❤️ by the open-source community. Special thanks to all contributors!
