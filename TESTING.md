# Testing Guide

This guide shows you how to test the Spark Resource Optimizer project locally.

## Prerequisites

- Python 3.8+ installed
- Virtual environment activated
- All dependencies installed

## 1. Environment Setup

```bash
# Activate virtual environment (already exists)
source venv/bin/activate

# Verify installation
spark-optimizer --version
# Should output: spark-optimizer, version 0.1.0

# View available commands
spark-optimizer --help
```

## 2. Database Setup

Initialize the SQLite database:

```bash
# Create database and tables
python -c "from spark_optimizer.storage.database import Database; db = Database('sqlite:///spark_optimizer.db'); db.create_tables(); print('Database created')"
```

Check if database was created:
```bash
ls -lh spark_optimizer.db
```

## 3. Running Unit Tests

The project has comprehensive unit tests covering all major components:

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_api/test_routes.py -v
pytest tests/test_storage/test_repository.py -v
pytest tests/test_collectors/test_event_log_collector.py -v
pytest tests/test_analyzer/test_similarity.py -v

# Run with coverage
pytest tests/ --cov=spark_optimizer --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Categories

| Test Module | Description | Tests |
|-------------|-------------|-------|
| `test_api/test_routes.py` | API endpoint tests | Health check, recommendations, job listing, analysis |
| `test_api/test_server.py` | Server integration tests | Full API workflow tests |
| `test_storage/test_repository.py` | Repository pattern tests | CRUD operations, search, filtering |
| `test_collectors/test_event_log_collector.py` | Event log parsing tests | Log parsing, metrics extraction |
| `test_analyzer/test_similarity.py` | Similarity calculation tests | Job matching, vector similarity |
| `test_recommender/*.py` | Recommender tests | All recommendation methods |
| `test_monitoring/*.py` | Monitoring module tests | SparkMonitor, AlertManager, WebSocketServer |
| `test_tuning/*.py` | Auto-tuning tests | AutoTuner, ConfigAdjuster, FeedbackLoop |
| `test_cost/*.py` | Cost modeling tests | CostModel, CostOptimizer, CloudPricing |

## 4. Testing What Currently Works

### A. Test Event Log Collection

The event log collector is **fully functional** and can parse Spark event logs:

```bash
# Create sample event log directory
mkdir -p sample_event_logs

# Test the collector (requires actual Spark event logs)
spark-optimizer collect --event-log-dir ./sample_event_logs --db-url sqlite:///spark_optimizer.db
```

**Note**: You'll need actual Spark event logs to test this. If you don't have any:
- Run a Spark job with event logging enabled: `--conf spark.eventLog.enabled=true`
- Or download sample logs from Spark examples

### B. Test History Server Collection

The History Server collector is **fully functional** and can fetch job metrics via REST API:

```bash
# Test with a running History Server
spark-optimizer collect-from-history-server \
  --history-server-url http://localhost:18080 \
  --db-url sqlite:///spark_optimizer.db \
  --max-apps 50

# Collect only completed applications
spark-optimizer collect-from-history-server \
  --history-server-url http://localhost:18080 \
  --status completed \
  --max-apps 100
```

### C. Test Recommendations

```bash
# Get recommendations using similarity-based method
spark-optimizer recommend \
  --input-size 10GB \
  --job-type etl \
  --priority balanced \
  --db-url sqlite:///spark_optimizer.db \
  --format table
```

### D. Test Job Analysis

```bash
# Analyze a specific job
spark-optimizer analyze --app-id <your-app-id> --db-url sqlite:///spark_optimizer.db
```

### E. Test API Server

Start the REST API server:

```bash
# Start server on port 8080
spark-optimizer serve --port 8080 --debug
```

In another terminal, test the API:

```bash
# Health check
curl http://localhost:8080/health

# Get recommendations
curl -X POST http://localhost:8080/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "input_size_bytes": 10737418240,
    "job_type": "etl",
    "priority": "balanced"
  }'

# List jobs
curl http://localhost:8080/api/v1/jobs

# Get job details
curl http://localhost:8080/api/v1/jobs/<app_id>

# Analyze a job
curl http://localhost:8080/api/v1/jobs/<app_id>/analyze

# Get statistics
curl http://localhost:8080/api/v1/stats
```

## 5. Running Example Scripts

The project includes working example scripts:

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced tuning example (compares multiple recommenders)
python examples/advanced_tuning.py
```

## 6. Testing with Python Code

Test the modules directly:

```python
# test_basic.py
from spark_optimizer.storage.database import Database
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender

# Test database
db = Database("sqlite:///test.db")
db.create_tables()
print("Database works")

# Test recommender
recommender = SimilarityRecommender(db)
rec = recommender.recommend(
    input_size_bytes=10 * 1024**3,  # 10GB
    job_type="etl",
    priority="balanced"
)
print(f"Recommender works: {rec}")
```

## 7. Web UI Dashboard

The Angular dashboard is in `web-ui-dashboard/`:

```bash
cd web-ui-dashboard

# Install dependencies (requires Node.js and pnpm)
pnpm install

# Start development server
ng serve

# Open browser to http://localhost:4200
```

## 8. Docker Testing (Optional)

If you have Docker:

```bash
# Build and run with docker-compose
docker-compose up

# Access API at http://localhost:8080
```

## Implementation Status

### Fully Implemented

| Component | Status | Description |
|-----------|--------|-------------|
| CLI Interface | Complete | All commands working |
| Event Log Collector | Complete | Parses Spark event logs |
| History Server Collector | Complete | REST API integration |
| EMR Collector | Complete | AWS EMR integration |
| Databricks Collector | Complete | Databricks integration |
| Dataproc Collector | Complete | GCP Dataproc integration |
| Similarity Recommender | Complete | Historical job matching |
| Rule-based Recommender | Complete | Heuristic optimization |
| ML Recommender | Complete | Random Forest/Gradient Boosting |
| Job Analyzer | Complete | Bottleneck detection, health scoring |
| Feature Extractor | Complete | ML feature engineering |
| Rule Engine | Complete | Rule-based optimization |
| Database/Storage | Complete | SQLAlchemy ORM, Repository pattern |
| REST API | Complete | All endpoints functional |
| Unit Tests | Complete | Comprehensive test coverage |
| Example Scripts | Complete | Working examples |

### Recently Implemented Features (v0.2.0)

| Feature | Status | Notes |
|---------|--------|-------|
| Real-time Monitoring | Complete | Streaming job metrics, alerts, WebSocket server |
| Auto-tuning | Complete | Automatic configuration adjustment, feedback loop |
| Advanced Cost Modeling | Complete | Multi-cloud pricing, cost optimization, instance comparison |

### Testing New Features

#### Real-time Monitoring

```python
from spark_optimizer.monitoring.monitor import SparkMonitor
from spark_optimizer.monitoring.alerts import AlertManager, AlertSeverity

# Start monitoring
monitor = SparkMonitor(poll_interval=5.0)
monitor.start()

# Add application to monitor
monitor.add_application("app-123", "my_spark_job")

# Update metrics
monitor.update_metrics("app-123", {"cpu_usage": 75.0, "memory_usage": 8192.0})

# Set up alerts
alert_manager = AlertManager()
alerts = alert_manager.evaluate_metrics("app-123", {"jvm_gc_time_percent": 15.0})

monitor.stop()
```

#### Auto-tuning

```python
from spark_optimizer.tuning.auto_tuner import AutoTuner, TuningStrategy

tuner = AutoTuner()

# Start a tuning session
session = tuner.start_session(
    app_id="app-123",
    app_name="my_spark_job",
    initial_config={"spark.executor.memory": 4096},
    strategy=TuningStrategy.MODERATE,
    target_metric="duration"
)

# Get recommendations based on metrics
adjustments = tuner.analyze_and_recommend(
    session.session_id,
    {"duration": 100.0, "memory_spill_ratio": 0.2}
)

# Apply and track adjustments
for adj in adjustments:
    tuner.apply_adjustment(session.session_id, adj)

# Get best configuration
best_config = tuner.get_best_config(session.session_id)
```

#### Cost Modeling

```python
from spark_optimizer.cost.cost_model import CostModel, InstanceType
from spark_optimizer.cost.cost_optimizer import CostOptimizer, OptimizationGoal
from spark_optimizer.cost.cloud_pricing import CloudPricing

# Estimate job cost
model = CostModel(cloud_provider="aws", region="us-east-1")
estimate = model.estimate_job_cost(
    num_executors=10,
    executor_cores=4,
    executor_memory_mb=8192,
    driver_memory_mb=4096,
    duration_hours=2.0,
    instance_type=InstanceType.ON_DEMAND
)
print(f"Estimated cost: ${estimate.total_cost:.2f}")

# Optimize configuration for cost
optimizer = CostOptimizer()
result = optimizer.optimize(
    current_config={"spark.executor.instances": 20},
    estimated_duration_hours=2.0,
    goal=OptimizationGoal.MINIMIZE_COST
)
print(f"Savings: ${result.savings:.2f} ({result.savings_percent:.1f}%)")

# Compare cloud providers
pricing = CloudPricing()
comparison = pricing.compare_providers(vcpus=4, memory_gb=16)
```

## Quick Smoke Test

Run this to verify everything is set up correctly:

```bash
# 1. Check CLI
spark-optimizer --version

# 2. Create database
python -c "from spark_optimizer.storage.database import Database; Database('sqlite:///test.db').create_tables(); print('DB OK')"

# 3. Test imports
python -c "from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender; print('Import OK')"

# 4. Run tests
pytest tests/ -v --tb=short

# 5. Start server (Ctrl+C to stop)
# spark-optimizer serve --port 8080
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'tabulate'`
**Solution**: `pip install tabulate`

**Issue**: Database not found
**Solution**: Run database creation step from section 2

**Issue**: No event logs to test with
**Solution**: Use the example scripts which create sample data, or point to real Spark event logs

**Issue**: Tests failing with import errors
**Solution**: Ensure you're in the project root and have the virtual environment activated

## Test Coverage Goals

| Module | Target Coverage |
|--------|-----------------|
| API Routes | 80%+ |
| Repository | 90%+ |
| Collectors | 85%+ |
| Recommenders | 85%+ |
| Analyzers | 80%+ |
| Monitoring | 85%+ |
| Tuning | 85%+ |
| Cost | 85%+ |
