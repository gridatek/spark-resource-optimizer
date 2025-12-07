# Local Testing Guide

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
python -c "from spark_optimizer.storage.database import Database; db = Database('sqlite:///spark_optimizer.db'); db.create_tables(); print('✓ Database created')"
```

Check if database was created:
```bash
ls -lh spark_optimizer.db
```

## 3. Testing What Currently Works

### A. Test Event Log Collection

The event log collector is **functional** and can parse Spark event logs. To test it:

```bash
# Create sample event log directory
mkdir -p sample_event_logs

# Test the collector (requires actual Spark event logs)
# If you have Spark event logs, copy them to sample_event_logs/
spark-optimizer collect --event-log-dir ./sample_event_logs --db-url sqlite:///spark_optimizer.db
```

**Note**: You'll need actual Spark event logs to test this. If you don't have any:
- Run a Spark job with event logging enabled: `--conf spark.eventLog.enabled=true`
- Or download sample logs from Spark examples

### B. Test Database Operations

```bash
# View database statistics
spark-optimizer stats --db-url sqlite:///spark_optimizer.db

# List jobs in database
spark-optimizer list-jobs --db-url sqlite:///spark_optimizer.db --limit 10
```

### C. Test Job Analysis (if you have data)

```bash
# Analyze a specific job (requires job data in DB)
spark-optimizer analyze --app-id <your-app-id> --db-url sqlite:///spark_optimizer.db
```

### D. Test Recommendations ⚠️ (Partially Implemented)

```bash
# Get recommendations (will use fallback logic since core logic is TODO)
spark-optimizer recommend \
  --input-size 10GB \
  --job-type etl \
  --priority balanced \
  --db-url sqlite:///spark_optimizer.db \
  --format table
```

**Expected behavior**: Should return a recommendation, but note that the similarity-based matching is not yet fully implemented (uses fallback logic).

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

# Get recommendations (API endpoint has TODOs)
curl -X POST http://localhost:8080/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "input_size_gb": 10,
    "job_type": "etl"
  }'

# List jobs
curl http://localhost:8080/api/v1/jobs
```

## 4. Testing with Python Code

Test the modules directly:

```python
# test_basic.py
from spark_optimizer.storage.database import Database
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender

# Test database
db = Database("sqlite:///test.db")
db.create_tables()
print("✓ Database works")

# Test recommender (uses fallback)
recommender = SimilarityRecommender(db)
rec = recommender.recommend(
    input_size_bytes=10 * 1024**3,  # 10GB
    job_type="etl",
    priority="balanced"
)
print(f"✓ Recommender works (using fallback): {rec}")
```

Run it:
```bash
python test_basic.py
```

## 5. Testing Web UI Dashboard (Separate Project)

The Angular dashboard is in `web-ui-dashboard/`:

```bash
cd web-ui-dashboard

# Install dependencies (requires Node.js and pnpm)
pnpm install

# Start development server
ng serve

# Open browser to http://localhost:4200
```

## 6. Running Unit Tests

Most tests are stubs (TODO), but you can run them:

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=spark_optimizer --cov-report=html

# View coverage report
open htmlcov/index.html
```

**Expected**: Most tests will pass but are empty placeholders.

## 7. Docker Testing (Optional)

If you have Docker:

```bash
# Build and run with docker-compose
docker-compose up

# Access API at http://localhost:8080
```

## What Works vs. What Doesn't

### ✅ Currently Working:
- CLI interface and command structure
- Event log parsing (EventLogCollector)
- Database schema and storage
- API server framework
- Basic recommendation fallback logic
- Job analysis CLI command

### ⚠️ Partially Working:
- Recommendations (uses fallback, not similarity-based)
- API endpoints (framework works, business logic has TODOs)

### ❌ Not Yet Implemented:
- Similarity-based job matching
- ML-based recommendations
- Rule-based optimization
- Cloud provider integrations (AWS, Databricks, GCP)
- Feature extraction and analysis algorithms
- Real-time monitoring
- Auto-tuning
- Most unit tests

## Quick Smoke Test

Run this to verify everything is set up correctly:

```bash
# 1. Check CLI
spark-optimizer --version

# 2. Create database
python -c "from spark_optimizer.storage.database import Database; Database('sqlite:///test.db').create_tables(); print('✓ DB OK')"

# 3. Test import
python -c "from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender; print('✓ Import OK')"

# 4. Start server (Ctrl+C to stop)
# spark-optimizer serve --port 8080
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'tabulate'`
**Solution**: `pip install tabulate`

**Issue**: Database not found
**Solution**: Run database creation step from section 2

**Issue**: No event logs to test with
**Solution**: Generate sample Spark event logs or use the fallback recommendation testing

## Next Steps for Development

To make this project fully functional, implement:
1. `similarity_recommender.py` - core similarity matching logic
2. `ml_recommender.py` - ML model training and prediction
3. Unit tests in `tests/` directory
4. Cloud collectors for AWS/Databricks/GCP
5. Web UI components in `web-ui-dashboard/`
