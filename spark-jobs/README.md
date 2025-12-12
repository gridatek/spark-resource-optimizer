# Spark Sample Jobs

This directory contains sample Spark applications used for Docker integration testing. These jobs are submitted to a real Spark cluster to generate realistic event logs for the Spark Resource Optimizer to analyze.

## Available Jobs

### 1. simple_wordcount.py
A basic word count job that processes text data using Spark RDD operations.

**Configuration:**
- Executor Memory: 1GB
- Executor Cores: 1
- Purpose: Demonstrates a simple, well-configured job

**Usage:**
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-jobs/simple_wordcount.py
```

### 2. data_processing_etl.py
A more complex ETL job that simulates a typical data processing workflow with multiple transformations and aggregations.

**Configuration:**
- Executor Memory: 2GB
- Executor Cores: 2
- Shuffle Partitions: 8
- Purpose: Demonstrates a realistic ETL workflow with good performance

**Usage:**
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-jobs/data_processing_etl.py
```

### 3. inefficient_job.py
A job with intentionally poor configuration to demonstrate optimization opportunities.

**Configuration:**
- Executor Memory: 512MB (too small, will cause spilling)
- Executor Cores: 1
- Shuffle Partitions: 200 (excessive)
- Purpose: Demonstrates a job that needs optimization

**Expected Issues:**
- Memory spilling to disk
- Excessive shuffles
- Low resource utilization
- High GC overhead

**Usage:**
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-jobs/inefficient_job.py
```

### 4. memory_intensive_job.py
A job that creates and caches large datasets to test memory optimization recommendations.

**Configuration:**
- Executor Memory: 1GB (moderate, may need increase)
- Executor Cores: 2
- Shuffle Partitions: 8
- Purpose: Tests memory usage optimization

**Operations:**
- Generates 10M rows with 20 columns of random data
- Caches large dataframes in memory
- Performs multiple aggregations on cached data
- Executes memory-intensive join operations

**Usage:**
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-jobs/memory_intensive_job.py
```

### 5. cpu_intensive_job.py
A job that performs complex CPU-intensive computations to test core allocation recommendations.

**Configuration:**
- Executor Memory: 2GB
- Executor Cores: 1 (limited to show CPU bottleneck)
- Shuffle Partitions: 16
- Purpose: Tests CPU/core optimization

**Operations:**
- Mathematical computations (sqrt, log, sin, cos)
- Complex string manipulations
- UDF-based transformations
- Processes 1M rows with intensive calculations

**Usage:**
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-jobs/cpu_intensive_job.py
```

### 6. skewed_data_job.py
A job with intentionally skewed data distribution to test skew detection and optimization.

**Configuration:**
- Executor Memory: 2GB
- Executor Cores: 2
- Shuffle Partitions: 8
- Purpose: Tests data skew detection

**Skew Characteristics:**
- 80% of records have the same key (key "A")
- Remaining 20% distributed across other keys
- Uneven partition sizes during aggregation
- Skewed join operations

**Usage:**
```bash
docker exec spark-master /opt/spark/bin/spark-submit \
  --master spark://spark-master:7077 \
  /opt/spark-jobs/skewed_data_job.py
```

## Integration Testing

These jobs are used by the Docker integration test script to validate the Spark Resource Optimizer's ability to:

1. Collect job data from Spark History Server
2. Analyze job performance and identify issues
3. Generate optimization recommendations
4. Track metrics like memory usage, spilling, and execution time

Run the integration test with:
```bash
python scripts/test_docker_integration.py --cleanup
```

## Event Logs

All jobs are configured with event logging enabled:
```python
.config("spark.eventLog.enabled", "true")
.config("spark.eventLog.dir", "file:///spark-events")
```

Event logs are written to `./spark-events/` and consumed by:
- Spark History Server (UI at http://localhost:18080)
- Spark Resource Optimizer (data collection API)

## Adding New Jobs

To add a new test job:

1. Create a new Python file in this directory
2. Configure the Spark session with event logging:
   ```python
   spark = SparkSession.builder \
       .appName("Your Job Name") \
       .config("spark.eventLog.enabled", "true") \
       .config("spark.eventLog.dir", "file:///spark-events") \
       .getOrCreate()
   ```
3. Make the script executable: `chmod +x spark-jobs/your_job.py`
4. Add the job to the test script in `scripts/test_docker_integration.py`
5. Test locally: `python scripts/test_docker_integration.py`
