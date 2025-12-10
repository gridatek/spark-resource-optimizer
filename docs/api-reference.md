# API Reference

## Overview

The Spark Resource Optimizer provides both a REST API and a CLI for accessing its functionality.

## REST API

Base URL: `http://localhost:8080` (default)

### Authentication

Currently, the API does not require authentication. This will be added in future versions.

### Response Format

All API responses follow this structure:

**Success Response**:
```json
{
    "status": "success",
    "data": { ... }
}
```

**Error Response**:
```json
{
    "status": "error",
    "error": "Error message",
    "code": "ERROR_CODE"
}
```

### Endpoints

---

#### Health Check

Check API server health status.

**Endpoint**: `GET /health`

**Response**:
```json
{
    "status": "healthy",
    "service": "spark-resource-optimizer",
    "version": "0.1.0"
}
```

**Example**:
```bash
curl http://localhost:8080/health
```

---

#### Get Recommendation

Get resource recommendations for a Spark job.

**Endpoint**: `POST /recommend`

**Request Body**:
```json
{
    "input_size_gb": 50.0,
    "job_type": "etl",
    "app_name": "daily_etl_job",
    "additional_params": {
        "output_format": "parquet",
        "num_partitions": 100
    }
}
```

**Parameters**:
- `input_size_gb` (required): Input data size in GB
- `job_type` (optional): Job type - `etl`, `ml`, `sql`, `streaming`
- `app_name` (optional): Application name for similarity matching
- `method` (optional): Recommendation method - `similarity`, `ml`, `rule_based`, `hybrid`
- `additional_params` (optional): Additional job-specific parameters

**Response**:
```json
{
    "configuration": {
        "executor_cores": 4,
        "executor_memory_mb": 8192,
        "num_executors": 10,
        "driver_memory_mb": 4096
    },
    "predicted_metrics": {
        "duration_minutes": 30,
        "cost_usd": 12.5
    },
    "confidence": 0.85,
    "method": "similarity",
    "metadata": {
        "similar_jobs": [
            {
                "app_id": "app-20240101-000001",
                "similarity": 0.92
            }
        ]
    }
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "input_size_gb": 50.0,
    "job_type": "etl",
    "app_name": "daily_etl_job"
  }'
```

**Python Example**:
```python
import requests

response = requests.post(
    "http://localhost:8080/recommend",
    json={
        "input_size_gb": 50.0,
        "job_type": "etl",
        "app_name": "daily_etl_job"
    }
)

recommendation = response.json()
print(f"Recommended executors: {recommendation['configuration']['num_executors']}")
```

---

#### Collect Job Data

Collect and store Spark job data from various sources.

**Endpoint**: `POST /collect`

**Request Body**:
```json
{
    "source_type": "event_log",
    "source_path": "/path/to/spark/event-logs",
    "config": {
        "recursive": true,
        "batch_size": 100
    }
}
```

**Parameters**:
- `source_type` (required): Source type - `event_log`, `history_server`, `metrics`
- `source_path` (required): Path or URL to data source
- `config` (optional): Source-specific configuration

**Response**:
```json
{
    "status": "success",
    "jobs_collected": 150,
    "jobs_stored": 145,
    "errors": 5,
    "duration_seconds": 42.5
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/collect \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "event_log",
    "source_path": "/path/to/logs"
  }'
```

---

#### List Jobs

List stored Spark jobs with optional filtering and pagination.

**Endpoint**: `GET /jobs`

**Query Parameters**:
- `limit` (optional, default: 50): Maximum number of results
- `offset` (optional, default: 0): Pagination offset
- `app_name` (optional): Filter by application name (partial match)
- `user` (optional): Filter by user
- `date_from` (optional): Start date (ISO format)
- `date_to` (optional): End date (ISO format)
- `status` (optional): Filter by status - `completed`, `failed`, `running`

**Response**:
```json
{
    "jobs": [
        {
            "app_id": "app-20240101-000001",
            "app_name": "daily_etl_job",
            "user": "data_engineer",
            "start_time": "2024-01-01T10:00:00Z",
            "end_time": "2024-01-01T10:30:00Z",
            "duration_ms": 1800000,
            "status": "completed",
            "configuration": {
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "num_executors": 10
            }
        }
    ],
    "total": 150,
    "limit": 50,
    "offset": 0
}
```

**Example**:
```bash
# List recent jobs
curl http://localhost:8080/jobs?limit=10

# Filter by app name
curl http://localhost:8080/jobs?app_name=etl

# Date range query
curl "http://localhost:8080/jobs?date_from=2024-01-01&date_to=2024-01-31"
```

---

#### Get Job Details

Get detailed information about a specific Spark job.

**Endpoint**: `GET /jobs/{app_id}`

**Path Parameters**:
- `app_id`: Spark application ID

**Response**:
```json
{
    "app_id": "app-20240101-000001",
    "app_name": "daily_etl_job",
    "user": "data_engineer",
    "submit_time": "2024-01-01T09:59:50Z",
    "start_time": "2024-01-01T10:00:00Z",
    "end_time": "2024-01-01T10:30:00Z",
    "duration_ms": 1800000,
    "spark_version": "3.4.0",
    "status": "completed",

    "configuration": {
        "executor_cores": 4,
        "executor_memory_mb": 8192,
        "num_executors": 10,
        "driver_memory_mb": 4096
    },

    "metrics": {
        "total_tasks": 500,
        "failed_tasks": 0,
        "total_stages": 10,
        "failed_stages": 0,
        "input_bytes": 53687091200,
        "output_bytes": 26843545600,
        "shuffle_read_bytes": 10737418240,
        "shuffle_write_bytes": 10737418240
    },

    "stages": [
        {
            "stage_id": 0,
            "stage_name": "collect at <console>:25",
            "num_tasks": 50,
            "duration_ms": 120000,
            "input_bytes": 5368709120
        }
    ]
}
```

**Example**:
```bash
curl http://localhost:8080/jobs/app-20240101-000001
```

---

#### Analyze Job

Analyze a specific job and return insights, bottlenecks, and suggestions.

**Endpoint**: `GET /analyze/{app_id}`

**Path Parameters**:
- `app_id`: Spark application ID

**Response**:
```json
{
    "app_id": "app-20240101-000001",
    "app_name": "daily_etl_job",

    "analysis": {
        "resource_efficiency": {
            "cpu_efficiency": 0.75,
            "memory_efficiency": 0.82,
            "io_efficiency": 0.68
        },

        "bottlenecks": [
            {
                "type": "memory",
                "severity": "medium",
                "description": "High memory spill detected in stage 3",
                "affected_stages": [3, 5]
            }
        ],

        "issues": [
            {
                "type": "data_skew",
                "severity": "high",
                "description": "Task duration variance > 3x in stage 2",
                "recommendation": "Repartition data or use salting technique"
            }
        ]
    },

    "suggestions": [
        {
            "category": "memory",
            "suggestion": "Increase executor memory to 12GB",
            "expected_improvement": "Eliminate disk spill, ~20% faster"
        },
        {
            "category": "parallelism",
            "suggestion": "Increase number of executors to 15",
            "expected_improvement": "Better resource utilization"
        }
    ],

    "comparison": {
        "similar_jobs": [
            {
                "app_id": "app-20240102-000001",
                "performance_difference": "+15%",
                "cost_difference": "+$5"
            }
        ]
    }
}
```

**Example**:
```bash
curl http://localhost:8080/analyze/app-20240101-000001
```

---

---

## Monitoring Endpoints

### Get Monitored Applications

Get list of currently monitored Spark applications.

**Endpoint**: `GET /monitoring/applications`

**Response**:
```json
{
    "applications": [
        {
            "app_id": "app-20241210-001",
            "app_name": "ETL Pipeline - Sales Data",
            "status": "running",
            "progress": 0.65,
            "active_tasks": 24,
            "completed_tasks": 156,
            "failed_tasks": 2,
            "current_memory_mb": 12288,
            "current_cpu_percent": 78.5,
            "executors": 8,
            "duration_seconds": 1234,
            "last_updated": "2024-12-10T10:30:00Z"
        }
    ],
    "total": 2
}
```

**Example**:
```bash
curl http://localhost:8080/monitoring/applications
```

---

### Get Application Status

Get detailed status of a specific monitored application.

**Endpoint**: `GET /monitoring/applications/{app_id}`

**Path Parameters**:
- `app_id`: Spark application ID

**Response**:
```json
{
    "app_id": "app-20241210-001",
    "app_name": "ETL Pipeline - Sales Data",
    "status": "running",
    "start_time": "2024-12-10T10:00:00Z",
    "progress": 0.65,
    "active_tasks": 24,
    "completed_tasks": 156,
    "failed_tasks": 2,
    "current_memory_mb": 12288,
    "current_cpu_percent": 78.5,
    "executors": 8,
    "metrics": {
        "gc_time_percent": 8.5,
        "shuffle_read_mb": 4096,
        "shuffle_write_mb": 2048
    }
}
```

**Example**:
```bash
curl http://localhost:8080/monitoring/applications/app-20241210-001
```

---

### Get Application Metrics

Get metrics history for a monitored application.

**Endpoint**: `GET /monitoring/applications/{app_id}/metrics`

**Query Parameters**:
- `metric_names` (optional): Comma-separated list of metric names
- `since` (optional): Start time (ISO format)
- `limit` (optional): Maximum data points (default: 100)

**Response**:
```json
{
    "app_id": "app-20241210-001",
    "metrics": [
        {
            "timestamp": "2024-12-10T10:30:00Z",
            "name": "cpu_percent",
            "value": 78.5,
            "labels": {"executor_id": "all"}
        },
        {
            "timestamp": "2024-12-10T10:30:00Z",
            "name": "memory_used_mb",
            "value": 12288,
            "labels": {"executor_id": "all"}
        }
    ]
}
```

---

### Get Active Alerts

Get active alerts across all monitored applications.

**Endpoint**: `GET /monitoring/alerts`

**Query Parameters**:
- `severity` (optional): Filter by severity (critical, error, warning, info)
- `app_id` (optional): Filter by application ID
- `acknowledged` (optional): Filter by acknowledgment status (true/false)

**Response**:
```json
{
    "alerts": [
        {
            "id": "alert-001",
            "app_id": "app-20241210-001",
            "severity": "warning",
            "title": "High GC Time",
            "message": "GC time is 12.5%, exceeding 10% threshold",
            "created_at": "2024-12-10T10:25:00Z",
            "acknowledged": false
        }
    ],
    "total": 1
}
```

---

### Acknowledge Alert

Acknowledge an alert.

**Endpoint**: `POST /monitoring/alerts/{alert_id}/acknowledge`

**Response**:
```json
{
    "status": "success",
    "message": "Alert acknowledged"
}
```

---

### WebSocket: Real-Time Updates

Connect to real-time monitoring updates via WebSocket.

**Endpoint**: `WS /monitoring/ws`

**Message Types**:

*Subscribe to application*:
```json
{
    "type": "subscribe",
    "app_id": "app-20241210-001"
}
```

*Metric update (server to client)*:
```json
{
    "type": "metric",
    "app_id": "app-20241210-001",
    "timestamp": "2024-12-10T10:30:00Z",
    "metrics": {
        "cpu_percent": 78.5,
        "memory_used_mb": 12288
    }
}
```

*Alert (server to client)*:
```json
{
    "type": "alert",
    "alert": {
        "id": "alert-001",
        "severity": "warning",
        "title": "High GC Time",
        "message": "GC time exceeds threshold"
    }
}
```

---

## Auto-Tuning Endpoints

### List Tuning Sessions

Get list of tuning sessions.

**Endpoint**: `GET /tuning/sessions`

**Query Parameters**:
- `status` (optional): Filter by status (active, paused, completed, failed)
- `app_id` (optional): Filter by application ID
- `limit` (optional): Maximum results (default: 50)

**Response**:
```json
{
    "sessions": [
        {
            "session_id": "tune-1702234567890",
            "app_id": "app-20241210-001",
            "app_name": "ETL Pipeline - Sales Data",
            "strategy": "moderate",
            "target_metric": "duration",
            "status": "active",
            "iterations": 5,
            "started_at": "2024-12-10T09:00:00Z",
            "best_metric_value": 85.2,
            "adjustments_count": 3
        }
    ],
    "total": 2
}
```

---

### Start Tuning Session

Start a new auto-tuning session.

**Endpoint**: `POST /tuning/sessions`

**Request Body**:
```json
{
    "app_id": "app-20241210-001",
    "strategy": "moderate",
    "target_metric": "duration",
    "initial_config": {
        "spark.executor.instances": 10,
        "spark.executor.cores": 4,
        "spark.executor.memory": "8g"
    },
    "constraints": {
        "min_executors": 5,
        "max_executors": 20,
        "min_memory_mb": 4096,
        "max_memory_mb": 16384
    }
}
```

**Parameters**:
- `app_id` (required): Application ID to tune
- `strategy` (optional): Tuning strategy - `conservative`, `moderate`, `aggressive` (default: moderate)
- `target_metric` (optional): Metric to optimize - `duration`, `cost`, `throughput` (default: duration)
- `initial_config` (optional): Starting configuration
- `constraints` (optional): Resource constraints

**Response**:
```json
{
    "session_id": "tune-1702234567890",
    "status": "active",
    "message": "Tuning session started"
}
```

---

### Get Tuning Session

Get details of a specific tuning session.

**Endpoint**: `GET /tuning/sessions/{session_id}`

**Response**:
```json
{
    "session_id": "tune-1702234567890",
    "app_id": "app-20241210-001",
    "app_name": "ETL Pipeline",
    "strategy": "moderate",
    "target_metric": "duration",
    "status": "active",
    "iterations": 5,
    "started_at": "2024-12-10T09:00:00Z",
    "initial_config": {
        "spark.executor.instances": 10,
        "spark.executor.cores": 4
    },
    "current_config": {
        "spark.executor.instances": 8,
        "spark.executor.cores": 4,
        "spark.executor.memory": "12g"
    },
    "best_config": {
        "spark.executor.instances": 8,
        "spark.executor.memory": "12g"
    },
    "best_metric_value": 85.2,
    "adjustments": [
        {
            "parameter": "spark.executor.memory",
            "old_value": "8g",
            "new_value": "12g",
            "reason": "Memory spilling detected",
            "applied": true,
            "timestamp": "2024-12-10T09:15:00Z"
        }
    ]
}
```

---

### Pause/Resume/End Tuning Session

Control a tuning session.

**Endpoint**: `POST /tuning/sessions/{session_id}/{action}`

**Path Parameters**:
- `session_id`: Session ID
- `action`: One of `pause`, `resume`, `end`

**Response**:
```json
{
    "status": "success",
    "session_status": "paused"
}
```

---

### Get Session Adjustments

Get adjustments made during a tuning session.

**Endpoint**: `GET /tuning/sessions/{session_id}/adjustments`

**Response**:
```json
{
    "adjustments": [
        {
            "parameter": "spark.executor.memory",
            "old_value": "8g",
            "new_value": "12g",
            "reason": "Memory spilling detected (ratio: 0.15)",
            "applied": true,
            "timestamp": "2024-12-10T09:15:00Z",
            "result": "Memory spill reduced by 80%"
        }
    ]
}
```

---

## Cost Optimization Endpoints

### Estimate Cost

Estimate cost for a Spark configuration.

**Endpoint**: `POST /cost/estimate`

**Request Body**:
```json
{
    "config": {
        "spark.executor.instances": 10,
        "spark.executor.cores": 4,
        "spark.executor.memory": 8192,
        "spark.driver.memory": 4096
    },
    "duration_hours": 2.0,
    "provider": "aws",
    "region": "us-east-1"
}
```

**Response**:
```json
{
    "total_cost": 12.50,
    "breakdown": [
        {
            "resource_type": "compute",
            "cost": 8.00,
            "quantity": 40,
            "unit": "vCPU-hours"
        },
        {
            "resource_type": "memory",
            "cost": 4.50,
            "quantity": 80,
            "unit": "GB-hours"
        }
    ],
    "instance_type": "m5.xlarge",
    "cloud_provider": "aws",
    "pricing_tier": "on_demand"
}
```

---

### Optimize Configuration

Optimize a configuration for cost, duration, or balance.

**Endpoint**: `POST /cost/optimize`

**Request Body**:
```json
{
    "config": {
        "spark.executor.instances": 20,
        "spark.executor.cores": 4,
        "spark.executor.memory": 16384
    },
    "duration_hours": 2.0,
    "goal": "minimize_cost",
    "budget": null,
    "provider": "aws",
    "constraints": {
        "min_executors": 5,
        "max_executors": 30
    }
}
```

**Parameters**:
- `config` (required): Current Spark configuration
- `duration_hours` (required): Estimated job duration
- `goal` (optional): Optimization goal - `minimize_cost`, `minimize_duration`, `balance`, `budget_constraint`
- `budget` (optional): Budget limit for budget_constraint goal
- `provider` (optional): Cloud provider (aws, gcp, azure)
- `constraints` (optional): Resource constraints

**Response**:
```json
{
    "original_config": {
        "spark.executor.instances": 20,
        "spark.executor.cores": 4,
        "spark.executor.memory": 16384
    },
    "optimized_config": {
        "spark.executor.instances": 12,
        "spark.executor.cores": 4,
        "spark.executor.memory": 12288
    },
    "original_cost": 25.00,
    "optimized_cost": 15.50,
    "savings": 9.50,
    "savings_percent": 38.0,
    "recommended_instance": "m5.xlarge",
    "recommendations": [
        "Reduce executor count from 20 to 12 for better utilization",
        "Consider using spot instances for additional 60-70% savings"
    ],
    "trade_offs": [
        "Job duration may increase by 15-20% with fewer executors"
    ]
}
```

---

### Compare Cloud Providers

Compare costs across different cloud providers.

**Endpoint**: `POST /cost/compare-providers`

**Request Body**:
```json
{
    "config": {
        "spark.executor.instances": 10,
        "spark.executor.cores": 4,
        "spark.executor.memory": 8192
    },
    "duration_hours": 2.0,
    "providers": ["aws", "gcp", "azure"]
}
```

**Response**:
```json
{
    "comparisons": [
        {
            "provider": "gcp",
            "instance_type": "n1-standard-4",
            "vcpus": 4,
            "memory_gb": 15,
            "hourly_price": 0.190,
            "total_cost": 3.80
        },
        {
            "provider": "aws",
            "instance_type": "m5.xlarge",
            "vcpus": 4,
            "memory_gb": 16,
            "hourly_price": 0.192,
            "total_cost": 3.84
        },
        {
            "provider": "azure",
            "instance_type": "Standard_D4s_v3",
            "vcpus": 4,
            "memory_gb": 16,
            "hourly_price": 0.192,
            "total_cost": 3.84
        }
    ],
    "cheapest": "gcp",
    "savings_vs_most_expensive": 0.04
}
```

---

### Get Spot Strategy Recommendation

Get recommendations for using spot/preemptible instances.

**Endpoint**: `POST /cost/spot-strategy`

**Request Body**:
```json
{
    "config": {
        "spark.executor.instances": 10,
        "spark.executor.cores": 4,
        "spark.executor.memory": 8192
    },
    "duration_hours": 2.0,
    "provider": "aws",
    "fault_tolerance": 0.8
}
```

**Parameters**:
- `fault_tolerance`: How tolerant the job is to interruption (0.0-1.0)

**Response**:
```json
{
    "strategy": "full_spot",
    "on_demand_cost": 12.50,
    "spot_cost": 3.75,
    "expected_cost": 4.00,
    "expected_savings": 8.50,
    "interruption_probability": 0.05,
    "recommendation": "Job is highly fault-tolerant. Use full spot instances for 68% savings."
}
```

---

### Get Cost-Duration Frontier

Get cost-duration trade-off options.

**Endpoint**: `POST /cost/frontier`

**Request Body**:
```json
{
    "config": {
        "spark.executor.instances": 10,
        "spark.executor.cores": 4,
        "spark.executor.memory": 8192
    },
    "base_duration_hours": 2.0,
    "num_points": 5
}
```

**Response**:
```json
{
    "frontier": [
        {
            "config": {"spark.executor.instances": 5},
            "cost": 6.25,
            "estimated_duration_hours": 3.5
        },
        {
            "config": {"spark.executor.instances": 8},
            "cost": 10.00,
            "estimated_duration_hours": 2.3
        },
        {
            "config": {"spark.executor.instances": 10},
            "cost": 12.50,
            "estimated_duration_hours": 2.0
        },
        {
            "config": {"spark.executor.instances": 15},
            "cost": 18.75,
            "estimated_duration_hours": 1.5
        },
        {
            "config": {"spark.executor.instances": 20},
            "cost": 25.00,
            "estimated_duration_hours": 1.2
        }
    ]
}
```

---

#### Submit Feedback

Submit feedback on a recommendation to improve future predictions.

**Endpoint**: `POST /feedback`

**Request Body**:
```json
{
    "recommendation_id": 123,
    "actual_performance": {
        "duration_ms": 1750000,
        "cost_usd": 11.5,
        "success": true
    },
    "satisfaction_score": 0.9,
    "comments": "Recommendation worked well, job completed faster than expected"
}
```

**Parameters**:
- `recommendation_id` (required): ID of the recommendation
- `actual_performance` (optional): Actual job performance metrics
- `satisfaction_score` (optional): User satisfaction (0.0 to 1.0)
- `comments` (optional): Additional feedback text

**Response**:
```json
{
    "status": "success",
    "message": "Feedback recorded successfully",
    "feedback_id": 456
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "recommendation_id": 123,
    "satisfaction_score": 0.9
  }'
```

---

## CLI Reference

### Installation

```bash
pip install spark-resource-optimizer
```

### Global Options

```bash
spark-optimizer [OPTIONS] COMMAND [ARGS]

Options:
  --config PATH          Configuration file path
  --log-level LEVEL      Logging level (DEBUG, INFO, WARNING, ERROR)
  --help                 Show help message
  --version              Show version
```

### Commands

---

#### collect

Collect Spark job data from various sources.

**Usage**:
```bash
spark-optimizer collect [OPTIONS]
```

**Options**:
- `--source TEXT`: Source type (event-logs, history-server, metrics) [required]
- `--path TEXT`: Path or URL to data source [required]
- `--batch-size INT`: Batch size for processing (default: 100)
- `--recursive`: Recursively process directories
- `--since TEXT`: Collect jobs since date (ISO format)
- `--user TEXT`: Filter by user
- `--dry-run`: Validate without storing

**Examples**:
```bash
# Collect from event logs
spark-optimizer collect --source event-logs --path /path/to/logs

# Collect with date filter
spark-optimizer collect --source event-logs --path /path/to/logs --since 2024-01-01

# Collect from History Server
spark-optimizer collect --source history-server --path http://localhost:18080

# Dry run
spark-optimizer collect --source event-logs --path /path/to/logs --dry-run
```

---

#### recommend

Get resource recommendations for a Spark job.

**Usage**:
```bash
spark-optimizer recommend [OPTIONS]
```

**Options**:
- `--input-size FLOAT`: Input data size in GB [required]
- `--job-type TEXT`: Job type (etl, ml, sql, streaming)
- `--app-name TEXT`: Application name for similarity matching
- `--method TEXT`: Recommendation method (similarity, ml, rule-based, hybrid)
- `--max-cost FLOAT`: Maximum acceptable cost in USD
- `--verbose`: Show detailed output
- `--output FORMAT`: Output format (text, json, yaml)

**Examples**:
```bash
# Basic recommendation
spark-optimizer recommend --input-size 50 --job-type etl

# With cost constraint
spark-optimizer recommend --input-size 100 --max-cost 50

# JSON output
spark-optimizer recommend --input-size 50 --output json

# Verbose output
spark-optimizer recommend --input-size 50 --verbose
```

**Output Example**:
```
Recommended Configuration:
  Executor Cores:    4
  Executor Memory:   8192 MB
  Number of Executors: 10
  Driver Memory:     4096 MB

Predicted Performance:
  Duration:          ~30 minutes
  Estimated Cost:    $12.50

Confidence:          85%
Method:             similarity
```

---

#### analyze

Analyze a Spark job and get insights.

**Usage**:
```bash
spark-optimizer analyze [OPTIONS] APP_ID
```

**Arguments**:
- `APP_ID`: Spark application ID [required]

**Options**:
- `--output FORMAT`: Output format (text, json, yaml)
- `--detailed`: Show detailed stage-level analysis

**Examples**:
```bash
# Analyze job
spark-optimizer analyze app-20240101-000001

# Detailed analysis
spark-optimizer analyze app-20240101-000001 --detailed

# JSON output
spark-optimizer analyze app-20240101-000001 --output json
```

---

#### list-jobs

List stored Spark jobs.

**Usage**:
```bash
spark-optimizer list-jobs [OPTIONS]
```

**Options**:
- `--limit INT`: Maximum number of results (default: 50)
- `--app-name TEXT`: Filter by application name
- `--user TEXT`: Filter by user
- `--since TEXT`: Start date (ISO format)
- `--until TEXT`: End date (ISO format)
- `--status TEXT`: Filter by status (completed, failed, running)
- `--output FORMAT`: Output format (table, json, csv)

**Examples**:
```bash
# List recent jobs
spark-optimizer list-jobs --limit 10

# Filter by app name
spark-optimizer list-jobs --app-name etl

# Date range
spark-optimizer list-jobs --since 2024-01-01 --until 2024-01-31

# CSV output
spark-optimizer list-jobs --output csv > jobs.csv
```

---

#### serve

Start the REST API server.

**Usage**:
```bash
spark-optimizer serve [OPTIONS]
```

**Options**:
- `--host TEXT`: Host address (default: 0.0.0.0)
- `--port INT`: Port number (default: 8080)
- `--debug`: Enable debug mode
- `--workers INT`: Number of worker processes (default: 4)

**Examples**:
```bash
# Start server
spark-optimizer serve

# Custom port
spark-optimizer serve --port 9090

# Debug mode
spark-optimizer serve --debug
```

---

#### train

Train ML models with historical job data.

**Usage**:
```bash
spark-optimizer train [OPTIONS]
```

**Options**:
- `--method TEXT`: Model type (ml, all)
- `--test-split FLOAT`: Test set fraction (default: 0.2)
- `--evaluate`: Evaluate model after training

**Examples**:
```bash
# Train ML models
spark-optimizer train --method ml

# Train and evaluate
spark-optimizer train --method ml --evaluate
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Malformed request body or parameters |
| `MISSING_PARAMETER` | Required parameter not provided |
| `JOB_NOT_FOUND` | Specified job ID does not exist |
| `INSUFFICIENT_DATA` | Not enough historical data for recommendation |
| `MODEL_NOT_TRAINED` | ML model has not been trained |
| `COLLECTION_FAILED` | Failed to collect data from source |
| `DATABASE_ERROR` | Database operation failed |
| `INTERNAL_ERROR` | Internal server error |

## Rate Limiting

Currently, no rate limiting is enforced. This will be added in future versions.

## Versioning

API version is included in responses and can be queried via the health endpoint. Breaking changes will result in a new major version.

## SDKs

### Python SDK

```python
from spark_optimizer import SparkOptimizer

# Initialize client
client = SparkOptimizer("http://localhost:8080")

# Get recommendation
rec = client.recommend(input_size_gb=50.0, job_type="etl")
print(rec.configuration)

# List jobs
jobs = client.list_jobs(limit=10, app_name="etl")

# Analyze job
analysis = client.analyze("app-20240101-000001")
print(analysis.suggestions)
```

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/gridatek/spark-resource-optimizer/issues
- Documentation: https://spark-optimizer.readthedocs.io
