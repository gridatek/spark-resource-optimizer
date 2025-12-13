
# GCP Dataproc Integration Guide

This guide explains how to use the Spark Resource Optimizer with Google Cloud Dataproc clusters.

## Overview

The Dataproc collector enables you to:
- 📊 Collect Spark application metrics from Dataproc clusters
- 🔍 Monitor job runs and execution details
- 💰 Track costs based on GCP machine type pricing
- 🎯 Get machine type recommendations for optimal performance
- ⚡ Optimize resource configurations for production workloads
- 🔄 Support for preemptible workers and autoscaling

## Free Testing Options

### GCP Free Tier & Credits (Best Free Option)

#### Option 1: GCP Free Trial (Recommended - $300 Free Credits)
**Get $300 in free credits valid for 90 days:**

1. Visit: https://cloud.google.com/free
2. Sign up for GCP (credit card required, but won't be charged)
3. Get **$300 in credits** (90 days)
4. Create Dataproc clusters and test extensively for free

**What you can do with $300:**
- Run ~400+ hours of small Dataproc clusters
- Test thoroughly without any cost
- Ideal for learning and integration testing

#### Option 2: GCP Always Free Tier
After free trial expires, GCP offers **Always Free** tier:
- **Note:** Dataproc itself is NOT included in Always Free
- However, you get free Compute Engine usage (limited):
  - 1 f1-micro VM instance per month
  - Too small for Dataproc clusters
- ⚠️ **Not suitable for Dataproc testing**

#### Option 3: Minimal Cost Testing (After Free Credits)
To keep costs under **$0.30 for testing**:

1. **Use smallest cluster configuration:**
   ```bash
   # Example cluster
   Master: n1-standard-2 (2 cores, 7.5 GB)
   Workers: 2x n1-standard-2
   Cost: ~$0.40/hour (Compute) + ~$0.01/hour (Dataproc)
   Total: ~$0.41/hour
   ```

2. **Use preemptible workers (up to 80% cheaper):**
   ```bash
   gcloud dataproc clusters create test-cluster \
     --region=us-central1 \
     --num-workers=2 \
     --num-preemptible-workers=2 \
     --worker-machine-type=n1-standard-2 \
     --preemptible-worker-boot-disk-size=30
   ```
   - Preemptible cost: ~$0.10/hour
   - Great for testing, may be interrupted

3. **Test quickly and delete:**
   - Run the integration test (~20-30 minutes)
   - Delete cluster immediately after
   - Total cost: **~$0.15-0.30**

4. **Auto-delete idle clusters:**
   ```bash
   gcloud dataproc clusters create test-cluster \
     --max-idle=30m \
     --region=us-central1
   ```

5. **Clean up immediately:**
   ```bash
   # Delete cluster after testing
   gcloud dataproc clusters delete test-cluster --region=us-central1
   ```

#### Option 4: Test with Existing Clusters
If you already have Dataproc clusters running:
- Set `submit_jobs: false` in the GitHub Actions workflow
- Test data collection from existing jobs
- No additional cost

**Pricing Calculator:** https://cloud.google.com/products/calculator

**Pro Tip:** With GCP's $300 free credits, you can test extensively for 90 days at absolutely no cost!

## Prerequisites

### 1. Install Google Cloud Libraries

```bash
pip install google-cloud-dataproc google-cloud-monitoring
```

### 2. Set Up Authentication

#### Option A: Application Default Credentials (Recommended for Development)

```bash
# Install gcloud CLI
# https://cloud.google.com/sdk/docs/install

# Authenticate
gcloud auth application-default login
```

#### Option B: Service Account (Recommended for Production)

1. Create a service account in GCP Console
2. Download the JSON key file
3. Set environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

#### Option C: Programmatic Credentials

```python
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file(
    '/path/to/service-account-key.json'
)

collector = DataprocCollector(
    project_id="my-project",
    region="us-central1",
    credentials=credentials
)
```

### 3. Required Permissions

The service account needs these IAM permissions:
- `dataproc.clusters.list` - List clusters
- `dataproc.clusters.get` - Get cluster details
- `dataproc.jobs.list` - List jobs
- `dataproc.jobs.get` - Get job details
- `monitoring.timeSeries.list` - Read Cloud Monitoring metrics (optional)

**Recommended Role:** `roles/dataproc.viewer`

```bash
# Grant permissions
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" \
    --role="roles/dataproc.viewer"
```

## Quick Start

### Basic Usage

```python
from spark_optimizer.collectors import DataprocCollector
from spark_optimizer.storage import Database

# Initialize database
db = Database("postgresql://user:pass@localhost/spark_metrics")

# Create Dataproc collector
collector = DataprocCollector(
    project_id="my-gcp-project",
    region="us-central1"
)

# Validate configuration
if collector.validate_config():
    # Collect data from all running clusters
    jobs = collector.collect()

    # Store in database
    for job in jobs:
        db.save_job(job)

    print(f"Collected {len(jobs)} Spark applications from Dataproc")
else:
    print("Failed to validate Dataproc configuration")
```

### Collect from Specific Clusters

```python
config = {
    "cluster_names": ["prod-cluster-1", "prod-cluster-2"],
}

collector = DataprocCollector(
    project_id="my-gcp-project",
    region="us-central1",
    config=config
)
jobs = collector.collect()
```

### Filter by Labels

```python
config = {
    "cluster_labels": {
        "env": "production",
        "team": "data-engineering"
    }
}

collector = DataprocCollector(
    project_id="my-gcp-project",
    region="us-central1",
    config=config
)
jobs = collector.collect()
```

### Advanced Configuration

```python
config = {
    # Specific clusters to monitor
    "cluster_names": [],  # Empty = monitor all clusters

    # Filter by cluster labels
    "cluster_labels": {"env": "prod"},

    # Maximum number of clusters to process
    "max_clusters": 20,

    # How many days back to collect data
    "days_back": 14,

    # Enable cost tracking
    "collect_costs": True,

    # Include preemptible worker costs
    "include_preemptible": True,

    # Preemptible pricing discount (default: 0.8 = 80% discount)
    "preemptible_discount": 0.8,
}

collector = DataprocCollector(
    project_id="my-gcp-project",
    region="us-central1",
    config=config
)
```

## Features

### 1. Automatic Cluster Discovery

```python
# Monitor all active clusters in project and region
collector = DataprocCollector(
    project_id="my-gcp-project",
    region="us-central1"
)
jobs = collector.collect()
```

### 2. Cost Tracking

Automatically calculate costs based on machine type pricing:

```python
config = {"collect_costs": True}
collector = DataprocCollector(
    project_id="my-gcp-project",
    region="us-central1",
    config=config
)
jobs = collector.collect()

# Each job includes:
# - estimated_cost (in USD based on machine types and duration)
```

### 3. Machine Type Recommendations

Get GCP-specific machine type recommendations:

```python
collector = DataprocCollector(
    project_id="my-gcp-project",
    region="us-central1"
)

workload_profile = {
    "memory_intensive": False,
    "compute_intensive": True,
    "job_type": "streaming"
}

recommendation = collector.get_machine_type_recommendations(
    current_machine_type="n1-standard-8",
    workload_profile=workload_profile
)

print(f"Current: {recommendation['current_machine_type']}")
print(f"Recommended: {recommendation['recommended_machine_type']}")
print(f"Reason: {recommendation['reason']}")
print(f"Cost change: {recommendation['estimated_cost_change_percent']:.2f}%")
print(f"Preemptible: {recommendation['preemptible_recommended']}")
```

**Recommendation Logic:**
- **Memory-intensive** (ML jobs) → N2 High-memory or N1 High-memory
- **Compute-intensive** (streaming) → C2 Compute-optimized
- **Cost-optimized** (batch) → E2 Cost-optimized
- **ETL workloads** → N2 Standard (balanced)

## Supported Machine Types

### N1 Standard - General Purpose
- n1-standard-4: 4 cores, 15 GB, $0.19/hr
- n1-standard-8: 8 cores, 30 GB, $0.38/hr
- n1-standard-16: 16 cores, 60 GB, $0.76/hr
- n1-standard-32: 32 cores, 120 GB, $1.52/hr

### N1 High-Memory - Memory-Intensive Workloads
- n1-highmem-4: 4 cores, 26 GB, $0.24/hr
- n1-highmem-8: 8 cores, 52 GB, $0.47/hr
- n1-highmem-16: 16 cores, 104 GB, $0.94/hr
- n1-highmem-32: 32 cores, 208 GB, $1.87/hr

### N1 High-CPU - Compute-Intensive Workloads
- n1-highcpu-4: 4 cores, 3.6 GB, $0.14/hr
- n1-highcpu-8: 8 cores, 7.2 GB, $0.28/hr
- n1-highcpu-16: 16 cores, 14.4 GB, $0.56/hr
- n1-highcpu-32: 32 cores, 28.8 GB, $1.13/hr

### N2 - Newer Generation (Better Performance)
- n2-standard-4: 4 cores, 16 GB, $0.19/hr
- n2-standard-8: 8 cores, 32 GB, $0.39/hr
- n2-standard-16: 16 cores, 64 GB, $0.78/hr
- n2-standard-32: 32 cores, 128 GB, $1.55/hr

### N2 High-Memory
- n2-highmem-4: 4 cores, 32 GB, $0.26/hr
- n2-highmem-8: 8 cores, 64 GB, $0.52/hr
- n2-highmem-16: 16 cores, 128 GB, $1.04/hr

### E2 - Cost-Optimized
- e2-standard-4: 4 cores, 16 GB, $0.13/hr
- e2-standard-8: 8 cores, 32 GB, $0.27/hr
- e2-standard-16: 16 cores, 64 GB, $0.54/hr

### C2 - Compute-Optimized (Highest Performance)
- c2-standard-4: 4 cores, 16 GB, $0.21/hr
- c2-standard-8: 8 cores, 32 GB, $0.42/hr
- c2-standard-16: 16 cores, 64 GB, $0.85/hr

*Note: Prices are approximate for us-central1 region and may vary.*

## Integration with REST API

Use the Dataproc collector with the REST API server:

```python
from spark_optimizer.api.server import init_app
from spark_optimizer.collectors import DataprocCollector

app = init_app("postgresql://user:pass@localhost/spark_metrics")

@app.route("/api/v1/collect/dataproc", methods=["POST"])
def collect_from_dataproc():
    data = request.get_json()

    collector = DataprocCollector(
        project_id=data.get("project_id"),
        region=data.get("region", "us-central1"),
        config={
            "cluster_names": data.get("cluster_names", []),
            "collect_costs": data.get("collect_costs", True),
        }
    )

    if not collector.validate_config():
        return jsonify({"error": "Invalid Dataproc configuration"}), 400

    jobs = collector.collect()

    for job in jobs:
        db.save_job(job)

    return jsonify({
        "success": True,
        "collected": len(jobs),
        "project": data.get("project_id"),
        "region": data.get("region")
    })
```

## Testing Integration with GitHub Actions

The project includes a manual GitHub Actions workflow to test Dataproc integration with real data.

### Running the Integration Test

1. **Navigate to Actions** in your GitHub repository
2. **Select "GCP Dataproc Integration Test"** workflow
3. **Click "Run workflow"** and provide:
   - **Project ID**: Your GCP project ID
   - **Region**: Your Dataproc region (e.g., `us-central1`)
   - **Cluster Name**: Your Dataproc cluster name (e.g., `my-cluster`)
   - **GCS Bucket**: Bucket for uploading jobs (e.g., `my-dataproc-bucket`)
   - **Max Clusters**: Maximum clusters to collect from (default: 5)
   - **Submit Jobs**: Whether to submit sample jobs (default: true)

### Required GitHub Secrets

Configure this secret in your repository settings:

- `GCP_CREDENTIALS`: Your GCP service account JSON key (base64 encoded or raw JSON)

### What the Workflow Does

1. **Uploads Sample Jobs** - Uploads Spark jobs from `spark-jobs/` to GCS
2. **Submits Jobs** - Runs 3 sample Spark applications via `gcloud dataproc jobs submit`:
   - Simple WordCount
   - Inefficient Job (demonstrates optimization opportunities)
   - Memory Intensive Job
3. **Waits for Completion** - Monitors job status until finished
4. **Collects Data** - Uses DataprocCollector to gather metrics
5. **Saves to Database** - Stores results in SQLite
6. **Uploads Artifact** - Database file available for download

### Example Output

```
✓ Jobs uploaded to gs://my-bucket/spark-jobs/
✓ Submitted Simple WordCount: job-abc123
✓ Submitted Inefficient Job: job-def456
✓ Submitted Memory Intensive Job: job-ghi789
✓ All jobs finished
✓ Collected data for 3 applications from cluster my-cluster
✓ Saved 3/3 jobs to database
Total applications in database: 3
```

### Using Without Job Submission

To test data collection from existing jobs without submitting new ones:

1. Set **Submit Jobs** to `false`
2. Provide existing **Cluster Name** with completed jobs
3. Workflow will only collect and analyze existing data

## Automated Collection

### Using Cron

Collect Dataproc metrics every hour:

```bash
# Add to crontab
0 * * * * /usr/bin/python3 /path/to/collect_dataproc.py
```

**collect_dataproc.py:**
```python
#!/usr/bin/env python3
import os
from spark_optimizer.collectors import DataprocCollector
from spark_optimizer.storage import Database

db = Database(os.environ["DATABASE_URL"])
collector = DataprocCollector(
    project_id=os.environ["GCP_PROJECT_ID"],
    region=os.environ.get("GCP_REGION", "us-central1")
)

jobs = collector.collect()
for job in jobs:
    db.save_job(job)

print(f"Collected {len(jobs)} jobs at {datetime.now()}")
```

### Using Cloud Scheduler

Create a Cloud Function or Cloud Run service and trigger it with Cloud Scheduler:

```yaml
# cloud-function-config.yaml
name: spark-metrics-collector
runtime: python39
entry_point: collect_metrics
environment_variables:
  DATABASE_URL: postgresql://...
  GCP_PROJECT_ID: my-project
  GCP_REGION: us-central1
```

```python
# main.py
from spark_optimizer.collectors import DataprocCollector
from spark_optimizer.storage import Database
import os

def collect_metrics(request):
    db = Database(os.environ["DATABASE_URL"])
    collector = DataprocCollector(
        project_id=os.environ["GCP_PROJECT_ID"],
        region=os.environ["GCP_REGION"]
    )

    jobs = collector.collect()
    for job in jobs:
        db.save_job(job)

    return {"collected": len(jobs)}, 200
```

## Multi-Region Support

Collect from multiple regions:

```python
regions = ["us-central1", "us-west1", "europe-west1"]
all_jobs = []

for region in regions:
    collector = DataprocCollector(
        project_id="my-project",
        region=region
    )
    jobs = collector.collect()
    all_jobs.extend(jobs)
    print(f"Region {region}: {len(jobs)} jobs")

print(f"Total: {len(all_jobs)} jobs")
```

## Troubleshooting

### Authentication Errors

**Problem:** `403 Forbidden` or `Permission Denied`

**Solution:**
1. Verify service account has `dataproc.viewer` role
2. Check GOOGLE_APPLICATION_CREDENTIALS is set correctly
3. Ensure ADC is configured with `gcloud auth application-default login`

### No Data Collected

**Problem:** `collect()` returns empty list

**Solution:**
1. Verify clusters are in RUNNING state
2. Check `days_back` configuration
3. Ensure jobs exist in the time range
4. Verify region is correct

### Rate Limiting

**Problem:** `429 Too Many Requests`

**Solution:**
1. Reduce `max_clusters` value
2. Increase collection interval
3. Use specific `cluster_names` instead of listing all

## Best Practices

1. **Use Service Accounts** - More secure than user credentials
2. **Store Credentials Securely** - Use Secret Manager or environment variables
3. **Filter Clusters** - Specify `cluster_names` or `cluster_labels` to avoid unnecessary API calls
4. **Monitor Costs** - Enable `collect_costs` to track spending
5. **Use Preemptible Workers** - For batch jobs to reduce costs by up to 80%
6. **Regional Deployment** - Run collector in same region as clusters
7. **Caching** - Store results in database to reduce repeated API calls

## Security Best Practices

### Store Credentials Securely

**Secret Manager:**
```bash
# Store service account key in Secret Manager
gcloud secrets create dataproc-collector-key \
    --data-file=/path/to/service-account-key.json

# Access in code
from google.cloud import secretmanager
client = secretmanager.SecretManagerServiceClient()
name = f"projects/PROJECT_ID/secrets/dataproc-collector-key/versions/latest"
response = client.access_secret_version(request={"name": name})
credentials_json = response.payload.data.decode("UTF-8")
```

**Environment Variables:**
```bash
export GCP_PROJECT_ID="my-project"
export GCP_REGION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
export DATABASE_URL="postgresql://user:pass@localhost/spark_metrics"
```

### Least Privilege Access

Create service accounts with minimal required permissions:
- Read-only access to Dataproc clusters and jobs
- Read-only access to Cloud Monitoring (optional)
- No write/admin permissions

```bash
# Create custom role with minimal permissions
gcloud iam roles create dataprocMetricsCollector \
    --project=PROJECT_ID \
    --title="Dataproc Metrics Collector" \
    --description="Minimal permissions for collecting Dataproc metrics" \
    --permissions=dataproc.clusters.list,dataproc.clusters.get,dataproc.jobs.list,dataproc.jobs.get
```

## Cost Optimization Tips

1. **Use Preemptible Workers** - Save up to 80% on worker node costs
2. **Right-size Clusters** - Use recommendations to select appropriate machine types
3. **Enable Autoscaling** - Automatically adjust cluster size based on workload
4. **Use E2 Machines** - For batch jobs that don't need maximum performance
5. **Monitor Idle Time** - Identify clusters running without active jobs
6. **Regional Pricing** - Some regions are cheaper than others

## Next Steps

- [Set up automated recommendations](./RECOMMENDATIONS.md)
- [Configure cost optimization](./COST_OPTIMIZATION.md)
- [Integrate with AWS EMR](./AWS_EMR_INTEGRATION.md)
- [Integrate with Databricks](./DATABRICKS_INTEGRATION.md)
- [Monitor multiple cloud providers](./MULTI_CLOUD.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/gridatek/spark-optimizer/issues
- Documentation: https://github.com/gridatek/spark-optimizer/docs
