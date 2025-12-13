

# Databricks Integration Guide

This guide explains how to use the Spark Resource Optimizer with Databricks workspaces.

## Overview

The Databricks collector enables you to:
- 📊 Collect Spark application metrics from Databricks clusters
- 🔍 Monitor job runs and execution details
- 💰 Track costs using DBU (Databricks Units) consumption
- 🎯 Get Databricks-specific cluster type recommendations
- ⚡ Optimize resource configurations for production workloads

## Free Testing Options

### Databricks Free Trial & Community Edition

#### Option 1: Databricks Free Trial (Recommended for Testing)
Get **14 days of free access** to a full Databricks workspace:

**AWS Databricks:**
1. Visit: https://databricks.com/try-databricks
2. Sign up for free trial
3. Get $200 in credits or 14-day trial
4. Full API access for testing the integration

**Azure Databricks:**
1. Visit: https://azure.microsoft.com/free/
2. Get $200 Azure credits (30 days)
3. Create Databricks workspace
4. Test with free credits

**GCP Databricks:**
1. Visit: https://cloud.google.com/free
2. Get $300 GCP credits (90 days)
3. Create Databricks workspace
4. Test with free credits

#### Option 2: Databricks Community Edition (Limited)
**Free forever, but with limitations:**
- **Cannot use API** (required for this integration)
- No custom clusters
- Single-node cluster only
- ⚠️ **Not suitable for testing this integration workflow**

#### Option 3: Minimal Cost Testing
To keep costs under **$0.50 for testing**:

1. **Use smallest cluster:**
   ```python
   # Example: Azure
   Cluster Type: Standard_DS3_v2 (4 cores, 14 GB)
   Cost: ~$0.19/hr (VM) + ~$0.30/hr (DBU) = ~$0.49/hr

   # Example: AWS
   Cluster Type: i3.xlarge (4 cores, 30.5 GB)
   Cost: ~$0.31/hr (EC2) + ~$0.23/hr (DBU) = ~$0.54/hr
   ```

2. **Test quickly and terminate:**
   - Run the integration test (~20-30 minutes)
   - Terminate cluster immediately after
   - Total cost: **~$0.25-0.50**

3. **Use cluster policies for auto-termination:**
   ```json
   {
     "autotermination_minutes": {
       "type": "fixed",
       "value": 30
     }
   }
   ```

4. **Clean up immediately:**
   ```bash
   # Terminate cluster via UI or CLI
   databricks clusters delete --cluster-id YOUR_CLUSTER_ID
   ```

#### Option 4: Test with Existing Clusters
If you already have Databricks clusters running:
- Set `submit_jobs: false` in the GitHub Actions workflow
- Test data collection from existing job runs
- No additional cost

**Pricing Calculator:** https://databricks.com/product/pricing

## Prerequisites

### 1. Install requests library

```bash
# requests is usually already installed, but if not:
pip install requests
```

### 2. Get Databricks Access Token

**Generate a Personal Access Token (PAT):**

1. Log in to your Databricks workspace
2. Click on your username in the top right
3. Select "User Settings"
4. Go to "Access Tokens" tab
5. Click "Generate New Token"
6. Copy and save the token securely

**Alternative: Service Principal (Production)**

For production deployments, use a service principal with OAuth:
```bash
# Azure Databricks
az ad sp create-for-rbac --name spark-optimizer-sp

# AWS Databricks
# Use the Databricks CLI to create a service principal
databricks service-principals create --display-name spark-optimizer-sp
```

### 3. Required Permissions

The token/service principal needs these permissions:
- `clusters/list` - List clusters
- `clusters/get` - Get cluster details
- `jobs/list` - List jobs
- `jobs/runs/list` - List job runs
- `sql/read` - Read SQL analytics (optional)

## Quick Start

### Basic Usage

```python
from spark_optimizer.collectors import DatabricksCollector
from spark_optimizer.storage import Database

# Initialize database
db = Database("postgresql://user:pass@localhost/spark_metrics")

# Create Databricks collector
collector = DatabricksCollector(
    workspace_url="https://dbc-xxx.cloud.databricks.com",
    token="dapi1234567890abcdef"
)

# Validate configuration
if collector.validate_config():
    # Collect data from all running clusters
    jobs = collector.collect()

    # Store in database
    for job in jobs:
        db.save_job(job)

    print(f"Collected {len(jobs)} Spark applications from Databricks")
else:
    print("Failed to validate Databricks configuration")
```

### Collect from Specific Clusters

```python
config = {
    "cluster_ids": ["0123-456789-abc123", "0123-456789-xyz789"],
}

collector = DatabricksCollector(
    workspace_url="https://dbc-xxx.cloud.databricks.com",
    token="dapi1234567890abcdef",
    config=config
)
jobs = collector.collect()
```

### Advanced Configuration

```python
config = {
    # Specific clusters to monitor
    "cluster_ids": [],  # Empty = monitor all clusters

    # Cluster states to include
    "cluster_states": ["RUNNING", "PENDING", "RESTARTING"],

    # Maximum number of clusters to process
    "max_clusters": 30,

    # How many days back to collect data
    "days_back": 14,

    # Include SQL analytics endpoints
    "collect_sql_analytics": True,

    # Enable DBU cost tracking
    "collect_costs": True,

    # DBU price in USD (varies by region and cloud provider)
    "dbu_price": 0.40,  # Standard pricing

    # Alternative: Basic Auth (not recommended)
    "username": "user@example.com",
    "password": "password"
}

collector = DatabricksCollector(
    workspace_url="https://dbc-xxx.cloud.databricks.com",
    token="dapi1234567890abcdef",
    config=config
)
```

## Features

### 1. Automatic Cluster Discovery

```python
# Monitor all active clusters in workspace
collector = DatabricksCollector(
    workspace_url="https://dbc-xxx.cloud.databricks.com",
    token="dapi1234567890abcdef"
)
jobs = collector.collect()
```

### 2. DBU Cost Tracking

Automatically calculate costs based on DBU consumption:

```python
config = {"collect_costs": True, "dbu_price": 0.40}
collector = DatabricksCollector(
    workspace_url="https://dbc-xxx.cloud.databricks.com",
    token="dapi1234567890abcdef",
    config=config
)
jobs = collector.collect()

# Each job includes:
# - estimated_cost (in USD based on DBU consumption)
```

### 3. Cluster Type Recommendations

Get Databricks-specific cluster recommendations:

```python
collector = DatabricksCollector(
    workspace_url="https://dbc-xxx.cloud.databricks.com",
    token="dapi1234567890abcdef"
)

workload_profile = {
    "memory_intensive": False,
    "io_intensive": True,
    "job_type": "streaming"
}

recommendation = collector.get_cluster_recommendations(
    current_cluster_type="Standard_DS3_v2",
    workload_profile=workload_profile
)

print(f"Current: {recommendation['current_cluster_type']}")
print(f"Recommended: {recommendation['recommended_cluster_type']}")
print(f"Reason: {recommendation['reason']}")
print(f"Cost change: {recommendation['estimated_cost_change_percent']:.2f}%")
print(f"Autoscaling: {recommendation['autoscaling_recommended']}")
```

**Recommendation Logic:**
- **Memory-intensive** (ML jobs) → E-series (Azure) or r5d (AWS)
- **I/O-intensive** (streaming) → DS-series (Azure) or i3 (AWS)
- **SQL analytics** → F-series (Azure) or i3 (AWS)
- **ETL workloads** → DS-series (Azure) or i3 (AWS)

## Supported Cluster Types

### Azure Databricks

#### General Purpose (DS-series)
- Standard_DS3_v2: 4 cores, 14 GB, 0.75 DBU/hr
- Standard_DS4_v2: 8 cores, 28 GB, 1.5 DBU/hr
- Standard_DS5_v2: 16 cores, 56 GB, 3.0 DBU/hr

#### Memory Optimized (E-series)
- Standard_E4s_v3: 4 cores, 32 GB, 1.0 DBU/hr
- Standard_E8s_v3: 8 cores, 64 GB, 2.0 DBU/hr
- Standard_E16s_v3: 16 cores, 128 GB, 4.0 DBU/hr

#### Compute Optimized (F-series)
- Standard_F4s: 4 cores, 8 GB, 0.6 DBU/hr
- Standard_F8s: 8 cores, 16 GB, 1.2 DBU/hr
- Standard_F16s: 16 cores, 32 GB, 2.4 DBU/hr

### AWS Databricks

#### Storage Optimized (i3)
- i3.xlarge: 4 cores, 30.5 GB, 0.75 DBU/hr
- i3.2xlarge: 8 cores, 61 GB, 1.5 DBU/hr
- i3.4xlarge: 16 cores, 122 GB, 3.0 DBU/hr

#### Memory Optimized (r5d)
- r5d.xlarge: 4 cores, 32 GB, 0.9 DBU/hr
- r5d.2xlarge: 8 cores, 64 GB, 1.8 DBU/hr
- r5d.4xlarge: 16 cores, 128 GB, 3.6 DBU/hr

*Note: DBU prices are approximate and vary by region and commitment level.*

## Integration with REST API

Use the Databricks collector with the REST API server:

```python
from spark_optimizer.api.server import init_app
from spark_optimizer.collectors import DatabricksCollector

app = init_app("postgresql://user:pass@localhost/spark_metrics")

@app.route("/api/v1/collect/databricks", methods=["POST"])
def collect_from_databricks():
    data = request.get_json()

    collector = DatabricksCollector(
        workspace_url=data.get("workspace_url"),
        token=data.get("token"),
        config={
            "cluster_ids": data.get("cluster_ids", []),
            "collect_costs": data.get("collect_costs", True),
            "dbu_price": data.get("dbu_price", 0.40),
        }
    )

    if not collector.validate_config():
        return jsonify({"error": "Invalid Databricks configuration"}), 400

    jobs = collector.collect()

    for job in jobs:
        db.save_job(job)

    return jsonify({
        "success": True,
        "collected": len(jobs),
        "workspace": data.get("workspace_url")
    })
```

## Testing Integration with GitHub Actions

The project includes a manual GitHub Actions workflow to test Databricks integration with real data.

### Running the Integration Test

1. **Navigate to Actions** in your GitHub repository
2. **Select "Databricks Integration Test"** workflow
3. **Click "Run workflow"** and provide:
   - **Workspace URL**: Your Databricks workspace URL (e.g., `https://dbc-xxx.cloud.databricks.com`)
   - **Cluster ID**: Your Databricks cluster ID (e.g., `0123-456789-abc123`)
   - **DBFS Path**: Path for uploading jobs (default: `/FileStore/spark-jobs`)
   - **Max Jobs**: Maximum jobs to collect (default: 10)
   - **Submit Jobs**: Whether to submit sample jobs (default: true)

### Required GitHub Secrets

Configure this secret in your repository settings:

- `DATABRICKS_TOKEN`: Your Databricks personal access token

### What the Workflow Does

1. **Uploads Sample Jobs** - Uploads Spark jobs from `spark-jobs/` to DBFS
2. **Submits Jobs** - Runs 3 sample Spark applications via Jobs API:
   - Simple WordCount
   - Inefficient Job (demonstrates optimization opportunities)
   - Memory Intensive Job
3. **Waits for Completion** - Monitors run status until finished
4. **Collects Data** - Uses DatabricksCollector to gather metrics
5. **Saves to Database** - Stores results in SQLite
6. **Uploads Artifact** - Database file available for download

### Example Output

```
✓ Jobs uploaded to DBFS:/FileStore/spark-jobs
✓ Submitted Simple WordCount: Run ID 123
✓ Submitted Inefficient Job: Run ID 456
✓ Submitted Memory Intensive Job: Run ID 789
✓ All jobs finished
✓ Collected 3 applications from cluster 0123-456789-abc123
✓ Saved 3/3 jobs to database
Total applications in database: 3
```

### Using Without Job Submission

To test data collection from existing jobs without submitting new ones:

1. Set **Submit Jobs** to `false`
2. Provide existing **Cluster ID** with completed runs
3. Workflow will only collect and analyze existing data

## Automated Collection

### Using Cron

Collect Databricks metrics every hour:

```bash
# Add to crontab
0 * * * * /usr/bin/python3 /path/to/collect_databricks.py
```

**collect_databricks.py:**
```python
#!/usr/bin/env python3
import os
from spark_optimizer.collectors import DatabricksCollector
from spark_optimizer.storage import Database

db = Database(os.environ["DATABASE_URL"])
collector = DatabricksCollector(
    workspace_url=os.environ["DATABRICKS_WORKSPACE_URL"],
    token=os.environ["DATABRICKS_TOKEN"]
)

jobs = collector.collect()
for job in jobs:
    db.save_job(job)

print(f"Collected {len(jobs)} jobs at {datetime.now()}")
```

### Using Databricks Jobs

Run collection as a Databricks job itself:

```python
# In a Databricks notebook
%pip install spark-optimizer

from spark_optimizer.collectors import DatabricksCollector
from spark_optimizer.storage import Database

# Use dbutils to get secrets
workspace_url = dbutils.secrets.get("spark-optimizer", "workspace-url")
token = dbutils.secrets.get("spark-optimizer", "token")
db_url = dbutils.secrets.get("spark-optimizer", "database-url")

db = Database(db_url)
collector = DatabricksCollector(workspace_url=workspace_url, token=token)

jobs = collector.collect()
for job in jobs:
    db.save_job(job)

print(f"✅ Collected {len(jobs)} jobs")
```

## Troubleshooting

### Authentication Errors

**Problem:** `401 Unauthorized`

**Solution:**
1. Verify token is valid and not expired
2. Check token has required permissions
3. Ensure workspace URL is correct

### No Data Collected

**Problem:** `collect()` returns empty list

**Solution:**
1. Verify clusters are in RUNNING/PENDING state
2. Check `days_back` configuration
3. Ensure clusters have job runs
4. Verify permissions to access job runs

### Rate Limiting

**Problem:** `429 Too Many Requests`

**Solution:**
1. Reduce `max_clusters` value
2. Increase delay between requests
3. Use specific `cluster_ids` instead of listing all

## Best Practices

1. **Use Service Principals** - More secure than personal access tokens
2. **Store Tokens Securely** - Use secret managers (Databricks Secrets, Azure Key Vault, AWS Secrets Manager)
3. **Filter Clusters** - Specify `cluster_ids` to avoid unnecessary API calls
4. **Monitor Costs** - Enable `collect_costs` to track spending
5. **Regional Deployment** - Deploy collectors in same region as workspace
6. **Rate Limiting** - Use `max_clusters` to avoid API throttling
7. **Caching** - Store results in database to reduce repeated API calls

## Security Best Practices

### Store Credentials Securely

**Databricks Secrets:**
```bash
# Create secret scope
databricks secrets create-scope --scope spark-optimizer

# Add secrets
databricks secrets put --scope spark-optimizer --key token
databricks secrets put --scope spark-optimizer --key database-url
```

**Environment Variables:**
```bash
export DATABRICKS_WORKSPACE_URL="https://dbc-xxx.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdef"
export DATABASE_URL="postgresql://user:pass@localhost/spark_metrics"
```

### Least Privilege Access

Create service principals with minimal required permissions:
- Read-only access to clusters
- Read-only access to jobs
- No write/admin permissions

## Cost Optimization Tips

1. **Track DBU Consumption** - Monitor which jobs consume most DBUs
2. **Right-size Clusters** - Use recommendations to select appropriate node types
3. **Enable Autoscaling** - Automatically adjust cluster size
4. **Use Spot/Preemptible** - Reduce costs with spot instances
5. **Monitor Idle Time** - Identify clusters running without active jobs

## Next Steps

- [Set up automated recommendations](./RECOMMENDATIONS.md)
- [Configure cost optimization](./COST_OPTIMIZATION.md)
- [Integrate with AWS EMR](./AWS_EMR_INTEGRATION.md)
- [Monitor multiple workspaces](./MULTI_CLOUD.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/gridatek/spark-optimizer/issues
- Documentation: https://github.com/gridatek/spark-optimizer/docs
