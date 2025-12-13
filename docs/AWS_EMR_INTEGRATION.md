# AWS EMR Integration Guide

This guide explains how to use the Spark Resource Optimizer with AWS EMR (Elastic MapReduce) clusters.

## Overview

The EMR collector enables you to:
- 📊 Collect Spark application metrics from EMR clusters
- 📈 Monitor CloudWatch metrics for detailed performance insights
- 💰 Track costs using AWS Cost Explorer
- 🎯 Get EMR-specific instance type recommendations
- ⚡ Optimize resource configurations for production EMR workloads

## Free Testing Options

### AWS Free Tier & Cost Optimization

**Important:** AWS EMR is **not included** in the AWS Free Tier. However, you can minimize costs:

#### Option 1: AWS Free Trial (New Accounts)
- New AWS accounts may receive promotional credits
- Check AWS Activate or AWS Educate programs for credits

#### Option 2: Minimal Cost Testing
To keep costs under **$0.50 for testing**:

1. **Use smallest instance types:**
   ```bash
   # Example cluster configuration
   Instance Type: m5.xlarge (master + 2 core nodes)
   Estimated cost: ~$0.72/hour
   ```

2. **Test quickly and terminate:**
   - Run the integration test (~20-30 minutes)
   - Terminate cluster immediately after
   - Total cost: **~$0.30-0.50**

3. **Use spot instances (up to 90% cheaper):**
   - Enable spot instances for core and task nodes
   - Not recommended for production, but great for testing

4. **Clean up immediately:**
   ```bash
   # Terminate cluster after testing
   aws emr terminate-clusters --cluster-ids j-XXXXXXXXXXXXX
   ```

#### Option 3: Test with Existing Clusters
If you already have EMR clusters running:
- Set `submit_jobs: false` in the GitHub Actions workflow
- Test data collection from existing jobs
- No additional cost

**Cost Calculator:** https://calculator.aws/#/addService/EMR

## Prerequisites

### 1. Install boto3

```bash
# Install with AWS support
pip install spark-optimizer[aws]

# Or install boto3 directly
pip install boto3
```

### 2. Configure AWS Credentials

The EMR collector uses boto3, which supports multiple authentication methods:

**Option A: AWS CLI Configuration (Recommended)**
```bash
aws configure
```

**Option B: Environment Variables**
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

**Option C: IAM Role (for EC2/ECS)**
If running on AWS infrastructure, use IAM roles for automatic credential management.

### 3. Required IAM Permissions

Create an IAM policy with these permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "emr:ListClusters",
        "emr:DescribeCluster",
        "emr:ListSteps",
        "emr:DescribeStep",
        "cloudwatch:GetMetricStatistics",
        "ce:GetCostAndUsage",
        "s3:GetObject"
      ],
      "Resource": "*"
    }
  ]
}
```

## Quick Start

### Basic Usage

```python
from spark_optimizer.collectors import EMRCollector
from spark_optimizer.storage import Database

# Initialize database
db = Database("postgresql://user:pass@localhost/spark_metrics")

# Create EMR collector
collector = EMRCollector(region_name="us-west-2")

# Validate configuration
if collector.validate_config():
    # Collect data from all running EMR clusters
    jobs = collector.collect()

    # Store in database
    for job in jobs:
        db.save_job(job)

    print(f"Collected {len(jobs)} Spark applications from EMR")
else:
    print("Failed to validate EMR configuration")
```

### Collect from Specific Clusters

```python
config = {
    "cluster_ids": ["j-1234567890ABC", "j-0987654321XYZ"],
}

collector = EMRCollector(region_name="us-east-1", config=config)
jobs = collector.collect()
```

### Advanced Configuration

```python
config = {
    # Specific clusters to monitor
    "cluster_ids": [],  # Empty list = monitor all clusters

    # Cluster states to include
    "cluster_states": ["RUNNING", "WAITING", "TERMINATING"],

    # Maximum number of clusters to process
    "max_clusters": 20,

    # How many days back to look for clusters
    "days_back": 14,

    # Enable CloudWatch metrics collection
    "collect_cloudwatch": True,

    # Enable cost data collection
    "collect_costs": True,

    # Optional: Explicit AWS credentials
    "aws_access_key_id": "YOUR_KEY",
    "aws_secret_access_key": "YOUR_SECRET",
    "aws_session_token": "YOUR_TOKEN",  # For temporary credentials
}

collector = EMRCollector(region_name="eu-west-1", config=config)
```

## Features

### 1. Automatic Cluster Discovery

The collector automatically discovers EMR clusters in your AWS account:

```python
# Monitor all running clusters
collector = EMRCollector()
jobs = collector.collect()
```

### 2. CloudWatch Metrics Integration

Get detailed performance metrics from CloudWatch:

```python
config = {"collect_cloudwatch": True}
collector = EMRCollector(config=config)
jobs = collector.collect()

# Each job will include:
# - avg_cpu_utilization
# - avg_memory_available_mb
```

### 3. Cost Tracking

Automatically calculate cluster costs:

```python
config = {"collect_costs": True}
collector = EMRCollector(config=config)
jobs = collector.collect()

# Each job will include:
# - estimated_cost (in USD)
```

### 4. Instance Type Recommendations

Get EMR-specific instance type recommendations:

```python
collector = EMRCollector()

workload_profile = {
    "cpu_utilization": 75,
    "memory_utilization": 45,
    "job_type": "etl"
}

recommendation = collector.get_instance_type_recommendations(
    current_instance_type="m5.2xlarge",
    workload_profile=workload_profile
)

print(f"Current: {recommendation['current_instance_type']}")
print(f"Recommended: {recommendation['recommended_instance_type']}")
print(f"Reason: {recommendation['reason']}")
print(f"Cost change: {recommendation['estimated_cost_change_percent']:.2f}%")
```

**Recommendation Logic:**
- **Memory-intensive workloads** (>70% memory usage or ML jobs) → r5 instances
- **CPU-intensive workloads** (>70% CPU usage or compute jobs) → c5 instances
- **Balanced workloads** (ETL, general processing) → m5 instances

## Supported Instance Types

The collector includes pricing and specifications for common EMR instance types:

### General Purpose (M5)
- m5.xlarge: 4 vCPU, 16 GB RAM, $0.192/hr
- m5.2xlarge: 8 vCPU, 32 GB RAM, $0.384/hr
- m5.4xlarge: 16 vCPU, 64 GB RAM, $0.768/hr
- m5.8xlarge: 32 vCPU, 128 GB RAM, $1.536/hr
- m5.12xlarge: 48 vCPU, 192 GB RAM, $2.304/hr
- m5.16xlarge: 64 vCPU, 256 GB RAM, $3.072/hr

### Memory Optimized (R5)
- r5.xlarge: 4 vCPU, 32 GB RAM, $0.252/hr
- r5.2xlarge: 8 vCPU, 64 GB RAM, $0.504/hr
- r5.4xlarge: 16 vCPU, 128 GB RAM, $1.008/hr
- r5.8xlarge: 32 vCPU, 256 GB RAM, $2.016/hr
- r5.12xlarge: 48 vCPU, 384 GB RAM, $3.024/hr

### Compute Optimized (C5)
- c5.xlarge: 4 vCPU, 8 GB RAM, $0.170/hr
- c5.2xlarge: 8 vCPU, 16 GB RAM, $0.340/hr
- c5.4xlarge: 16 vCPU, 32 GB RAM, $0.680/hr
- c5.9xlarge: 36 vCPU, 72 GB RAM, $1.530/hr

*Note: Prices are for us-east-1 region and may vary by region and over time.*

## Integration with REST API

Use the EMR collector with the REST API server:

```python
from spark_optimizer.api.server import init_app
from spark_optimizer.collectors import EMRCollector

# Initialize the app with database
app = init_app("postgresql://user:pass@localhost/spark_metrics")

# In your collection endpoint:
@app.route("/api/v1/collect/emr", methods=["POST"])
def collect_from_emr():
    data = request.get_json()

    collector = EMRCollector(
        region_name=data.get("region", "us-east-1"),
        config={
            "cluster_ids": data.get("cluster_ids", []),
            "collect_cloudwatch": data.get("collect_cloudwatch", True),
            "collect_costs": data.get("collect_costs", True),
        }
    )

    if not collector.validate_config():
        return jsonify({"error": "Invalid AWS configuration"}), 400

    jobs = collector.collect()

    # Store jobs in database
    for job in jobs:
        db.save_job(job)

    return jsonify({
        "success": True,
        "collected": len(jobs),
        "region": data.get("region", "us-east-1")
    })
```

## Testing Integration with GitHub Actions

The project includes a manual GitHub Actions workflow to test EMR integration with real data.

### Running the Integration Test

1. **Navigate to Actions** in your GitHub repository
2. **Select "AWS EMR Integration Test"** workflow
3. **Click "Run workflow"** and provide:
   - **AWS Region**: Your EMR cluster region (e.g., `us-west-2`)
   - **Cluster ID**: Your EMR cluster ID (e.g., `j-1234567890ABC`)
   - **S3 Bucket**: Bucket for uploading Spark jobs (e.g., `my-emr-bucket`)
   - **Max Clusters**: Maximum clusters to collect from (default: 5)
   - **Submit Jobs**: Whether to submit sample jobs (default: true)

### Required GitHub Secrets

Configure these secrets in your repository settings:

- `AWS_ACCESS_KEY_ID`: Your AWS access key
- `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key

### What the Workflow Does

1. **Uploads Sample Jobs** - Uploads Spark jobs from `spark-jobs/` to S3
2. **Submits Jobs** - Runs 3 sample Spark applications:
   - Simple WordCount
   - Inefficient Job (demonstrates optimization opportunities)
   - Memory Intensive Job
3. **Waits for Completion** - Monitors job status until finished
4. **Collects Data** - Uses EMRCollector to gather metrics
5. **Saves to Database** - Stores results in SQLite
6. **Uploads Artifact** - Database file available for download

### Example Output

```
✓ Jobs uploaded to s3://my-bucket/spark-jobs/
✓ Submitted Simple WordCount: s-ABC123
✓ Submitted Inefficient Job: s-DEF456
✓ Submitted Memory Intensive Job: s-GHI789
✓ All jobs finished
✓ Collected data for 3 applications
✓ Saved 3/3 jobs to database
Total applications in database: 3
```

### Using Without Job Submission

To test data collection from existing jobs without submitting new ones:

1. Set **Submit Jobs** to `false`
2. Provide existing **Cluster ID** with completed jobs
3. Workflow will only collect and analyze existing data

## Automated Collection

### Using Cron

Collect EMR metrics every hour:

```bash
# Add to crontab
0 * * * * /usr/bin/python3 /path/to/collect_emr.py
```

**collect_emr.py:**
```python
#!/usr/bin/env python3
from spark_optimizer.collectors import EMRCollector
from spark_optimizer.storage import Database

db = Database("postgresql://user:pass@localhost/spark_metrics")
collector = EMRCollector(region_name="us-west-2")

jobs = collector.collect()
for job in jobs:
    db.save_job(job)

print(f"Collected {len(jobs)} jobs at {datetime.now()}")
```

### Using AWS Lambda

Deploy as a Lambda function for serverless collection:

```python
import json
from spark_optimizer.collectors import EMRCollector
from spark_optimizer.storage import Database

def lambda_handler(event, context):
    db = Database(os.environ["DATABASE_URL"])

    collector = EMRCollector(
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )

    jobs = collector.collect()
    for job in jobs:
        db.save_job(job)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "collected": len(jobs),
            "timestamp": datetime.now().isoformat()
        })
    }
```

## Troubleshooting

### Authentication Errors

**Problem:** `botocore.exceptions.NoCredentialsError`

**Solution:**
1. Verify AWS credentials are configured:
   ```bash
   aws sts get-caller-identity
   ```
2. Check environment variables are set
3. Ensure IAM role is attached (if running on AWS)

### Permission Denied

**Problem:** `botocore.exceptions.ClientError: An error occurred (AccessDenied)`

**Solution:**
1. Verify IAM permissions listed above
2. Check AWS region is correct
3. Ensure cluster IDs exist and are accessible

### No Data Collected

**Problem:** `collect()` returns empty list

**Solution:**
1. Verify clusters are in RUNNING or WAITING state
2. Check `days_back` configuration
3. Ensure clusters have completed steps
4. Verify region matches cluster location

### CloudWatch Metrics Missing

**Problem:** No CloudWatch metrics in collected data

**Solution:**
1. Ensure `collect_cloudwatch: True` in config
2. Verify CloudWatch permissions
3. Check that metrics exist for the time period
4. EMR must have CloudWatch metrics enabled

## Best Practices

1. **Use IAM Roles** - Prefer IAM roles over access keys for better security
2. **Filter Clusters** - Specify `cluster_ids` for production to avoid unnecessary API calls
3. **Monitor Costs** - Enable `collect_costs` to track spending
4. **Regional Deployment** - Deploy collectors in the same region as EMR clusters
5. **Rate Limiting** - Use `max_clusters` to avoid AWS API throttling
6. **Caching** - Store results in database to reduce repeated API calls

## Next Steps

- [Set up automated recommendations](./RECOMMENDATIONS.md)
- [Configure cost optimization](./COST_OPTIMIZATION.md)
- [Integrate with Databricks](./DATABRICKS_INTEGRATION.md)
- [Monitor multiple cloud providers](./MULTI_CLOUD.md)

## Support

For issues or questions:
- GitHub Issues: https://github.com/gridatek/spark-optimizer/issues
- Documentation: https://github.com/gridatek/spark-optimizer/docs
