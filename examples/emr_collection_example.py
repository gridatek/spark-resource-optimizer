#!/usr/bin/env python3
"""
Example script for collecting Spark metrics from AWS EMR clusters.

This script demonstrates how to:
1. Initialize the EMR collector
2. Validate AWS credentials
3. Collect metrics from EMR clusters
4. Store data in database
5. Get instance type recommendations

Prerequisites:
- AWS credentials configured (aws configure)
- boto3 installed (pip install boto3)
- Database configured
"""

import os
import sys
from datetime import datetime

try:
    from spark_optimizer.collectors import EMRCollector
    from spark_optimizer.storage import Database
except ImportError:
    print("Error: spark_optimizer package not found.")
    print("Install it with: pip install -e .[aws]")
    sys.exit(1)


def main():
    """Main function to demonstrate EMR collection."""

    print("=" * 70)
    print("Spark Resource Optimizer - AWS EMR Collection Example")
    print("=" * 70)
    print()

    # Configuration
    AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
    DATABASE_URL = os.environ.get(
        "DATABASE_URL", "postgresql://localhost/spark_optimizer"
    )

    # Optional: Collect from specific clusters
    CLUSTER_IDS = os.environ.get("EMR_CLUSTER_IDS", "").split(",")
    CLUSTER_IDS = [cid.strip() for cid in CLUSTER_IDS if cid.strip()]

    print(f"📍 AWS Region: {AWS_REGION}")
    print(f"💾 Database: {DATABASE_URL}")
    if CLUSTER_IDS:
        print(f"🎯 Target Clusters: {', '.join(CLUSTER_IDS)}")
    else:
        print("🎯 Target: All active clusters")
    print()

    # Step 1: Initialize EMR Collector
    print("🔧 Initializing EMR collector...")
    config = {
        "cluster_ids": CLUSTER_IDS,
        "cluster_states": ["RUNNING", "WAITING"],
        "max_clusters": 10,
        "days_back": 7,
        "collect_cloudwatch": True,
        "collect_costs": True,
    }

    try:
        collector = EMRCollector(region_name=AWS_REGION, config=config)
        print("✅ EMR collector initialized")
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("Install boto3 with: pip install boto3")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing collector: {e}")
        sys.exit(1)

    # Step 2: Validate Configuration
    print("\n🔍 Validating AWS credentials and permissions...")
    if not collector.validate_config():
        print("❌ AWS configuration validation failed!")
        print("\nPossible issues:")
        print("  1. AWS credentials not configured")
        print("  2. Insufficient IAM permissions")
        print("  3. Invalid AWS region")
        print("\nRun 'aws sts get-caller-identity' to test credentials")
        sys.exit(1)

    print("✅ AWS configuration validated")

    # Step 3: Collect Data
    print(f"\n📊 Collecting data from EMR clusters...")
    print("This may take a few minutes...")

    try:
        jobs = collector.collect()
        print(f"\n✅ Successfully collected {len(jobs)} Spark applications")
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not jobs:
        print("\n⚠️  No Spark applications found.")
        print("Possible reasons:")
        print("  - No active EMR clusters in the region")
        print("  - Clusters don't have completed Spark steps")
        print("  - Clusters are older than 'days_back' configuration")
        return

    # Step 4: Display Sample Data
    print("\n" + "=" * 70)
    print("📈 Sample Application Metrics")
    print("=" * 70)

    for i, job in enumerate(jobs[:3], 1):  # Show first 3 jobs
        print(f"\n{i}. {job.get('app_name', 'Unknown')}")
        print(f"   App ID: {job.get('app_id')}")
        print(f"   Duration: {job.get('duration_ms', 0) / 1000 / 60:.2f} minutes")
        print(f"   Executors: {job.get('num_executors')}")
        print(
            f"   Executor Config: {job.get('executor_cores')} cores, "
            f"{job.get('executor_memory_mb')} MB"
        )

        if "estimated_cost" in job:
            print(f"   Estimated Cost: ${job['estimated_cost']:.4f}")

        if "avg_cpu_utilization" in job:
            print(f"   Avg CPU: {job['avg_cpu_utilization']:.1f}%")

        if "tags" in job:
            cluster_name = job["tags"].get("cluster_name", "N/A")
            instance_type = job["tags"].get("instance_type", "N/A")
            print(f"   Cluster: {cluster_name}")
            print(f"   Instance Type: {instance_type}")

    if len(jobs) > 3:
        print(f"\n   ... and {len(jobs) - 3} more applications")

    # Step 5: Get Instance Type Recommendation (for first job with CloudWatch data)
    print("\n" + "=" * 70)
    print("🎯 Instance Type Recommendation")
    print("=" * 70)

    for job in jobs:
        if "avg_cpu_utilization" in job and "tags" in job:
            current_instance = job["tags"].get("instance_type", "m5.xlarge")

            workload_profile = {
                "cpu_utilization": job.get("avg_cpu_utilization", 50),
                "memory_utilization": 50,  # Would calculate from CloudWatch data
                "job_type": "etl",
            }

            recommendation = collector.get_instance_type_recommendations(
                current_instance, workload_profile
            )

            print(f"\nApplication: {job.get('app_name')}")
            print(f"Current Instance: {recommendation['current_instance_type']}")
            print(
                f"  Cost: ${recommendation['current_hourly_cost']:.3f}/hour"
            )
            print(
                f"\nRecommended Instance: {recommendation['recommended_instance_type']}"
            )
            print(
                f"  Cost: ${recommendation['recommended_hourly_cost']:.3f}/hour"
            )
            print(f"  Change: {recommendation['estimated_cost_change_percent']:+.1f}%")
            print(f"\nReason: {recommendation['reason']}")
            break

    # Step 6: Store in Database (Optional)
    if input("\n\n💾 Store data in database? (y/N): ").lower() == "y":
        print("\n🔄 Connecting to database...")
        try:
            db = Database(DATABASE_URL)
            print("✅ Database connected")

            print(f"💾 Storing {len(jobs)} applications...")
            for job in jobs:
                db.save_job(job)

            print("✅ Data stored successfully!")

        except Exception as e:
            print(f"❌ Database error: {e}")
            print("Data collected but not stored.")

    # Summary
    print("\n" + "=" * 70)
    print("✅ Collection Complete!")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Region: {AWS_REGION}")
    print(f"Applications Collected: {len(jobs)}")

    total_cost = sum(job.get("estimated_cost", 0) for job in jobs)
    if total_cost > 0:
        print(f"Total Estimated Cost: ${total_cost:.2f}")

    print("\nNext Steps:")
    print("  1. Run recommendations: spark-optimizer recommend")
    print("  2. View dashboard: spark-optimizer dashboard")
    print("  3. Analyze trends: spark-optimizer analyze")
    print()


if __name__ == "__main__":
    main()
