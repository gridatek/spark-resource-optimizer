"""Collector for AWS EMR (Elastic MapReduce) clusters and Spark applications."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

from .base_collector import BaseCollector


class EMRCollector(BaseCollector):
    """Collects Spark job data from AWS EMR clusters.

    Features:
    - Lists EMR clusters via boto3
    - Fetches application history from EMR
    - Pulls CloudWatch metrics for detailed performance data
    - Retrieves cost data from AWS Cost Explorer
    - Provides EMR-specific instance type recommendations

    Prerequisites:
    - boto3 library installed (pip install boto3)
    - AWS credentials configured (via environment variables, ~/.aws/credentials, or IAM role)
    - Required IAM permissions:
        - emr:ListClusters
        - emr:DescribeCluster
        - emr:ListSteps
        - emr:DescribeStep
        - cloudwatch:GetMetricStatistics
        - ce:GetCostAndUsage (for Cost Explorer)
        - s3:GetObject (for event logs if enabled)
    """

    # EMR instance type to vCPU and memory mapping
    EMR_INSTANCE_TYPES = {
        "m5.xlarge": {"vcpu": 4, "memory_gb": 16, "cost_per_hour": 0.192},
        "m5.2xlarge": {"vcpu": 8, "memory_gb": 32, "cost_per_hour": 0.384},
        "m5.4xlarge": {"vcpu": 16, "memory_gb": 64, "cost_per_hour": 0.768},
        "m5.8xlarge": {"vcpu": 32, "memory_gb": 128, "cost_per_hour": 1.536},
        "m5.12xlarge": {"vcpu": 48, "memory_gb": 192, "cost_per_hour": 2.304},
        "m5.16xlarge": {"vcpu": 64, "memory_gb": 256, "cost_per_hour": 3.072},
        "r5.xlarge": {"vcpu": 4, "memory_gb": 32, "cost_per_hour": 0.252},
        "r5.2xlarge": {"vcpu": 8, "memory_gb": 64, "cost_per_hour": 0.504},
        "r5.4xlarge": {"vcpu": 16, "memory_gb": 128, "cost_per_hour": 1.008},
        "r5.8xlarge": {"vcpu": 32, "memory_gb": 256, "cost_per_hour": 2.016},
        "r5.12xlarge": {"vcpu": 48, "memory_gb": 384, "cost_per_hour": 3.024},
        "c5.xlarge": {"vcpu": 4, "memory_gb": 8, "cost_per_hour": 0.17},
        "c5.2xlarge": {"vcpu": 8, "memory_gb": 16, "cost_per_hour": 0.34},
        "c5.4xlarge": {"vcpu": 16, "memory_gb": 32, "cost_per_hour": 0.68},
        "c5.9xlarge": {"vcpu": 36, "memory_gb": 72, "cost_per_hour": 1.53},
    }

    def __init__(
        self,
        region_name: str = "us-east-1",
        config: Optional[Dict] = None,
    ):
        """Initialize the EMR collector.

        Args:
            region_name: AWS region where EMR clusters are located
            config: Optional configuration dictionary with:
                - cluster_ids: List of specific cluster IDs to collect from
                - cluster_states: List of cluster states to filter (default: ['RUNNING', 'WAITING'])
                - max_clusters: Maximum number of clusters to process (default: 10)
                - days_back: How many days back to collect data (default: 7)
                - collect_cloudwatch: Whether to collect CloudWatch metrics (default: True)
                - collect_costs: Whether to collect cost data (default: True)
                - aws_access_key_id: AWS access key (optional, uses default credentials if not set)
                - aws_secret_access_key: AWS secret key (optional)
                - aws_session_token: AWS session token (optional)

        Raises:
            ImportError: If boto3 is not installed
        """
        if not BOTO3_AVAILABLE:
            raise ImportError(
                "boto3 is required for EMR collector. Install it with: pip install boto3"
            )

        super().__init__(config)
        self.region_name = region_name

        # AWS credentials (optional, will use default credential chain if not provided)
        aws_credentials = {}
        if self.config.get("aws_access_key_id"):
            aws_credentials["aws_access_key_id"] = self.config["aws_access_key_id"]
        if self.config.get("aws_secret_access_key"):
            aws_credentials["aws_secret_access_key"] = self.config[
                "aws_secret_access_key"
            ]
        if self.config.get("aws_session_token"):
            aws_credentials["aws_session_token"] = self.config["aws_session_token"]

        # Initialize AWS clients
        self.emr_client = boto3.client(
            "emr", region_name=region_name, **aws_credentials
        )
        self.cloudwatch_client = boto3.client(
            "cloudwatch", region_name=region_name, **aws_credentials
        )
        self.ce_client = boto3.client("ce", region_name="us-east-1", **aws_credentials)

        # Configuration
        self.cluster_ids = self.config.get("cluster_ids", [])
        self.cluster_states = self.config.get("cluster_states", ["RUNNING", "WAITING"])
        self.max_clusters = self.config.get("max_clusters", 10)
        self.days_back = self.config.get("days_back", 7)
        self.collect_cloudwatch = self.config.get("collect_cloudwatch", True)
        self.collect_costs = self.config.get("collect_costs", True)

    def collect(self) -> List[Dict]:
        """Collect job data from EMR clusters.

        Returns:
            List of dictionaries containing job metrics in the format expected
            by the database storage layer.

        Raises:
            ClientError: If AWS API calls fail
        """
        clusters = self._list_clusters()
        job_data = []

        for cluster in clusters:
            cluster_id = cluster["Id"]
            try:
                # Get cluster details
                cluster_details = self._describe_cluster(cluster_id)

                # Get Spark applications from cluster
                applications = self._fetch_applications(cluster_id, cluster_details)

                # Enrich with CloudWatch metrics if enabled
                if self.collect_cloudwatch:
                    for app in applications:
                        cloudwatch_metrics = self._fetch_cloudwatch_metrics(
                            cluster_id, app.get("app_id")
                        )
                        app.update(cloudwatch_metrics)

                # Add cost data if enabled
                if self.collect_costs and applications:
                    cluster_cost = self._fetch_cluster_cost(
                        cluster_id, cluster_details
                    )
                    # Distribute cost across applications proportionally
                    for app in applications:
                        app["estimated_cost"] = cluster_cost / len(applications)

                job_data.extend(applications)

            except Exception as e:
                print(f"Error collecting data from cluster {cluster_id}: {e}")
                continue

        return job_data

    def validate_config(self) -> bool:
        """Validate the collector configuration.

        Returns:
            True if configuration is valid and AWS credentials work
        """
        try:
            # Test EMR access
            self.emr_client.list_clusters(ClusterStates=["RUNNING"], MaxResults=1)
            return True
        except Exception as e:
            print(f"EMR validation failed: {e}")
            return False

    def get_instance_type_recommendations(
        self, current_instance_type: str, workload_profile: Dict
    ) -> Dict:
        """Get EMR instance type recommendations based on workload profile.

        Args:
            current_instance_type: Current EMR instance type
            workload_profile: Dictionary with workload characteristics:
                - cpu_utilization: Average CPU utilization (0-100)
                - memory_utilization: Average memory utilization (0-100)
                - io_intensive: Whether workload is I/O intensive
                - job_type: Type of job ('etl', 'ml', 'streaming', 'compute')

        Returns:
            Dictionary with recommendations:
                - recommended_instance_type: Suggested instance type
                - reason: Explanation for recommendation
                - estimated_cost_change: Percentage cost change
        """
        cpu_util = workload_profile.get("cpu_utilization", 50)
        mem_util = workload_profile.get("memory_utilization", 50)
        job_type = workload_profile.get("job_type", "etl")

        # Memory-intensive workloads (ML, caching)
        if mem_util > 70 or job_type == "ml":
            recommended = "r5.4xlarge"  # Memory-optimized
            reason = "High memory utilization detected - recommending memory-optimized instance"

        # CPU-intensive workloads (data processing, transformations)
        elif cpu_util > 70 or job_type == "compute":
            recommended = "c5.4xlarge"  # Compute-optimized
            reason = "High CPU utilization detected - recommending compute-optimized instance"

        # Balanced workloads (most ETL jobs)
        else:
            recommended = "m5.4xlarge"  # General purpose
            reason = "Balanced workload - recommending general purpose instance"

        # Calculate cost change
        current_cost = self.EMR_INSTANCE_TYPES.get(current_instance_type, {}).get(
            "cost_per_hour", 0
        )
        recommended_cost = self.EMR_INSTANCE_TYPES.get(recommended, {}).get(
            "cost_per_hour", 0
        )

        if current_cost > 0:
            cost_change = ((recommended_cost - current_cost) / current_cost) * 100
        else:
            cost_change = 0

        return {
            "recommended_instance_type": recommended,
            "current_instance_type": current_instance_type,
            "reason": reason,
            "estimated_cost_change_percent": round(cost_change, 2),
            "current_hourly_cost": current_cost,
            "recommended_hourly_cost": recommended_cost,
        }

    def _list_clusters(self) -> List[Dict]:
        """List EMR clusters based on configuration.

        Returns:
            List of cluster metadata dictionaries
        """
        if self.cluster_ids:
            # Fetch specific clusters
            clusters = []
            for cluster_id in self.cluster_ids:
                try:
                    response = self.emr_client.describe_cluster(ClusterId=cluster_id)
                    clusters.append(response["Cluster"])
                except Exception as e:
                    print(f"Error fetching cluster {cluster_id}: {e}")
            return clusters

        # List clusters by state
        created_after = datetime.utcnow() - timedelta(days=self.days_back)

        try:
            response = self.emr_client.list_clusters(
                ClusterStates=self.cluster_states,
                CreatedAfter=created_after,
            )
            return response.get("Clusters", [])[:self.max_clusters]
        except Exception as e:
            print(f"Error listing clusters: {e}")
            return []

    def _describe_cluster(self, cluster_id: str) -> Dict:
        """Get detailed information about a cluster.

        Args:
            cluster_id: EMR cluster ID

        Returns:
            Cluster details dictionary
        """
        response = self.emr_client.describe_cluster(ClusterId=cluster_id)
        return response["Cluster"]

    def _fetch_applications(
        self, cluster_id: str, cluster_details: Dict
    ) -> List[Dict]:
        """Fetch Spark applications from an EMR cluster.

        This attempts to get application data from EMR steps.
        For more detailed metrics, you would typically need to:
        1. Access the Spark History Server running on the master node
        2. Or parse event logs from S3

        Args:
            cluster_id: EMR cluster ID
            cluster_details: Cluster metadata

        Returns:
            List of application metrics dictionaries
        """
        applications = []

        try:
            # List steps (Spark jobs) on the cluster
            response = self.emr_client.list_steps(ClusterId=cluster_id)
            steps = response.get("Steps", [])

            for step in steps:
                step_id = step["Id"]
                step_details = self.emr_client.describe_step(
                    ClusterId=cluster_id, StepId=step_id
                )
                step_info = step_details["Step"]

                # Extract metrics from step
                app_metrics = self._convert_step_to_metrics(
                    cluster_id, cluster_details, step_info
                )
                if app_metrics:
                    applications.append(app_metrics)

        except Exception as e:
            print(f"Error fetching applications for cluster {cluster_id}: {e}")

        return applications

    def _convert_step_to_metrics(
        self, cluster_id: str, cluster_details: Dict, step: Dict
    ) -> Optional[Dict]:
        """Convert EMR step to application metrics format.

        Args:
            cluster_id: EMR cluster ID
            cluster_details: Cluster metadata
            step: Step details from EMR API

        Returns:
            Application metrics dictionary or None if step is not a Spark job
        """
        # Extract instance configuration
        instance_groups = cluster_details.get("InstanceGroups", [])
        core_instances = next(
            (ig for ig in instance_groups if ig["InstanceGroupType"] == "CORE"), {}
        )
        master_instance = next(
            (ig for ig in instance_groups if ig["InstanceGroupType"] == "MASTER"), {}
        )

        instance_type = core_instances.get("InstanceType", "m5.xlarge")
        instance_count = core_instances.get("RequestedInstanceCount", 1)

        # Get instance specs
        instance_specs = self.EMR_INSTANCE_TYPES.get(
            instance_type, {"vcpu": 4, "memory_gb": 16}
        )

        # Calculate executor configuration
        executor_cores = instance_specs["vcpu"]
        executor_memory_mb = int(instance_specs["memory_gb"] * 1024 * 0.9)  # 90% for executor
        num_executors = instance_count

        # Extract timing information
        start_time = step.get("Status", {}).get("Timeline", {}).get("CreationDateTime")
        end_time = step.get("Status", {}).get("Timeline", {}).get("EndDateTime")

        if not start_time:
            return None

        duration_ms = 0
        if end_time:
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

        return {
            "app_id": f"{cluster_id}-{step['Id']}",
            "app_name": step.get("Name", "Unknown"),
            "user": cluster_details.get("Tags", {}).get("Owner", "unknown"),
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "num_executors": num_executors,
            "executor_cores": executor_cores,
            "executor_memory_mb": executor_memory_mb,
            "driver_memory_mb": 4096,  # Typically runs on master
            "total_tasks": 0,  # Not available from EMR step API
            "failed_tasks": 0,
            "total_stages": 0,
            "input_bytes": 0,  # Would need to parse from logs
            "output_bytes": 0,
            "shuffle_read_bytes": 0,
            "shuffle_write_bytes": 0,
            "disk_spilled_bytes": 0,
            "memory_spilled_bytes": 0,
            "executor_run_time_ms": duration_ms,
            "jvm_gc_time_ms": 0,
            "estimated_cost": 0.0,  # Filled in later if enabled
            "tags": {
                "cluster_id": cluster_id,
                "cluster_name": cluster_details.get("Name", ""),
                "instance_type": instance_type,
                "emr_release": cluster_details.get("ReleaseLabel", ""),
                "step_state": step.get("Status", {}).get("State", ""),
            },
        }

    def _fetch_cloudwatch_metrics(
        self, cluster_id: str, app_id: Optional[str] = None
    ) -> Dict:
        """Fetch CloudWatch metrics for a cluster/application.

        Args:
            cluster_id: EMR cluster ID
            app_id: Optional application ID

        Returns:
            Dictionary with CloudWatch metrics
        """
        metrics = {}
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        try:
            # Cluster-level metrics
            # CPU utilization
            cpu_response = self.cloudwatch_client.get_metric_statistics(
                Namespace="AWS/ElasticMapReduce",
                MetricName="CPUUtilization",
                Dimensions=[{"Name": "JobFlowId", "Value": cluster_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5 minutes
                Statistics=["Average"],
            )

            if cpu_response.get("Datapoints"):
                avg_cpu = sum(
                    dp["Average"] for dp in cpu_response["Datapoints"]
                ) / len(cpu_response["Datapoints"])
                metrics["avg_cpu_utilization"] = avg_cpu

            # Memory utilization
            mem_response = self.cloudwatch_client.get_metric_statistics(
                Namespace="AWS/ElasticMapReduce",
                MetricName="MemoryAvailableMB",
                Dimensions=[{"Name": "JobFlowId", "Value": cluster_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=["Average"],
            )

            if mem_response.get("Datapoints"):
                avg_mem_available = sum(
                    dp["Average"] for dp in mem_response["Datapoints"]
                ) / len(mem_response["Datapoints"])
                metrics["avg_memory_available_mb"] = avg_mem_available

        except Exception as e:
            print(f"Error fetching CloudWatch metrics for {cluster_id}: {e}")

        return metrics

    def _fetch_cluster_cost(self, cluster_id: str, cluster_details: Dict) -> float:
        """Fetch cost data for a cluster from AWS Cost Explorer.

        Args:
            cluster_id: EMR cluster ID
            cluster_details: Cluster metadata

        Returns:
            Estimated cluster cost in dollars
        """
        try:
            # Get cluster runtime
            start_time = cluster_details.get("Status", {}).get("Timeline", {}).get("CreationDateTime")
            end_time = cluster_details.get("Status", {}).get("Timeline", {}).get("EndDateTime")

            if not start_time:
                return 0.0

            if not end_time:
                end_time = datetime.utcnow()

            # Calculate runtime in hours
            runtime_hours = (end_time - start_time).total_seconds() / 3600

            # Calculate cost based on instance types and counts
            total_cost = 0.0
            instance_groups = cluster_details.get("InstanceGroups", [])

            for ig in instance_groups:
                instance_type = ig.get("InstanceType", "")
                instance_count = ig.get("RequestedInstanceCount", 0)

                instance_cost = self.EMR_INSTANCE_TYPES.get(instance_type, {}).get(
                    "cost_per_hour", 0
                )
                total_cost += instance_cost * instance_count * runtime_hours

            return total_cost

        except Exception as e:
            print(f"Error calculating cost for cluster {cluster_id}: {e}")
            return 0.0
