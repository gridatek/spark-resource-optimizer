"""Collector for Databricks workspaces and Spark applications."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time

try:
    import requests
    from requests.auth import HTTPBasicAuth

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from .base_collector import BaseCollector


class DatabricksCollector(BaseCollector):
    """Collects Spark job data from Databricks workspaces.

    Features:
    - Lists clusters via Databricks REST API
    - Fetches job runs and execution details
    - Pulls cluster metrics and configurations
    - Retrieves cost data based on DBU (Databricks Units) consumption
    - Provides Databricks-specific cluster type recommendations

    Prerequisites:
    - requests library installed (pip install requests)
    - Databricks workspace access token or service principal credentials
    - Required permissions:
        - clusters:list
        - clusters:get
        - jobs:list
        - jobs:get
        - sql:read (for SQL analytics)

    Authentication Methods:
    1. Personal Access Token (PAT)
    2. Service Principal with client ID and secret
    3. Azure AD token (for Azure Databricks)
    """

    # Databricks cluster types and their characteristics
    CLUSTER_TYPES = {
        # Standard clusters
        "Standard_DS3_v2": {"cores": 4, "memory_gb": 14, "dbu_per_hour": 0.75},
        "Standard_DS4_v2": {"cores": 8, "memory_gb": 28, "dbu_per_hour": 1.5},
        "Standard_DS5_v2": {"cores": 16, "memory_gb": 56, "dbu_per_hour": 3.0},
        # Memory optimized
        "Standard_E4s_v3": {"cores": 4, "memory_gb": 32, "dbu_per_hour": 1.0},
        "Standard_E8s_v3": {"cores": 8, "memory_gb": 64, "dbu_per_hour": 2.0},
        "Standard_E16s_v3": {"cores": 16, "memory_gb": 128, "dbu_per_hour": 4.0},
        # Compute optimized
        "Standard_F4s": {"cores": 4, "memory_gb": 8, "dbu_per_hour": 0.6},
        "Standard_F8s": {"cores": 8, "memory_gb": 16, "dbu_per_hour": 1.2},
        "Standard_F16s": {"cores": 16, "memory_gb": 32, "dbu_per_hour": 2.4},
        # AWS instance types
        "i3.xlarge": {"cores": 4, "memory_gb": 30.5, "dbu_per_hour": 0.75},
        "i3.2xlarge": {"cores": 8, "memory_gb": 61, "dbu_per_hour": 1.5},
        "i3.4xlarge": {"cores": 16, "memory_gb": 122, "dbu_per_hour": 3.0},
        "r5d.xlarge": {"cores": 4, "memory_gb": 32, "dbu_per_hour": 0.9},
        "r5d.2xlarge": {"cores": 8, "memory_gb": 64, "dbu_per_hour": 1.8},
        "r5d.4xlarge": {"cores": 16, "memory_gb": 128, "dbu_per_hour": 3.6},
    }

    def __init__(
        self,
        workspace_url: str,
        token: Optional[str] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize the Databricks collector.

        Args:
            workspace_url: Databricks workspace URL (e.g., https://dbc-xxx.cloud.databricks.com)
            token: Personal access token or service principal token
            config: Optional configuration dictionary with:
                - cluster_ids: List of specific cluster IDs to collect from
                - cluster_states: List of cluster states to filter (default: ['RUNNING', 'PENDING'])
                - max_clusters: Maximum number of clusters to process (default: 20)
                - days_back: How many days back to collect data (default: 7)
                - collect_sql_analytics: Include SQL analytics endpoints (default: True)
                - collect_costs: Whether to collect DBU cost data (default: True)
                - dbu_price: Price per DBU in USD (default: 0.40)
                - username: Username for basic auth (alternative to token)
                - password: Password for basic auth (alternative to token)

        Raises:
            ImportError: If requests library is not installed
            ValueError: If neither token nor username/password provided
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError(
                "requests library is required for Databricks collector. "
                "Install it with: pip install requests"
            )

        super().__init__(config)
        self.workspace_url = workspace_url.rstrip("/")
        self.token = token

        # Authentication setup
        if not self.token and not (
            self.config.get("username") and self.config.get("password")
        ):
            raise ValueError(
                "Either 'token' or 'username' and 'password' must be provided"
            )

        # Setup headers
        self.headers = {"Content-Type": "application/json"}
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"

        # Basic auth (if username/password provided)
        self.auth = None
        if self.config.get("username") and self.config.get("password"):
            self.auth = HTTPBasicAuth(self.config["username"], self.config["password"])

        # Configuration
        self.cluster_ids = self.config.get("cluster_ids", [])
        self.cluster_states = self.config.get("cluster_states", ["RUNNING", "PENDING"])
        self.max_clusters = self.config.get("max_clusters", 20)
        self.days_back = self.config.get("days_back", 7)
        self.collect_sql_analytics = self.config.get("collect_sql_analytics", True)
        self.collect_costs = self.config.get("collect_costs", True)
        self.dbu_price = self.config.get("dbu_price", 0.40)  # Default DBU price

        # API endpoints
        self.api_base = f"{self.workspace_url}/api/2.0"
        self.api_base_v2_1 = f"{self.workspace_url}/api/2.1"

    def collect(self) -> List[Dict]:
        """Collect job data from Databricks workspace.

        Returns:
            List of dictionaries containing job metrics in the format expected
            by the database storage layer.

        Raises:
            requests.HTTPError: If API requests fail
        """
        clusters = self._list_clusters()
        job_data = []

        for cluster in clusters:
            cluster_id = cluster["cluster_id"]
            try:
                # Get cluster details
                cluster_details = self._get_cluster(cluster_id)

                # Get Spark applications from cluster
                applications = self._fetch_applications(cluster_id, cluster_details)

                # Add cost data if enabled
                if self.collect_costs and applications:
                    cluster_cost = self._calculate_cluster_cost(cluster_details)
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
            True if configuration is valid and Databricks API is accessible
        """
        try:
            # Test API access by listing clusters
            response = requests.get(
                f"{self.api_base}/clusters/list",
                headers=self.headers,
                auth=self.auth,
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Databricks validation failed: {e}")
            return False

    def get_cluster_recommendations(
        self, current_cluster_type: str, workload_profile: Dict
    ) -> Dict:
        """Get Databricks cluster type recommendations based on workload profile.

        Args:
            current_cluster_type: Current cluster node type
            workload_profile: Dictionary with workload characteristics:
                - memory_intensive: Whether workload is memory intensive
                - io_intensive: Whether workload is I/O intensive
                - job_type: Type of job ('etl', 'ml', 'streaming', 'sql')
                - autoscaling: Whether to recommend autoscaling

        Returns:
            Dictionary with recommendations:
                - recommended_cluster_type: Suggested node type
                - reason: Explanation for recommendation
                - estimated_cost_change: Percentage cost change
                - autoscaling_recommended: Whether to enable autoscaling
        """
        memory_intensive = workload_profile.get("memory_intensive", False)
        io_intensive = workload_profile.get("io_intensive", False)
        job_type = workload_profile.get("job_type", "etl")

        # Determine cloud provider from cluster type
        is_azure = current_cluster_type.startswith("Standard_")
        is_aws = current_cluster_type.startswith(("i3", "r5", "m5", "c5"))

        # Memory-intensive workloads (ML, caching)
        if memory_intensive or job_type == "ml":
            recommended = "Standard_E8s_v3" if is_azure else "r5d.2xlarge"
            reason = (
                "Memory-intensive workload - recommending memory-optimized instances"
            )

        # I/O-intensive workloads (streaming, real-time)
        elif io_intensive or job_type == "streaming":
            recommended = "Standard_DS4_v2" if is_azure else "i3.2xlarge"
            reason = "I/O-intensive workload - recommending instances with local SSDs"

        # SQL analytics
        elif job_type == "sql":
            recommended = "Standard_F8s" if is_azure else "i3.xlarge"
            reason = "SQL analytics workload - recommending compute-optimized instances"

        # Balanced ETL workloads
        else:
            recommended = "Standard_DS4_v2" if is_azure else "i3.2xlarge"
            reason = "Balanced ETL workload - recommending general purpose instances"

        # Calculate cost change
        current_dbu = self.CLUSTER_TYPES.get(current_cluster_type, {}).get(
            "dbu_per_hour", 0
        )
        recommended_dbu = self.CLUSTER_TYPES.get(recommended, {}).get("dbu_per_hour", 0)

        if current_dbu > 0:
            cost_change = ((recommended_dbu - current_dbu) / current_dbu) * 100
        else:
            cost_change = 0

        # Autoscaling recommendation
        autoscaling_recommended = job_type in ["etl", "streaming"]

        return {
            "recommended_cluster_type": recommended,
            "current_cluster_type": current_cluster_type,
            "reason": reason,
            "estimated_cost_change_percent": round(cost_change, 2),
            "current_dbu_per_hour": current_dbu,
            "recommended_dbu_per_hour": recommended_dbu,
            "autoscaling_recommended": autoscaling_recommended,
        }

    def _list_clusters(self) -> List[Dict]:
        """List Databricks clusters based on configuration.

        Returns:
            List of cluster dictionaries
        """
        if self.cluster_ids:
            # Fetch specific clusters
            clusters = []
            for cluster_id in self.cluster_ids:
                try:
                    cluster = self._get_cluster(cluster_id)
                    clusters.append(cluster)
                except Exception as e:
                    print(f"Error fetching cluster {cluster_id}: {e}")
            return clusters

        # List all clusters
        try:
            response = requests.get(
                f"{self.api_base}/clusters/list",
                headers=self.headers,
                auth=self.auth,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            all_clusters = data.get("clusters", [])

            # Filter by state if specified
            if self.cluster_states:
                filtered_clusters = [
                    c for c in all_clusters if c.get("state") in self.cluster_states
                ]
            else:
                filtered_clusters = all_clusters

            return filtered_clusters[: self.max_clusters]

        except Exception as e:
            print(f"Error listing clusters: {e}")
            return []

    def _get_cluster(self, cluster_id: str) -> Dict:
        """Get detailed information about a cluster.

        Args:
            cluster_id: Databricks cluster ID

        Returns:
            Cluster details dictionary
        """
        response = requests.get(
            f"{self.api_base}/clusters/get",
            params={"cluster_id": cluster_id},
            headers=self.headers,
            auth=self.auth,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    def _fetch_applications(self, cluster_id: str, cluster_details: Dict) -> List[Dict]:
        """Fetch Spark applications from a Databricks cluster.

        This fetches job runs associated with the cluster.

        Args:
            cluster_id: Databricks cluster ID
            cluster_details: Cluster metadata

        Returns:
            List of application metrics dictionaries
        """
        applications = []

        try:
            # Calculate time range
            end_time_ms = int(time.time() * 1000)
            start_time_ms = end_time_ms - (self.days_back * 24 * 60 * 60 * 1000)

            # List job runs for this cluster
            response = requests.get(
                f"{self.api_base}/jobs/runs/list",
                params={
                    "limit": 100,
                    "start_time_from": start_time_ms,
                    "start_time_to": end_time_ms,
                },
                headers=self.headers,
                auth=self.auth,
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            runs = data.get("runs", [])

            # Filter runs for this cluster
            cluster_runs = [
                r
                for r in runs
                if r.get("cluster_instance", {}).get("cluster_id") == cluster_id
            ]

            for run in cluster_runs:
                app_metrics = self._convert_run_to_metrics(
                    cluster_id, cluster_details, run
                )
                if app_metrics:
                    applications.append(app_metrics)

        except Exception as e:
            print(f"Error fetching applications for cluster {cluster_id}: {e}")

        return applications

    def _convert_run_to_metrics(
        self, cluster_id: str, cluster_details: Dict, run: Dict
    ) -> Optional[Dict]:
        """Convert Databricks job run to application metrics format.

        Args:
            cluster_id: Databricks cluster ID
            cluster_details: Cluster metadata
            run: Job run details from Databricks API

        Returns:
            Application metrics dictionary or None if invalid
        """
        # Extract cluster configuration
        node_type_id = cluster_details.get("node_type_id", "Standard_DS3_v2")
        num_workers = cluster_details.get("num_workers", 0)
        autoscale = cluster_details.get("autoscale", {})

        if autoscale:
            num_workers = autoscale.get("max_workers", num_workers)

        # Get node specs
        node_specs = self.CLUSTER_TYPES.get(node_type_id, {"cores": 4, "memory_gb": 14})

        # Calculate executor configuration
        executor_cores = node_specs["cores"]
        executor_memory_mb = int(node_specs["memory_gb"] * 1024 * 0.9)
        num_executors = num_workers

        # Extract timing information
        start_time_ms = run.get("start_time")
        end_time_ms = run.get("end_time")

        if not start_time_ms:
            return None

        start_time = datetime.fromtimestamp(start_time_ms / 1000.0)
        end_time = datetime.fromtimestamp(end_time_ms / 1000.0) if end_time_ms else None

        duration_ms = 0
        if end_time_ms and start_time_ms:
            duration_ms = end_time_ms - start_time_ms

        # Extract task information
        tasks = run.get("tasks", [])
        state = run.get("state", {})
        result_state = state.get("result_state", "UNKNOWN")

        return {
            "app_id": f"{cluster_id}-{run.get('run_id')}",
            "app_name": run.get("run_name", "Unknown"),
            "user": run.get("creator_user_name", "unknown"),
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "num_executors": num_executors,
            "executor_cores": executor_cores,
            "executor_memory_mb": executor_memory_mb,
            "driver_memory_mb": 4096,
            "total_tasks": len(tasks),
            "failed_tasks": 1 if result_state == "FAILED" else 0,
            "total_stages": 0,
            "input_bytes": 0,
            "output_bytes": 0,
            "shuffle_read_bytes": 0,
            "shuffle_write_bytes": 0,
            "disk_spilled_bytes": 0,
            "memory_spilled_bytes": 0,
            "executor_run_time_ms": duration_ms,
            "jvm_gc_time_ms": 0,
            "estimated_cost": 0.0,
            "tags": {
                "cluster_id": cluster_id,
                "cluster_name": cluster_details.get("cluster_name", ""),
                "node_type": node_type_id,
                "databricks_runtime": cluster_details.get("spark_version", "unknown"),
                "job_id": str(run.get("job_id", "")),
                "run_id": str(run.get("run_id", "")),
                "run_state": result_state,
            },
        }

    def _calculate_cluster_cost(self, cluster_details: Dict) -> float:
        """Calculate cost for a cluster based on DBU consumption.

        Args:
            cluster_details: Cluster metadata

        Returns:
            Estimated cluster cost in dollars
        """
        try:
            # Get cluster runtime
            start_time_ms = cluster_details.get("start_time")
            state_message = cluster_details.get("state_message", "")

            if not start_time_ms:
                return 0.0

            # Calculate runtime in hours
            current_time_ms = int(time.time() * 1000)
            runtime_ms = current_time_ms - start_time_ms
            runtime_hours = runtime_ms / (1000 * 60 * 60)

            # Get node type and count
            node_type_id = cluster_details.get("node_type_id", "Standard_DS3_v2")
            num_workers = cluster_details.get("num_workers", 0)
            autoscale = cluster_details.get("autoscale", {})

            if autoscale:
                # Use average of min and max for autoscaling
                min_workers = autoscale.get("min_workers", 0)
                max_workers = autoscale.get("max_workers", num_workers)
                num_workers = (min_workers + max_workers) / 2

            # Get DBU rate
            dbu_per_hour = self.CLUSTER_TYPES.get(node_type_id, {}).get(
                "dbu_per_hour", 0.75
            )

            # Calculate total cost
            # Cost = (workers + 1 driver) * DBU_rate * hours * price_per_DBU
            total_dbu = (num_workers + 1) * dbu_per_hour * runtime_hours
            total_cost = total_dbu * self.dbu_price

            return total_cost

        except Exception as e:
            print(f"Error calculating cost for cluster: {e}")
            return 0.0
