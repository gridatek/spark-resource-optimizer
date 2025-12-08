"""Collector for Google Cloud Dataproc Spark metrics.

This module provides functionality to collect Spark application metrics from
Google Cloud Dataproc clusters using the Dataproc API.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

try:
    from google.cloud import dataproc_v1
    from google.cloud import monitoring_v3
    from google.api_core import exceptions as google_exceptions
    from google.oauth2 import service_account

    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False

from .base_collector import BaseCollector


class DataprocCollector(BaseCollector):
    """Collector for Google Cloud Dataproc clusters.

    Collects Spark application metrics from Dataproc clusters using the
    Dataproc API and Cloud Monitoring API.

    Features:
    - Automatic cluster discovery
    - Job and metrics collection
    - GCP cost tracking based on machine types
    - Machine type recommendations
    - Support for multiple regions
    - Integration with Cloud Monitoring

    Attributes:
        project_id: GCP project ID
        region: GCP region (e.g., 'us-central1')
        cluster_client: Dataproc cluster controller client
        job_client: Dataproc job controller client
        monitoring_client: Cloud Monitoring client
    """

    # GCP machine type specifications (cores, memory_gb, price_per_hour_usd)
    MACHINE_TYPES = {
        # Standard (n1) - General purpose
        "n1-standard-4": {"cores": 4, "memory_gb": 15, "price": 0.19},
        "n1-standard-8": {"cores": 8, "memory_gb": 30, "price": 0.38},
        "n1-standard-16": {"cores": 16, "memory_gb": 60, "price": 0.76},
        "n1-standard-32": {"cores": 32, "memory_gb": 120, "price": 1.52},
        # High-memory (n1) - Memory-intensive workloads
        "n1-highmem-4": {"cores": 4, "memory_gb": 26, "price": 0.24},
        "n1-highmem-8": {"cores": 8, "memory_gb": 52, "price": 0.47},
        "n1-highmem-16": {"cores": 16, "memory_gb": 104, "price": 0.94},
        "n1-highmem-32": {"cores": 32, "memory_gb": 208, "price": 1.87},
        # High-CPU (n1) - Compute-intensive workloads
        "n1-highcpu-4": {"cores": 4, "memory_gb": 3.6, "price": 0.14},
        "n1-highcpu-8": {"cores": 8, "memory_gb": 7.2, "price": 0.28},
        "n1-highcpu-16": {"cores": 16, "memory_gb": 14.4, "price": 0.56},
        "n1-highcpu-32": {"cores": 32, "memory_gb": 28.8, "price": 1.13},
        # N2 - Newer generation, better performance
        "n2-standard-4": {"cores": 4, "memory_gb": 16, "price": 0.19},
        "n2-standard-8": {"cores": 8, "memory_gb": 32, "price": 0.39},
        "n2-standard-16": {"cores": 16, "memory_gb": 64, "price": 0.78},
        "n2-standard-32": {"cores": 32, "memory_gb": 128, "price": 1.55},
        # N2 High-memory
        "n2-highmem-4": {"cores": 4, "memory_gb": 32, "price": 0.26},
        "n2-highmem-8": {"cores": 8, "memory_gb": 64, "price": 0.52},
        "n2-highmem-16": {"cores": 16, "memory_gb": 128, "price": 1.04},
        # E2 - Cost-optimized
        "e2-standard-4": {"cores": 4, "memory_gb": 16, "price": 0.13},
        "e2-standard-8": {"cores": 8, "memory_gb": 32, "price": 0.27},
        "e2-standard-16": {"cores": 16, "memory_gb": 64, "price": 0.54},
        # C2 - Compute-optimized (highest performance)
        "c2-standard-4": {"cores": 4, "memory_gb": 16, "price": 0.21},
        "c2-standard-8": {"cores": 8, "memory_gb": 32, "price": 0.42},
        "c2-standard-16": {"cores": 16, "memory_gb": 64, "price": 0.85},
    }

    def __init__(
        self,
        project_id: str,
        region: str = "us-central1",
        credentials: Optional[Any] = None,
        config: Optional[Dict] = None,
    ):
        """Initialize Dataproc collector.

        Args:
            project_id: GCP project ID
            region: GCP region (default: us-central1)
            credentials: GCP credentials (optional, uses ADC if not provided)
            config: Additional configuration options:
                - cluster_names: List of specific clusters to monitor
                - cluster_labels: Dict of labels to filter clusters
                - max_clusters: Maximum clusters to process (default: 10)
                - days_back: Days of history to collect (default: 7)
                - collect_costs: Enable cost tracking (default: True)
                - include_preemptible: Include preemptible worker costs (default: True)
                - preemptible_discount: Preemptible pricing discount (default: 0.8)
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError(
                "google-cloud-dataproc is required for Dataproc collector. "
                "Install with: pip install google-cloud-dataproc google-cloud-monitoring"
            )

        super().__init__()

        self.project_id = project_id
        self.region = region
        self.credentials = credentials

        # Configuration
        config = config or {}
        self.cluster_names = config.get("cluster_names", [])
        self.cluster_labels = config.get("cluster_labels", {})
        self.max_clusters = config.get("max_clusters", 10)
        self.days_back = config.get("days_back", 7)
        self.collect_costs = config.get("collect_costs", True)
        self.include_preemptible = config.get("include_preemptible", True)
        self.preemptible_discount = config.get("preemptible_discount", 0.8)

        # Initialize clients
        self._init_clients()

    def _init_clients(self):
        """Initialize GCP clients."""
        client_options = {"api_endpoint": f"{self.region}-dataproc.googleapis.com"}

        if self.credentials:
            self.cluster_client = dataproc_v1.ClusterControllerClient(
                credentials=self.credentials, client_options=client_options
            )
            self.job_client = dataproc_v1.JobControllerClient(
                credentials=self.credentials, client_options=client_options
            )
            self.monitoring_client = monitoring_v3.MetricServiceClient(
                credentials=self.credentials
            )
        else:
            self.cluster_client = dataproc_v1.ClusterControllerClient(
                client_options=client_options
            )
            self.job_client = dataproc_v1.JobControllerClient(
                client_options=client_options
            )
            self.monitoring_client = monitoring_v3.MetricServiceClient()

    def validate_config(self) -> bool:
        """Validate Dataproc configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            # Try to list clusters to validate credentials
            request = dataproc_v1.ListClustersRequest(
                project_id=self.project_id, region=self.region, page_size=1
            )
            list(self.cluster_client.list_clusters(request=request))
            return True
        except google_exceptions.GoogleAPIError as e:
            print(f"Validation failed: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error during validation: {e}")
            return False

    def collect(self) -> List[Dict]:
        """Collect metrics from Dataproc clusters.

        Returns:
            List of job metrics dictionaries
        """
        all_jobs = []

        # Get clusters
        clusters = self._list_clusters()

        if not clusters:
            print("No Dataproc clusters found")
            return []

        print(f"Found {len(clusters)} Dataproc cluster(s)")

        # Collect jobs from each cluster
        for cluster in clusters:
            cluster_name = cluster.cluster_name

            try:
                # Get cluster details
                cluster_details = self._get_cluster_details(cluster_name)

                # Get jobs for this cluster
                jobs = self._get_cluster_jobs(cluster_name)

                print(f"Cluster '{cluster_name}': Found {len(jobs)} job(s)")

                # Convert jobs to metrics format
                for job in jobs:
                    metrics = self._convert_job_to_metrics(
                        cluster_name, cluster_details, job
                    )
                    if metrics:
                        all_jobs.append(metrics)

            except Exception as e:
                print(f"Error collecting from cluster '{cluster_name}': {e}")
                continue

        return all_jobs

    def _list_clusters(self) -> List:
        """List Dataproc clusters based on configuration.

        Returns:
            List of cluster objects
        """
        try:
            request = dataproc_v1.ListClustersRequest(
                project_id=self.project_id,
                region=self.region,
                filter=self._build_cluster_filter(),
            )

            clusters = list(self.cluster_client.list_clusters(request=request))

            # Apply max_clusters limit
            if self.max_clusters:
                clusters = clusters[: self.max_clusters]

            return clusters

        except google_exceptions.GoogleAPIError as e:
            print(f"Error listing clusters: {e}")
            return []

    def _build_cluster_filter(self) -> str:
        """Build cluster filter string.

        Returns:
            Filter string for ListClusters API
        """
        filters = []

        # Filter by cluster names
        if self.cluster_names:
            name_filters = [f'clusterName = "{name}"' for name in self.cluster_names]
            filters.append(f"({' OR '.join(name_filters)})")

        # Filter by labels
        if self.cluster_labels:
            for key, value in self.cluster_labels.items():
                filters.append(f'labels.{key} = "{value}"')

        # Only active clusters
        filters.append("status.state = RUNNING OR status.state = UPDATING")

        return " AND ".join(filters) if filters else ""

    def _get_cluster_details(self, cluster_name: str) -> Dict:
        """Get detailed cluster information.

        Args:
            cluster_name: Name of the cluster

        Returns:
            Dictionary with cluster details
        """
        try:
            request = dataproc_v1.GetClusterRequest(
                project_id=self.project_id,
                region=self.region,
                cluster_name=cluster_name,
            )

            cluster = self.cluster_client.get_cluster(request=request)

            # Extract relevant details
            config = cluster.config

            details = {
                "cluster_name": cluster.cluster_name,
                "cluster_uuid": cluster.cluster_uuid,
                "project_id": self.project_id,
                "region": self.region,
                "state": cluster.status.state.name,
                "master_config": {
                    "num_instances": config.master_config.num_instances,
                    "machine_type": config.master_config.machine_type_uri.split("/")[
                        -1
                    ],
                    "disk_size_gb": config.master_config.disk_config.boot_disk_size_gb,
                },
                "worker_config": {
                    "num_instances": config.worker_config.num_instances,
                    "machine_type": config.worker_config.machine_type_uri.split("/")[
                        -1
                    ],
                    "disk_size_gb": config.worker_config.disk_config.boot_disk_size_gb,
                },
                "labels": dict(cluster.labels),
                "create_time": cluster.status.state_start_time.timestamp(),
            }

            # Add preemptible worker config if exists
            if config.secondary_worker_config.num_instances > 0:
                details["preemptible_worker_config"] = {
                    "num_instances": config.secondary_worker_config.num_instances,
                    "machine_type": config.secondary_worker_config.machine_type_uri.split(
                        "/"
                    )[
                        -1
                    ],
                    "disk_size_gb": config.secondary_worker_config.disk_config.boot_disk_size_gb,
                }

            return details

        except google_exceptions.GoogleAPIError as e:
            print(f"Error getting cluster details for '{cluster_name}': {e}")
            return {}

    def _get_cluster_jobs(self, cluster_name: str) -> List:
        """Get jobs for a specific cluster.

        Args:
            cluster_name: Name of the cluster

        Returns:
            List of job objects
        """
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.days_back)

            request = dataproc_v1.ListJobsRequest(
                project_id=self.project_id,
                region=self.region,
                cluster_name=cluster_name,
                job_state_matcher=dataproc_v1.ListJobsRequest.JobStateMatcher.ALL,
            )

            jobs = []
            for job in self.job_client.list_jobs(request=request):
                # Filter by time range
                job_start = job.status.state_start_time.timestamp()
                if start_time.timestamp() <= job_start <= end_time.timestamp():
                    jobs.append(job)

            return jobs

        except google_exceptions.GoogleAPIError as e:
            print(f"Error getting jobs for cluster '{cluster_name}': {e}")
            return []

    def _convert_job_to_metrics(
        self, cluster_name: str, cluster_details: Dict, job: Any
    ) -> Optional[Dict]:
        """Convert Dataproc job to metrics format.

        Args:
            cluster_name: Cluster name
            cluster_details: Cluster configuration details
            job: Dataproc job object

        Returns:
            Dictionary in standardized metrics format
        """
        try:
            # Extract job details
            job_id = job.reference.job_id
            job_type = "unknown"

            # Determine job type
            if job.spark_job:
                job_type = "spark"
            elif job.pyspark_job:
                job_type = "pyspark"
            elif job.spark_sql_job:
                job_type = "spark_sql"

            # Get timing information
            start_time = job.status.state_start_time.timestamp()
            end_time = None
            duration_ms = 0

            if job.status.state in [
                dataproc_v1.JobStatus.State.DONE,
                dataproc_v1.JobStatus.State.ERROR,
                dataproc_v1.JobStatus.State.CANCELLED,
            ]:
                end_time = job.status.state_start_time.timestamp()
                if hasattr(job.status, "details"):
                    duration_ms = int((end_time - start_time) * 1000)

            # Extract machine type information
            worker_config = cluster_details.get("worker_config", {})
            machine_type = worker_config.get("machine_type", "n1-standard-4")
            num_workers = worker_config.get("num_instances", 2)

            # Get machine specs
            machine_specs = self.MACHINE_TYPES.get(
                machine_type, {"cores": 4, "memory_gb": 15, "price": 0.19}
            )

            # Build metrics dictionary
            metrics = {
                "app_id": f"{cluster_name}-{job_id}",
                "app_name": job.reference.job_id,
                "user": (
                    job.status.details if hasattr(job.status, "details") else "unknown"
                ),
                "start_time": int(start_time * 1000),
                "end_time": int(end_time * 1000) if end_time else None,
                "duration_ms": duration_ms,
                "num_executors": num_workers,
                "executor_cores": machine_specs["cores"],
                "executor_memory_mb": int(machine_specs["memory_gb"] * 1024),
                "driver_memory_mb": int(machine_specs["memory_gb"] * 1024),
                "total_tasks": 0,  # Not directly available
                "failed_tasks": (
                    1 if job.status.state == dataproc_v1.JobStatus.State.ERROR else 0
                ),
                "succeeded_tasks": (
                    1 if job.status.state == dataproc_v1.JobStatus.State.DONE else 0
                ),
                "input_bytes": 0,  # Would need metrics API
                "output_bytes": 0,  # Would need metrics API
                "shuffle_read_bytes": 0,
                "shuffle_write_bytes": 0,
                "tags": {
                    "cluster_name": cluster_name,
                    "cluster_uuid": cluster_details.get("cluster_uuid", ""),
                    "project_id": self.project_id,
                    "region": self.region,
                    "machine_type": machine_type,
                    "job_type": job_type,
                    "job_state": job.status.state.name,
                },
            }

            # Add cost information
            if self.collect_costs:
                cost = self._calculate_job_cost(cluster_details, duration_ms)
                metrics["estimated_cost"] = cost

            return metrics

        except Exception as e:
            print(f"Error converting job to metrics: {e}")
            return None

    def _calculate_job_cost(
        self, cluster_details: Dict, duration_ms: int
    ) -> float:
        """Calculate job cost based on cluster configuration and duration.

        Args:
            cluster_details: Cluster configuration
            duration_ms: Job duration in milliseconds

        Returns:
            Estimated cost in USD
        """
        if duration_ms == 0:
            return 0.0

        total_cost = 0.0
        duration_hours = duration_ms / 1000 / 3600

        # Master node cost
        master_config = cluster_details.get("master_config", {})
        master_type = master_config.get("machine_type", "n1-standard-4")
        master_count = master_config.get("num_instances", 1)
        master_specs = self.MACHINE_TYPES.get(
            master_type, {"price": 0.19}
        )
        total_cost += (
            master_count * master_specs["price"] * duration_hours
        )

        # Worker node cost
        worker_config = cluster_details.get("worker_config", {})
        worker_type = worker_config.get("machine_type", "n1-standard-4")
        worker_count = worker_config.get("num_instances", 2)
        worker_specs = self.MACHINE_TYPES.get(
            worker_type, {"price": 0.19}
        )
        total_cost += (
            worker_count * worker_specs["price"] * duration_hours
        )

        # Preemptible worker cost (if enabled and exists)
        if self.include_preemptible and "preemptible_worker_config" in cluster_details:
            preempt_config = cluster_details["preemptible_worker_config"]
            preempt_type = preempt_config.get("machine_type", "n1-standard-4")
            preempt_count = preempt_config.get("num_instances", 0)
            preempt_specs = self.MACHINE_TYPES.get(
                preempt_type, {"price": 0.19}
            )
            preempt_cost = (
                preempt_count
                * preempt_specs["price"]
                * duration_hours
                * self.preemptible_discount
            )
            total_cost += preempt_cost

        return round(total_cost, 4)

    def get_machine_type_recommendations(
        self, current_machine_type: str, workload_profile: Dict
    ) -> Dict:
        """Get machine type recommendations based on workload.

        Args:
            current_machine_type: Current machine type
            workload_profile: Dictionary with workload characteristics:
                - memory_intensive: bool
                - compute_intensive: bool
                - job_type: str (ml, streaming, etl, sql, batch)
                - cost_optimized: bool

        Returns:
            Dictionary with recommendation details
        """
        current_specs = self.MACHINE_TYPES.get(
            current_machine_type,
            {"cores": 4, "memory_gb": 15, "price": 0.19},
        )

        # Determine recommended machine type based on workload
        recommended_type = current_machine_type
        reason = "Current configuration is appropriate"

        if workload_profile.get("memory_intensive") or workload_profile.get(
            "job_type"
        ) == "ml":
            # Recommend high-memory instances
            if "highmem" not in current_machine_type:
                cores = current_specs["cores"]
                if cores <= 8:
                    recommended_type = "n2-highmem-8"
                elif cores <= 16:
                    recommended_type = "n2-highmem-16"
                else:
                    recommended_type = "n2-highmem-16"
                reason = "Memory-intensive workload benefits from high-memory instances"

        elif workload_profile.get(
            "compute_intensive"
        ) or workload_profile.get("job_type") in ["streaming", "realtime"]:
            # Recommend compute-optimized instances
            if not current_machine_type.startswith("c2-"):
                cores = current_specs["cores"]
                if cores <= 8:
                    recommended_type = "c2-standard-8"
                elif cores <= 16:
                    recommended_type = "c2-standard-16"
                else:
                    recommended_type = "c2-standard-16"
                reason = "Compute-intensive workload benefits from C2 instances"

        elif workload_profile.get("cost_optimized") or workload_profile.get(
            "job_type"
        ) == "batch":
            # Recommend E2 cost-optimized instances
            if not current_machine_type.startswith("e2-"):
                cores = current_specs["cores"]
                if cores <= 8:
                    recommended_type = "e2-standard-8"
                elif cores <= 16:
                    recommended_type = "e2-standard-16"
                else:
                    recommended_type = "e2-standard-16"
                reason = "Batch workload can use cost-optimized E2 instances"

        elif workload_profile.get("job_type") in ["etl", "sql"]:
            # Recommend balanced N2 instances
            if not current_machine_type.startswith("n2-"):
                cores = current_specs["cores"]
                if cores <= 8:
                    recommended_type = "n2-standard-8"
                elif cores <= 16:
                    recommended_type = "n2-standard-16"
                else:
                    recommended_type = "n2-standard-32"
                reason = "ETL/SQL workload benefits from balanced N2 instances"

        # Calculate cost impact
        recommended_specs = self.MACHINE_TYPES.get(
            recommended_type, current_specs
        )
        cost_change_percent = (
            (
                (recommended_specs["price"] - current_specs["price"])
                / current_specs["price"]
            )
            * 100
        )

        # Determine if preemptible workers are recommended
        preemptible_recommended = workload_profile.get("job_type") in [
            "batch",
            "etl",
        ] or workload_profile.get("cost_optimized", False)

        return {
            "current_machine_type": current_machine_type,
            "recommended_machine_type": recommended_type,
            "reason": reason,
            "current_specs": current_specs,
            "recommended_specs": recommended_specs,
            "estimated_cost_change_percent": cost_change_percent,
            "preemptible_recommended": preemptible_recommended,
        }
