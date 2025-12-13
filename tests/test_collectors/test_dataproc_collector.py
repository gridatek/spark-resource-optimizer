"""Tests for GCP Dataproc collector."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

try:
    from google.cloud import dataproc_v1
    from google.api_core import exceptions as google_exceptions
    from google.protobuf.timestamp_pb2 import Timestamp

    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    Timestamp = None  # type: ignore

if GOOGLE_CLOUD_AVAILABLE:
    from spark_optimizer.collectors.dataproc_collector import DataprocCollector


@pytest.mark.skipif(
    not GOOGLE_CLOUD_AVAILABLE, reason="google-cloud-dataproc not installed"
)
class TestDataprocCollector:
    """Tests for GCP Dataproc collector."""

    def test_initialization_basic(self):
        """Test basic Dataproc collector initialization."""
        with (
            patch(
                "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
            ),
            patch(
                "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
            ),
            patch(
                "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
            ),
        ):
            collector = DataprocCollector(
                project_id="test-project", region="us-central1"
            )

            assert collector.project_id == "test-project"
            assert collector.region == "us-central1"
            assert collector.max_clusters == 10
            assert collector.days_back == 7
            assert collector.collect_costs is True

    def test_initialization_with_config(self):
        """Test Dataproc collector initialization with custom config."""
        config = {
            "cluster_names": ["cluster1", "cluster2"],
            "cluster_labels": {"env": "prod"},
            "max_clusters": 5,
            "days_back": 14,
            "collect_costs": False,
            "include_preemptible": False,
            "preemptible_discount": 0.7,
        }

        with (
            patch(
                "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
            ),
            patch(
                "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
            ),
            patch(
                "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
            ),
        ):
            collector = DataprocCollector(
                project_id="test-project",
                region="us-west1",
                config=config,
            )

            assert collector.cluster_names == ["cluster1", "cluster2"]
            assert collector.cluster_labels == {"env": "prod"}
            assert collector.max_clusters == 5
            assert collector.days_back == 14
            assert collector.collect_costs is False
            assert collector.include_preemptible is False
            assert collector.preemptible_discount == 0.7

    def test_initialization_without_library_raises_error(self):
        """Test that initialization fails without google-cloud-dataproc."""
        with patch(
            "spark_optimizer.collectors.dataproc_collector.GOOGLE_CLOUD_AVAILABLE",
            False,
        ):
            with pytest.raises(ImportError, match="google-cloud-dataproc is required"):
                DataprocCollector(project_id="test-project")

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_validate_config_success(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test configuration validation with valid credentials."""
        mock_client_instance = Mock()
        mock_client_instance.list_clusters.return_value = iter([])
        mock_cluster_client.return_value = mock_client_instance

        collector = DataprocCollector(project_id="test-project", region="us-central1")
        result = collector.validate_config()

        assert result is True
        mock_client_instance.list_clusters.assert_called_once()

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_validate_config_failure(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test configuration validation with invalid credentials."""
        mock_client_instance = Mock()
        mock_client_instance.list_clusters.side_effect = (
            google_exceptions.PermissionDenied("Invalid credentials")
        )
        mock_cluster_client.return_value = mock_client_instance

        collector = DataprocCollector(project_id="test-project", region="us-central1")
        result = collector.validate_config()

        assert result is False

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_build_cluster_filter(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test cluster filter building."""
        config = {
            "cluster_names": ["cluster1", "cluster2"],
            "cluster_labels": {"env": "prod", "team": "data"},
        }

        collector = DataprocCollector(
            project_id="test-project",
            region="us-central1",
            config=config,
        )
        filter_str = collector._build_cluster_filter()

        assert 'clusterName = "cluster1"' in filter_str
        assert 'clusterName = "cluster2"' in filter_str
        assert 'labels.env = "prod"' in filter_str
        assert 'labels.team = "data"' in filter_str
        assert "status.state = RUNNING" in filter_str

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_get_cluster_details(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test getting cluster details."""
        # Mock cluster response
        mock_cluster = Mock()
        mock_cluster.cluster_name = "test-cluster"
        mock_cluster.cluster_uuid = "uuid-123"
        mock_cluster.status.state.name = "RUNNING"
        mock_cluster.labels = {"env": "test"}

        # Mock state_start_time
        timestamp = Timestamp()
        timestamp.FromDatetime(datetime.now())
        mock_cluster.status.state_start_time = timestamp

        # Mock config
        mock_config = Mock()

        # Master config
        mock_config.master_config.num_instances = 1
        mock_config.master_config.machine_type_uri = (
            "projects/test/zones/us-central1-a/machineTypes/n1-standard-4"
        )
        mock_config.master_config.disk_config.boot_disk_size_gb = 100

        # Worker config
        mock_config.worker_config.num_instances = 3
        mock_config.worker_config.machine_type_uri = (
            "projects/test/zones/us-central1-a/machineTypes/n1-standard-8"
        )
        mock_config.worker_config.disk_config.boot_disk_size_gb = 500

        # Secondary worker config
        mock_config.secondary_worker_config.num_instances = 2
        mock_config.secondary_worker_config.machine_type_uri = (
            "projects/test/zones/us-central1-a/machineTypes/n1-standard-4"
        )
        mock_config.secondary_worker_config.disk_config.boot_disk_size_gb = 100

        mock_cluster.config = mock_config

        mock_client_instance = Mock()
        mock_client_instance.get_cluster.return_value = mock_cluster
        mock_cluster_client.return_value = mock_client_instance

        collector = DataprocCollector(project_id="test-project", region="us-central1")
        details = collector._get_cluster_details("test-cluster")

        assert details["cluster_name"] == "test-cluster"
        assert details["cluster_uuid"] == "uuid-123"
        assert details["master_config"]["machine_type"] == "n1-standard-4"
        assert details["worker_config"]["machine_type"] == "n1-standard-8"
        assert details["worker_config"]["num_instances"] == 3
        assert details["preemptible_worker_config"]["num_instances"] == 2

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_convert_job_to_metrics(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test converting Dataproc job to metrics format."""
        cluster_details = {
            "cluster_name": "test-cluster",
            "cluster_uuid": "uuid-123",
            "project_id": "test-project",
            "region": "us-central1",
            "worker_config": {
                "num_instances": 4,
                "machine_type": "n1-standard-8",
            },
        }

        # Mock job
        mock_job = Mock()
        mock_job.reference.job_id = "job-123"

        # Set job type
        mock_job.spark_job = Mock()
        mock_job.pyspark_job = None
        mock_job.spark_sql_job = None

        # Set status
        timestamp_start = Timestamp()
        timestamp_start.FromDatetime(datetime.now() - timedelta(hours=1))
        mock_job.status.state_start_time = timestamp_start
        mock_job.status.state = dataproc_v1.JobStatus.State.DONE
        mock_job.status.details = "user@example.com"

        collector = DataprocCollector(project_id="test-project", region="us-central1")
        metrics = collector._convert_job_to_metrics(
            "test-cluster", cluster_details, mock_job
        )

        assert metrics is not None
        assert metrics["app_id"] == "test-cluster-job-123"
        assert metrics["app_name"] == "job-123"
        assert metrics["num_executors"] == 4
        assert metrics["executor_cores"] == 8  # n1-standard-8 has 8 cores
        assert metrics["tags"]["machine_type"] == "n1-standard-8"
        assert metrics["tags"]["job_type"] == "spark"
        assert metrics["tags"]["job_state"] == "DONE"

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_calculate_job_cost(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test job cost calculation."""
        cluster_details = {
            "master_config": {
                "num_instances": 1,
                "machine_type": "n1-standard-4",
            },
            "worker_config": {
                "num_instances": 3,
                "machine_type": "n1-standard-8",
            },
        }

        collector = DataprocCollector(project_id="test-project", region="us-central1")

        # 1 hour = 3,600,000 ms
        cost = collector._calculate_job_cost(cluster_details, 3600000)

        # Expected: 1 master (n1-standard-4: $0.19) + 3 workers (n1-standard-8: $0.38 each)
        # = 0.19 + (3 * 0.38) = 0.19 + 1.14 = 1.33
        assert cost == pytest.approx(1.33, rel=0.01)

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_calculate_job_cost_with_preemptible(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test cost calculation with preemptible workers."""
        cluster_details = {
            "master_config": {
                "num_instances": 1,
                "machine_type": "n1-standard-4",
            },
            "worker_config": {
                "num_instances": 2,
                "machine_type": "n1-standard-4",
            },
            "preemptible_worker_config": {
                "num_instances": 4,
                "machine_type": "n1-standard-4",
            },
        }

        collector = DataprocCollector(project_id="test-project", region="us-central1")

        cost = collector._calculate_job_cost(cluster_details, 3600000)

        # Expected: 1 master ($0.19) + 2 workers ($0.19 * 2) + 4 preemptible ($0.19 * 4 * 0.8)
        # = 0.19 + 0.38 + 0.608 = 1.178
        assert cost == pytest.approx(1.178, rel=0.01)

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_get_machine_type_recommendations_memory_intensive(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test machine type recommendations for memory-intensive workload."""
        collector = DataprocCollector(project_id="test-project", region="us-central1")

        workload = {"memory_intensive": True, "job_type": "ml"}

        recommendation = collector.get_machine_type_recommendations(
            "n1-standard-8", workload
        )

        assert recommendation["recommended_machine_type"] == "n2-highmem-8"
        assert "memory" in recommendation["reason"].lower()

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_get_machine_type_recommendations_compute_intensive(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test machine type recommendations for compute-intensive workload."""
        collector = DataprocCollector(project_id="test-project", region="us-central1")

        workload = {"compute_intensive": True, "job_type": "streaming"}

        recommendation = collector.get_machine_type_recommendations(
            "n1-standard-8", workload
        )

        assert recommendation["recommended_machine_type"] == "c2-standard-8"
        assert "compute" in recommendation["reason"].lower()

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_get_machine_type_recommendations_cost_optimized(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test machine type recommendations for cost-optimized workload."""
        collector = DataprocCollector(project_id="test-project", region="us-central1")

        workload = {"cost_optimized": True, "job_type": "batch"}

        recommendation = collector.get_machine_type_recommendations(
            "n1-standard-8", workload
        )

        assert recommendation["recommended_machine_type"] == "e2-standard-8"
        assert "cost" in recommendation["reason"].lower()

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_get_machine_type_recommendations_etl(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test machine type recommendations for ETL workload."""
        collector = DataprocCollector(project_id="test-project", region="us-central1")

        workload = {"job_type": "etl"}

        recommendation = collector.get_machine_type_recommendations(
            "n1-standard-8", workload
        )

        assert recommendation["recommended_machine_type"] == "n2-standard-8"
        assert "n2" in recommendation["reason"].lower()

    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.ClusterControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.dataproc_v1.JobControllerClient"
    )
    @patch(
        "spark_optimizer.collectors.dataproc_collector.monitoring_v3.MetricServiceClient"
    )
    def test_get_machine_type_recommendations_preemptible(
        self, mock_monitoring, mock_job_client, mock_cluster_client
    ):
        """Test preemptible worker recommendations."""
        collector = DataprocCollector(project_id="test-project", region="us-central1")

        # Batch workload should recommend preemptible
        workload = {"job_type": "batch"}
        recommendation = collector.get_machine_type_recommendations(
            "n1-standard-4", workload
        )
        assert recommendation["preemptible_recommended"] is True

        # ETL workload should recommend preemptible
        workload = {"job_type": "etl"}
        recommendation = collector.get_machine_type_recommendations(
            "n1-standard-4", workload
        )
        assert recommendation["preemptible_recommended"] is True

        # Cost-optimized should recommend preemptible
        workload = {"cost_optimized": True}
        recommendation = collector.get_machine_type_recommendations(
            "n1-standard-4", workload
        )
        assert recommendation["preemptible_recommended"] is True


@pytest.mark.skipif(GOOGLE_CLOUD_AVAILABLE, reason="Testing import error handling")
def test_dataproc_collector_import_error():
    """Test that DataprocCollector raises ImportError when google-cloud not installed."""
    pass
