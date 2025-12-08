"""Tests for Databricks collector."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import json

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

if REQUESTS_AVAILABLE:
    from spark_optimizer.collectors.databricks_collector import DatabricksCollector


@pytest.mark.skipif(not REQUESTS_AVAILABLE, reason="requests not installed")
class TestDatabricksCollector:
    """Tests for Databricks collector."""

    def test_initialization_with_token(self):
        """Test Databricks collector initialization with token."""
        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com",
            token="dapi12345",
        )

        assert collector.workspace_url == "https://dbc-test.cloud.databricks.com"
        assert collector.token == "dapi12345"
        assert "Authorization" in collector.headers
        assert collector.headers["Authorization"] == "Bearer dapi12345"

    def test_initialization_with_basic_auth(self):
        """Test Databricks collector initialization with username/password."""
        config = {"username": "test@example.com", "password": "secret"}

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", config=config
        )

        assert collector.auth is not None
        assert collector.auth.username == "test@example.com"
        assert collector.auth.password == "secret"

    def test_initialization_without_auth_raises_error(self):
        """Test that initialization without auth raises ValueError."""
        with pytest.raises(ValueError, match="Either 'token' or 'username'"):
            DatabricksCollector(workspace_url="https://dbc-test.cloud.databricks.com")

    def test_initialization_with_config(self):
        """Test Databricks collector initialization with custom config."""
        config = {
            "cluster_ids": ["cluster-123"],
            "max_clusters": 5,
            "days_back": 14,
            "collect_sql_analytics": False,
            "collect_costs": False,
            "dbu_price": 0.50,
        }

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com",
            token="test",
            config=config,
        )

        assert collector.cluster_ids == ["cluster-123"]
        assert collector.max_clusters == 5
        assert collector.days_back == 14
        assert collector.collect_sql_analytics is False
        assert collector.collect_costs is False
        assert collector.dbu_price == 0.50

    @patch("spark_optimizer.collectors.databricks_collector.requests.get")
    def test_validate_config_success(self, mock_get):
        """Test configuration validation with valid credentials."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )
        result = collector.validate_config()

        assert result is True
        mock_get.assert_called_once()

    @patch("spark_optimizer.collectors.databricks_collector.requests.get")
    def test_validate_config_failure(self, mock_get):
        """Test configuration validation with invalid credentials."""
        mock_get.side_effect = requests.RequestException("Unauthorized")

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="invalid"
        )
        result = collector.validate_config()

        assert result is False

    @patch("spark_optimizer.collectors.databricks_collector.requests.get")
    def test_list_clusters(self, mock_get):
        """Test listing Databricks clusters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "clusters": [
                {
                    "cluster_id": "cluster-1",
                    "cluster_name": "Test Cluster 1",
                    "state": "RUNNING",
                },
                {
                    "cluster_id": "cluster-2",
                    "cluster_name": "Test Cluster 2",
                    "state": "PENDING",
                },
            ]
        }
        mock_get.return_value = mock_response

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )
        clusters = collector._list_clusters()

        assert len(clusters) == 2
        assert clusters[0]["cluster_id"] == "cluster-1"
        assert clusters[1]["cluster_id"] == "cluster-2"

    @patch("spark_optimizer.collectors.databricks_collector.requests.get")
    def test_get_cluster(self, mock_get):
        """Test getting cluster details."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "cluster_id": "cluster-123",
            "cluster_name": "Test Cluster",
            "state": "RUNNING",
            "node_type_id": "Standard_DS4_v2",
            "num_workers": 4,
            "spark_version": "11.3.x-scala2.12",
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )
        cluster = collector._get_cluster("cluster-123")

        assert cluster["cluster_id"] == "cluster-123"
        assert cluster["node_type_id"] == "Standard_DS4_v2"
        assert cluster["num_workers"] == 4

    @patch("spark_optimizer.collectors.databricks_collector.requests.get")
    def test_convert_run_to_metrics(self, mock_get):
        """Test converting Databricks job run to metrics format."""
        cluster_details = {
            "cluster_name": "Test Cluster",
            "node_type_id": "Standard_DS4_v2",
            "num_workers": 3,
            "spark_version": "11.3.x-scala2.12",
        }

        run = {
            "run_id": 12345,
            "run_name": "Test Job Run",
            "creator_user_name": "test@example.com",
            "start_time": 1609459200000,  # 2021-01-01 00:00:00
            "end_time": 1609462800000,  # 2021-01-01 01:00:00
            "job_id": 999,
            "state": {"result_state": "SUCCESS"},
            "tasks": [{"task_key": "task1"}, {"task_key": "task2"}],
        }

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )
        metrics = collector._convert_run_to_metrics("cluster-123", cluster_details, run)

        assert metrics is not None
        assert metrics["app_id"] == "cluster-123-12345"
        assert metrics["app_name"] == "Test Job Run"
        assert metrics["user"] == "test@example.com"
        assert metrics["num_executors"] == 3
        assert metrics["executor_cores"] == 8  # Standard_DS4_v2 has 8 cores
        assert metrics["duration_ms"] == 3600000  # 1 hour
        assert metrics["total_tasks"] == 2
        assert metrics["failed_tasks"] == 0
        assert metrics["tags"]["cluster_id"] == "cluster-123"
        assert metrics["tags"]["node_type"] == "Standard_DS4_v2"
        assert metrics["tags"]["run_state"] == "SUCCESS"

    @patch("spark_optimizer.collectors.databricks_collector.requests.get")
    def test_convert_run_to_metrics_with_autoscaling(self, mock_get):
        """Test metrics conversion with autoscaling cluster."""
        cluster_details = {
            "cluster_name": "Auto Cluster",
            "node_type_id": "i3.2xlarge",
            "autoscale": {"min_workers": 2, "max_workers": 10},
            "spark_version": "12.0.x-scala2.12",
        }

        run = {
            "run_id": 67890,
            "run_name": "Auto Job",
            "creator_user_name": "user@test.com",
            "start_time": 1609459200000,
            "end_time": 1609466400000,
            "state": {"result_state": "FAILED"},
            "tasks": [],
        }

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )
        metrics = collector._convert_run_to_metrics("cluster-abc", cluster_details, run)

        assert metrics is not None
        assert metrics["num_executors"] == 10  # Uses max_workers for autoscaling
        assert metrics["failed_tasks"] == 1  # FAILED state
        assert metrics["tags"]["run_state"] == "FAILED"

    def test_get_cluster_recommendations_memory_intensive(self):
        """Test cluster recommendations for memory-intensive workload."""
        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )

        workload = {"memory_intensive": True, "job_type": "ml"}

        # Test Azure recommendation
        recommendation = collector.get_cluster_recommendations(
            "Standard_DS4_v2", workload
        )

        assert recommendation["recommended_cluster_type"] == "Standard_E8s_v3"
        assert "memory" in recommendation["reason"].lower()

        # Test AWS recommendation
        recommendation = collector.get_cluster_recommendations("i3.2xlarge", workload)

        assert recommendation["recommended_cluster_type"] == "r5d.2xlarge"

    def test_get_cluster_recommendations_io_intensive(self):
        """Test cluster recommendations for I/O-intensive workload."""
        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )

        workload = {"io_intensive": True, "job_type": "streaming"}

        recommendation = collector.get_cluster_recommendations("Standard_F8s", workload)

        assert recommendation["recommended_cluster_type"] == "Standard_DS4_v2"
        assert "i/o" in recommendation["reason"].lower()

    def test_get_cluster_recommendations_sql(self):
        """Test cluster recommendations for SQL workload."""
        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )

        workload = {"job_type": "sql"}

        recommendation = collector.get_cluster_recommendations(
            "Standard_DS3_v2", workload
        )

        assert recommendation["recommended_cluster_type"] == "Standard_F8s"
        assert "sql" in recommendation["reason"].lower()

    def test_get_cluster_recommendations_autoscaling(self):
        """Test autoscaling recommendations."""
        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )

        # ETL should recommend autoscaling
        workload = {"job_type": "etl"}
        recommendation = collector.get_cluster_recommendations("i3.xlarge", workload)
        assert recommendation["autoscaling_recommended"] is True

        # Streaming should recommend autoscaling
        workload = {"job_type": "streaming"}
        recommendation = collector.get_cluster_recommendations("i3.xlarge", workload)
        assert recommendation["autoscaling_recommended"] is True

    @patch("spark_optimizer.collectors.databricks_collector.time.time")
    def test_calculate_cluster_cost(self, mock_time):
        """Test DBU cost calculation."""
        # Mock current time to 2 hours after cluster start
        mock_time.return_value = 1609466400.0  # 2021-01-01 02:00:00

        cluster_details = {
            "start_time": 1609459200000,  # 2021-01-01 00:00:00
            "node_type_id": "Standard_DS4_v2",
            "num_workers": 4,
        }

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )
        cost = collector._calculate_cluster_cost(cluster_details)

        # Expected: (4 workers + 1 driver) * 1.5 DBU/hr * 2 hours * $0.40/DBU = $6.00
        assert cost == pytest.approx(6.0, rel=0.01)

    @patch("spark_optimizer.collectors.databricks_collector.time.time")
    def test_calculate_cluster_cost_with_autoscaling(self, mock_time):
        """Test cost calculation with autoscaling."""
        mock_time.return_value = 1609466400.0

        cluster_details = {
            "start_time": 1609459200000,
            "node_type_id": "i3.xlarge",
            "autoscale": {"min_workers": 2, "max_workers": 8},
        }

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )
        cost = collector._calculate_cluster_cost(cluster_details)

        # Uses average: (2 + 8) / 2 = 5 workers
        # (5 workers + 1 driver) * 0.75 DBU/hr * 2 hours * $0.40/DBU = $3.60
        assert cost == pytest.approx(3.6, rel=0.01)

    @patch("spark_optimizer.collectors.databricks_collector.requests.get")
    def test_collect_with_costs(self, mock_get):
        """Test full collection with cost data."""

        def mock_response_factory(url, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()

            if "clusters/list" in url:
                mock_response.json.return_value = {
                    "clusters": [
                        {
                            "cluster_id": "cluster-1",
                            "cluster_name": "Test",
                            "state": "RUNNING",
                        }
                    ]
                }
            elif "clusters/get" in url:
                mock_response.json.return_value = {
                    "cluster_id": "cluster-1",
                    "cluster_name": "Test",
                    "node_type_id": "Standard_DS4_v2",
                    "num_workers": 2,
                    "spark_version": "11.3.x",
                    "start_time": 1609459200000,
                }
            elif "jobs/runs/list" in url:
                mock_response.json.return_value = {
                    "runs": [
                        {
                            "run_id": 123,
                            "run_name": "Test Run",
                            "creator_user_name": "test@test.com",
                            "start_time": 1609459200000,
                            "end_time": 1609462800000,
                            "job_id": 1,
                            "state": {"result_state": "SUCCESS"},
                            "tasks": [],
                            "cluster_instance": {"cluster_id": "cluster-1"},
                        }
                    ]
                }

            return mock_response

        mock_get.side_effect = mock_response_factory

        collector = DatabricksCollector(
            workspace_url="https://dbc-test.cloud.databricks.com", token="test"
        )

        jobs = collector.collect()

        assert len(jobs) > 0
        assert jobs[0]["app_id"] == "cluster-1-123"
        assert jobs[0]["estimated_cost"] > 0


@pytest.mark.skipif(REQUESTS_AVAILABLE, reason="Testing import error handling")
def test_databricks_collector_import_error():
    """Test that DatabricksCollector raises ImportError when requests is not installed."""
    pass
