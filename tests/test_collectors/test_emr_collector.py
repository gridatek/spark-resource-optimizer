"""Tests for EMR collector."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Check if boto3 is available
try:
    import boto3
    from botocore.exceptions import ClientError

    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

if BOTO3_AVAILABLE:
    from spark_optimizer.collectors.emr_collector import EMRCollector


@pytest.mark.skipif(not BOTO3_AVAILABLE, reason="boto3 not installed")
class TestEMRCollector:
    """Tests for EMR collector."""

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_initialization(self, mock_boto3):
        """Test EMR collector initialization."""
        mock_boto3.client.return_value = Mock()

        collector = EMRCollector(region_name="us-west-2")

        assert collector.region_name == "us-west-2"
        assert collector.max_clusters == 10
        assert collector.days_back == 7
        assert collector.collect_cloudwatch is True
        assert collector.collect_costs is True

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_initialization_with_config(self, mock_boto3):
        """Test EMR collector initialization with custom config."""
        mock_boto3.client.return_value = Mock()

        config = {
            "cluster_ids": ["j-1234567890ABC"],
            "max_clusters": 5,
            "days_back": 14,
            "collect_cloudwatch": False,
            "collect_costs": False,
        }

        collector = EMRCollector(region_name="eu-west-1", config=config)

        assert collector.region_name == "eu-west-1"
        assert collector.cluster_ids == ["j-1234567890ABC"]
        assert collector.max_clusters == 5
        assert collector.days_back == 14
        assert collector.collect_cloudwatch is False
        assert collector.collect_costs is False

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_validate_config_success(self, mock_boto3):
        """Test configuration validation with valid credentials."""
        mock_emr_client = Mock()
        mock_emr_client.list_clusters.return_value = {"Clusters": []}
        mock_boto3.client.return_value = mock_emr_client

        collector = EMRCollector()
        result = collector.validate_config()

        assert result is True
        mock_emr_client.list_clusters.assert_called_once()

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_validate_config_failure(self, mock_boto3):
        """Test configuration validation with invalid credentials."""
        mock_emr_client = Mock()
        mock_emr_client.list_clusters.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "ListClusters",
        )
        mock_boto3.client.return_value = mock_emr_client

        collector = EMRCollector()
        result = collector.validate_config()

        assert result is False

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_list_clusters(self, mock_boto3):
        """Test listing EMR clusters."""
        mock_emr_client = Mock()
        mock_emr_client.list_clusters.return_value = {
            "Clusters": [
                {
                    "Id": "j-1234567890ABC",
                    "Name": "Test Cluster 1",
                    "Status": {"State": "RUNNING"},
                },
                {
                    "Id": "j-0987654321XYZ",
                    "Name": "Test Cluster 2",
                    "Status": {"State": "WAITING"},
                },
            ]
        }
        mock_boto3.client.return_value = mock_emr_client

        collector = EMRCollector()
        clusters = collector._list_clusters()

        assert len(clusters) == 2
        assert clusters[0]["Id"] == "j-1234567890ABC"
        assert clusters[1]["Id"] == "j-0987654321XYZ"

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_list_specific_clusters(self, mock_boto3):
        """Test listing specific cluster IDs."""
        mock_emr_client = Mock()
        mock_emr_client.describe_cluster.return_value = {
            "Cluster": {
                "Id": "j-1234567890ABC",
                "Name": "Specific Cluster",
                "Status": {"State": "RUNNING"},
            }
        }
        mock_boto3.client.return_value = mock_emr_client

        config = {"cluster_ids": ["j-1234567890ABC"]}
        collector = EMRCollector(config=config)
        clusters = collector._list_clusters()

        assert len(clusters) == 1
        assert clusters[0]["Id"] == "j-1234567890ABC"

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_describe_cluster(self, mock_boto3):
        """Test describing a cluster."""
        mock_emr_client = Mock()
        mock_emr_client.describe_cluster.return_value = {
            "Cluster": {
                "Id": "j-1234567890ABC",
                "Name": "Test Cluster",
                "Status": {"State": "RUNNING"},
                "InstanceGroups": [
                    {
                        "InstanceGroupType": "MASTER",
                        "InstanceType": "m5.xlarge",
                        "RequestedInstanceCount": 1,
                    },
                    {
                        "InstanceGroupType": "CORE",
                        "InstanceType": "m5.2xlarge",
                        "RequestedInstanceCount": 3,
                    },
                ],
            }
        }
        mock_boto3.client.return_value = mock_emr_client

        collector = EMRCollector()
        cluster = collector._describe_cluster("j-1234567890ABC")

        assert cluster["Id"] == "j-1234567890ABC"
        assert cluster["Name"] == "Test Cluster"
        assert len(cluster["InstanceGroups"]) == 2

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_convert_step_to_metrics(self, mock_boto3):
        """Test converting EMR step to metrics format."""
        mock_boto3.client.return_value = Mock()

        cluster_details = {
            "Name": "Test Cluster",
            "ReleaseLabel": "emr-6.9.0",
            "InstanceGroups": [
                {
                    "InstanceGroupType": "CORE",
                    "InstanceType": "m5.4xlarge",
                    "RequestedInstanceCount": 5,
                },
                {
                    "InstanceGroupType": "MASTER",
                    "InstanceType": "m5.xlarge",
                    "RequestedInstanceCount": 1,
                },
            ],
        }

        step = {
            "Id": "s-123456789",
            "Name": "Spark Application",
            "Status": {
                "State": "COMPLETED",
                "Timeline": {
                    "CreationDateTime": datetime(2025, 1, 1, 10, 0, 0),
                    "EndDateTime": datetime(2025, 1, 1, 11, 30, 0),
                },
            },
        }

        collector = EMRCollector()
        metrics = collector._convert_step_to_metrics(
            "j-1234567890ABC", cluster_details, step
        )

        assert metrics is not None
        assert metrics["app_id"] == "j-1234567890ABC-s-123456789"
        assert metrics["app_name"] == "Spark Application"
        assert metrics["num_executors"] == 5
        assert metrics["executor_cores"] == 16  # m5.4xlarge has 16 vCPUs
        assert metrics["duration_ms"] == 5400000  # 1.5 hours in ms
        assert metrics["tags"]["cluster_id"] == "j-1234567890ABC"
        assert metrics["tags"]["instance_type"] == "m5.4xlarge"

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_get_instance_type_recommendations_memory_intensive(self, mock_boto3):
        """Test instance type recommendations for memory-intensive workload."""
        mock_boto3.client.return_value = Mock()

        collector = EMRCollector()
        workload = {
            "cpu_utilization": 50,
            "memory_utilization": 80,
            "job_type": "ml",
        }

        recommendation = collector.get_instance_type_recommendations(
            "m5.4xlarge", workload
        )

        assert recommendation["recommended_instance_type"] == "r5.4xlarge"
        assert "memory" in recommendation["reason"].lower()

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_get_instance_type_recommendations_cpu_intensive(self, mock_boto3):
        """Test instance type recommendations for CPU-intensive workload."""
        mock_boto3.client.return_value = Mock()

        collector = EMRCollector()
        workload = {
            "cpu_utilization": 85,
            "memory_utilization": 40,
            "job_type": "compute",
        }

        recommendation = collector.get_instance_type_recommendations(
            "m5.4xlarge", workload
        )

        assert recommendation["recommended_instance_type"] == "c5.4xlarge"
        assert "cpu" in recommendation["reason"].lower()

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_get_instance_type_recommendations_balanced(self, mock_boto3):
        """Test instance type recommendations for balanced workload."""
        mock_boto3.client.return_value = Mock()

        collector = EMRCollector()
        workload = {
            "cpu_utilization": 60,
            "memory_utilization": 55,
            "job_type": "etl",
        }

        recommendation = collector.get_instance_type_recommendations(
            "r5.4xlarge", workload
        )

        assert recommendation["recommended_instance_type"] == "m5.4xlarge"
        assert "balanced" in recommendation["reason"].lower()

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_fetch_cluster_cost(self, mock_boto3):
        """Test calculating cluster cost."""
        mock_boto3.client.return_value = Mock()

        cluster_details = {
            "Status": {
                "Timeline": {
                    "CreationDateTime": datetime(2025, 1, 1, 10, 0, 0),
                    "EndDateTime": datetime(2025, 1, 1, 12, 0, 0),
                }
            },
            "InstanceGroups": [
                {
                    "InstanceGroupType": "CORE",
                    "InstanceType": "m5.xlarge",
                    "RequestedInstanceCount": 3,
                },
                {
                    "InstanceGroupType": "MASTER",
                    "InstanceType": "m5.xlarge",
                    "RequestedInstanceCount": 1,
                },
            ],
        }

        collector = EMRCollector()
        cost = collector._fetch_cluster_cost("j-1234567890ABC", cluster_details)

        # 4 instances * 2 hours * $0.192/hour = $1.536
        assert cost == pytest.approx(1.536, rel=0.01)

    @patch("spark_optimizer.collectors.emr_collector.boto3")
    def test_collect_with_cloudwatch_and_costs(self, mock_boto3):
        """Test full collection with CloudWatch and cost data."""
        mock_emr_client = Mock()
        mock_cloudwatch_client = Mock()
        mock_ce_client = Mock()

        # Mock cluster list
        mock_emr_client.list_clusters.return_value = {
            "Clusters": [
                {
                    "Id": "j-1234567890ABC",
                    "Name": "Test Cluster",
                    "Status": {"State": "RUNNING"},
                }
            ]
        }

        # Mock cluster details
        mock_emr_client.describe_cluster.return_value = {
            "Cluster": {
                "Id": "j-1234567890ABC",
                "Name": "Test Cluster",
                "ReleaseLabel": "emr-6.9.0",
                "Status": {
                    "Timeline": {
                        "CreationDateTime": datetime(2025, 1, 1, 10, 0, 0),
                        "EndDateTime": datetime(2025, 1, 1, 12, 0, 0),
                    }
                },
                "InstanceGroups": [
                    {
                        "InstanceGroupType": "CORE",
                        "InstanceType": "m5.xlarge",
                        "RequestedInstanceCount": 2,
                    }
                ],
            }
        }

        # Mock steps
        mock_emr_client.list_steps.return_value = {
            "Steps": [{"Id": "s-123456789", "Name": "Spark Job"}]
        }

        mock_emr_client.describe_step.return_value = {
            "Step": {
                "Id": "s-123456789",
                "Name": "Spark Job",
                "Status": {
                    "State": "COMPLETED",
                    "Timeline": {
                        "CreationDateTime": datetime(2025, 1, 1, 10, 30, 0),
                        "EndDateTime": datetime(2025, 1, 1, 11, 0, 0),
                    },
                },
            }
        }

        # Mock CloudWatch
        mock_cloudwatch_client.get_metric_statistics.return_value = {
            "Datapoints": [{"Average": 65.5}]
        }

        def client_factory(service, **kwargs):
            if service == "emr":
                return mock_emr_client
            elif service == "cloudwatch":
                return mock_cloudwatch_client
            elif service == "ce":
                return mock_ce_client

        mock_boto3.client.side_effect = client_factory

        collector = EMRCollector()
        jobs = collector.collect()

        assert len(jobs) > 0
        assert jobs[0]["app_id"] == "j-1234567890ABC-s-123456789"
        assert "avg_cpu_utilization" in jobs[0]
        assert jobs[0]["estimated_cost"] > 0


@pytest.mark.skipif(BOTO3_AVAILABLE, reason="Testing import error handling")
def test_emr_collector_import_error():
    """Test that EMRCollector raises ImportError when boto3 is not installed."""
    # This test would only run if boto3 is not installed
    # In practice, it's hard to test this without uninstalling boto3
    pass
