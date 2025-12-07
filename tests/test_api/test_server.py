"""Comprehensive tests for REST API server endpoints."""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from flask import Flask

from spark_optimizer.api.server import app, init_app
from spark_optimizer.storage.models import SparkApplication
from spark_optimizer.analyzer.rule_engine import Severity


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config["TESTING"] = True

    # Initialize with test database
    init_app("sqlite:///:memory:")

    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_job_data():
    """Sample job data for testing."""
    return {
        "app_id": "app-test-123",
        "app_name": "Test Spark Job",
        "user": "test_user",
        "duration_ms": 300000,
        "num_executors": 10,
        "executor_cores": 4,
        "executor_memory_mb": 8192,
        "driver_memory_mb": 4096,
        "total_tasks": 1000,
        "failed_tasks": 0,
        "total_stages": 10,
        "input_bytes": 10 * 1024**3,
        "output_bytes": 5 * 1024**3,
        "shuffle_read_bytes": 2 * 1024**3,
        "shuffle_write_bytes": 2 * 1024**3,
        "disk_spilled_bytes": 0,
        "memory_spilled_bytes": 0,
        "executor_run_time_ms": 250000,
        "jvm_gc_time_ms": 10000,
    }


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_success(self, client):
        """Test health check returns 200."""
        response = client.get("/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert "version" in data


class TestRecommendEndpoint:
    """Tests for /api/v1/recommend endpoint."""

    def test_recommend_success(self, client):
        """Test successful recommendation request."""
        request_data = {
            "input_size_bytes": 10 * 1024**3,
            "job_type": "etl",
            "priority": "balanced",
        }

        response = client.post(
            "/api/v1/recommend",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert "configuration" in data
        assert "confidence" in data
        assert "metadata" in data

        config = data["configuration"]
        assert config["num_executors"] > 0
        assert config["executor_cores"] > 0
        assert config["executor_memory_mb"] > 0
        assert config["driver_memory_mb"] > 0

    def test_recommend_missing_input_size(self, client):
        """Test recommendation request without input_size_bytes."""
        request_data = {"job_type": "etl", "priority": "balanced"}

        response = client.post(
            "/api/v1/recommend",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_recommend_invalid_priority(self, client):
        """Test recommendation request with invalid priority."""
        request_data = {
            "input_size_bytes": 10 * 1024**3,
            "priority": "super_fast",  # Invalid priority
        }

        response = client.post(
            "/api/v1/recommend",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        # Should still work but default to balanced
        assert response.status_code in [200, 400]

    def test_recommend_different_job_types(self, client):
        """Test recommendations for different job types."""
        job_types = ["etl", "ml", "streaming"]

        for job_type in job_types:
            request_data = {
                "input_size_bytes": 10 * 1024**3,
                "job_type": job_type,
                "priority": "balanced",
            }

            response = client.post(
                "/api/v1/recommend",
                data=json.dumps(request_data),
                content_type="application/json",
            )

            assert response.status_code == 200
            data = json.loads(response.data)
            assert data["metadata"]["job_type"] == job_type


class TestJobsEndpoint:
    """Tests for /api/v1/jobs endpoints."""

    def test_list_jobs_empty(self, client):
        """Test listing jobs when database is empty."""
        response = client.get("/api/v1/jobs")

        assert response.status_code == 200
        data = json.loads(response.data)

        assert "jobs" in data
        assert "total" in data
        assert "limit" in data
        assert data["total"] == 0
        assert len(data["jobs"]) == 0

    def test_get_job_not_found(self, client):
        """Test getting a job that doesn't exist."""
        response = client.get("/api/v1/jobs/nonexistent-app-id")

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


class TestAnalyzeEndpoint:
    """Tests for /api/v1/jobs/<app_id>/analyze endpoint."""

    @patch("spark_optimizer.api.server.db")
    @patch("spark_optimizer.api.server.rule_based_recommender")
    def test_analyze_job_success(self, mock_recommender, mock_db, client):
        """Test successful job analysis."""
        # Mock database session
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        # Create mock job
        mock_job = Mock(spec=SparkApplication)
        mock_job.app_id = "app-test-123"
        mock_job.app_name = "Test Job"
        mock_job.input_bytes = 10 * 1024**3
        mock_job.disk_spilled_bytes = 0
        mock_job.memory_spilled_bytes = 0
        mock_job.num_executors = 10
        mock_job.executor_cores = 4
        mock_job.executor_memory_mb = 8192
        mock_job.driver_memory_mb = 4096
        mock_job.duration_ms = 300000
        mock_job.jvm_gc_time_ms = 10000
        mock_job.total_tasks = 1000
        mock_job.failed_tasks = 0
        mock_job.total_stages = 10
        mock_job.shuffle_read_bytes = 0
        mock_job.shuffle_write_bytes = 0
        mock_job.output_bytes = 0

        mock_session.query.return_value.filter.return_value.first.return_value = (
            mock_job
        )

        # Mock recommender analysis
        mock_recommender.analyze_job.return_value = {
            "app_id": "app-test-123",
            "app_name": "Test Job",
            "analysis": {
                "total_recommendations": 0,
                "critical": 0,
                "warnings": 0,
                "info": 0,
                "health_score": 100.0,
            },
            "current_configuration": {
                "num_executors": 10,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "driver_memory_mb": 4096,
            },
            "recommended_configuration": {
                "num_executors": 10,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "driver_memory_mb": 4096,
            },
            "spark_configs": {},
            "recommendations": [],
        }

        response = client.get("/api/v1/jobs/app-test-123/analyze")

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["app_id"] == "app-test-123"
        assert "analysis" in data
        assert "recommendations" in data
        assert data["analysis"]["health_score"] == 100.0

    def test_analyze_job_not_found(self, client):
        """Test analyzing a job that doesn't exist."""
        response = client.get("/api/v1/jobs/nonexistent-app/analyze")

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


class TestCollectEndpoint:
    """Tests for /api/v1/collect endpoint."""

    def test_collect_missing_url(self, client):
        """Test collect request without history_server_url."""
        request_data = {"max_apps": 100}

        response = client.post(
            "/api/v1/collect",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "history_server_url" in data["error"].lower()

    @patch("spark_optimizer.api.server.HistoryServerCollector")
    def test_collect_invalid_server(self, mock_collector_class, client):
        """Test collect with unreachable History Server."""
        # Mock collector that fails validation
        mock_collector = Mock()
        mock_collector.validate_config.return_value = False
        mock_collector_class.return_value = mock_collector

        request_data = {"history_server_url": "http://localhost:18080"}

        response = client.post(
            "/api/v1/collect",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 503
        data = json.loads(response.data)
        assert "error" in data

    @patch("spark_optimizer.api.server.HistoryServerCollector")
    @patch("spark_optimizer.api.server.db")
    def test_collect_success(
        self, mock_db, mock_collector_class, client, sample_job_data
    ):
        """Test successful job collection."""
        # Mock collector
        mock_collector = Mock()
        mock_collector.validate_config.return_value = True
        mock_collector.collect.return_value = [sample_job_data]
        mock_collector_class.return_value = mock_collector

        # Mock database session
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = (
            None  # No existing job
        )

        request_data = {"history_server_url": "http://localhost:18080", "max_apps": 10}

        response = client.post(
            "/api/v1/collect",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200
        data = json.loads(response.data)

        assert data["success"] is True
        assert data["collected"] == 1
        assert data["failed"] == 0


class TestCompareEndpoint:
    """Tests for /api/v1/compare endpoint."""

    def test_compare_missing_app_ids(self, client):
        """Test compare request without app_ids."""
        request_data = {}

        response = client.post(
            "/api/v1/compare",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_compare_too_few_jobs(self, client):
        """Test compare with less than 2 jobs."""
        request_data = {"app_ids": ["app-1"]}

        response = client.post(
            "/api/v1/compare",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "at least 2" in data["error"].lower()

    def test_compare_too_many_jobs(self, client):
        """Test compare with more than 10 jobs."""
        request_data = {"app_ids": [f"app-{i}" for i in range(15)]}

        response = client.post(
            "/api/v1/compare",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "maximum" in data["error"].lower()

    def test_compare_jobs_not_found(self, client):
        """Test compare with non-existent jobs."""
        request_data = {"app_ids": ["nonexistent-1", "nonexistent-2"]}

        response = client.post(
            "/api/v1/compare",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data


class TestStatsEndpoint:
    """Tests for /api/v1/stats endpoint."""

    def test_get_stats_empty_database(self, client):
        """Test getting stats from empty database."""
        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = json.loads(response.data)

        assert "total_jobs" in data
        assert "avg_duration_ms" in data or "avg_duration" in data
