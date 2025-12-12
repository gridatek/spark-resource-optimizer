"""Tests for API routes."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from flask import Flask

from spark_optimizer.api.routes import api_bp
from spark_optimizer.storage.database import Database
from spark_optimizer.storage.models import SparkApplication


@pytest.fixture
def app():
    """Create a Flask test app with routes registered."""
    flask_app = Flask(__name__)
    flask_app.config["TESTING"] = True
    flask_app.config["DATABASE_URL"] = "sqlite:///:memory:"
    flask_app.register_blueprint(api_bp, url_prefix="/api/v1")
    return flask_app


@pytest.fixture
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture
def mock_db():
    """Create a mock database."""
    with patch("spark_optimizer.api.routes.get_db") as mock:
        db = MagicMock(spec=Database)
        mock.return_value = db
        yield db


@pytest.fixture
def mock_recommender():
    """Create a mock recommender."""
    with patch("spark_optimizer.api.routes.get_recommender") as mock:
        recommender = MagicMock()
        mock.return_value = recommender
        yield recommender


@pytest.fixture
def sample_spark_application():
    """Create a sample SparkApplication for testing."""
    app = SparkApplication(
        app_id="app-test-123",
        app_name="Test Spark Job",
        user="test_user",
        status="COMPLETED",
        spark_version="3.4.0",
        submit_time=None,
        start_time=datetime(2024, 1, 1, 10, 0, 0),
        end_time=datetime(2024, 1, 1, 10, 30, 0),
        duration_ms=1800000,
        num_executors=10,
        executor_cores=4,
        executor_memory_mb=8192,
        driver_memory_mb=4096,
        total_tasks=1000,
        failed_tasks=0,
        total_stages=10,
        failed_stages=0,
        input_bytes=10 * 1024**3,
        output_bytes=5 * 1024**3,
        shuffle_read_bytes=2 * 1024**3,
        shuffle_write_bytes=2 * 1024**3,
        memory_spilled_bytes=0,
        disk_spilled_bytes=0,
        executor_run_time_ms=1500000,
        executor_cpu_time_ms=1400000,
        jvm_gc_time_ms=50000,
        peak_memory_usage=6 * 1024**3,
        cluster_type="standalone",
        instance_type="m5.xlarge",
        estimated_cost=5.50,
        tags={"job_type": "etl"},
        spark_configs={"spark.sql.shuffle.partitions": "200"},
    )
    return app


class TestAPIRoutes:
    """Test cases for API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint returns healthy status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "healthy"
        assert data["service"] == "spark-resource-optimizer"

    def test_get_recommendation_success(self, client, mock_recommender):
        """Test successful recommendation request."""
        mock_recommender.recommend.return_value = {
            "configuration": {
                "num_executors": 10,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "driver_memory_mb": 4096,
            },
            "confidence": 0.85,
            "metadata": {
                "method": "similarity",
                "job_type": "etl",
            },
        }

        request_data = {
            "input_size_gb": 10.0,
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
        assert data["configuration"]["num_executors"] == 10
        assert data["confidence"] == 0.85

    def test_get_recommendation_missing_input_size(self, client):
        """Test recommendation request without input size returns error."""
        request_data = {
            "job_type": "etl",
            "priority": "balanced",
        }

        response = client.post(
            "/api/v1/recommend",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_get_recommendation_invalid_priority(self, client, mock_recommender):
        """Test recommendation request with invalid priority returns error."""
        request_data = {
            "input_size_gb": 10.0,
            "priority": "invalid_priority",
        }

        response = client.post(
            "/api/v1/recommend",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "priority" in data["error"].lower()

    def test_list_jobs_empty(self, client, mock_db):
        """Test listing jobs when database is empty."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_query.count.return_value = 0
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = (
            []
        )
        mock_session.query.return_value = mock_query

        response = client.get("/api/v1/jobs")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert "jobs" in data
        assert data["total"] == 0
        assert len(data["jobs"]) == 0

    def test_list_jobs_with_results(self, client, mock_db, sample_spark_application):
        """Test listing jobs returns job data."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            sample_spark_application
        ]
        mock_session.query.return_value = mock_query

        response = client.get("/api/v1/jobs")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["total"] == 1
        assert len(data["jobs"]) == 1
        assert data["jobs"][0]["app_id"] == "app-test-123"

    def test_list_jobs_with_filters(self, client, mock_db, sample_spark_application):
        """Test listing jobs with query filters."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            sample_spark_application
        ]
        mock_session.query.return_value = mock_query

        response = client.get("/api/v1/jobs?limit=10&offset=0&app_name=Test")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_get_job_details_success(self, client, mock_db, sample_spark_application):
        """Test getting job details by app_id."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = (
            sample_spark_application
        )

        response = client.get("/api/v1/jobs/app-test-123")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["app_id"] == "app-test-123"
        assert data["app_name"] == "Test Spark Job"
        assert "configuration" in data
        assert "metrics" in data

    def test_get_job_details_not_found(self, client, mock_db):
        """Test getting job details for non-existent job."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        response = client.get("/api/v1/jobs/nonexistent-app")

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data

    def test_collect_missing_source_type(self, client):
        """Test collect request without source_type returns error."""
        request_data = {
            "source_path": "/path/to/logs",
        }

        response = client.post(
            "/api/v1/collect",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_collect_event_log_missing_path(self, client):
        """Test collect event_log without source_path returns error."""
        request_data = {
            "source_type": "event_log",
        }

        response = client.post(
            "/api/v1/collect",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert "source_path" in data["error"]

    def test_collect_unknown_source_type(self, client):
        """Test collect with unknown source_type returns error."""
        request_data = {
            "source_type": "unknown_source",
            "source_path": "/path/to/data",
        }

        response = client.post(
            "/api/v1/collect",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data
        assert (
            "unknown_source" in data["error"].lower()
            or "unknown" in data["error"].lower()
        )

    @patch("spark_optimizer.api.routes.JobAnalyzer")
    def test_analyze_job_success(
        self, mock_analyzer_class, client, mock_db, sample_spark_application
    ):
        """Test analyzing a job returns analysis results."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = (
            sample_spark_application
        )

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_job.return_value = {
            "resource_efficiency": {
                "cpu_efficiency": 0.85,
                "memory_efficiency": 0.75,
            },
            "bottlenecks": ["shuffle"],
            "issues": [
                {
                    "type": "high_gc_time",
                    "severity": "warning",
                    "description": "GC time is 10% of total time",
                    "recommendation": "Increase executor memory",
                }
            ],
            "health_score": 75.0,
        }

        with patch(
            "spark_optimizer.api.routes.get_analyzer", return_value=mock_analyzer
        ):
            response = client.get("/api/v1/analyze/app-test-123")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["app_id"] == "app-test-123"
        assert "analysis" in data
        assert "suggestions" in data

    def test_analyze_job_not_found(self, client, mock_db):
        """Test analyzing non-existent job returns 404."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.filter.return_value.first.return_value = None

        with patch("spark_optimizer.api.routes.get_analyzer"):
            response = client.get("/api/v1/analyze/nonexistent-app")

        assert response.status_code == 404
        data = json.loads(response.data)
        assert "error" in data

    def test_submit_feedback_success(self, client, mock_db):
        """Test submitting feedback for a recommendation."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        mock_repo = MagicMock()
        with patch(
            "spark_optimizer.storage.repository.JobRecommendationRepository",
            return_value=mock_repo,
        ):
            request_data = {
                "recommendation_id": 1,
                "satisfaction_score": 0.9,
            }

            response = client.post(
                "/api/v1/feedback",
                data=json.dumps(request_data),
                content_type="application/json",
            )

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["status"] == "success"

    def test_submit_feedback_missing_id(self, client):
        """Test submitting feedback without recommendation_id returns error."""
        request_data = {
            "satisfaction_score": 0.9,
        }

        response = client.post(
            "/api/v1/feedback",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_submit_feedback_invalid_score(self, client, mock_db):
        """Test submitting feedback with invalid score returns error."""
        request_data = {
            "recommendation_id": 1,
            "satisfaction_score": 1.5,  # Invalid: should be 0.0 to 1.0
        }

        response = client.post(
            "/api/v1/feedback",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 400
        data = json.loads(response.data)
        assert "error" in data

    def test_get_stats_empty_database(self, client, mock_db):
        """Test getting stats from empty database."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session
        mock_session.query.return_value.scalar.return_value = 0
        mock_session.query.return_value.group_by.return_value.all.return_value = []

        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["total_jobs"] == 0

    def test_get_stats_with_data(self, client, mock_db):
        """Test getting stats with data in database."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        # Mock different query results
        mock_session.query.return_value.scalar.side_effect = [
            100,  # total_jobs
            300000,  # avg_duration
            100 * 1024**4,  # total_input
            50 * 1024**4,  # total_output
        ]
        mock_session.query.return_value.group_by.return_value.all.return_value = [
            ("etl_job", 50),
            ("ml_job", 30),
            ("streaming_job", 20),
        ]

        response = client.get("/api/v1/stats")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["total_jobs"] == 100


class TestRecommendationValidation:
    """Test input validation for recommendation endpoint."""

    def test_empty_request_body(self, client):
        """Test recommendation with empty request body."""
        response = client.post(
            "/api/v1/recommend",
            data=json.dumps({}),
            content_type="application/json",
        )

        assert response.status_code == 400

    def test_no_content_type(self, client):
        """Test recommendation without content-type header."""
        response = client.post("/api/v1/recommend", data='{"input_size_gb": 10}')

        # Flask should handle this gracefully
        assert response.status_code in [400, 415]

    def test_input_size_bytes_format(self, client, mock_recommender):
        """Test recommendation with input_size_bytes instead of input_size_gb."""
        mock_recommender.recommend.return_value = {
            "configuration": {
                "num_executors": 10,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "driver_memory_mb": 4096,
            },
            "confidence": 0.85,
        }

        request_data = {
            "input_size_bytes": 10 * 1024**3,  # 10 GB in bytes
            "priority": "balanced",
        }

        response = client.post(
            "/api/v1/recommend",
            data=json.dumps(request_data),
            content_type="application/json",
        )

        assert response.status_code == 200


class TestJobFiltering:
    """Test job listing with various filters."""

    def test_filter_by_date_range(self, client, mock_db, sample_spark_application):
        """Test filtering jobs by date range."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            sample_spark_application
        ]
        mock_session.query.return_value = mock_query

        response = client.get(
            "/api/v1/jobs?date_from=2024-01-01T00:00:00Z&date_to=2024-12-31T23:59:59Z"
        )

        assert response.status_code == 200

    def test_filter_by_duration(self, client, mock_db, sample_spark_application):
        """Test filtering jobs by duration range."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            sample_spark_application
        ]
        mock_session.query.return_value = mock_query

        response = client.get("/api/v1/jobs?min_duration=60&max_duration=3600")

        assert response.status_code == 200

    def test_pagination(self, client, mock_db, sample_spark_application):
        """Test job listing pagination."""
        mock_session = MagicMock()
        mock_db.get_session.return_value.__enter__.return_value = mock_session

        mock_query = MagicMock()
        mock_query.count.return_value = 100
        mock_query.order_by.return_value.offset.return_value.limit.return_value.all.return_value = [
            sample_spark_application
        ]
        mock_session.query.return_value = mock_query

        response = client.get("/api/v1/jobs?limit=10&offset=20")

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["limit"] == 10
        assert data["offset"] == 20
