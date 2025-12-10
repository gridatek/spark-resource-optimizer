"""Tests for repository pattern implementation."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from spark_optimizer.storage.models import (
    Base,
    SparkApplication,
    SparkStage,
    JobRecommendation,
)
from spark_optimizer.storage.repository import (
    SparkApplicationRepository,
    JobRecommendationRepository,
    SparkStageRepository,
)


@pytest.fixture
def engine():
    """Create an in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def sample_app_data():
    """Sample application data for testing."""
    return {
        "app_id": "app-test-001",
        "app_name": "Test Spark Application",
        "user": "test_user",
        "status": "COMPLETED",
        "spark_version": "3.4.0",
        "start_time": datetime.utcnow() - timedelta(hours=1),
        "end_time": datetime.utcnow(),
        "duration_ms": 3600000,
        "num_executors": 10,
        "executor_cores": 4,
        "executor_memory_mb": 8192,
        "driver_memory_mb": 4096,
        "total_tasks": 1000,
        "failed_tasks": 0,
        "total_stages": 10,
        "failed_stages": 0,
        "input_bytes": 10 * 1024**3,
        "output_bytes": 5 * 1024**3,
        "shuffle_read_bytes": 2 * 1024**3,
        "shuffle_write_bytes": 2 * 1024**3,
    }


@pytest.fixture
def sample_stage_data():
    """Sample stage data for testing."""
    return {
        "stage_id": 1,
        "name": "map at TestJob.scala:42",
        "num_tasks": 100,
        "status": "COMPLETE",
        "duration_ms": 30000,
        "input_bytes": 1024**3,
        "output_bytes": 512 * 1024**2,
        "shuffle_read_bytes": 0,
        "shuffle_write_bytes": 256 * 1024**2,
    }


class TestSparkApplicationRepository:
    """Test cases for SparkApplicationRepository."""

    def test_create_application(self, session, sample_app_data):
        """Test creating a new application record."""
        repo = SparkApplicationRepository(session)

        app = repo.create(sample_app_data)

        assert app.id is not None
        assert app.app_id == "app-test-001"
        assert app.app_name == "Test Spark Application"
        assert app.num_executors == 10
        assert app.executor_memory_mb == 8192

    def test_get_by_app_id(self, session, sample_app_data):
        """Test retrieving application by ID."""
        repo = SparkApplicationRepository(session)

        # Create an application first
        created_app = repo.create(sample_app_data)
        session.commit()

        # Retrieve it
        retrieved_app = repo.get_by_app_id("app-test-001")

        assert retrieved_app is not None
        assert retrieved_app.id == created_app.id
        assert retrieved_app.app_id == "app-test-001"

    def test_get_by_app_id_not_found(self, session):
        """Test retrieving non-existent application returns None."""
        repo = SparkApplicationRepository(session)

        result = repo.get_by_app_id("nonexistent-app")

        assert result is None

    def test_search_similar_by_name_pattern(self, session, sample_app_data):
        """Test searching for similar applications by name pattern."""
        repo = SparkApplicationRepository(session)

        # Create multiple applications
        for i in range(3):
            data = sample_app_data.copy()
            data["app_id"] = f"app-test-{i:03d}"
            data["app_name"] = f"ETL Job {i}"
            repo.create(data)
        session.commit()

        # Search by pattern
        criteria = {"app_name_pattern": "ETL"}
        results = repo.search_similar(criteria)

        assert len(results) == 3
        for app in results:
            assert "ETL" in app.app_name

    def test_search_similar_by_input_size_range(self, session, sample_app_data):
        """Test searching for similar applications by input size range."""
        repo = SparkApplicationRepository(session)

        # Create applications with different input sizes
        sizes = [1 * 1024**3, 5 * 1024**3, 10 * 1024**3, 50 * 1024**3]
        for i, size in enumerate(sizes):
            data = sample_app_data.copy()
            data["app_id"] = f"app-size-{i:03d}"
            data["input_bytes"] = size
            repo.create(data)
        session.commit()

        # Search for apps with 5-15 GB input
        criteria = {"input_size_range": (4 * 1024**3, 15 * 1024**3)}
        results = repo.search_similar(criteria)

        assert len(results) == 2
        for app in results:
            assert 4 * 1024**3 <= app.input_bytes <= 15 * 1024**3

    def test_search_similar_by_input_bytes_with_tolerance(
        self, session, sample_app_data
    ):
        """Test searching for applications with input size tolerance."""
        repo = SparkApplicationRepository(session)

        # Create applications with similar input sizes
        target_size = 10 * 1024**3
        sizes = [
            target_size * 0.8,  # Within 50% tolerance
            target_size * 1.2,  # Within 50% tolerance
            target_size * 2.0,  # Outside tolerance
        ]
        for i, size in enumerate(sizes):
            data = sample_app_data.copy()
            data["app_id"] = f"app-tol-{i:03d}"
            data["input_bytes"] = int(size)
            repo.create(data)
        session.commit()

        # Search with 50% tolerance (default)
        criteria = {"input_bytes": target_size, "size_tolerance": 0.5}
        results = repo.search_similar(criteria)

        assert len(results) == 2

    def test_search_similar_successful_only(self, session, sample_app_data):
        """Test searching for only successful applications."""
        repo = SparkApplicationRepository(session)

        # Create successful and failed applications
        for i in range(3):
            data = sample_app_data.copy()
            data["app_id"] = f"app-success-{i:03d}"
            data["failed_tasks"] = 0
            repo.create(data)

        for i in range(2):
            data = sample_app_data.copy()
            data["app_id"] = f"app-failed-{i:03d}"
            data["failed_tasks"] = 10
            repo.create(data)

        session.commit()

        criteria = {"successful_only": True}
        results = repo.search_similar(criteria)

        assert len(results) == 3
        for app in results:
            assert app.failed_tasks == 0

    def test_get_recent(self, session, sample_app_data):
        """Test getting recent applications."""
        repo = SparkApplicationRepository(session)

        # Create applications with different submit times
        now = datetime.utcnow()
        for i in range(5):
            data = sample_app_data.copy()
            data["app_id"] = f"app-recent-{i:03d}"
            data["submit_time"] = now - timedelta(days=i * 10)
            repo.create(data)
        session.commit()

        # Get apps from last 30 days
        results = repo.get_recent(days=30, limit=10)

        # Should get apps from last 30 days (indices 0, 1, 2)
        assert len(results) == 3

    def test_get_by_input_size_range(self, session, sample_app_data):
        """Test getting applications by input size range."""
        repo = SparkApplicationRepository(session)

        # Create applications with different sizes
        for i in range(4):
            data = sample_app_data.copy()
            data["app_id"] = f"app-range-{i:03d}"
            data["input_bytes"] = (i + 1) * 5 * 1024**3
            data["failed_tasks"] = 0
            repo.create(data)
        session.commit()

        # Get apps with 5-15 GB input
        results = repo.get_by_input_size_range(
            min_bytes=4 * 1024**3,
            max_bytes=16 * 1024**3,
            limit=10,
        )

        assert len(results) == 3

    def test_get_successful_jobs(self, session, sample_app_data):
        """Test getting successfully completed jobs."""
        repo = SparkApplicationRepository(session)

        # Create successful and failed jobs
        for i in range(3):
            data = sample_app_data.copy()
            data["app_id"] = f"app-ok-{i:03d}"
            data["failed_tasks"] = 0
            repo.create(data)

        for i in range(2):
            data = sample_app_data.copy()
            data["app_id"] = f"app-fail-{i:03d}"
            data["failed_tasks"] = 5
            repo.create(data)

        session.commit()

        results = repo.get_successful_jobs(limit=10)

        assert len(results) == 3

    def test_update_application(self, session, sample_app_data):
        """Test updating an application record."""
        repo = SparkApplicationRepository(session)

        # Create an application
        repo.create(sample_app_data)
        session.commit()

        # Update it
        updates = {"status": "FAILED", "failed_tasks": 50}
        updated_app = repo.update("app-test-001", updates)

        assert updated_app is not None
        assert updated_app.status == "FAILED"
        assert updated_app.failed_tasks == 50

    def test_delete_application(self, session, sample_app_data):
        """Test deleting an application record."""
        repo = SparkApplicationRepository(session)

        # Create an application
        repo.create(sample_app_data)
        session.commit()

        # Verify it exists
        assert repo.get_by_app_id("app-test-001") is not None

        # Delete it
        result = repo.delete("app-test-001")
        session.commit()

        assert result is True
        assert repo.get_by_app_id("app-test-001") is None

    def test_delete_nonexistent_application(self, session):
        """Test deleting non-existent application returns False."""
        repo = SparkApplicationRepository(session)

        result = repo.delete("nonexistent-app")

        assert result is False

    def test_get_statistics(self, session, sample_app_data):
        """Test getting aggregate statistics."""
        repo = SparkApplicationRepository(session)

        # Create multiple applications
        for i in range(5):
            data = sample_app_data.copy()
            data["app_id"] = f"app-stats-{i:03d}"
            data["duration_ms"] = 1000000 * (i + 1)
            data["input_bytes"] = 1024**3 * (i + 1)
            data["output_bytes"] = 512 * 1024**2 * (i + 1)
            repo.create(data)
        session.commit()

        stats = repo.get_statistics()

        assert stats["total_jobs"] == 5
        assert stats["avg_duration_ms"] > 0
        assert stats["total_input_bytes"] > 0
        assert stats["total_output_bytes"] > 0


class TestJobRecommendationRepository:
    """Test cases for JobRecommendationRepository."""

    @pytest.fixture
    def sample_recommendation_data(self):
        """Sample recommendation data for testing."""
        return {
            "job_signature": "sig-test-001",
            "input_size_bytes": 10 * 1024**3,
            "job_type": "etl",
            "recommended_executors": 10,
            "recommended_executor_cores": 4,
            "recommended_executor_memory_mb": 8192,
            "recommended_driver_memory_mb": 4096,
            "confidence_score": 0.85,
            "recommendation_method": "similarity",
        }

    def test_create_recommendation(self, session, sample_recommendation_data):
        """Test creating a new recommendation record."""
        repo = JobRecommendationRepository(session)

        rec = repo.create(sample_recommendation_data)

        assert rec.id is not None
        assert rec.job_signature == "sig-test-001"
        assert rec.recommended_executors == 10
        assert rec.confidence_score == 0.85

    def test_get_by_signature(self, session, sample_recommendation_data):
        """Test retrieving recommendation by job signature."""
        repo = JobRecommendationRepository(session)

        # Create a recommendation
        repo.create(sample_recommendation_data)
        session.commit()

        # Retrieve it
        rec = repo.get_by_signature("sig-test-001")

        assert rec is not None
        assert rec.job_signature == "sig-test-001"

    def test_get_by_signature_returns_most_recent(
        self, session, sample_recommendation_data
    ):
        """Test that get_by_signature returns the most recent recommendation."""
        repo = JobRecommendationRepository(session)

        # Create multiple recommendations with same signature
        for i in range(3):
            data = sample_recommendation_data.copy()
            data["recommended_executors"] = 10 + i
            repo.create(data)
            session.commit()

        # Should get the most recent one
        rec = repo.get_by_signature("sig-test-001")

        assert rec is not None
        assert rec.recommended_executors == 12  # Last one created

    def test_get_by_id(self, session, sample_recommendation_data):
        """Test retrieving recommendation by ID."""
        repo = JobRecommendationRepository(session)

        # Create a recommendation
        created = repo.create(sample_recommendation_data)
        session.commit()

        # Retrieve it by ID
        rec = repo.get_by_id(created.id)

        assert rec is not None
        assert rec.id == created.id

    def test_record_usage(self, session, sample_recommendation_data):
        """Test recording recommendation usage."""
        repo = JobRecommendationRepository(session)

        # Create a recommendation
        created = repo.create(sample_recommendation_data)
        session.commit()

        assert created.used_at is None

        # Record usage
        repo.record_usage(created.id)
        session.commit()

        # Verify used_at is set
        rec = repo.get_by_id(created.id)
        assert rec.used_at is not None

    def test_add_feedback(self, session, sample_recommendation_data):
        """Test adding feedback to a recommendation."""
        repo = JobRecommendationRepository(session)

        # Create a recommendation
        created = repo.create(sample_recommendation_data)
        session.commit()

        assert created.feedback_score is None

        # Add feedback
        repo.add_feedback(created.id, 0.9)
        session.commit()

        # Verify feedback is set
        rec = repo.get_by_id(created.id)
        assert rec.feedback_score == 0.9

    def test_get_recent_recommendations(self, session, sample_recommendation_data):
        """Test getting recent recommendations."""
        repo = JobRecommendationRepository(session)

        # Create multiple recommendations
        for i in range(5):
            data = sample_recommendation_data.copy()
            data["job_signature"] = f"sig-recent-{i:03d}"
            repo.create(data)
        session.commit()

        results = repo.get_recent_recommendations(limit=3)

        assert len(results) == 3

    def test_get_recent_recommendations_with_feedback(
        self, session, sample_recommendation_data
    ):
        """Test getting recent recommendations with feedback filter."""
        repo = JobRecommendationRepository(session)

        # Create recommendations, some with feedback
        for i in range(5):
            data = sample_recommendation_data.copy()
            data["job_signature"] = f"sig-fb-{i:03d}"
            rec = repo.create(data)
            if i % 2 == 0:
                rec.feedback_score = 0.8
        session.commit()

        results = repo.get_recent_recommendations(limit=10, with_feedback=True)

        assert len(results) == 3
        for rec in results:
            assert rec.feedback_score is not None

    def test_get_average_feedback(self, session, sample_recommendation_data):
        """Test getting average feedback score."""
        repo = JobRecommendationRepository(session)

        # Create recommendations with feedback
        scores = [0.7, 0.8, 0.9, 1.0]
        for i, score in enumerate(scores):
            data = sample_recommendation_data.copy()
            data["job_signature"] = f"sig-avg-{i:03d}"
            rec = repo.create(data)
            rec.feedback_score = score
        session.commit()

        avg = repo.get_average_feedback()

        assert avg == pytest.approx(0.85, rel=0.01)

    def test_get_average_feedback_no_feedback(
        self, session, sample_recommendation_data
    ):
        """Test average feedback returns 0.0 when no feedback exists."""
        repo = JobRecommendationRepository(session)

        # Create recommendations without feedback
        for i in range(3):
            data = sample_recommendation_data.copy()
            data["job_signature"] = f"sig-nofb-{i:03d}"
            repo.create(data)
        session.commit()

        avg = repo.get_average_feedback()

        assert avg == 0.0


class TestSparkStageRepository:
    """Test cases for SparkStageRepository."""

    def test_create_stage(self, session, sample_app_data, sample_stage_data):
        """Test creating a stage record."""
        app_repo = SparkApplicationRepository(session)
        stage_repo = SparkStageRepository(session)

        # Create application first
        app = app_repo.create(sample_app_data)
        session.commit()

        # Create stage
        sample_stage_data["application_id"] = app.id
        stage = stage_repo.create(sample_stage_data)
        session.commit()

        assert stage.id is not None
        assert stage.stage_id == 1
        assert stage.application_id == app.id

    def test_get_by_application(self, session, sample_app_data, sample_stage_data):
        """Test getting all stages for an application."""
        app_repo = SparkApplicationRepository(session)
        stage_repo = SparkStageRepository(session)

        # Create application
        app = app_repo.create(sample_app_data)
        session.commit()

        # Create multiple stages
        for i in range(5):
            data = sample_stage_data.copy()
            data["stage_id"] = i
            data["application_id"] = app.id
            stage_repo.create(data)
        session.commit()

        stages = stage_repo.get_by_application(app.id)

        assert len(stages) == 5
        # Should be ordered by stage_id
        for i, stage in enumerate(stages):
            assert stage.stage_id == i

    def test_get_slowest_stages(self, session, sample_app_data, sample_stage_data):
        """Test getting slowest stages for an application."""
        app_repo = SparkApplicationRepository(session)
        stage_repo = SparkStageRepository(session)

        # Create application
        app = app_repo.create(sample_app_data)
        session.commit()

        # Create stages with different durations
        durations = [10000, 50000, 30000, 100000, 20000]
        for i, duration in enumerate(durations):
            data = sample_stage_data.copy()
            data["stage_id"] = i
            data["application_id"] = app.id
            data["duration_ms"] = duration
            stage_repo.create(data)
        session.commit()

        slowest = stage_repo.get_slowest_stages(app.id, limit=3)

        assert len(slowest) == 3
        # Should be ordered by duration descending
        assert slowest[0].duration_ms == 100000
        assert slowest[1].duration_ms == 50000
        assert slowest[2].duration_ms == 30000


class TestRepositoryEdgeCases:
    """Test edge cases and error handling."""

    def test_create_with_minimal_data(self, session):
        """Test creating application with minimal required data."""
        repo = SparkApplicationRepository(session)

        minimal_data = {"app_id": "app-minimal"}
        app = repo.create(minimal_data)

        assert app.id is not None
        assert app.app_id == "app-minimal"
        assert app.app_name is None

    def test_search_with_empty_criteria(self, session, sample_app_data):
        """Test search with empty criteria returns all apps."""
        repo = SparkApplicationRepository(session)

        # Create some applications
        for i in range(3):
            data = sample_app_data.copy()
            data["app_id"] = f"app-empty-{i:03d}"
            repo.create(data)
        session.commit()

        results = repo.search_similar({}, limit=10)

        assert len(results) == 3

    def test_search_with_multiple_criteria(self, session, sample_app_data):
        """Test search with multiple criteria combines them."""
        repo = SparkApplicationRepository(session)

        # Create applications with different characteristics
        data1 = sample_app_data.copy()
        data1["app_id"] = "app-multi-001"
        data1["app_name"] = "ETL Job"
        data1["input_bytes"] = 10 * 1024**3
        repo.create(data1)

        data2 = sample_app_data.copy()
        data2["app_id"] = "app-multi-002"
        data2["app_name"] = "ETL Job"
        data2["input_bytes"] = 50 * 1024**3
        repo.create(data2)

        data3 = sample_app_data.copy()
        data3["app_id"] = "app-multi-003"
        data3["app_name"] = "ML Job"
        data3["input_bytes"] = 10 * 1024**3
        repo.create(data3)

        session.commit()

        # Search for ETL jobs with ~10GB input
        criteria = {
            "app_name_pattern": "ETL",
            "input_size_range": (5 * 1024**3, 15 * 1024**3),
        }
        results = repo.search_similar(criteria)

        assert len(results) == 1
        assert results[0].app_id == "app-multi-001"
