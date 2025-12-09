"""Repository pattern for data access."""

from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
from .models import SparkApplication, SparkStage, JobRecommendation


class SparkApplicationRepository:
    """Repository for SparkApplication CRUD operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy session
        """
        self.session = session

    def create(self, app_data: Dict) -> SparkApplication:
        """Create a new Spark application record.

        Args:
            app_data: Dictionary containing application data

        Returns:
            Created SparkApplication instance
        """
        app = SparkApplication(**app_data)
        self.session.add(app)
        self.session.flush()
        return app

    def get_by_app_id(self, app_id: str) -> Optional[SparkApplication]:
        """Get application by Spark application ID.

        Args:
            app_id: Spark application ID

        Returns:
            SparkApplication instance or None
        """
        return self.session.query(SparkApplication).filter_by(app_id=app_id).first()

    def get_recent(self, days: int = 30, limit: int = 100) -> List[SparkApplication]:
        """Get recent applications.

        Args:
            days: Number of days to look back
            limit: Maximum number of results

        Returns:
            List of SparkApplication instances
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        return (
            self.session.query(SparkApplication)
            .filter(SparkApplication.submit_time >= cutoff_date)
            .order_by(SparkApplication.submit_time.desc())
            .limit(limit)
            .all()
        )

    def search_similar(
        self, criteria: Dict, limit: int = 10
    ) -> List[SparkApplication]:
        """Search for similar applications based on criteria.

        Supports multiple search criteria for finding similar jobs:
        - app_name_pattern: Partial match on application name
        - input_size_range: Tuple of (min_bytes, max_bytes) for input size
        - executor_count_range: Tuple of (min, max) for number of executors
        - duration_range: Tuple of (min_ms, max_ms) for duration
        - job_type: Tag-based job type filter
        - status: Filter by job status (completed, failed, etc.)
        - min_success_rate: Minimum task success rate (0-1)

        Args:
            criteria: Dictionary of search criteria
            limit: Maximum number of results

        Returns:
            List of similar SparkApplication instances
        """
        query = self.session.query(SparkApplication)

        # Filter by application name pattern
        if "app_name_pattern" in criteria:
            pattern = criteria["app_name_pattern"]
            query = query.filter(
                SparkApplication.app_name.ilike(f"%{pattern}%")
            )

        # Filter by input size range
        if "input_size_range" in criteria:
            min_size, max_size = criteria["input_size_range"]
            if min_size is not None:
                query = query.filter(SparkApplication.input_bytes >= min_size)
            if max_size is not None:
                query = query.filter(SparkApplication.input_bytes <= max_size)

        # Filter by input size with tolerance (for similarity matching)
        if "input_bytes" in criteria:
            target_size = criteria["input_bytes"]
            tolerance = criteria.get("size_tolerance", 0.5)  # Default 50% tolerance
            min_size = int(target_size * (1 - tolerance))
            max_size = int(target_size * (1 + tolerance))
            query = query.filter(
                and_(
                    SparkApplication.input_bytes >= min_size,
                    SparkApplication.input_bytes <= max_size,
                )
            )

        # Filter by executor count range
        if "executor_count_range" in criteria:
            min_exec, max_exec = criteria["executor_count_range"]
            if min_exec is not None:
                query = query.filter(SparkApplication.num_executors >= min_exec)
            if max_exec is not None:
                query = query.filter(SparkApplication.num_executors <= max_exec)

        # Filter by duration range
        if "duration_range" in criteria:
            min_dur, max_dur = criteria["duration_range"]
            if min_dur is not None:
                query = query.filter(SparkApplication.duration_ms >= min_dur)
            if max_dur is not None:
                query = query.filter(SparkApplication.duration_ms <= max_dur)

        # Filter by status
        if "status" in criteria:
            query = query.filter(SparkApplication.status == criteria["status"])

        # Filter for successful jobs only (no failed tasks)
        if criteria.get("successful_only", False):
            query = query.filter(
                or_(
                    SparkApplication.failed_tasks == 0,
                    SparkApplication.failed_tasks.is_(None),
                )
            )

        # Filter by minimum success rate
        if "min_success_rate" in criteria:
            min_rate = criteria["min_success_rate"]
            # Only include jobs where (total_tasks - failed_tasks) / total_tasks >= min_rate
            query = query.filter(
                SparkApplication.total_tasks > 0,
                (SparkApplication.total_tasks - func.coalesce(SparkApplication.failed_tasks, 0))
                >= SparkApplication.total_tasks * min_rate,
            )

        # Filter by date range
        if "date_from" in criteria:
            query = query.filter(SparkApplication.start_time >= criteria["date_from"])
        if "date_to" in criteria:
            query = query.filter(SparkApplication.start_time <= criteria["date_to"])

        # Filter by cluster type
        if "cluster_type" in criteria:
            query = query.filter(SparkApplication.cluster_type == criteria["cluster_type"])

        # Filter by Spark version
        if "spark_version" in criteria:
            query = query.filter(
                SparkApplication.spark_version.ilike(f"{criteria['spark_version']}%")
            )

        # Order by most recent first, or by similarity score if calculable
        if criteria.get("order_by") == "input_size" and "input_bytes" in criteria:
            # Order by closest input size
            target = criteria["input_bytes"]
            query = query.order_by(
                func.abs(SparkApplication.input_bytes - target)
            )
        else:
            # Default: order by most recent
            query = query.order_by(SparkApplication.end_time.desc())

        return query.limit(limit).all()

    def get_by_input_size_range(
        self,
        min_bytes: int,
        max_bytes: int,
        limit: int = 50,
    ) -> List[SparkApplication]:
        """Get applications within an input size range.

        Args:
            min_bytes: Minimum input size in bytes
            max_bytes: Maximum input size in bytes
            limit: Maximum number of results

        Returns:
            List of SparkApplication instances
        """
        return (
            self.session.query(SparkApplication)
            .filter(
                and_(
                    SparkApplication.input_bytes >= min_bytes,
                    SparkApplication.input_bytes <= max_bytes,
                    SparkApplication.failed_tasks == 0,  # Only successful jobs
                )
            )
            .order_by(SparkApplication.end_time.desc())
            .limit(limit)
            .all()
        )

    def get_successful_jobs(self, limit: int = 100) -> List[SparkApplication]:
        """Get successfully completed jobs.

        Args:
            limit: Maximum number of results

        Returns:
            List of successful SparkApplication instances
        """
        return (
            self.session.query(SparkApplication)
            .filter(
                or_(
                    SparkApplication.failed_tasks == 0,
                    SparkApplication.failed_tasks.is_(None),
                )
            )
            .filter(SparkApplication.duration_ms.isnot(None))
            .filter(SparkApplication.end_time.isnot(None))
            .order_by(SparkApplication.end_time.desc())
            .limit(limit)
            .all()
        )

    def update(self, app_id: str, updates: Dict) -> Optional[SparkApplication]:
        """Update an existing application.

        Args:
            app_id: Spark application ID
            updates: Dictionary of fields to update

        Returns:
            Updated SparkApplication or None if not found
        """
        app = self.get_by_app_id(app_id)
        if app:
            for key, value in updates.items():
                if hasattr(app, key):
                    setattr(app, key, value)
            app.updated_at = datetime.utcnow()
            self.session.flush()
        return app

    def delete(self, app_id: str) -> bool:
        """Delete an application by ID.

        Args:
            app_id: Spark application ID

        Returns:
            True if deleted, False if not found
        """
        app = self.get_by_app_id(app_id)
        if app:
            self.session.delete(app)
            self.session.flush()
            return True
        return False

    def get_statistics(self) -> Dict:
        """Get aggregate statistics for all applications.

        Returns:
            Dictionary with statistics
        """
        total = self.session.query(func.count(SparkApplication.id)).scalar() or 0
        avg_duration = (
            self.session.query(func.avg(SparkApplication.duration_ms)).scalar() or 0
        )
        total_input = (
            self.session.query(func.sum(SparkApplication.input_bytes)).scalar() or 0
        )
        total_output = (
            self.session.query(func.sum(SparkApplication.output_bytes)).scalar() or 0
        )

        return {
            "total_jobs": total,
            "avg_duration_ms": float(avg_duration),
            "total_input_bytes": total_input,
            "total_output_bytes": total_output,
        }


class JobRecommendationRepository:
    """Repository for JobRecommendation CRUD operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy session
        """
        self.session = session

    def create(self, recommendation_data: Dict) -> JobRecommendation:
        """Create a new recommendation record.

        Args:
            recommendation_data: Dictionary containing recommendation data

        Returns:
            Created JobRecommendation instance
        """
        recommendation = JobRecommendation(**recommendation_data)
        self.session.add(recommendation)
        self.session.flush()
        return recommendation

    def get_by_signature(self, job_signature: str) -> Optional[JobRecommendation]:
        """Get most recent recommendation for a job signature.

        Args:
            job_signature: Job signature hash

        Returns:
            Most recent JobRecommendation or None
        """
        return (
            self.session.query(JobRecommendation)
            .filter_by(job_signature=job_signature)
            .order_by(JobRecommendation.created_at.desc())
            .first()
        )

    def get_by_id(self, recommendation_id: int) -> Optional[JobRecommendation]:
        """Get recommendation by ID.

        Args:
            recommendation_id: Recommendation ID

        Returns:
            JobRecommendation or None
        """
        return self.session.query(JobRecommendation).get(recommendation_id)

    def record_usage(self, recommendation_id: int):
        """Record that a recommendation was used.

        Args:
            recommendation_id: Recommendation ID
        """
        recommendation = self.get_by_id(recommendation_id)
        if recommendation:
            recommendation.used_at = datetime.utcnow()
            self.session.flush()

    def add_feedback(self, recommendation_id: int, score: float):
        """Add user feedback for a recommendation.

        Args:
            recommendation_id: Recommendation ID
            score: Feedback score (0.0 to 1.0)
        """
        recommendation = self.get_by_id(recommendation_id)
        if recommendation:
            recommendation.feedback_score = score
            self.session.flush()

    def get_recent_recommendations(
        self, limit: int = 50, with_feedback: bool = False
    ) -> List[JobRecommendation]:
        """Get recent recommendations.

        Args:
            limit: Maximum number of results
            with_feedback: If True, only return recommendations with feedback

        Returns:
            List of JobRecommendation instances
        """
        query = self.session.query(JobRecommendation)

        if with_feedback:
            query = query.filter(JobRecommendation.feedback_score.isnot(None))

        return (
            query.order_by(JobRecommendation.created_at.desc())
            .limit(limit)
            .all()
        )

    def get_average_feedback(self) -> float:
        """Get average feedback score across all recommendations.

        Returns:
            Average feedback score or 0.0 if no feedback
        """
        result = (
            self.session.query(func.avg(JobRecommendation.feedback_score))
            .filter(JobRecommendation.feedback_score.isnot(None))
            .scalar()
        )
        return float(result) if result else 0.0


class SparkStageRepository:
    """Repository for SparkStage CRUD operations."""

    def __init__(self, session: Session):
        """Initialize repository with database session.

        Args:
            session: SQLAlchemy session
        """
        self.session = session

    def create(self, stage_data: Dict) -> SparkStage:
        """Create a new stage record.

        Args:
            stage_data: Dictionary containing stage data

        Returns:
            Created SparkStage instance
        """
        stage = SparkStage(**stage_data)
        self.session.add(stage)
        self.session.flush()
        return stage

    def get_by_application(self, application_id: int) -> List[SparkStage]:
        """Get all stages for an application.

        Args:
            application_id: Internal application ID

        Returns:
            List of SparkStage instances
        """
        return (
            self.session.query(SparkStage)
            .filter_by(application_id=application_id)
            .order_by(SparkStage.stage_id)
            .all()
        )

    def get_slowest_stages(
        self, application_id: int, limit: int = 5
    ) -> List[SparkStage]:
        """Get slowest stages for an application.

        Args:
            application_id: Internal application ID
            limit: Maximum number of stages to return

        Returns:
            List of SparkStage instances ordered by duration
        """
        return (
            self.session.query(SparkStage)
            .filter_by(application_id=application_id)
            .order_by(SparkStage.duration_ms.desc())
            .limit(limit)
            .all()
        )
