"""Repository pattern for data access."""

from typing import List, Optional, Dict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
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

    def search_similar(self, criteria: Dict, limit: int = 10) -> List[SparkApplication]:
        """Search for similar applications based on criteria.

        Args:
            criteria: Dictionary of search criteria
            limit: Maximum number of results

        Returns:
            List of similar SparkApplication instances
        """
        # TODO: Implement similarity search logic
        # Consider: input size, app name pattern, executor config, etc.
        query = self.session.query(SparkApplication)

        if "app_name_pattern" in criteria:
            query = query.filter(
                SparkApplication.app_name.like(f"%{criteria['app_name_pattern']}%")
            )

        return query.limit(limit).all()


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

    def record_usage(self, recommendation_id: int):
        """Record that a recommendation was used.

        Args:
            recommendation_id: Recommendation ID
        """
        recommendation = self.session.query(JobRecommendation).get(recommendation_id)
        if recommendation:
            recommendation.used_at = datetime.utcnow()
            self.session.flush()

    def add_feedback(self, recommendation_id: int, score: float):
        """Add user feedback for a recommendation.

        Args:
            recommendation_id: Recommendation ID
            score: Feedback score (0.0 to 1.0)
        """
        recommendation = self.session.query(JobRecommendation).get(recommendation_id)
        if recommendation:
            recommendation.feedback_score = score
            self.session.flush()
