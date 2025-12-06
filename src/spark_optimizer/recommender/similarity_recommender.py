"""Similarity-based recommender using historical job matching."""

from typing import TYPE_CHECKING, Dict, List, Optional
from .base_recommender import BaseRecommender

if TYPE_CHECKING:
    from spark_optimizer.storage.database import Database


class SimilarityRecommender(BaseRecommender):
    """Recommends resources based on similar historical jobs."""

    def __init__(self, db: Optional["Database"] = None, config: Optional[Dict] = None):
        """Initialize similarity-based recommender.

        Args:
            db: Optional Database instance for loading historical jobs
            config: Optional configuration dictionary
        """
        super().__init__(config)
        self.db = db
        self.historical_jobs: List[Dict] = []
        self.min_similarity_threshold = (
            config.get("min_similarity", 0.7) if config else 0.7
        )

    def recommend(
        self,
        input_size_bytes: int,
        job_type: Optional[str] = None,
        sla_minutes: Optional[int] = None,
        budget_dollars: Optional[float] = None,
        priority: str = "balanced",
    ) -> Dict:
        """Generate recommendations based on similar jobs.

        Args:
            input_size_bytes: Expected input data size in bytes
            job_type: Type of job (e.g., etl, ml, streaming)
            sla_minutes: Maximum acceptable duration in minutes
            budget_dollars: Maximum acceptable cost in dollars
            priority: Optimization priority (performance, cost, or balanced)

        Returns:
            Recommendation dictionary
        """
        # Build job requirements dict for internal use
        job_requirements = {
            "input_size_bytes": input_size_bytes,
            "job_type": job_type,
            "sla_minutes": sla_minutes,
            "budget_dollars": budget_dollars,
            "priority": priority,
        }

        # TODO: Implement similarity-based recommendation
        # 1. Find similar jobs from history
        # 2. Analyze their resource usage
        # 3. Recommend based on successful configurations
        # 4. Adjust for differences in scale

        if not self.historical_jobs:
            return self._fallback_recommendation(job_requirements)

        # Placeholder implementation
        similar_jobs = self._find_similar_jobs(job_requirements)

        if not similar_jobs:
            return self._fallback_recommendation(job_requirements)

        # Average the configurations of similar jobs
        avg_config = self._average_configurations(similar_jobs)

        return self._create_recommendation_response(
            executor_cores=avg_config["executor_cores"],
            executor_memory_mb=avg_config["executor_memory_mb"],
            num_executors=avg_config["num_executors"],
            driver_memory_mb=avg_config["driver_memory_mb"],
            confidence=0.8,
            metadata={
                "method": "similarity",
                "similar_jobs_count": len(similar_jobs),
            },
        )

    def train(self, historical_jobs: List[Dict]):
        """Train with historical job data.

        Args:
            historical_jobs: List of historical job data
        """
        # TODO: Process and store historical jobs
        self.historical_jobs = historical_jobs

    def _find_similar_jobs(self, job_requirements: Dict, top_k: int = 5) -> List[Dict]:
        """Find similar jobs from historical data.

        Args:
            job_requirements: Job requirements
            top_k: Number of similar jobs to find

        Returns:
            List of similar jobs
        """
        # TODO: Implement similarity search
        # - Use JobSimilarityCalculator
        # - Filter by minimum similarity threshold
        return []

    def _average_configurations(self, jobs: List[Dict]) -> Dict:
        """Calculate average configuration from jobs.

        Args:
            jobs: List of job data

        Returns:
            Averaged configuration
        """
        # TODO: Implement configuration averaging
        # - Weight by job success/performance
        # - Consider different scaling factors
        return {
            "executor_cores": 4,
            "executor_memory_mb": 8192,
            "num_executors": 10,
            "driver_memory_mb": 4096,
        }

    def _fallback_recommendation(self, job_requirements: Dict) -> Dict:
        """Provide fallback recommendation when no similar jobs found.

        Args:
            job_requirements: Job requirements

        Returns:
            Fallback recommendation
        """
        # TODO: Implement rule-based fallback
        # - Based on input size
        # - Conservative resource allocation
        input_gb = job_requirements.get("input_size_gb", 10)

        return self._create_recommendation_response(
            executor_cores=4,
            executor_memory_mb=int(8192 * max(1, input_gb / 100)),
            num_executors=max(5, int(input_gb / 10)),
            driver_memory_mb=4096,
            confidence=0.5,
            metadata={"method": "fallback"},
        )
