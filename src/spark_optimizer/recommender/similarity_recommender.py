"""Similarity-based recommender using historical job matching."""

from typing import TYPE_CHECKING, Dict, List, Optional
from .base_recommender import BaseRecommender
from spark_optimizer.analyzer.similarity import JobSimilarityCalculator

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
        self.similarity_calculator = JobSimilarityCalculator(
            weights=config.get("similarity_weights") if config else None
        )
        self.top_k_similar = config.get("top_k_similar", 5) if config else 5

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

        # Load historical jobs from database if available and not loaded yet
        if not self.historical_jobs and self.db:
            self.load_historical_jobs_from_db()

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
        self.historical_jobs = historical_jobs

    def load_historical_jobs_from_db(self, limit: int = 1000):
        """Load historical jobs from database.

        Args:
            limit: Maximum number of jobs to load
        """
        if not self.db:
            return

        from spark_optimizer.storage.models import SparkApplication

        with self.db.get_session() as session:
            # Load successful jobs (no failures, completed successfully)
            apps = (
                session.query(SparkApplication)
                .filter(SparkApplication.failed_tasks == 0)
                .filter(SparkApplication.duration_ms.isnot(None))
                .filter(SparkApplication.end_time.isnot(None))
                .order_by(SparkApplication.end_time.desc())
                .limit(limit)
                .all()
            )

            # Convert to dictionaries
            self.historical_jobs = []
            for app in apps:
                job_dict = {
                    "app_id": app.app_id,
                    "app_name": app.app_name,
                    "input_bytes": app.input_bytes or 0,
                    "output_bytes": app.output_bytes or 0,
                    "shuffle_write_bytes": app.shuffle_write_bytes or 0,
                    "shuffle_read_bytes": app.shuffle_read_bytes or 0,
                    "total_stages": app.total_stages or 0,
                    "total_tasks": app.total_tasks or 0,
                    "failed_tasks": app.failed_tasks or 0,
                    "duration_ms": app.duration_ms,
                    "num_executors": app.num_executors or 0,
                    "executor_cores": app.executor_cores or 0,
                    "executor_memory_mb": app.executor_memory_mb or 0,
                    "driver_memory_mb": app.driver_memory_mb or 0,
                    "executor_run_time_ms": app.executor_run_time_ms or 0,
                    "disk_spilled_bytes": app.disk_spilled_bytes or 0,
                    "memory_spilled_bytes": app.memory_spilled_bytes or 0,
                }
                self.historical_jobs.append(job_dict)

    def _find_similar_jobs(
        self, job_requirements: Dict, top_k: int = None
    ) -> List[Dict]:
        """Find similar jobs from historical data.

        Args:
            job_requirements: Job requirements
            top_k: Number of similar jobs to find

        Returns:
            List of similar jobs
        """
        if top_k is None:
            top_k = self.top_k_similar

        # Use similarity calculator to find similar jobs
        similar_jobs_with_scores = self.similarity_calculator.find_similar_jobs(
            target_job=job_requirements,
            candidate_jobs=self.historical_jobs,
            top_k=top_k * 2,
        )

        # Filter by minimum similarity threshold
        filtered_jobs = [
            job
            for job, score in similar_jobs_with_scores
            if score >= self.min_similarity_threshold
        ]

        # Return top K after filtering
        return filtered_jobs[:top_k]

    def _average_configurations(self, jobs: List[Dict]) -> Dict:
        """Calculate average configuration from jobs.

        Args:
            jobs: List of job data

        Returns:
            Averaged configuration
        """
        if not jobs:
            return {
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "num_executors": 5,
                "driver_memory_mb": 4096,
            }

        # Calculate weights based on job performance
        # Jobs with better performance (less spilling, faster) get higher weight
        weights = []
        for job in jobs:
            weight = 1.0

            # Penalize jobs with spilling
            if (
                job.get("disk_spilled_bytes", 0) > 0
                or job.get("memory_spilled_bytes", 0) > 0
            ):
                weight *= 0.7

            # Bonus for faster jobs (relative to data processed)
            duration = job.get("duration_ms", 1)
            input_bytes = job.get("input_bytes", 1)
            throughput = input_bytes / duration if duration > 0 else 0
            if throughput > 0:
                weight *= min(2.0, 1.0 + throughput / 1000000)  # Normalize throughput

            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            weights = [1.0] * len(jobs)
            total_weight = len(jobs)

        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted averages
        avg_executor_cores = sum(
            job.get("executor_cores", 4) * w for job, w in zip(jobs, normalized_weights)
        )
        avg_executor_memory_mb = sum(
            job.get("executor_memory_mb", 8192) * w
            for job, w in zip(jobs, normalized_weights)
        )
        avg_num_executors = sum(
            job.get("num_executors", 5) * w for job, w in zip(jobs, normalized_weights)
        )
        avg_driver_memory_mb = sum(
            job.get("driver_memory_mb", 4096) * w
            for job, w in zip(jobs, normalized_weights)
        )

        # Round to reasonable values
        return {
            "executor_cores": max(1, int(round(avg_executor_cores))),
            "executor_memory_mb": max(
                1024, int(round(avg_executor_memory_mb / 1024) * 1024)
            ),
            "num_executors": max(1, int(round(avg_num_executors))),
            "driver_memory_mb": max(
                1024, int(round(avg_driver_memory_mb / 1024) * 1024)
            ),
        }

    def _fallback_recommendation(self, job_requirements: Dict) -> Dict:
        """Provide fallback recommendation when no similar jobs found.

        Uses rule-based heuristics based on input size and job type.

        Args:
            job_requirements: Job requirements

        Returns:
            Fallback recommendation
        """
        input_bytes = job_requirements.get("input_size_bytes", 10 * 1024**3)
        input_gb = input_bytes / (1024**3)
        job_type = job_requirements.get("job_type")
        priority = job_requirements.get("priority", "balanced")

        # Base configuration (for ~10GB input)
        base_executor_cores = 4
        base_executor_memory_mb = 8192
        base_num_executors = 5
        base_driver_memory_mb = 4096

        # Scale based on input size
        # Rule of thumb: 1 executor per 2-5GB of input data
        scale_factor = max(0.5, min(5.0, input_gb / 10))

        num_executors = max(2, int(base_num_executors * scale_factor))
        executor_memory_mb = int(
            base_executor_memory_mb * min(2.0, 1 + scale_factor / 10)
        )

        # Adjust based on job type
        if job_type == "ml":
            # ML jobs typically need more memory
            executor_memory_mb = int(executor_memory_mb * 1.5)
            base_driver_memory_mb = int(base_driver_memory_mb * 1.5)
        elif job_type == "streaming":
            # Streaming jobs need more executors, less memory
            num_executors = int(num_executors * 1.3)
            executor_memory_mb = int(executor_memory_mb * 0.8)

        # Adjust based on priority
        if priority == "performance":
            # More resources for performance
            num_executors = int(num_executors * 1.2)
            executor_memory_mb = int(executor_memory_mb * 1.2)
            base_executor_cores = 6
        elif priority == "cost":
            # Fewer resources for cost optimization
            num_executors = max(2, int(num_executors * 0.8))
            executor_memory_mb = int(executor_memory_mb * 0.8)

        # Round memory to nearest GB
        executor_memory_mb = max(2048, int(round(executor_memory_mb / 1024) * 1024))
        driver_memory_mb = max(2048, int(round(base_driver_memory_mb / 1024) * 1024))

        return self._create_recommendation_response(
            executor_cores=base_executor_cores,
            executor_memory_mb=executor_memory_mb,
            num_executors=num_executors,
            driver_memory_mb=driver_memory_mb,
            confidence=0.5,
            metadata={
                "method": "fallback",
                "input_gb": round(input_gb, 2),
                "job_type": job_type or "unknown",
                "priority": priority,
            },
        )
