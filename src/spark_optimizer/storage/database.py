"""Database connection and session management."""

from contextlib import contextmanager
from typing import Any, Dict, Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()  # type: ignore[misc]


class Database:
    """Database connection manager."""

    def __init__(self, connection_string: str):
        """Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
        """
        self.engine = create_engine(connection_string)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session.

        Yields:
            SQLAlchemy session
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_job(self, job_dict: Dict) -> None:
        """Save a Spark job to the database.

        Args:
            job_dict: Dictionary containing job metrics
        """
        from .models import SparkApplication

        # Extract nested configuration and metrics if present
        config = job_dict.get("configuration", {})
        metrics = job_dict.get("metrics", {})

        # Convert memory strings to MB if needed
        executor_memory_mb = self._parse_memory_to_mb(
            config.get("executor_memory_mb") or job_dict.get("executor_memory", "0")
        )
        driver_memory_mb = self._parse_memory_to_mb(
            config.get("driver_memory_mb") or job_dict.get("driver_memory", "0")
        )

        with self.get_session() as session:
            app = SparkApplication(
                app_id=job_dict.get("app_id"),
                app_name=job_dict.get("app_name"),
                user=job_dict.get("user"),
                submit_time=job_dict.get("submit_time"),
                start_time=job_dict.get("start_time"),
                end_time=job_dict.get("end_time"),
                duration_ms=job_dict.get("duration_ms"),
                spark_version=job_dict.get("spark_version"),
                executor_cores=config.get("executor_cores")
                or job_dict.get("executor_cores"),
                executor_memory_mb=executor_memory_mb,
                num_executors=config.get("num_executors")
                or job_dict.get("num_executors"),
                driver_memory_mb=driver_memory_mb,
                total_tasks=metrics.get("total_tasks") or job_dict.get("total_tasks"),
                failed_tasks=metrics.get("failed_tasks")
                or job_dict.get("failed_tasks"),
                total_stages=metrics.get("total_stages")
                or job_dict.get("total_stages"),
                failed_stages=metrics.get("failed_stages")
                or job_dict.get("failed_stages"),
                input_bytes=metrics.get("input_bytes") or job_dict.get("input_bytes"),
                output_bytes=metrics.get("output_bytes")
                or job_dict.get("output_bytes"),
                shuffle_read_bytes=metrics.get("shuffle_read_bytes")
                or job_dict.get("shuffle_read_bytes"),
                shuffle_write_bytes=metrics.get("shuffle_write_bytes")
                or job_dict.get("shuffle_write_bytes"),
                cpu_time_ms=metrics.get("cpu_time_ms")
                or job_dict.get("executor_cpu_time"),
                memory_spilled_bytes=metrics.get("memory_spilled_bytes")
                or job_dict.get("spill_memory_bytes"),
                disk_spilled_bytes=metrics.get("disk_spilled_bytes")
                or job_dict.get("spill_disk_bytes"),
                executor_run_time_ms=job_dict.get("executor_run_time"),
                executor_cpu_time_ms=job_dict.get("executor_cpu_time"),
                jvm_gc_time_ms=job_dict.get("jvm_gc_time"),
                peak_memory_usage=job_dict.get("peak_memory_usage"),
                cluster_type=job_dict.get("cluster_type"),
                instance_type=job_dict.get("instance_type"),
                estimated_cost=job_dict.get("estimated_cost"),
                tags=job_dict.get("tags"),
                environment=job_dict.get("environment"),
            )
            session.add(app)

    def save_recommendation(self, rec_dict: Dict) -> None:
        """Save a job recommendation to the database.

        Args:
            rec_dict: Dictionary containing recommendation data
        """
        from .models import JobRecommendation

        with self.get_session() as session:
            rec = JobRecommendation(
                job_signature=self._generate_job_signature(rec_dict),
                recommended_executor_cores=rec_dict.get("recommended_executor_cores"),
                recommended_executor_memory_mb=rec_dict.get(
                    "recommended_executor_memory_mb"
                ),
                recommended_num_executors=rec_dict.get("recommended_executors"),
                recommended_driver_memory_mb=rec_dict.get(
                    "recommended_driver_memory_mb"
                ),
                predicted_duration_ms=rec_dict.get("predicted_duration_ms"),
                predicted_cost=rec_dict.get("predicted_cost"),
                confidence_score=rec_dict.get("confidence_score"),
                recommendation_method=rec_dict.get("recommendation_method"),
                similar_jobs=rec_dict.get("similar_job_ids"),
            )
            session.add(rec)

    @staticmethod
    def _parse_memory_to_mb(memory_str: str) -> int:
        """Parse memory string (e.g., '4g', '512m') to MB.

        Args:
            memory_str: Memory string with unit

        Returns:
            Memory in MB
        """
        if isinstance(memory_str, int):
            return memory_str

        memory_str = str(memory_str).lower().strip()

        if memory_str.endswith("g"):
            return int(float(memory_str[:-1]) * 1024)
        elif memory_str.endswith("m"):
            return int(float(memory_str[:-1]))
        elif memory_str.endswith("k"):
            return int(float(memory_str[:-1]) / 1024)
        else:
            # Assume MB if no unit
            return int(float(memory_str))

    @staticmethod
    def _generate_job_signature(rec_dict: Dict) -> str:
        """Generate a unique signature for a job based on its characteristics.

        Args:
            rec_dict: Recommendation dictionary

        Returns:
            Job signature string
        """
        import hashlib
        import json

        signature_data = {
            "input_size": rec_dict.get("input_size_bytes", 0),
            "job_type": rec_dict.get("job_type", "unknown"),
        }

        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(
            signature_str.encode(), usedforsecurity=False
        ).hexdigest()  # nosec B324
