"""Database models for storing Spark job data."""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base


class SparkApplication(Base):
    """Model for Spark application metadata."""

    __tablename__ = "spark_applications"

    id = Column(Integer, primary_key=True, index=True)
    app_id = Column(String, unique=True, index=True, nullable=False)
    app_name = Column(String, nullable=False)
    user = Column(String)
    submit_time = Column(DateTime)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_ms = Column(Integer)
    spark_version = Column(String)

    # Resource configuration
    executor_cores = Column(Integer)
    executor_memory_mb = Column(Integer)
    num_executors = Column(Integer)
    driver_memory_mb = Column(Integer)

    # Execution metrics
    total_tasks = Column(Integer)
    failed_tasks = Column(Integer)
    total_stages = Column(Integer)
    failed_stages = Column(Integer)

    # Storage metrics
    input_bytes = Column(Integer)
    output_bytes = Column(Integer)
    shuffle_read_bytes = Column(Integer)
    shuffle_write_bytes = Column(Integer)

    # Computed metrics
    cpu_time_ms = Column(Integer)
    memory_spilled_bytes = Column(Integer)
    disk_spilled_bytes = Column(Integer)

    # Additional metadata
    tags = Column(JSON)
    environment = Column(JSON)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    stages = relationship(
        "SparkStage", back_populates="application", cascade="all, delete-orphan"
    )


class SparkStage(Base):
    """Model for Spark stage metrics."""

    __tablename__ = "spark_stages"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(
        Integer, ForeignKey("spark_applications.id"), nullable=False
    )
    stage_id = Column(Integer, nullable=False)
    stage_name = Column(String)

    num_tasks = Column(Integer)
    submission_time = Column(DateTime)
    completion_time = Column(DateTime)
    duration_ms = Column(Integer)

    # Resource usage
    executor_run_time_ms = Column(Integer)
    executor_cpu_time_ms = Column(Integer)
    memory_bytes_spilled = Column(Integer)
    disk_bytes_spilled = Column(Integer)

    # I/O metrics
    input_bytes = Column(Integer)
    output_bytes = Column(Integer)
    shuffle_read_bytes = Column(Integer)
    shuffle_write_bytes = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    application = relationship("SparkApplication", back_populates="stages")


class JobRecommendation(Base):
    """Model for storing job recommendations."""

    __tablename__ = "job_recommendations"

    id = Column(Integer, primary_key=True, index=True)
    job_signature = Column(String, index=True)  # Hash of job characteristics

    # Recommended configuration
    recommended_executor_cores = Column(Integer)
    recommended_executor_memory_mb = Column(Integer)
    recommended_num_executors = Column(Integer)
    recommended_driver_memory_mb = Column(Integer)

    # Predicted metrics
    predicted_duration_ms = Column(Integer)
    predicted_cost = Column(Float)
    confidence_score = Column(Float)

    # Recommendation metadata
    recommendation_method = Column(String)  # similarity, ml, rule-based
    similar_jobs = Column(JSON)  # List of similar job IDs used for recommendation

    created_at = Column(DateTime, default=datetime.utcnow)
    used_at = Column(DateTime)  # When recommendation was actually used
    feedback_score = Column(Float)  # User feedback on recommendation quality
