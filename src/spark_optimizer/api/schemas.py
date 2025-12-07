"""Pydantic schemas for API request/response validation."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl


class RecommendationRequest(BaseModel):
    """Request schema for /api/v1/recommend endpoint."""

    input_size_bytes: int = Field(
        ..., gt=0, description="Expected input data size in bytes"
    )
    job_type: Optional[str] = Field(
        None, description="Type of job (etl, ml, streaming)"
    )
    sla_minutes: Optional[int] = Field(
        None, gt=0, description="Maximum acceptable duration in minutes"
    )
    budget_dollars: Optional[float] = Field(
        None, gt=0, description="Maximum acceptable cost in dollars"
    )
    priority: str = Field(
        "balanced", description="Optimization priority (performance, cost, or balanced)"
    )

    @validator("priority")
    def validate_priority(cls, v):
        """Validate priority value."""
        if v not in ["performance", "cost", "balanced"]:
            raise ValueError("priority must be one of: performance, cost, balanced")
        return v

    @validator("job_type")
    def validate_job_type(cls, v):
        """Validate job_type value."""
        if v is not None and v not in [
            "etl",
            "ml",
            "streaming",
            "batch",
            "interactive",
        ]:
            raise ValueError(
                "job_type must be one of: etl, ml, streaming, batch, interactive"
            )
        return v


class ResourceConfiguration(BaseModel):
    """Resource configuration schema."""

    executor_cores: int = Field(..., gt=0, description="Number of cores per executor")
    executor_memory_mb: int = Field(..., gt=0, description="Memory per executor in MB")
    num_executors: int = Field(..., gt=0, description="Number of executors")
    driver_memory_mb: int = Field(..., gt=0, description="Driver memory in MB")


class RecommendationMetadata(BaseModel):
    """Metadata for recommendation response."""

    method: str = Field(..., description="Recommendation method used")
    job_type: str = Field(..., description="Type of job")
    priority: str = Field(..., description="Optimization priority")
    rules_applied: Optional[int] = Field(None, description="Number of rules applied")
    optimization_hints: Optional[List[Dict[str, Any]]] = Field(
        None, description="Optimization hints"
    )


class RecommendationResponse(BaseModel):
    """Response schema for /api/v1/recommend endpoint."""

    configuration: ResourceConfiguration
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    metadata: RecommendationMetadata


class CollectRequest(BaseModel):
    """Request schema for /api/v1/collect endpoint."""

    history_server_url: str = Field(..., description="URL of the Spark History Server")
    max_apps: int = Field(
        100, gt=0, le=1000, description="Maximum number of applications to fetch"
    )
    status: str = Field("completed", description="Application status filter")
    min_date: Optional[str] = Field(
        None, description="Minimum date for applications (ISO format)"
    )

    @validator("history_server_url")
    def validate_url(cls, v):
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("history_server_url must start with http:// or https://")
        return v.rstrip("/")

    @validator("status")
    def validate_status(cls, v):
        """Validate status value."""
        if v not in ["completed", "running", "failed"]:
            raise ValueError("status must be one of: completed, running, failed")
        return v


class CollectResponse(BaseModel):
    """Response schema for /api/v1/collect endpoint."""

    success: bool
    collected: int = Field(..., ge=0)
    failed: int = Field(..., ge=0)
    skipped: int = Field(..., ge=0)
    message: str


class CompareRequest(BaseModel):
    """Request schema for /api/v1/compare endpoint."""

    app_ids: List[str] = Field(
        ...,
        min_length=2,
        max_length=10,
        description="List of application IDs to compare",
    )
    metrics: Optional[List[str]] = Field(
        None, description="Specific metrics to compare (all if not specified)"
    )

    @validator("app_ids")
    def validate_app_ids(cls, v):
        """Validate app_ids list."""
        if len(v) != len(set(v)):
            raise ValueError("app_ids must contain unique values")
        return v


class JobMetrics(BaseModel):
    """Job metrics schema for comparison."""

    app_id: str
    app_name: str
    duration_ms: int = Field(..., ge=0)
    executor_memory_mb: int = Field(..., ge=0)
    executor_cores: int = Field(..., ge=0)
    num_executors: int = Field(..., ge=0)
    driver_memory_mb: int = Field(..., ge=0)
    input_bytes: int = Field(..., ge=0)
    output_bytes: int = Field(..., ge=0)
    shuffle_read_bytes: int = Field(..., ge=0)
    shuffle_write_bytes: int = Field(..., ge=0)
    disk_spilled_bytes: int = Field(..., ge=0)
    memory_spilled_bytes: int = Field(..., ge=0)
    total_tasks: int = Field(..., ge=0)
    failed_tasks: int = Field(..., ge=0)
    jvm_gc_time_ms: int = Field(..., ge=0)
    throughput_gbps: float = Field(..., ge=0.0)
    data_per_executor_hour_gb: float = Field(..., ge=0.0)
    spill_ratio: float = Field(..., ge=0.0)


class PerformanceSummary(BaseModel):
    """Performance summary for job comparison."""

    app_id: str
    app_name: str
    duration_ms: Optional[int] = None
    throughput_gbps: Optional[float] = None


class ComparisonSummary(BaseModel):
    """Summary schema for job comparison."""

    fastest: Optional[PerformanceSummary] = None
    slowest: Optional[PerformanceSummary] = None
    most_efficient: Optional[PerformanceSummary] = None
    least_efficient: Optional[PerformanceSummary] = None


class ComparisonRecommendation(BaseModel):
    """Recommendation from job comparison."""

    type: str = Field(..., description="Type of recommendation")
    observation: str = Field(..., description="What was observed")
    recommendation: str = Field(..., description="Recommended action")


class CompareResponse(BaseModel):
    """Response schema for /api/v1/compare endpoint."""

    jobs: List[JobMetrics]
    summary: ComparisonSummary
    recommendations: List[ComparisonRecommendation]
    compared_count: int = Field(..., ge=2)


class OptimizationRecommendation(BaseModel):
    """Individual optimization recommendation."""

    rule_id: str
    title: str
    description: str
    severity: str = Field(..., description="Severity level (critical, warning, info)")
    current_value: Any
    recommended_value: Any
    expected_improvement: str
    spark_configs: Dict[str, str]

    @validator("severity")
    def validate_severity(cls, v):
        """Validate severity value."""
        if v not in ["critical", "warning", "info"]:
            raise ValueError("severity must be one of: critical, warning, info")
        return v


class AnalysisResult(BaseModel):
    """Analysis result statistics."""

    total_recommendations: int = Field(..., ge=0)
    critical: int = Field(..., ge=0)
    warnings: int = Field(..., ge=0)
    info: int = Field(..., ge=0)
    health_score: float = Field(..., ge=0.0, le=100.0)


class AnalyzeResponse(BaseModel):
    """Response schema for /api/v1/jobs/<app_id>/analyze endpoint."""

    app_id: str
    app_name: str
    analysis: AnalysisResult
    current_configuration: ResourceConfiguration
    recommended_configuration: ResourceConfiguration
    spark_configs: Dict[str, str]
    recommendations: List[OptimizationRecommendation]


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str
    message: Optional[str] = None
