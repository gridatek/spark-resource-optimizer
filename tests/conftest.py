"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_job_data():
    """Sample job data for testing.

    Returns:
        Dictionary containing sample job data
    """
    return {
        "app_id": "app-20240101-000001",
        "app_name": "sample_etl_job",
        "duration_ms": 120000,
        "executor_cores": 4,
        "executor_memory_mb": 8192,
        "num_executors": 10,
        "input_bytes": 10 * 1024**3,  # 10 GB
        "output_bytes": 5 * 1024**3,  # 5 GB
        "total_stages": 5,
        "total_tasks": 100,
    }


@pytest.fixture
def sample_job_requirements():
    """Sample job requirements for recommendations.

    Returns:
        Dictionary containing job requirements
    """
    return {
        "input_size_gb": 10.0,
        "job_type": "etl",
        "app_name": "test_job",
    }


@pytest.fixture
def sample_historical_jobs():
    """Sample historical jobs for testing.

    Returns:
        List of historical job data
    """
    return [
        {
            "app_id": f"app-2024-{i:06d}",
            "app_name": "historical_job",
            "duration_ms": 100000 + i * 1000,
            "executor_cores": 4,
            "executor_memory_mb": 8192,
            "num_executors": 10,
            "input_bytes": (10 + i) * 1024**3,
        }
        for i in range(10)
    ]
