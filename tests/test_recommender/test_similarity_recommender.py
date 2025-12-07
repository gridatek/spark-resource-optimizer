"""Tests for similarity-based recommender."""

import pytest
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender
from spark_optimizer.analyzer.similarity import JobSimilarityCalculator


class TestJobSimilarityCalculator:
    """Test the JobSimilarityCalculator class."""

    def test_calculate_similarity_identical_jobs(self):
        """Test similarity calculation for identical jobs."""
        calc = JobSimilarityCalculator()

        job = {
            "app_name": "test_job",
            "input_bytes": 10 * 1024**3,
            "total_stages": 5,
            "num_executors": 10,
            "executor_cores": 4,
            "executor_memory_mb": 8192,
        }

        score = calc.calculate_similarity(job, job)
        assert score == pytest.approx(
            1.0, abs=0.01
        ), "Identical jobs should have similarity of 1.0"

    def test_calculate_similarity_similar_jobs(self):
        """Test similarity calculation for similar jobs."""
        calc = JobSimilarityCalculator()

        job1 = {
            "app_name": "my_etl_job",
            "input_bytes": 10 * 1024**3,
            "total_stages": 5,
            "num_executors": 10,
            "executor_cores": 4,
            "executor_memory_mb": 8192,
        }

        job2 = {
            "app_name": "my_etl_job_v2",
            "input_bytes": 12 * 1024**3,
            "total_stages": 6,
            "num_executors": 12,
            "executor_cores": 4,
            "executor_memory_mb": 8192,
        }

        score = calc.calculate_similarity(job1, job2)
        assert (
            0.7 <= score <= 1.0
        ), f"Similar jobs should have high similarity score, got {score}"

    def test_calculate_similarity_different_jobs(self):
        """Test similarity calculation for very different jobs."""
        calc = JobSimilarityCalculator()

        job1 = {
            "app_name": "etl_job",
            "input_bytes": 10 * 1024**3,
            "total_stages": 5,
            "num_executors": 10,
            "executor_cores": 4,
            "executor_memory_mb": 8192,
        }

        job2 = {
            "app_name": "ml_training",
            "input_bytes": 1000 * 1024**3,
            "total_stages": 50,
            "num_executors": 100,
            "executor_cores": 8,
            "executor_memory_mb": 16384,
        }

        score = calc.calculate_similarity(job1, job2)
        assert (
            0.0 <= score <= 0.5
        ), f"Different jobs should have low similarity score, got {score}"

    def test_size_similarity(self):
        """Test size similarity calculation."""
        calc = JobSimilarityCalculator()

        # Same size
        assert calc._size_similarity(1000, 1000) == 1.0

        # Similar sizes
        assert calc._size_similarity(1000, 1200) == pytest.approx(0.833, abs=0.01)

        # Very different sizes
        assert calc._size_similarity(1000, 10000) == 0.1

        # Zero handling
        assert calc._size_similarity(0, 0) == 1.0
        assert calc._size_similarity(0, 1000) == 0.0

    def test_text_similarity(self):
        """Test text similarity calculation."""
        calc = JobSimilarityCalculator()

        # Exact match
        assert calc._text_similarity("my_job", "my_job") == 1.0

        # Similar names
        score = calc._text_similarity("my_etl_job", "my_etl_job_v2")
        assert (
            score > 0.7
        ), f"Similar job names should have high text similarity, got {score}"

        # Different names
        score = calc._text_similarity("etl_job", "ml_training")
        assert (
            score < 0.5
        ), f"Different job names should have low text similarity, got {score}"

    def test_find_similar_jobs(self):
        """Test finding similar jobs from candidates."""
        calc = JobSimilarityCalculator()

        target = {
            "app_name": "my_job",
            "input_bytes": 10 * 1024**3,
            "total_stages": 5,
        }

        candidates = [
            {"app_name": "my_job_v1", "input_bytes": 11 * 1024**3, "total_stages": 5},
            {"app_name": "my_job_v2", "input_bytes": 12 * 1024**3, "total_stages": 6},
            {"app_name": "other_job", "input_bytes": 100 * 1024**3, "total_stages": 50},
        ]

        similar = calc.find_similar_jobs(target, candidates, top_k=2)
        assert len(similar) == 2
        assert all(isinstance(item, tuple) and len(item) == 2 for item in similar)
        # Most similar should be first
        assert similar[0][1] >= similar[1][1]


class TestSimilarityRecommender:
    """Test the SimilarityRecommender class."""

    def test_recommend_with_no_historical_data(self):
        """Test recommendation when no historical data is available."""
        recommender = SimilarityRecommender()

        rec = recommender.recommend(
            input_size_bytes=10 * 1024**3,
            job_type="etl",
            priority="balanced",
        )

        assert "configuration" in rec
        assert "confidence" in rec
        assert "metadata" in rec
        assert rec["metadata"]["method"] == "fallback"
        assert rec["confidence"] == 0.5

        config = rec["configuration"]
        assert config["num_executors"] > 0
        assert config["executor_cores"] > 0
        assert config["executor_memory_mb"] > 0

    def test_recommend_with_historical_data(self):
        """Test recommendation with historical data."""
        recommender = SimilarityRecommender()

        # Train with historical data
        historical_jobs = [
            {
                "app_name": "etl_job",
                "input_bytes": 10 * 1024**3,
                "output_bytes": 5 * 1024**3,
                "shuffle_write_bytes": 2 * 1024**3,
                "total_stages": 5,
                "num_executors": 10,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "driver_memory_mb": 4096,
                "duration_ms": 300000,
                "disk_spilled_bytes": 0,
                "memory_spilled_bytes": 0,
            },
            {
                "app_name": "etl_job_v2",
                "input_bytes": 12 * 1024**3,
                "output_bytes": 6 * 1024**3,
                "shuffle_write_bytes": 2.5 * 1024**3,
                "total_stages": 6,
                "num_executors": 12,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "driver_memory_mb": 4096,
                "duration_ms": 350000,
                "disk_spilled_bytes": 0,
                "memory_spilled_bytes": 0,
            },
        ]

        recommender.train(historical_jobs)

        # Lower similarity threshold for this test
        recommender.min_similarity_threshold = 0.3

        rec = recommender.recommend(
            input_size_bytes=11 * 1024**3,
            job_type="etl",
            priority="balanced",
        )

        assert "configuration" in rec
        # Should find similar jobs since we lowered threshold
        if rec["metadata"]["method"] == "similarity":
            assert rec["confidence"] == 0.8
            assert rec["metadata"]["similar_jobs_count"] > 0
        else:
            # Fallback is also acceptable if similarity threshold not met
            assert rec["metadata"]["method"] == "fallback"

    def test_fallback_recommendation_ml_job(self):
        """Test fallback recommendation for ML job."""
        recommender = SimilarityRecommender()

        rec = recommender.recommend(
            input_size_bytes=50 * 1024**3,
            job_type="ml",
            priority="balanced",
        )

        config = rec["configuration"]
        # ML jobs should get more memory
        assert config["executor_memory_mb"] >= 8192
        assert config["num_executors"] >= 5

    def test_fallback_recommendation_performance_priority(self):
        """Test fallback recommendation with performance priority."""
        recommender = SimilarityRecommender()

        rec_balanced = recommender.recommend(
            input_size_bytes=10 * 1024**3,
            priority="balanced",
        )

        rec_performance = recommender.recommend(
            input_size_bytes=10 * 1024**3,
            priority="performance",
        )

        # Performance priority should allocate more resources
        assert (
            rec_performance["configuration"]["num_executors"]
            >= rec_balanced["configuration"]["num_executors"]
        )
        assert (
            rec_performance["configuration"]["executor_memory_mb"]
            >= rec_balanced["configuration"]["executor_memory_mb"]
        )

    def test_fallback_recommendation_cost_priority(self):
        """Test fallback recommendation with cost priority."""
        recommender = SimilarityRecommender()

        rec_balanced = recommender.recommend(
            input_size_bytes=10 * 1024**3,
            priority="balanced",
        )

        rec_cost = recommender.recommend(
            input_size_bytes=10 * 1024**3,
            priority="cost",
        )

        # Cost priority should allocate fewer resources
        assert (
            rec_cost["configuration"]["num_executors"]
            <= rec_balanced["configuration"]["num_executors"]
        )
        assert (
            rec_cost["configuration"]["executor_memory_mb"]
            <= rec_balanced["configuration"]["executor_memory_mb"]
        )

    def test_average_configurations(self):
        """Test configuration averaging."""
        recommender = SimilarityRecommender()

        jobs = [
            {
                "num_executors": 10,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "driver_memory_mb": 4096,
                "disk_spilled_bytes": 0,
                "memory_spilled_bytes": 0,
                "duration_ms": 300000,
                "input_bytes": 10 * 1024**3,
            },
            {
                "num_executors": 12,
                "executor_cores": 4,
                "executor_memory_mb": 10240,
                "driver_memory_mb": 4096,
                "disk_spilled_bytes": 0,
                "memory_spilled_bytes": 0,
                "duration_ms": 350000,
                "input_bytes": 12 * 1024**3,
            },
        ]

        avg_config = recommender._average_configurations(jobs)

        assert avg_config["executor_cores"] == 4
        assert 10 <= avg_config["num_executors"] <= 12
        assert 8192 <= avg_config["executor_memory_mb"] <= 10240

    def test_average_configurations_empty_list(self):
        """Test configuration averaging with empty list."""
        recommender = SimilarityRecommender()

        avg_config = recommender._average_configurations([])

        # Should return default values
        assert avg_config["executor_cores"] > 0
        assert avg_config["num_executors"] > 0
        assert avg_config["executor_memory_mb"] > 0
        assert avg_config["driver_memory_mb"] > 0
