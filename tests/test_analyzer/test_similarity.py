"""Tests for job similarity calculation."""

import pytest
import numpy as np

from spark_optimizer.analyzer.similarity import JobSimilarityCalculator


@pytest.fixture
def calculator():
    """Create a similarity calculator with default weights."""
    return JobSimilarityCalculator()


@pytest.fixture
def custom_weights():
    """Custom weights for testing."""
    return {
        "input_size": 0.5,
        "app_name": 0.2,
        "num_stages": 0.1,
        "executor_config": 0.1,
        "shuffle_size": 0.05,
        "output_size": 0.05,
    }


@pytest.fixture
def sample_job1():
    """Sample job data for testing."""
    return {
        "app_id": "app-001",
        "app_name": "ETL Data Pipeline",
        "input_bytes": 10 * 1024**3,  # 10 GB
        "output_bytes": 5 * 1024**3,  # 5 GB
        "shuffle_read_bytes": 2 * 1024**3,
        "shuffle_write_bytes": 2 * 1024**3,
        "total_stages": 10,
        "total_tasks": 1000,
        "num_executors": 10,
        "executor_cores": 4,
        "executor_memory_mb": 8192,
        "duration_ms": 600000,
    }


@pytest.fixture
def sample_job2():
    """Another sample job for comparison."""
    return {
        "app_id": "app-002",
        "app_name": "ETL Data Processing",
        "input_bytes": 12 * 1024**3,  # 12 GB - similar to job1
        "output_bytes": 6 * 1024**3,  # 6 GB
        "shuffle_read_bytes": 2.5 * 1024**3,
        "shuffle_write_bytes": 2.5 * 1024**3,
        "total_stages": 12,
        "total_tasks": 1200,
        "num_executors": 12,
        "executor_cores": 4,
        "executor_memory_mb": 8192,
        "duration_ms": 720000,
    }


@pytest.fixture
def dissimilar_job():
    """A job that is very different from sample_job1."""
    return {
        "app_id": "app-003",
        "app_name": "ML Model Training",
        "input_bytes": 100 * 1024**3,  # 100 GB - much larger
        "output_bytes": 1 * 1024**3,
        "shuffle_read_bytes": 50 * 1024**3,
        "shuffle_write_bytes": 50 * 1024**3,
        "total_stages": 50,
        "total_tasks": 10000,
        "num_executors": 50,
        "executor_cores": 8,
        "executor_memory_mb": 32768,
        "duration_ms": 7200000,
    }


class TestJobSimilarityCalculator:
    """Test cases for JobSimilarityCalculator."""

    def test_calculate_similarity_identical_jobs(self, calculator, sample_job1):
        """Test similarity between identical jobs is 1.0."""
        similarity = calculator.calculate_similarity(sample_job1, sample_job1)

        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_calculate_similarity_similar_jobs(
        self, calculator, sample_job1, sample_job2
    ):
        """Test similarity between similar jobs is high."""
        similarity = calculator.calculate_similarity(sample_job1, sample_job2)

        # Similar jobs should have high similarity (> 0.7)
        assert similarity > 0.7
        assert similarity < 1.0

    def test_calculate_similarity_dissimilar_jobs(
        self, calculator, sample_job1, dissimilar_job
    ):
        """Test similarity between dissimilar jobs is low."""
        similarity = calculator.calculate_similarity(sample_job1, dissimilar_job)

        # Dissimilar jobs should have lower similarity
        assert similarity < 0.7

    def test_calculate_similarity_symmetric(self, calculator, sample_job1, sample_job2):
        """Test that similarity calculation is symmetric."""
        sim1 = calculator.calculate_similarity(sample_job1, sample_job2)
        sim2 = calculator.calculate_similarity(sample_job2, sample_job1)

        assert sim1 == pytest.approx(sim2, rel=0.01)

    def test_calculate_similarity_with_custom_weights(
        self, custom_weights, sample_job1, sample_job2
    ):
        """Test similarity calculation with custom weights."""
        calculator = JobSimilarityCalculator(weights=custom_weights)

        similarity = calculator.calculate_similarity(sample_job1, sample_job2)

        # With higher input_size weight, small size differences matter more
        assert 0.0 <= similarity <= 1.0

    def test_size_similarity_identical(self, calculator):
        """Test size similarity for identical sizes."""
        sim = calculator._size_similarity(1024**3, 1024**3)
        assert sim == 1.0

    def test_size_similarity_different_sizes(self, calculator):
        """Test size similarity for different sizes."""
        # 1 GB vs 2 GB -> ratio should be 0.5
        sim = calculator._size_similarity(1024**3, 2 * 1024**3)
        assert sim == pytest.approx(0.5, rel=0.01)

    def test_size_similarity_zero_values(self, calculator):
        """Test size similarity with zero values."""
        # Both zero -> 1.0
        assert calculator._size_similarity(0, 0) == 1.0
        # One zero -> 0.0
        assert calculator._size_similarity(0, 1024**3) == 0.0
        assert calculator._size_similarity(1024**3, 0) == 0.0

    def test_size_similarity_order_invariant(self, calculator):
        """Test that size similarity is order invariant."""
        sim1 = calculator._size_similarity(1024**3, 2 * 1024**3)
        sim2 = calculator._size_similarity(2 * 1024**3, 1024**3)
        assert sim1 == sim2

    def test_text_similarity_exact_match(self, calculator):
        """Test text similarity for exact matches."""
        sim = calculator._text_similarity("ETL Job", "ETL Job")
        assert sim == 1.0

    def test_text_similarity_case_insensitive(self, calculator):
        """Test text similarity is case insensitive."""
        sim = calculator._text_similarity("etl job", "ETL JOB")
        assert sim == 1.0

    def test_text_similarity_containment(self, calculator):
        """Test text similarity when one string contains another."""
        sim = calculator._text_similarity("ETL", "ETL Data Pipeline")
        assert sim == 0.8  # Containment score

    def test_text_similarity_partial_match(self, calculator):
        """Test text similarity for partial matches."""
        sim = calculator._text_similarity("ETL Data Pipeline", "ETL Data Processing")
        # Should have some similarity due to shared n-grams
        assert 0.0 < sim < 1.0

    def test_text_similarity_no_match(self, calculator):
        """Test text similarity for completely different strings."""
        sim = calculator._text_similarity("ABC", "XYZ")
        # Should have low similarity
        assert sim < 0.5

    def test_text_similarity_empty_strings(self, calculator):
        """Test text similarity with empty strings."""
        assert calculator._text_similarity("", "test") == 0.0
        assert calculator._text_similarity("test", "") == 0.0
        assert calculator._text_similarity("", "") == 0.0

    def test_count_similarity_identical(self, calculator):
        """Test count similarity for identical counts."""
        sim = calculator._count_similarity(100, 100)
        assert sim == 1.0

    def test_count_similarity_different(self, calculator):
        """Test count similarity for different counts."""
        sim = calculator._count_similarity(50, 100)
        assert sim == pytest.approx(0.5, rel=0.01)

    def test_count_similarity_zero_values(self, calculator):
        """Test count similarity with zero values."""
        assert calculator._count_similarity(0, 0) == 1.0
        assert calculator._count_similarity(0, 100) == 0.0
        assert calculator._count_similarity(100, 0) == 0.0

    def test_config_similarity_identical(self, calculator, sample_job1):
        """Test config similarity for identical configurations."""
        sim = calculator._config_similarity(sample_job1, sample_job1)
        assert sim == pytest.approx(1.0, rel=0.01)

    def test_config_similarity_different(self, calculator, sample_job1, dissimilar_job):
        """Test config similarity for different configurations."""
        sim = calculator._config_similarity(sample_job1, dissimilar_job)
        # Very different configs should have low similarity
        assert sim < 0.5

    def test_find_similar_jobs_returns_sorted_list(
        self, calculator, sample_job1, sample_job2, dissimilar_job
    ):
        """Test find_similar_jobs returns jobs sorted by similarity."""
        candidates = [sample_job2, dissimilar_job]

        results = calculator.find_similar_jobs(sample_job1, candidates, top_k=2)

        assert len(results) == 2
        # Results should be sorted by similarity (descending)
        assert results[0][1] >= results[1][1]
        # First result should be sample_job2 (more similar)
        assert results[0][0]["app_id"] == "app-002"

    def test_find_similar_jobs_top_k(self, calculator, sample_job1):
        """Test find_similar_jobs respects top_k parameter."""
        # Create multiple candidate jobs
        candidates = []
        for i in range(10):
            job = {
                "app_id": f"app-{i:03d}",
                "app_name": f"Job {i}",
                "input_bytes": (i + 1) * 1024**3,
                "output_bytes": (i + 1) * 512 * 1024**2,
            }
            candidates.append(job)

        results = calculator.find_similar_jobs(sample_job1, candidates, top_k=3)

        assert len(results) == 3

    def test_find_similar_jobs_empty_candidates(self, calculator, sample_job1):
        """Test find_similar_jobs with empty candidates list."""
        results = calculator.find_similar_jobs(sample_job1, [], top_k=5)

        assert len(results) == 0

    def test_find_similar_jobs_returns_similarity_scores(
        self, calculator, sample_job1, sample_job2
    ):
        """Test find_similar_jobs returns proper similarity scores."""
        results = calculator.find_similar_jobs(sample_job1, [sample_job2], top_k=1)

        assert len(results) == 1
        job, score = results[0]
        assert job["app_id"] == "app-002"
        assert 0.0 <= score <= 1.0

    def test_calculate_feature_vector(self, calculator, sample_job1):
        """Test feature vector extraction."""
        vector = calculator.calculate_feature_vector(sample_job1)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 10  # Number of features
        # All values should be non-negative (after log1p)
        assert all(v >= 0 for v in vector)

    def test_calculate_feature_vector_handles_missing_fields(self, calculator):
        """Test feature vector handles missing fields gracefully."""
        minimal_job = {
            "app_id": "app-minimal",
            "input_bytes": 1024**3,
        }

        vector = calculator.calculate_feature_vector(minimal_job)

        assert isinstance(vector, np.ndarray)
        assert len(vector) == 10

    def test_calculate_cosine_similarity(self, calculator):
        """Test cosine similarity calculation."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([1.0, 2.0, 3.0])

        sim = calculator.calculate_cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(1.0, rel=0.01)

    def test_calculate_cosine_similarity_orthogonal(self, calculator):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        sim = calculator.calculate_cosine_similarity(vec1, vec2)

        assert sim == pytest.approx(0.0, rel=0.01)

    def test_calculate_cosine_similarity_zero_vector(self, calculator):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        sim = calculator.calculate_cosine_similarity(vec1, vec2)

        assert sim == 0.0

    def test_calculate_cosine_similarity_empty_vectors(self, calculator):
        """Test cosine similarity with empty vectors."""
        vec1 = np.array([])
        vec2 = np.array([])

        sim = calculator.calculate_cosine_similarity(vec1, vec2)

        assert sim == 0.0

    def test_calculate_euclidean_distance_identical(self, calculator):
        """Test Euclidean distance for identical vectors."""
        vec1 = np.array([1.0, 2.0, 3.0])

        dist = calculator.calculate_euclidean_distance(vec1, vec1)

        assert dist == pytest.approx(0.0, abs=0.001)

    def test_calculate_euclidean_distance_different(self, calculator):
        """Test Euclidean distance for different vectors."""
        vec1 = np.array([0.0, 0.0, 0.0])
        vec2 = np.array([3.0, 4.0, 0.0])

        dist = calculator.calculate_euclidean_distance(vec1, vec2)

        assert dist == pytest.approx(5.0, rel=0.01)  # 3-4-5 triangle

    def test_calculate_euclidean_distance_empty_vectors(self, calculator):
        """Test Euclidean distance with empty vectors."""
        vec1 = np.array([])
        vec2 = np.array([])

        dist = calculator.calculate_euclidean_distance(vec1, vec2)

        assert dist == float("inf")

    def test_find_similar_jobs_vectorized_cosine(
        self, calculator, sample_job1, sample_job2, dissimilar_job
    ):
        """Test vectorized similarity search with cosine method."""
        candidates = [sample_job2, dissimilar_job]

        results = calculator.find_similar_jobs_vectorized(
            sample_job1, candidates, top_k=2, method="cosine"
        )

        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1]

    def test_find_similar_jobs_vectorized_euclidean(
        self, calculator, sample_job1, sample_job2, dissimilar_job
    ):
        """Test vectorized similarity search with euclidean method."""
        candidates = [sample_job2, dissimilar_job]

        results = calculator.find_similar_jobs_vectorized(
            sample_job1, candidates, top_k=2, method="euclidean"
        )

        assert len(results) == 2
        # Results should be sorted by similarity
        assert results[0][1] >= results[1][1]

    def test_find_similar_jobs_vectorized_empty_candidates(
        self, calculator, sample_job1
    ):
        """Test vectorized search with empty candidates."""
        results = calculator.find_similar_jobs_vectorized(sample_job1, [], top_k=5)

        assert len(results) == 0

    def test_find_similar_jobs_vectorized_invalid_method(
        self, calculator, sample_job1, sample_job2
    ):
        """Test vectorized search with invalid method raises error."""
        with pytest.raises(ValueError):
            calculator.find_similar_jobs_vectorized(
                sample_job1, [sample_job2], method="invalid_method"
            )

    def test_get_feature_importance(self, calculator):
        """Test getting feature importance."""
        importance = calculator.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) > 0
        # All importance values should sum to 1.0
        assert sum(importance.values()) == pytest.approx(1.0, rel=0.01)

    def test_get_feature_importance_with_custom_weights(self, custom_weights):
        """Test feature importance with custom weights."""
        calculator = JobSimilarityCalculator(weights=custom_weights)
        importance = calculator.get_feature_importance()

        # Custom weights gave input_size 0.5 out of 1.0 total
        assert importance["input_size"] == pytest.approx(0.5, rel=0.01)

    def test_default_weights_sum_to_one(self, calculator):
        """Test that default weights sum to 1.0."""
        weights = calculator._default_weights()
        total = sum(weights.values())
        assert total == pytest.approx(1.0, rel=0.01)


class TestSimilarityEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_similarity_with_missing_fields(self, calculator):
        """Test similarity when jobs have missing fields."""
        job1 = {"app_id": "app-001", "input_bytes": 1024**3}
        job2 = {"app_id": "app-002", "output_bytes": 512 * 1024**2}

        # Should not raise an error
        similarity = calculator.calculate_similarity(job1, job2)

        assert 0.0 <= similarity <= 1.0

    def test_similarity_with_input_size_bytes_field(self, calculator):
        """Test similarity handles input_size_bytes field (from job_requirements)."""
        job1 = {"app_id": "app-001", "input_size_bytes": 10 * 1024**3}
        job2 = {"app_id": "app-002", "input_bytes": 10 * 1024**3}

        similarity = calculator.calculate_similarity(job1, job2)

        # Should match input_size_bytes with input_bytes
        assert similarity > 0.0

    def test_similarity_with_very_large_values(self, calculator):
        """Test similarity with very large byte values."""
        job1 = {"app_id": "app-001", "input_bytes": 1024**5}  # 1 PB
        job2 = {"app_id": "app-002", "input_bytes": 1024**5}

        similarity = calculator.calculate_similarity(job1, job2)

        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_feature_vector_with_zero_values(self, calculator):
        """Test feature vector extraction with all zero values."""
        job = {
            "app_id": "app-zero",
            "input_bytes": 0,
            "output_bytes": 0,
            "shuffle_read_bytes": 0,
            "shuffle_write_bytes": 0,
            "total_stages": 0,
            "total_tasks": 0,
            "num_executors": 0,
            "executor_cores": 0,
            "executor_memory_mb": 0,
            "duration_ms": 0,
        }

        vector = calculator.calculate_feature_vector(job)

        # log1p(0) = 0, so most values should be 0
        assert isinstance(vector, np.ndarray)
        assert all(v == 0 for v in vector)


class TestSimilarityWithRealWorldData:
    """Test with realistic job data patterns."""

    @pytest.fixture
    def etl_jobs(self):
        """Create a set of similar ETL jobs."""
        return [
            {
                "app_id": f"etl-{i}",
                "app_name": f"Daily ETL Pipeline {i}",
                "input_bytes": 10 * 1024**3 + i * 1024**3,
                "output_bytes": 5 * 1024**3,
                "shuffle_write_bytes": 2 * 1024**3,
                "total_stages": 10,
                "num_executors": 10,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
            }
            for i in range(5)
        ]

    @pytest.fixture
    def ml_jobs(self):
        """Create a set of similar ML jobs."""
        return [
            {
                "app_id": f"ml-{i}",
                "app_name": f"Model Training Job {i}",
                "input_bytes": 100 * 1024**3 + i * 10 * 1024**3,
                "output_bytes": 1 * 1024**3,
                "shuffle_write_bytes": 50 * 1024**3,
                "total_stages": 50,
                "num_executors": 50,
                "executor_cores": 8,
                "executor_memory_mb": 32768,
            }
            for i in range(5)
        ]

    def test_etl_jobs_similar_to_each_other(self, calculator, etl_jobs):
        """Test that ETL jobs are similar to each other."""
        target = etl_jobs[0]
        candidates = etl_jobs[1:]

        results = calculator.find_similar_jobs(target, candidates, top_k=4)

        # All ETL jobs should have high similarity
        for job, score in results:
            assert score > 0.7

    def test_ml_jobs_more_similar_to_ml_jobs(self, calculator, etl_jobs, ml_jobs):
        """Test that ML jobs are more similar to other ML jobs than ETL jobs."""
        target_ml = ml_jobs[0]
        all_candidates = etl_jobs + ml_jobs[1:]

        results = calculator.find_similar_jobs(target_ml, all_candidates, top_k=8)

        # Top results should be ML jobs
        top_results = results[:4]
        ml_count = sum(1 for job, _ in top_results if job["app_id"].startswith("ml-"))
        assert ml_count >= 3  # At least 3 of top 4 should be ML jobs
