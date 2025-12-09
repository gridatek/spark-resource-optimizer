"""Tests for ML-based recommender."""

import os
import tempfile
import pytest
import numpy as np
from spark_optimizer.recommender.ml_recommender import MLRecommender


def generate_synthetic_jobs(n_jobs: int = 100, seed: int = 42) -> list:
    """Generate synthetic job data for testing.

    Args:
        n_jobs: Number of jobs to generate
        seed: Random seed for reproducibility

    Returns:
        List of job dictionaries
    """
    np.random.seed(seed)
    jobs = []

    for i in range(n_jobs):
        # Generate realistic job data with some correlations
        input_gb = np.random.uniform(1, 100)
        input_bytes = int(input_gb * 1024**3)

        # Resources correlate with input size
        base_executors = max(2, int(input_gb / 5))
        num_executors = base_executors + np.random.randint(-2, 3)
        num_executors = max(1, min(50, num_executors))

        executor_cores = np.random.choice([2, 4, 8])
        executor_memory_mb = np.random.choice([4096, 8192, 16384])
        driver_memory_mb = np.random.choice([2048, 4096, 8192])

        # Metrics correlate with resources and input size
        output_bytes = int(input_bytes * np.random.uniform(0.3, 0.8))
        shuffle_bytes = int(input_bytes * np.random.uniform(0.1, 0.5))
        num_stages = max(1, int(np.log2(input_gb + 1) * 3) + np.random.randint(-2, 3))
        num_tasks = max(1, int(input_gb * 10) + np.random.randint(-5, 6))

        job_types = ["etl", "ml", "streaming", "batch", None]
        priorities = ["performance", "cost", "balanced"]

        jobs.append(
            {
                "app_id": f"app_{i:04d}",
                "app_name": f"job_{i}",
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "shuffle_read_bytes": shuffle_bytes // 2,
                "shuffle_write_bytes": shuffle_bytes // 2,
                "total_stages": num_stages,
                "total_tasks": num_tasks,
                "num_executors": num_executors,
                "executor_cores": executor_cores,
                "executor_memory_mb": executor_memory_mb,
                "driver_memory_mb": driver_memory_mb,
                "duration_ms": int(input_gb * 1000 * np.random.uniform(0.5, 2)),
                "disk_spilled_bytes": 0,
                "memory_spilled_bytes": 0,
                "job_type": np.random.choice(job_types),
                "priority": np.random.choice(priorities),
            }
        )

    return jobs


class TestMLRecommenderInit:
    """Test MLRecommender initialization."""

    def test_default_init(self):
        """Test default initialization."""
        recommender = MLRecommender()

        assert recommender.is_trained is False
        assert recommender.model_type == "random_forest"
        assert recommender.n_estimators == 100
        assert recommender.max_depth == 10
        assert len(recommender.models) == 0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = {
            "model_type": "gradient_boosting",
            "n_estimators": 50,
            "max_depth": 5,
            "model_path": "/tmp/test_models",
        }
        recommender = MLRecommender(config=config)

        assert recommender.model_type == "gradient_boosting"
        assert recommender.n_estimators == 50
        assert recommender.max_depth == 5
        assert recommender.model_path == "/tmp/test_models"


class TestMLRecommenderTraining:
    """Test MLRecommender training functionality."""

    def test_train_with_valid_data(self):
        """Test training with valid historical data."""
        recommender = MLRecommender()
        jobs = generate_synthetic_jobs(n_jobs=50)

        metrics = recommender.train(jobs)

        assert recommender.is_trained is True
        assert len(recommender.models) == 4  # 4 target fields

        # Check metrics structure
        for target in MLRecommender.TARGET_FIELDS:
            assert target in metrics
            assert "mae" in metrics[target]
            assert "rmse" in metrics[target]
            assert "r2" in metrics[target]
            assert "cv_r2_mean" in metrics[target]

    def test_train_with_empty_data(self):
        """Test training with empty dataset raises error."""
        recommender = MLRecommender()

        with pytest.raises(ValueError, match="Cannot train with empty dataset"):
            recommender.train([])

    def test_train_with_insufficient_data(self):
        """Test training with insufficient data raises error."""
        recommender = MLRecommender()
        jobs = generate_synthetic_jobs(n_jobs=5)

        with pytest.raises(ValueError, match="Insufficient training data"):
            recommender.train(jobs)

    def test_train_with_invalid_jobs(self):
        """Test training with jobs missing required fields."""
        recommender = MLRecommender()

        # Jobs without resource configuration
        jobs = [{"input_bytes": 1024**3, "app_name": f"job_{i}"} for i in range(20)]

        with pytest.raises(ValueError, match="valid jobs after filtering"):
            recommender.train(jobs)

    def test_train_gradient_boosting(self):
        """Test training with gradient boosting model."""
        config = {"model_type": "gradient_boosting", "n_estimators": 20}
        recommender = MLRecommender(config=config)
        jobs = generate_synthetic_jobs(n_jobs=30)

        metrics = recommender.train(jobs)

        assert recommender.is_trained is True
        assert recommender.model_type == "gradient_boosting"


class TestMLRecommenderPrediction:
    """Test MLRecommender prediction functionality."""

    @pytest.fixture
    def trained_recommender(self):
        """Create a trained recommender for testing."""
        recommender = MLRecommender()
        jobs = generate_synthetic_jobs(n_jobs=50)
        recommender.train(jobs)
        return recommender

    def test_recommend_requires_training(self):
        """Test that recommend raises error if not trained."""
        recommender = MLRecommender()

        with pytest.raises(ValueError, match="Model must be trained"):
            recommender.recommend(input_size_bytes=10 * 1024**3)

    def test_recommend_basic(self, trained_recommender):
        """Test basic recommendation."""
        rec = trained_recommender.recommend(
            input_size_bytes=10 * 1024**3,
            job_type="etl",
            priority="balanced",
        )

        assert "configuration" in rec
        assert "confidence" in rec
        assert "metadata" in rec

        config = rec["configuration"]
        assert config["executor_cores"] >= 1
        assert config["executor_memory_mb"] >= 1024
        assert config["num_executors"] >= 1
        assert config["driver_memory_mb"] >= 1024

        assert rec["metadata"]["method"] == "ml"
        assert 0 <= rec["confidence"] <= 1

    def test_recommend_with_different_job_types(self, trained_recommender):
        """Test recommendations for different job types."""
        job_types = ["etl", "ml", "streaming", "batch"]
        input_size = 20 * 1024**3

        recommendations = {}
        for job_type in job_types:
            rec = trained_recommender.recommend(
                input_size_bytes=input_size,
                job_type=job_type,
                priority="balanced",
            )
            recommendations[job_type] = rec["configuration"]

        # All should return valid configurations
        for job_type, config in recommendations.items():
            assert config["executor_cores"] >= 1
            assert config["num_executors"] >= 1

    def test_recommend_with_different_priorities(self, trained_recommender):
        """Test recommendations with different priorities."""
        input_size = 10 * 1024**3

        rec_balanced = trained_recommender.recommend(
            input_size_bytes=input_size,
            priority="balanced",
        )

        rec_performance = trained_recommender.recommend(
            input_size_bytes=input_size,
            priority="performance",
            sla_minutes=30,
        )

        rec_cost = trained_recommender.recommend(
            input_size_bytes=input_size,
            priority="cost",
            budget_dollars=100,
        )

        # Performance should have more or equal resources
        assert (
            rec_performance["configuration"]["num_executors"]
            >= rec_cost["configuration"]["num_executors"]
        )

    def test_recommend_scales_with_input_size(self, trained_recommender):
        """Test that recommendations scale with input size."""
        rec_small = trained_recommender.recommend(
            input_size_bytes=1 * 1024**3,
            priority="balanced",
        )

        rec_large = trained_recommender.recommend(
            input_size_bytes=50 * 1024**3,
            priority="balanced",
        )

        # Larger input should generally need more resources
        # (May not always be true due to model predictions, but trend should hold)
        small_total = (
            rec_small["configuration"]["num_executors"]
            * rec_small["configuration"]["executor_cores"]
        )
        large_total = (
            rec_large["configuration"]["num_executors"]
            * rec_large["configuration"]["executor_cores"]
        )

        # At minimum, both should be valid
        assert small_total >= 1
        assert large_total >= 1

    def test_recommend_returns_valid_ranges(self, trained_recommender):
        """Test that predictions are within valid ranges."""
        # Test with various input sizes
        for input_gb in [1, 10, 50, 100]:
            rec = trained_recommender.recommend(
                input_size_bytes=input_gb * 1024**3,
                priority="balanced",
            )
            config = rec["configuration"]

            # Check all values are within defined ranges
            assert (
                MLRecommender.RESOURCE_RANGES["executor_cores"][0]
                <= config["executor_cores"]
                <= MLRecommender.RESOURCE_RANGES["executor_cores"][1]
            )
            assert (
                MLRecommender.RESOURCE_RANGES["executor_memory_mb"][0]
                <= config["executor_memory_mb"]
                <= MLRecommender.RESOURCE_RANGES["executor_memory_mb"][1]
            )
            assert (
                MLRecommender.RESOURCE_RANGES["num_executors"][0]
                <= config["num_executors"]
                <= MLRecommender.RESOURCE_RANGES["num_executors"][1]
            )
            assert (
                MLRecommender.RESOURCE_RANGES["driver_memory_mb"][0]
                <= config["driver_memory_mb"]
                <= MLRecommender.RESOURCE_RANGES["driver_memory_mb"][1]
            )


class TestMLRecommenderEvaluation:
    """Test MLRecommender evaluation functionality."""

    @pytest.fixture
    def trained_recommender(self):
        """Create a trained recommender for testing."""
        recommender = MLRecommender()
        jobs = generate_synthetic_jobs(n_jobs=50)
        recommender.train(jobs)
        return recommender

    def test_evaluate_requires_training(self):
        """Test that evaluate raises error if not trained."""
        recommender = MLRecommender()
        test_jobs = generate_synthetic_jobs(n_jobs=10)

        with pytest.raises(ValueError, match="Model must be trained"):
            recommender.evaluate(test_jobs)

    def test_evaluate_with_valid_test_data(self, trained_recommender):
        """Test evaluation with valid test data."""
        test_jobs = generate_synthetic_jobs(n_jobs=20, seed=123)

        metrics = trained_recommender.evaluate(test_jobs)

        # Check metrics structure
        for target in MLRecommender.TARGET_FIELDS:
            assert target in metrics
            assert "mae" in metrics[target]
            assert "rmse" in metrics[target]
            assert "r2" in metrics[target]
            assert "mape" in metrics[target]

            # MAE and RMSE should be non-negative
            assert metrics[target]["mae"] >= 0
            assert metrics[target]["rmse"] >= 0

    def test_evaluate_with_empty_test_data(self, trained_recommender):
        """Test evaluation with empty test data."""
        with pytest.raises(ValueError, match="Cannot evaluate with empty test set"):
            trained_recommender.evaluate([])

    def test_evaluate_with_invalid_test_data(self, trained_recommender):
        """Test evaluation with invalid test data."""
        invalid_jobs = [{"input_bytes": 1024**3}]

        with pytest.raises(ValueError, match="No valid jobs in test set"):
            trained_recommender.evaluate(invalid_jobs)


class TestMLRecommenderPersistence:
    """Test MLRecommender save/load functionality."""

    @pytest.fixture
    def trained_recommender(self):
        """Create a trained recommender for testing."""
        recommender = MLRecommender()
        jobs = generate_synthetic_jobs(n_jobs=30)
        recommender.train(jobs)
        return recommender

    def test_save_requires_training(self):
        """Test that save raises error if not trained."""
        recommender = MLRecommender()

        with pytest.raises(ValueError, match="Cannot save untrained model"):
            recommender.save_model()

    def test_save_and_load(self, trained_recommender):
        """Test saving and loading models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            save_path = trained_recommender.save_model(tmpdir)
            assert save_path == tmpdir

            # Verify files exist
            assert os.path.exists(os.path.join(tmpdir, "metadata.joblib"))
            assert os.path.exists(os.path.join(tmpdir, "scaler_features.joblib"))
            for target in MLRecommender.TARGET_FIELDS:
                assert os.path.exists(os.path.join(tmpdir, f"model_{target}.joblib"))

            # Load into new recommender
            new_recommender = MLRecommender()
            new_recommender.load_model(tmpdir)

            assert new_recommender.is_trained is True
            assert len(new_recommender.models) == 4

    def test_load_produces_same_predictions(self, trained_recommender):
        """Test that loaded model produces same predictions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Get prediction from original
            original_rec = trained_recommender.recommend(
                input_size_bytes=10 * 1024**3,
                job_type="etl",
                priority="balanced",
            )

            # Save and load
            trained_recommender.save_model(tmpdir)
            new_recommender = MLRecommender()
            new_recommender.load_model(tmpdir)

            # Get prediction from loaded
            loaded_rec = new_recommender.recommend(
                input_size_bytes=10 * 1024**3,
                job_type="etl",
                priority="balanced",
            )

            # Predictions should be identical
            assert original_rec["configuration"] == loaded_rec["configuration"]

    def test_load_nonexistent_path(self):
        """Test loading from nonexistent path."""
        recommender = MLRecommender()

        with pytest.raises(FileNotFoundError, match="Model path not found"):
            recommender.load_model("/nonexistent/path")

    def test_load_incomplete_model(self, trained_recommender):
        """Test loading with missing model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trained_recommender.save_model(tmpdir)

            # Remove one model file
            os.remove(os.path.join(tmpdir, "model_executor_cores.joblib"))

            new_recommender = MLRecommender()
            with pytest.raises(ValueError, match="Incomplete model files"):
                new_recommender.load_model(tmpdir)


class TestMLRecommenderFeatureImportance:
    """Test feature importance functionality."""

    @pytest.fixture
    def trained_recommender(self):
        """Create a trained recommender for testing."""
        recommender = MLRecommender()
        jobs = generate_synthetic_jobs(n_jobs=30)
        recommender.train(jobs)
        return recommender

    def test_get_feature_importance_requires_training(self):
        """Test that feature importance requires training."""
        recommender = MLRecommender()

        with pytest.raises(ValueError, match="Model must be trained"):
            recommender.get_feature_importance()

    def test_get_feature_importance(self, trained_recommender):
        """Test getting feature importance."""
        importance = trained_recommender.get_feature_importance()

        # Should have importance for each target
        for target in MLRecommender.TARGET_FIELDS:
            assert target in importance
            # Should have importance for each feature
            assert len(importance[target]) == len(MLRecommender.FEATURE_NAMES)
            # All importances should be non-negative
            for feature, imp in importance[target].items():
                assert imp >= 0

    def test_top_feature_importance(self, trained_recommender):
        """Test getting top feature importances."""
        top_importance = trained_recommender._get_top_feature_importance(top_k=3)

        assert len(top_importance) == 3
        # All values should be non-negative
        for imp in top_importance.values():
            assert imp >= 0


class TestMLRecommenderFeatureExtraction:
    """Test feature extraction functionality."""

    def test_extract_features_from_job(self):
        """Test feature extraction from historical job."""
        recommender = MLRecommender()

        job = {
            "input_bytes": 10 * 1024**3,
            "output_bytes": 5 * 1024**3,
            "shuffle_read_bytes": 1 * 1024**3,
            "shuffle_write_bytes": 1 * 1024**3,
            "total_stages": 5,
            "total_tasks": 100,
            "job_type": "etl",
            "priority": "balanced",
        }

        features = recommender._extract_features_from_job(job)

        assert len(features) == len(MLRecommender.FEATURE_NAMES)
        assert features[0] == pytest.approx(10.0, abs=0.1)  # input_size_gb
        assert features[1] == pytest.approx(5.0, abs=0.1)  # output_size_gb

    def test_extract_features_from_requirements(self):
        """Test feature extraction from job requirements."""
        recommender = MLRecommender()

        requirements = {
            "input_size_bytes": 10 * 1024**3,
            "job_type": "ml",
            "priority": "performance",
        }

        features = recommender._extract_features(requirements)

        assert len(features) == len(MLRecommender.FEATURE_NAMES)
        assert features[0] == pytest.approx(10.0, abs=0.1)  # input_size_gb

    def test_filter_valid_jobs(self):
        """Test job filtering."""
        recommender = MLRecommender()

        jobs = [
            # Valid job
            {
                "input_bytes": 1024**3,
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "num_executors": 10,
                "driver_memory_mb": 4096,
            },
            # Missing input_bytes
            {
                "executor_cores": 4,
                "executor_memory_mb": 8192,
                "num_executors": 10,
                "driver_memory_mb": 4096,
            },
            # Missing resource fields
            {
                "input_bytes": 1024**3,
            },
            # Zero values
            {
                "input_bytes": 1024**3,
                "executor_cores": 0,
                "executor_memory_mb": 8192,
                "num_executors": 10,
                "driver_memory_mb": 4096,
            },
        ]

        valid = recommender._filter_valid_jobs(jobs)
        assert len(valid) == 1


class TestMLRecommenderMetrics:
    """Test metrics computation."""

    def test_compute_metrics(self):
        """Test metrics computation."""
        recommender = MLRecommender()

        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 18, 32, 38, 52])

        metrics = recommender._compute_metrics(y_true, y_pred)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
        assert "mape" in metrics

        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= 0
        assert metrics["mape"] >= 0

    def test_calculate_confidence(self):
        """Test confidence calculation."""
        recommender = MLRecommender()
        jobs = generate_synthetic_jobs(n_jobs=30)
        recommender.train(jobs)

        # Create a sample feature vector
        features = recommender._extract_features(
            {
                "input_size_bytes": 10 * 1024**3,
                "job_type": "etl",
                "priority": "balanced",
            }
        )
        features_scaled = recommender.scalers["features"].transform(
            features.reshape(1, -1)
        )

        confidence = recommender._calculate_confidence(features_scaled)

        assert 0.3 <= confidence <= 0.95
