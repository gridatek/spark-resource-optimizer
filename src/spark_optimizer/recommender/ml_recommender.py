"""Machine learning-based recommender using trained models."""

import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from .base_recommender import BaseRecommender
from spark_optimizer.analyzer.feature_extractor import FeatureExtractor


class MLRecommender(BaseRecommender):
    """ML-based recommender using trained models for resource prediction."""

    # Target resource fields that we predict
    TARGET_FIELDS = [
        "executor_cores",
        "executor_memory_mb",
        "num_executors",
        "driver_memory_mb",
    ]

    # Feature names used for prediction (in order)
    FEATURE_NAMES = [
        "input_size_gb",
        "output_size_gb",
        "shuffle_size_gb",
        "num_stages",
        "num_tasks",
        "avg_tasks_per_stage",
        "io_ratio",
        "shuffle_to_input_ratio",
        "job_type_etl",
        "job_type_ml",
        "job_type_streaming",
        "job_type_batch",
        "job_type_other",
        "priority_performance",
        "priority_cost",
        "priority_balanced",
    ]

    # Valid resource ranges for post-processing predictions
    RESOURCE_RANGES = {
        "executor_cores": (1, 32),
        "executor_memory_mb": (1024, 65536),
        "num_executors": (1, 500),
        "driver_memory_mb": (1024, 32768),
    }

    def __init__(self, config: Optional[Dict] = None):
        """Initialize ML recommender.

        Args:
            config: Optional configuration dictionary with keys:
                - model_type: 'random_forest' or 'gradient_boosting'
                - n_estimators: Number of trees (default: 100)
                - max_depth: Maximum tree depth (default: 10)
                - model_path: Path to save/load models
        """
        super().__init__(config)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {
            "features": StandardScaler(),
        }
        self.is_trained = False
        self.feature_extractor = FeatureExtractor()
        self.training_stats: Dict[str, Any] = {}

        # Model configuration
        self.model_type = self.config.get("model_type", "random_forest")
        self.n_estimators = self.config.get("n_estimators", 100)
        self.max_depth = self.config.get("max_depth", 10)
        self.model_path = self.config.get("model_path", "models")

    def recommend(
        self,
        input_size_bytes: int,
        job_type: Optional[str] = None,
        sla_minutes: Optional[int] = None,
        budget_dollars: Optional[float] = None,
        priority: str = "balanced",
    ) -> Dict:
        """Generate ML-based recommendations.

        Args:
            input_size_bytes: Expected input data size in bytes
            job_type: Type of job (e.g., etl, ml, streaming, batch)
            sla_minutes: Maximum acceptable duration in minutes
            budget_dollars: Maximum acceptable cost in dollars
            priority: Optimization priority (performance, cost, or balanced)

        Returns:
            Recommendation dictionary with configuration, confidence, and metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making recommendations")

        # Build job requirements dict
        job_requirements = {
            "input_size_bytes": input_size_bytes,
            "job_type": job_type,
            "sla_minutes": sla_minutes,
            "budget_dollars": budget_dollars,
            "priority": priority,
        }

        # Extract and normalize features
        features = self._extract_features(job_requirements)
        features_scaled = self.scalers["features"].transform(features.reshape(1, -1))

        # Make predictions for each resource
        predictions = self._predict(features_scaled)

        # Apply constraints based on SLA and budget
        predictions = self._apply_constraints(predictions, job_requirements)

        # Calculate confidence based on model performance and feature similarity
        confidence = self._calculate_confidence(features_scaled)

        return self._create_recommendation_response(
            executor_cores=predictions["executor_cores"],
            executor_memory_mb=predictions["executor_memory_mb"],
            num_executors=predictions["num_executors"],
            driver_memory_mb=predictions["driver_memory_mb"],
            confidence=confidence,
            metadata={
                "method": "ml",
                "model_type": self.model_type,
                "training_samples": self.training_stats.get("n_samples", 0),
                "feature_importance": self._get_top_feature_importance(),
            },
        )

    def train(self, historical_jobs: List[Dict], test_size: float = 0.2) -> Dict:
        """Train ML models with historical data.

        Args:
            historical_jobs: List of historical job data dictionaries
            test_size: Fraction of data to use for testing (default: 0.2)

        Returns:
            Dictionary containing training metrics
        """
        if not historical_jobs:
            raise ValueError("Cannot train with empty dataset")

        if len(historical_jobs) < 10:
            raise ValueError(
                f"Insufficient training data: {len(historical_jobs)} jobs. "
                "Need at least 10 jobs for training."
            )

        # Filter jobs with valid resource configurations
        valid_jobs = self._filter_valid_jobs(historical_jobs)
        if len(valid_jobs) < 10:
            raise ValueError(
                f"Only {len(valid_jobs)} valid jobs after filtering. "
                "Need at least 10 jobs with complete resource configurations."
            )

        # Extract features and targets
        X, y_dict = self._prepare_training_data(valid_jobs)

        # Split data
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, np.arange(len(X)), test_size=test_size, random_state=42
        )

        # Fit feature scaler
        self.scalers["features"].fit(X_train)
        X_train_scaled = self.scalers["features"].transform(X_train)
        X_test_scaled = self.scalers["features"].transform(X_test)

        # Train a model for each target resource
        training_metrics = {}
        for target_name in self.TARGET_FIELDS:
            y = y_dict[target_name]
            y_train = y[indices_train]
            y_test = y[indices_test]

            # Create and train model
            model = self._create_model()
            model.fit(X_train_scaled, y_train)
            self.models[target_name] = model

            # Evaluate on test set
            y_pred = model.predict(X_test_scaled)
            metrics = self._compute_metrics(y_test, y_pred)
            training_metrics[target_name] = metrics

            # Cross-validation score
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=min(5, len(X_train)), scoring="r2"
            )
            training_metrics[target_name]["cv_r2_mean"] = float(np.mean(cv_scores))
            training_metrics[target_name]["cv_r2_std"] = float(np.std(cv_scores))

        # Store training statistics
        self.training_stats = {
            "n_samples": len(valid_jobs),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "metrics": training_metrics,
            "feature_names": self.FEATURE_NAMES,
        }

        self.is_trained = True
        return training_metrics

    def _filter_valid_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Filter jobs with valid resource configurations.

        Args:
            jobs: List of job dictionaries

        Returns:
            Filtered list of valid jobs
        """
        valid_jobs = []
        for job in jobs:
            # Check if job has all required target fields with valid values
            has_valid_targets = all(
                job.get(field, 0) > 0 for field in self.TARGET_FIELDS
            )
            # Check if job has input size information
            has_input_size = (
                job.get("input_bytes", 0) > 0 or job.get("input_size_bytes", 0) > 0
            )

            if has_valid_targets and has_input_size:
                valid_jobs.append(job)

        return valid_jobs

    def _prepare_training_data(
        self, jobs: List[Dict]
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare feature matrix and target vectors from historical jobs.

        Args:
            jobs: List of valid job dictionaries

        Returns:
            Tuple of (feature_matrix, target_dict)
        """
        X_list = []
        y_dict_list: Dict[str, List[Any]] = {field: [] for field in self.TARGET_FIELDS}

        for job in jobs:
            # Extract features
            features = self._extract_features_from_job(job)
            X_list.append(features)

            # Extract targets
            for field in self.TARGET_FIELDS:
                y_dict_list[field].append(job.get(field, 0))

        X = np.array(X_list)
        y_dict: Dict[str, np.ndarray] = {
            field: np.array(values) for field, values in y_dict_list.items()
        }

        return X, y_dict

    def _extract_features_from_job(self, job: Dict) -> np.ndarray:
        """Extract feature vector from a historical job.

        Args:
            job: Job data dictionary

        Returns:
            Feature vector as numpy array
        """
        # Size features (convert to GB)
        input_bytes = job.get("input_bytes", 0) or job.get("input_size_bytes", 0) or 0
        input_size_gb = input_bytes / (1024**3)
        output_size_gb = job.get("output_bytes", 0) / (1024**3)
        shuffle_bytes = job.get("shuffle_read_bytes", 0) + job.get(
            "shuffle_write_bytes", 0
        )
        shuffle_size_gb = shuffle_bytes / (1024**3)

        # Complexity features
        num_stages = job.get("total_stages", 0) or job.get("num_stages", 1)
        num_tasks = job.get("total_tasks", 0) or job.get("num_tasks", 0)
        avg_tasks_per_stage = num_tasks / max(num_stages, 1)

        # I/O features
        io_ratio = (
            job.get("output_bytes", 0) / max(input_bytes, 1) if input_bytes > 0 else 0
        )
        shuffle_to_input_ratio = shuffle_bytes / max(input_bytes, 1)

        # Job type encoding (one-hot)
        job_type = job.get("job_type", "").lower() if job.get("job_type") else ""
        job_type_etl = 1.0 if job_type == "etl" else 0.0
        job_type_ml = 1.0 if job_type == "ml" else 0.0
        job_type_streaming = 1.0 if job_type == "streaming" else 0.0
        job_type_batch = 1.0 if job_type == "batch" else 0.0
        job_type_other = (
            1.0
            if job_type and job_type not in ["etl", "ml", "streaming", "batch"]
            else 0.0
        )

        # Priority encoding (one-hot)
        priority = (
            job.get("priority", "balanced").lower()
            if job.get("priority")
            else "balanced"
        )
        priority_performance = 1.0 if priority == "performance" else 0.0
        priority_cost = 1.0 if priority == "cost" else 0.0
        priority_balanced = 1.0 if priority == "balanced" else 0.0

        return np.array(
            [
                input_size_gb,
                output_size_gb,
                shuffle_size_gb,
                num_stages,
                num_tasks,
                avg_tasks_per_stage,
                io_ratio,
                shuffle_to_input_ratio,
                job_type_etl,
                job_type_ml,
                job_type_streaming,
                job_type_batch,
                job_type_other,
                priority_performance,
                priority_cost,
                priority_balanced,
            ]
        )

    def _extract_features(self, job_requirements: Dict) -> np.ndarray:
        """Extract features from job requirements for prediction.

        Args:
            job_requirements: Job requirements dictionary

        Returns:
            Feature array
        """
        # Size features
        input_bytes = job_requirements.get("input_size_bytes", 0)
        input_size_gb = input_bytes / (1024**3)

        # For prediction, we don't have output/shuffle info yet, estimate based on job type
        job_type = (
            job_requirements.get("job_type", "").lower()
            if job_requirements.get("job_type")
            else ""
        )

        # Estimate output and shuffle based on job type heuristics
        if job_type == "etl":
            output_size_gb = input_size_gb * 0.8  # ETL usually compresses/filters
            shuffle_size_gb = input_size_gb * 0.3
        elif job_type == "ml":
            output_size_gb = input_size_gb * 0.1  # ML outputs models, small
            shuffle_size_gb = input_size_gb * 0.5  # ML has more shuffling
        elif job_type == "streaming":
            output_size_gb = input_size_gb * 0.5
            shuffle_size_gb = input_size_gb * 0.2
        else:
            output_size_gb = input_size_gb * 0.5
            shuffle_size_gb = input_size_gb * 0.3

        # Estimate complexity based on input size
        num_stages = max(1, int(np.log2(input_size_gb + 1) * 3))
        num_tasks = max(1, int(input_size_gb * 10))  # ~10 tasks per GB
        avg_tasks_per_stage = num_tasks / max(num_stages, 1)

        # I/O ratios
        io_ratio = output_size_gb / max(input_size_gb, 0.001)
        shuffle_to_input_ratio = shuffle_size_gb / max(input_size_gb, 0.001)

        # Job type encoding
        job_type_etl = 1.0 if job_type == "etl" else 0.0
        job_type_ml = 1.0 if job_type == "ml" else 0.0
        job_type_streaming = 1.0 if job_type == "streaming" else 0.0
        job_type_batch = 1.0 if job_type == "batch" else 0.0
        job_type_other = (
            1.0
            if job_type and job_type not in ["etl", "ml", "streaming", "batch"]
            else 0.0
        )

        # Priority encoding
        priority = (
            job_requirements.get("priority", "balanced").lower()
            if job_requirements.get("priority")
            else "balanced"
        )
        priority_performance = 1.0 if priority == "performance" else 0.0
        priority_cost = 1.0 if priority == "cost" else 0.0
        priority_balanced = 1.0 if priority == "balanced" else 0.0

        return np.array(
            [
                input_size_gb,
                output_size_gb,
                shuffle_size_gb,
                num_stages,
                num_tasks,
                avg_tasks_per_stage,
                io_ratio,
                shuffle_to_input_ratio,
                job_type_etl,
                job_type_ml,
                job_type_streaming,
                job_type_batch,
                job_type_other,
                priority_performance,
                priority_cost,
                priority_balanced,
            ]
        )

    def _create_model(self) -> Any:
        """Create a new model instance based on configuration.

        Returns:
            Scikit-learn model instance
        """
        if self.model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                learning_rate=0.1,
            )
        else:
            # Default to Random Forest
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,  # Use all CPU cores
            )

    def _predict(self, features_scaled: np.ndarray) -> Dict:
        """Make predictions using trained models.

        Args:
            features_scaled: Scaled feature array (1, n_features)

        Returns:
            Dictionary of predictions
        """
        predictions = {}

        for target_name, model in self.models.items():
            # Get raw prediction
            raw_pred = model.predict(features_scaled)[0]

            # Clip to valid range
            min_val, max_val = self.RESOURCE_RANGES[target_name]
            pred_value = max(min_val, min(max_val, raw_pred))

            # Round to appropriate values
            if target_name == "executor_memory_mb":
                # Round memory to nearest 512MB
                pred_value = int(round(pred_value / 512) * 512)
            elif target_name == "driver_memory_mb":
                # Round driver memory to nearest 512MB
                pred_value = int(round(pred_value / 512) * 512)
            else:
                pred_value = int(round(pred_value))

            predictions[target_name] = pred_value

        return predictions

    def _apply_constraints(self, predictions: Dict, job_requirements: Dict) -> Dict:
        """Apply SLA and budget constraints to predictions.

        Args:
            predictions: Raw predictions dictionary
            job_requirements: Job requirements with constraints

        Returns:
            Adjusted predictions
        """
        sla_minutes = job_requirements.get("sla_minutes")
        budget_dollars = job_requirements.get("budget_dollars")
        priority = job_requirements.get("priority", "balanced")

        # Adjust based on priority
        if priority == "performance" and sla_minutes:
            # Increase resources for performance priority with SLA
            predictions["num_executors"] = int(predictions["num_executors"] * 1.2)
            predictions["executor_memory_mb"] = int(
                predictions["executor_memory_mb"] * 1.1
            )
        elif priority == "cost" and budget_dollars:
            # Reduce resources for cost priority
            predictions["num_executors"] = max(
                1, int(predictions["num_executors"] * 0.8)
            )
            predictions["executor_memory_mb"] = max(
                1024, int(predictions["executor_memory_mb"] * 0.9)
            )

        # Ensure values are within valid ranges after adjustments
        for field, (min_val, max_val) in self.RESOURCE_RANGES.items():
            predictions[field] = max(min_val, min(max_val, predictions[field]))

        return predictions

    def _calculate_confidence(self, features_scaled: np.ndarray) -> float:
        """Calculate confidence score for the prediction.

        Based on model performance metrics and feature coverage.

        Args:
            features_scaled: Scaled feature array

        Returns:
            Confidence score between 0 and 1
        """
        if not self.training_stats.get("metrics"):
            return 0.5

        # Base confidence from average R2 scores
        r2_scores = []
        for target_name, metrics in self.training_stats["metrics"].items():
            r2 = metrics.get("r2", 0)
            r2_scores.append(max(0, r2))  # Clip negative R2

        avg_r2 = np.mean(r2_scores) if r2_scores else 0.5

        # Adjust confidence based on how well the input matches training data
        # Using distance from scaled feature mean (should be ~0 for normalized data)
        feature_distance = np.mean(np.abs(features_scaled))
        distance_penalty = min(0.2, feature_distance * 0.1)

        confidence = max(0.3, min(0.95, avg_r2 - distance_penalty))
        return round(confidence, 2)

    def _compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: True target values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(
                np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
            ),
        }

    def _get_top_feature_importance(self, top_k: int = 5) -> Dict[str, float]:
        """Get top feature importances across all models.

        Args:
            top_k: Number of top features to return

        Returns:
            Dictionary of feature name to average importance
        """
        if not self.models:
            return {}

        # Average feature importance across all models
        importances = np.zeros(len(self.FEATURE_NAMES))
        for model in self.models.values():
            if hasattr(model, "feature_importances_"):
                importances += model.feature_importances_

        importances /= len(self.models)

        # Get top K features
        top_indices = np.argsort(importances)[-top_k:][::-1]
        return {
            self.FEATURE_NAMES[i]: round(float(importances[i]), 4) for i in top_indices
        }

    def evaluate(self, test_jobs: List[Dict]) -> Dict:
        """Evaluate model performance on test data.

        Args:
            test_jobs: List of test job data

        Returns:
            Dictionary of evaluation metrics per target
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        if not test_jobs:
            raise ValueError("Cannot evaluate with empty test set")

        valid_jobs = self._filter_valid_jobs(test_jobs)
        if not valid_jobs:
            raise ValueError("No valid jobs in test set")

        X, y_dict = self._prepare_training_data(valid_jobs)
        X_scaled = self.scalers["features"].transform(X)

        metrics = {}
        for target_name in self.TARGET_FIELDS:
            y_true = y_dict[target_name]
            y_pred = self.models[target_name].predict(X_scaled)

            # Apply valid range clipping
            min_val, max_val = self.RESOURCE_RANGES[target_name]
            y_pred = np.clip(y_pred, min_val, max_val)

            metrics[target_name] = self._compute_metrics(y_true, y_pred)

        return metrics

    def save_model(self, path: Optional[str] = None) -> str:
        """Save trained models and scalers to disk.

        Args:
            path: Directory path to save models (default: self.model_path)

        Returns:
            Path where models were saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        save_path = path or self.model_path
        os.makedirs(save_path, exist_ok=True)

        # Save models
        for target_name, model in self.models.items():
            model_file = os.path.join(save_path, f"model_{target_name}.joblib")
            joblib.dump(model, model_file)

        # Save scaler
        scaler_file = os.path.join(save_path, "scaler_features.joblib")
        joblib.dump(self.scalers["features"], scaler_file)

        # Save metadata
        metadata = {
            "is_trained": self.is_trained,
            "model_type": self.model_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "training_stats": self.training_stats,
            "feature_names": self.FEATURE_NAMES,
            "target_fields": self.TARGET_FIELDS,
        }
        metadata_file = os.path.join(save_path, "metadata.joblib")
        joblib.dump(metadata, metadata_file)

        return save_path

    def load_model(self, path: Optional[str] = None) -> None:
        """Load trained models and scalers from disk.

        Args:
            path: Directory path to load models from (default: self.model_path)
        """
        load_path = path or self.model_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model path not found: {load_path}")

        # Load metadata
        metadata_file = os.path.join(load_path, "metadata.joblib")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        metadata = joblib.load(metadata_file)
        self.model_type = metadata.get("model_type", "random_forest")
        self.n_estimators = metadata.get("n_estimators", 100)
        self.max_depth = metadata.get("max_depth", 10)
        self.training_stats = metadata.get("training_stats", {})

        # Load scaler
        scaler_file = os.path.join(load_path, "scaler_features.joblib")
        if os.path.exists(scaler_file):
            self.scalers["features"] = joblib.load(scaler_file)

        # Load models
        self.models = {}
        for target_name in self.TARGET_FIELDS:
            model_file = os.path.join(load_path, f"model_{target_name}.joblib")
            if os.path.exists(model_file):
                self.models[target_name] = joblib.load(model_file)

        if len(self.models) == len(self.TARGET_FIELDS):
            self.is_trained = True
        else:
            raise ValueError(
                f"Incomplete model files. Found {len(self.models)} of "
                f"{len(self.TARGET_FIELDS)} required models."
            )

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance for each target model.

        Returns:
            Dictionary mapping target names to feature importance dicts
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to get feature importance")

        importance_dict = {}
        for target_name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importance_dict[target_name] = {
                    name: round(float(imp), 4)
                    for name, imp in zip(self.FEATURE_NAMES, model.feature_importances_)
                }

        return importance_dict
