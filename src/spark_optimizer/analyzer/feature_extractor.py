"""Feature extraction for machine learning models."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class FeatureExtractor:
    """Extract ML features from Spark job data."""

    # Feature names in order
    FEATURE_NAMES = [
        "input_size_gb",
        "output_size_gb",
        "shuffle_size_gb",
        "total_cores",
        "total_memory_gb",
        "executor_cores",
        "num_stages",
        "num_tasks",
        "avg_tasks_per_stage",
        "io_ratio",
        "shuffle_to_input_ratio",
    ]

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = list(self.FEATURE_NAMES)
        # Statistics for normalization (fitted from data)
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._mins: Optional[np.ndarray] = None
        self._maxs: Optional[np.ndarray] = None
        self._is_fitted = False

    def extract_features(self, job_data: Dict) -> Dict:
        """Extract features from job data.

        Args:
            job_data: Raw job data

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Basic job characteristics
        features.update(self._extract_size_features(job_data))
        features.update(self._extract_resource_features(job_data))
        features.update(self._extract_complexity_features(job_data))
        features.update(self._extract_io_features(job_data))

        return features

    def _extract_size_features(self, job_data: Dict) -> Dict:
        """Extract data size related features.

        Args:
            job_data: Job data

        Returns:
            Size features
        """
        return {
            "input_size_gb": job_data.get("input_bytes", 0) / (1024**3),
            "output_size_gb": job_data.get("output_bytes", 0) / (1024**3),
            "shuffle_size_gb": job_data.get("shuffle_read_bytes", 0) / (1024**3),
        }

    def _extract_resource_features(self, job_data: Dict) -> Dict:
        """Extract resource configuration features.

        Args:
            job_data: Job data

        Returns:
            Resource features
        """
        num_executors = job_data.get("num_executors", 0)
        executor_cores = job_data.get("executor_cores", 0)
        executor_memory_mb = job_data.get("executor_memory_mb", 0)

        return {
            "total_cores": num_executors * executor_cores,
            "total_memory_gb": num_executors * executor_memory_mb / 1024,
            "executor_cores": executor_cores,
        }

    def _extract_complexity_features(self, job_data: Dict) -> Dict:
        """Extract job complexity features.

        Args:
            job_data: Job data

        Returns:
            Complexity features
        """
        total_stages = job_data.get("total_stages", 0)
        total_tasks = job_data.get("total_tasks", 0)

        return {
            "num_stages": total_stages,
            "num_tasks": total_tasks,
            "avg_tasks_per_stage": total_tasks / max(total_stages, 1),
        }

    def _extract_io_features(self, job_data: Dict) -> Dict:
        """Extract I/O related features.

        Args:
            job_data: Job data

        Returns:
            I/O features
        """
        input_bytes = job_data.get("input_bytes", 0)
        output_bytes = job_data.get("output_bytes", 0)
        shuffle_read = job_data.get("shuffle_read_bytes", 0)

        return {
            "io_ratio": output_bytes / max(input_bytes, 1),
            "shuffle_to_input_ratio": shuffle_read / max(input_bytes, 1),
        }

    def create_feature_matrix(self, jobs: List[Dict]) -> np.ndarray:
        """Create feature matrix for multiple jobs.

        Args:
            jobs: List of job data dictionaries

        Returns:
            NumPy array of shape (n_jobs, n_features)
        """
        if not jobs:
            return np.array([])

        feature_list = []
        for job in jobs:
            features = self.extract_features(job)
            # Ensure consistent ordering
            feature_vector = [features.get(name, 0.0) for name in self.feature_names]
            feature_list.append(feature_vector)

        matrix = np.array(feature_list, dtype=np.float64)

        # Handle missing values (replace NaN/Inf with 0)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return matrix

    def fit(self, feature_matrix: np.ndarray) -> "FeatureExtractor":
        """Fit the normalizer to training data.

        Args:
            feature_matrix: Training feature matrix

        Returns:
            Self for method chaining
        """
        if feature_matrix.size == 0:
            return self

        # Compute statistics for standardization
        self._means = np.mean(feature_matrix, axis=0)
        self._stds = np.std(feature_matrix, axis=0)
        # Avoid division by zero
        self._stds = np.where(self._stds == 0, 1.0, self._stds)

        # Compute min/max for min-max scaling
        self._mins = np.min(feature_matrix, axis=0)
        self._maxs = np.max(feature_matrix, axis=0)
        # Avoid division by zero
        ranges = self._maxs - self._mins
        ranges = np.where(ranges == 0, 1.0, ranges)
        self._maxs = self._mins + ranges

        self._is_fitted = True
        return self

    def normalize_features(
        self,
        feature_matrix: np.ndarray,
        method: str = "standard",
    ) -> np.ndarray:
        """Normalize feature matrix.

        Args:
            feature_matrix: Raw feature matrix
            method: Normalization method - "standard" (z-score), "minmax", or "log"

        Returns:
            Normalized feature matrix
        """
        if feature_matrix.size == 0:
            return feature_matrix

        # Auto-fit if not fitted yet
        if not self._is_fitted:
            self.fit(feature_matrix)

        if method == "standard":
            # Z-score normalization: (x - mean) / std
            normalized = (feature_matrix - self._means) / self._stds

        elif method == "minmax":
            # Min-max normalization: (x - min) / (max - min)
            assert self._mins is not None and self._maxs is not None  # nosec B101
            ranges = self._maxs - self._mins
            normalized = (feature_matrix - self._mins) / ranges

        elif method == "log":
            # Log transformation for skewed features
            # Add 1 to avoid log(0), then apply z-score
            log_matrix = np.log1p(np.abs(feature_matrix))
            # Fit on log-transformed data
            log_means = np.mean(log_matrix, axis=0)
            log_stds = np.std(log_matrix, axis=0)
            log_stds = np.where(log_stds == 0, 1.0, log_stds)
            normalized = (log_matrix - log_means) / log_stds

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Handle any remaining NaN/Inf values
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

        return normalized

    def fit_transform(
        self,
        feature_matrix: np.ndarray,
        method: str = "standard",
    ) -> np.ndarray:
        """Fit and transform feature matrix in one step.

        Args:
            feature_matrix: Raw feature matrix
            method: Normalization method

        Returns:
            Normalized feature matrix
        """
        self.fit(feature_matrix)
        return self.normalize_features(feature_matrix, method=method)

    def transform(
        self,
        feature_matrix: np.ndarray,
        method: str = "standard",
    ) -> np.ndarray:
        """Transform feature matrix using previously fitted statistics.

        Args:
            feature_matrix: Raw feature matrix
            method: Normalization method

        Returns:
            Normalized feature matrix

        Raises:
            ValueError: If not fitted yet
        """
        if not self._is_fitted:
            raise ValueError("FeatureExtractor not fitted. Call fit() first.")

        return self.normalize_features(feature_matrix, method=method)

    def get_feature_names(self) -> List[str]:
        """Get list of feature names.

        Returns:
            List of feature names in order
        """
        return list(self.feature_names)

    def extract_feature_vector(self, job_data: Dict) -> np.ndarray:
        """Extract feature vector for a single job.

        Args:
            job_data: Job data dictionary

        Returns:
            NumPy array of features
        """
        features = self.extract_features(job_data)
        return np.array([features.get(name, 0.0) for name in self.feature_names])

    def get_statistics(self) -> Dict:
        """Get fitted statistics.

        Returns:
            Dictionary with means, stds, mins, maxs for each feature
        """
        if not self._is_fitted:
            return {}

        assert (  # nosec B101
            self._means is not None
            and self._stds is not None
            and self._mins is not None
            and self._maxs is not None
        )
        return {
            "means": dict(zip(self.feature_names, self._means.tolist())),
            "stds": dict(zip(self.feature_names, self._stds.tolist())),
            "mins": dict(zip(self.feature_names, self._mins.tolist())),
            "maxs": dict(zip(self.feature_names, self._maxs.tolist())),
        }
