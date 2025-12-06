"""Feature extraction for machine learning models."""

from typing import Dict, List
import numpy as np


class FeatureExtractor:
    """Extract ML features from Spark job data."""

    def __init__(self):
        """Initialize feature extractor."""
        self.feature_names = []

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
        # TODO: Implement size feature extraction
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
        # TODO: Implement resource feature extraction
        return {
            "total_cores": job_data.get("num_executors", 0)
            * job_data.get("executor_cores", 0),
            "total_memory_gb": job_data.get("num_executors", 0)
            * job_data.get("executor_memory_mb", 0)
            / 1024,
            "executor_cores": job_data.get("executor_cores", 0),
        }

    def _extract_complexity_features(self, job_data: Dict) -> Dict:
        """Extract job complexity features.

        Args:
            job_data: Job data

        Returns:
            Complexity features
        """
        # TODO: Implement complexity feature extraction
        return {
            "num_stages": job_data.get("total_stages", 0),
            "num_tasks": job_data.get("total_tasks", 0),
            "avg_tasks_per_stage": job_data.get("total_tasks", 0)
            / max(job_data.get("total_stages", 1), 1),
        }

    def _extract_io_features(self, job_data: Dict) -> Dict:
        """Extract I/O related features.

        Args:
            job_data: Job data

        Returns:
            I/O features
        """
        # TODO: Implement I/O feature extraction
        input_bytes = job_data.get("input_bytes", 0)
        output_bytes = job_data.get("output_bytes", 0)

        return {
            "io_ratio": output_bytes / max(input_bytes, 1),
            "shuffle_to_input_ratio": job_data.get("shuffle_read_bytes", 0)
            / max(input_bytes, 1),
        }

    def create_feature_matrix(self, jobs: List[Dict]) -> np.ndarray:
        """Create feature matrix for multiple jobs.

        Args:
            jobs: List of job data dictionaries

        Returns:
            NumPy array of shape (n_jobs, n_features)
        """
        # TODO: Implement matrix creation
        # - Extract features for all jobs
        # - Stack into matrix
        # - Handle missing values
        feature_list = []
        for job in jobs:
            features = self.extract_features(job)
            feature_list.append(list(features.values()))

        return np.array(feature_list) if feature_list else np.array([])

    def normalize_features(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Normalize feature matrix.

        Args:
            feature_matrix: Raw feature matrix

        Returns:
            Normalized feature matrix
        """
        # TODO: Implement normalization
        # - Standard scaling
        # - Min-max scaling
        # - Log transformation for skewed features
        return feature_matrix
