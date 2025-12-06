"""Job similarity calculation for matching historical jobs."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class JobSimilarityCalculator:
    """Calculate similarity between Spark jobs."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """Initialize similarity calculator.

        Args:
            weights: Feature weights for similarity calculation
        """
        self.weights = weights or self._default_weights()

    def _default_weights(self) -> Dict[str, float]:
        """Get default feature weights.

        Returns:
            Dictionary of feature weights
        """
        return {
            "input_size": 0.3,
            "app_name": 0.2,
            "num_stages": 0.15,
            "executor_config": 0.15,
            "shuffle_size": 0.1,
            "output_size": 0.1,
        }

    def calculate_similarity(self, job1: Dict, job2: Dict) -> float:
        """Calculate similarity score between two jobs.

        Args:
            job1: First job data
            job2: Second job data

        Returns:
            Similarity score between 0 and 1
        """
        # TODO: Implement similarity calculation
        # - Compare input sizes
        # - Compare application names (text similarity)
        # - Compare number of stages/tasks
        # - Compare resource configurations
        # - Weighted combination of all factors

        scores = []

        # Input size similarity
        if "input_bytes" in job1 and "input_bytes" in job2:
            input_sim = self._size_similarity(job1["input_bytes"], job2["input_bytes"])
            scores.append(input_sim * self.weights["input_size"])

        # TODO: Add more similarity metrics

        return sum(scores) if scores else 0.0

    def _size_similarity(self, size1: int, size2: int) -> float:
        """Calculate similarity between two sizes.

        Args:
            size1: First size in bytes
            size2: Second size in bytes

        Returns:
            Similarity score between 0 and 1
        """
        if size1 == 0 and size2 == 0:
            return 1.0
        if size1 == 0 or size2 == 0:
            return 0.0

        ratio = min(size1, size2) / max(size1, size2)
        return ratio

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between job names.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # TODO: Implement text similarity
        # - Use edit distance, n-grams, or other methods
        return 0.0

    def find_similar_jobs(
        self, target_job: Dict, candidate_jobs: List[Dict], top_k: int = 5
    ) -> List[Tuple[Dict, float]]:
        """Find most similar jobs from candidates.

        Args:
            target_job: Job to find matches for
            candidate_jobs: List of candidate jobs
            top_k: Number of top matches to return

        Returns:
            List of (job, similarity_score) tuples, sorted by similarity
        """
        # TODO: Implement similarity ranking
        similarities = []
        for candidate in candidate_jobs:
            score = self.calculate_similarity(target_job, candidate)
            similarities.append((candidate, score))

        # Sort by similarity score descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def calculate_feature_vector(self, job: Dict) -> np.ndarray:
        """Extract feature vector from job data.

        Args:
            job: Job data dictionary

        Returns:
            NumPy array of features
        """
        # TODO: Implement feature extraction
        # - Normalize features
        # - Handle missing values
        # - Create fixed-length vector
        features: List[float] = []
        return np.array(features)
