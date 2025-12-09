"""Job similarity calculation for matching historical jobs."""

from typing import Dict, List, Optional, Tuple
import numpy as np


class JobSimilarityCalculator:
    """Calculate similarity between Spark jobs."""

    # Feature names for vector representation
    FEATURE_NAMES = [
        "input_bytes",
        "output_bytes",
        "shuffle_read_bytes",
        "shuffle_write_bytes",
        "total_stages",
        "total_tasks",
        "num_executors",
        "executor_cores",
        "executor_memory_mb",
        "duration_ms",
    ]

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
        scores = []
        total_weight = 0.0

        # Input size similarity
        if "input_bytes" in job1 and "input_bytes" in job2:
            input_sim = self._size_similarity(job1["input_bytes"], job2["input_bytes"])
            weight = self.weights["input_size"]
            scores.append(input_sim * weight)
            total_weight += weight
        elif "input_size_bytes" in job1 and "input_bytes" in job2:
            # Handle job_requirements format
            input_sim = self._size_similarity(job1["input_size_bytes"], job2["input_bytes"])
            weight = self.weights["input_size"]
            scores.append(input_sim * weight)
            total_weight += weight

        # Application name similarity (text)
        if "app_name" in job1 and "app_name" in job2:
            name_sim = self._text_similarity(job1["app_name"], job2["app_name"])
            weight = self.weights["app_name"]
            scores.append(name_sim * weight)
            total_weight += weight

        # Number of stages similarity
        if "total_stages" in job1 and "total_stages" in job2:
            stages_sim = self._count_similarity(
                job1["total_stages"], job2["total_stages"]
            )
            weight = self.weights["num_stages"]
            scores.append(stages_sim * weight)
            total_weight += weight

        # Executor configuration similarity
        if all(
            k in job1 and k in job2
            for k in ["num_executors", "executor_cores", "executor_memory_mb"]
        ):
            config_sim = self._config_similarity(job1, job2)
            weight = self.weights["executor_config"]
            scores.append(config_sim * weight)
            total_weight += weight

        # Shuffle size similarity
        if "shuffle_write_bytes" in job1 and "shuffle_write_bytes" in job2:
            shuffle_sim = self._size_similarity(
                job1["shuffle_write_bytes"], job2["shuffle_write_bytes"]
            )
            weight = self.weights["shuffle_size"]
            scores.append(shuffle_sim * weight)
            total_weight += weight

        # Output size similarity
        if "output_bytes" in job1 and "output_bytes" in job2:
            output_sim = self._size_similarity(
                job1["output_bytes"], job2["output_bytes"]
            )
            weight = self.weights["output_size"]
            scores.append(output_sim * weight)
            total_weight += weight

        # Normalize by total weight to get score between 0 and 1
        return sum(scores) / total_weight if total_weight > 0 else 0.0

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
        if not text1 or not text2:
            return 0.0

        text1 = text1.lower()
        text2 = text2.lower()

        # Exact match
        if text1 == text2:
            return 1.0

        # Check if one contains the other
        if text1 in text2 or text2 in text1:
            return 0.8

        # Calculate Jaccard similarity using character n-grams
        n = 2  # Use bigrams
        ngrams1 = set(text1[i : i + n] for i in range(len(text1) - n + 1))
        ngrams2 = set(text2[i : i + n] for i in range(len(text2) - n + 1))

        if not ngrams1 and not ngrams2:
            return 1.0
        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)

        return intersection / union if union > 0 else 0.0

    def _count_similarity(self, count1: int, count2: int) -> float:
        """Calculate similarity between two counts.

        Args:
            count1: First count
            count2: Second count

        Returns:
            Similarity score between 0 and 1
        """
        if count1 == 0 and count2 == 0:
            return 1.0
        if count1 == 0 or count2 == 0:
            return 0.0

        ratio = min(count1, count2) / max(count1, count2)
        return ratio

    def _config_similarity(self, job1: Dict, job2: Dict) -> float:
        """Calculate similarity between executor configurations.

        Args:
            job1: First job data
            job2: Second job data

        Returns:
            Similarity score between 0 and 1
        """
        scores = []

        # Compare number of executors
        executors_sim = self._count_similarity(
            job1.get("num_executors", 0), job2.get("num_executors", 0)
        )
        scores.append(executors_sim)

        # Compare cores per executor
        cores_sim = self._count_similarity(
            job1.get("executor_cores", 0), job2.get("executor_cores", 0)
        )
        scores.append(cores_sim)

        # Compare memory per executor
        memory_sim = self._size_similarity(
            job1.get("executor_memory_mb", 0), job2.get("executor_memory_mb", 0)
        )
        scores.append(memory_sim)

        return sum(scores) / len(scores) if scores else 0.0

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
            NumPy array of normalized features
        """
        # Extract raw features
        features = []

        # Size features (log-scaled to handle large values)
        input_bytes = job.get("input_bytes", job.get("input_size_bytes", 0))
        output_bytes = job.get("output_bytes", 0)
        shuffle_read = job.get("shuffle_read_bytes", 0)
        shuffle_write = job.get("shuffle_write_bytes", 0)

        # Use log1p to handle zeros and normalize large values
        features.append(np.log1p(input_bytes))
        features.append(np.log1p(output_bytes))
        features.append(np.log1p(shuffle_read))
        features.append(np.log1p(shuffle_write))

        # Count features (log-scaled)
        features.append(np.log1p(job.get("total_stages", 0)))
        features.append(np.log1p(job.get("total_tasks", 0)))

        # Resource configuration features
        features.append(np.log1p(job.get("num_executors", 0)))
        features.append(job.get("executor_cores", 0))
        features.append(np.log1p(job.get("executor_memory_mb", 0)))

        # Performance feature
        features.append(np.log1p(job.get("duration_ms", 0)))

        return np.array(features, dtype=np.float64)

    def calculate_cosine_similarity(
        self, vector1: np.ndarray, vector2: np.ndarray
    ) -> float:
        """Calculate cosine similarity between two feature vectors.

        Args:
            vector1: First feature vector
            vector2: Second feature vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        if vector1.size == 0 or vector2.size == 0:
            return 0.0

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vector1, vector2) / (norm1 * norm2))

    def calculate_euclidean_distance(
        self, vector1: np.ndarray, vector2: np.ndarray
    ) -> float:
        """Calculate Euclidean distance between two feature vectors.

        Args:
            vector1: First feature vector
            vector2: Second feature vector

        Returns:
            Euclidean distance
        """
        if vector1.size == 0 or vector2.size == 0:
            return float("inf")

        return float(np.linalg.norm(vector1 - vector2))

    def find_similar_jobs_vectorized(
        self,
        target_job: Dict,
        candidate_jobs: List[Dict],
        top_k: int = 5,
        method: str = "cosine",
    ) -> List[Tuple[Dict, float]]:
        """Find similar jobs using vectorized similarity calculation.

        This method is faster for large candidate sets.

        Args:
            target_job: Job to find matches for
            candidate_jobs: List of candidate jobs
            top_k: Number of top matches to return
            method: Similarity method - "cosine" or "euclidean"

        Returns:
            List of (job, similarity_score) tuples
        """
        if not candidate_jobs:
            return []

        # Extract feature vector for target
        target_vector = self.calculate_feature_vector(target_job)

        # Build feature matrix for candidates
        candidate_vectors = np.array([
            self.calculate_feature_vector(job) for job in candidate_jobs
        ])

        if method == "cosine":
            # Normalize vectors
            target_norm = target_vector / (np.linalg.norm(target_vector) + 1e-10)
            candidate_norms = candidate_vectors / (
                np.linalg.norm(candidate_vectors, axis=1, keepdims=True) + 1e-10
            )
            # Calculate cosine similarities
            similarities = np.dot(candidate_norms, target_norm)

        elif method == "euclidean":
            # Calculate Euclidean distances and convert to similarity
            distances = np.linalg.norm(candidate_vectors - target_vector, axis=1)
            # Convert distance to similarity (higher is better)
            similarities = 1 / (1 + distances)

        else:
            raise ValueError(f"Unknown similarity method: {method}")

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return jobs with their similarity scores
        results = [
            (candidate_jobs[i], float(similarities[i]))
            for i in top_indices
        ]

        return results

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on weights.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        total_weight = sum(self.weights.values())
        return {
            name: weight / total_weight
            for name, weight in self.weights.items()
        }
