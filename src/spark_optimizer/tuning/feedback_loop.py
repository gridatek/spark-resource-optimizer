"""Feedback loop for continuous improvement of tuning recommendations."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class TuningFeedback:
    """Feedback on a tuning recommendation."""

    feedback_id: str
    session_id: str
    app_id: str
    config_applied: Dict
    expected_improvement: float  # Expected % improvement
    actual_improvement: float  # Actual % improvement
    metric_name: str
    metric_before: float
    metric_after: float
    success: bool  # Did it improve as expected?
    timestamp: datetime = field(default_factory=datetime.utcnow)
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "session_id": self.session_id,
            "app_id": self.app_id,
            "config_applied": self.config_applied,
            "expected_improvement": self.expected_improvement,
            "actual_improvement": self.actual_improvement,
            "metric_name": self.metric_name,
            "metric_before": self.metric_before,
            "metric_after": self.metric_after,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "notes": self.notes,
        }


@dataclass
class LearningRecord:
    """Record of learned tuning patterns."""

    pattern_id: str
    condition: Dict  # Metric conditions that triggered this pattern
    action: Dict  # Configuration changes recommended
    success_count: int = 0
    failure_count: int = 0
    avg_improvement: float = 0.0
    last_used: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def confidence(self) -> float:
        """Calculate confidence score based on sample size and success rate."""
        total = self.success_count + self.failure_count
        if total < 5:
            return 0.0
        # Simple confidence: success rate * sample factor
        sample_factor = min(1.0, total / 20)
        return self.success_rate * sample_factor

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "pattern_id": self.pattern_id,
            "condition": self.condition,
            "action": self.action,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "confidence": self.confidence,
            "avg_improvement": self.avg_improvement,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "created_at": self.created_at.isoformat(),
        }


class FeedbackLoop:
    """Continuous learning system for tuning recommendations.

    Uses feedback from applied configurations to:
    - Learn which adjustments work for specific conditions
    - Improve future recommendations
    - Build a knowledge base of tuning patterns
    """

    def __init__(self, min_confidence: float = 0.6):
        """Initialize the feedback loop.

        Args:
            min_confidence: Minimum confidence to use a learned pattern
        """
        self._min_confidence = min_confidence
        self._feedback: List[TuningFeedback] = []
        self._patterns: Dict[str, LearningRecord] = {}
        self._feedback_counter = 0
        self._pattern_counter = 0

    def record_feedback(
        self,
        session_id: str,
        app_id: str,
        config_applied: Dict,
        metric_name: str,
        metric_before: float,
        metric_after: float,
        expected_improvement: float = 0.0,
        notes: str = "",
    ) -> TuningFeedback:
        """Record feedback on a tuning recommendation.

        Args:
            session_id: Tuning session ID
            app_id: Application ID
            config_applied: Configuration that was applied
            metric_name: Metric being optimized
            metric_before: Metric value before change
            metric_after: Metric value after change
            expected_improvement: Expected improvement percentage
            notes: Additional notes

        Returns:
            TuningFeedback record
        """
        self._feedback_counter += 1
        feedback_id = (
            f"fb-{self._feedback_counter}-{int(datetime.utcnow().timestamp())}"
        )

        # Calculate actual improvement
        if metric_before != 0:
            # For metrics where lower is better (like duration)
            if metric_name in ["duration", "cost", "gc_time", "latency"]:
                actual_improvement = (
                    (metric_before - metric_after) / metric_before * 100
                )
            else:
                actual_improvement = (
                    (metric_after - metric_before) / metric_before * 100
                )
        else:
            actual_improvement = 0.0

        # Determine success
        success = actual_improvement > 0 or (
            expected_improvement > 0
            and actual_improvement >= expected_improvement * 0.5
        )

        feedback = TuningFeedback(
            feedback_id=feedback_id,
            session_id=session_id,
            app_id=app_id,
            config_applied=config_applied,
            expected_improvement=expected_improvement,
            actual_improvement=actual_improvement,
            metric_name=metric_name,
            metric_before=metric_before,
            metric_after=metric_after,
            success=success,
            notes=notes,
        )

        self._feedback.append(feedback)

        # Update learning patterns
        self._update_patterns(feedback)

        logger.info(
            f"Recorded feedback {feedback_id}: "
            f"{'success' if success else 'failure'} "
            f"({actual_improvement:+.1f}% vs expected {expected_improvement:+.1f}%)"
        )

        return feedback

    def get_feedback(
        self,
        session_id: Optional[str] = None,
        app_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[TuningFeedback]:
        """Get feedback records.

        Args:
            session_id: Filter by session ID
            app_id: Filter by application ID
            since: Only return feedback after this time
            limit: Maximum records to return

        Returns:
            List of feedback records
        """
        feedback = self._feedback.copy()

        if session_id:
            feedback = [f for f in feedback if f.session_id == session_id]

        if app_id:
            feedback = [f for f in feedback if f.app_id == app_id]

        if since:
            feedback = [f for f in feedback if f.timestamp > since]

        return sorted(feedback, key=lambda f: f.timestamp, reverse=True)[:limit]

    def get_recommendation(
        self,
        metrics: Dict[str, float],
        current_config: Dict,
    ) -> Optional[Tuple[Dict, float]]:
        """Get a learned recommendation based on current metrics.

        Args:
            metrics: Current metric values
            current_config: Current configuration

        Returns:
            Tuple of (recommended changes, confidence) or None
        """
        # Find matching patterns
        matching_patterns = []

        for pattern in self._patterns.values():
            if pattern.confidence < self._min_confidence:
                continue

            # Check if pattern conditions match current metrics
            match = True
            for metric, condition in pattern.condition.items():
                if metric not in metrics:
                    match = False
                    break

                value = metrics[metric]
                op = condition.get("operator", "gt")
                threshold = condition.get("threshold", 0)

                if op == "gt" and not value > threshold:
                    match = False
                elif op == "gte" and not value >= threshold:
                    match = False
                elif op == "lt" and not value < threshold:
                    match = False
                elif op == "lte" and not value <= threshold:
                    match = False
                elif op == "eq" and not value == threshold:
                    match = False

                if not match:
                    break

            if match:
                matching_patterns.append(pattern)

        if not matching_patterns:
            return None

        # Select best pattern (highest confidence)
        best_pattern = max(matching_patterns, key=lambda p: p.confidence)

        # Update last used
        best_pattern.last_used = datetime.utcnow()

        return best_pattern.action, best_pattern.confidence

    def get_patterns(
        self,
        min_samples: int = 5,
        min_confidence: Optional[float] = None,
    ) -> List[LearningRecord]:
        """Get learned patterns.

        Args:
            min_samples: Minimum sample size
            min_confidence: Minimum confidence score

        Returns:
            List of learning records
        """
        if min_confidence is None:
            min_confidence = self._min_confidence

        patterns = [
            p
            for p in self._patterns.values()
            if (p.success_count + p.failure_count) >= min_samples
            and p.confidence >= min_confidence
        ]

        return sorted(patterns, key=lambda p: p.confidence, reverse=True)

    def get_statistics(self) -> Dict:
        """Get feedback loop statistics.

        Returns:
            Statistics dictionary
        """
        if not self._feedback:
            return {
                "total_feedback": 0,
                "success_rate": 0.0,
                "avg_improvement": 0.0,
                "patterns_learned": 0,
                "high_confidence_patterns": 0,
            }

        successes = sum(1 for f in self._feedback if f.success)
        improvements = [f.actual_improvement for f in self._feedback]

        high_confidence = sum(
            1 for p in self._patterns.values() if p.confidence >= self._min_confidence
        )

        return {
            "total_feedback": len(self._feedback),
            "success_rate": successes / len(self._feedback),
            "avg_improvement": sum(improvements) / len(improvements),
            "patterns_learned": len(self._patterns),
            "high_confidence_patterns": high_confidence,
        }

    def _update_patterns(self, feedback: TuningFeedback) -> None:
        """Update learning patterns based on feedback.

        Args:
            feedback: New feedback record
        """
        # Create pattern ID from config changes
        config_key = self._config_to_key(feedback.config_applied)
        pattern_id = f"pattern-{config_key}"

        if pattern_id not in self._patterns:
            # Create new pattern
            condition = self._infer_condition(feedback)
            self._patterns[pattern_id] = LearningRecord(
                pattern_id=pattern_id,
                condition=condition,
                action=feedback.config_applied,
            )

        pattern = self._patterns[pattern_id]

        # Update counts
        if feedback.success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1

        # Update average improvement
        total = pattern.success_count + pattern.failure_count
        pattern.avg_improvement = (
            pattern.avg_improvement * (total - 1) + feedback.actual_improvement
        ) / total

        pattern.last_used = datetime.utcnow()

    def _config_to_key(self, config: Dict) -> str:
        """Convert config dict to a stable key.

        Args:
            config: Configuration dictionary

        Returns:
            String key
        """
        # Sort keys and create deterministic string
        items = sorted(config.items())
        return "-".join(f"{k}_{v}" for k, v in items)

    def _infer_condition(self, feedback: TuningFeedback) -> Dict:
        """Infer conditions from feedback.

        Args:
            feedback: Feedback record

        Returns:
            Condition dictionary
        """
        # Simple inference based on the metric being optimized
        # In a more sophisticated system, this would analyze metric patterns

        conditions = {}

        metric = feedback.metric_name
        before = feedback.metric_before

        if metric in ["duration", "cost", "gc_time", "latency"]:
            # These are "lower is better" metrics
            # The condition is when the metric is high
            conditions[metric] = {
                "operator": "gt",
                "threshold": before * 0.8,  # 80% of the "before" value
            }
        else:
            # "Higher is better" metrics like throughput
            conditions[metric] = {
                "operator": "lt",
                "threshold": before * 1.2,  # 120% of the "before" value
            }

        return conditions

    def export_patterns(self) -> str:
        """Export learned patterns as JSON.

        Returns:
            JSON string of patterns
        """
        patterns_data = [p.to_dict() for p in self._patterns.values()]
        return json.dumps(patterns_data, indent=2)

    def import_patterns(self, patterns_json: str) -> int:
        """Import patterns from JSON.

        Args:
            patterns_json: JSON string of patterns

        Returns:
            Number of patterns imported
        """
        patterns_data = json.loads(patterns_json)
        imported = 0

        for data in patterns_data:
            pattern = LearningRecord(
                pattern_id=data["pattern_id"],
                condition=data["condition"],
                action=data["action"],
                success_count=data.get("success_count", 0),
                failure_count=data.get("failure_count", 0),
                avg_improvement=data.get("avg_improvement", 0.0),
            )

            if data.get("last_used"):
                pattern.last_used = datetime.fromisoformat(data["last_used"])
            if data.get("created_at"):
                pattern.created_at = datetime.fromisoformat(data["created_at"])

            self._patterns[pattern.pattern_id] = pattern
            imported += 1

        logger.info(f"Imported {imported} patterns")
        return imported
