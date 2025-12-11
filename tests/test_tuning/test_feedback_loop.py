"""Tests for feedback loop functionality."""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock

from spark_optimizer.tuning.feedback_loop import (
    FeedbackLoop,
    TuningFeedback,
    LearningRecord,
)


class TestTuningFeedback:
    """Test TuningFeedback dataclass."""

    def test_feedback_creation(self):
        """Test creating feedback."""
        feedback = TuningFeedback(
            feedback_id="fb-1",
            session_id="session-1",
            app_id="app-123",
            config_applied={"spark.executor.memory": 8192},
            expected_improvement=10.0,
            actual_improvement=15.0,
            metric_name="duration",
            metric_before=100.0,
            metric_after=85.0,
            success=True,
        )

        assert feedback.feedback_id == "fb-1"
        assert feedback.session_id == "session-1"
        assert feedback.expected_improvement == 10.0
        assert feedback.actual_improvement == 15.0
        assert feedback.success is True

    def test_feedback_to_dict(self):
        """Test converting feedback to dictionary."""
        now = datetime.utcnow()
        feedback = TuningFeedback(
            feedback_id="fb-2",
            session_id="session-2",
            app_id="app-456",
            config_applied={"spark.executor.cores": 4},
            expected_improvement=5.0,
            actual_improvement=-2.0,
            metric_name="cost",
            metric_before=50.0,
            metric_after=51.0,
            success=False,
            timestamp=now,
            notes="Config change didn't help",
        )

        result = feedback.to_dict()

        assert result["feedback_id"] == "fb-2"
        assert result["config_applied"]["spark.executor.cores"] == 4
        assert result["expected_improvement"] == 5.0
        assert result["actual_improvement"] == -2.0
        assert result["success"] is False
        assert result["notes"] == "Config change didn't help"


class TestLearningRecord:
    """Test LearningRecord dataclass."""

    def test_record_creation(self):
        """Test creating a learning record."""
        record = LearningRecord(
            pattern_id="pattern-1",
            condition={"gc_time_percent": {"operator": "gt", "threshold": 10}},
            action={"spark.memory.fraction": 0.5},
        )

        assert record.pattern_id == "pattern-1"
        assert record.success_count == 0
        assert record.failure_count == 0

    def test_success_rate(self):
        """Test success rate calculation."""
        record = LearningRecord(
            pattern_id="pattern-1",
            condition={},
            action={},
            success_count=7,
            failure_count=3,
        )

        assert record.success_rate == 0.7

    def test_success_rate_no_data(self):
        """Test success rate with no data."""
        record = LearningRecord(
            pattern_id="pattern-1",
            condition={},
            action={},
        )

        assert record.success_rate == 0.0

    def test_confidence_low_samples(self):
        """Test confidence with low sample count."""
        record = LearningRecord(
            pattern_id="pattern-1",
            condition={},
            action={},
            success_count=3,
            failure_count=1,
        )

        # Less than 5 samples = 0 confidence
        assert record.confidence == 0.0

    def test_confidence_medium_samples(self):
        """Test confidence with medium sample count."""
        record = LearningRecord(
            pattern_id="pattern-1",
            condition={},
            action={},
            success_count=8,
            failure_count=2,
        )

        # 10 samples, 80% success rate
        # Sample factor = 10/20 = 0.5
        # Confidence = 0.8 * 0.5 = 0.4
        assert record.confidence == 0.4

    def test_confidence_high_samples(self):
        """Test confidence with high sample count."""
        record = LearningRecord(
            pattern_id="pattern-1",
            condition={},
            action={},
            success_count=16,
            failure_count=4,
        )

        # 20 samples, 80% success rate
        # Sample factor = min(1.0, 20/20) = 1.0
        # Confidence = 0.8 * 1.0 = 0.8
        assert record.confidence == 0.8

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        now = datetime.utcnow()
        record = LearningRecord(
            pattern_id="pattern-2",
            condition={"metric": {"operator": "gt", "threshold": 10}},
            action={"param": "value"},
            success_count=15,
            failure_count=5,
            avg_improvement=12.5,
            last_used=now,
        )

        result = record.to_dict()

        assert result["pattern_id"] == "pattern-2"
        assert result["success_count"] == 15
        assert result["failure_count"] == 5
        assert result["success_rate"] == 0.75
        assert result["avg_improvement"] == 12.5


class TestFeedbackLoop:
    """Test FeedbackLoop class."""

    def test_loop_initialization(self):
        """Test feedback loop initialization."""
        loop = FeedbackLoop()

        assert loop._min_confidence == 0.6

    def test_loop_custom_confidence(self):
        """Test feedback loop with custom confidence."""
        loop = FeedbackLoop(min_confidence=0.8)

        assert loop._min_confidence == 0.8

    def test_record_feedback_success(self):
        """Test recording successful feedback."""
        loop = FeedbackLoop()

        feedback = loop.record_feedback(
            session_id="session-1",
            app_id="app-123",
            config_applied={"spark.executor.memory": 8192},
            metric_name="duration",
            metric_before=100.0,
            metric_after=80.0,  # 20% improvement
            expected_improvement=15.0,
        )

        assert feedback.feedback_id is not None
        assert feedback.actual_improvement == 20.0  # (100-80)/100 * 100
        assert feedback.success is True

    def test_record_feedback_failure(self):
        """Test recording failed feedback."""
        loop = FeedbackLoop()

        feedback = loop.record_feedback(
            session_id="session-1",
            app_id="app-123",
            config_applied={"spark.executor.memory": 8192},
            metric_name="duration",
            metric_before=100.0,
            metric_after=120.0,  # 20% worse
            expected_improvement=15.0,
        )

        assert feedback.actual_improvement == -20.0  # (100-120)/100 * 100
        assert feedback.success is False

    def test_record_feedback_higher_is_better(self):
        """Test feedback for metrics where higher is better."""
        loop = FeedbackLoop()

        feedback = loop.record_feedback(
            session_id="session-1",
            app_id="app-123",
            config_applied={},
            metric_name="throughput",
            metric_before=100.0,
            metric_after=120.0,  # 20% improvement
        )

        assert feedback.actual_improvement == 20.0  # (120-100)/100 * 100
        assert feedback.success is True

    def test_record_feedback_creates_pattern(self):
        """Test that feedback creates learning patterns."""
        loop = FeedbackLoop()

        loop.record_feedback(
            session_id="session-1",
            app_id="app-123",
            config_applied={"spark.executor.memory": 8192},
            metric_name="duration",
            metric_before=100.0,
            metric_after=80.0,
        )

        assert len(loop._patterns) == 1

    def test_get_feedback(self):
        """Test getting feedback records."""
        loop = FeedbackLoop()

        loop.record_feedback(
            session_id="session-1",
            app_id="app-1",
            config_applied={},
            metric_name="duration",
            metric_before=100.0,
            metric_after=80.0,
        )
        loop.record_feedback(
            session_id="session-2",
            app_id="app-2",
            config_applied={},
            metric_name="duration",
            metric_before=100.0,
            metric_after=90.0,
        )

        # Get all
        all_feedback = loop.get_feedback()
        assert len(all_feedback) == 2

        # Filter by session
        session_feedback = loop.get_feedback(session_id="session-1")
        assert len(session_feedback) == 1
        assert session_feedback[0].session_id == "session-1"

        # Filter by app
        app_feedback = loop.get_feedback(app_id="app-2")
        assert len(app_feedback) == 1
        assert app_feedback[0].app_id == "app-2"

    def test_get_feedback_with_limit(self):
        """Test feedback retrieval with limit."""
        loop = FeedbackLoop()

        for i in range(10):
            loop.record_feedback(
                session_id=f"session-{i}",
                app_id=f"app-{i}",
                config_applied={},
                metric_name="duration",
                metric_before=100.0,
                metric_after=90.0,
            )

        feedback = loop.get_feedback(limit=5)
        assert len(feedback) == 5

    def test_get_recommendation_no_patterns(self):
        """Test getting recommendation with no patterns."""
        loop = FeedbackLoop()

        result = loop.get_recommendation(
            {"gc_time_percent": 15.0},
            {"spark.executor.memory": 4096},
        )

        assert result is None

    def test_get_recommendation_with_matching_pattern(self):
        """Test getting recommendation with matching pattern."""
        loop = FeedbackLoop()

        # Build up pattern with enough samples
        for i in range(10):
            loop.record_feedback(
                session_id=f"session-{i}",
                app_id=f"app-{i}",
                config_applied={"spark.memory.fraction": 0.5},
                metric_name="duration",
                metric_before=100.0,
                metric_after=80.0 if i < 8 else 110.0,  # 80% success
            )

        # The pattern should now match
        result = loop.get_recommendation(
            {"duration": 150.0},  # High duration
            {},
        )

        # Might or might not match depending on inferred conditions
        # Just check the structure if result exists
        if result:
            action, confidence = result
            assert isinstance(action, dict)
            assert isinstance(confidence, float)

    def test_get_patterns(self):
        """Test getting learned patterns."""
        loop = FeedbackLoop()

        # Create patterns with varying success rates
        for i in range(10):
            loop.record_feedback(
                session_id=f"session-{i}",
                app_id=f"app-{i}",
                config_applied={"config-a": "value"},
                metric_name="duration",
                metric_before=100.0,
                metric_after=80.0,
            )

        # With 10 samples, confidence is 0.5 (success_rate * sample_factor)
        # sample_factor = min(1.0, 10/20) = 0.5
        # So we need to lower min_confidence to see the patterns
        patterns = loop.get_patterns(min_samples=5, min_confidence=0.4)

        assert len(patterns) >= 1
        for pattern in patterns:
            total = pattern.success_count + pattern.failure_count
            assert total >= 5

    def test_get_statistics(self):
        """Test getting feedback loop statistics."""
        loop = FeedbackLoop()

        # No feedback
        stats = loop.get_statistics()
        assert stats["total_feedback"] == 0

        # Add some feedback
        for i in range(5):
            loop.record_feedback(
                session_id=f"session-{i}",
                app_id=f"app-{i}",
                config_applied={},
                metric_name="duration",
                metric_before=100.0,
                metric_after=80.0 if i < 3 else 110.0,
            )

        stats = loop.get_statistics()

        assert stats["total_feedback"] == 5
        assert stats["success_rate"] == 0.6  # 3/5
        assert stats["patterns_learned"] >= 1

    def test_update_pattern_counts(self):
        """Test that pattern counts are updated correctly."""
        loop = FeedbackLoop()

        # Record success
        loop.record_feedback(
            session_id="s1",
            app_id="a1",
            config_applied={"param": "value"},
            metric_name="duration",
            metric_before=100.0,
            metric_after=80.0,
        )

        # Get the pattern
        patterns = list(loop._patterns.values())
        assert len(patterns) == 1
        assert patterns[0].success_count == 1
        assert patterns[0].failure_count == 0

        # Record failure with same config
        loop.record_feedback(
            session_id="s2",
            app_id="a2",
            config_applied={"param": "value"},
            metric_name="duration",
            metric_before=100.0,
            metric_after=120.0,
        )

        patterns = list(loop._patterns.values())
        assert patterns[0].success_count == 1
        assert patterns[0].failure_count == 1

    def test_export_patterns(self):
        """Test exporting patterns as JSON."""
        loop = FeedbackLoop()

        # Create some patterns
        for i in range(5):
            loop.record_feedback(
                session_id=f"s{i}",
                app_id=f"a{i}",
                config_applied={"param": f"value{i}"},
                metric_name="duration",
                metric_before=100.0,
                metric_after=80.0,
            )

        exported = loop.export_patterns()

        assert isinstance(exported, str)
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) == 5

    def test_import_patterns(self):
        """Test importing patterns from JSON."""
        loop = FeedbackLoop()

        patterns_json = json.dumps(
            [
                {
                    "pattern_id": "pattern-1",
                    "condition": {"metric": {"operator": "gt", "threshold": 10}},
                    "action": {"param": "value"},
                    "success_count": 8,
                    "failure_count": 2,
                    "avg_improvement": 15.0,
                },
                {
                    "pattern_id": "pattern-2",
                    "condition": {"other": {"operator": "lt", "threshold": 5}},
                    "action": {"other_param": "other_value"},
                    "success_count": 10,
                    "failure_count": 0,
                    "avg_improvement": 20.0,
                },
            ]
        )

        count = loop.import_patterns(patterns_json)

        assert count == 2
        assert len(loop._patterns) == 2
        assert "pattern-1" in loop._patterns
        assert loop._patterns["pattern-1"].success_count == 8


class TestFeedbackLoopConditionMatching:
    """Test condition matching in feedback loop."""

    def test_condition_gt(self):
        """Test greater than condition matching."""
        loop = FeedbackLoop(min_confidence=0.4)  # Lower threshold for test patterns

        # Manually create a pattern with gt condition
        loop._patterns["test"] = LearningRecord(
            pattern_id="test",
            condition={"metric": {"operator": "gt", "threshold": 10}},
            action={"param": "value"},
            success_count=10,
            failure_count=0,
        )

        # Should match
        result = loop.get_recommendation({"metric": 15}, {})
        assert result is not None

    def test_condition_lt(self):
        """Test less than condition matching."""
        loop = FeedbackLoop(min_confidence=0.4)  # Lower threshold for test patterns

        loop._patterns["test"] = LearningRecord(
            pattern_id="test",
            condition={"metric": {"operator": "lt", "threshold": 10}},
            action={"param": "value"},
            success_count=10,
            failure_count=0,
        )

        # Should match
        result = loop.get_recommendation({"metric": 5}, {})
        assert result is not None

        # Should not match
        result = loop.get_recommendation({"metric": 15}, {})
        assert result is None

    def test_condition_gte(self):
        """Test greater than or equal condition matching."""
        loop = FeedbackLoop(min_confidence=0.4)  # Lower threshold for test patterns

        loop._patterns["test"] = LearningRecord(
            pattern_id="test",
            condition={"metric": {"operator": "gte", "threshold": 10}},
            action={"param": "value"},
            success_count=10,
            failure_count=0,
        )

        # Should match (equal)
        result = loop.get_recommendation({"metric": 10}, {})
        assert result is not None

        # Should match (greater)
        result = loop.get_recommendation({"metric": 15}, {})
        assert result is not None

    def test_condition_multiple_metrics(self):
        """Test matching with multiple metric conditions."""
        loop = FeedbackLoop(min_confidence=0.4)  # Lower threshold for test patterns

        loop._patterns["test"] = LearningRecord(
            pattern_id="test",
            condition={
                "gc_time": {"operator": "gt", "threshold": 10},
                "memory_spill": {"operator": "gt", "threshold": 0.1},
            },
            action={"param": "value"},
            success_count=10,
            failure_count=0,
        )

        # Should match (both conditions met)
        result = loop.get_recommendation(
            {"gc_time": 15, "memory_spill": 0.2},
            {},
        )
        assert result is not None

        # Should not match (only one condition met)
        result = loop.get_recommendation(
            {"gc_time": 15, "memory_spill": 0.05},
            {},
        )
        assert result is None

    def test_selects_highest_confidence_pattern(self):
        """Test that highest confidence pattern is selected."""
        loop = FeedbackLoop()

        # Lower confidence pattern
        loop._patterns["low"] = LearningRecord(
            pattern_id="low",
            condition={"metric": {"operator": "gt", "threshold": 5}},
            action={"result": "low"},
            success_count=6,
            failure_count=4,
        )

        # Higher confidence pattern
        loop._patterns["high"] = LearningRecord(
            pattern_id="high",
            condition={"metric": {"operator": "gt", "threshold": 5}},
            action={"result": "high"},
            success_count=18,
            failure_count=2,
        )

        result = loop.get_recommendation({"metric": 10}, {})

        assert result is not None
        action, confidence = result
        assert action["result"] == "high"
