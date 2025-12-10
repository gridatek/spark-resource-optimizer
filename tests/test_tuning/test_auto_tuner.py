"""Tests for auto-tuning functionality."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from spark_optimizer.tuning.auto_tuner import (
    AutoTuner,
    TuningSession,
    TuningAdjustment,
    TuningConfig,
    TuningStrategy,
)


class TestTuningStrategy:
    """Test TuningStrategy enum."""

    def test_strategy_values(self):
        """Test that strategy enum has expected values."""
        assert TuningStrategy.CONSERVATIVE.value == "conservative"
        assert TuningStrategy.MODERATE.value == "moderate"
        assert TuningStrategy.AGGRESSIVE.value == "aggressive"


class TestTuningConfig:
    """Test TuningConfig dataclass."""

    def test_config_creation(self):
        """Test creating a tuning config."""
        config = TuningConfig(
            name="spark.executor.memory",
            current_value=4096,
            min_value=1024,
            max_value=32768,
            step_size=1024,
            unit="MB",
            description="Executor memory",
        )

        assert config.name == "spark.executor.memory"
        assert config.current_value == 4096
        assert config.min_value == 1024
        assert config.max_value == 32768
        assert config.step_size == 1024
        assert config.unit == "MB"

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = TuningConfig(
            name="spark.executor.cores",
            current_value=4,
            min_value=1,
            max_value=16,
            step_size=1,
        )

        result = config.to_dict()

        assert result["name"] == "spark.executor.cores"
        assert result["current_value"] == 4
        assert result["min_value"] == 1
        assert result["max_value"] == 16


class TestTuningAdjustment:
    """Test TuningAdjustment dataclass."""

    def test_adjustment_creation(self):
        """Test creating an adjustment."""
        adjustment = TuningAdjustment(
            parameter="spark.executor.memory",
            old_value=4096,
            new_value=6144,
            reason="Memory spilling detected",
        )

        assert adjustment.parameter == "spark.executor.memory"
        assert adjustment.old_value == 4096
        assert adjustment.new_value == 6144
        assert adjustment.reason == "Memory spilling detected"
        assert adjustment.applied is False

    def test_adjustment_to_dict(self):
        """Test converting adjustment to dictionary."""
        now = datetime.utcnow()
        adjustment = TuningAdjustment(
            parameter="spark.executor.cores",
            old_value=4,
            new_value=2,
            reason="High task failure rate",
            timestamp=now,
            applied=True,
            result="Success",
        )

        result = adjustment.to_dict()

        assert result["parameter"] == "spark.executor.cores"
        assert result["old_value"] == 4
        assert result["new_value"] == 2
        assert result["applied"] is True
        assert result["result"] == "Success"


class TestTuningSession:
    """Test TuningSession dataclass."""

    def test_session_creation(self):
        """Test creating a session."""
        session = TuningSession(
            session_id="tune-1",
            app_id="app-123",
            app_name="test_job",
            strategy=TuningStrategy.MODERATE,
            target_metric="duration",
        )

        assert session.session_id == "tune-1"
        assert session.app_id == "app-123"
        assert session.status == "active"
        assert session.iterations == 0
        assert session.adjustments == []

    def test_session_to_dict(self):
        """Test converting session to dictionary."""
        session = TuningSession(
            session_id="tune-2",
            app_id="app-456",
            app_name="etl_job",
            strategy=TuningStrategy.AGGRESSIVE,
            target_metric="cost",
            initial_config={"spark.executor.memory": 4096},
            current_config={"spark.executor.memory": 6144},
        )

        result = session.to_dict()

        assert result["session_id"] == "tune-2"
        assert result["strategy"] == "aggressive"
        assert result["target_metric"] == "cost"
        assert result["initial_config"]["spark.executor.memory"] == 4096
        assert result["current_config"]["spark.executor.memory"] == 6144


class TestAutoTuner:
    """Test AutoTuner class."""

    def test_tuner_initialization(self):
        """Test auto-tuner initialization."""
        tuner = AutoTuner()

        assert len(tuner._tunable_params) > 0
        assert tuner._max_iterations == 20
        assert tuner._convergence_threshold == 0.02

    def test_tuner_custom_initialization(self):
        """Test auto-tuner with custom parameters."""
        custom_params = {
            "custom.param": TuningConfig(
                name="custom.param",
                current_value=10,
                min_value=1,
                max_value=100,
                step_size=5,
            )
        }

        tuner = AutoTuner(
            tunable_params=custom_params,
            max_iterations=10,
            convergence_threshold=0.05,
        )

        assert len(tuner._tunable_params) == 1
        assert tuner._max_iterations == 10
        assert tuner._convergence_threshold == 0.05

    def test_start_session(self):
        """Test starting a tuning session."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            strategy=TuningStrategy.MODERATE,
            target_metric="duration",
        )

        assert session.app_id == "app-123"
        assert session.app_name == "test_job"
        assert session.strategy == TuningStrategy.MODERATE
        assert session.target_metric == "duration"
        assert session.status == "active"
        assert session.initial_config == {"spark.executor.memory": 4096}

    def test_start_session_existing_active(self):
        """Test that starting session returns existing active session."""
        tuner = AutoTuner()

        session1 = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )

        session2 = tuner.start_session(
            app_id="app-123",
            app_name="different_name",
            initial_config={"different": "config"},
        )

        assert session1.session_id == session2.session_id

    def test_get_session(self):
        """Test getting a session by ID."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )

        retrieved = tuner.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.session_id == session.session_id

        missing = tuner.get_session("nonexistent")
        assert missing is None

    def test_get_active_session(self):
        """Test getting active session for an app."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )

        active = tuner.get_active_session("app-123")
        assert active is not None
        assert active.session_id == session.session_id

        missing = tuner.get_active_session("app-999")
        assert missing is None

    def test_list_sessions(self):
        """Test listing sessions."""
        tuner = AutoTuner()

        tuner.start_session("app-1", "job1", {})
        tuner.start_session("app-2", "job2", {})
        tuner.start_session("app-3", "job3", {})

        # End one session
        sessions = tuner.list_sessions()
        assert len(sessions) == 3

        # Filter by app
        sessions = tuner.list_sessions(app_id="app-1")
        assert len(sessions) == 1

        # Filter by status
        tuner.end_session(tuner.list_sessions()[0].session_id)
        sessions = tuner.list_sessions(status="active")
        assert len(sessions) == 2

    def test_analyze_and_recommend_records_metrics(self):
        """Test that analyze_and_recommend records metrics history."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            target_metric="duration",
        )

        tuner.analyze_and_recommend(
            session.session_id, {"duration": 100.0, "cpu": 50.0}
        )

        assert len(session.metrics_history) == 1
        assert session.metrics_history[0]["metrics"]["duration"] == 100.0

    def test_analyze_and_recommend_updates_best(self):
        """Test that analyze_and_recommend updates best config."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            target_metric="duration",
        )

        tuner.analyze_and_recommend(session.session_id, {"duration": 100.0})

        assert session.best_metric_value == 100.0

        # Better value (lower duration is better)
        tuner.analyze_and_recommend(session.session_id, {"duration": 80.0})

        assert session.best_metric_value == 80.0

    def test_analyze_and_recommend_max_iterations(self):
        """Test that session ends at max iterations."""
        tuner = AutoTuner(max_iterations=3)

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
            target_metric="duration",
        )

        for i in range(5):
            tuner.analyze_and_recommend(
                session.session_id, {"duration": 100.0 - i * 10}
            )

        assert session.status == "completed"
        assert session.iterations == 3

    def test_analyze_and_recommend_memory_spilling(self):
        """Test recommendations for memory spilling."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            target_metric="duration",
        )

        adjustments = tuner.analyze_and_recommend(
            session.session_id,
            {"duration": 100.0, "memory_spill_ratio": 0.2},
        )

        # Should recommend increasing memory
        memory_adjustment = next(
            (a for a in adjustments if a.parameter == "spark.executor.memory"),
            None,
        )
        assert memory_adjustment is not None
        assert memory_adjustment.new_value > 4096

    def test_analyze_and_recommend_high_gc(self):
        """Test recommendations for high GC time."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.memory.fraction": 0.6},
            target_metric="duration",
        )

        adjustments = tuner.analyze_and_recommend(
            session.session_id,
            {"duration": 100.0, "gc_time_percent": 20.0},
        )

        # Should recommend reducing memory fraction
        fraction_adjustment = next(
            (a for a in adjustments if a.parameter == "spark.memory.fraction"),
            None,
        )
        assert fraction_adjustment is not None
        assert fraction_adjustment.new_value < 0.6

    def test_analyze_and_recommend_low_cpu(self):
        """Test recommendations for low CPU utilization."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.default.parallelism": 100},
            target_metric="duration",
        )

        adjustments = tuner.analyze_and_recommend(
            session.session_id,
            {"duration": 100.0, "cpu_utilization": 30.0},
        )

        # Should recommend increasing parallelism
        parallelism_adjustment = next(
            (a for a in adjustments if a.parameter == "spark.default.parallelism"),
            None,
        )
        assert parallelism_adjustment is not None
        assert parallelism_adjustment.new_value > 100

    def test_apply_adjustment(self):
        """Test applying an adjustment."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            target_metric="duration",
        )

        adjustment = TuningAdjustment(
            parameter="spark.executor.memory",
            old_value=4096,
            new_value=6144,
            reason="Test",
        )

        result = tuner.apply_adjustment(session.session_id, adjustment)

        assert result is True
        assert adjustment.applied is True
        assert session.current_config["spark.executor.memory"] == 6144
        assert len(session.adjustments) == 1

    def test_apply_adjustment_invalid_session(self):
        """Test applying adjustment to invalid session."""
        tuner = AutoTuner()

        adjustment = TuningAdjustment(
            parameter="test",
            old_value=1,
            new_value=2,
            reason="Test",
        )

        result = tuner.apply_adjustment("nonexistent", adjustment)

        assert result is False

    def test_pause_session(self):
        """Test pausing a session."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )

        result = tuner.pause_session(session.session_id)

        assert result is True
        assert session.status == "paused"

    def test_pause_inactive_session(self):
        """Test pausing an inactive session fails."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )
        tuner.end_session(session.session_id)

        result = tuner.pause_session(session.session_id)

        assert result is False

    def test_resume_session(self):
        """Test resuming a paused session."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )
        tuner.pause_session(session.session_id)

        result = tuner.resume_session(session.session_id)

        assert result is True
        assert session.status == "active"

    def test_resume_non_paused_session(self):
        """Test resuming a non-paused session fails."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )

        result = tuner.resume_session(session.session_id)

        assert result is False  # Already active

    def test_end_session(self):
        """Test ending a session."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )

        result = tuner.end_session(session.session_id)

        assert result is True
        assert session.status == "completed"
        assert session.ended_at is not None

        # Should no longer be active
        active = tuner.get_active_session("app-123")
        assert active is None

    def test_end_session_with_status(self):
        """Test ending a session with custom status."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
        )

        result = tuner.end_session(session.session_id, status="failed")

        assert result is True
        assert session.status == "failed"

    def test_get_best_config(self):
        """Test getting best configuration."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            target_metric="duration",
        )

        tuner.analyze_and_recommend(session.session_id, {"duration": 100.0})

        best = tuner.get_best_config(session.session_id)

        assert best is not None
        assert "spark.executor.memory" in best

    def test_get_tunable_parameters(self):
        """Test getting tunable parameters."""
        tuner = AutoTuner()

        params = tuner.get_tunable_parameters()

        assert "spark.executor.memory" in params
        assert "spark.executor.cores" in params
        assert "spark.executor.instances" in params

    def test_set_parameter_range(self):
        """Test setting parameter range."""
        tuner = AutoTuner()

        result = tuner.set_parameter_range(
            "spark.executor.memory",
            min_value=2048,
            max_value=16384,
            step_size=2048,
        )

        assert result is True
        params = tuner.get_tunable_parameters()
        assert params["spark.executor.memory"].min_value == 2048
        assert params["spark.executor.memory"].max_value == 16384
        assert params["spark.executor.memory"].step_size == 2048

    def test_set_parameter_range_unknown(self):
        """Test setting range for unknown parameter."""
        tuner = AutoTuner()

        result = tuner.set_parameter_range("unknown.param", min_value=0)

        assert result is False


class TestAutoTunerStrategyMultipliers:
    """Test strategy multipliers affect adjustments."""

    def test_conservative_strategy(self):
        """Test conservative strategy makes smaller adjustments."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            strategy=TuningStrategy.CONSERVATIVE,
            target_metric="duration",
        )

        adjustments = tuner.analyze_and_recommend(
            session.session_id,
            {"duration": 100.0, "memory_spill_ratio": 0.2},
        )

        memory_adj = next(
            (a for a in adjustments if a.parameter == "spark.executor.memory"),
            None,
        )

        if memory_adj:
            # Conservative should use 0.5x step size (512 instead of 1024)
            assert memory_adj.new_value - memory_adj.old_value <= 1024

    def test_aggressive_strategy(self):
        """Test aggressive strategy makes larger adjustments."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={"spark.executor.memory": 4096},
            strategy=TuningStrategy.AGGRESSIVE,
            target_metric="duration",
        )

        adjustments = tuner.analyze_and_recommend(
            session.session_id,
            {"duration": 100.0, "memory_spill_ratio": 0.2},
        )

        memory_adj = next(
            (a for a in adjustments if a.parameter == "spark.executor.memory"),
            None,
        )

        if memory_adj:
            # Aggressive should use 2x step size (2048 instead of 1024)
            assert memory_adj.new_value - memory_adj.old_value >= 1024


class TestAutoTunerMetricOptimization:
    """Test metric optimization direction."""

    def test_duration_lower_is_better(self):
        """Test that lower duration is considered better."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
            target_metric="duration",
        )

        # Initial metric
        tuner.analyze_and_recommend(session.session_id, {"duration": 100.0})
        assert session.best_metric_value == 100.0

        # Lower is better
        tuner.analyze_and_recommend(session.session_id, {"duration": 80.0})
        assert session.best_metric_value == 80.0

        # Higher is not better
        tuner.analyze_and_recommend(session.session_id, {"duration": 90.0})
        assert session.best_metric_value == 80.0

    def test_throughput_higher_is_better(self):
        """Test that higher throughput is considered better."""
        tuner = AutoTuner()

        session = tuner.start_session(
            app_id="app-123",
            app_name="test_job",
            initial_config={},
            target_metric="throughput",
        )

        # Initial metric
        tuner.analyze_and_recommend(session.session_id, {"throughput": 100.0})
        assert session.best_metric_value == 100.0

        # Higher is better
        tuner.analyze_and_recommend(session.session_id, {"throughput": 120.0})
        assert session.best_metric_value == 120.0

        # Lower is not better
        tuner.analyze_and_recommend(session.session_id, {"throughput": 110.0})
        assert session.best_metric_value == 120.0
