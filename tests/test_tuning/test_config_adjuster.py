"""Tests for configuration adjustment functionality."""

import pytest
from datetime import datetime
from unittest.mock import Mock

from spark_optimizer.tuning.config_adjuster import (
    ConfigAdjuster,
    ConfigChange,
    AdjustmentAction,
)


class TestAdjustmentAction:
    """Test AdjustmentAction enum."""

    def test_action_values(self):
        """Test that action enum has expected values."""
        assert AdjustmentAction.INCREASE.value == "increase"
        assert AdjustmentAction.DECREASE.value == "decrease"
        assert AdjustmentAction.SET.value == "set"
        assert AdjustmentAction.RESET.value == "reset"


class TestConfigChange:
    """Test ConfigChange dataclass."""

    def test_config_change_creation(self):
        """Test creating a config change."""
        change = ConfigChange(
            parameter="spark.executor.memory",
            action=AdjustmentAction.SET,
            old_value=4096,
            new_value=8192,
            source="auto",
            reason="Memory pressure detected",
        )

        assert change.parameter == "spark.executor.memory"
        assert change.action == AdjustmentAction.SET
        assert change.old_value == 4096
        assert change.new_value == 8192
        assert change.source == "auto"

    def test_config_change_to_dict(self):
        """Test converting config change to dictionary."""
        now = datetime.utcnow()
        change = ConfigChange(
            parameter="spark.executor.cores",
            action=AdjustmentAction.DECREASE,
            old_value=8,
            new_value=4,
            timestamp=now,
            source="manual",
            reason="Cost optimization",
        )

        result = change.to_dict()

        assert result["parameter"] == "spark.executor.cores"
        assert result["action"] == "decrease"
        assert result["old_value"] == 8
        assert result["new_value"] == 4
        assert result["timestamp"] == now.isoformat()


class TestConfigAdjuster:
    """Test ConfigAdjuster class."""

    def test_adjuster_initialization(self):
        """Test adjuster initialization."""
        adjuster = ConfigAdjuster()

        assert adjuster.config == {}
        assert adjuster.history == []

    def test_adjuster_with_initial_config(self):
        """Test adjuster with initial configuration."""
        initial = {
            "spark.executor.memory": 4096,
            "spark.executor.cores": 4,
        }

        adjuster = ConfigAdjuster(initial_config=initial)

        assert adjuster.config == initial
        # Should not modify original
        initial["new_key"] = "value"
        assert "new_key" not in adjuster.config

    def test_get_value(self):
        """Test getting a configuration value."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.memory": 4096,
            }
        )

        assert adjuster.get("spark.executor.memory") == 4096
        assert adjuster.get("missing.key") is None
        assert adjuster.get("missing.key", 1024) == 1024

    def test_set_value(self):
        """Test setting a configuration value."""
        adjuster = ConfigAdjuster()

        change = adjuster.set(
            "spark.executor.memory",
            4096,
            source="manual",
            reason="Initial setup",
        )

        assert adjuster.get("spark.executor.memory") == 4096
        assert change.parameter == "spark.executor.memory"
        assert change.action == AdjustmentAction.SET
        assert change.new_value == 4096
        assert len(adjuster.history) == 1

    def test_set_with_validation(self):
        """Test setting with validation."""
        adjuster = ConfigAdjuster()

        # Valid value
        adjuster.set("spark.executor.memory", 4096)
        assert adjuster.get("spark.executor.memory") == 4096

        # Invalid value (below min)
        with pytest.raises(ValueError):
            adjuster.set("spark.executor.memory", 100)

    def test_set_without_validation(self):
        """Test setting without validation."""
        adjuster = ConfigAdjuster()

        # Would fail validation but we disable it
        change = adjuster.set(
            "spark.executor.memory",
            100,
            validate=False,
        )

        assert adjuster.get("spark.executor.memory") == 100

    def test_adjust_increase(self):
        """Test increasing a value."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.memory": 4096,
            }
        )

        change = adjuster.adjust(
            "spark.executor.memory",
            AdjustmentAction.INCREASE,
            amount=1024,
        )

        assert adjuster.get("spark.executor.memory") == 5120
        assert change.old_value == 4096
        assert change.new_value == 5120

    def test_adjust_decrease(self):
        """Test decreasing a value."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.cores": 8,
            }
        )

        change = adjuster.adjust(
            "spark.executor.cores",
            AdjustmentAction.DECREASE,
            amount=2,
        )

        assert adjuster.get("spark.executor.cores") == 6
        assert change.old_value == 8
        assert change.new_value == 6

    def test_adjust_reset(self):
        """Test resetting a value to default."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.memory": 16384,
            }
        )

        change = adjuster.adjust(
            "spark.executor.memory",
            AdjustmentAction.RESET,
        )

        # Default for executor.memory is 4096
        assert adjuster.get("spark.executor.memory") == 4096
        assert change.old_value == 16384
        assert change.new_value == 4096

    def test_adjust_set(self):
        """Test SET action in adjust."""
        adjuster = ConfigAdjuster()

        change = adjuster.adjust(
            "spark.executor.memory",
            AdjustmentAction.SET,
            amount=8192,
        )

        assert adjuster.get("spark.executor.memory") == 8192

    def test_adjust_respects_constraints(self):
        """Test that adjust respects min/max constraints."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.memory": 1024,
            }
        )

        # Try to decrease below minimum
        change = adjuster.adjust(
            "spark.executor.memory",
            AdjustmentAction.DECREASE,
            amount=1000,
        )

        # Should be clamped to minimum (512)
        assert adjuster.get("spark.executor.memory") >= 512

    def test_adjust_float_parameter(self):
        """Test adjusting float parameters."""
        adjuster = ConfigAdjuster(
            {
                "spark.memory.fraction": 0.6,
            }
        )

        change = adjuster.adjust(
            "spark.memory.fraction",
            AdjustmentAction.INCREASE,
            amount=0.1,
        )

        assert adjuster.get("spark.memory.fraction") == pytest.approx(0.7)

    def test_rollback_single_step(self):
        """Test rolling back a single change."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.memory": 4096,
            }
        )

        adjuster.set("spark.executor.memory", 8192)
        adjuster.set("spark.executor.memory", 16384)

        rolled_back = adjuster.rollback(1)

        assert len(rolled_back) == 1
        assert adjuster.get("spark.executor.memory") == 8192

    def test_rollback_multiple_steps(self):
        """Test rolling back multiple changes."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.memory": 4096,
            }
        )

        adjuster.set("spark.executor.memory", 8192)
        adjuster.set("spark.executor.memory", 16384)
        adjuster.set("spark.executor.memory", 32768)

        rolled_back = adjuster.rollback(2)

        assert len(rolled_back) == 2
        assert adjuster.get("spark.executor.memory") == 8192

    def test_rollback_removes_new_value(self):
        """Test that rollback removes values that didn't exist before."""
        adjuster = ConfigAdjuster()

        adjuster.set("new.parameter", "value")

        rolled_back = adjuster.rollback(1)

        assert adjuster.get("new.parameter") is None

    def test_rollback_more_than_history(self):
        """Test rolling back more steps than available."""
        adjuster = ConfigAdjuster()

        adjuster.set("test.param", "value")

        rolled_back = adjuster.rollback(10)

        assert len(rolled_back) == 1

    def test_validate_change_type(self):
        """Test type validation."""
        adjuster = ConfigAdjuster()

        # Integer parameter
        assert adjuster.validate_change("spark.executor.cores", 4) is True
        assert adjuster.validate_change("spark.executor.cores", "four") is False

        # Float parameter
        assert adjuster.validate_change("spark.memory.fraction", 0.5) is True
        assert adjuster.validate_change("spark.memory.fraction", "half") is False

        # Boolean parameter
        assert adjuster.validate_change("spark.shuffle.compress", True) is True
        assert adjuster.validate_change("spark.shuffle.compress", "true") is False

    def test_validate_change_range(self):
        """Test range validation."""
        adjuster = ConfigAdjuster()

        # Within range
        assert adjuster.validate_change("spark.executor.cores", 4) is True

        # Below minimum
        assert adjuster.validate_change("spark.executor.cores", 0) is False

        # Above maximum
        assert adjuster.validate_change("spark.executor.cores", 100) is False

    def test_custom_validator(self):
        """Test adding custom validators."""
        adjuster = ConfigAdjuster()

        def must_be_even(param, value, config):
            if param == "custom.param":
                return value % 2 == 0
            return True

        adjuster.add_validator(must_be_even)

        assert adjuster.validate_change("custom.param", 4) is True
        assert adjuster.validate_change("custom.param", 3) is False

    def test_get_related_params(self):
        """Test getting related parameters."""
        adjuster = ConfigAdjuster()

        related = adjuster.get_related_params("spark.executor.memory")

        assert "spark.memory.fraction" in related

        # Non-existent parameter
        related = adjuster.get_related_params("unknown.param")
        assert related == []

    def test_suggest_related_changes_memory(self):
        """Test suggesting related changes for memory."""
        adjuster = ConfigAdjuster()

        suggestions = adjuster.suggest_related_changes(
            "spark.executor.memory",
            8192,
        )

        assert "spark.executor.memoryOverhead" in suggestions
        # Memory overhead should be ~10% of executor memory
        assert suggestions["spark.executor.memoryOverhead"] >= 384

    def test_suggest_related_changes_cores(self):
        """Test suggesting related changes for cores."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.instances": 5,
            }
        )

        suggestions = adjuster.suggest_related_changes(
            "spark.executor.cores",
            4,
        )

        assert "spark.default.parallelism" in suggestions
        # Parallelism should be cores * executors * 2
        assert suggestions["spark.default.parallelism"] == 4 * 5 * 2

    def test_suggest_related_changes_instances_dynamic(self):
        """Test suggesting related changes for instances with dynamic allocation."""
        adjuster = ConfigAdjuster(
            {
                "spark.dynamicAllocation.enabled": True,
            }
        )

        suggestions = adjuster.suggest_related_changes(
            "spark.executor.instances",
            10,
        )

        assert "spark.dynamicAllocation.minExecutors" in suggestions
        assert "spark.dynamicAllocation.maxExecutors" in suggestions

    def test_to_spark_submit_args(self):
        """Test converting to spark-submit arguments."""
        adjuster = ConfigAdjuster(
            {
                "spark.executor.memory": 4096,
                "spark.executor.cores": 4,
                "spark.shuffle.compress": True,
            }
        )

        args = adjuster.to_spark_submit_args()

        assert "--conf" in args
        assert "spark.executor.memory=4096" in args
        assert "spark.executor.cores=4" in args
        assert "spark.shuffle.compress=true" in args

    def test_to_dict(self):
        """Test converting to dictionary."""
        initial = {
            "spark.executor.memory": 4096,
            "spark.executor.cores": 4,
        }
        adjuster = ConfigAdjuster(initial)

        result = adjuster.to_dict()

        assert result == initial
        # Should be a copy
        result["new_key"] = "value"
        assert "new_key" not in adjuster.config

    def test_from_dict(self):
        """Test loading from dictionary."""
        adjuster = ConfigAdjuster()

        config = {
            "spark.executor.memory": 4096,
            "spark.executor.cores": 4,
        }

        adjuster.from_dict(config)

        assert adjuster.get("spark.executor.memory") == 4096
        assert adjuster.get("spark.executor.cores") == 4

    def test_from_dict_with_validation(self):
        """Test loading with validation."""
        adjuster = ConfigAdjuster()

        config = {
            "spark.executor.memory": 100,  # Below minimum
        }

        with pytest.raises(ValueError):
            adjuster.from_dict(config)

    def test_from_dict_without_validation(self):
        """Test loading without validation."""
        adjuster = ConfigAdjuster()

        config = {
            "spark.executor.memory": 100,  # Below minimum
        }

        adjuster.from_dict(config, validate=False)

        assert adjuster.get("spark.executor.memory") == 100


class TestConfigAdjusterConstraints:
    """Test built-in constraints."""

    def test_constraints_exist(self):
        """Test that constraints are defined."""
        assert len(ConfigAdjuster.CONSTRAINTS) > 0

    def test_memory_constraints(self):
        """Test memory parameter constraints."""
        constraints = ConfigAdjuster.CONSTRAINTS["spark.executor.memory"]

        assert constraints["type"] == "memory"
        assert constraints["min"] > 0
        assert constraints["max"] > constraints["min"]
        assert "default" in constraints

    def test_cores_constraints(self):
        """Test cores parameter constraints."""
        constraints = ConfigAdjuster.CONSTRAINTS["spark.executor.cores"]

        assert constraints["type"] == "integer"
        assert constraints["min"] >= 1
        assert constraints["max"] > constraints["min"]

    def test_fraction_constraints(self):
        """Test fraction parameter constraints."""
        constraints = ConfigAdjuster.CONSTRAINTS["spark.memory.fraction"]

        assert constraints["type"] == "float"
        assert 0 < constraints["min"] < 1
        assert 0 < constraints["max"] <= 1

    def test_boolean_constraints(self):
        """Test boolean parameter constraints."""
        constraints = ConfigAdjuster.CONSTRAINTS["spark.shuffle.compress"]

        assert constraints["type"] == "boolean"
        assert "default" in constraints


class TestConfigAdjusterHistory:
    """Test change history functionality."""

    def test_history_records_all_changes(self):
        """Test that all changes are recorded in history."""
        adjuster = ConfigAdjuster()

        adjuster.set("param1", "value1")
        adjuster.set("param2", "value2")
        adjuster.set("param1", "value3")

        assert len(adjuster.history) == 3

    def test_history_preserves_order(self):
        """Test that history maintains chronological order."""
        adjuster = ConfigAdjuster()

        adjuster.set("param", "first")
        adjuster.set("param", "second")
        adjuster.set("param", "third")

        assert adjuster.history[0].new_value == "first"
        assert adjuster.history[1].new_value == "second"
        assert adjuster.history[2].new_value == "third"

    def test_history_is_copy(self):
        """Test that history property returns a copy."""
        adjuster = ConfigAdjuster()

        adjuster.set("param", "value")

        history = adjuster.history
        history.clear()

        assert len(adjuster.history) == 1
