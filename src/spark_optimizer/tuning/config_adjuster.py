"""Configuration adjustment utilities for auto-tuning."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AdjustmentAction(Enum):
    """Types of configuration adjustments."""

    INCREASE = "increase"
    DECREASE = "decrease"
    SET = "set"
    RESET = "reset"


@dataclass
class ConfigChange:
    """A configuration change record."""

    parameter: str
    action: AdjustmentAction
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "auto"  # auto, manual, rule
    reason: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "action": self.action.value,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "reason": self.reason,
        }


class ConfigAdjuster:
    """Handles Spark configuration adjustments.

    Provides utilities for:
    - Validating configuration changes
    - Applying changes safely
    - Rolling back changes
    - Tracking change history
    """

    # Spark configuration constraints
    CONSTRAINTS = {
        "spark.executor.memory": {
            "type": "memory",
            "min": 512,  # MB
            "max": 65536,  # MB
            "default": 4096,
        },
        "spark.executor.cores": {
            "type": "integer",
            "min": 1,
            "max": 32,
            "default": 4,
        },
        "spark.executor.instances": {
            "type": "integer",
            "min": 1,
            "max": 500,
            "default": 2,
        },
        "spark.driver.memory": {
            "type": "memory",
            "min": 512,
            "max": 32768,
            "default": 2048,
        },
        "spark.driver.cores": {
            "type": "integer",
            "min": 1,
            "max": 16,
            "default": 1,
        },
        "spark.sql.shuffle.partitions": {
            "type": "integer",
            "min": 1,
            "max": 10000,
            "default": 200,
        },
        "spark.default.parallelism": {
            "type": "integer",
            "min": 1,
            "max": 10000,
            "default": 100,
        },
        "spark.memory.fraction": {
            "type": "float",
            "min": 0.1,
            "max": 0.95,
            "default": 0.6,
        },
        "spark.memory.storageFraction": {
            "type": "float",
            "min": 0.1,
            "max": 0.9,
            "default": 0.5,
        },
        "spark.shuffle.compress": {
            "type": "boolean",
            "default": True,
        },
        "spark.shuffle.spill.compress": {
            "type": "boolean",
            "default": True,
        },
        "spark.dynamicAllocation.enabled": {
            "type": "boolean",
            "default": False,
        },
        "spark.dynamicAllocation.minExecutors": {
            "type": "integer",
            "min": 0,
            "max": 100,
            "default": 0,
        },
        "spark.dynamicAllocation.maxExecutors": {
            "type": "integer",
            "min": 1,
            "max": 500,
            "default": 10,
        },
    }

    # Interdependent parameters that should be adjusted together
    RELATED_PARAMS = {
        "spark.executor.memory": [
            "spark.memory.fraction",
            "spark.executor.memoryOverhead",
        ],
        "spark.executor.cores": ["spark.task.cpus", "spark.default.parallelism"],
        "spark.executor.instances": [
            "spark.dynamicAllocation.minExecutors",
            "spark.dynamicAllocation.maxExecutors",
        ],
    }

    def __init__(self, initial_config: Optional[Dict] = None):
        """Initialize the config adjuster.

        Args:
            initial_config: Initial Spark configuration
        """
        self._config = initial_config.copy() if initial_config else {}
        self._history: List[ConfigChange] = []
        self._validators: List[Callable[[str, Any, Dict], bool]] = []

    @property
    def config(self) -> Dict:
        """Get current configuration."""
        return self._config.copy()

    @property
    def history(self) -> List[ConfigChange]:
        """Get change history."""
        return self._history.copy()

    def add_validator(self, validator: Callable[[str, Any, Dict], bool]) -> None:
        """Add a custom validator function.

        Args:
            validator: Function(param_name, new_value, full_config) -> bool
        """
        self._validators.append(validator)

    def get(self, parameter: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            parameter: Parameter name
            default: Default value if not set

        Returns:
            Parameter value or default
        """
        return self._config.get(parameter, default)

    def set(
        self,
        parameter: str,
        value: Any,
        source: str = "manual",
        reason: str = "",
        validate: bool = True,
    ) -> ConfigChange:
        """Set a configuration value.

        Args:
            parameter: Parameter name
            value: New value
            source: Source of change (auto, manual, rule)
            reason: Reason for change
            validate: Whether to validate the change

        Returns:
            ConfigChange record

        Raises:
            ValueError: If validation fails
        """
        if validate and not self.validate_change(parameter, value):
            raise ValueError(f"Invalid value {value} for parameter {parameter}")

        old_value = self._config.get(parameter)
        self._config[parameter] = value

        change = ConfigChange(
            parameter=parameter,
            action=AdjustmentAction.SET,
            old_value=old_value,
            new_value=value,
            source=source,
            reason=reason,
        )
        self._history.append(change)

        logger.info(f"Config set: {parameter} = {value} (was: {old_value})")
        return change

    def adjust(
        self,
        parameter: str,
        action: AdjustmentAction,
        amount: Optional[Any] = None,
        source: str = "auto",
        reason: str = "",
    ) -> ConfigChange:
        """Adjust a configuration value.

        Args:
            parameter: Parameter name
            action: Type of adjustment
            amount: Amount to adjust by (for increase/decrease)
            source: Source of change
            reason: Reason for change

        Returns:
            ConfigChange record

        Raises:
            ValueError: If adjustment is invalid
        """
        old_value = self._config.get(parameter)
        constraint = self.CONSTRAINTS.get(parameter, {})
        param_type = constraint.get("type", "string")

        if action == AdjustmentAction.RESET:
            new_value = constraint.get("default", old_value)

        elif action == AdjustmentAction.SET:
            new_value = amount

        elif action == AdjustmentAction.INCREASE:
            if param_type in ["integer", "memory"]:
                new_value = (old_value or 0) + (amount or 1)
            elif param_type == "float":
                new_value = (old_value or 0) + (amount or 0.1)
            else:
                raise ValueError(f"Cannot increase parameter of type {param_type}")

        elif action == AdjustmentAction.DECREASE:
            if param_type in ["integer", "memory"]:
                new_value = (old_value or 0) - (amount or 1)
            elif param_type == "float":
                new_value = (old_value or 0) - (amount or 0.1)
            else:
                raise ValueError(f"Cannot decrease parameter of type {param_type}")

        else:
            raise ValueError(f"Unknown action: {action}")

        # Apply constraints
        new_value = self._apply_constraints(parameter, new_value)

        if not self.validate_change(parameter, new_value):
            raise ValueError(f"Invalid value {new_value} for parameter {parameter}")

        self._config[parameter] = new_value

        change = ConfigChange(
            parameter=parameter,
            action=action,
            old_value=old_value,
            new_value=new_value,
            source=source,
            reason=reason,
        )
        self._history.append(change)

        logger.info(f"Config adjusted: {parameter} {action.value} to {new_value}")
        return change

    def rollback(self, steps: int = 1) -> List[ConfigChange]:
        """Rollback recent configuration changes.

        Args:
            steps: Number of changes to rollback

        Returns:
            List of rolled back changes
        """
        rolled_back = []

        for _ in range(min(steps, len(self._history))):
            if not self._history:
                break

            change = self._history.pop()

            # Restore old value
            if change.old_value is None:
                self._config.pop(change.parameter, None)
            else:
                self._config[change.parameter] = change.old_value

            rolled_back.append(change)
            logger.info(f"Rolled back: {change.parameter} to {change.old_value}")

        return rolled_back

    def validate_change(self, parameter: str, value: Any) -> bool:
        """Validate a configuration change.

        Args:
            parameter: Parameter name
            value: Proposed value

        Returns:
            True if valid
        """
        # Check built-in constraints
        constraint = self.CONSTRAINTS.get(parameter)

        if constraint:
            param_type = constraint.get("type")

            # Type validation
            if param_type == "integer":
                if not isinstance(value, (int, float)):
                    return False
                value = int(value)

            elif param_type == "float":
                if not isinstance(value, (int, float)):
                    return False

            elif param_type == "boolean":
                if not isinstance(value, bool):
                    return False

            elif param_type == "memory":
                if not isinstance(value, (int, float)):
                    return False
                value = int(value)

            # Range validation
            min_val = constraint.get("min")
            max_val = constraint.get("max")

            if min_val is not None and value < min_val:
                return False
            if max_val is not None and value > max_val:
                return False

        # Run custom validators
        for validator in self._validators:
            try:
                if not validator(parameter, value, self._config):
                    return False
            except Exception as e:
                logger.warning(f"Validator error: {e}")
                return False

        return True

    def _apply_constraints(self, parameter: str, value: Any) -> Any:
        """Apply constraints to a value.

        Args:
            parameter: Parameter name
            value: Value to constrain

        Returns:
            Constrained value
        """
        constraint = self.CONSTRAINTS.get(parameter)

        if not constraint:
            return value

        param_type = constraint.get("type")

        # Apply type conversion
        if param_type == "integer":
            value = int(value)
        elif param_type == "memory":
            value = int(value)
        elif param_type == "float":
            value = float(value)

        # Apply min/max
        min_val = constraint.get("min")
        max_val = constraint.get("max")

        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)

        return value

    def get_related_params(self, parameter: str) -> List[str]:
        """Get parameters that should be considered when adjusting this one.

        Args:
            parameter: Parameter name

        Returns:
            List of related parameter names
        """
        return self.RELATED_PARAMS.get(parameter, [])

    def suggest_related_changes(
        self,
        parameter: str,
        new_value: Any,
    ) -> Dict[str, Any]:
        """Suggest changes to related parameters.

        Args:
            parameter: Parameter being changed
            new_value: New value for parameter

        Returns:
            Dictionary of suggested related changes
        """
        suggestions = {}

        if parameter == "spark.executor.memory":
            # Suggest memory overhead (10% of executor memory, min 384MB)
            overhead = max(384, int(new_value * 0.1))
            suggestions["spark.executor.memoryOverhead"] = overhead

        elif parameter == "spark.executor.cores":
            # Suggest parallelism based on cores
            executor_count = self._config.get("spark.executor.instances", 2)
            parallelism = new_value * executor_count * 2
            suggestions["spark.default.parallelism"] = parallelism

        elif parameter == "spark.executor.instances":
            # If using dynamic allocation, update min/max
            if self._config.get("spark.dynamicAllocation.enabled"):
                suggestions["spark.dynamicAllocation.minExecutors"] = max(
                    1, new_value // 2
                )
                suggestions["spark.dynamicAllocation.maxExecutors"] = new_value * 2

        return suggestions

    def to_spark_submit_args(self) -> List[str]:
        """Convert configuration to spark-submit arguments.

        Returns:
            List of spark-submit arguments
        """
        args = []

        for param, value in self._config.items():
            if isinstance(value, bool):
                value = "true" if value else "false"
            args.extend(["--conf", f"{param}={value}"])

        return args

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()

    def from_dict(self, config: Dict, validate: bool = True) -> None:
        """Load configuration from dictionary.

        Args:
            config: Configuration dictionary
            validate: Whether to validate values

        Raises:
            ValueError: If validation fails and validate=True
        """
        for param, value in config.items():
            if validate and not self.validate_change(param, value):
                raise ValueError(f"Invalid value {value} for parameter {param}")
            self._config[param] = value
