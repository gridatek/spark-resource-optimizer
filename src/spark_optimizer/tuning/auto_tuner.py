"""Auto-tuning capabilities for Spark applications."""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
import time

logger = logging.getLogger(__name__)


class TuningStrategy(Enum):
    """Tuning strategy types."""

    CONSERVATIVE = "conservative"  # Small, safe adjustments
    MODERATE = "moderate"  # Balanced adjustments
    AGGRESSIVE = "aggressive"  # Larger adjustments for faster convergence


@dataclass
class TuningConfig:
    """Configuration for a tuning parameter."""

    name: str
    current_value: Any
    min_value: Any
    max_value: Any
    step_size: Any
    unit: str = ""
    description: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step_size": self.step_size,
            "unit": self.unit,
            "description": self.description,
        }


@dataclass
class TuningAdjustment:
    """A configuration adjustment made by the auto-tuner."""

    parameter: str
    old_value: Any
    new_value: Any
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    applied: bool = False
    result: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "parameter": self.parameter,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat(),
            "applied": self.applied,
            "result": self.result,
        }


@dataclass
class TuningSession:
    """A tuning session for an application."""

    session_id: str
    app_id: str
    app_name: str
    strategy: TuningStrategy
    target_metric: str  # Metric to optimize (e.g., "duration", "cost", "throughput")
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    status: str = "active"  # active, paused, completed, failed
    iterations: int = 0
    adjustments: List[TuningAdjustment] = field(default_factory=list)
    initial_config: Dict = field(default_factory=dict)
    current_config: Dict = field(default_factory=dict)
    best_config: Dict = field(default_factory=dict)
    best_metric_value: Optional[float] = None
    metrics_history: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "app_id": self.app_id,
            "app_name": self.app_name,
            "strategy": self.strategy.value,
            "target_metric": self.target_metric,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "status": self.status,
            "iterations": self.iterations,
            "adjustments": [a.to_dict() for a in self.adjustments],
            "initial_config": self.initial_config,
            "current_config": self.current_config,
            "best_config": self.best_config,
            "best_metric_value": self.best_metric_value,
        }


class AutoTuner:
    """Automatic configuration tuner for Spark applications.

    Provides dynamic configuration adjustment based on:
    - Real-time metrics analysis
    - Historical performance data
    - Configurable tuning strategies
    - Feedback loop learning
    """

    # Default tunable parameters with their ranges
    DEFAULT_TUNABLE_PARAMS = {
        "spark.executor.memory": TuningConfig(
            name="spark.executor.memory",
            current_value=4096,
            min_value=1024,
            max_value=32768,
            step_size=1024,
            unit="MB",
            description="Executor memory allocation",
        ),
        "spark.executor.cores": TuningConfig(
            name="spark.executor.cores",
            current_value=4,
            min_value=1,
            max_value=16,
            step_size=1,
            unit="cores",
            description="Number of cores per executor",
        ),
        "spark.executor.instances": TuningConfig(
            name="spark.executor.instances",
            current_value=5,
            min_value=1,
            max_value=100,
            step_size=1,
            unit="executors",
            description="Number of executor instances",
        ),
        "spark.driver.memory": TuningConfig(
            name="spark.driver.memory",
            current_value=2048,
            min_value=512,
            max_value=16384,
            step_size=512,
            unit="MB",
            description="Driver memory allocation",
        ),
        "spark.sql.shuffle.partitions": TuningConfig(
            name="spark.sql.shuffle.partitions",
            current_value=200,
            min_value=10,
            max_value=2000,
            step_size=50,
            unit="partitions",
            description="Number of shuffle partitions",
        ),
        "spark.default.parallelism": TuningConfig(
            name="spark.default.parallelism",
            current_value=100,
            min_value=10,
            max_value=1000,
            step_size=10,
            unit="tasks",
            description="Default parallelism level",
        ),
        "spark.memory.fraction": TuningConfig(
            name="spark.memory.fraction",
            current_value=0.6,
            min_value=0.3,
            max_value=0.9,
            step_size=0.05,
            unit="fraction",
            description="Fraction of heap for execution/storage",
        ),
        "spark.memory.storageFraction": TuningConfig(
            name="spark.memory.storageFraction",
            current_value=0.5,
            min_value=0.2,
            max_value=0.8,
            step_size=0.05,
            unit="fraction",
            description="Fraction of memory pool for storage",
        ),
    }

    # Strategy multipliers for adjustment step sizes
    STRATEGY_MULTIPLIERS = {
        TuningStrategy.CONSERVATIVE: 0.5,
        TuningStrategy.MODERATE: 1.0,
        TuningStrategy.AGGRESSIVE: 2.0,
    }

    def __init__(
        self,
        tunable_params: Optional[Dict[str, TuningConfig]] = None,
        max_iterations: int = 20,
        convergence_threshold: float = 0.02,
    ):
        """Initialize the auto-tuner.

        Args:
            tunable_params: Dictionary of tunable parameters
            max_iterations: Maximum tuning iterations
            convergence_threshold: Stop when improvement is below this threshold
        """
        self._tunable_params = tunable_params or dict(self.DEFAULT_TUNABLE_PARAMS)
        self._max_iterations = max_iterations
        self._convergence_threshold = convergence_threshold

        self._sessions: Dict[str, TuningSession] = {}
        self._active_sessions: Dict[str, str] = {}  # app_id -> session_id
        self._lock = threading.Lock()
        self._session_counter = 0

    def start_session(
        self,
        app_id: str,
        app_name: str,
        initial_config: Dict,
        strategy: TuningStrategy = TuningStrategy.MODERATE,
        target_metric: str = "duration",
    ) -> TuningSession:
        """Start a new tuning session for an application.

        Args:
            app_id: Spark application ID
            app_name: Application name
            initial_config: Initial Spark configuration
            strategy: Tuning strategy to use
            target_metric: Metric to optimize

        Returns:
            New tuning session
        """
        with self._lock:
            # Check for existing active session
            if app_id in self._active_sessions:
                existing = self._sessions.get(self._active_sessions[app_id])
                if existing and existing.status == "active":
                    logger.warning(
                        f"Active session exists for {app_id}, returning existing"
                    )
                    return existing

            # Create new session
            self._session_counter += 1
            session_id = (
                f"tune-{self._session_counter}-{int(datetime.utcnow().timestamp())}"
            )

            session = TuningSession(
                session_id=session_id,
                app_id=app_id,
                app_name=app_name,
                strategy=strategy,
                target_metric=target_metric,
                initial_config=initial_config.copy(),
                current_config=initial_config.copy(),
                best_config=initial_config.copy(),
            )

            self._sessions[session_id] = session
            self._active_sessions[app_id] = session_id

            logger.info(f"Started tuning session {session_id} for {app_id}")
            return session

    def get_session(self, session_id: str) -> Optional[TuningSession]:
        """Get a tuning session by ID.

        Args:
            session_id: Session ID

        Returns:
            Tuning session or None
        """
        with self._lock:
            return self._sessions.get(session_id)

    def get_active_session(self, app_id: str) -> Optional[TuningSession]:
        """Get active tuning session for an application.

        Args:
            app_id: Spark application ID

        Returns:
            Active tuning session or None
        """
        with self._lock:
            session_id = self._active_sessions.get(app_id)
            if session_id:
                return self._sessions.get(session_id)
        return None

    def list_sessions(
        self,
        app_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[TuningSession]:
        """List tuning sessions.

        Args:
            app_id: Filter by application ID
            status: Filter by status

        Returns:
            List of tuning sessions
        """
        with self._lock:
            sessions = list(self._sessions.values())

            if app_id:
                sessions = [s for s in sessions if s.app_id == app_id]

            if status:
                sessions = [s for s in sessions if s.status == status]

            return sorted(sessions, key=lambda s: s.started_at, reverse=True)

    def analyze_and_recommend(
        self,
        session_id: str,
        current_metrics: Dict[str, float],
    ) -> List[TuningAdjustment]:
        """Analyze metrics and recommend configuration adjustments.

        Args:
            session_id: Tuning session ID
            current_metrics: Current performance metrics

        Returns:
            List of recommended adjustments
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session or session.status != "active":
                return []

            # Record metrics
            session.metrics_history.append(
                {
                    "iteration": session.iterations,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metrics": current_metrics.copy(),
                }
            )

            # Get current target metric value
            target_value = current_metrics.get(session.target_metric)
            if target_value is None:
                logger.warning(f"Target metric {session.target_metric} not in metrics")
                return []

            # Update best config if improved
            improved = False
            if session.best_metric_value is None:
                session.best_metric_value = target_value
                session.best_config = session.current_config.copy()
                improved = True
            elif self._is_better(
                target_value, session.best_metric_value, session.target_metric
            ):
                improvement = abs(target_value - session.best_metric_value) / max(
                    abs(session.best_metric_value), 1
                )
                session.best_metric_value = target_value
                session.best_config = session.current_config.copy()
                improved = True

                # Check convergence
                if improvement < self._convergence_threshold:
                    logger.info(
                        f"Session {session_id} converged (improvement: {improvement:.4f})"
                    )
                    session.status = "completed"
                    session.ended_at = datetime.utcnow()
                    return []

            # Check max iterations
            session.iterations += 1
            if session.iterations >= self._max_iterations:
                logger.info(f"Session {session_id} reached max iterations")
                session.status = "completed"
                session.ended_at = datetime.utcnow()
                return []

            # Generate recommendations based on metrics analysis
            adjustments = self._generate_adjustments(session, current_metrics)

            return adjustments

    def apply_adjustment(
        self,
        session_id: str,
        adjustment: TuningAdjustment,
    ) -> bool:
        """Mark an adjustment as applied and update session config.

        Args:
            session_id: Tuning session ID
            adjustment: Adjustment that was applied

        Returns:
            True if successful
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False

            adjustment.applied = True
            session.adjustments.append(adjustment)
            session.current_config[adjustment.parameter] = adjustment.new_value

            logger.info(
                f"Applied adjustment: {adjustment.parameter} "
                f"{adjustment.old_value} -> {adjustment.new_value}"
            )
            return True

    def pause_session(self, session_id: str) -> bool:
        """Pause a tuning session.

        Args:
            session_id: Session ID

        Returns:
            True if paused successfully
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == "active":
                session.status = "paused"
                return True
        return False

    def resume_session(self, session_id: str) -> bool:
        """Resume a paused tuning session.

        Args:
            session_id: Session ID

        Returns:
            True if resumed successfully
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status == "paused":
                session.status = "active"
                return True
        return False

    def end_session(self, session_id: str, status: str = "completed") -> bool:
        """End a tuning session.

        Args:
            session_id: Session ID
            status: Final status (completed, failed, cancelled)

        Returns:
            True if ended successfully
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and session.status in ["active", "paused"]:
                session.status = status
                session.ended_at = datetime.utcnow()

                # Remove from active sessions
                if session.app_id in self._active_sessions:
                    if self._active_sessions[session.app_id] == session_id:
                        del self._active_sessions[session.app_id]

                return True
        return False

    def get_best_config(self, session_id: str) -> Optional[Dict]:
        """Get the best configuration found in a session.

        Args:
            session_id: Session ID

        Returns:
            Best configuration or None
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session.best_config.copy()
        return None

    def _is_better(self, new_value: float, old_value: float, metric: str) -> bool:
        """Check if new metric value is better than old.

        Args:
            new_value: New metric value
            old_value: Old metric value
            metric: Metric name

        Returns:
            True if new is better
        """
        # For these metrics, lower is better
        lower_is_better = ["duration", "cost", "gc_time", "spill_bytes", "latency"]

        if any(m in metric.lower() for m in lower_is_better):
            return new_value < old_value
        else:
            # For throughput, efficiency, etc., higher is better
            return new_value > old_value

    def _generate_adjustments(
        self,
        session: TuningSession,
        metrics: Dict[str, float],
    ) -> List[TuningAdjustment]:
        """Generate configuration adjustments based on metrics.

        Args:
            session: Tuning session
            metrics: Current metrics

        Returns:
            List of recommended adjustments
        """
        adjustments = []
        strategy_mult = self.STRATEGY_MULTIPLIERS[session.strategy]

        # Analyze specific issues and recommend fixes

        # 1. Memory pressure / spilling
        memory_spill = metrics.get("memory_spill_ratio", 0)
        disk_spill = metrics.get("disk_spill_bytes", 0)

        if memory_spill > 0.1 or disk_spill > 0:
            # Increase executor memory
            param = "spark.executor.memory"
            if param in self._tunable_params:
                config = self._tunable_params[param]
                current = session.current_config.get(param, config.current_value)
                step = int(config.step_size * strategy_mult)
                new_value = min(current + step, config.max_value)

                if new_value != current:
                    adjustments.append(
                        TuningAdjustment(
                            parameter=param,
                            old_value=current,
                            new_value=new_value,
                            reason=f"Memory spilling detected (ratio: {memory_spill:.2f})",
                        )
                    )

        # 2. GC pressure
        gc_time_percent = metrics.get("gc_time_percent", 0)

        if gc_time_percent > 10:
            # Reduce memory fraction or increase memory
            param = "spark.memory.fraction"
            if param in self._tunable_params:
                config = self._tunable_params[param]
                current = session.current_config.get(param, config.current_value)
                step = config.step_size * strategy_mult
                new_value = max(current - step, config.min_value)

                if new_value != current:
                    adjustments.append(
                        TuningAdjustment(
                            parameter=param,
                            old_value=current,
                            new_value=round(new_value, 2),
                            reason=f"High GC time ({gc_time_percent:.1f}%)",
                        )
                    )

        # 3. CPU underutilization
        cpu_utilization = metrics.get("cpu_utilization", 100)

        if cpu_utilization < 50:
            # Increase parallelism
            param = "spark.default.parallelism"
            if param in self._tunable_params:
                config = self._tunable_params[param]
                current = session.current_config.get(param, config.current_value)
                step = int(config.step_size * strategy_mult)
                new_value = min(current + step, config.max_value)

                if new_value != current:
                    adjustments.append(
                        TuningAdjustment(
                            parameter=param,
                            old_value=current,
                            new_value=new_value,
                            reason=f"Low CPU utilization ({cpu_utilization:.1f}%)",
                        )
                    )

        # 4. Shuffle performance
        shuffle_spill_ratio = metrics.get("shuffle_spill_ratio", 0)

        if shuffle_spill_ratio > 0.2:
            # Increase shuffle partitions
            param = "spark.sql.shuffle.partitions"
            if param in self._tunable_params:
                config = self._tunable_params[param]
                current = session.current_config.get(param, config.current_value)
                step = int(config.step_size * strategy_mult)
                new_value = min(current + step, config.max_value)

                if new_value != current:
                    adjustments.append(
                        TuningAdjustment(
                            parameter=param,
                            old_value=current,
                            new_value=new_value,
                            reason=f"High shuffle spill ratio ({shuffle_spill_ratio:.2f})",
                        )
                    )

        # 5. Task failures / stragglers
        task_failure_rate = metrics.get("task_failure_rate", 0)

        if task_failure_rate > 0.05:
            # Could indicate memory issues - increase memory or reduce cores
            param = "spark.executor.cores"
            if param in self._tunable_params:
                config = self._tunable_params[param]
                current = session.current_config.get(param, config.current_value)
                step = int(config.step_size * strategy_mult)
                new_value = max(current - step, config.min_value)

                if new_value != current:
                    adjustments.append(
                        TuningAdjustment(
                            parameter=param,
                            old_value=current,
                            new_value=new_value,
                            reason=f"High task failure rate ({task_failure_rate:.1%})",
                        )
                    )

        # 6. Cost optimization (if targeting cost)
        if session.target_metric == "cost":
            cost_per_task = metrics.get("cost_per_task", 0)
            task_throughput = metrics.get("task_throughput", 0)

            # If throughput is good, try reducing executors
            if task_throughput > 0 and task_failure_rate < 0.01:
                param = "spark.executor.instances"
                if param in self._tunable_params:
                    config = self._tunable_params[param]
                    current = session.current_config.get(param, config.current_value)

                    # Conservative reduction
                    new_value = max(current - 1, config.min_value)

                    if new_value != current:
                        adjustments.append(
                            TuningAdjustment(
                                parameter=param,
                                old_value=current,
                                new_value=new_value,
                                reason="Cost optimization - reducing executors",
                            )
                        )

        return adjustments

    def get_tunable_parameters(self) -> Dict[str, TuningConfig]:
        """Get all tunable parameters.

        Returns:
            Dictionary of tunable parameters
        """
        return self._tunable_params.copy()

    def set_parameter_range(
        self,
        param_name: str,
        min_value: Any = None,
        max_value: Any = None,
        step_size: Any = None,
    ) -> bool:
        """Update range for a tunable parameter.

        Args:
            param_name: Parameter name
            min_value: New minimum value
            max_value: New maximum value
            step_size: New step size

        Returns:
            True if parameter exists and was updated
        """
        if param_name not in self._tunable_params:
            return False

        config = self._tunable_params[param_name]

        if min_value is not None:
            config.min_value = min_value
        if max_value is not None:
            config.max_value = max_value
        if step_size is not None:
            config.step_size = step_size

        return True
