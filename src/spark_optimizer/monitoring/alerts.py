"""Alert management for real-time monitoring."""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """An alert triggered by monitoring conditions."""

    id: str
    app_id: str
    severity: AlertSeverity
    title: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "app_id": self.app_id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "acknowledged_by": self.acknowledged_by,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
        }


@dataclass
class AlertRule:
    """A rule for triggering alerts."""

    name: str
    metric_name: str
    condition: str  # "gt", "lt", "gte", "lte", "eq", "neq"
    threshold: float
    severity: AlertSeverity
    title_template: str
    message_template: str
    cooldown_minutes: int = 5  # Minimum time between repeated alerts
    enabled: bool = True

    def evaluate(self, value: float) -> bool:
        """Evaluate if the condition is met.

        Args:
            value: Metric value to evaluate

        Returns:
            True if condition is met, False otherwise
        """
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        elif self.condition == "gte":
            return value >= self.threshold
        elif self.condition == "lte":
            return value <= self.threshold
        elif self.condition == "eq":
            return value == self.threshold
        elif self.condition == "neq":
            return value != self.threshold
        return False


class AlertManager:
    """Manages alerts for monitored applications.

    Supports:
    - Threshold-based alerting
    - Alert cooldowns to prevent flooding
    - Alert acknowledgement and resolution
    - Alert history
    """

    # Default alert rules for common issues
    DEFAULT_RULES = [
        AlertRule(
            name="high_gc_time",
            metric_name="jvm_gc_time_percent",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="High GC Time",
            message_template="GC time is {value:.1f}%, exceeding {threshold}% threshold",
        ),
        AlertRule(
            name="critical_gc_time",
            metric_name="jvm_gc_time_percent",
            condition="gt",
            threshold=25.0,
            severity=AlertSeverity.CRITICAL,
            title_template="Critical GC Time",
            message_template="GC time is critically high at {value:.1f}%",
        ),
        AlertRule(
            name="memory_spilling",
            metric_name="memory_spill_ratio",
            condition="gt",
            threshold=0.1,
            severity=AlertSeverity.WARNING,
            title_template="Memory Spilling Detected",
            message_template="Memory spill ratio is {value:.2f}, data is spilling to disk",
        ),
        AlertRule(
            name="high_task_failure_rate",
            metric_name="task_failure_rate",
            condition="gt",
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            title_template="High Task Failure Rate",
            message_template="Task failure rate is {value:.1%}, exceeding {threshold:.1%}",
        ),
        AlertRule(
            name="executor_lost",
            metric_name="executors_lost",
            condition="gt",
            threshold=0,
            severity=AlertSeverity.ERROR,
            title_template="Executor Lost",
            message_template="{value:.0f} executor(s) have been lost",
        ),
        AlertRule(
            name="low_cpu_utilization",
            metric_name="cpu_utilization",
            condition="lt",
            threshold=30.0,
            severity=AlertSeverity.INFO,
            title_template="Low CPU Utilization",
            message_template="CPU utilization is only {value:.1f}%, resources may be over-provisioned",
        ),
        AlertRule(
            name="high_shuffle_spill",
            metric_name="shuffle_spill_ratio",
            condition="gt",
            threshold=0.2,
            severity=AlertSeverity.WARNING,
            title_template="High Shuffle Spill",
            message_template="Shuffle spill ratio is {value:.2f}, consider increasing executor memory",
        ),
    ]

    def __init__(self, rules: Optional[List[AlertRule]] = None):
        """Initialize the alert manager.

        Args:
            rules: List of alert rules. If None, uses DEFAULT_RULES
        """
        self._rules = rules if rules is not None else list(self.DEFAULT_RULES)
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._last_alert_time: Dict[str, datetime] = {}  # rule_name:app_id -> time
        self._subscribers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self._alert_counter = 0

    def add_rule(self, rule: AlertRule) -> None:
        """Add a new alert rule.

        Args:
            rule: Alert rule to add
        """
        with self._lock:
            self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule by name.

        Args:
            name: Rule name

        Returns:
            True if rule was removed, False if not found
        """
        with self._lock:
            for i, rule in enumerate(self._rules):
                if rule.name == name:
                    self._rules.pop(i)
                    return True
        return False

    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules.

        Returns:
            List of alert rules
        """
        with self._lock:
            return list(self._rules)

    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        """Subscribe to alert notifications.

        Args:
            callback: Function to call when alert is triggered
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Alert], None]) -> None:
        """Unsubscribe from alert notifications.

        Args:
            callback: Previously subscribed callback
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def evaluate_metrics(self, app_id: str, metrics: Dict[str, float]) -> List[Alert]:
        """Evaluate metrics against all rules and trigger alerts.

        Args:
            app_id: Spark application ID
            metrics: Dictionary of metric name to value

        Returns:
            List of newly triggered alerts
        """
        new_alerts = []

        with self._lock:
            for rule in self._rules:
                if not rule.enabled:
                    continue

                if rule.metric_name not in metrics:
                    continue

                value = metrics[rule.metric_name]

                if rule.evaluate(value):
                    # Check cooldown
                    cooldown_key = f"{rule.name}:{app_id}"
                    last_time = self._last_alert_time.get(cooldown_key)

                    if last_time:
                        cooldown = timedelta(minutes=rule.cooldown_minutes)
                        if datetime.utcnow() - last_time < cooldown:
                            continue

                    # Create alert
                    alert = self._create_alert(app_id, rule, value)
                    new_alerts.append(alert)
                    self._last_alert_time[cooldown_key] = datetime.utcnow()

        # Notify subscribers
        for alert in new_alerts:
            self._notify_subscribers(alert)

        return new_alerts

    def get_active_alerts(self, app_id: Optional[str] = None) -> List[Alert]:
        """Get active (unresolved) alerts.

        Args:
            app_id: Optional app ID to filter by

        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = [a for a in self._alerts.values() if not a.resolved]
            if app_id:
                alerts = [a for a in alerts if a.app_id == app_id]
            return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID.

        Args:
            alert_id: Alert ID

        Returns:
            Alert or None if not found
        """
        with self._lock:
            return self._alerts.get(alert_id)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "user") -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: User or system that acknowledged

        Returns:
            True if alert was acknowledged, False if not found
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and not alert.acknowledged:
                alert.acknowledged = True
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if alert was resolved, False if not found
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                self._alert_history.append(alert)
                return True
        return False

    def get_alert_history(
        self,
        app_id: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alert history.

        Args:
            app_id: Optional app ID to filter by
            since: Only return alerts after this time
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        with self._lock:
            history = self._alert_history.copy()

            if app_id:
                history = [a for a in history if a.app_id == app_id]

            if since:
                history = [a for a in history if a.created_at > since]

            return sorted(history, key=lambda a: a.created_at, reverse=True)[:limit]

    def _create_alert(self, app_id: str, rule: AlertRule, value: float) -> Alert:
        """Create a new alert from a triggered rule.

        Args:
            app_id: Spark application ID
            rule: Triggered rule
            value: Metric value that triggered the rule

        Returns:
            New alert
        """
        self._alert_counter += 1
        alert_id = f"alert-{self._alert_counter}-{int(datetime.utcnow().timestamp())}"

        title = rule.title_template.format(value=value, threshold=rule.threshold)
        message = rule.message_template.format(value=value, threshold=rule.threshold)

        alert = Alert(
            id=alert_id,
            app_id=app_id,
            severity=rule.severity,
            title=title,
            message=message,
            metric_name=rule.metric_name,
            metric_value=value,
            threshold=rule.threshold,
        )

        self._alerts[alert_id] = alert
        logger.info(f"Alert triggered: {alert.title} for {app_id}")

        return alert

    def _notify_subscribers(self, alert: Alert) -> None:
        """Notify all subscribers of a new alert.

        Args:
            alert: New alert
        """
        with self._lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error notifying alert subscriber: {e}")
