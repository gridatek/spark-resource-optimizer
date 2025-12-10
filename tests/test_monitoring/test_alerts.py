"""Tests for alert management functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock

from spark_optimizer.monitoring.alerts import (
    AlertManager,
    AlertRule,
    Alert,
    AlertSeverity,
)


class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_severity_values(self):
        """Test that severity enum has expected values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = Alert(
            id="alert-1",
            app_id="app-123",
            severity=AlertSeverity.WARNING,
            title="High GC Time",
            message="GC time is 15%",
        )

        assert alert.id == "alert-1"
        assert alert.app_id == "app-123"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "High GC Time"
        assert not alert.acknowledged
        assert not alert.resolved

    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        now = datetime.utcnow()
        alert = Alert(
            id="alert-1",
            app_id="app-123",
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            message="Test message",
            metric_name="gc_time",
            metric_value=25.0,
            threshold=10.0,
            created_at=now,
        )

        result = alert.to_dict()

        assert result["id"] == "alert-1"
        assert result["severity"] == "error"
        assert result["metric_name"] == "gc_time"
        assert result["metric_value"] == 25.0
        assert result["threshold"] == 10.0
        assert result["created_at"] == now.isoformat()
        assert result["acknowledged"] is False
        assert result["resolved"] is False


class TestAlertRule:
    """Test AlertRule dataclass."""

    def test_rule_creation(self):
        """Test creating an alert rule."""
        rule = AlertRule(
            name="high_gc",
            metric_name="gc_time_percent",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="High GC Time",
            message_template="GC time is {value}%",
        )

        assert rule.name == "high_gc"
        assert rule.condition == "gt"
        assert rule.threshold == 10.0
        assert rule.enabled

    def test_rule_evaluate_gt(self):
        """Test evaluating greater than condition."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )

        assert rule.evaluate(15.0) is True
        assert rule.evaluate(10.0) is False
        assert rule.evaluate(5.0) is False

    def test_rule_evaluate_lt(self):
        """Test evaluating less than condition."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="lt",
            threshold=30.0,
            severity=AlertSeverity.INFO,
            title_template="Test",
            message_template="Test",
        )

        assert rule.evaluate(20.0) is True
        assert rule.evaluate(30.0) is False
        assert rule.evaluate(40.0) is False

    def test_rule_evaluate_gte(self):
        """Test evaluating greater than or equal condition."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gte",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )

        assert rule.evaluate(15.0) is True
        assert rule.evaluate(10.0) is True
        assert rule.evaluate(5.0) is False

    def test_rule_evaluate_lte(self):
        """Test evaluating less than or equal condition."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="lte",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )

        assert rule.evaluate(5.0) is True
        assert rule.evaluate(10.0) is True
        assert rule.evaluate(15.0) is False

    def test_rule_evaluate_eq(self):
        """Test evaluating equal condition."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="eq",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )

        assert rule.evaluate(100.0) is True
        assert rule.evaluate(99.0) is False

    def test_rule_evaluate_neq(self):
        """Test evaluating not equal condition."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="neq",
            threshold=0,
            severity=AlertSeverity.ERROR,
            title_template="Test",
            message_template="Test",
        )

        assert rule.evaluate(1) is True
        assert rule.evaluate(0) is False

    def test_rule_evaluate_unknown_condition(self):
        """Test evaluating unknown condition returns False."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="unknown",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )

        assert rule.evaluate(15.0) is False


class TestAlertManager:
    """Test AlertManager class."""

    def test_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()

        assert len(manager._rules) == len(AlertManager.DEFAULT_RULES)

    def test_manager_custom_rules(self):
        """Test manager with custom rules."""
        custom_rules = [
            AlertRule(
                name="custom",
                metric_name="custom_metric",
                condition="gt",
                threshold=50.0,
                severity=AlertSeverity.WARNING,
                title_template="Custom Alert",
                message_template="Value: {value}",
            )
        ]

        manager = AlertManager(rules=custom_rules)

        assert len(manager._rules) == 1
        assert manager._rules[0].name == "custom"

    def test_add_rule(self):
        """Test adding a rule."""
        manager = AlertManager(rules=[])

        rule = AlertRule(
            name="new_rule",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )

        manager.add_rule(rule)

        rules = manager.get_rules()
        assert len(rules) == 1
        assert rules[0].name == "new_rule"

    def test_remove_rule(self):
        """Test removing a rule."""
        rule = AlertRule(
            name="test_rule",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )
        manager = AlertManager(rules=[rule])

        result = manager.remove_rule("test_rule")
        assert result is True
        assert len(manager.get_rules()) == 0

        # Removing non-existent rule
        result = manager.remove_rule("nonexistent")
        assert result is False

    def test_subscribe_unsubscribe(self):
        """Test subscribing and unsubscribing to alerts."""
        manager = AlertManager()
        callback = Mock()

        manager.subscribe(callback)
        assert callback in manager._subscribers

        manager.unsubscribe(callback)
        assert callback not in manager._subscribers

    def test_evaluate_metrics_triggers_alert(self):
        """Test that metrics evaluation triggers alerts."""
        rule = AlertRule(
            name="high_gc",
            metric_name="gc_time_percent",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="High GC Time",
            message_template="GC is {value:.1f}%",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"gc_time_percent": 15.0})

        assert len(alerts) == 1
        assert alerts[0].app_id == "app-123"
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "15.0" in alerts[0].message

    def test_evaluate_metrics_no_trigger(self):
        """Test that metrics within threshold don't trigger alerts."""
        rule = AlertRule(
            name="high_gc",
            metric_name="gc_time_percent",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"gc_time_percent": 5.0})

        assert len(alerts) == 0

    def test_evaluate_metrics_disabled_rule(self):
        """Test that disabled rules don't trigger alerts."""
        rule = AlertRule(
            name="disabled_rule",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            enabled=False,
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"test": 100.0})

        assert len(alerts) == 0

    def test_evaluate_metrics_cooldown(self):
        """Test that cooldown prevents repeated alerts."""
        rule = AlertRule(
            name="high_gc",
            metric_name="gc_time_percent",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=5,
        )
        manager = AlertManager(rules=[rule])

        # First evaluation triggers alert
        alerts1 = manager.evaluate_metrics("app-123", {"gc_time_percent": 15.0})
        assert len(alerts1) == 1

        # Second evaluation within cooldown doesn't trigger
        alerts2 = manager.evaluate_metrics("app-123", {"gc_time_percent": 20.0})
        assert len(alerts2) == 0

    def test_evaluate_metrics_notifies_subscribers(self):
        """Test that subscribers are notified of new alerts."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])
        callback = Mock()
        manager.subscribe(callback)

        manager.evaluate_metrics("app-123", {"test": 15.0})

        callback.assert_called_once()
        alert = callback.call_args[0][0]
        assert alert.app_id == "app-123"

    def test_get_active_alerts(self):
        """Test getting active alerts."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        manager.evaluate_metrics("app-1", {"test": 15.0})
        manager.evaluate_metrics("app-2", {"test": 20.0})

        active = manager.get_active_alerts()
        assert len(active) == 2

        # Filter by app_id
        active_app1 = manager.get_active_alerts(app_id="app-1")
        assert len(active_app1) == 1
        assert active_app1[0].app_id == "app-1"

    def test_get_alert(self):
        """Test getting a specific alert."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"test": 15.0})
        alert_id = alerts[0].id

        retrieved = manager.get_alert(alert_id)
        assert retrieved is not None
        assert retrieved.id == alert_id

        missing = manager.get_alert("nonexistent")
        assert missing is None

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"test": 15.0})
        alert_id = alerts[0].id

        result = manager.acknowledge_alert(alert_id, acknowledged_by="admin")

        assert result is True
        alert = manager.get_alert(alert_id)
        assert alert.acknowledged is True
        assert alert.acknowledged_by == "admin"
        assert alert.acknowledged_at is not None

    def test_acknowledge_alert_idempotent(self):
        """Test that acknowledging twice doesn't change anything."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"test": 15.0})
        alert_id = alerts[0].id

        manager.acknowledge_alert(alert_id)
        result = manager.acknowledge_alert(alert_id)

        assert result is False  # Already acknowledged

    def test_acknowledge_nonexistent_alert(self):
        """Test acknowledging a nonexistent alert."""
        manager = AlertManager()

        result = manager.acknowledge_alert("nonexistent")

        assert result is False

    def test_resolve_alert(self):
        """Test resolving an alert."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"test": 15.0})
        alert_id = alerts[0].id

        result = manager.resolve_alert(alert_id)

        assert result is True
        alert = manager.get_alert(alert_id)
        assert alert.resolved is True
        assert alert.resolved_at is not None

        # Should not appear in active alerts
        active = manager.get_active_alerts()
        assert len(active) == 0

    def test_resolve_alert_idempotent(self):
        """Test that resolving twice doesn't change anything."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        alerts = manager.evaluate_metrics("app-123", {"test": 15.0})
        alert_id = alerts[0].id

        manager.resolve_alert(alert_id)
        result = manager.resolve_alert(alert_id)

        assert result is False  # Already resolved

    def test_get_alert_history(self):
        """Test getting alert history."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        # Create and resolve some alerts
        alerts1 = manager.evaluate_metrics("app-1", {"test": 15.0})
        alerts2 = manager.evaluate_metrics("app-2", {"test": 20.0})

        manager.resolve_alert(alerts1[0].id)
        manager.resolve_alert(alerts2[0].id)

        history = manager.get_alert_history()
        assert len(history) == 2

        # Filter by app_id
        history_app1 = manager.get_alert_history(app_id="app-1")
        assert len(history_app1) == 1

    def test_get_alert_history_with_limit(self):
        """Test alert history with limit."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        # Create and resolve multiple alerts
        for i in range(5):
            alerts = manager.evaluate_metrics(f"app-{i}", {"test": 15.0})
            manager.resolve_alert(alerts[0].id)

        history = manager.get_alert_history(limit=3)
        assert len(history) == 3

    def test_subscriber_error_handling(self):
        """Test that subscriber errors don't crash manager."""
        rule = AlertRule(
            name="test",
            metric_name="test",
            condition="gt",
            threshold=10.0,
            severity=AlertSeverity.WARNING,
            title_template="Test",
            message_template="Test",
            cooldown_minutes=0,
        )
        manager = AlertManager(rules=[rule])

        def bad_callback(alert):
            raise Exception("Callback error")

        good_callback = Mock()

        manager.subscribe(bad_callback)
        manager.subscribe(good_callback)

        # Should not raise
        alerts = manager.evaluate_metrics("app-123", {"test": 15.0})

        assert len(alerts) == 1
        good_callback.assert_called_once()


class TestDefaultAlertRules:
    """Test default alert rules."""

    def test_default_rules_exist(self):
        """Test that default rules are defined."""
        assert len(AlertManager.DEFAULT_RULES) > 0

    def test_gc_time_rules(self):
        """Test GC time alert rules."""
        manager = AlertManager()

        # Warning level GC
        alerts = manager.evaluate_metrics("app-1", {"jvm_gc_time_percent": 15.0})
        warning_alerts = [a for a in alerts if a.severity == AlertSeverity.WARNING]
        assert len(warning_alerts) >= 1

        # Critical level GC
        alerts = manager.evaluate_metrics("app-2", {"jvm_gc_time_percent": 30.0})
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) >= 1

    def test_memory_spill_rule(self):
        """Test memory spill alert rule."""
        manager = AlertManager()

        alerts = manager.evaluate_metrics("app-1", {"memory_spill_ratio": 0.2})

        assert len(alerts) >= 1
        spill_alerts = [a for a in alerts if "spill" in a.title.lower() or "memory" in a.title.lower()]
        assert len(spill_alerts) >= 1

    def test_task_failure_rule(self):
        """Test task failure rate alert rule."""
        manager = AlertManager()

        alerts = manager.evaluate_metrics("app-1", {"task_failure_rate": 0.1})

        assert len(alerts) >= 1
        failure_alerts = [a for a in alerts if "failure" in a.title.lower()]
        assert len(failure_alerts) >= 1

    def test_executor_lost_rule(self):
        """Test executor lost alert rule."""
        manager = AlertManager()

        alerts = manager.evaluate_metrics("app-1", {"executors_lost": 2})

        assert len(alerts) >= 1
        lost_alerts = [a for a in alerts if "executor" in a.title.lower()]
        assert len(lost_alerts) >= 1
