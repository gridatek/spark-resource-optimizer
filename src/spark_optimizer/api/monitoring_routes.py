"""API routes for real-time monitoring."""

from flask import Blueprint, request, jsonify, g
from typing import Optional
import logging

from spark_optimizer.monitoring import SparkMonitor, AlertManager, WebSocketServer

logger = logging.getLogger(__name__)

monitoring_bp = Blueprint("monitoring", __name__, url_prefix="/monitoring")

# Global monitor and websocket instances (initialized by server)
_monitor: Optional[SparkMonitor] = None
_websocket_server: Optional[WebSocketServer] = None
_alert_manager: Optional[AlertManager] = None


def init_monitoring(
    metrics_endpoint: Optional[str] = None,
    history_server_url: Optional[str] = None,
    websocket_port: int = 8765,
) -> None:
    """Initialize monitoring components.

    Args:
        metrics_endpoint: Prometheus metrics endpoint
        history_server_url: Spark History Server URL
        websocket_port: WebSocket server port
    """
    global _monitor, _websocket_server, _alert_manager

    # Initialize monitor
    _monitor = SparkMonitor(
        metrics_endpoint=metrics_endpoint,
        history_server_url=history_server_url,
    )

    # Initialize alert manager
    _alert_manager = AlertManager()

    # Initialize WebSocket server
    _websocket_server = WebSocketServer(port=websocket_port)

    # Connect components
    def on_monitor_event(event_type: str, data: dict) -> None:
        """Forward monitor events to WebSocket clients."""
        if _websocket_server:
            _websocket_server.broadcast(event_type, data)

        # Evaluate alerts on metrics updates
        if event_type == "metrics_updated" and _alert_manager:
            app_id = data.get("app_id")
            metrics = data.get("metrics", {})
            if app_id and metrics:
                alerts = _alert_manager.evaluate_metrics(app_id, metrics)
                for alert in alerts:
                    if _websocket_server:
                        _websocket_server.broadcast("alert", alert.to_dict())

    _monitor.subscribe(on_monitor_event)


def start_monitoring() -> None:
    """Start monitoring services."""
    if _monitor:
        _monitor.start()
    if _websocket_server:
        _websocket_server.start()


def stop_monitoring() -> None:
    """Stop monitoring services."""
    if _monitor:
        _monitor.stop()
    if _websocket_server:
        _websocket_server.stop()


def get_monitor() -> Optional[SparkMonitor]:
    """Get the monitor instance."""
    return _monitor


def get_alert_manager() -> Optional[AlertManager]:
    """Get the alert manager instance."""
    return _alert_manager


@monitoring_bp.route("/status", methods=["GET"])
def get_monitoring_status():
    """Get monitoring service status.

    Returns:
        JSON response with monitoring status
    """
    monitor = get_monitor()

    return (
        jsonify(
            {
                "monitoring_active": (
                    monitor is not None and monitor._running if monitor else False
                ),
                "websocket_active": (
                    _websocket_server is not None and _websocket_server._running
                    if _websocket_server
                    else False
                ),
                "websocket_port": _websocket_server.port if _websocket_server else None,
                "connected_clients": (
                    _websocket_server.client_count if _websocket_server else 0
                ),
            }
        ),
        200,
    )


@monitoring_bp.route("/applications", methods=["GET"])
def list_monitored_applications():
    """List all currently monitored applications.

    Returns:
        JSON response with list of applications
    """
    monitor = get_monitor()

    if not monitor:
        return jsonify({"error": "Monitoring not initialized"}), 503

    apps = monitor.get_applications()

    return (
        jsonify(
            {
                "applications": [app.to_dict() for app in apps],
                "count": len(apps),
            }
        ),
        200,
    )


@monitoring_bp.route("/applications/<app_id>", methods=["GET"])
def get_application_status(app_id: str):
    """Get real-time status for a specific application.

    Args:
        app_id: Spark application ID

    Returns:
        JSON response with application status
    """
    monitor = get_monitor()

    if not monitor:
        return jsonify({"error": "Monitoring not initialized"}), 503

    app = monitor.get_application(app_id)

    if not app:
        return jsonify({"error": "Application not found or not being monitored"}), 404

    return jsonify(app.to_dict()), 200


@monitoring_bp.route("/applications/<app_id>/metrics", methods=["GET"])
def get_application_metrics(app_id: str):
    """Get metric history for an application.

    Query parameters:
        - metric: Metric name (required)
        - since: ISO timestamp to get metrics since (optional)

    Args:
        app_id: Spark application ID

    Returns:
        JSON response with metric history
    """
    monitor = get_monitor()

    if not monitor:
        return jsonify({"error": "Monitoring not initialized"}), 503

    metric_name = request.args.get("metric")
    since_str = request.args.get("since")

    if not metric_name:
        return jsonify({"error": "Missing required parameter: metric"}), 400

    since = None
    if since_str:
        from datetime import datetime

        try:
            since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "Invalid timestamp format"}), 400

    history = monitor.get_metric_history(app_id, metric_name, since=since)

    return (
        jsonify(
            {
                "app_id": app_id,
                "metric": metric_name,
                "points": [p.to_dict() for p in history],
                "count": len(history),
            }
        ),
        200,
    )


@monitoring_bp.route("/applications/<app_id>/track", methods=["POST"])
def track_application(app_id: str):
    """Start tracking an application.

    Request body:
        {
            "app_name": str (optional)
        }

    Args:
        app_id: Spark application ID

    Returns:
        JSON response confirming tracking started
    """
    monitor = get_monitor()

    if not monitor:
        return jsonify({"error": "Monitoring not initialized"}), 503

    data = request.get_json() or {}
    app_name = data.get("app_name", app_id)

    status = monitor.add_application(app_id, app_name)

    return (
        jsonify(
            {
                "status": "tracking",
                "application": status.to_dict(),
            }
        ),
        200,
    )


@monitoring_bp.route("/applications/<app_id>/metrics", methods=["POST"])
def push_metrics(app_id: str):
    """Push metrics for an application.

    Request body:
        {
            "metrics": {"metric_name": value, ...}
        }

    Args:
        app_id: Spark application ID

    Returns:
        JSON response confirming metrics received
    """
    monitor = get_monitor()

    if not monitor:
        return jsonify({"error": "Monitoring not initialized"}), 503

    data = request.get_json()

    if not data or "metrics" not in data:
        return jsonify({"error": "Missing required field: metrics"}), 400

    metrics = data["metrics"]

    if not isinstance(metrics, dict):
        return jsonify({"error": "metrics must be a dictionary"}), 400

    # Ensure app is being tracked
    if not monitor.get_application(app_id):
        monitor.add_application(app_id, app_id)

    monitor.update_metrics(app_id, metrics)

    return (
        jsonify(
            {
                "status": "received",
                "app_id": app_id,
                "metrics_count": len(metrics),
            }
        ),
        200,
    )


@monitoring_bp.route("/applications/<app_id>/status", methods=["PUT"])
def update_application_status(app_id: str):
    """Update status for an application.

    Request body:
        {
            "status": str (optional),
            "progress": float (optional),
            "active_tasks": int (optional),
            "completed_tasks": int (optional),
            "failed_tasks": int (optional),
            ...
        }

    Args:
        app_id: Spark application ID

    Returns:
        JSON response confirming status updated
    """
    monitor = get_monitor()

    if not monitor:
        return jsonify({"error": "Monitoring not initialized"}), 503

    data = request.get_json() or {}

    if not monitor.get_application(app_id):
        return jsonify({"error": "Application not found or not being monitored"}), 404

    monitor.update_status(
        app_id,
        status=data.get("status"),
        progress=data.get("progress"),
        active_tasks=data.get("active_tasks"),
        completed_tasks=data.get("completed_tasks"),
        failed_tasks=data.get("failed_tasks"),
        active_stages=data.get("active_stages"),
        completed_stages=data.get("completed_stages"),
        current_memory_mb=data.get("current_memory_mb"),
        current_cpu_percent=data.get("current_cpu_percent"),
        executors=data.get("executors"),
    )

    return (
        jsonify(
            {
                "status": "updated",
                "app_id": app_id,
            }
        ),
        200,
    )


@monitoring_bp.route("/alerts", methods=["GET"])
def list_alerts():
    """List active alerts.

    Query parameters:
        - app_id: Filter by application ID (optional)

    Returns:
        JSON response with list of alerts
    """
    alert_manager = get_alert_manager()

    if not alert_manager:
        return jsonify({"error": "Alert manager not initialized"}), 503

    app_id = request.args.get("app_id")

    alerts = alert_manager.get_active_alerts(app_id=app_id)

    return (
        jsonify(
            {
                "alerts": [alert.to_dict() for alert in alerts],
                "count": len(alerts),
            }
        ),
        200,
    )


@monitoring_bp.route("/alerts/<alert_id>", methods=["GET"])
def get_alert(alert_id: str):
    """Get a specific alert.

    Args:
        alert_id: Alert ID

    Returns:
        JSON response with alert details
    """
    alert_manager = get_alert_manager()

    if not alert_manager:
        return jsonify({"error": "Alert manager not initialized"}), 503

    alert = alert_manager.get_alert(alert_id)

    if not alert:
        return jsonify({"error": "Alert not found"}), 404

    return jsonify(alert.to_dict()), 200


@monitoring_bp.route("/alerts/<alert_id>/acknowledge", methods=["POST"])
def acknowledge_alert(alert_id: str):
    """Acknowledge an alert.

    Request body:
        {
            "acknowledged_by": str (optional)
        }

    Args:
        alert_id: Alert ID

    Returns:
        JSON response confirming acknowledgement
    """
    alert_manager = get_alert_manager()

    if not alert_manager:
        return jsonify({"error": "Alert manager not initialized"}), 503

    data = request.get_json() or {}
    acknowledged_by = data.get("acknowledged_by", "api")

    if alert_manager.acknowledge_alert(alert_id, acknowledged_by):
        return (
            jsonify(
                {
                    "status": "acknowledged",
                    "alert_id": alert_id,
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "Alert not found or already acknowledged"}), 404


@monitoring_bp.route("/alerts/<alert_id>/resolve", methods=["POST"])
def resolve_alert(alert_id: str):
    """Resolve an alert.

    Args:
        alert_id: Alert ID

    Returns:
        JSON response confirming resolution
    """
    alert_manager = get_alert_manager()

    if not alert_manager:
        return jsonify({"error": "Alert manager not initialized"}), 503

    if alert_manager.resolve_alert(alert_id):
        return (
            jsonify(
                {
                    "status": "resolved",
                    "alert_id": alert_id,
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "Alert not found or already resolved"}), 404


@monitoring_bp.route("/alerts/history", methods=["GET"])
def get_alert_history():
    """Get alert history.

    Query parameters:
        - app_id: Filter by application ID (optional)
        - since: ISO timestamp to get alerts since (optional)
        - limit: Maximum number of alerts to return (default: 100)

    Returns:
        JSON response with alert history
    """
    alert_manager = get_alert_manager()

    if not alert_manager:
        return jsonify({"error": "Alert manager not initialized"}), 503

    app_id = request.args.get("app_id")
    since_str = request.args.get("since")
    limit = request.args.get("limit", 100, type=int)

    since = None
    if since_str:
        from datetime import datetime

        try:
            since = datetime.fromisoformat(since_str.replace("Z", "+00:00"))
        except ValueError:
            return jsonify({"error": "Invalid timestamp format"}), 400

    history = alert_manager.get_alert_history(app_id=app_id, since=since, limit=limit)

    return (
        jsonify(
            {
                "alerts": [alert.to_dict() for alert in history],
                "count": len(history),
            }
        ),
        200,
    )


@monitoring_bp.route("/alerts/rules", methods=["GET"])
def list_alert_rules():
    """List configured alert rules.

    Returns:
        JSON response with list of rules
    """
    alert_manager = get_alert_manager()

    if not alert_manager:
        return jsonify({"error": "Alert manager not initialized"}), 503

    rules = alert_manager.get_rules()

    return (
        jsonify(
            {
                "rules": [
                    {
                        "name": rule.name,
                        "metric_name": rule.metric_name,
                        "condition": rule.condition,
                        "threshold": rule.threshold,
                        "severity": rule.severity.value,
                        "cooldown_minutes": rule.cooldown_minutes,
                        "enabled": rule.enabled,
                    }
                    for rule in rules
                ],
                "count": len(rules),
            }
        ),
        200,
    )
