"""Real-time monitoring example for Spark Resource Optimizer.

This example demonstrates the real-time monitoring functionality:
1. Setting up a SparkMonitor
2. Registering applications for monitoring
3. Receiving metrics and status updates
4. Setting up alerts for threshold violations

To run this example:
    python examples/monitoring_example.py

Note: This example uses simulated data. To test with a real cluster,
configure the metrics_endpoint and history_server_url parameters.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spark_optimizer.monitoring.monitor import (
    SparkMonitor,
    ApplicationStatus,
    MetricPoint,
)
from spark_optimizer.monitoring.alerts import (
    AlertManager,
    AlertRule,
    AlertSeverity,
)


def metrics_callback(metrics: dict):
    """Callback for receiving metrics updates."""
    print(f"  Received metrics update: {len(metrics)} data points")
    for name, value in list(metrics.items())[:3]:
        print(f"    - {name}: {value}")


def alert_callback(alert: dict):
    """Callback for receiving alerts."""
    print(f"\n  ALERT [{alert.get('severity', 'info').upper()}]: {alert.get('title', 'Unknown')}")
    print(f"    Message: {alert.get('message', 'No message')}")
    print(f"    App ID: {alert.get('app_id', 'Unknown')}")


def main():
    """Demonstrate real-time monitoring capabilities."""
    print("=" * 60)
    print("Spark Resource Optimizer - Real-Time Monitoring Example")
    print("=" * 60)

    # 1. Initialize the SparkMonitor
    print("\nStep 1: Initializing SparkMonitor...")

    # For demo, we use None for endpoints (simulated mode)
    # In production, configure these:
    #   metrics_endpoint="http://your-spark-cluster:4040/metrics/json"
    #   history_server_url="http://your-history-server:18080"
    monitor = SparkMonitor(
        metrics_endpoint=None,  # Set to your Spark metrics endpoint
        history_server_url=None,  # Set to your history server URL
        poll_interval=5.0,
    )
    print("  SparkMonitor initialized")

    # 2. Initialize the AlertManager
    print("\nStep 2: Setting up AlertManager...")
    alert_manager = AlertManager()

    # Add alert rules
    alert_manager.add_rule(AlertRule(
        rule_id="high-memory",
        name="High Memory Usage",
        condition="memory_percent > 80",
        severity=AlertSeverity.WARNING,
        threshold=80.0,
        metric_name="memory_percent",
        description="Alert when memory usage exceeds 80%",
    ))

    alert_manager.add_rule(AlertRule(
        rule_id="high-gc-time",
        name="High GC Time",
        condition="gc_time_percent > 10",
        severity=AlertSeverity.WARNING,
        threshold=10.0,
        metric_name="gc_time_percent",
        description="Alert when GC time exceeds 10% of total time",
    ))

    alert_manager.add_rule(AlertRule(
        rule_id="task-failures",
        name="Task Failures Detected",
        condition="failed_tasks > 5",
        severity=AlertSeverity.ERROR,
        threshold=5.0,
        metric_name="failed_tasks",
        description="Alert when more than 5 tasks have failed",
    ))

    print(f"  Added {len(alert_manager.get_rules())} alert rules")

    # 3. Register callback handlers
    print("\nStep 3: Registering callback handlers...")
    monitor.register_callback("metrics", metrics_callback)
    alert_manager.register_callback(alert_callback)
    print("  Callbacks registered")

    # 4. Simulate application monitoring
    print("\nStep 4: Simulating application monitoring...")

    # Create a simulated application status
    app_status = ApplicationStatus(
        app_id="app-20241210-001",
        app_name="ETL Pipeline - Sales Data",
        status="running",
        start_time=datetime.utcnow(),
        duration_seconds=0,
        progress=0.0,
        active_tasks=24,
        completed_tasks=0,
        failed_tasks=0,
        current_memory_mb=12288,
        current_cpu_percent=45.0,
        executors=8,
    )

    # Register the application
    monitor.register_application(app_status)
    print(f"  Registered application: {app_status.app_id}")
    print(f"    Name: {app_status.app_name}")
    print(f"    Executors: {app_status.executors}")
    print(f"    Memory: {app_status.current_memory_mb} MB")

    # 5. Simulate monitoring updates
    print("\nStep 5: Simulating monitoring updates (5 iterations)...")

    for i in range(5):
        print(f"\n  --- Update {i + 1}/5 ---")

        # Update application progress
        app_status.progress = min(1.0, app_status.progress + 0.2)
        app_status.completed_tasks += 30
        app_status.active_tasks = max(0, app_status.active_tasks - 5)
        app_status.duration_seconds += 60
        app_status.current_cpu_percent = 50 + (i * 10)  # Increasing CPU
        app_status.current_memory_mb = 12288 + (i * 2048)  # Increasing memory

        # Simulate some task failures on iteration 3
        if i == 3:
            app_status.failed_tasks = 7

        monitor.update_application(app_status)

        print(f"  Progress: {app_status.progress:.0%}")
        print(f"  CPU: {app_status.current_cpu_percent}%")
        print(f"  Memory: {app_status.current_memory_mb} MB")
        print(f"  Tasks: {app_status.active_tasks} active, {app_status.completed_tasks} completed, {app_status.failed_tasks} failed")

        # Record metrics
        metrics = {
            "cpu_percent": app_status.current_cpu_percent,
            "memory_percent": (app_status.current_memory_mb / 32768) * 100,  # Assume 32GB max
            "gc_time_percent": 5 + i * 3,  # Simulate increasing GC time
            "failed_tasks": app_status.failed_tasks,
        }

        # Check alerts
        for metric_name, value in metrics.items():
            alerts = alert_manager.check_metric(
                app_id=app_status.app_id,
                metric_name=metric_name,
                value=value,
            )
            for alert in alerts:
                alert_callback(alert.to_dict())

        time.sleep(1)  # Brief pause between updates

    # 6. Get final status
    print("\nStep 6: Final application status...")
    app_status.status = "completed"
    app_status.progress = 1.0
    monitor.update_application(app_status)

    final_status = monitor.get_application(app_status.app_id)
    if final_status:
        print(f"  Status: {final_status.status}")
        print(f"  Final Progress: {final_status.progress:.0%}")
        print(f"  Duration: {final_status.duration_seconds} seconds")
        print(f"  Total Tasks Completed: {final_status.completed_tasks}")
        print(f"  Failed Tasks: {final_status.failed_tasks}")

    # 7. Get alert summary
    print("\nStep 7: Alert summary...")
    active_alerts = alert_manager.get_active_alerts()
    print(f"  Total active alerts: {len(active_alerts)}")

    for alert in active_alerts[:5]:  # Show up to 5
        print(f"    - [{alert.severity.value}] {alert.title}")

    # 8. Display monitoring configuration for production
    print("\nStep 8: Production configuration example:")
    print("-" * 60)
    config_example = """
# SparkMonitor configuration for production:

monitor = SparkMonitor(
    # Spark UI metrics endpoint (usually port 4040)
    metrics_endpoint="http://spark-driver:4040/metrics/json",

    # Spark History Server URL
    history_server_url="http://history-server:18080",

    # Polling interval in seconds
    poll_interval=5.0,
)

# AlertManager with persistence:
alert_manager = AlertManager(
    db_url="postgresql://user:pass@localhost/spark_optimizer",
    cooldown_seconds=300,  # 5-minute alert cooldown
)
"""
    print(config_example)
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Monitoring example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
