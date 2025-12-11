"""Tests for real-time monitoring functionality."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import threading
import time

from spark_optimizer.monitoring.monitor import (
    SparkMonitor,
    MetricPoint,
    ApplicationStatus,
)


class TestMetricPoint:
    """Test the MetricPoint dataclass."""

    def test_metric_point_creation(self):
        """Test creating a metric point."""
        now = datetime.utcnow()
        point = MetricPoint(
            timestamp=now,
            name="cpu_usage",
            value=75.5,
            labels={"app_id": "app-123"},
        )

        assert point.timestamp == now
        assert point.name == "cpu_usage"
        assert point.value == 75.5
        assert point.labels == {"app_id": "app-123"}

    def test_metric_point_to_dict(self):
        """Test converting metric point to dictionary."""
        now = datetime.utcnow()
        point = MetricPoint(
            timestamp=now,
            name="memory_usage",
            value=1024.0,
        )

        result = point.to_dict()

        assert result["name"] == "memory_usage"
        assert result["value"] == 1024.0
        assert result["timestamp"] == now.isoformat()
        assert result["labels"] == {}

    def test_metric_point_default_labels(self):
        """Test that labels default to empty dict."""
        point = MetricPoint(
            timestamp=datetime.utcnow(),
            name="test",
            value=1.0,
        )

        assert point.labels == {}


class TestApplicationStatus:
    """Test the ApplicationStatus dataclass."""

    def test_application_status_creation(self):
        """Test creating an application status."""
        status = ApplicationStatus(
            app_id="app-123",
            app_name="test_job",
            status="running",
        )

        assert status.app_id == "app-123"
        assert status.app_name == "test_job"
        assert status.status == "running"
        assert status.progress == 0.0
        assert status.active_tasks == 0

    def test_application_status_to_dict(self):
        """Test converting application status to dictionary."""
        start_time = datetime.utcnow()
        status = ApplicationStatus(
            app_id="app-456",
            app_name="etl_job",
            status="running",
            start_time=start_time,
            progress=0.5,
            active_tasks=10,
            completed_tasks=50,
            executors=5,
        )

        result = status.to_dict()

        assert result["app_id"] == "app-456"
        assert result["app_name"] == "etl_job"
        assert result["status"] == "running"
        assert result["start_time"] == start_time.isoformat()
        assert result["progress"] == 0.5
        assert result["active_tasks"] == 10
        assert result["completed_tasks"] == 50
        assert result["executors"] == 5

    def test_application_status_default_values(self):
        """Test default values for application status."""
        status = ApplicationStatus(
            app_id="app-789",
            app_name="test",
            status="pending",
        )

        assert status.duration_seconds == 0
        assert status.progress == 0.0
        assert status.active_tasks == 0
        assert status.completed_tasks == 0
        assert status.failed_tasks == 0
        assert status.current_memory_mb == 0
        assert status.current_cpu_percent == 0
        assert status.metrics == {}


class TestSparkMonitor:
    """Test the SparkMonitor class."""

    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = SparkMonitor(
            metrics_endpoint="http://localhost:9090",
            history_server_url="http://localhost:18080",
            poll_interval=10.0,
        )

        assert monitor.metrics_endpoint == "http://localhost:9090"
        assert monitor.history_server_url == "http://localhost:18080"
        assert monitor.poll_interval == 10.0
        assert not monitor._running

    def test_monitor_default_initialization(self):
        """Test monitor with default values."""
        monitor = SparkMonitor()

        assert monitor.metrics_endpoint is None
        assert monitor.history_server_url is None
        assert monitor.poll_interval == 5.0

    def test_start_stop(self):
        """Test starting and stopping the monitor."""
        monitor = SparkMonitor()

        monitor.start()
        assert monitor._running
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()

        monitor.stop()
        assert not monitor._running
        # Give thread time to stop
        time.sleep(0.2)

    def test_start_idempotent(self):
        """Test that starting twice doesn't create multiple threads."""
        monitor = SparkMonitor()

        monitor.start()
        thread1 = monitor._monitor_thread

        monitor.start()
        thread2 = monitor._monitor_thread

        assert thread1 is thread2

        monitor.stop()

    def test_subscribe_unsubscribe(self):
        """Test subscribing and unsubscribing to events."""
        monitor = SparkMonitor()
        callback = Mock()

        monitor.subscribe(callback)
        assert callback in monitor._subscribers

        monitor.unsubscribe(callback)
        assert callback not in monitor._subscribers

    def test_add_application(self):
        """Test adding an application to monitor."""
        monitor = SparkMonitor()

        status = monitor.add_application("app-123", "test_job")

        assert status.app_id == "app-123"
        assert status.app_name == "test_job"
        assert status.status == "running"
        assert status.start_time is not None

    def test_add_application_idempotent(self):
        """Test that adding same app twice returns existing."""
        monitor = SparkMonitor()

        status1 = monitor.add_application("app-123", "test_job")
        status2 = monitor.add_application("app-123", "test_job_2")

        assert status1 is status2
        assert status1.app_name == "test_job"  # Original name preserved

    def test_add_application_notifies_subscribers(self):
        """Test that adding app notifies subscribers."""
        monitor = SparkMonitor()
        callback = Mock()
        monitor.subscribe(callback)

        monitor.add_application("app-123", "test_job")

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "application_started"
        assert data["app_id"] == "app-123"

    def test_get_applications(self):
        """Test getting all monitored applications."""
        monitor = SparkMonitor()

        monitor.add_application("app-1", "job1")
        monitor.add_application("app-2", "job2")
        monitor.add_application("app-3", "job3")

        apps = monitor.get_applications()

        assert len(apps) == 3
        app_ids = [app.app_id for app in apps]
        assert "app-1" in app_ids
        assert "app-2" in app_ids
        assert "app-3" in app_ids

    def test_get_application(self):
        """Test getting a specific application."""
        monitor = SparkMonitor()

        monitor.add_application("app-123", "test_job")

        app = monitor.get_application("app-123")
        assert app is not None
        assert app.app_id == "app-123"

        missing = monitor.get_application("app-999")
        assert missing is None

    def test_update_metrics(self):
        """Test updating metrics for an application."""
        monitor = SparkMonitor()
        callback = Mock()
        monitor.subscribe(callback)

        monitor.add_application("app-123", "test_job")
        callback.reset_mock()

        monitor.update_metrics(
            "app-123",
            {
                "cpu_usage": 75.0,
                "memory_usage": 8192.0,
            },
        )

        app = monitor.get_application("app-123")
        assert app.metrics["cpu_usage"] == 75.0
        assert app.metrics["memory_usage"] == 8192.0

        callback.assert_called_once()
        event_type, data = callback.call_args[0]
        assert event_type == "metrics_updated"
        assert data["app_id"] == "app-123"

    def test_update_metrics_unknown_app(self):
        """Test that updating metrics for unknown app is no-op."""
        monitor = SparkMonitor()
        callback = Mock()
        monitor.subscribe(callback)

        monitor.update_metrics("app-unknown", {"cpu": 50.0})

        callback.assert_not_called()

    def test_update_status(self):
        """Test updating application status."""
        monitor = SparkMonitor()
        callback = Mock()
        monitor.subscribe(callback)

        monitor.add_application("app-123", "test_job")
        callback.reset_mock()

        monitor.update_status(
            "app-123",
            status="running",
            progress=0.5,
            active_tasks=10,
            completed_tasks=50,
            current_memory_mb=4096,
            executors=5,
        )

        app = monitor.get_application("app-123")
        assert app.progress == 0.5
        assert app.active_tasks == 10
        assert app.completed_tasks == 50
        assert app.current_memory_mb == 4096
        assert app.executors == 5

    def test_update_status_change_notification(self):
        """Test that status changes notify subscribers."""
        monitor = SparkMonitor()
        callback = Mock()
        monitor.subscribe(callback)

        monitor.add_application("app-123", "test_job")
        callback.reset_mock()

        monitor.update_status("app-123", status="completed")

        # Should have two calls: status_changed and application_updated
        assert callback.call_count == 2
        calls = [call[0] for call in callback.call_args_list]
        event_types = [call[0] for call in calls]
        assert "status_changed" in event_types
        assert "application_updated" in event_types

    def test_get_metric_history(self):
        """Test getting metric history."""
        monitor = SparkMonitor()

        monitor.add_application("app-123", "test_job")
        monitor.update_metrics("app-123", {"cpu": 50.0})
        monitor.update_metrics("app-123", {"cpu": 60.0})
        monitor.update_metrics("app-123", {"cpu": 70.0})

        history = monitor.get_metric_history("app-123", "cpu")

        assert len(history) == 3
        assert history[0].value == 50.0
        assert history[1].value == 60.0
        assert history[2].value == 70.0

    def test_get_metric_history_with_since(self):
        """Test getting metric history with time filter."""
        monitor = SparkMonitor()

        monitor.add_application("app-123", "test_job")
        monitor.update_metrics("app-123", {"cpu": 50.0})

        # Get history from future (should be empty)
        future = datetime.utcnow() + timedelta(hours=1)
        history = monitor.get_metric_history("app-123", "cpu", since=future)

        assert len(history) == 0

    def test_trim_history_by_count(self):
        """Test that history is trimmed by count."""
        monitor = SparkMonitor()
        monitor._max_history_points = 5

        monitor.add_application("app-123", "test_job")

        # Add more than max points
        for i in range(10):
            monitor.update_metrics("app-123", {"cpu": float(i)})

        history = monitor.get_metric_history("app-123", "cpu")
        assert len(history) <= 5

    def test_subscriber_error_handling(self):
        """Test that subscriber errors don't crash monitor."""
        monitor = SparkMonitor()

        def bad_callback(event_type, data):
            raise Exception("Callback error")

        good_callback = Mock()

        monitor.subscribe(bad_callback)
        monitor.subscribe(good_callback)

        # Should not raise, and good callback should still be called
        monitor.add_application("app-123", "test_job")

        good_callback.assert_called_once()

    def test_update_status_duration_calculation(self):
        """Test that duration is calculated when start_time is set."""
        monitor = SparkMonitor()

        status = monitor.add_application("app-123", "test_job")
        start_time = status.start_time

        # Small sleep to ensure duration > 0
        time.sleep(0.1)

        monitor.update_status("app-123", progress=0.5)

        app = monitor.get_application("app-123")
        assert app.duration_seconds > 0


class TestSparkMonitorPolling:
    """Test polling functionality with mocked HTTP."""

    @patch("requests.get")
    def test_poll_prometheus_success(self, mock_get):
        """Test successful Prometheus polling."""
        monitor = SparkMonitor(metrics_endpoint="http://localhost:9090")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "result": [
                    {
                        "metric": {"app_id": "app-123", "app_name": "test_job"},
                        "value": [1234567890, "5"],
                    }
                ]
            }
        }
        mock_get.return_value = mock_response

        monitor._poll_prometheus()

        apps = monitor.get_applications()
        assert len(apps) == 1
        assert apps[0].app_id == "app-123"
        assert apps[0].executors == 5

    @patch("requests.get")
    def test_poll_prometheus_error(self, mock_get):
        """Test Prometheus polling handles errors gracefully."""
        monitor = SparkMonitor(metrics_endpoint="http://localhost:9090")

        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        # Should not raise
        monitor._poll_prometheus()

        # No apps should be added
        assert len(monitor.get_applications()) == 0

    @patch("requests.get")
    def test_poll_history_server_success(self, mock_get):
        """Test successful History Server polling."""
        monitor = SparkMonitor(history_server_url="http://localhost:18080")

        # First response for listing apps
        list_response = Mock()
        list_response.status_code = 200
        list_response.json.return_value = [{"id": "app-123", "name": "test_job"}]

        # Second response for app details
        detail_response = Mock()
        detail_response.status_code = 200
        detail_response.json.return_value = {"attempts": [{"completed": False}]}

        mock_get.side_effect = [list_response, detail_response]

        monitor._poll_history_server()

        apps = monitor.get_applications()
        assert len(apps) == 1
        assert apps[0].app_id == "app-123"
        assert apps[0].status == "running"

    @patch("requests.get")
    def test_poll_history_server_completed_app(self, mock_get):
        """Test History Server polling with completed app."""
        monitor = SparkMonitor(history_server_url="http://localhost:18080")

        list_response = Mock()
        list_response.status_code = 200
        list_response.json.return_value = [{"id": "app-456", "name": "completed_job"}]

        detail_response = Mock()
        detail_response.status_code = 200
        detail_response.json.return_value = {"attempts": [{"completed": True}]}

        mock_get.side_effect = [list_response, detail_response]

        monitor._poll_history_server()

        apps = monitor.get_applications()
        assert len(apps) == 1
        assert apps[0].status == "completed"
