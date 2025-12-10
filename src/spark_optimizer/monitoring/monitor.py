"""Real-time monitoring for Spark applications."""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
import threading
import time
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """A single metric data point."""

    timestamp: datetime
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
        }


@dataclass
class ApplicationStatus:
    """Status of a monitored Spark application."""

    app_id: str
    app_name: str
    status: str  # running, completed, failed
    start_time: Optional[datetime] = None
    duration_seconds: float = 0
    progress: float = 0.0  # 0.0 to 1.0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    active_stages: int = 0
    completed_stages: int = 0
    current_memory_mb: float = 0
    current_cpu_percent: float = 0
    executors: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "app_id": self.app_id,
            "app_name": self.app_name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "duration_seconds": self.duration_seconds,
            "progress": self.progress,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "active_stages": self.active_stages,
            "completed_stages": self.completed_stages,
            "current_memory_mb": self.current_memory_mb,
            "current_cpu_percent": self.current_cpu_percent,
            "executors": self.executors,
            "metrics": self.metrics,
            "last_updated": self.last_updated.isoformat(),
        }


class SparkMonitor:
    """Real-time monitor for Spark applications.

    Provides streaming metrics and status updates for running Spark jobs.
    """

    def __init__(
        self,
        metrics_endpoint: Optional[str] = None,
        history_server_url: Optional[str] = None,
        poll_interval: float = 5.0,
    ):
        """Initialize the Spark monitor.

        Args:
            metrics_endpoint: Prometheus metrics endpoint URL
            history_server_url: Spark History Server URL
            poll_interval: Polling interval in seconds
        """
        self.metrics_endpoint = metrics_endpoint
        self.history_server_url = history_server_url
        self.poll_interval = poll_interval

        self._applications: Dict[str, ApplicationStatus] = {}
        self._metric_history: Dict[str, List[MetricPoint]] = {}
        self._subscribers: List[Callable[[str, Dict], None]] = []
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Metric retention settings
        self._max_history_points = 1000
        self._max_history_age = timedelta(hours=1)

    def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Spark monitor started")

    def stop(self) -> None:
        """Stop the monitoring loop."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        logger.info("Spark monitor stopped")

    def subscribe(self, callback: Callable[[str, Dict], None]) -> None:
        """Subscribe to monitoring events.

        Args:
            callback: Function to call with (event_type, data) for each event
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[str, Dict], None]) -> None:
        """Unsubscribe from monitoring events.

        Args:
            callback: Previously subscribed callback function
        """
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def get_applications(self) -> List[ApplicationStatus]:
        """Get all monitored applications.

        Returns:
            List of application statuses
        """
        with self._lock:
            return list(self._applications.values())

    def get_application(self, app_id: str) -> Optional[ApplicationStatus]:
        """Get status for a specific application.

        Args:
            app_id: Spark application ID

        Returns:
            Application status or None if not found
        """
        with self._lock:
            return self._applications.get(app_id)

    def get_metric_history(
        self,
        app_id: str,
        metric_name: str,
        since: Optional[datetime] = None,
    ) -> List[MetricPoint]:
        """Get historical metric values for an application.

        Args:
            app_id: Spark application ID
            metric_name: Name of the metric
            since: Only return points after this time

        Returns:
            List of metric points
        """
        key = f"{app_id}:{metric_name}"
        with self._lock:
            history = self._metric_history.get(key, [])
            if since:
                history = [p for p in history if p.timestamp > since]
            return history

    def add_application(self, app_id: str, app_name: str) -> ApplicationStatus:
        """Manually add an application to monitor.

        Args:
            app_id: Spark application ID
            app_name: Application name

        Returns:
            The created application status
        """
        with self._lock:
            if app_id not in self._applications:
                status = ApplicationStatus(
                    app_id=app_id,
                    app_name=app_name,
                    status="running",
                    start_time=datetime.utcnow(),
                )
                self._applications[app_id] = status
                self._notify("application_started", status.to_dict())
            return self._applications[app_id]

    def update_metrics(self, app_id: str, metrics: Dict[str, float]) -> None:
        """Update metrics for an application.

        Args:
            app_id: Spark application ID
            metrics: Dictionary of metric name to value
        """
        now = datetime.utcnow()

        with self._lock:
            if app_id not in self._applications:
                return

            app = self._applications[app_id]
            app.metrics.update(metrics)
            app.last_updated = now

            # Store metric history
            for name, value in metrics.items():
                key = f"{app_id}:{name}"
                point = MetricPoint(timestamp=now, name=name, value=value)

                if key not in self._metric_history:
                    self._metric_history[key] = []

                self._metric_history[key].append(point)

                # Trim history
                self._trim_history(key)

            # Notify subscribers
            self._notify(
                "metrics_updated",
                {"app_id": app_id, "metrics": metrics, "timestamp": now.isoformat()},
            )

    def update_status(
        self,
        app_id: str,
        status: Optional[str] = None,
        progress: Optional[float] = None,
        active_tasks: Optional[int] = None,
        completed_tasks: Optional[int] = None,
        failed_tasks: Optional[int] = None,
        active_stages: Optional[int] = None,
        completed_stages: Optional[int] = None,
        current_memory_mb: Optional[float] = None,
        current_cpu_percent: Optional[float] = None,
        executors: Optional[int] = None,
    ) -> None:
        """Update status for an application.

        Args:
            app_id: Spark application ID
            status: New status (running, completed, failed)
            progress: Job progress (0.0 to 1.0)
            active_tasks: Number of active tasks
            completed_tasks: Number of completed tasks
            failed_tasks: Number of failed tasks
            active_stages: Number of active stages
            completed_stages: Number of completed stages
            current_memory_mb: Current memory usage in MB
            current_cpu_percent: Current CPU usage percentage
            executors: Number of executors
        """
        with self._lock:
            if app_id not in self._applications:
                return

            app = self._applications[app_id]

            if status is not None:
                old_status = app.status
                app.status = status
                if old_status != status:
                    self._notify(
                        "status_changed",
                        {
                            "app_id": app_id,
                            "old_status": old_status,
                            "new_status": status,
                        },
                    )

            if progress is not None:
                app.progress = progress
            if active_tasks is not None:
                app.active_tasks = active_tasks
            if completed_tasks is not None:
                app.completed_tasks = completed_tasks
            if failed_tasks is not None:
                app.failed_tasks = failed_tasks
            if active_stages is not None:
                app.active_stages = active_stages
            if completed_stages is not None:
                app.completed_stages = completed_stages
            if current_memory_mb is not None:
                app.current_memory_mb = current_memory_mb
            if current_cpu_percent is not None:
                app.current_cpu_percent = current_cpu_percent
            if executors is not None:
                app.executors = executors

            # Update duration
            if app.start_time:
                app.duration_seconds = (
                    datetime.utcnow() - app.start_time
                ).total_seconds()

            app.last_updated = datetime.utcnow()

            # Notify subscribers
            self._notify("application_updated", app.to_dict())

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                self._poll_metrics()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")

            time.sleep(self.poll_interval)

    def _poll_metrics(self) -> None:
        """Poll metrics from configured sources."""
        # Poll from Prometheus if configured
        if self.metrics_endpoint:
            self._poll_prometheus()

        # Poll from History Server if configured
        if self.history_server_url:
            self._poll_history_server()

    def _poll_prometheus(self) -> None:
        """Poll metrics from Prometheus."""
        try:
            import requests

            # Query for running applications
            response = requests.get(
                f"{self.metrics_endpoint}/api/v1/query",
                params={"query": "spark_executor_count"},
                timeout=10,
            )

            if response.status_code != 200:
                return

            data = response.json()
            results = data.get("data", {}).get("result", [])

            for result in results:
                labels = result.get("metric", {})
                app_id = labels.get("app_id")

                if not app_id:
                    continue

                app_name = labels.get("app_name", app_id)

                # Add or update application
                self.add_application(app_id, app_name)

                # Get executor count
                value = result.get("value", [None, 0])
                if len(value) >= 2:
                    try:
                        executors = int(float(value[1]))
                        self.update_status(app_id, executors=executors)
                    except (ValueError, TypeError):
                        pass

        except ImportError:
            logger.warning("requests library not available for Prometheus polling")
        except Exception as e:
            logger.error(f"Error polling Prometheus: {e}")

    def _poll_history_server(self) -> None:
        """Poll status from Spark History Server."""
        try:
            import requests

            # Query for running applications
            response = requests.get(
                f"{self.history_server_url}/api/v1/applications",
                params={"status": "running"},
                timeout=10,
            )

            if response.status_code != 200:
                return

            apps = response.json()

            for app_data in apps:
                app_id = app_data.get("id")
                app_name = app_data.get("name", app_id)

                if not app_id:
                    continue

                # Add or update application
                self.add_application(app_id, app_name)

                # Get detailed status
                detail_response = requests.get(
                    f"{self.history_server_url}/api/v1/applications/{app_id}",
                    timeout=10,
                )

                if detail_response.status_code == 200:
                    detail = detail_response.json()
                    attempts = detail.get("attempts", [])

                    if attempts:
                        latest = attempts[-1]
                        self.update_status(
                            app_id,
                            status=(
                                "running"
                                if not latest.get("completed")
                                else "completed"
                            ),
                        )

        except ImportError:
            logger.warning("requests library not available for History Server polling")
        except Exception as e:
            logger.error(f"Error polling History Server: {e}")

    def _trim_history(self, key: str) -> None:
        """Trim metric history to stay within limits.

        Args:
            key: Metric history key (app_id:metric_name)
        """
        if key not in self._metric_history:
            return

        history = self._metric_history[key]

        # Trim by count
        if len(history) > self._max_history_points:
            history = history[-self._max_history_points :]

        # Trim by age
        cutoff = datetime.utcnow() - self._max_history_age
        history = [p for p in history if p.timestamp > cutoff]

        self._metric_history[key] = history

    def _notify(self, event_type: str, data: Dict) -> None:
        """Notify all subscribers of an event.

        Args:
            event_type: Type of event
            data: Event data
        """
        for callback in self._subscribers:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
