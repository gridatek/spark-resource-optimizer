"""Collector for Spark metrics and monitoring systems."""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

try:
    import requests
except ImportError:
    requests = None  # type: ignore

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class MetricsCollector(BaseCollector):
    """Collects job data from metrics systems (Prometheus, Grafana, etc.)."""

    # Default Prometheus metrics for Spark
    SPARK_METRICS = {
        "executor_count": 'spark_executor_count{app_id=~".*"}',
        "executor_memory": 'spark_executor_memory_bytes{app_id=~".*"}',
        "executor_cores": 'spark_executor_cores{app_id=~".*"}',
        "task_count": 'spark_task_count{app_id=~".*"}',
        "task_duration": 'spark_task_duration_seconds{app_id=~".*"}',
        "shuffle_read": 'spark_shuffle_read_bytes{app_id=~".*"}',
        "shuffle_write": 'spark_shuffle_write_bytes{app_id=~".*"}',
        "input_bytes": 'spark_input_bytes{app_id=~".*"}',
        "output_bytes": 'spark_output_bytes{app_id=~".*"}',
        "gc_time": 'spark_jvm_gc_time_seconds{app_id=~".*"}',
        "memory_used": 'spark_memory_used_bytes{app_id=~".*"}',
        "cpu_time": 'spark_cpu_time_seconds{app_id=~".*"}',
    }

    def __init__(self, metrics_endpoint: str, config: Optional[Dict] = None):
        """Initialize the Metrics collector.

        Args:
            metrics_endpoint: URL of the metrics endpoint (e.g., http://prometheus:9090)
            config: Optional configuration dictionary with keys:
                - timeout: Request timeout in seconds (default: 30)
                - lookback_hours: Hours to look back for metrics (default: 24)
                - step: Query step interval (default: "1m")
                - custom_metrics: Dict of custom metric queries
                - verify_ssl: Whether to verify SSL certificates (default: True)
        """
        super().__init__(config)
        self.metrics_endpoint = metrics_endpoint.rstrip("/")
        self.timeout = self.config.get("timeout", 30)
        self.lookback_hours = self.config.get("lookback_hours", 24)
        self.step = self.config.get("step", "1m")
        self.verify_ssl = self.config.get("verify_ssl", True)

        # Merge custom metrics with defaults
        self.metrics = dict(self.SPARK_METRICS)
        if self.config.get("custom_metrics"):
            self.metrics.update(self.config["custom_metrics"])

    def collect(self) -> List[Dict]:
        """Collect job data from metrics system.

        Returns:
            List of dictionaries containing job metrics
        """
        if requests is None:
            raise ImportError(
                "requests library is required for MetricsCollector. "
                "Install it with: pip install requests"
            )

        if not self.validate_config():
            raise ConnectionError(
                f"Cannot connect to metrics endpoint: {self.metrics_endpoint}"
            )

        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=self.lookback_hours)

        # Collect metrics for each query
        raw_metrics = {}
        for metric_name, query in self.metrics.items():
            try:
                result = self._query_prometheus(
                    query,
                    start_time=start_time,
                    end_time=end_time,
                )
                raw_metrics[metric_name] = result
            except Exception as e:
                logger.warning(f"Failed to collect metric {metric_name}: {e}")
                raw_metrics[metric_name] = []

        # Aggregate metrics by application
        jobs = self._aggregate_metrics(raw_metrics)

        return jobs

    def validate_config(self) -> bool:
        """Validate the collector configuration.

        Returns:
            True if configuration is valid, False otherwise
        """
        if requests is None:
            logger.error("requests library not available")
            return False

        try:
            # Try to reach the Prometheus API
            response = requests.get(
                f"{self.metrics_endpoint}/api/v1/status/config",
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to metrics endpoint: {e}")
            return False

    def _query_prometheus(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict]:
        """Query Prometheus metrics.

        Args:
            query: PromQL query string
            start_time: Start of time range (default: 1 hour ago)
            end_time: End of time range (default: now)

        Returns:
            List of query results
        """
        if requests is None:
            return []

        # Set default time range
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=1)

        # Convert to Unix timestamps
        start_ts = start_time.timestamp()
        end_ts = end_time.timestamp()

        # Query range API for time series data
        url = f"{self.metrics_endpoint}/api/v1/query_range"
        params = {
            "query": query,
            "start": start_ts,
            "end": end_ts,
            "step": self.step,
        }

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "success":
                logger.warning(
                    f"Prometheus query failed: {data.get('error', 'Unknown error')}"
                )
                return []

            return data.get("data", {}).get("result", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Prometheus: {e}")
            return []

    def _query_instant(self, query: str) -> List[Dict]:
        """Query Prometheus for instant values.

        Args:
            query: PromQL query string

        Returns:
            List of query results
        """
        if requests is None:
            return []

        url = f"{self.metrics_endpoint}/api/v1/query"
        params = {"query": query}

        try:
            response = requests.get(
                url,
                params=params,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            response.raise_for_status()

            data = response.json()

            if data.get("status") != "success":
                return []

            return data.get("data", {}).get("result", [])

        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying Prometheus: {e}")
            return []

    def _aggregate_metrics(self, raw_metrics: Dict[str, List[Dict]]) -> List[Dict]:
        """Aggregate raw metrics into job-level statistics.

        Args:
            raw_metrics: Dictionary mapping metric names to Prometheus results

        Returns:
            List of job dictionaries with aggregated metrics
        """
        # Group metrics by app_id
        apps: Dict[str, Dict] = {}

        for metric_name, results in raw_metrics.items():
            for result in results:
                labels = result.get("metric", {})
                app_id = labels.get("app_id") or labels.get("application_id")

                if not app_id:
                    continue

                if app_id not in apps:
                    apps[app_id] = {
                        "app_id": app_id,
                        "app_name": labels.get("app_name", app_id),
                        "metrics": {},
                        "labels": labels,
                    }

                # Extract values from time series
                values = result.get("values", [])
                if values:
                    # Get the latest value and calculate statistics
                    numeric_values = []
                    for ts, val in values:
                        try:
                            numeric_values.append(float(val))
                        except (ValueError, TypeError):
                            continue

                    if numeric_values:
                        apps[app_id]["metrics"][metric_name] = {
                            "latest": numeric_values[-1],
                            "max": max(numeric_values),
                            "min": min(numeric_values),
                            "avg": sum(numeric_values) / len(numeric_values),
                            "count": len(numeric_values),
                        }

        # Convert aggregated data to job format
        jobs = []
        for app_id, app_data in apps.items():
            metrics = app_data.get("metrics", {})

            job = {
                "app_id": app_id,
                "app_name": app_data.get("app_name", app_id),
                "source": "prometheus",
                "collected_at": datetime.utcnow().isoformat(),
                # Resource configuration
                "num_executors": int(
                    metrics.get("executor_count", {}).get("latest", 0)
                ),
                "executor_memory_mb": int(
                    metrics.get("executor_memory", {}).get("latest", 0) / (1024 * 1024)
                ),
                "executor_cores": int(
                    metrics.get("executor_cores", {}).get("latest", 0)
                ),
                # Task metrics
                "total_tasks": int(metrics.get("task_count", {}).get("max", 0)),
                "duration_ms": int(
                    metrics.get("task_duration", {}).get("latest", 0) * 1000
                ),
                # I/O metrics
                "input_bytes": int(metrics.get("input_bytes", {}).get("max", 0)),
                "output_bytes": int(metrics.get("output_bytes", {}).get("max", 0)),
                "shuffle_read_bytes": int(
                    metrics.get("shuffle_read", {}).get("max", 0)
                ),
                "shuffle_write_bytes": int(
                    metrics.get("shuffle_write", {}).get("max", 0)
                ),
                # Performance metrics
                "jvm_gc_time_ms": int(
                    metrics.get("gc_time", {}).get("latest", 0) * 1000
                ),
                "peak_memory_usage": int(metrics.get("memory_used", {}).get("max", 0)),
                "executor_cpu_time_ms": int(
                    metrics.get("cpu_time", {}).get("latest", 0) * 1000
                ),
                # Raw metrics for additional analysis
                "raw_metrics": metrics,
            }

            jobs.append(job)

        return jobs

    def get_application_metrics(self, app_id: str) -> Dict:
        """Get detailed metrics for a specific application.

        Args:
            app_id: Spark application ID

        Returns:
            Dictionary with detailed application metrics
        """
        metrics = {}

        for metric_name, query_template in self.metrics.items():
            # Modify query to filter by app_id
            query = query_template.replace('app_id=~".*"', f'app_id="{app_id}"')

            result = self._query_prometheus(query)
            if result:
                values = result[0].get("values", [])
                if values:
                    numeric_values = [float(v) for _, v in values]
                    metrics[metric_name] = {
                        "values": values,
                        "latest": numeric_values[-1] if numeric_values else 0,
                        "max": max(numeric_values) if numeric_values else 0,
                        "min": min(numeric_values) if numeric_values else 0,
                        "avg": (
                            sum(numeric_values) / len(numeric_values)
                            if numeric_values
                            else 0
                        ),
                    }

        return {
            "app_id": app_id,
            "metrics": metrics,
            "collected_at": datetime.utcnow().isoformat(),
        }

    def list_applications(self) -> List[str]:
        """List all applications with metrics in Prometheus.

        Returns:
            List of application IDs
        """
        # Query for unique app_ids
        query = "count by (app_id) (spark_executor_count)"
        results = self._query_instant(query)

        app_ids = []
        for result in results:
            app_id = result.get("metric", {}).get("app_id")
            if app_id:
                app_ids.append(app_id)

        return app_ids
