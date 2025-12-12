"""Collector for Spark History Server API."""

from typing import Dict, List, Optional
from datetime import datetime
import requests
import logging
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class HistoryServerCollector(BaseCollector):
    """Collects job data from Spark History Server REST API.

    The Spark History Server provides a REST API to retrieve information about
    completed Spark applications. This collector uses the API to fetch application
    metrics and store them in the database.

    API Documentation:
    https://spark.apache.org/docs/latest/monitoring.html#rest-api
    """

    def __init__(self, history_server_url: str, config: Optional[Dict] = None):
        """Initialize the History Server collector.

        Args:
            history_server_url: URL of the Spark History Server (e.g., http://localhost:18080)
            config: Optional configuration dictionary with:
                - timeout: Request timeout in seconds (default: 30)
                - max_apps: Maximum number of applications to fetch (default: 100)
                - min_date: Minimum date for applications (ISO format)
                - status: Application status filter ('completed', 'running', etc.)
        """
        super().__init__(config)
        self.history_server_url = history_server_url.rstrip("/")
        self.timeout = self.config.get("timeout", 30)
        self.max_apps = self.config.get("max_apps", 100)
        self.min_date = self.config.get("min_date")
        self.status = self.config.get("status", "completed")

    def collect(self) -> List[Dict]:
        """Collect job data from History Server.

        Returns:
            List of dictionaries containing job metrics in the format expected
            by the database storage layer.

        Raises:
            requests.RequestException: If API requests fail
        """
        logger.info(f"Collecting applications from {self.history_server_url}")
        applications = self._fetch_applications()
        logger.info(f"Found {len(applications)} applications")

        job_data = []
        for app in applications:
            try:
                app_id = app["id"]
                app_name = app.get("name", "Unknown")
                attempts = app.get("attempts", [])

                if not attempts:
                    logger.warning(f"No attempts found for {app_id}, skipping")
                    continue

                attempt_id = attempts[-1].get("attemptId") if attempts else None
                logger.debug(f"Processing {app_name} ({app_id})")

                # Fetch detailed application data
                app_details = self._fetch_application_details(app_id, attempt_id)
                executors = self._fetch_executors(app_id, attempt_id)
                environment = self._fetch_environment(app_id, attempt_id)

                # Convert to normalized format
                metrics = self._convert_to_metrics(
                    app, app_details, executors, environment
                )
                if metrics:
                    job_data.append(metrics)
                    logger.info(f"Collected metrics for {app_name}")
                else:
                    logger.warning(f"Failed to convert metrics for {app_name}")

            except Exception as e:
                logger.error(f"Error fetching details for {app_id}: {e}", exc_info=True)
                continue

        logger.info(f"Successfully collected {len(job_data)} jobs")
        return job_data

    def validate_config(self) -> bool:
        """Validate the collector configuration.

        Returns:
            True if configuration is valid and History Server is accessible
        """
        try:
            response = requests.get(
                f"{self.history_server_url}/api/v1/applications",
                params={"limit": 1},
                timeout=5,
            )
            return response.status_code == 200
        except Exception:
            return False

    def _fetch_applications(self) -> List[Dict]:
        """Fetch list of applications from History Server.

        Returns:
            List of application metadata

        Raises:
            requests.RequestException: If request fails
        """
        params = {"limit": self.max_apps}

        # Note: The Spark History Server API's 'status' parameter may not work as expected
        # in all versions. We fetch all applications and filter afterward if needed.
        if self.min_date:
            params["minDate"] = self.min_date

        logger.debug(f"Fetching applications with params: {params}")
        response = requests.get(
            f"{self.history_server_url}/api/v1/applications",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()

        applications = response.json()

        # Filter by status if specified (doing it client-side for compatibility)
        if self.status and self.status != "all":
            filtered = []
            for app in applications:
                attempts = app.get("attempts", [])
                if attempts:
                    latest_attempt = attempts[-1]
                    # Check if attempt is completed
                    if self.status == "completed" and "endTime" in latest_attempt:
                        filtered.append(app)
                    elif self.status == "running" and "endTime" not in latest_attempt:
                        filtered.append(app)
                    else:
                        filtered.append(app)  # Include if status doesn't match known types
            logger.debug(f"Filtered {len(filtered)}/{len(applications)} applications by status={self.status}")
            return filtered

        return applications

    def _fetch_application_details(
        self, app_id: str, attempt_id: Optional[str] = None
    ) -> Dict:
        """Fetch detailed metrics for a specific application.

        Args:
            app_id: Application ID
            attempt_id: Attempt ID (optional, uses latest if not provided)

        Returns:
            Dictionary containing application details

        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.history_server_url}/api/v1/applications/{app_id}"
        if attempt_id:
            url += f"/{attempt_id}"

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        return response.json()

    def _fetch_executors(
        self, app_id: str, attempt_id: Optional[str] = None
    ) -> List[Dict]:
        """Fetch executor information for an application.

        Args:
            app_id: Application ID
            attempt_id: Attempt ID (optional)

        Returns:
            List of executor metadata
        """
        url = f"{self.history_server_url}/api/v1/applications/{app_id}"
        if attempt_id:
            url += f"/{attempt_id}"
        url += "/allexecutors"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception:
            return []

    def _fetch_environment(self, app_id: str, attempt_id: Optional[str] = None) -> Dict:
        """Fetch environment and configuration for an application.

        Args:
            app_id: Application ID
            attempt_id: Attempt ID (optional)

        Returns:
            Dictionary containing environment info and Spark properties
        """
        url = f"{self.history_server_url}/api/v1/applications/{app_id}"
        if attempt_id:
            url += f"/{attempt_id}"
        url += "/environment"

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}

    def _convert_to_metrics(
        self,
        app: Dict,
        app_details: Dict,
        executors: List[Dict],
        environment: Dict,
    ) -> Optional[Dict]:
        """Convert History Server API response to normalized metrics format.

        Args:
            app: Application metadata from list endpoint
            app_details: Detailed application info
            executors: List of executor info
            environment: Environment and configuration info

        Returns:
            Dictionary in the format expected by database storage, or None if conversion fails
        """
        try:
            # Extract Spark properties
            spark_props = {}
            if "sparkProperties" in environment:
                spark_props = {
                    prop[0]: prop[1] for prop in environment["sparkProperties"]
                }

            # Parse memory strings to MB
            executor_memory_str = spark_props.get("spark.executor.memory", "1g")
            driver_memory_str = spark_props.get("spark.driver.memory", "1g")

            executor_memory_mb = self._parse_memory_to_mb(executor_memory_str)
            driver_memory_mb = self._parse_memory_to_mb(driver_memory_str)

            # Get executor configuration
            executor_cores = int(spark_props.get("spark.executor.cores", "1"))
            num_executors = len([e for e in executors if e.get("id") != "driver"])

            # Parse timestamps and user
            start_time = None
            end_time = None
            duration_ms = None
            user = "unknown"

            attempts = app.get("attempts", [])
            if attempts:
                latest_attempt = attempts[-1]

                # Use epoch timestamps (in milliseconds) instead of ISO strings
                if "startTimeEpoch" in latest_attempt:
                    start_time = self._parse_timestamp(latest_attempt["startTimeEpoch"])
                elif "startTime" in latest_attempt:
                    # Fallback to parsing ISO string if epoch not available
                    try:
                        from datetime import datetime
                        start_time = datetime.fromisoformat(latest_attempt["startTime"].replace("GMT", "+00:00"))
                    except Exception:
                        logger.warning(f"Failed to parse startTime: {latest_attempt['startTime']}")

                if "endTimeEpoch" in latest_attempt:
                    end_time = self._parse_timestamp(latest_attempt["endTimeEpoch"])
                elif "endTime" in latest_attempt:
                    # Fallback to parsing ISO string if epoch not available
                    try:
                        from datetime import datetime
                        end_time = datetime.fromisoformat(latest_attempt["endTime"].replace("GMT", "+00:00"))
                    except Exception:
                        logger.warning(f"Failed to parse endTime: {latest_attempt['endTime']}")

                if "duration" in latest_attempt:
                    duration_ms = latest_attempt["duration"]

                user = latest_attempt.get("sparkUser", "unknown")

            # Aggregate executor metrics
            total_input_bytes = 0
            total_output_bytes = 0
            total_shuffle_read = 0
            total_shuffle_write = 0
            total_disk_spilled = 0
            total_memory_spilled = 0
            total_executor_run_time = 0

            for executor in executors:
                if executor.get("id") == "driver":
                    continue

                total_input_bytes += executor.get("totalInputBytes", 0)
                total_shuffle_read += executor.get("totalShuffleRead", 0)
                total_shuffle_write += executor.get("totalShuffleWrite", 0)
                total_disk_spilled += executor.get("totalDiskBytesSpilled", 0)
                total_memory_spilled += executor.get("totalMemoryBytesSpilled", 0)
                total_executor_run_time += executor.get("totalDuration", 0)

            # Get task and stage counts from app details
            total_tasks = app_details.get("totalTasks", 0)
            failed_tasks = app_details.get("failedTasks", 0)
            total_stages = app_details.get("totalStages", 0)

            return {
                "app_id": app["id"],
                "app_name": app["name"],
                "user": user,
                "start_time": start_time,
                "end_time": end_time,
                "duration_ms": duration_ms,
                "num_executors": num_executors,
                "executor_cores": executor_cores,
                "executor_memory_mb": executor_memory_mb,
                "driver_memory_mb": driver_memory_mb,
                "total_tasks": total_tasks,
                "failed_tasks": failed_tasks,
                "total_stages": total_stages,
                "input_bytes": total_input_bytes,
                "output_bytes": total_output_bytes,
                "shuffle_read_bytes": total_shuffle_read,
                "shuffle_write_bytes": total_shuffle_write,
                "disk_spilled_bytes": total_disk_spilled,
                "memory_spilled_bytes": total_memory_spilled,
                "executor_run_time_ms": total_executor_run_time,
            }

        except KeyError as e:
            logger.error(f"Missing required field in metrics conversion: {e}")
            logger.debug(f"App data: {app}")
            return None
        except Exception as e:
            logger.error(f"Error converting metrics: {e}", exc_info=True)
            return None

    def _parse_memory_to_mb(self, memory_str: str) -> int:
        """Parse Spark memory string (e.g., '4g', '512m') to megabytes.

        Args:
            memory_str: Memory string in Spark format

        Returns:
            Memory in megabytes
        """
        if not memory_str:
            return 1024  # Default to 1GB if not specified

        memory_str = str(memory_str).lower().strip()

        units = {
            "k": 1 / 1024,  # KB to MB
            "m": 1,  # MB to MB
            "g": 1024,  # GB to MB
            "t": 1024 * 1024,  # TB to MB
        }

        if memory_str[-1] in units:
            return int(float(memory_str[:-1]) * units[memory_str[-1]])

        # Assume bytes if no unit
        return int(float(memory_str) / (1024 * 1024))
