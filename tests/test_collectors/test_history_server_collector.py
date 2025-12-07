"""Tests for Spark History Server collector."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from spark_optimizer.collectors.history_server_collector import HistoryServerCollector


class TestHistoryServerCollector:
    """Test the HistoryServerCollector class."""

    def test_init(self):
        """Test collector initialization."""
        collector = HistoryServerCollector("http://localhost:18080")
        assert collector.history_server_url == "http://localhost:18080"
        assert collector.timeout == 30
        assert collector.max_apps == 100
        assert collector.status == "completed"

    def test_init_with_config(self):
        """Test collector initialization with custom config."""
        config = {
            "timeout": 60,
            "max_apps": 50,
            "status": "running",
            "min_date": "2024-01-01",
        }
        collector = HistoryServerCollector("http://localhost:18080", config)
        assert collector.timeout == 60
        assert collector.max_apps == 50
        assert collector.status == "running"
        assert collector.min_date == "2024-01-01"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from URL."""
        collector = HistoryServerCollector("http://localhost:18080/")
        assert collector.history_server_url == "http://localhost:18080"

    @patch("requests.get")
    def test_validate_config_success(self, mock_get):
        """Test successful configuration validation."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        collector = HistoryServerCollector("http://localhost:18080")
        assert collector.validate_config() is True

        mock_get.assert_called_once_with(
            "http://localhost:18080/api/v1/applications",
            params={"limit": 1},
            timeout=5,
        )

    @patch("requests.get")
    def test_validate_config_failure(self, mock_get):
        """Test configuration validation failure."""
        mock_get.side_effect = Exception("Connection error")

        collector = HistoryServerCollector("http://localhost:18080")
        assert collector.validate_config() is False

    @patch("requests.get")
    def test_fetch_applications(self, mock_get):
        """Test fetching applications list."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "app-1", "name": "Test App 1"},
            {"id": "app-2", "name": "Test App 2"},
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        collector = HistoryServerCollector("http://localhost:18080")
        apps = collector._fetch_applications()

        assert len(apps) == 2
        assert apps[0]["id"] == "app-1"
        assert apps[1]["id"] == "app-2"

        mock_get.assert_called_once_with(
            "http://localhost:18080/api/v1/applications",
            params={"limit": 100, "status": "completed"},
            timeout=30,
        )

    @patch("requests.get")
    def test_fetch_applications_with_filters(self, mock_get):
        """Test fetching applications with filters."""
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        config = {"max_apps": 50, "status": "running", "min_date": "2024-01-01"}
        collector = HistoryServerCollector("http://localhost:18080", config)
        collector._fetch_applications()

        mock_get.assert_called_once_with(
            "http://localhost:18080/api/v1/applications",
            params={"limit": 50, "status": "running", "minDate": "2024-01-01"},
            timeout=30,
        )

    @patch("requests.get")
    def test_fetch_application_details(self, mock_get):
        """Test fetching application details."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": "app-1",
            "totalTasks": 100,
            "failedTasks": 0,
            "totalStages": 10,
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        collector = HistoryServerCollector("http://localhost:18080")
        details = collector._fetch_application_details("app-1")

        assert details["id"] == "app-1"
        assert details["totalTasks"] == 100

        mock_get.assert_called_once_with(
            "http://localhost:18080/api/v1/applications/app-1", timeout=30
        )

    @patch("requests.get")
    def test_fetch_application_details_with_attempt(self, mock_get):
        """Test fetching application details with attempt ID."""
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        collector = HistoryServerCollector("http://localhost:18080")
        collector._fetch_application_details("app-1", "1")

        mock_get.assert_called_once_with(
            "http://localhost:18080/api/v1/applications/app-1/1", timeout=30
        )

    @patch("requests.get")
    def test_fetch_executors(self, mock_get):
        """Test fetching executor information."""
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": "driver", "totalInputBytes": 0},
            {"id": "1", "totalInputBytes": 1000000},
            {"id": "2", "totalInputBytes": 2000000},
        ]
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        collector = HistoryServerCollector("http://localhost:18080")
        executors = collector._fetch_executors("app-1")

        assert len(executors) == 3
        assert executors[0]["id"] == "driver"
        assert executors[1]["totalInputBytes"] == 1000000

        mock_get.assert_called_once_with(
            "http://localhost:18080/api/v1/applications/app-1/allexecutors", timeout=30
        )

    @patch("requests.get")
    def test_fetch_executors_error(self, mock_get):
        """Test fetching executors handles errors gracefully."""
        mock_get.side_effect = Exception("API error")

        collector = HistoryServerCollector("http://localhost:18080")
        executors = collector._fetch_executors("app-1")

        assert executors == []

    @patch("requests.get")
    def test_fetch_environment(self, mock_get):
        """Test fetching environment information."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "sparkProperties": [
                ["spark.executor.memory", "4g"],
                ["spark.executor.cores", "2"],
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        collector = HistoryServerCollector("http://localhost:18080")
        env = collector._fetch_environment("app-1")

        assert "sparkProperties" in env
        assert len(env["sparkProperties"]) == 2

        mock_get.assert_called_once_with(
            "http://localhost:18080/api/v1/applications/app-1/environment", timeout=30
        )

    @patch("requests.get")
    def test_fetch_environment_error(self, mock_get):
        """Test fetching environment handles errors gracefully."""
        mock_get.side_effect = Exception("API error")

        collector = HistoryServerCollector("http://localhost:18080")
        env = collector._fetch_environment("app-1")

        assert env == {}

    def test_parse_memory_to_mb(self):
        """Test memory string parsing."""
        collector = HistoryServerCollector("http://localhost:18080")

        # Test different units
        assert collector._parse_memory_to_mb("1g") == 1024
        assert collector._parse_memory_to_mb("2048m") == 2048
        assert collector._parse_memory_to_mb("512k") == 0  # Rounds down
        assert collector._parse_memory_to_mb("1t") == 1024 * 1024

        # Test case insensitivity
        assert collector._parse_memory_to_mb("4G") == 4096
        assert collector._parse_memory_to_mb("8M") == 8

        # Test decimal values
        assert collector._parse_memory_to_mb("1.5g") == 1536
        assert collector._parse_memory_to_mb("0.5g") == 512

    def test_convert_to_metrics_complete_data(self):
        """Test converting API response to metrics with complete data."""
        collector = HistoryServerCollector("http://localhost:18080")

        app = {
            "id": "app-20240101-001",
            "name": "Test ETL Job",
            "attempts": [
                {
                    "attemptId": "1",
                    "sparkUser": "test_user",
                    "startTime": 1704067200000,  # 2024-01-01 00:00:00
                    "endTime": 1704070800000,  # 2024-01-01 01:00:00
                    "duration": 3600000,  # 1 hour
                }
            ],
        }

        app_details = {"totalTasks": 100, "failedTasks": 0, "totalStages": 10}

        executors = [
            {"id": "driver", "totalInputBytes": 0},
            {
                "id": "1",
                "totalInputBytes": 1000000,
                "totalShuffleRead": 500000,
                "totalShuffleWrite": 300000,
                "totalDiskBytesSpilled": 0,
                "totalMemoryBytesSpilled": 0,
                "totalDuration": 100000,
            },
            {
                "id": "2",
                "totalInputBytes": 2000000,
                "totalShuffleRead": 1000000,
                "totalShuffleWrite": 600000,
                "totalDiskBytesSpilled": 0,
                "totalMemoryBytesSpilled": 0,
                "totalDuration": 120000,
            },
        ]

        environment = {
            "sparkProperties": [
                ["spark.executor.memory", "8g"],
                ["spark.executor.cores", "4"],
                ["spark.driver.memory", "4g"],
            ]
        }

        metrics = collector._convert_to_metrics(
            app, app_details, executors, environment
        )

        assert metrics is not None
        assert metrics["app_id"] == "app-20240101-001"
        assert metrics["app_name"] == "Test ETL Job"
        assert metrics["user"] == "test_user"
        assert metrics["duration_ms"] == 3600000
        assert metrics["num_executors"] == 2  # Excludes driver
        assert metrics["executor_cores"] == 4
        assert metrics["executor_memory_mb"] == 8192  # 8g = 8192mb
        assert metrics["driver_memory_mb"] == 4096  # 4g = 4096mb
        assert metrics["total_tasks"] == 100
        assert metrics["failed_tasks"] == 0
        assert metrics["total_stages"] == 10
        assert metrics["input_bytes"] == 3000000  # 1M + 2M
        assert metrics["shuffle_read_bytes"] == 1500000  # 500K + 1M
        assert metrics["shuffle_write_bytes"] == 900000  # 300K + 600K
        assert metrics["executor_run_time_ms"] == 220000  # 100K + 120K

    def test_convert_to_metrics_minimal_data(self):
        """Test converting API response with minimal data."""
        collector = HistoryServerCollector("http://localhost:18080")

        app = {"id": "app-1", "name": "Test App", "attempts": []}

        app_details = {}
        executors = []
        environment = {}

        metrics = collector._convert_to_metrics(
            app, app_details, executors, environment
        )

        assert metrics is not None
        assert metrics["app_id"] == "app-1"
        assert metrics["app_name"] == "Test App"
        assert metrics["user"] == "unknown"
        assert metrics["num_executors"] == 0
        assert metrics["executor_cores"] == 1  # Default
        assert metrics["executor_memory_mb"] == 1024  # Default 1g
        assert metrics["driver_memory_mb"] == 1024  # Default 1g

    def test_convert_to_metrics_error_handling(self):
        """Test that conversion handles errors gracefully."""
        collector = HistoryServerCollector("http://localhost:18080")

        # Invalid data that will cause exception
        app = None
        metrics = collector._convert_to_metrics(app, {}, [], {})

        assert metrics is None

    @patch("requests.get")
    def test_collect_full_integration(self, mock_get):
        """Test full collection workflow."""

        # Mock responses for different API calls
        def mock_api_response(url, **kwargs):
            response = Mock()
            response.raise_for_status = Mock()

            if "/applications" in url and "app-" not in url:
                # Applications list
                response.json.return_value = [
                    {
                        "id": "app-1",
                        "name": "Test App",
                        "attempts": [
                            {
                                "attemptId": "1",
                                "sparkUser": "test",
                                "startTime": 1704067200000,
                                "endTime": 1704070800000,
                                "duration": 3600000,
                            }
                        ],
                    }
                ]
            elif "/environment" in url:
                # Environment
                response.json.return_value = {
                    "sparkProperties": [
                        ["spark.executor.memory", "4g"],
                        ["spark.executor.cores", "2"],
                        ["spark.driver.memory", "2g"],
                    ]
                }
            elif "/allexecutors" in url:
                # Executors
                response.json.return_value = [
                    {"id": "driver"},
                    {"id": "1", "totalInputBytes": 1000000},
                ]
            else:
                # Application details
                response.json.return_value = {"totalTasks": 10, "totalStages": 2}

            return response

        mock_get.side_effect = mock_api_response

        collector = HistoryServerCollector("http://localhost:18080")
        job_data = collector.collect()

        assert len(job_data) == 1
        assert job_data[0]["app_id"] == "app-1"
        assert job_data[0]["app_name"] == "Test App"
        assert job_data[0]["num_executors"] == 1

    @patch("requests.get")
    def test_collect_handles_errors(self, mock_get):
        """Test that collect handles individual app errors gracefully."""

        def mock_api_response(url, **kwargs):
            response = Mock()
            response.raise_for_status = Mock()

            if "/applications" in url and "app-" not in url:
                # Return two apps
                response.json.return_value = [
                    {"id": "app-1", "name": "Good App", "attempts": []},
                    {"id": "app-2", "name": "Bad App", "attempts": []},
                ]
            elif "app-2" in url:
                # Fail for app-2
                raise Exception("API error")
            else:
                response.json.return_value = {}

            return response

        mock_get.side_effect = mock_api_response

        collector = HistoryServerCollector("http://localhost:18080")

        # Should not raise exception, just skip bad apps
        job_data = collector.collect()

        # Only app-1 should be processed
        assert len(job_data) == 1
        assert job_data[0]["app_id"] == "app-1"
