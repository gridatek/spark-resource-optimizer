"""Tests for event log collector."""

import pytest
import json
import tempfile
import gzip
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from spark_optimizer.collectors.event_log_collector import (
    EventLogCollector,
    SparkJobMetrics,
    memory_string_to_bytes,
)


@pytest.fixture
def sample_spark_events():
    """Create sample Spark event log content."""
    events = [
        {
            "Event": "SparkListenerApplicationStart",
            "App ID": "app-test-20240101120000-0001",
            "App Name": "Test ETL Job",
            "User": "spark_user",
            "Timestamp": 1704110400000,  # 2024-01-01 12:00:00
        },
        {
            "Event": "SparkListenerEnvironmentUpdate",
            "Spark Properties": [
                ["spark.executor.memory", "8g"],
                ["spark.executor.cores", "4"],
                ["spark.executor.instances", "10"],
                ["spark.driver.memory", "4g"],
                ["spark.driver.cores", "2"],
                ["spark.sql.shuffle.partitions", "200"],
            ],
        },
        {
            "Event": "SparkListenerStageCompleted",
            "Stage Info": {"Stage ID": 0, "Stage Name": "map"},
        },
        {
            "Event": "SparkListenerStageCompleted",
            "Stage Info": {"Stage ID": 1, "Stage Name": "reduce"},
        },
        {
            "Event": "SparkListenerTaskEnd",
            "Task Info": {"Task ID": 0, "Successful": True},
            "Task Metrics": {
                "Input Metrics": {"Bytes Read": 1073741824},  # 1 GB
                "Output Metrics": {"Bytes Written": 536870912},  # 512 MB
                "Shuffle Read Metrics": {"Total Bytes Read": 268435456},  # 256 MB
                "Shuffle Write Metrics": {"Bytes Written": 268435456},  # 256 MB
                "Executor Run Time": 5000,
                "Executor CPU Time": 4500,
                "JVM GC Time": 200,
                "Memory Bytes Spilled": 0,
                "Disk Bytes Spilled": 0,
                "Peak Execution Memory": 4294967296,  # 4 GB
            },
        },
        {
            "Event": "SparkListenerTaskEnd",
            "Task Info": {"Task ID": 1, "Successful": True},
            "Task Metrics": {
                "Input Metrics": {"Bytes Read": 1073741824},  # 1 GB
                "Output Metrics": {"Bytes Written": 536870912},  # 512 MB
                "Shuffle Read Metrics": {"Total Bytes Read": 268435456},  # 256 MB
                "Shuffle Write Metrics": {"Bytes Written": 268435456},  # 256 MB
                "Executor Run Time": 6000,
                "Executor CPU Time": 5500,
                "JVM GC Time": 300,
                "Memory Bytes Spilled": 0,
                "Disk Bytes Spilled": 0,
                "Peak Execution Memory": 5368709120,  # 5 GB
            },
        },
        {
            "Event": "SparkListenerApplicationEnd",
            "Timestamp": 1704114000000,  # 2024-01-01 13:00:00
        },
    ]
    return events


@pytest.fixture
def event_log_file(sample_spark_events):
    """Create a temporary event log file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
        for event in sample_spark_events:
            f.write(json.dumps(event) + "\n")
        temp_path = f.name
    yield Path(temp_path)
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def event_log_gzip(sample_spark_events):
    """Create a gzip-compressed event log file."""
    with tempfile.NamedTemporaryFile(suffix=".log.gz", delete=False) as f:
        temp_path = f.name

    with gzip.open(temp_path, "wt", encoding="utf-8") as f:
        for event in sample_spark_events:
            f.write(json.dumps(event) + "\n")

    yield Path(temp_path)
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def event_log_dir(sample_spark_events):
    """Create a temporary directory with multiple event logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple event log files
        for i in range(3):
            events = sample_spark_events.copy()
            events[0] = events[0].copy()
            events[0]["App ID"] = f"app-test-{i:04d}"
            events[0]["App Name"] = f"Test Job {i}"

            with open(Path(tmpdir) / f"event_log_{i}.log", "w") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")

        yield Path(tmpdir)


class TestEventLogCollector:
    """Test cases for EventLogCollector."""

    def test_collector_initialization(self, event_log_dir):
        """Test collector can be initialized."""
        collector = EventLogCollector(str(event_log_dir))

        assert collector.event_log_dir == event_log_dir
        assert collector.event_log_dir.exists()

    def test_collector_initialization_with_path_object(self, event_log_dir):
        """Test collector initialization with Path object."""
        collector = EventLogCollector(event_log_dir)

        assert collector.event_log_dir == event_log_dir

    def test_collect_from_event_log(self, event_log_file):
        """Test collecting data from a single event log file."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        assert metrics is not None
        assert metrics.app_id == "app-test-20240101120000-0001"
        assert metrics.app_name == "Test ETL Job"
        assert metrics.user == "spark_user"

    def test_collect_from_gzip_file(self, event_log_gzip):
        """Test collecting data from gzip-compressed event log."""
        collector = EventLogCollector(str(event_log_gzip.parent))

        metrics = collector.parse_event_log(event_log_gzip)

        assert metrics is not None
        assert metrics.app_id == "app-test-20240101120000-0001"

    def test_collect_all_from_directory(self, event_log_dir):
        """Test collecting all event logs from directory."""
        collector = EventLogCollector(str(event_log_dir))

        all_metrics = list(collector.collect_all())

        assert len(all_metrics) == 3
        app_ids = {m.app_id for m in all_metrics}
        assert "app-test-0000" in app_ids
        assert "app-test-0001" in app_ids
        assert "app-test-0002" in app_ids

    def test_parse_application_data(self, event_log_file):
        """Test parsing application metadata."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        assert metrics.app_name == "Test ETL Job"
        assert metrics.user == "spark_user"
        assert metrics.start_time is not None
        assert metrics.end_time is not None
        assert metrics.duration_ms == 3600000  # 1 hour

    def test_parse_resource_configuration(self, event_log_file):
        """Test parsing resource configuration from environment update."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        assert metrics.executor_memory == "8g"
        assert metrics.executor_cores == 4
        assert metrics.num_executors == 10
        assert metrics.driver_memory == "4g"
        assert metrics.driver_cores == 2

    def test_aggregate_task_metrics(self, event_log_file):
        """Test aggregation of task metrics."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        # Should sum up metrics from all tasks
        assert metrics.total_tasks == 2
        assert metrics.failed_tasks == 0
        assert metrics.input_bytes == 2 * 1073741824  # 2 GB total
        assert metrics.output_bytes == 2 * 536870912  # 1 GB total
        assert metrics.shuffle_read_bytes == 2 * 268435456  # 512 MB total
        assert metrics.shuffle_write_bytes == 2 * 268435456  # 512 MB total

    def test_aggregate_stage_metrics(self, event_log_file):
        """Test aggregation of stage metrics."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        assert metrics.total_stages == 2

    def test_timing_metrics(self, event_log_file):
        """Test parsing timing metrics."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        assert metrics.executor_run_time == 11000  # 5000 + 6000
        assert metrics.executor_cpu_time == 10000  # 4500 + 5500
        assert metrics.jvm_gc_time == 500  # 200 + 300

    def test_memory_metrics(self, event_log_file):
        """Test parsing memory metrics."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        # Should take max peak memory
        assert metrics.peak_memory_usage == 5368709120  # 5 GB (max)
        assert metrics.spill_memory_bytes == 0
        assert metrics.spill_disk_bytes == 0

    def test_spark_configs_extracted(self, event_log_file):
        """Test Spark configurations are extracted."""
        collector = EventLogCollector(str(event_log_file.parent))

        metrics = collector.parse_event_log(event_log_file)

        assert "spark.sql.shuffle.partitions" in metrics.spark_configs
        assert metrics.spark_configs["spark.sql.shuffle.partitions"] == "200"

    def test_validate_config(self, event_log_dir):
        """Test configuration validation."""
        collector = EventLogCollector(str(event_log_dir))

        # Event log collector doesn't have validate_config, but the dir should exist
        assert collector.event_log_dir.exists()
        assert collector.event_log_dir.is_dir()

    def test_parse_empty_event_log(self):
        """Test parsing empty event log returns None."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            temp_path = Path(f.name)

        try:
            collector = EventLogCollector(str(temp_path.parent))
            metrics = collector.parse_event_log(temp_path)

            assert metrics is None
        finally:
            temp_path.unlink(missing_ok=True)

    def test_parse_log_without_app_start(self):
        """Test parsing log without application start event returns None."""
        events = [
            {"Event": "SparkListenerTaskEnd", "Task Info": {"Task ID": 0}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
            temp_path = Path(f.name)

        try:
            collector = EventLogCollector(str(temp_path.parent))
            metrics = collector.parse_event_log(temp_path)

            assert metrics is None
        finally:
            temp_path.unlink(missing_ok=True)

    def test_parse_log_without_app_end(self, sample_spark_events):
        """Test parsing log without application end event."""
        # Remove app end event
        events = [
            e
            for e in sample_spark_events
            if e["Event"] != "SparkListenerApplicationEnd"
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
            temp_path = Path(f.name)

        try:
            collector = EventLogCollector(str(temp_path.parent))
            metrics = collector.parse_event_log(temp_path)

            assert metrics is not None
            assert metrics.end_time is None
            assert metrics.duration_ms is None
        finally:
            temp_path.unlink(missing_ok=True)

    def test_handle_malformed_json_lines(self, sample_spark_events):
        """Test handling malformed JSON lines gracefully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("not valid json\n")
            for event in sample_spark_events:
                f.write(json.dumps(event) + "\n")
            f.write("{incomplete json\n")
            temp_path = Path(f.name)

        try:
            collector = EventLogCollector(str(temp_path.parent))
            metrics = collector.parse_event_log(temp_path)

            # Should still parse valid events
            assert metrics is not None
            assert metrics.app_id == "app-test-20240101120000-0001"
        finally:
            temp_path.unlink(missing_ok=True)

    def test_skip_hidden_files(self, event_log_dir):
        """Test that hidden files are skipped."""
        # Create a hidden file
        hidden_file = event_log_dir / ".hidden_log"
        hidden_file.write_text('{"Event": "test"}')

        collector = EventLogCollector(str(event_log_dir))
        all_metrics = list(collector.collect_all())

        # Should not include hidden file
        assert len(all_metrics) == 3

    def test_handle_failed_tasks(self, sample_spark_events):
        """Test handling failed task events."""
        events = sample_spark_events.copy()
        # Add a failed task
        events.append(
            {
                "Event": "SparkListenerTaskEnd",
                "Task Info": {"Task ID": 2, "Successful": False},
                "Task Metrics": {
                    "Input Metrics": {"Bytes Read": 0},
                    "Output Metrics": {"Bytes Written": 0},
                    "Shuffle Read Metrics": {"Total Bytes Read": 0},
                    "Shuffle Write Metrics": {"Bytes Written": 0},
                    "Executor Run Time": 1000,
                    "Executor CPU Time": 800,
                    "JVM GC Time": 50,
                    "Memory Bytes Spilled": 0,
                    "Disk Bytes Spilled": 0,
                },
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            for event in events:
                f.write(json.dumps(event) + "\n")
            temp_path = Path(f.name)

        try:
            collector = EventLogCollector(str(temp_path.parent))
            metrics = collector.parse_event_log(temp_path)

            assert metrics.total_tasks == 3
            assert metrics.failed_tasks == 1
        finally:
            temp_path.unlink(missing_ok=True)


class TestMemoryStringConversion:
    """Test cases for memory string to bytes conversion."""

    def test_convert_gigabytes(self):
        """Test converting gigabyte strings."""
        assert memory_string_to_bytes("4g") == 4 * 1024**3
        assert memory_string_to_bytes("4G") == 4 * 1024**3
        assert memory_string_to_bytes("1g") == 1024**3

    def test_convert_megabytes(self):
        """Test converting megabyte strings."""
        assert memory_string_to_bytes("512m") == 512 * 1024**2
        assert memory_string_to_bytes("512M") == 512 * 1024**2
        assert memory_string_to_bytes("1024m") == 1024**3

    def test_convert_kilobytes(self):
        """Test converting kilobyte strings."""
        assert memory_string_to_bytes("1024k") == 1024 * 1024
        assert memory_string_to_bytes("1024K") == 1024 * 1024

    def test_convert_terabytes(self):
        """Test converting terabyte strings."""
        assert memory_string_to_bytes("1t") == 1024**4
        assert memory_string_to_bytes("2T") == 2 * 1024**4

    def test_convert_decimal_values(self):
        """Test converting decimal memory values."""
        assert memory_string_to_bytes("1.5g") == int(1.5 * 1024**3)
        assert memory_string_to_bytes("2.5m") == int(2.5 * 1024**2)

    def test_convert_plain_bytes(self):
        """Test converting plain byte values."""
        assert memory_string_to_bytes("1073741824") == 1073741824

    def test_handle_whitespace(self):
        """Test handling strings with whitespace."""
        assert memory_string_to_bytes("  4g  ") == 4 * 1024**3
        assert memory_string_to_bytes(" 512m ") == 512 * 1024**2


class TestSparkJobMetricsDataclass:
    """Test cases for SparkJobMetrics dataclass."""

    def test_create_metrics_with_required_fields(self):
        """Test creating metrics with all required fields."""
        metrics = SparkJobMetrics(
            app_id="app-test-001",
            app_name="Test Job",
            user="test_user",
            start_time=datetime.now(),
            end_time=None,
            duration_ms=None,
            executor_cores=4,
            executor_memory="8g",
            num_executors=10,
            driver_memory="4g",
            driver_cores=2,
            total_tasks=100,
            failed_tasks=0,
            total_stages=5,
            input_bytes=1024**3,
            output_bytes=512 * 1024**2,
            shuffle_read_bytes=256 * 1024**2,
            shuffle_write_bytes=256 * 1024**2,
            peak_memory_usage=4 * 1024**3,
            jvm_heap_memory=2 * 1024**3,
            spill_disk_bytes=0,
            spill_memory_bytes=0,
            executor_run_time=50000,
            executor_cpu_time=45000,
            jvm_gc_time=1000,
            spark_configs={"spark.sql.shuffle.partitions": "200"},
        )

        assert metrics.app_id == "app-test-001"
        assert metrics.app_name == "Test Job"
        assert metrics.num_executors == 10
        assert metrics.cluster_type is None  # Optional field

    def test_create_metrics_with_optional_fields(self):
        """Test creating metrics with optional fields."""
        metrics = SparkJobMetrics(
            app_id="app-test-001",
            app_name="Test Job",
            user="test_user",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_ms=60000,
            executor_cores=4,
            executor_memory="8g",
            num_executors=10,
            driver_memory="4g",
            driver_cores=2,
            total_tasks=100,
            failed_tasks=0,
            total_stages=5,
            input_bytes=1024**3,
            output_bytes=512 * 1024**2,
            shuffle_read_bytes=256 * 1024**2,
            shuffle_write_bytes=256 * 1024**2,
            peak_memory_usage=4 * 1024**3,
            jvm_heap_memory=2 * 1024**3,
            spill_disk_bytes=0,
            spill_memory_bytes=0,
            executor_run_time=50000,
            executor_cpu_time=45000,
            jvm_gc_time=1000,
            spark_configs={},
            cluster_type="emr",
            instance_type="r5.xlarge",
            estimated_cost=5.50,
        )

        assert metrics.cluster_type == "emr"
        assert metrics.instance_type == "r5.xlarge"
        assert metrics.estimated_cost == 5.50
