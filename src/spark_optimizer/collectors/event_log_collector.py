"""
Event Log Collector - Parses Spark event logs to extract job metadata and metrics
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from datetime import datetime
from dataclasses import dataclass, asdict


@dataclass
class SparkJobMetrics:
    """Structured representation of Spark job metrics"""

    app_id: str
    app_name: str
    user: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[int]

    # Resource configuration
    executor_cores: int
    executor_memory: str
    num_executors: int
    driver_memory: str
    driver_cores: int

    # Performance metrics
    total_tasks: int
    failed_tasks: int
    total_stages: int
    input_bytes: int
    output_bytes: int
    shuffle_read_bytes: int
    shuffle_write_bytes: int

    # Memory metrics
    peak_memory_usage: int
    jvm_heap_memory: int
    spill_disk_bytes: int
    spill_memory_bytes: int

    # Timing metrics
    executor_run_time: int
    executor_cpu_time: int
    jvm_gc_time: int

    # Configuration
    spark_configs: Dict[str, str]

    # Cost estimation (if available)
    cluster_type: Optional[str] = None
    instance_type: Optional[str] = None
    estimated_cost: Optional[float] = None


class EventLogCollector:
    """Collects and parses Spark event logs"""

    def __init__(self, event_log_dir: str):
        self.event_log_dir = Path(event_log_dir)

    def collect_all(self) -> Iterator[SparkJobMetrics]:
        """Collect metrics from all event logs in the directory"""
        for log_file in self.event_log_dir.rglob("*"):
            if log_file.is_file() and not log_file.name.startswith("."):
                try:
                    metrics = self.parse_event_log(log_file)
                    if metrics:
                        yield metrics
                except Exception as e:
                    print(f"Error parsing {log_file}: {e}")

    def parse_event_log(self, log_file: Path) -> Optional[SparkJobMetrics]:
        """Parse a single Spark event log file"""
        events = list(self._read_events(log_file))

        if not events:
            return None

        # Extract key events
        app_start = self._find_event(events, "SparkListenerApplicationStart")
        app_end = self._find_event(events, "SparkListenerApplicationEnd")
        env_update = self._find_event(events, "SparkListenerEnvironmentUpdate")

        if not app_start:
            return None

        # Aggregate metrics from all task and stage events
        task_metrics = self._aggregate_task_metrics(events)
        stage_metrics = self._aggregate_stage_metrics(events)

        # Extract configuration
        spark_configs = {}
        if env_update:
            spark_props = env_update.get("Spark Properties", [])
            spark_configs = dict(spark_props)

        # Build metrics object
        app_id = app_start.get("App ID", "unknown")
        app_name = app_start.get("App Name", "unknown")
        user = app_start.get("User", "unknown")
        start_time = datetime.fromtimestamp(app_start["Timestamp"] / 1000)
        end_time = (
            datetime.fromtimestamp(app_end["Timestamp"] / 1000) if app_end else None
        )
        duration_ms = app_end["Timestamp"] - app_start["Timestamp"] if app_end else None

        # Extract resource config from spark properties
        executor_memory = spark_configs.get("spark.executor.memory", "1g")
        executor_cores = int(spark_configs.get("spark.executor.cores", "1"))
        num_executors = int(spark_configs.get("spark.executor.instances", "1"))
        driver_memory = spark_configs.get("spark.driver.memory", "1g")
        driver_cores = int(spark_configs.get("spark.driver.cores", "1"))

        return SparkJobMetrics(
            app_id=app_id,
            app_name=app_name,
            user=user,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            num_executors=num_executors,
            driver_memory=driver_memory,
            driver_cores=driver_cores,
            total_tasks=task_metrics["total_tasks"],
            failed_tasks=task_metrics["failed_tasks"],
            total_stages=stage_metrics["total_stages"],
            input_bytes=task_metrics["input_bytes"],
            output_bytes=task_metrics["output_bytes"],
            shuffle_read_bytes=task_metrics["shuffle_read_bytes"],
            shuffle_write_bytes=task_metrics["shuffle_write_bytes"],
            peak_memory_usage=task_metrics["peak_memory"],
            jvm_heap_memory=task_metrics["jvm_heap_memory"],
            spill_disk_bytes=task_metrics["spill_disk_bytes"],
            spill_memory_bytes=task_metrics["spill_memory_bytes"],
            executor_run_time=task_metrics["executor_run_time"],
            executor_cpu_time=task_metrics["executor_cpu_time"],
            jvm_gc_time=task_metrics["jvm_gc_time"],
            spark_configs=spark_configs,
        )

    def _read_events(self, log_file: Path) -> Iterator[Dict]:
        """Read events from log file (handles gzip compression)"""
        open_func = gzip.open if log_file.suffix == ".gz" else open

        with open_func(log_file, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue

    def _find_event(self, events: List[Dict], event_type: str) -> Optional[Dict]:
        """Find first event of given type"""
        for event in events:
            if event.get("Event") == event_type:
                return event
        return None

    def _aggregate_task_metrics(self, events: List[Dict]) -> Dict:
        """Aggregate metrics from all task end events"""
        metrics = {
            "total_tasks": 0,
            "failed_tasks": 0,
            "input_bytes": 0,
            "output_bytes": 0,
            "shuffle_read_bytes": 0,
            "shuffle_write_bytes": 0,
            "peak_memory": 0,
            "jvm_heap_memory": 0,
            "spill_disk_bytes": 0,
            "spill_memory_bytes": 0,
            "executor_run_time": 0,
            "executor_cpu_time": 0,
            "jvm_gc_time": 0,
        }

        for event in events:
            if event.get("Event") == "SparkListenerTaskEnd":
                metrics["total_tasks"] += 1

                task_info = event.get("Task Info", {})
                if not task_info.get("Successful", True):
                    metrics["failed_tasks"] += 1

                task_metrics = event.get("Task Metrics", {})
                if task_metrics:
                    metrics["input_bytes"] += task_metrics.get("Input Metrics", {}).get(
                        "Bytes Read", 0
                    )
                    metrics["output_bytes"] += task_metrics.get(
                        "Output Metrics", {}
                    ).get("Bytes Written", 0)
                    metrics["shuffle_read_bytes"] += task_metrics.get(
                        "Shuffle Read Metrics", {}
                    ).get("Total Bytes Read", 0)
                    metrics["shuffle_write_bytes"] += task_metrics.get(
                        "Shuffle Write Metrics", {}
                    ).get("Bytes Written", 0)

                    memory_metrics = task_metrics.get("Memory Bytes Spilled", 0)
                    disk_metrics = task_metrics.get("Disk Bytes Spilled", 0)

                    metrics["spill_memory_bytes"] += memory_metrics
                    metrics["spill_disk_bytes"] += disk_metrics
                    metrics["peak_memory"] = max(
                        metrics["peak_memory"],
                        task_metrics.get("Peak Execution Memory", 0),
                    )

                    metrics["executor_run_time"] += task_metrics.get(
                        "Executor Run Time", 0
                    )
                    metrics["executor_cpu_time"] += task_metrics.get(
                        "Executor CPU Time", 0
                    )
                    metrics["jvm_gc_time"] += task_metrics.get("JVM GC Time", 0)

        return metrics

    def _aggregate_stage_metrics(self, events: List[Dict]) -> Dict:
        """Aggregate metrics from stage events"""
        stages = set()

        for event in events:
            if event.get("Event") == "SparkListenerStageCompleted":
                stage_info = event.get("Stage Info", {})
                stages.add(stage_info.get("Stage ID"))

        return {"total_stages": len(stages)}


def memory_string_to_bytes(memory_str: str) -> int:
    """Convert Spark memory string (e.g., '4g', '512m') to bytes"""
    memory_str = memory_str.lower().strip()

    units = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}

    if memory_str[-1] in units:
        return int(float(memory_str[:-1]) * units[memory_str[-1]])

    return int(memory_str)


# Example usage
if __name__ == "__main__":
    collector = EventLogCollector("/path/to/spark/event/logs")

    for job_metrics in collector.collect_all():
        print(f"Job: {job_metrics.app_name}")
        duration_sec = (
            (job_metrics.duration_ms / 1000) if job_metrics.duration_ms else 0
        )
        print(f"Duration: {duration_sec:.2f}s")
        print(f"Executors: {job_metrics.num_executors}")
        print(f"Input: {job_metrics.input_bytes / (1024**3):.2f} GB")
        print(f"Shuffle: {job_metrics.shuffle_write_bytes / (1024**3):.2f} GB")
        print("-" * 50)
