"""Job analysis and feature extraction."""

from typing import Any, Dict, List, Optional
from datetime import datetime
import statistics


class JobAnalyzer:
    """Analyzes Spark job characteristics and performance."""

    # Thresholds for analysis
    HIGH_GC_RATIO = 0.1  # 10% of execution time
    HIGH_SHUFFLE_RATIO = 0.5  # Shuffle > 50% of input
    HIGH_SPILL_RATIO = 0.05  # Spill > 5% of processed data
    LOW_CPU_UTILIZATION = 0.3  # Less than 30% CPU usage
    HIGH_TASK_FAILURE_RATE = 0.01  # More than 1% task failures
    LOW_PARALLELISM_RATIO = 0.5  # Less than 50% of optimal parallelism

    def __init__(self):
        """Initialize the job analyzer."""
        pass

    def analyze_job(self, job_data: Dict) -> Dict:
        """Analyze a single job and extract key characteristics.

        Args:
            job_data: Dictionary containing job execution data

        Returns:
            Dictionary containing analyzed job characteristics
        """
        analysis = {
            "job_id": job_data.get("app_id"),
            "app_name": job_data.get("app_name"),
            "duration_ms": job_data.get("duration_ms"),
            "resource_efficiency": self._calculate_resource_efficiency(job_data),
            "bottlenecks": self._identify_bottlenecks(job_data),
            "issues": self._detect_issues(job_data),
            "health_score": self._calculate_health_score(job_data),
        }
        return analysis

    def _calculate_resource_efficiency(self, job_data: Dict) -> Dict:
        """Calculate resource utilization efficiency.

        Args:
            job_data: Job execution data

        Returns:
            Dictionary with efficiency metrics
        """
        # Calculate CPU efficiency
        # CPU efficiency = actual CPU time / (total executor time * cores)
        executor_cpu_time_ms = job_data.get("executor_cpu_time_ms", 0)
        executor_run_time_ms = job_data.get("executor_run_time_ms", 0)
        num_executors = job_data.get("num_executors", 1)
        executor_cores = job_data.get("executor_cores", 1)

        total_available_cpu_time = executor_run_time_ms * num_executors * executor_cores
        cpu_efficiency = (
            executor_cpu_time_ms / total_available_cpu_time
            if total_available_cpu_time > 0
            else 0.0
        )
        cpu_efficiency = min(1.0, cpu_efficiency)  # Cap at 100%

        # Calculate memory efficiency
        # Memory efficiency = peak memory usage / total allocated memory
        peak_memory_usage = job_data.get("peak_memory_usage", 0)
        executor_memory_mb = job_data.get("executor_memory_mb", 1)
        total_allocated_memory = executor_memory_mb * 1024 * 1024 * num_executors

        memory_efficiency = (
            peak_memory_usage / total_allocated_memory
            if total_allocated_memory > 0
            else 0.0
        )
        memory_efficiency = min(1.0, memory_efficiency)  # Cap at 100%

        # Calculate I/O efficiency
        # I/O efficiency based on throughput (bytes processed per ms)
        input_bytes = job_data.get("input_bytes", 0)
        output_bytes = job_data.get("output_bytes", 0)
        duration_ms = job_data.get("duration_ms", 1)

        total_bytes = input_bytes + output_bytes
        throughput = total_bytes / duration_ms if duration_ms > 0 else 0

        # Normalize throughput to 0-1 scale (assume 100 MB/s is optimal)
        optimal_throughput = 100 * 1024 * 1024 / 1000  # 100 MB/s in bytes/ms
        io_efficiency = (
            min(1.0, throughput / optimal_throughput) if optimal_throughput > 0 else 0.0
        )

        # Calculate shuffle efficiency
        shuffle_read = job_data.get("shuffle_read_bytes", 0)
        shuffle_write = job_data.get("shuffle_write_bytes", 0)
        shuffle_ratio = (
            (shuffle_read + shuffle_write) / input_bytes if input_bytes > 0 else 0.0
        )
        # Lower shuffle ratio is better, invert for efficiency score
        shuffle_efficiency = max(0.0, 1.0 - min(1.0, shuffle_ratio))

        return {
            "cpu_efficiency": round(cpu_efficiency, 3),
            "memory_efficiency": round(memory_efficiency, 3),
            "io_efficiency": round(io_efficiency, 3),
            "shuffle_efficiency": round(shuffle_efficiency, 3),
            "overall_efficiency": round(
                (
                    cpu_efficiency
                    + memory_efficiency
                    + io_efficiency
                    + shuffle_efficiency
                )
                / 4,
                3,
            ),
        }

    def _identify_bottlenecks(self, job_data: Dict) -> List[str]:
        """Identify performance bottlenecks.

        Args:
            job_data: Job execution data

        Returns:
            List of identified bottlenecks
        """
        bottlenecks: List[str] = []

        # Check for CPU-bound operations
        executor_cpu_time_ms = job_data.get("executor_cpu_time_ms", 0)
        executor_run_time_ms = job_data.get("executor_run_time_ms", 0)
        num_executors = job_data.get("num_executors", 1)
        executor_cores = job_data.get("executor_cores", 1)

        if executor_run_time_ms > 0:
            total_available = executor_run_time_ms * num_executors * executor_cores
            cpu_utilization = (
                executor_cpu_time_ms / total_available if total_available > 0 else 0
            )

            if cpu_utilization > 0.85:
                bottlenecks.append("CPU_BOUND")
            elif cpu_utilization < self.LOW_CPU_UTILIZATION:
                bottlenecks.append("CPU_UNDERUTILIZED")

        # Check for memory pressure
        memory_spilled = job_data.get("memory_spilled_bytes", 0)
        disk_spilled = job_data.get("disk_spilled_bytes", 0)
        input_bytes = job_data.get("input_bytes", 1)

        if memory_spilled > 0 or disk_spilled > 0:
            spill_ratio = (
                (memory_spilled + disk_spilled) / input_bytes if input_bytes > 0 else 0
            )
            if spill_ratio > self.HIGH_SPILL_RATIO:
                bottlenecks.append("MEMORY_PRESSURE")

        # Check for GC pressure
        jvm_gc_time_ms = job_data.get("jvm_gc_time_ms", 0)
        if executor_run_time_ms > 0:
            gc_ratio = jvm_gc_time_ms / executor_run_time_ms
            if gc_ratio > self.HIGH_GC_RATIO:
                bottlenecks.append("GC_OVERHEAD")

        # Check for I/O wait times (high shuffle relative to compute)
        shuffle_read = job_data.get("shuffle_read_bytes", 0)
        shuffle_write = job_data.get("shuffle_write_bytes", 0)
        if input_bytes > 0:
            shuffle_ratio = (shuffle_read + shuffle_write) / input_bytes
            if shuffle_ratio > self.HIGH_SHUFFLE_RATIO:
                bottlenecks.append("SHUFFLE_HEAVY")

        # Check for task parallelism
        total_tasks = job_data.get("total_tasks", 0)
        optimal_tasks = (
            num_executors * executor_cores * 2
        )  # Rule of thumb: 2-3 tasks per core
        if total_tasks > 0 and optimal_tasks > 0:
            parallelism_ratio = total_tasks / optimal_tasks
            if parallelism_ratio < self.LOW_PARALLELISM_RATIO:
                bottlenecks.append("LOW_PARALLELISM")
            elif parallelism_ratio > 10:
                bottlenecks.append("EXCESSIVE_TASKS")

        return bottlenecks

    def _detect_issues(self, job_data: Dict) -> List[Dict]:
        """Detect common Spark job issues.

        Args:
            job_data: Job execution data

        Returns:
            List of detected issues with descriptions
        """
        issues: List[Dict] = []

        # Data skew detection
        # Check if shuffle read/write is disproportionate
        shuffle_read = job_data.get("shuffle_read_bytes", 0)
        shuffle_write = job_data.get("shuffle_write_bytes", 0)
        input_bytes = job_data.get("input_bytes", 1)

        if shuffle_write > 0 and input_bytes > 0:
            # High amplification might indicate skew
            amplification = shuffle_write / input_bytes
            if amplification > 2.0:
                issues.append(
                    {
                        "type": "DATA_SKEW",
                        "severity": "HIGH" if amplification > 5.0 else "MEDIUM",
                        "description": f"High data amplification ({amplification:.1f}x) may indicate data skew",
                        "recommendation": "Consider salting keys or repartitioning data",
                    }
                )

        # Excessive spill to disk
        disk_spilled = job_data.get("disk_spilled_bytes", 0)
        memory_spilled = job_data.get("memory_spilled_bytes", 0)
        if disk_spilled > 0:
            spill_gb = disk_spilled / (1024**3)
            issues.append(
                {
                    "type": "DISK_SPILL",
                    "severity": "HIGH" if spill_gb > 10 else "MEDIUM",
                    "description": f"Spilled {spill_gb:.2f} GB to disk due to memory pressure",
                    "recommendation": "Increase executor memory or reduce partition size",
                }
            )

        if memory_spilled > 0:
            spill_gb = memory_spilled / (1024**3)
            issues.append(
                {
                    "type": "MEMORY_SPILL",
                    "severity": "MEDIUM",
                    "description": f"Spilled {spill_gb:.2f} GB from memory",
                    "recommendation": "Increase spark.memory.fraction or executor memory",
                }
            )

        # Task failures
        failed_tasks = job_data.get("failed_tasks", 0)
        total_tasks = job_data.get("total_tasks", 1)
        if failed_tasks > 0:
            failure_rate = failed_tasks / total_tasks if total_tasks > 0 else 0
            issues.append(
                {
                    "type": "TASK_FAILURES",
                    "severity": "HIGH" if failure_rate > 0.05 else "MEDIUM",
                    "description": f"{failed_tasks} tasks failed ({failure_rate:.1%} failure rate)",
                    "recommendation": "Check executor logs for OOM errors or data issues",
                }
            )

        # Stage failures
        failed_stages = job_data.get("failed_stages", 0)
        if failed_stages > 0:
            issues.append(
                {
                    "type": "STAGE_FAILURES",
                    "severity": "HIGH",
                    "description": f"{failed_stages} stages failed",
                    "recommendation": "Review stage details for root cause",
                }
            )

        # Insufficient parallelism
        num_executors = job_data.get("num_executors", 1)
        executor_cores = job_data.get("executor_cores", 1)
        optimal_partitions = num_executors * executor_cores * 2

        if total_tasks > 0 and total_tasks < optimal_partitions / 2:
            issues.append(
                {
                    "type": "LOW_PARALLELISM",
                    "severity": "MEDIUM",
                    "description": f"Only {total_tasks} tasks for {num_executors * executor_cores} cores",
                    "recommendation": "Increase partition count with repartition() or coalesce()",
                }
            )

        # High GC time
        jvm_gc_time_ms = job_data.get("jvm_gc_time_ms", 0)
        executor_run_time_ms = job_data.get("executor_run_time_ms", 1)
        if jvm_gc_time_ms > 0 and executor_run_time_ms > 0:
            gc_ratio = jvm_gc_time_ms / executor_run_time_ms
            if gc_ratio > self.HIGH_GC_RATIO:
                issues.append(
                    {
                        "type": "HIGH_GC_TIME",
                        "severity": "MEDIUM" if gc_ratio < 0.2 else "HIGH",
                        "description": f"GC time is {gc_ratio:.1%} of execution time",
                        "recommendation": "Increase executor memory or tune GC settings",
                    }
                )

        return issues

    def _calculate_health_score(self, job_data: Dict) -> float:
        """Calculate overall health score for a job.

        Args:
            job_data: Job execution data

        Returns:
            Health score between 0 and 1
        """
        score = 1.0

        # Penalize for task failures
        failed_tasks = job_data.get("failed_tasks", 0)
        total_tasks = job_data.get("total_tasks", 1)
        if failed_tasks > 0 and total_tasks > 0:
            failure_rate = failed_tasks / total_tasks
            score -= min(0.3, failure_rate * 3)  # Up to 30% penalty

        # Penalize for disk spill
        disk_spilled = job_data.get("disk_spilled_bytes", 0)
        input_bytes = job_data.get("input_bytes", 1)
        if disk_spilled > 0 and input_bytes > 0:
            spill_ratio = disk_spilled / input_bytes
            score -= min(0.2, spill_ratio * 0.5)  # Up to 20% penalty

        # Penalize for high GC
        jvm_gc_time_ms = job_data.get("jvm_gc_time_ms", 0)
        executor_run_time_ms = job_data.get("executor_run_time_ms", 1)
        if jvm_gc_time_ms > 0 and executor_run_time_ms > 0:
            gc_ratio = jvm_gc_time_ms / executor_run_time_ms
            if gc_ratio > self.HIGH_GC_RATIO:
                score -= min(0.15, gc_ratio * 0.5)  # Up to 15% penalty

        # Penalize for stage failures
        failed_stages = job_data.get("failed_stages", 0)
        if failed_stages > 0:
            score -= min(0.2, failed_stages * 0.1)  # Up to 20% penalty

        return max(0.0, round(score, 3))

    def compare_jobs(self, job1: Dict, job2: Dict) -> Dict:
        """Compare two jobs and highlight differences.

        Args:
            job1: First job data
            job2: Second job data

        Returns:
            Dictionary containing comparison results
        """
        comparison: Dict[str, Any] = {
            "job1_id": job1.get("app_id"),
            "job2_id": job2.get("app_id"),
            "resource_comparison": {},
            "performance_comparison": {},
            "efficiency_comparison": {},
            "recommendations": [],
        }

        # Compare resource configurations
        resource_fields = [
            ("num_executors", "Number of Executors"),
            ("executor_cores", "Executor Cores"),
            ("executor_memory_mb", "Executor Memory (MB)"),
            ("driver_memory_mb", "Driver Memory (MB)"),
        ]

        for field, label in resource_fields:
            val1 = job1.get(field, 0)
            val2 = job2.get(field, 0)
            diff = val2 - val1 if val1 and val2 else 0
            diff_pct = (diff / val1 * 100) if val1 and val1 != 0 else 0

            comparison["resource_comparison"][field] = {
                "job1": val1,
                "job2": val2,
                "difference": diff,
                "difference_percent": round(diff_pct, 1),
            }

        # Compare performance metrics
        performance_fields = [
            ("duration_ms", "Duration (ms)"),
            ("total_tasks", "Total Tasks"),
            ("input_bytes", "Input Bytes"),
            ("output_bytes", "Output Bytes"),
            ("shuffle_read_bytes", "Shuffle Read Bytes"),
            ("shuffle_write_bytes", "Shuffle Write Bytes"),
        ]

        for field, label in performance_fields:
            val1 = job1.get(field, 0)
            val2 = job2.get(field, 0)
            diff = val2 - val1 if val1 is not None and val2 is not None else 0
            diff_pct = (diff / val1 * 100) if val1 and val1 != 0 else 0

            comparison["performance_comparison"][field] = {
                "job1": val1,
                "job2": val2,
                "difference": diff,
                "difference_percent": round(diff_pct, 1),
            }

        # Compare efficiency
        efficiency1 = self._calculate_resource_efficiency(job1)
        efficiency2 = self._calculate_resource_efficiency(job2)

        for key in efficiency1:
            comparison["efficiency_comparison"][key] = {
                "job1": efficiency1[key],
                "job2": efficiency2[key],
                "difference": round(efficiency2[key] - efficiency1[key], 3),
            }

        # Generate recommendations based on comparison
        recommendations = []

        # Duration improvement
        dur1 = job1.get("duration_ms", 0)
        dur2 = job2.get("duration_ms", 0)
        if dur1 > 0 and dur2 > 0:
            if dur2 < dur1 * 0.8:
                recommendations.append(
                    f"Job 2 is {((dur1 - dur2) / dur1 * 100):.0f}% faster. "
                    "Consider adopting its resource configuration."
                )
            elif dur1 < dur2 * 0.8:
                recommendations.append(
                    f"Job 1 is {((dur2 - dur1) / dur2 * 100):.0f}% faster. "
                    "Consider adopting its resource configuration."
                )

        # Resource efficiency
        if efficiency2["overall_efficiency"] > efficiency1["overall_efficiency"] + 0.1:
            recommendations.append(
                "Job 2 has better overall resource efficiency. "
                "Review its configuration for optimization opportunities."
            )
        elif (
            efficiency1["overall_efficiency"] > efficiency2["overall_efficiency"] + 0.1
        ):
            recommendations.append(
                "Job 1 has better overall resource efficiency. "
                "Review its configuration for optimization opportunities."
            )

        comparison["recommendations"] = recommendations
        return comparison

    def generate_summary(self, jobs: List[Dict]) -> Dict:
        """Generate summary statistics for multiple jobs.

        Args:
            jobs: List of job data dictionaries

        Returns:
            Summary statistics
        """
        if not jobs:
            return {
                "total_jobs": 0,
                "avg_duration_ms": 0,
                "total_input_bytes": 0,
                "total_output_bytes": 0,
            }

        durations = [j.get("duration_ms", 0) for j in jobs if j.get("duration_ms")]
        input_bytes_list = [j.get("input_bytes", 0) for j in jobs]
        output_bytes_list = [j.get("output_bytes", 0) for j in jobs]
        failed_tasks_list = [j.get("failed_tasks", 0) for j in jobs]
        total_tasks_list = [j.get("total_tasks", 0) for j in jobs]

        # Calculate success rate
        total_failed = sum(failed_tasks_list)
        total_tasks = sum(total_tasks_list)
        success_rate = (
            (total_tasks - total_failed) / total_tasks if total_tasks > 0 else 1.0
        )

        # Calculate average efficiency
        efficiencies = [self._calculate_resource_efficiency(j) for j in jobs]
        avg_efficiency = {
            "cpu_efficiency": (
                statistics.mean([e["cpu_efficiency"] for e in efficiencies])
                if efficiencies
                else 0
            ),
            "memory_efficiency": (
                statistics.mean([e["memory_efficiency"] for e in efficiencies])
                if efficiencies
                else 0
            ),
            "io_efficiency": (
                statistics.mean([e["io_efficiency"] for e in efficiencies])
                if efficiencies
                else 0
            ),
            "overall_efficiency": (
                statistics.mean([e["overall_efficiency"] for e in efficiencies])
                if efficiencies
                else 0
            ),
        }

        # Identify common issues
        all_issues: Dict[str, int] = {}
        for job in jobs:
            issues = self._detect_issues(job)
            for issue in issues:
                issue_type = issue["type"]
                all_issues[issue_type] = all_issues.get(issue_type, 0) + 1

        common_issues = sorted(
            [
                {"type": k, "count": v, "percentage": v / len(jobs) * 100}
                for k, v in all_issues.items()
            ],
            key=lambda x: x["count"],  # type: ignore[arg-type, return-value]
            reverse=True,
        )

        return {
            "total_jobs": len(jobs),
            "avg_duration_ms": statistics.mean(durations) if durations else 0,
            "median_duration_ms": statistics.median(durations) if durations else 0,
            "min_duration_ms": min(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "std_duration_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "total_input_bytes": sum(input_bytes_list),
            "total_output_bytes": sum(output_bytes_list),
            "avg_input_bytes": (
                statistics.mean(input_bytes_list) if input_bytes_list else 0
            ),
            "avg_output_bytes": (
                statistics.mean(output_bytes_list) if output_bytes_list else 0
            ),
            "task_success_rate": round(success_rate, 4),
            "avg_efficiency": avg_efficiency,
            "common_issues": common_issues[:5],  # Top 5 issues
        }
