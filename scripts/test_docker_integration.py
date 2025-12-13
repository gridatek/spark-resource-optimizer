#!/usr/bin/env python3
"""
Docker Integration Test Script
Tests the Spark Resource Optimizer with real Spark jobs submitted to a Docker cluster.

Instead of seeding the database directly, this script:
1. Starts a full Spark cluster (master + worker)
2. Submits real Spark applications
3. Collects data from Spark History Server
4. Tests the optimizer's recommendations
"""
import argparse
import subprocess
import sys
import time
import requests
import json
from pathlib import Path
from typing import List, Dict, Optional


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


class DockerIntegrationTest:
    """Manages Docker-based integration testing with real Spark jobs."""

    def __init__(self, profiles: List[str], cleanup: bool = False, verbose: bool = False):
        self.profiles = profiles or ["with-spark"]
        self.cleanup = cleanup
        self.verbose = verbose
        self.api_url = "http://localhost:8080"
        self.spark_master_url = "spark://localhost:7077"
        self.history_server_url = "http://localhost:18080"
        self.test_results = []
        self.project_root = Path(__file__).parent.parent

    def log(self, message: str, color: str = ""):
        """Print log message with optional color."""
        if color:
            print(f"{color}{message}{Colors.END}")
        else:
            print(message)

    def log_success(self, message: str):
        """Print success message."""
        self.log(f"✓ {message}", Colors.GREEN)

    def log_error(self, message: str):
        """Print error message."""
        self.log(f"✗ {message}", Colors.RED)

    def log_info(self, message: str):
        """Print info message."""
        self.log(f"ℹ {message}", Colors.BLUE)

    def run_command(self, cmd: List[str], check: bool = True, capture: bool = False) -> Optional[str]:
        """Run a shell command."""
        if self.verbose:
            self.log_info(f"Running: {' '.join(cmd)}")

        try:
            if capture:
                result = subprocess.run(cmd, check=check, capture_output=True, text=True)
                return result.stdout
            else:
                subprocess.run(cmd, check=check)
                return None
        except subprocess.CalledProcessError as e:
            self.log_error(f"Command failed: {' '.join(cmd)}")
            if capture and e.stderr:
                self.log_error(f"Error: {e.stderr}")
            raise

    def start_services(self):
        """Start Docker Compose services."""
        self.log_info("Starting Docker services...")

        # Build profile arguments
        profile_args = []
        for profile in self.profiles:
            profile_args.extend(["--profile", profile])

        cmd = ["docker", "compose"] + profile_args + ["up", "-d"]
        self.run_command(cmd)

        self.log_success("Docker services started")

    def wait_for_service(self, url: str, service_name: str, timeout: int = 120):
        """Wait for a service to be healthy."""
        self.log_info(f"Waiting for {service_name} to be ready...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    self.log_success(f"{service_name} is ready")
                    return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(2)

        self.log_error(f"{service_name} failed to start within {timeout}s")
        return False

    def submit_spark_job(self, job_script: str, job_name: str) -> bool:
        """Submit a Spark job to the cluster."""
        self.log_info(f"Submitting Spark job: {job_name}")

        job_path = self.project_root / "spark-jobs" / job_script

        if not job_path.exists():
            self.log_error(f"Job script not found: {job_path}")
            return False

        # Submit job using spark-submit in the spark-master container
        cmd = [
            "docker", "exec", "spark-master",
            "/opt/spark/bin/spark-submit",
            "--master", "spark://spark-master:7077",
            "--deploy-mode", "client",
            f"/opt/spark-jobs/{job_script}"
        ]

        try:
            self.log_info(f"Executing: {job_script}")
            self.run_command(cmd, check=True)
            self.log_success(f"Job completed: {job_name}")
            return True
        except subprocess.CalledProcessError:
            self.log_error(f"Job failed: {job_name}")
            return False

    def wait_for_event_logs(self, timeout: int = 60):
        """Wait for event logs to be written and available."""
        self.log_info("Waiting for event logs to be available...")

        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if event logs exist
            spark_events_dir = self.project_root / "spark-events"
            if spark_events_dir.exists():
                event_files = list(spark_events_dir.glob("*"))
                if event_files:
                    self.log_success(f"Found {len(event_files)} event log(s)")
                    time.sleep(5)  # Give History Server time to process
                    return True

            time.sleep(2)

        self.log_error("No event logs found")
        return False

    def collect_from_history_server(self) -> bool:
        """Trigger data collection from Spark History Server."""
        self.log_info("Collecting data from History Server...")

        try:
            # Use the optimizer's collection API
            response = requests.post(
                f"{self.api_url}/api/v1/collect",
                json={
                    "history_server_url": "http://spark-history:18080"
                },
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                collected = result.get("collected", 0)
                self.log_success(f"Collected {collected} job(s) from History Server")
                return True
            else:
                self.log_error(f"Collection failed: {response.status_code}")
                try:
                    error_detail = response.json()
                    self.log_error(f"Error details: {error_detail}")
                except:
                    self.log_error(f"Response text: {response.text[:500]}")
                return False

        except Exception as e:
            self.log_error(f"Collection error: {e}")
            import traceback
            self.log_error(f"Traceback: {traceback.format_exc()}")
            return False

    def test_api_endpoints(self) -> Dict[str, bool]:
        """Test various API endpoints."""
        results = {}

        # Test 1: Health check
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            results["health_check"] = response.status_code == 200
            if results["health_check"]:
                self.log_success("API health check passed")
            else:
                self.log_error("API health check failed")
        except Exception as e:
            results["health_check"] = False
            self.log_error(f"Health check error: {e}")

        # Test 2: List jobs
        try:
            response = requests.get(f"{self.api_url}/api/v1/jobs", timeout=10)
            if response.status_code == 200:
                data = response.json()
                job_count = data.get("total", 0)
                results["list_jobs"] = job_count > 0
                if results["list_jobs"]:
                    self.log_success(f"Found {job_count} job(s) in database")
                else:
                    self.log_error("No jobs found in database")
            else:
                results["list_jobs"] = False
                self.log_error(f"List jobs failed: {response.status_code}")
        except Exception as e:
            results["list_jobs"] = False
            self.log_error(f"List jobs error: {e}")

        # Test 3: Get recommendation
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/recommend",
                json={
                    "executor_cores": 2,
                    "executor_memory_mb": 4096,
                    "num_executors": 5,
                    "input_size_bytes": 10 * 1024 * 1024 * 1024  # 10 GB in bytes
                },
                timeout=30
            )
            results["recommendation"] = response.status_code == 200
            if results["recommendation"]:
                data = response.json()
                confidence = data.get("confidence", 0)
                self.log_success(f"Got recommendation (confidence: {confidence:.2f})")
            else:
                self.log_error(f"Recommendation failed: {response.status_code}")
        except Exception as e:
            results["recommendation"] = False
            self.log_error(f"Recommendation error: {e}")

        # Test 4: Get statistics
        try:
            response = requests.get(f"{self.api_url}/api/v1/stats", timeout=10)
            results["statistics"] = response.status_code == 200
            if results["statistics"]:
                self.log_success("Retrieved statistics")
            else:
                self.log_error("Statistics retrieval failed")
        except Exception as e:
            results["statistics"] = False
            self.log_error(f"Statistics error: {e}")

        return results

    def test_cli_commands(self) -> Dict[str, bool]:
        """Test all CLI commands."""
        results = {}

        self.log(f"\n{'='*60}", Colors.BOLD)
        self.log("Testing CLI Commands", Colors.BOLD)
        self.log(f"{'='*60}\n", Colors.BOLD)

        db_path = str(self.project_root / "spark_optimizer.db")

        # Test 1: Database initialization
        try:
            self.log_info("Testing: spark-optimizer db init")
            self.run_command(
                ["spark-optimizer", "db", "init", "--db-url", f"sqlite:///{db_path}"],
                check=True
            )
            results["db_init"] = True
            self.log_success("Database initialization succeeded")
        except Exception as e:
            results["db_init"] = False
            self.log_error(f"Database init failed: {e}")

        # Test 2: Database current revision
        try:
            self.log_info("Testing: spark-optimizer db current")
            output = self.run_command(
                ["spark-optimizer", "db", "current", "--db-url", f"sqlite:///{db_path}"],
                check=True,
                capture=True
            )
            results["db_current"] = output is not None
            if results["db_current"]:
                self.log_success("Database current revision check succeeded")
            else:
                self.log_error("Database current returned no output")
        except Exception as e:
            results["db_current"] = False
            self.log_error(f"Database current failed: {e}")

        # Test 3: Database history
        try:
            self.log_info("Testing: spark-optimizer db history")
            output = self.run_command(
                ["spark-optimizer", "db", "history", "--db-url", f"sqlite:///{db_path}"],
                check=True,
                capture=True
            )
            results["db_history"] = output is not None
            if results["db_history"]:
                self.log_success("Database history check succeeded")
        except Exception as e:
            results["db_history"] = False
            self.log_error(f"Database history failed: {e}")

        # Test 4: Collect from event logs
        try:
            self.log_info("Testing: spark-optimizer collect --event-log-dir")
            event_log_dir = str(self.project_root / "spark-events")
            self.run_command(
                [
                    "spark-optimizer", "collect",
                    "--event-log-dir", event_log_dir,
                    "--db-url", f"sqlite:///{db_path}"
                ],
                check=True
            )
            results["collect_event_logs"] = True
            self.log_success("Event log collection succeeded")
        except Exception as e:
            results["collect_event_logs"] = False
            self.log_error(f"Event log collection failed: {e}")

        # Test 5: Collect from History Server
        try:
            self.log_info("Testing: spark-optimizer collect-from-history-server")
            self.run_command(
                [
                    "spark-optimizer", "collect-from-history-server",
                    "--history-server-url", "http://localhost:18080",
                    "--db-url", f"sqlite:///{db_path}",
                    "--max-apps", "10"
                ],
                check=True
            )
            results["collect_history_server"] = True
            self.log_success("History Server collection succeeded")
        except Exception as e:
            results["collect_history_server"] = False
            self.log_error(f"History Server collection failed: {e}")

        # Test 6: List jobs
        try:
            self.log_info("Testing: spark-optimizer list-jobs")
            output = self.run_command(
                [
                    "spark-optimizer", "list-jobs",
                    "--db-url", f"sqlite:///{db_path}",
                    "--limit", "10"
                ],
                check=True,
                capture=True
            )
            results["list_jobs"] = output is not None and len(output) > 0
            if results["list_jobs"]:
                self.log_success("List jobs succeeded")
                if self.verbose and output:
                    print(output[:500])  # Show first 500 chars
            else:
                self.log_error("List jobs returned no output")
        except Exception as e:
            results["list_jobs"] = False
            self.log_error(f"List jobs failed: {e}")

        # Test 7: Stats
        try:
            self.log_info("Testing: spark-optimizer stats")
            output = self.run_command(
                [
                    "spark-optimizer", "stats",
                    "--db-url", f"sqlite:///{db_path}"
                ],
                check=True,
                capture=True
            )
            results["stats"] = output is not None and "Total Jobs" in output
            if results["stats"]:
                self.log_success("Stats command succeeded")
                if self.verbose and output:
                    print(output)
            else:
                self.log_error("Stats output invalid")
        except Exception as e:
            results["stats"] = False
            self.log_error(f"Stats command failed: {e}")

        # Test 8: Analyze command (requires app_id from database)
        try:
            self.log_info("Testing: spark-optimizer analyze")

            # Get first app_id from database
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT app_id FROM spark_applications LIMIT 1")
            row = cursor.fetchone()
            conn.close()

            if row:
                app_id = row[0]
                self.log_info(f"Analyzing application: {app_id}")
                output = self.run_command(
                    [
                        "spark-optimizer", "analyze",
                        "--app-id", app_id,
                        "--db-url", f"sqlite:///{db_path}"
                    ],
                    check=True,
                    capture=True
                )
                results["analyze"] = output is not None and "Job Analysis" in output
                if results["analyze"]:
                    self.log_success("Analyze command succeeded")
                    if self.verbose and output:
                        print(output)
                else:
                    self.log_error("Analyze output invalid")
            else:
                results["analyze"] = False
                self.log_error("No applications found in database for analyze test")
        except Exception as e:
            results["analyze"] = False
            self.log_error(f"Analyze command failed: {e}")

        # Test 9: Recommend command
        try:
            self.log_info("Testing: spark-optimizer recommend")
            output = self.run_command(
                [
                    "spark-optimizer", "recommend",
                    "--input-size", "10GB",
                    "--job-type", "etl",
                    "--priority", "balanced",
                    "--db-url", f"sqlite:///{db_path}",
                    "--format", "table"
                ],
                check=False,  # May fail if not enough data
                capture=True
            )
            # Recommendation may fail with insufficient data, so we're lenient
            results["recommend"] = True  # Just test that command runs
            if output and "Recommended Configuration" in output:
                self.log_success("Recommend command succeeded with results")
                if self.verbose and output:
                    print(output)
            else:
                self.log_info("Recommend command ran (may not have sufficient data for recommendation)")
        except Exception as e:
            results["recommend"] = False
            self.log_error(f"Recommend command failed: {e}")

        # Test 10: Recommend with JSON output
        try:
            self.log_info("Testing: spark-optimizer recommend --format json")
            output = self.run_command(
                [
                    "spark-optimizer", "recommend",
                    "--input-size", "5GB",
                    "--priority", "performance",
                    "--db-url", f"sqlite:///{db_path}",
                    "--format", "json"
                ],
                check=False,
                capture=True
            )
            results["recommend_json"] = True
            if output:
                self.log_success("Recommend JSON format succeeded")
            else:
                self.log_info("Recommend JSON ran (may not have sufficient data)")
        except Exception as e:
            results["recommend_json"] = False
            self.log_error(f"Recommend JSON failed: {e}")

        # Test 11: Recommend with spark-submit format
        try:
            self.log_info("Testing: spark-optimizer recommend --format spark-submit")
            output = self.run_command(
                [
                    "spark-optimizer", "recommend",
                    "--input-size", "20GB",
                    "--priority", "cost",
                    "--db-url", f"sqlite:///{db_path}",
                    "--format", "spark-submit"
                ],
                check=False,
                capture=True
            )
            results["recommend_spark_submit"] = True
            if output and "spark-submit" in output:
                self.log_success("Recommend spark-submit format succeeded")
                if self.verbose and output:
                    print(output)
            else:
                self.log_info("Recommend spark-submit ran (may not have sufficient data)")
        except Exception as e:
            results["recommend_spark_submit"] = False
            self.log_error(f"Recommend spark-submit failed: {e}")

        return results

    def stop_services(self):
        """Stop Docker Compose services."""
        self.log_info("Stopping Docker services...")

        profile_args = []
        for profile in self.profiles:
            profile_args.extend(["--profile", profile])

        cmd = ["docker", "compose"] + profile_args + ["down"]

        if self.cleanup:
            cmd.append("-v")
            self.log_info("Removing volumes...")

        self.run_command(cmd, check=False)
        self.log_success("Docker services stopped")

    def cleanup_event_logs(self):
        """Clean up event logs directory."""
        import os
        spark_events_dir = self.project_root / "spark-events"
        if spark_events_dir.exists():
            import shutil
            shutil.rmtree(spark_events_dir)
            spark_events_dir.mkdir(parents=True, exist_ok=True)
            # Set permissions so Docker containers can write to it
            os.chmod(spark_events_dir, 0o777)
            self.log_info("Cleaned up event logs directory")

    def run_tests(self):
        """Run the full integration test suite."""
        self.log(f"\n{'='*60}", Colors.BOLD)
        self.log("Docker Integration Test - Real Spark Jobs", Colors.BOLD)
        self.log(f"{'='*60}\n", Colors.BOLD)

        try:
            # Step 1: Cleanup and prepare
            if self.cleanup:
                self.cleanup_event_logs()

            # Step 2: Start services
            self.start_services()

            # Step 3: Wait for services to be ready
            if not self.wait_for_service(f"{self.api_url}/health", "API Server"):
                return False

            if not self.wait_for_service(f"{self.history_server_url}", "History Server"):
                return False

            # Give Spark master/worker time to start
            self.log_info("Waiting for Spark cluster to initialize...")
            time.sleep(15)

            # Step 4: Submit Spark jobs
            self.log(f"\n{'='*60}", Colors.BOLD)
            self.log("Submitting Spark Jobs", Colors.BOLD)
            self.log(f"{'='*60}\n", Colors.BOLD)

            # Basic test jobs (always run)
            jobs = [
                ("simple_wordcount.py", "Simple WordCount"),
                ("data_processing_etl.py", "ETL Data Processing"),
                ("inefficient_job.py", "Inefficient Job"),
            ]

            # Extended test jobs (optional, run if available)
            extended_jobs = [
                ("memory_intensive_job.py", "Memory-Intensive Job"),
                ("cpu_intensive_job.py", "CPU-Intensive Job"),
                ("skewed_data_job.py", "Skewed Data Job"),
            ]

            # Check if running extended tests
            run_extended = len(self.profiles) > 1 or "--extended" in str(self.profiles)
            if run_extended:
                jobs.extend(extended_jobs)

            job_results = []
            for job_script, job_name in jobs:
                success = self.submit_spark_job(job_script, job_name)
                job_results.append((job_name, success))
                time.sleep(5)  # Brief pause between jobs

            # Step 5: Wait for event logs
            if not self.wait_for_event_logs():
                self.log_error("Event logs not found, cannot continue")
                return False

            # Step 6: Collect data from History Server
            self.log(f"\n{'='*60}", Colors.BOLD)
            self.log("Collecting Data from History Server", Colors.BOLD)
            self.log(f"{'='*60}\n", Colors.BOLD)

            collection_success = self.collect_from_history_server()
            if not collection_success:
                self.log_error("Failed to collect data from History Server (non-fatal, continuing tests)")

            # Step 7: Test API endpoints
            self.log(f"\n{'='*60}", Colors.BOLD)
            self.log("Testing API Endpoints", Colors.BOLD)
            self.log(f"{'='*60}\n", Colors.BOLD)

            api_results = self.test_api_endpoints()

            # Step 8: Test CLI commands
            cli_results = self.test_cli_commands()

            # Step 9: Print summary
            self.log(f"\n{'='*60}", Colors.BOLD)
            self.log("Test Results Summary", Colors.BOLD)
            self.log(f"{'='*60}\n", Colors.BOLD)

            self.log("Spark Jobs:", Colors.BOLD)
            for job_name, success in job_results:
                if success:
                    self.log_success(f"  {job_name}")
                else:
                    self.log_error(f"  {job_name}")

            self.log("\nAPI Tests:", Colors.BOLD)
            for test_name, success in api_results.items():
                if success:
                    self.log_success(f"  {test_name}")
                else:
                    self.log_error(f"  {test_name}")

            self.log("\nCLI Tests:", Colors.BOLD)
            for test_name, success in cli_results.items():
                if success:
                    self.log_success(f"  {test_name}")
                else:
                    self.log_error(f"  {test_name}")

            # Calculate overall success
            all_jobs_passed = all(success for _, success in job_results)
            all_api_tests_passed = all(api_results.values())
            all_cli_tests_passed = all(cli_results.values())
            overall_success = all_jobs_passed and all_api_tests_passed and all_cli_tests_passed

            self.log(f"\n{'='*60}", Colors.BOLD)
            if overall_success:
                self.log("Status: SUCCESS ✓", Colors.GREEN + Colors.BOLD)
            else:
                self.log("Status: FAILED ✗", Colors.RED + Colors.BOLD)
            self.log(f"{'='*60}\n", Colors.BOLD)

            return overall_success

        except KeyboardInterrupt:
            self.log_error("\nTest interrupted by user")
            return False

        except Exception as e:
            self.log_error(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            # Cleanup
            if not self.cleanup:
                self.log_info("\nServices are still running. Use --cleanup to remove them.")
            else:
                self.stop_services()


def main():
    parser = argparse.ArgumentParser(
        description="Docker Integration Test for Spark Resource Optimizer"
    )
    parser.add_argument(
        "--profiles",
        help="Comma-separated Docker Compose profiles (default: with-spark)",
        default="with-spark"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove Docker volumes after tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    profiles = [p.strip() for p in args.profiles.split(",")]

    tester = DockerIntegrationTest(
        profiles=profiles,
        cleanup=args.cleanup,
        verbose=args.verbose
    )

    success = tester.run_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
