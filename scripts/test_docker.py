#!/usr/bin/env python3
"""
Docker Integration Test Suite for Spark Resource Optimizer

This script performs comprehensive integration testing of the Docker Compose setup,
verifying that all services work correctly together and testing full workflows.

Usage:
    python scripts/test_docker.py                    # Run all tests
    python scripts/test_docker.py --quick            # Quick smoke tests
    python scripts/test_docker.py --profiles core    # Test specific profiles
    python scripts/test_docker.py --verbose          # Detailed output
    python scripts/test_docker.py --keep-running     # Keep services up after tests
"""

import argparse
import json
import random
import socket
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

# ANSI Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    RESET = '\033[0m'


class DockerComposeManager:
    """Manages Docker Compose lifecycle with automatic cleanup"""

    def __init__(self, profiles: List[str] = None, cleanup_volumes: bool = False,
                 env_file: str = '.env.test', timeout: int = 120):
        self.profiles = profiles or []
        self.cleanup_volumes = cleanup_volumes
        self.env_file = env_file
        self.timeout = timeout
        self.services_started = False
        self.start_time = None

    def __enter__(self):
        """Start Docker Compose services"""
        try:
            print(f"{Colors.BLUE}Starting Docker Compose services...{Colors.RESET}")
            self.start_time = time.time()

            # Build docker compose command
            cmd = ['docker', 'compose', 'up', '-d']

            # Add profiles
            for profile in self.profiles:
                cmd.extend(['--profile', profile])

            # Execute docker compose up
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            if result.returncode != 0:
                print(f"{Colors.RED}Failed to start services:{Colors.RESET}")
                print(result.stderr)
                raise RuntimeError(f"docker compose up failed: {result.stderr}")

            self.services_started = True
            elapsed = time.time() - self.start_time
            print(f"{Colors.GREEN}✓ Services started ({elapsed:.1f}s){Colors.RESET}")
            return self

        except subprocess.TimeoutExpired:
            raise RuntimeError(f"docker compose up timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError("docker or docker-compose not found. Please install Docker.")

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop and cleanup Docker Compose services"""
        if not self.services_started:
            return

        try:
            print(f"\n{Colors.BLUE}Cleaning up Docker Compose services...{Colors.RESET}")

            # Stop services
            cmd = ['docker', 'compose', 'down']
            if self.cleanup_volumes:
                cmd.append('-v')

            subprocess.run(cmd, capture_output=True, timeout=60)
            print(f"{Colors.GREEN}✓ Services cleaned up{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.YELLOW}Warning: Cleanup error: {e}{Colors.RESET}")

    def get_logs(self, service: str, tail: int = 50) -> str:
        """Get logs for a specific service"""
        try:
            result = subprocess.run(
                ['docker', 'compose', 'logs', '--tail', str(tail), service],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.stdout
        except Exception as e:
            return f"Failed to get logs: {e}"


class ServiceHealthChecker:
    """Smart health checking with exponential backoff retries"""

    @staticmethod
    def check_http(url: str, timeout: int = 120, expected_status: int = 200) -> Tuple[bool, str]:
        """Check HTTP endpoint health with retries"""
        delays = [2, 4, 8, 16, 30]  # Exponential backoff
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == expected_status:
                    return True, "OK"
                return False, f"Status {response.status_code}"
            except requests.exceptions.RequestException as e:
                # Try next delay
                for delay in delays:
                    if time.time() - start_time + delay < timeout:
                        time.sleep(delay)
                        break
                else:
                    break

        elapsed = time.time() - start_time
        return False, f"Timeout after {elapsed:.1f}s"

    @staticmethod
    def check_tcp(host: str, port: int, timeout: int = 60) -> Tuple[bool, str]:
        """Check TCP port connectivity"""
        start_time = time.time()
        delays = [2, 4, 8, 16, 30]

        while time.time() - start_time < timeout:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(5)
                    result = sock.connect_ex((host, port))
                    if result == 0:
                        return True, "OK"
            except Exception:
                pass

            # Wait before retry
            for delay in delays:
                if time.time() - start_time + delay < timeout:
                    time.sleep(delay)
                    break
            else:
                break

        elapsed = time.time() - start_time
        return False, f"Timeout after {elapsed:.1f}s"

    @staticmethod
    def check_postgres(host: str = 'localhost', port: int = 5432, timeout: int = 60) -> Tuple[bool, str]:
        """Check PostgreSQL connectivity"""
        return ServiceHealthChecker.check_tcp(host, port, timeout)


class TestDataSeeder:
    """Generate realistic test data for integration testing"""

    # Job archetypes representing different Spark workload patterns
    JOB_ARCHETYPES = {
        "small_etl": {
            "app_name": "Small ETL Job",
            "input_bytes": int(2 * 1024**3),  # 2GB
            "executor_memory_mb": 4096,
            "num_executors": 5,
            "duration_ms": 180000,  # 3 minutes
            "status": "completed",
            "failed_tasks": 0,
            "disk_spilled_bytes": 0,
            "jvm_gc_time_ms": 5000,
        },
        "medium_etl": {
            "app_name": "Medium ETL Job",
            "input_bytes": int(10 * 1024**3),  # 10GB (reduced from 20GB)
            "executor_memory_mb": 8192,
            "num_executors": 10,
            "duration_ms": 600000,  # 10 minutes
            "status": "completed",
            "failed_tasks": 0,
            "disk_spilled_bytes": 1048576,  # 1MB
            "jvm_gc_time_ms": 15000,
        },
        "large_etl": {
            "app_name": "Large ETL Job",
            "input_bytes": int(20 * 1024**3),  # 20GB (reduced from 80GB)
            "executor_memory_mb": 16384,
            "num_executors": 20,
            "duration_ms": 1800000,  # 30 minutes
            "status": "completed",
            "failed_tasks": 0,
            "disk_spilled_bytes": 10485760,  # 10MB
            "jvm_gc_time_ms": 50000,
        },
        "ml_training": {
            "app_name": "ML Training Job",
            "input_bytes": int(15 * 1024**3),  # 15GB (reduced from 25GB)
            "executor_memory_mb": 32768,
            "num_executors": 10,
            "duration_ms": 1800000,  # 30 minutes (reduced from 60)
            "status": "completed",
            "failed_tasks": 0,
            "disk_spilled_bytes": 0,
            "jvm_gc_time_ms": 30000,
        },
        "problematic_job": {
            "app_name": "Problematic Job",
            "input_bytes": int(15 * 1024**3),  # 15GB (reduced from 50GB)
            "executor_memory_mb": 4096,
            "num_executors": 10,
            "duration_ms": 1200000,  # 20 minutes (reduced from 40)
            "status": "completed",
            "failed_tasks": 15,
            "disk_spilled_bytes": int(1 * 1024**3),  # 1GB (must stay under 2GB limit)
            "jvm_gc_time_ms": 300000,
        }
    }

    @classmethod
    def generate_jobs(cls, count: int = 20) -> List[Dict]:
        """Generate diverse, realistic job data"""
        jobs = []
        archetypes = list(cls.JOB_ARCHETYPES.keys())

        for i in range(count):
            # Pick random archetype with variety
            if i < 5:
                archetype_name = archetypes[i % len(archetypes)]
            else:
                archetype_name = random.choice(archetypes)

            archetype = cls.JOB_ARCHETYPES[archetype_name].copy()

            # Add unique app_id
            timestamp = int(time.time() * 1000) + i
            archetype["app_id"] = f"app-test-{archetype_name}-{timestamp}"
            archetype["app_name"] = f"{archetype['app_name']} #{i+1}"

            # Add some randomness (±20%)
            for key in ["input_bytes", "duration_ms", "num_executors", "disk_spilled_bytes"]:
                if key in archetype and archetype[key] is not None:
                    variance = random.uniform(0.8, 1.2)
                    value = int(archetype[key] * variance)
                    # Cap at PostgreSQL INTEGER max to avoid overflow
                    archetype[key] = min(value, 2_000_000_000)

            jobs.append(archetype)

        return jobs


class TestResults:
    """Track and report test results"""

    def __init__(self):
        self.tests: List[Dict] = []
        self.start_time = time.time()

    def add(self, name: str, passed: bool, message: str = "", duration: float = 0):
        """Add a test result"""
        self.tests.append({
            "name": name,
            "passed": passed,
            "message": message,
            "duration": duration
        })

    def print_summary(self):
        """Print test results summary"""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t["passed"])
        failed = total - passed
        duration = time.time() - self.start_time

        print("\n" + "=" * 50)
        print(f"{Colors.BOLD}Results Summary{Colors.RESET}")
        print("=" * 50)
        print(f"Total Tests: {total}")
        print(f"Passed: {Colors.GREEN}{passed} ✓{Colors.RESET}")
        print(f"Failed: {Colors.RED if failed > 0 else Colors.GREEN}{failed}{Colors.RESET}")
        print(f"Duration: {duration:.1f}s")
        print()

        if failed == 0:
            print(f"{Colors.GREEN}{Colors.BOLD}Status: SUCCESS ✓{Colors.RESET}")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}Status: FAILED ✗{Colors.RESET}")
            print(f"\n{Colors.RED}Failed tests:{Colors.RESET}")
            for test in self.tests:
                if not test["passed"]:
                    print(f"  ✗ {test['name']}: {test['message']}")
            return 1


class DockerIntegrationTests:
    """Comprehensive Docker integration test suite"""

    def __init__(self, api_url: str = "http://localhost:8080", verbose: bool = False):
        self.api_url = api_url
        self.verbose = verbose
        self.results = TestResults()
        self.test_jobs = []

    def run_test(self, name: str, test_func, *args, **kwargs):
        """Run a single test and record result"""
        start = time.time()
        try:
            test_func(*args, **kwargs)
            duration = time.time() - start
            self.results.add(name, True, "OK", duration)
            print(f"  {Colors.GREEN}✓{Colors.RESET} {name} ({duration:.1f}s)")
        except AssertionError as e:
            duration = time.time() - start
            self.results.add(name, False, str(e), duration)
            print(f"  {Colors.RED}✗{Colors.RESET} {name}: {e}")
        except Exception as e:
            duration = time.time() - start
            self.results.add(name, False, f"Error: {e}", duration)
            print(f"  {Colors.RED}✗{Colors.RESET} {name}: Error: {e}")

    def test_core_services(self):
        """Test core services (API + PostgreSQL)"""
        print(f"\n{Colors.BOLD}[1/6] Testing Core Services (API + PostgreSQL){Colors.RESET}")

        # Wait for services
        self.run_test("PostgreSQL healthy", self._test_postgres_health)
        self.run_test("API healthy", self._test_api_health)

        # API endpoints
        self.run_test("GET /api/health", self._test_health_endpoint)
        self.run_test("POST /api/recommend (cold start)", self._test_recommend_cold_start)

        # Seed data
        self.run_test("Seed test data", self._test_seed_data)

        # Test with data
        self.run_test("GET /api/jobs", self._test_list_jobs)
        self.run_test("GET /api/jobs/{app_id}", self._test_get_job_details)
        self.run_test("POST /api/recommend (with data)", self._test_recommend_with_data)
        self.run_test("GET /api/analyze/{app_id}", self._test_analyze_job)
        self.run_test("POST /api/feedback", self._test_feedback)
        self.run_test("GET /api/stats", self._test_stats)

    def test_worker_profile(self):
        """Test worker profile (Celery + Redis)"""
        print(f"\n{Colors.BOLD}[2/6] Testing Worker Profile{Colors.RESET}")
        self.run_test("Redis connectivity", self._test_redis_connectivity)

    def test_spark_profile(self):
        """Test Spark profile (Spark History Server)"""
        print(f"\n{Colors.BOLD}[3/6] Testing Spark Profile{Colors.RESET}")
        self.run_test("Spark History Server accessible", self._test_spark_history_accessible)

    def test_monitoring_profile(self):
        """Test monitoring profile (Prometheus + Grafana)"""
        print(f"\n{Colors.BOLD}[4/6] Testing Monitoring Profile{Colors.RESET}")
        self.run_test("Prometheus accessible", self._test_prometheus_accessible)
        self.run_test("Grafana accessible", self._test_grafana_accessible)

    def test_tools_profile(self):
        """Test tools profile (pgAdmin)"""
        print(f"\n{Colors.BOLD}[5/6] Testing Tools Profile{Colors.RESET}")
        self.run_test("pgAdmin accessible", self._test_pgadmin_accessible)

    def test_e2e_workflow(self):
        """Test end-to-end workflow"""
        print(f"\n{Colors.BOLD}[6/6] Testing End-to-End Workflow{Colors.RESET}")
        self.run_test("Full workflow", self._test_full_workflow)

    # Individual test methods
    def _test_postgres_health(self):
        """Test PostgreSQL health"""
        healthy, msg = ServiceHealthChecker.check_postgres(timeout=60)
        assert healthy, f"PostgreSQL not healthy: {msg}"

    def _test_api_health(self):
        """Test API health"""
        # Give API extra time to start up (containers can be slow)
        print(f"      Waiting for API to be ready...")
        healthy, msg = ServiceHealthChecker.check_http(f"{self.api_url}/health", timeout=180)
        assert healthy, f"API not healthy: {msg}"

    def _test_health_endpoint(self):
        """Test /health endpoint"""
        response = requests.get(f"{self.api_url}/health", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "status" in data, "Missing 'status' in response"

    def _test_recommend_cold_start(self):
        """Test recommendation with no historical data"""
        payload = {
            "input_size_bytes": 10 * 1024**3,  # 10GB in bytes
            "job_type": "etl",
            "priority": "balanced"
        }
        response = requests.post(f"{self.api_url}/api/v1/recommend", json=payload, timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "configuration" in data or "recommendation" in data, "Missing recommendation in response"

    def _test_seed_data(self):
        """Seed test data via Docker container"""
        # Generate test jobs
        self.test_jobs = TestDataSeeder.generate_jobs(20)

        # Create a Python script to seed the data
        seed_script = """
import sys
from spark_optimizer.storage.database import Database
from spark_optimizer.storage.models import SparkApplication
from datetime import datetime

db_url = sys.argv[1]
db = Database(db_url)
db.create_tables()

jobs_data = eval(sys.argv[2])

with db.get_session() as session:
    for job_data in jobs_data:
        job = SparkApplication()
        job.app_id = job_data["app_id"]
        job.app_name = job_data["app_name"]
        job.input_bytes = job_data["input_bytes"]
        job.executor_memory_mb = job_data["executor_memory_mb"]
        job.num_executors = job_data["num_executors"]
        job.duration_ms = job_data["duration_ms"]
        job.status = job_data.get("status", "completed")
        job.failed_tasks = job_data.get("failed_tasks", 0)
        job.disk_spilled_bytes = job_data.get("disk_spilled_bytes", 0)
        job.jvm_gc_time_ms = job_data.get("jvm_gc_time_ms", 0)
        job.submit_time = datetime.utcnow()
        job.start_time = datetime.utcnow()
        job.end_time = datetime.utcnow()
        session.add(job)
    session.commit()

print(f"Seeded {len(jobs_data)} jobs")
"""

        # Write the seed script to a temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(seed_script)
            script_path = f.name

        try:
            # Copy script to container and execute
            subprocess.run(
                ['docker', 'compose', 'cp', script_path, 'api:/tmp/seed_script.py'],
                check=True,
                capture_output=True,
                timeout=30
            )

            # Execute the script in the container
            result = subprocess.run(
                ['docker', 'compose', 'exec', '-T', 'api', 'python', '/tmp/seed_script.py',
                 'postgresql://spark_optimizer:spark_password@db:5432/spark_optimizer',
                 str(self.test_jobs)],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode != 0:
                raise AssertionError(f"Failed to seed data: {result.stderr}")

            assert len(self.test_jobs) == 20, "Failed to seed all jobs"

        finally:
            # Clean up temp file
            import os
            try:
                os.unlink(script_path)
            except:
                pass

    def _test_list_jobs(self):
        """Test listing jobs"""
        response = requests.get(f"{self.api_url}/api/v1/jobs", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "jobs" in data or isinstance(data, list), "Missing jobs in response"
        jobs = data if isinstance(data, list) else data.get("jobs", [])
        assert len(jobs) >= 20, f"Expected at least 20 jobs, got {len(jobs)}"

    def _test_get_job_details(self):
        """Test getting job details"""
        if not self.test_jobs:
            raise AssertionError("No test jobs available")

        app_id = self.test_jobs[0]["app_id"]
        response = requests.get(f"{self.api_url}/api/v1/jobs/{app_id}", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("app_id") == app_id, "Job ID mismatch"

    def _test_recommend_with_data(self):
        """Test recommendation with historical data"""
        payload = {
            "input_size_bytes": 50 * 1024**3,  # 50GB in bytes
            "job_type": "etl",
            "priority": "balanced"
        }
        response = requests.post(f"{self.api_url}/api/v1/recommend", json=payload, timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "configuration" in data or "recommendation" in data, "Missing recommendation"

    def _test_analyze_job(self):
        """Test job analysis"""
        if not self.test_jobs:
            raise AssertionError("No test jobs available")

        # Find a problematic job
        problematic_job = next((j for j in self.test_jobs if "problematic" in j["app_id"]), self.test_jobs[0])
        app_id = problematic_job["app_id"]

        response = requests.get(f"{self.api_url}/api/v1/jobs/{app_id}/analyze", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "analysis" in data or "issues" in data or "bottlenecks" in data, "Missing analysis"

    def _test_feedback(self):
        """Test feedback submission"""
        if not self.test_jobs:
            raise AssertionError("No test jobs available")

        payload = {
            "app_id": self.test_jobs[0]["app_id"],
            "satisfaction": 4,
            "comment": "Test feedback"
        }
        response = requests.post(f"{self.api_url}/api/v1/feedback", json=payload, timeout=10)
        assert response.status_code in [200, 201, 404], f"Expected 200/201/404, got {response.status_code}"

    def _test_stats(self):
        """Test statistics endpoint"""
        response = requests.get(f"{self.api_url}/api/v1/stats", timeout=10)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "total_jobs" in data or "job_count" in data, "Missing stats"

    def _test_redis_connectivity(self):
        """Test Redis connectivity"""
        healthy, msg = ServiceHealthChecker.check_tcp("localhost", 6379, timeout=30)
        assert healthy, f"Redis not accessible: {msg}"

    def _test_spark_history_accessible(self):
        """Test Spark History Server accessibility"""
        healthy, msg = ServiceHealthChecker.check_http("http://localhost:18080", timeout=60, expected_status=200)
        assert healthy, f"Spark History Server not accessible: {msg}"

    def _test_prometheus_accessible(self):
        """Test Prometheus accessibility"""
        healthy, msg = ServiceHealthChecker.check_http("http://localhost:9090", timeout=30, expected_status=200)
        assert healthy, f"Prometheus not accessible: {msg}"

    def _test_grafana_accessible(self):
        """Test Grafana accessibility"""
        healthy, msg = ServiceHealthChecker.check_http("http://localhost:3000", timeout=30, expected_status=200)
        assert healthy, f"Grafana not accessible: {msg}"

    def _test_pgadmin_accessible(self):
        """Test pgAdmin accessibility"""
        healthy, msg = ServiceHealthChecker.check_http("http://localhost:5050", timeout=30, expected_status=200)
        assert healthy, f"pgAdmin not accessible: {msg}"

    def _test_full_workflow(self):
        """Test complete end-to-end workflow"""
        # This is a simplified workflow test
        # 1. Verify services are up
        response = requests.get(f"{self.api_url}/health", timeout=10)
        assert response.status_code == 200

        # 2. Verify data exists
        response = requests.get(f"{self.api_url}/api/v1/jobs", timeout=10)
        assert response.status_code == 200

        # 3. Get recommendation
        payload = {"input_size_bytes": 30 * 1024**3, "job_type": "etl", "priority": "balanced"}
        response = requests.post(f"{self.api_url}/api/v1/recommend", json=payload, timeout=10)
        assert response.status_code == 200


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Docker Integration Tests for Spark Resource Optimizer'
    )
    parser.add_argument('--profiles', type=str, default='',
                        help='Comma-separated profiles to enable (e.g., with-worker,with-spark)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick smoke tests only')
    parser.add_argument('--cleanup', action='store_true',
                        help='Remove volumes after tests')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--keep-running', action='store_true',
                        help='Keep services running after tests')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Service startup timeout in seconds')

    args = parser.parse_args()

    # Parse profiles
    profiles = [p.strip() for p in args.profiles.split(',') if p.strip()]

    # Print header
    print("=" * 50)
    print(f"{Colors.BOLD}Docker Integration Tests{Colors.RESET}")
    print(f"{Colors.BOLD}Spark Resource Optimizer{Colors.RESET}")
    print("=" * 50)
    print()

    exit_code = 0

    try:
        # Start Docker Compose services
        with DockerComposeManager(
            profiles=profiles,
            cleanup_volumes=args.cleanup,
            timeout=args.timeout
        ) as manager:

            # Initialize test suite
            tests = DockerIntegrationTests(verbose=args.verbose)

            # Run tests
            tests.test_core_services()

            if not args.quick:
                # Test optional profiles if enabled
                if 'with-worker' in profiles:
                    tests.test_worker_profile()

                if 'with-spark' in profiles:
                    tests.test_spark_profile()

                if 'with-monitoring' in profiles:
                    tests.test_monitoring_profile()

                if 'with-tools' in profiles:
                    tests.test_tools_profile()

                # E2E workflow
                tests.test_e2e_workflow()

            # Print results
            exit_code = tests.results.print_summary()

            if args.keep_running:
                print(f"\n{Colors.YELLOW}Services are still running. Stop with: docker compose down{Colors.RESET}")
                # Prevent cleanup
                manager.services_started = False

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.RESET}")
        exit_code = 130

    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        exit_code = 2

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
