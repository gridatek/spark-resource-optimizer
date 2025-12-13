"""
Command-line interface for Spark Resource Optimizer
"""

import click
import sys
from pathlib import Path
from tabulate import tabulate
import json
from datetime import datetime
from typing import Optional

# Import our modules (adjust imports based on your package structure)
from spark_optimizer.collectors.event_log_collector import EventLogCollector
from spark_optimizer.collectors.history_server_collector import HistoryServerCollector
from spark_optimizer.collectors.metrics_collector import MetricsCollector
from spark_optimizer.storage.database import Database
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Spark Resource Optimizer - Optimize Spark job resources based on historical data"""
    pass


@cli.command()
@click.option(
    "--event-log-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing Spark event logs",
)
@click.option(
    "--db-url",
    default="sqlite:///spark_optimizer.db",
    help="Database URL (default: sqlite:///spark_optimizer.db)",
)
@click.option(
    "--batch-size",
    default=100,
    type=int,
    help="Number of logs to process in each batch",
)
def collect(event_log_dir: str, db_url: str, batch_size: int):
    """Collect metrics from Spark event logs and store in database"""

    click.echo(f"Collecting metrics from: {event_log_dir}")

    collector = EventLogCollector(event_log_dir)
    db = Database(db_url)

    processed = 0
    errors = 0

    with click.progressbar(
        collector.collect_all(), label="Processing event logs", length=None
    ) as bar:
        for job_metrics in bar:
            try:
                # Convert dataclass to dict
                from dataclasses import asdict

                job_dict = asdict(job_metrics)

                db.save_job(job_dict)
                processed += 1

                if processed % batch_size == 0:
                    click.echo(f"\nProcessed {processed} jobs so far...")

            except Exception as e:
                errors += 1
                if errors <= 5:  # Only show first 5 errors
                    click.echo(f"\nError processing job: {e}", err=True)

    click.echo(f"\n✓ Successfully processed {processed} jobs")
    if errors > 0:
        click.echo(f"✗ Failed to process {errors} jobs", err=True)


@cli.command(name="collect-from-history-server")
@click.option(
    "--history-server-url",
    required=True,
    help="URL of Spark History Server (e.g., http://localhost:18080)",
)
@click.option(
    "--db-url",
    default="sqlite:///spark_optimizer.db",
    help="Database URL (default: sqlite:///spark_optimizer.db)",
)
@click.option(
    "--max-apps",
    default=100,
    type=int,
    help="Maximum number of applications to fetch (default: 100)",
)
@click.option(
    "--status",
    default="completed",
    type=click.Choice(["completed", "running", "all"]),
    help="Application status filter (default: completed)",
)
@click.option(
    "--timeout",
    default=30,
    type=int,
    help="Request timeout in seconds (default: 30)",
)
def collect_from_history_server(
    history_server_url: str, db_url: str, max_apps: int, status: str, timeout: int
):
    """Collect metrics from Spark History Server REST API and store in database"""

    click.echo(f"Connecting to History Server: {history_server_url}")

    # Initialize collector
    config = {"max_apps": max_apps, "status": status, "timeout": timeout}
    collector = HistoryServerCollector(history_server_url, config)

    # Validate connection
    if not collector.validate_config():
        click.echo(
            "✗ Failed to connect to History Server. Please check the URL and try again.",
            err=True,
        )
        sys.exit(1)

    click.echo("✓ Successfully connected to History Server")

    # Initialize database
    db = Database(db_url)

    # Collect jobs
    click.echo(f"Fetching up to {max_apps} applications with status={status}...")

    try:
        job_data = collector.collect()

        if not job_data:
            click.echo("No applications found matching criteria")
            return

        click.echo(f"Retrieved {len(job_data)} applications")

        # Save to database
        processed = 0
        errors = 0

        with click.progressbar(
            job_data, label="Saving to database", show_pos=True
        ) as bar:
            for job in bar:
                try:
                    db.save_job(job)
                    processed += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:  # Only show first 5 errors
                        click.echo(
                            f"\nError saving job {job.get('app_id')}: {e}", err=True
                        )

        click.echo(f"\n✓ Successfully processed {processed} applications")
        if errors > 0:
            click.echo(f"✗ Failed to process {errors} applications", err=True)

    except Exception as e:
        click.echo(f"✗ Error collecting from History Server: {e}", err=True)
        sys.exit(1)


@cli.command(name="collect-from-metrics")
@click.option(
    "--metrics-endpoint",
    required=True,
    help="URL of Prometheus/metrics endpoint (e.g., http://localhost:9090)",
)
@click.option(
    "--db-url",
    default="sqlite:///spark_optimizer.db",
    help="Database URL (default: sqlite:///spark_optimizer.db)",
)
@click.option(
    "--lookback-hours",
    default=24,
    type=int,
    help="Hours to look back for metrics (default: 24)",
)
@click.option(
    "--timeout",
    default=30,
    type=int,
    help="Request timeout in seconds (default: 30)",
)
@click.option(
    "--step",
    default="1m",
    help="Query step interval (default: 1m)",
)
@click.option(
    "--no-verify-ssl",
    is_flag=True,
    help="Disable SSL certificate verification",
)
def collect_from_metrics(
    metrics_endpoint: str,
    db_url: str,
    lookback_hours: int,
    timeout: int,
    step: str,
    no_verify_ssl: bool,
):
    """Collect metrics from Prometheus or other metrics systems"""

    click.echo(f"Connecting to metrics endpoint: {metrics_endpoint}")

    # Initialize collector
    config = {
        "lookback_hours": lookback_hours,
        "timeout": timeout,
        "step": step,
        "verify_ssl": not no_verify_ssl,
    }

    collector = MetricsCollector(metrics_endpoint, config=config)

    # Validate connection
    if not collector.validate_config():
        click.echo(
            "✗ Failed to connect to metrics endpoint. Please check the URL and connectivity.",
            err=True,
        )
        sys.exit(1)

    click.echo("✓ Successfully connected to metrics endpoint")

    # Initialize database
    db = Database(db_url)

    # Collect metrics
    click.echo(
        f"Collecting metrics from the last {lookback_hours} hours with step={step}..."
    )

    try:
        job_data = collector.collect()

        if not job_data:
            click.echo("No job metrics found")
            return

        click.echo(f"Retrieved metrics for {len(job_data)} jobs")

        # Save to database
        processed = 0
        errors = 0

        with click.progressbar(
            job_data, label="Saving to database", show_pos=True
        ) as bar:
            for job in bar:
                try:
                    db.save_job(job)
                    processed += 1
                except Exception as e:
                    errors += 1
                    if errors <= 5:  # Only show first 5 errors
                        click.echo(
                            f"\nError saving job {job.get('app_id', 'unknown')}: {e}",
                            err=True,
                        )

        click.echo(f"\n✓ Successfully processed {processed} jobs")
        if errors > 0:
            click.echo(f"✗ Failed to process {errors} jobs", err=True)

    except Exception as e:
        click.echo(f"✗ Error collecting metrics: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--input-size", required=True, help="Expected input data size (e.g., 10GB, 500MB)"
)
@click.option("--job-type", default=None, help="Type of job (e.g., etl, ml, streaming)")
@click.option(
    "--sla", default=None, type=int, help="Maximum acceptable duration in minutes"
)
@click.option(
    "--budget", default=None, type=float, help="Maximum acceptable cost in dollars"
)
@click.option(
    "--priority",
    default="balanced",
    type=click.Choice(["performance", "cost", "balanced"]),
    help="Optimization priority",
)
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
@click.option(
    "--format",
    "output_format",
    default="table",
    type=click.Choice(["table", "json", "spark-submit"]),
    help="Output format",
)
def recommend(
    input_size: str,
    job_type: Optional[str],
    sla: Optional[int],
    budget: Optional[float],
    priority: str,
    db_url: str,
    output_format: str,
):
    """Get resource recommendations for a Spark job"""

    # Parse input size
    input_bytes = parse_size_string(input_size)

    if input_bytes is None:
        click.echo(
            "Error: Invalid input size format. Use format like '10GB' or '500MB'",
            err=True,
        )
        sys.exit(1)

    # Get recommendation
    db = Database(db_url)
    recommender = SimilarityRecommender(db)

    click.echo(f"Analyzing similar jobs for {input_size} input...")

    try:
        rec = recommender.recommend(
            input_size_bytes=input_bytes,
            job_type=job_type,
            sla_minutes=sla,
            budget_dollars=budget,
            priority=priority,
        )

        # Output recommendation
        if output_format == "json":
            output_json(rec)
        elif output_format == "spark-submit":
            output_spark_submit(rec)
        else:
            output_table(rec)

    except Exception as e:
        click.echo(f"Error generating recommendation: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
@click.option("--limit", default=20, type=int, help="Number of jobs to display")
@click.option("--job-type", default=None, help="Filter by job type")
def list_jobs(db_url: str, limit: int, job_type: Optional[str]):
    """List recent Spark jobs in the database"""

    db = Database(db_url)

    from spark_optimizer.storage.models import SparkApplication

    with db.get_session() as session:
        query = session.query(SparkApplication)

        if job_type:
            query = query.filter(SparkApplication.tags.contains({"job_type": job_type}))

        query = query.order_by(SparkApplication.start_time.desc()).limit(limit)
        jobs = query.all()

        if not jobs:
            click.echo("No jobs found in database")
            return

        # Format for display
        table_data = []
        for job in jobs:
            duration_min = job.duration_ms / 60000 if job.duration_ms else 0
            input_gb = job.input_bytes / (1024**3)

            table_data.append(
                [
                    job.app_id[:20],
                    job.app_name[:30],
                    job.start_time.strftime("%Y-%m-%d %H:%M"),
                    f"{duration_min:.1f}",
                    f"{input_gb:.2f}",
                    f"{job.num_executors}",
                    f"{job.executor_cores}",
                    f"{job.executor_memory_mb // 1024}GB",
                ]
            )

        headers = [
            "App ID",
            "Name",
            "Start Time",
            "Duration (min)",
            "Input (GB)",
            "Executors",
            "Cores",
            "Memory",
        ]

        click.echo(tabulate(table_data, headers=headers, tablefmt="grid"))
        click.echo(f"\nTotal jobs: {len(jobs)}")


@cli.command()
@click.option("--app-id", required=True, help="Application ID to analyze")
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
def analyze(app_id: str, db_url: str):
    """Analyze a specific Spark job and provide optimization suggestions"""

    db = Database(db_url)

    from spark_optimizer.storage.models import SparkApplication

    with db.get_session() as session:
        job = (
            session.query(SparkApplication)
            .filter(SparkApplication.app_id == app_id)
            .first()
        )

        if not job:
            click.echo(f"Job {app_id} not found in database", err=True)
            sys.exit(1)

        # Display job details
        click.echo(f"\n{'='*60}")
        click.echo(f"Job Analysis: {job.app_name}")
        click.echo(f"{'='*60}\n")

        click.echo(f"Application ID: {job.app_id}")
        click.echo(f"Duration: {job.duration_ms / 60000:.2f} minutes")
        click.echo(f"Input Data: {job.input_bytes / (1024**3):.2f} GB")
        click.echo(f"Shuffle Data: {job.shuffle_write_bytes / (1024**3):.2f} GB")

        click.echo(f"\nResource Configuration:")
        click.echo(f"  Executors: {job.num_executors}")
        click.echo(f"  Cores per executor: {job.executor_cores}")
        click.echo(f"  Memory per executor: {job.executor_memory_mb} MB")

        # Analyze for issues
        click.echo(f"\n{'='*60}")
        click.echo("Optimization Opportunities:")
        click.echo(f"{'='*60}\n")

        issues_found = False

        # Check for spilling
        if job.disk_spilled_bytes and job.disk_spilled_bytes > 0:
            click.echo(
                f"⚠ Disk spill detected: {job.disk_spilled_bytes / (1024**3):.2f} GB"
            )
            click.echo(
                "  → Consider increasing executor memory or reducing partition size\n"
            )
            issues_found = True

        # Check for GC issues
        if job.jvm_gc_time_ms and job.executor_run_time_ms:
            gc_ratio = (
                job.jvm_gc_time_ms / job.executor_run_time_ms
                if job.executor_run_time_ms > 0
                else 0
            )
            if gc_ratio > 0.1:
                click.echo(f"⚠ High GC time: {gc_ratio*100:.1f}% of execution time")
                click.echo(
                    "  → Consider increasing executor memory or reducing memory pressure\n"
                )
                issues_found = True

        # Check for task failures
        if job.failed_tasks and job.failed_tasks > 0:
            click.echo(f"⚠ Task failures: {job.failed_tasks} out of {job.total_tasks}")
            click.echo(
                "  → Investigate failure causes and consider adjusting resources\n"
            )
            issues_found = True

        # Check shuffle ratio
        if job.input_bytes and job.input_bytes > 0:
            shuffle_ratio = job.shuffle_write_bytes / job.input_bytes
            if shuffle_ratio > 0.5:
                click.echo(f"⚠ High shuffle ratio: {shuffle_ratio*100:.1f}%")
                click.echo(
                    "  → Consider optimizing join strategies or increasing shuffle partitions\n"
                )
                issues_found = True

        if not issues_found:
            click.echo("✓ No major issues detected. Job appears well-optimized.")


@cli.command()
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
def stats(db_url: str):
    """Display statistics about collected data"""

    db = Database(db_url)

    from spark_optimizer.storage.models import SparkApplication
    from sqlalchemy import func

    with db.get_session() as session:
        total_jobs = session.query(func.count(SparkApplication.id)).scalar()

        if total_jobs == 0:
            click.echo("No jobs in database")
            return

        # Calculate statistics
        avg_duration = (
            session.query(func.avg(SparkApplication.duration_ms)).scalar() or 0
        )
        total_input = (
            session.query(func.sum(SparkApplication.input_bytes)).scalar() or 0
        )

        click.echo(f"\n{'='*60}")
        click.echo("Database Statistics")
        click.echo(f"{'='*60}\n")

        click.echo(f"Total Jobs: {total_jobs}")
        click.echo(f"Average Duration: {avg_duration / 60000:.2f} minutes")
        click.echo(f"Total Data Processed: {total_input / (1024**3):.2f} GB")

        # Job type distribution
        click.echo(f"\nJob Type Distribution:")
        job_types = (
            session.query(SparkApplication.app_name, func.count(SparkApplication.id))
            .group_by(SparkApplication.app_name)
            .all()
        )

        for job_type, count in job_types:
            if job_type:
                click.echo(f"  {job_type}: {count}")


@cli.command()
@click.option(
    "--host", default="0.0.0.0", help="Host to bind the server to"  # nosec B104
)
@click.option("--port", default=8080, type=int, help="Port to bind the server to")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option(
    "--db-url",
    default="sqlite:///spark_optimizer.db",
    help="Database URL",
    envvar="SPARK_OPTIMIZER_DB_URL",
)
def serve(host: str, port: int, debug: bool, db_url: str):
    """Start the API server"""
    from spark_optimizer.api.server import run_server

    click.echo(f"Starting Spark Resource Optimizer API on {host}:{port}")
    click.echo(f"Database: {db_url}")
    click.echo(f"Debug mode: {'enabled' if debug else 'disabled'}")

    run_server(host=host, port=port, debug=debug, db_url=db_url)


# Database migration commands
@cli.group()
def db():
    """Database migration commands"""
    pass


@db.command(name="init")
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
def db_init(db_url: str):
    """Initialize database schema (run all migrations)"""
    click.echo(f"Initializing database: {db_url}")

    try:
        database = Database(db_url)
        database.create_tables()
        click.echo("✓ Database initialized successfully")
    except Exception as e:
        click.echo(f"✗ Failed to initialize database: {e}", err=True)
        sys.exit(1)


@db.command(name="upgrade")
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
@click.option(
    "--revision",
    default="head",
    help="Target revision (default: head for latest)",
)
def db_upgrade(db_url: str, revision: str):
    """Upgrade database to a later version"""
    click.echo(f"Upgrading database to revision: {revision}")

    try:
        database = Database(db_url)
        database.run_migrations(revision)
        click.echo("✓ Database upgrade completed successfully")
    except Exception as e:
        click.echo(f"✗ Failed to upgrade database: {e}", err=True)
        sys.exit(1)


@db.command(name="downgrade")
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
@click.option(
    "--revision",
    default="-1",
    help="Target revision (default: -1 for previous)",
)
def db_downgrade(db_url: str, revision: str):
    """Downgrade database to a previous version"""
    click.echo(f"Downgrading database to revision: {revision}")

    try:
        database = Database(db_url)
        database.downgrade_migration(revision)
        click.echo("✓ Database downgrade completed successfully")
    except Exception as e:
        click.echo(f"✗ Failed to downgrade database: {e}", err=True)
        sys.exit(1)


@db.command(name="current")
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
def db_current(db_url: str):
    """Display current database revision"""
    from alembic import command
    from alembic.config import Config

    try:
        # Get the project root directory (where alembic.ini is located)
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini_path = project_root / "alembic.ini"

        if not alembic_ini_path.exists():
            click.echo(
                f"✗ Alembic configuration not found at {alembic_ini_path}", err=True
            )
            sys.exit(1)

        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # Show current revision
        click.echo("Current database revision:")
        command.current(alembic_cfg, verbose=True)

    except Exception as e:
        click.echo(f"✗ Failed to get current revision: {e}", err=True)
        sys.exit(1)


@db.command(name="history")
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed migration information",
)
def db_history(db_url: str, verbose: bool):
    """Display migration history"""
    from alembic import command
    from alembic.config import Config

    try:
        # Get the project root directory (where alembic.ini is located)
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini_path = project_root / "alembic.ini"

        if not alembic_ini_path.exists():
            click.echo(
                f"✗ Alembic configuration not found at {alembic_ini_path}", err=True
            )
            sys.exit(1)

        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # Show migration history
        click.echo("Migration history:")
        command.history(alembic_cfg, verbose=verbose)

    except Exception as e:
        click.echo(f"✗ Failed to get migration history: {e}", err=True)
        sys.exit(1)


@db.command(name="stamp")
@click.option("--db-url", default="sqlite:///spark_optimizer.db", help="Database URL")
@click.option(
    "--revision",
    required=True,
    help="Revision to stamp database with",
)
def db_stamp(db_url: str, revision: str):
    """Mark database as being at a specific revision without running migrations"""
    from alembic import command
    from alembic.config import Config

    click.echo(f"Stamping database with revision: {revision}")

    try:
        # Get the project root directory (where alembic.ini is located)
        project_root = Path(__file__).parent.parent.parent.parent
        alembic_ini_path = project_root / "alembic.ini"

        if not alembic_ini_path.exists():
            click.echo(
                f"✗ Alembic configuration not found at {alembic_ini_path}", err=True
            )
            sys.exit(1)

        # Create Alembic config
        alembic_cfg = Config(str(alembic_ini_path))
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

        # Stamp database
        command.stamp(alembic_cfg, revision)
        click.echo("✓ Database stamped successfully")

    except Exception as e:
        click.echo(f"✗ Failed to stamp database: {e}", err=True)
        sys.exit(1)


def parse_size_string(size_str: str) -> Optional[int]:
    """Parse size string like '10GB' to bytes"""
    size_str = size_str.upper().strip()

    # Check units from longest to shortest to avoid matching "B" in "GB"
    units = [
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("B", 1),
    ]

    for unit, multiplier in units:
        if size_str.endswith(unit):
            try:
                value = float(size_str[: -len(unit)].strip())
                return int(value * multiplier)
            except ValueError:
                return None

    return None


def output_table(rec):
    """Output recommendation as formatted table"""
    click.echo(f"\n{'='*60}")
    click.echo("Recommended Configuration")
    click.echo(f"{'='*60}\n")

    config = rec["configuration"]

    config_data = [
        ["Executors", config.get("num_executors", "N/A")],
        ["Cores per executor", config.get("executor_cores", "N/A")],
        ["Memory per executor", f"{config.get('executor_memory_mb', 0)} MB"],
        ["Driver memory", f"{config.get('driver_memory_mb', 0)} MB"],
    ]

    click.echo(tabulate(config_data, headers=["Parameter", "Value"], tablefmt="grid"))

    click.echo(f"\n{'='*60}")
    click.echo("Predictions")
    click.echo(f"{'='*60}\n")

    click.echo(f"Confidence: {rec.get('confidence', 0):.0%}")
    click.echo(f"Method: {rec.get('metadata', {}).get('method', 'unknown')}")

    if rec.get("metadata", {}).get("similar_jobs_count"):
        click.echo(
            f"\nBased on {rec['metadata']['similar_jobs_count']} similar historical jobs"
        )


def output_json(rec):
    """Output recommendation as JSON"""
    # rec is already a dict, not a dataclass
    click.echo(json.dumps(rec, indent=2, default=str))


def output_spark_submit(rec):
    """Output recommendation as spark-submit command"""
    config = rec["configuration"]

    click.echo("spark-submit \\")
    click.echo(f"  --num-executors {config.get('num_executors', 5)} \\")
    click.echo(f"  --executor-cores {config.get('executor_cores', 4)} \\")
    click.echo(f"  --executor-memory {config.get('executor_memory_mb', 8192)}m \\")
    click.echo(f"  --driver-memory {config.get('driver_memory_mb', 4096)}m \\")
    click.echo("  your-application.jar")


def main():
    """Main entry point for the CLI"""
    cli()


if __name__ == "__main__":
    main()
