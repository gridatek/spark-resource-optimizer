"""
REST API Server for Spark Resource Optimizer
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from typing import Optional
import logging
import os

from spark_optimizer.storage.database import Database
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender
from spark_optimizer.recommender.rule_based_recommender import RuleBasedRecommender
from spark_optimizer.collectors.history_server_collector import HistoryServerCollector
from spark_optimizer.storage.models import SparkApplication

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize database and recommender - needs to be configured with connection string
# This will be properly initialized when the app starts
db: Optional[Database] = None
recommender: Optional[SimilarityRecommender] = None
rule_based_recommender: Optional[RuleBasedRecommender] = None


def init_app(db_url: str = "sqlite:///spark_optimizer.db"):
    """Initialize the application with database and recommender.

    Args:
        db_url: Database connection string
    """
    global db, recommender, rule_based_recommender
    db = Database(db_url)
    db.create_tables()
    recommender = SimilarityRecommender(db=db)
    rule_based_recommender = RuleBasedRecommender()


def _ensure_initialized() -> (
    tuple[Database, SimilarityRecommender, RuleBasedRecommender]
):
    """Ensure database and recommender are initialized.

    Returns:
        Tuple of (db, recommender, rule_based_recommender) for type checking

    Raises:
        RuntimeError: If application is not initialized
    """
    if db is None or recommender is None or rule_based_recommender is None:
        raise RuntimeError("Application not initialized. Call init_app() first.")
    return (db, recommender, rule_based_recommender)


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {"status": "healthy", "service": "spark-resource-optimizer", "version": "0.1.0"}
    )


@app.route("/api/v1/openapi.yaml", methods=["GET"])
def get_openapi_spec():
    """Serve the OpenAPI specification in YAML format"""
    try:
        # Get the path to the openapi.yaml file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        openapi_path = os.path.join(current_dir, "openapi.yaml")

        if not os.path.exists(openapi_path):
            return jsonify({"error": "OpenAPI specification not found"}), 404

        return send_file(
            openapi_path,
            mimetype="application/x-yaml",
            as_attachment=False,
            download_name="openapi.yaml",
        )
    except Exception as e:
        logger.error(f"Error serving OpenAPI spec: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/recommend", methods=["POST"])
def get_recommendation():
    """
    Get resource recommendations for a Spark job

    Request body:
    {
        "input_size_bytes": 10737418240,  // 10 GB
        "job_type": "etl",  // optional
        "sla_minutes": 30,  // optional
        "budget_dollars": 5.0,  // optional
        "priority": "balanced"  // performance, cost, or balanced
    }
    """
    try:
        db, recommender, _ = _ensure_initialized()

        data = request.get_json()

        # Validate required fields
        if "input_size_bytes" not in data:
            return jsonify({"error": "input_size_bytes is required"}), 400

        input_size = data["input_size_bytes"]
        job_type = data.get("job_type")
        sla_minutes = data.get("sla_minutes")
        budget_dollars = data.get("budget_dollars")
        priority = data.get("priority", "balanced")

        # Validate priority
        if priority not in ["performance", "cost", "balanced"]:
            return (
                jsonify(
                    {"error": "priority must be one of: performance, cost, balanced"}
                ),
                400,
            )

        # Get recommendation
        logger.info(
            f"Generating recommendation for {input_size} bytes, priority={priority}"
        )

        rec = recommender.recommend(
            input_size_bytes=input_size,
            job_type=job_type,
            sla_minutes=sla_minutes,
            budget_dollars=budget_dollars,
            priority=priority,
        )

        # rec is already a dict, no need to convert
        # Extract configuration from rec
        config = rec.get("configuration", rec)

        # Save recommendation to database
        db.save_recommendation(
            {
                "user": request.remote_addr,
                "job_name": data.get("job_name", "unknown"),
                "input_size_bytes": input_size,
                "job_type": job_type,
                "sla_minutes": sla_minutes,
                "budget_dollars": budget_dollars,
                "recommended_executors": config.get(
                    "num_executors", rec.get("num_executors")
                ),
                "recommended_executor_cores": config.get(
                    "executor_cores", rec.get("executor_cores")
                ),
                "recommended_executor_memory_mb": config.get(
                    "executor_memory_mb", rec.get("executor_memory_mb")
                ),
                "recommended_configs": {
                    "shuffle_partitions": rec.get("shuffle_partitions"),
                    "dynamic_allocation": rec.get("dynamic_allocation"),
                },
                "predicted_duration_ms": rec.get("predicted_duration_ms"),
                "predicted_cost": rec.get("predicted_cost"),
                "confidence_score": rec.get("confidence", rec.get("confidence_score")),
                "recommendation_method": rec.get("metadata", {}).get(
                    "method", rec.get("recommendation_method", "similarity")
                ),
                "similar_job_ids": rec.get("similar_jobs", []),
            }
        )

        return jsonify(rec)

    except Exception as e:
        logger.error(f"Error generating recommendation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/jobs", methods=["GET"])
def list_jobs():
    """
    List recent Spark jobs

    Query parameters:
    - limit: Number of jobs to return (default: 20)
    - job_type: Filter by job type
    - min_duration: Minimum duration in seconds
    - max_duration: Maximum duration in seconds
    """
    try:
        db, _, _ = _ensure_initialized()

        limit = request.args.get("limit", 20, type=int)
        job_type = request.args.get("job_type")
        min_duration = request.args.get("min_duration", type=int)
        max_duration = request.args.get("max_duration", type=int)

        with db.get_session() as session:
            query = session.query(SparkApplication)

            if job_type:
                query = query.filter(
                    SparkApplication.tags.contains({"job_type": job_type})
                )

            if min_duration:
                query = query.filter(
                    SparkApplication.duration_ms >= min_duration * 1000
                )

            if max_duration:
                query = query.filter(
                    SparkApplication.duration_ms <= max_duration * 1000
                )

            query = query.order_by(SparkApplication.start_time.desc()).limit(limit)
            jobs = query.all()

            # Convert to dict
            jobs_list = []
            for job in jobs:
                jobs_list.append(
                    {
                        "app_id": job.app_id,
                        "app_name": job.app_name,
                        "user": job.user,
                        "start_time": job.start_time.isoformat(),
                        "end_time": job.end_time.isoformat() if job.end_time else None,
                        "duration_ms": job.duration_ms,
                        "status": job.status,
                        "estimated_cost": job.estimated_cost,
                        "configuration": {
                            "num_executors": job.num_executors,
                            "executor_cores": job.executor_cores,
                            "executor_memory_mb": job.executor_memory_mb,
                            "driver_memory_mb": job.driver_memory_mb,
                        },
                        "metrics": {
                            "total_tasks": job.total_tasks,
                            "failed_tasks": job.failed_tasks,
                            "total_stages": job.total_stages,
                            "input_bytes": job.input_bytes,
                            "output_bytes": job.output_bytes,
                            "shuffle_read_bytes": job.shuffle_read_bytes,
                            "shuffle_write_bytes": job.shuffle_write_bytes,
                            "memory_spilled_bytes": job.memory_spilled_bytes,
                            "disk_spilled_bytes": job.disk_spilled_bytes,
                        },
                    }
                )

            return jsonify({"jobs": jobs_list, "total": len(jobs_list), "limit": limit})

    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/jobs/<app_id>", methods=["GET"])
def get_job(app_id: str):
    """Get details for a specific job"""
    try:
        db, _, _ = _ensure_initialized()

        with db.get_session() as session:
            job = (
                session.query(SparkApplication)
                .filter(SparkApplication.app_id == app_id)
                .first()
            )

            if not job:
                return jsonify({"error": "Job not found"}), 404

            # Build detailed response
            job_dict = {
                "app_id": job.app_id,
                "app_name": job.app_name,
                "user": job.user,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "duration_ms": job.duration_ms,
                "status": job.status,
                "resource_config": {
                    "num_executors": job.num_executors,
                    "executor_cores": job.executor_cores,
                    "executor_memory_mb": job.executor_memory_mb,
                    "driver_memory_mb": job.driver_memory_mb,
                },
                "metrics": {
                    "total_tasks": job.total_tasks,
                    "failed_tasks": job.failed_tasks,
                    "total_stages": job.total_stages,
                    "input_bytes": job.input_bytes,
                    "output_bytes": job.output_bytes,
                    "shuffle_read_bytes": job.shuffle_read_bytes,
                    "shuffle_write_bytes": job.shuffle_write_bytes,
                    "peak_memory_usage": job.peak_memory_usage,
                    "disk_spilled_bytes": job.disk_spilled_bytes,
                    "memory_spilled_bytes": job.memory_spilled_bytes,
                    "executor_run_time_ms": job.executor_run_time_ms,
                    "executor_cpu_time_ms": job.executor_cpu_time_ms,
                    "jvm_gc_time_ms": job.jvm_gc_time_ms,
                },
                "environment": {
                    "cluster_type": job.cluster_type,
                    "instance_type": job.instance_type,
                    "spark_version": job.spark_version,
                },
                "cost": {"estimated_cost": job.estimated_cost},
                "spark_configs": job.spark_configs,
            }

            return jsonify(job_dict)

    except Exception as e:
        logger.error(f"Error getting job: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/jobs/<app_id>/analyze", methods=["GET"])
def analyze_job(app_id: str):
    """Analyze a job and provide optimization suggestions using rule-based engine"""
    try:
        db, _, rb_recommender = _ensure_initialized()

        with db.get_session() as session:
            job = (
                session.query(SparkApplication)
                .filter(SparkApplication.app_id == app_id)
                .first()
            )

            if not job:
                return jsonify({"error": "Job not found"}), 404

            # Build job_data dictionary for rule-based recommender
            job_data = {
                "app_id": job.app_id,
                "app_name": job.app_name,
                "input_bytes": job.input_bytes or 0,
                "output_bytes": job.output_bytes or 0,
                "shuffle_read_bytes": job.shuffle_read_bytes or 0,
                "shuffle_write_bytes": job.shuffle_write_bytes or 0,
                "disk_spilled_bytes": job.disk_spilled_bytes or 0,
                "memory_spilled_bytes": job.memory_spilled_bytes or 0,
                "num_executors": job.num_executors or 0,
                "executor_cores": job.executor_cores or 0,
                "executor_memory_mb": job.executor_memory_mb or 0,
                "driver_memory_mb": job.driver_memory_mb or 0,
                "duration_ms": job.duration_ms or 0,
                "jvm_gc_time": job.jvm_gc_time_ms or 0,
                "total_tasks": job.total_tasks or 0,
                "failed_tasks": job.failed_tasks or 0,
                "total_stages": job.total_stages or 0,
            }

            # Use rule-based recommender for comprehensive analysis
            analysis = rb_recommender.analyze_job(job_data)

            # Return the full analysis with backward-compatible format
            return jsonify(analysis)

    except Exception as e:
        logger.error(f"Error analyzing job: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/collect", methods=["POST"])
def collect_from_history_server():
    """Collect jobs from Spark History Server and store in database

    Request body:
    {
        "history_server_url": "http://localhost:18080",
        "max_apps": 100,  // optional, default 100
        "status": "completed",  // optional, default "completed"
        "min_date": "2024-01-01T00:00:00"  // optional, ISO format
    }

    Returns:
    {
        "success": true,
        "collected": 42,
        "failed": 2,
        "skipped": 5,
        "message": "Successfully collected 42 jobs"
    }
    """
    try:
        db, _, _ = _ensure_initialized()

        data = request.get_json()

        # Validate required fields
        if not data or "history_server_url" not in data:
            return jsonify({"error": "Missing required field: history_server_url"}), 400

        history_server_url = data["history_server_url"]

        # Optional configuration
        config = {
            "max_apps": data.get("max_apps", 100),
            "status": data.get("status", "completed"),
        }

        if "min_date" in data:
            config["min_date"] = data["min_date"]

        # Initialize collector
        collector = HistoryServerCollector(history_server_url, config=config)

        # Validate connectivity
        if not collector.validate_config():
            return (
                jsonify(
                    {
                        "error": f"Cannot connect to History Server at {history_server_url}",
                        "message": "Please verify the URL and ensure the History Server is running",
                    }
                ),
                503,
            )

        # Collect jobs
        logger.info(f"Collecting jobs from {history_server_url}")
        job_data = collector.collect()

        # Store in database
        collected = 0
        failed = 0
        skipped = 0

        with db.get_session() as session:
            for job in job_data:
                try:
                    # Check if job already exists
                    existing = (
                        session.query(SparkApplication)
                        .filter(SparkApplication.app_id == job["app_id"])
                        .first()
                    )

                    if existing:
                        logger.debug(f"Skipping existing job: {job['app_id']}")
                        skipped += 1
                        continue

                    # Create new application record
                    app = SparkApplication(**job)
                    session.add(app)
                    collected += 1

                except Exception as e:
                    logger.error(
                        f"Error storing job {job.get('app_id', 'unknown')}: {e}"
                    )
                    failed += 1
                    continue

            session.commit()

        logger.info(
            f"Collection complete: {collected} collected, {failed} failed, {skipped} skipped"
        )

        return jsonify(
            {
                "success": True,
                "collected": collected,
                "failed": failed,
                "skipped": skipped,
                "message": f"Successfully collected {collected} jobs from History Server",
            }
        )

    except Exception as e:
        logger.error(f"Error collecting from History Server: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/compare", methods=["POST"])
def compare_jobs():
    """Compare multiple Spark jobs to identify patterns and differences

    Request body:
    {
        "app_ids": ["app-1", "app-2", "app-3"],
        "metrics": ["duration_ms", "executor_memory_mb", ...]  // optional, all metrics if not specified
    }

    Returns:
    {
        "jobs": [
            {"app_id": "app-1", "app_name": "...", "duration_ms": 100000, ...},
            {"app_id": "app-2", "app_name": "...", "duration_ms": 150000, ...}
        ],
        "summary": {
            "fastest": {"app_id": "app-1", "duration_ms": 100000},
            "slowest": {"app_id": "app-2", "duration_ms": 150000},
            "most_efficient": {"app_id": "app-1", "efficiency_ratio": 0.85},
            "least_efficient": {"app_id": "app-2", "efficiency_ratio": 0.42}
        },
        "recommendations": [
            {
                "metric": "executor_memory_mb",
                "observation": "app-1 uses 50% less memory but runs 33% faster",
                "recommendation": "Consider reducing executor memory for better cost-efficiency"
            }
        ]
    }
    """
    try:
        db, _, rb_recommender = _ensure_initialized()

        data = request.get_json()

        # Validate required fields
        if not data or "app_ids" not in data:
            return jsonify({"error": "Missing required field: app_ids"}), 400

        app_ids = data["app_ids"]

        if not isinstance(app_ids, list) or len(app_ids) < 2:
            return (
                jsonify(
                    {"error": "app_ids must be a list with at least 2 application IDs"}
                ),
                400,
            )

        if len(app_ids) > 10:
            return jsonify({"error": "Maximum 10 jobs can be compared at once"}), 400

        # Fetch jobs from database
        with db.get_session() as session:
            jobs = (
                session.query(SparkApplication)
                .filter(SparkApplication.app_id.in_(app_ids))
                .all()
            )

            if len(jobs) != len(app_ids):
                found_ids = {job.app_id for job in jobs}
                missing = set(app_ids) - found_ids
                return (
                    jsonify({"error": f"Some jobs not found: {', '.join(missing)}"}),
                    404,
                )

            # Convert jobs to dictionaries with key metrics
            job_data = []
            for job in jobs:
                job_dict = {
                    "app_id": job.app_id,
                    "app_name": job.app_name,
                    "duration_ms": job.duration_ms or 0,
                    "executor_memory_mb": job.executor_memory_mb or 0,
                    "executor_cores": job.executor_cores or 0,
                    "num_executors": job.num_executors or 0,
                    "driver_memory_mb": job.driver_memory_mb or 0,
                    "input_bytes": job.input_bytes or 0,
                    "output_bytes": job.output_bytes or 0,
                    "shuffle_read_bytes": job.shuffle_read_bytes or 0,
                    "shuffle_write_bytes": job.shuffle_write_bytes or 0,
                    "disk_spilled_bytes": job.disk_spilled_bytes or 0,
                    "memory_spilled_bytes": job.memory_spilled_bytes or 0,
                    "total_tasks": job.total_tasks or 0,
                    "failed_tasks": job.failed_tasks or 0,
                    "jvm_gc_time_ms": job.jvm_gc_time_ms or 0,
                }

                # Calculate efficiency metrics
                if job_dict["duration_ms"] > 0 and job_dict["input_bytes"] > 0:
                    # Throughput: GB/second
                    job_dict["throughput_gbps"] = (
                        job_dict["input_bytes"] / (1024**3)
                    ) / (job_dict["duration_ms"] / 1000)

                    # Resource efficiency: input processed per executor-hour
                    total_executor_hours = (
                        job_dict["num_executors"]
                        * job_dict["duration_ms"]
                        / (1000 * 3600)
                    )
                    if total_executor_hours > 0:
                        job_dict["data_per_executor_hour_gb"] = (
                            job_dict["input_bytes"] / (1024**3)
                        ) / total_executor_hours
                else:
                    job_dict["throughput_gbps"] = 0
                    job_dict["data_per_executor_hour_gb"] = 0

                # Calculate spill ratio
                if job_dict["input_bytes"] > 0:
                    total_spilled = (
                        job_dict["disk_spilled_bytes"]
                        + job_dict["memory_spilled_bytes"]
                    )
                    job_dict["spill_ratio"] = total_spilled / job_dict["input_bytes"]
                else:
                    job_dict["spill_ratio"] = 0

                job_data.append(job_dict)

            # Find best and worst performers
            jobs_with_duration = [j for j in job_data if j["duration_ms"] > 0]
            jobs_with_throughput = [j for j in job_data if j["throughput_gbps"] > 0]

            summary = {}

            if jobs_with_duration:
                fastest = min(jobs_with_duration, key=lambda x: x["duration_ms"])
                slowest = max(jobs_with_duration, key=lambda x: x["duration_ms"])
                summary["fastest"] = {
                    "app_id": fastest["app_id"],
                    "app_name": fastest["app_name"],
                    "duration_ms": fastest["duration_ms"],
                }
                summary["slowest"] = {
                    "app_id": slowest["app_id"],
                    "app_name": slowest["app_name"],
                    "duration_ms": slowest["duration_ms"],
                }

            if jobs_with_throughput:
                most_efficient = max(
                    jobs_with_throughput, key=lambda x: x["throughput_gbps"]
                )
                least_efficient = min(
                    jobs_with_throughput, key=lambda x: x["throughput_gbps"]
                )
                summary["most_efficient"] = {
                    "app_id": most_efficient["app_id"],
                    "app_name": most_efficient["app_name"],
                    "throughput_gbps": round(most_efficient["throughput_gbps"], 3),
                }
                summary["least_efficient"] = {
                    "app_id": least_efficient["app_id"],
                    "app_name": least_efficient["app_name"],
                    "throughput_gbps": round(least_efficient["throughput_gbps"], 3),
                }

            # Generate recommendations based on comparison
            recommendations = []

            # Check for spilling differences
            jobs_with_spill = [j for j in job_data if j["spill_ratio"] > 0]
            jobs_without_spill = [j for j in job_data if j["spill_ratio"] == 0]

            if jobs_with_spill and jobs_without_spill:
                recommendations.append(
                    {
                        "type": "spilling",
                        "observation": f"{len(jobs_with_spill)} job(s) experienced spilling while "
                        f"{len(jobs_without_spill)} did not",
                        "recommendation": "Jobs without spilling had sufficient memory allocation. "
                        "Consider matching their executor memory configuration.",
                    }
                )

            # Check for resource allocation differences
            if jobs_with_throughput:
                avg_memory = sum(j["executor_memory_mb"] for j in job_data) / len(
                    job_data
                )
                high_throughput_jobs = [
                    j
                    for j in jobs_with_throughput
                    if j["throughput_gbps"] > most_efficient["throughput_gbps"] * 0.8
                ]

                if high_throughput_jobs:
                    avg_memory_efficient = sum(
                        j["executor_memory_mb"] for j in high_throughput_jobs
                    ) / len(high_throughput_jobs)

                    if avg_memory_efficient < avg_memory * 0.8:
                        recommendations.append(
                            {
                                "type": "resource_optimization",
                                "observation": "Higher-throughput jobs use less memory on average",
                                "recommendation": f"Consider reducing executor memory to around {int(avg_memory_efficient)}MB "
                                "for better cost-efficiency",
                            }
                        )

            # Check for task failure patterns
            jobs_with_failures = [j for j in job_data if j["failed_tasks"] > 0]
            jobs_without_failures = [j for j in job_data if j["failed_tasks"] == 0]

            if jobs_with_failures and jobs_without_failures:
                recommendations.append(
                    {
                        "type": "reliability",
                        "observation": f"{len(jobs_with_failures)} job(s) had task failures",
                        "recommendation": "Analyze successful jobs' configurations and apply similar settings. "
                        "Consider enabling adaptive execution.",
                    }
                )

            return jsonify(
                {
                    "jobs": job_data,
                    "summary": summary,
                    "recommendations": recommendations,
                    "compared_count": len(job_data),
                }
            )

    except Exception as e:
        logger.error(f"Error comparing jobs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/v1/stats", methods=["GET"])
def get_stats():
    """Get aggregate statistics"""
    try:
        db, _, _ = _ensure_initialized()

        from sqlalchemy import func

        with db.get_session() as session:
            total_jobs = session.query(func.count(SparkApplication.id)).scalar()
            avg_duration = session.query(
                func.avg(SparkApplication.duration_ms)
            ).scalar()
            total_input = session.query(func.sum(SparkApplication.input_bytes)).scalar()

            # Job type distribution
            job_types = (
                session.query(
                    SparkApplication.app_name, func.count(SparkApplication.id)
                )
                .group_by(SparkApplication.app_name)
                .all()
            )

            job_type_dist = {jt: count for jt, count in job_types if jt}

            return jsonify(
                {
                    "total_jobs": total_jobs or 0,
                    "avg_duration_ms": int(avg_duration) if avg_duration else 0,
                    "total_data_processed_bytes": total_input or 0,
                    "job_type_distribution": job_type_dist,
                }
            )

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def run_server(
    host="0.0.0.0",  # nosec B104
    port=8080,
    debug=False,
    db_url: str = "sqlite:///spark_optimizer.db",
):
    """Run the Flask server"""
    init_app(db_url=db_url)
    logger.info(f"Starting Spark Resource Optimizer API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_server(debug=True)
