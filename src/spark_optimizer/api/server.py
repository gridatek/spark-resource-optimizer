"""
REST API Server for Spark Resource Optimizer
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from dataclasses import asdict
from typing import Optional
import logging

from spark_optimizer.storage.database import Database
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender
from spark_optimizer.recommender.rule_based_recommender import RuleBasedRecommender
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


def _ensure_initialized() -> tuple[Database, SimilarityRecommender, RuleBasedRecommender]:
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

        # Convert to dict
        rec_dict = asdict(rec)

        # Save recommendation to database
        db.save_recommendation(
            {
                "user": request.remote_addr,
                "job_name": data.get("job_name", "unknown"),
                "input_size_bytes": input_size,
                "job_type": job_type,
                "sla_minutes": sla_minutes,
                "budget_dollars": budget_dollars,
                "recommended_executors": rec.num_executors,
                "recommended_executor_cores": rec.executor_cores,
                "recommended_executor_memory_mb": rec.executor_memory_mb,
                "recommended_configs": {
                    "shuffle_partitions": rec.shuffle_partitions,
                    "dynamic_allocation": rec.dynamic_allocation,
                },
                "predicted_duration_ms": rec.predicted_duration_ms,
                "predicted_cost": rec.predicted_cost,
                "confidence_score": rec.confidence_score,
                "recommendation_method": rec.recommendation_method,
                "similar_job_ids": rec.similar_jobs,
            }
        )

        return jsonify(rec_dict)

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
                        "num_executors": job.num_executors,
                        "executor_cores": job.executor_cores,
                        "executor_memory_mb": job.executor_memory_mb,
                        "input_bytes": job.input_bytes,
                        "shuffle_write_bytes": job.shuffle_write_bytes,
                        "total_tasks": job.total_tasks,
                        "failed_tasks": job.failed_tasks,
                        "estimated_cost": job.estimated_cost,
                    }
                )

            return jsonify({"jobs": jobs_list, "count": len(jobs_list)})

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
