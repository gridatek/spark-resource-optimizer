"""API routes for the Spark Resource Optimizer."""

from flask import Blueprint, request, jsonify, current_app, g
from typing import Dict, Optional
import logging

from spark_optimizer.storage.database import Database
from spark_optimizer.storage.models import SparkApplication
from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender
from spark_optimizer.recommender.rule_based_recommender import RuleBasedRecommender
from spark_optimizer.analyzer.job_analyzer import JobAnalyzer
from spark_optimizer.collectors.event_log_collector import EventLogCollector
from spark_optimizer.collectors.history_server_collector import HistoryServerCollector
from spark_optimizer.collectors.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


def get_db() -> Database:
    """Get or create database connection for this request.

    Returns:
        Database instance
    """
    if "db" not in g:
        db_url = current_app.config.get("DATABASE_URL", "sqlite:///spark_optimizer.db")
        g.db = Database(db_url)
    return g.db


def get_recommender() -> SimilarityRecommender:
    """Get or create recommender for this request.

    Returns:
        SimilarityRecommender instance
    """
    if "recommender" not in g:
        g.recommender = SimilarityRecommender(db=get_db())
    return g.recommender


def get_rule_recommender() -> RuleBasedRecommender:
    """Get or create rule-based recommender for this request.

    Returns:
        RuleBasedRecommender instance
    """
    if "rule_recommender" not in g:
        g.rule_recommender = RuleBasedRecommender()
    return g.rule_recommender


def get_analyzer() -> JobAnalyzer:
    """Get or create job analyzer for this request.

    Returns:
        JobAnalyzer instance
    """
    if "analyzer" not in g:
        g.analyzer = JobAnalyzer()
    return g.analyzer


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint.

    Returns:
        JSON response with health status
    """
    return jsonify({"status": "healthy", "service": "spark-resource-optimizer"})


@api_bp.route("/recommend", methods=["POST"])
def get_recommendation():
    """Get resource recommendations for a job.

    Request body:
        {
            "input_size_gb": float,
            "job_type": str (optional),
            "app_name": str (optional),
            "sla_minutes": int (optional),
            "budget_dollars": float (optional),
            "priority": str (optional, default: "balanced")
        }

    Returns:
        JSON response with recommendations
    """
    try:
        # Handle missing or incorrect Content-Type header
        try:
            data = request.get_json(force=False, silent=False)
        except Exception as json_error:
            return (
                jsonify(
                    {
                        "error": "Invalid or missing Content-Type header. Expected 'application/json'"
                    }
                ),
                415,
            )

        if not data:
            return jsonify({"error": "Request body is required"}), 400

        # Support both input_size_gb and input_size_bytes
        if "input_size_gb" in data:
            input_size_bytes = int(data["input_size_gb"] * (1024**3))
        elif "input_size_bytes" in data:
            input_size_bytes = data["input_size_bytes"]
        else:
            return (
                jsonify(
                    {
                        "error": "Missing required parameter: input_size_gb or input_size_bytes"
                    }
                ),
                400,
            )

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
        recommender = get_recommender()
        recommendation = recommender.recommend(
            input_size_bytes=input_size_bytes,
            job_type=job_type,
            sla_minutes=sla_minutes,
            budget_dollars=budget_dollars,
            priority=priority,
        )

        return jsonify(recommendation), 200

    except Exception as e:
        logger.error(f"Error generating recommendation: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/collect", methods=["POST"])
def collect_job_data():
    """Collect and store job data.

    Request body:
        {
            "source_type": str (event_log, history_server, metrics),
            "source_path": str (path or URL depending on source_type),
            "config": dict (optional)
        }

    Returns:
        JSON response with collection status
    """
    try:
        data = request.get_json()

        if not data or "source_type" not in data:
            return jsonify({"error": "Missing required parameter: source_type"}), 400

        source_type = data["source_type"]
        source_path = data.get("source_path")
        config = data.get("config", {})

        # Initialize appropriate collector
        if source_type == "event_log":
            if not source_path:
                return (
                    jsonify(
                        {"error": "source_path is required for event_log collector"}
                    ),
                    400,
                )
            collector = EventLogCollector(source_path)

        elif source_type == "history_server":
            if not source_path:
                return (
                    jsonify(
                        {
                            "error": "source_path (URL) is required for history_server collector"
                        }
                    ),
                    400,
                )
            collector = HistoryServerCollector(source_path, config)
            if not collector.validate_config():
                return (
                    jsonify(
                        {
                            "error": f"Cannot connect to History Server at {source_path}",
                            "message": "Please verify the URL and ensure the History Server is running",
                        }
                    ),
                    503,
                )

        elif source_type == "metrics":
            if not source_path:
                return (
                    jsonify(
                        {
                            "error": "source_path (endpoint URL) is required for metrics collector"
                        }
                    ),
                    400,
                )
            collector = MetricsCollector(source_path, config)
            if not collector.validate_config():
                return (
                    jsonify(
                        {
                            "error": f"Cannot connect to metrics endpoint at {source_path}",
                            "message": "Please verify the URL and ensure the metrics server is running",
                        }
                    ),
                    503,
                )

        else:
            return (
                jsonify(
                    {
                        "error": f"Unknown source_type: {source_type}. Supported: event_log, history_server, metrics"
                    }
                ),
                400,
            )

        # Collect jobs
        logger.info(f"Collecting jobs from {source_type}: {source_path}")
        job_data = collector.collect()

        # Store in database
        db = get_db()
        collected = 0
        failed = 0
        skipped = 0

        with db.get_session() as session:
            for job in job_data:
                try:
                    # Check if job already exists
                    existing = (
                        session.query(SparkApplication)
                        .filter(SparkApplication.app_id == job.get("app_id"))
                        .first()
                    )

                    if existing:
                        skipped += 1
                        continue

                    # Handle dataclass conversion if needed
                    if hasattr(job, "__dataclass_fields__"):
                        from dataclasses import asdict

                        job = asdict(job)

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

        return (
            jsonify(
                {
                    "status": "success",
                    "jobs_collected": collected,
                    "jobs_skipped": skipped,
                    "jobs_failed": failed,
                    "message": f"Collected {collected} jobs from {source_type}",
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error collecting job data: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/jobs", methods=["GET"])
def list_jobs():
    """List stored jobs with optional filtering.

    Query parameters:
        - limit: int (default 50)
        - offset: int (default 0)
        - app_name: str (optional)
        - date_from: str (optional, ISO format)
        - date_to: str (optional, ISO format)
        - min_duration: int (optional, in seconds)
        - max_duration: int (optional, in seconds)

    Returns:
        JSON response with job list
    """
    try:
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        app_name = request.args.get("app_name")
        date_from = request.args.get("date_from")
        date_to = request.args.get("date_to")
        min_duration = request.args.get("min_duration", type=int)
        max_duration = request.args.get("max_duration", type=int)

        db = get_db()

        with db.get_session() as session:
            from sqlalchemy import func
            from datetime import datetime

            query = session.query(SparkApplication)

            # Apply filters
            if app_name:
                query = query.filter(SparkApplication.app_name.ilike(f"%{app_name}%"))

            if date_from:
                try:
                    from_date = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
                    query = query.filter(SparkApplication.start_time >= from_date)
                except ValueError:
                    pass

            if date_to:
                try:
                    to_date = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
                    query = query.filter(SparkApplication.start_time <= to_date)
                except ValueError:
                    pass

            if min_duration:
                query = query.filter(
                    SparkApplication.duration_ms >= min_duration * 1000
                )

            if max_duration:
                query = query.filter(
                    SparkApplication.duration_ms <= max_duration * 1000
                )

            # Get total count before pagination
            total = query.count()

            # Apply pagination
            query = query.order_by(SparkApplication.start_time.desc())
            query = query.offset(offset).limit(limit)
            jobs = query.all()

            # Convert to list of dicts
            jobs_list = []
            for job in jobs:
                jobs_list.append(
                    {
                        "app_id": job.app_id,
                        "app_name": job.app_name,
                        "user": job.user,
                        "start_time": (
                            job.start_time.isoformat() if job.start_time else None
                        ),
                        "end_time": job.end_time.isoformat() if job.end_time else None,
                        "duration_ms": job.duration_ms,
                        "status": job.status,
                        "configuration": {
                            "num_executors": job.num_executors,
                            "executor_cores": job.executor_cores,
                            "executor_memory_mb": job.executor_memory_mb,
                            "driver_memory_mb": job.driver_memory_mb,
                        },
                        "metrics": {
                            "total_tasks": job.total_tasks,
                            "failed_tasks": job.failed_tasks,
                            "input_bytes": job.input_bytes,
                            "output_bytes": job.output_bytes,
                        },
                    }
                )

            return (
                jsonify(
                    {
                        "jobs": jobs_list,
                        "total": total,
                        "limit": limit,
                        "offset": offset,
                    }
                ),
                200,
            )

    except Exception as e:
        logger.error(f"Error listing jobs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/jobs/<app_id>", methods=["GET"])
def get_job_details(app_id: str):
    """Get detailed information about a specific job.

    Args:
        app_id: Spark application ID

    Returns:
        JSON response with job details
    """
    try:
        db = get_db()

        with db.get_session() as session:
            job = (
                session.query(SparkApplication)
                .filter(SparkApplication.app_id == app_id)
                .first()
            )

            if not job:
                return jsonify({"error": "Job not found"}), 404

            job_dict = {
                "app_id": job.app_id,
                "app_name": job.app_name,
                "user": job.user,
                "status": job.status,
                "spark_version": job.spark_version,
                "submit_time": job.submit_time.isoformat() if job.submit_time else None,
                "start_time": job.start_time.isoformat() if job.start_time else None,
                "end_time": job.end_time.isoformat() if job.end_time else None,
                "duration_ms": job.duration_ms,
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
                    "failed_stages": job.failed_stages,
                    "input_bytes": job.input_bytes,
                    "output_bytes": job.output_bytes,
                    "shuffle_read_bytes": job.shuffle_read_bytes,
                    "shuffle_write_bytes": job.shuffle_write_bytes,
                    "memory_spilled_bytes": job.memory_spilled_bytes,
                    "disk_spilled_bytes": job.disk_spilled_bytes,
                    "executor_run_time_ms": job.executor_run_time_ms,
                    "executor_cpu_time_ms": job.executor_cpu_time_ms,
                    "jvm_gc_time_ms": job.jvm_gc_time_ms,
                    "peak_memory_usage": job.peak_memory_usage,
                },
                "environment": {
                    "cluster_type": job.cluster_type,
                    "instance_type": job.instance_type,
                },
                "cost": {
                    "estimated_cost": job.estimated_cost,
                },
                "tags": job.tags,
                "spark_configs": job.spark_configs,
            }

            return jsonify(job_dict), 200

    except Exception as e:
        logger.error(f"Error getting job details: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/analyze/<app_id>", methods=["GET"])
def analyze_job(app_id: str):
    """Analyze a specific job and return insights.

    Args:
        app_id: Spark application ID

    Returns:
        JSON response with analysis
    """
    try:
        db = get_db()
        analyzer = get_analyzer()

        with db.get_session() as session:
            job = (
                session.query(SparkApplication)
                .filter(SparkApplication.app_id == app_id)
                .first()
            )

            if not job:
                return jsonify({"error": "Job not found"}), 404

            # Build job data dictionary for analyzer
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
                "jvm_gc_time_ms": job.jvm_gc_time_ms or 0,
                "executor_run_time_ms": job.executor_run_time_ms or 0,
                "executor_cpu_time_ms": job.executor_cpu_time_ms or 0,
                "total_tasks": job.total_tasks or 0,
                "failed_tasks": job.failed_tasks or 0,
                "total_stages": job.total_stages or 0,
                "failed_stages": job.failed_stages or 0,
                "peak_memory_usage": job.peak_memory_usage or 0,
            }

            # Run analysis
            analysis = analyzer.analyze_job(job_data)

            # Generate optimization suggestions based on issues
            suggestions = []
            for issue in analysis.get("issues", []):
                suggestions.append(
                    {
                        "issue": issue.get("type"),
                        "severity": issue.get("severity"),
                        "description": issue.get("description"),
                        "recommendation": issue.get("recommendation"),
                    }
                )

            return (
                jsonify(
                    {
                        "app_id": app_id,
                        "analysis": {
                            "resource_efficiency": analysis.get(
                                "resource_efficiency", {}
                            ),
                            "bottlenecks": analysis.get("bottlenecks", []),
                            "issues": analysis.get("issues", []),
                            "health_score": analysis.get("health_score", 0),
                        },
                        "suggestions": suggestions,
                    }
                ),
                200,
            )

    except Exception as e:
        logger.error(f"Error analyzing job: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/feedback", methods=["POST"])
def submit_feedback():
    """Submit feedback on a recommendation.

    Request body:
        {
            "recommendation_id": int,
            "actual_performance": dict (optional),
            "satisfaction_score": float (0.0 to 1.0)
        }

    Returns:
        JSON response confirming feedback submission
    """
    try:
        data = request.get_json()

        if not data or "recommendation_id" not in data:
            return (
                jsonify({"error": "Missing required parameter: recommendation_id"}),
                400,
            )

        recommendation_id = data["recommendation_id"]
        satisfaction_score = data.get("satisfaction_score")

        if satisfaction_score is not None:
            if not (0.0 <= satisfaction_score <= 1.0):
                return (
                    jsonify(
                        {"error": "satisfaction_score must be between 0.0 and 1.0"}
                    ),
                    400,
                )

        db = get_db()

        from spark_optimizer.storage.repository import JobRecommendationRepository

        with db.get_session() as session:
            repo = JobRecommendationRepository(session)

            if satisfaction_score is not None:
                repo.add_feedback(recommendation_id, satisfaction_score)

            # Record that feedback was provided
            repo.record_usage(recommendation_id)

            session.commit()

        return (
            jsonify(
                {
                    "status": "success",
                    "message": "Feedback recorded",
                    "recommendation_id": recommendation_id,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error submitting feedback: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@api_bp.route("/stats", methods=["GET"])
def get_stats():
    """Get statistics about stored jobs.

    Returns:
        JSON response with statistics
    """
    try:
        db = get_db()

        from sqlalchemy import func

        with db.get_session() as session:
            total_jobs = session.query(func.count(SparkApplication.id)).scalar() or 0

            if total_jobs == 0:
                return (
                    jsonify(
                        {
                            "total_jobs": 0,
                            "avg_duration_ms": 0,
                            "total_input_bytes": 0,
                            "total_output_bytes": 0,
                            "job_name_distribution": {},
                        }
                    ),
                    200,
                )

            avg_duration = (
                session.query(func.avg(SparkApplication.duration_ms)).scalar() or 0
            )
            total_input = (
                session.query(func.sum(SparkApplication.input_bytes)).scalar() or 0
            )
            total_output = (
                session.query(func.sum(SparkApplication.output_bytes)).scalar() or 0
            )

            # Job name distribution
            job_names = (
                session.query(
                    SparkApplication.app_name, func.count(SparkApplication.id)
                )
                .group_by(SparkApplication.app_name)
                .all()
            )

            job_name_dist = {name: count for name, count in job_names if name}

            return (
                jsonify(
                    {
                        "total_jobs": total_jobs,
                        "avg_duration_ms": int(avg_duration),
                        "total_input_bytes": total_input,
                        "total_output_bytes": total_output,
                        "job_name_distribution": job_name_dist,
                    }
                ),
                200,
            )

    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
