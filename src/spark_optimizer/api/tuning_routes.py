"""API routes for auto-tuning capabilities."""

from flask import Blueprint, request, jsonify
from typing import Optional
import logging

from spark_optimizer.tuning import (
    AutoTuner,
    TuningStrategy,
    ConfigAdjuster,
    FeedbackLoop,
)

logger = logging.getLogger(__name__)

tuning_bp = Blueprint("tuning", __name__, url_prefix="/tuning")

# Global instances
_auto_tuner: Optional[AutoTuner] = None
_feedback_loop: Optional[FeedbackLoop] = None


def init_tuning() -> None:
    """Initialize tuning components."""
    global _auto_tuner, _feedback_loop

    _auto_tuner = AutoTuner()
    _feedback_loop = FeedbackLoop()


def get_auto_tuner() -> Optional[AutoTuner]:
    """Get the auto-tuner instance."""
    return _auto_tuner


def get_feedback_loop() -> Optional[FeedbackLoop]:
    """Get the feedback loop instance."""
    return _feedback_loop


@tuning_bp.route("/status", methods=["GET"])
def get_tuning_status():
    """Get tuning service status.

    Returns:
        JSON response with tuning service status
    """
    auto_tuner = get_auto_tuner()
    feedback_loop = get_feedback_loop()

    return (
        jsonify(
            {
                "tuning_available": auto_tuner is not None,
                "feedback_loop_available": feedback_loop is not None,
                "active_sessions": (
                    len(auto_tuner.list_sessions(status="active")) if auto_tuner else 0
                ),
                "tunable_parameters": (
                    len(auto_tuner.get_tunable_parameters()) if auto_tuner else 0
                ),
            }
        ),
        200,
    )


@tuning_bp.route("/sessions", methods=["GET"])
def list_tuning_sessions():
    """List tuning sessions.

    Query parameters:
        - app_id: Filter by application ID (optional)
        - status: Filter by status (optional)

    Returns:
        JSON response with list of sessions
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    app_id = request.args.get("app_id")
    status = request.args.get("status")

    sessions = auto_tuner.list_sessions(app_id=app_id, status=status)

    return (
        jsonify(
            {
                "sessions": [s.to_dict() for s in sessions],
                "count": len(sessions),
            }
        ),
        200,
    )


@tuning_bp.route("/sessions", methods=["POST"])
def start_tuning_session():
    """Start a new tuning session.

    Request body:
        {
            "app_id": str,
            "app_name": str (optional),
            "initial_config": dict,
            "strategy": str (optional, default: "moderate"),
            "target_metric": str (optional, default: "duration")
        }

    Returns:
        JSON response with created session
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    data = request.get_json()

    if not data or "app_id" not in data:
        return jsonify({"error": "Missing required field: app_id"}), 400

    if "initial_config" not in data:
        return jsonify({"error": "Missing required field: initial_config"}), 400

    app_id = data["app_id"]
    app_name = data.get("app_name", app_id)
    initial_config = data["initial_config"]

    # Parse strategy
    strategy_str = data.get("strategy", "moderate").lower()
    try:
        strategy = TuningStrategy(strategy_str)
    except ValueError:
        return (
            jsonify(
                {
                    "error": f"Invalid strategy: {strategy_str}. "
                    f"Valid options: conservative, moderate, aggressive"
                }
            ),
            400,
        )

    target_metric = data.get("target_metric", "duration")

    session = auto_tuner.start_session(
        app_id=app_id,
        app_name=app_name,
        initial_config=initial_config,
        strategy=strategy,
        target_metric=target_metric,
    )

    return (
        jsonify(
            {
                "status": "created",
                "session": session.to_dict(),
            }
        ),
        201,
    )


@tuning_bp.route("/sessions/<session_id>", methods=["GET"])
def get_tuning_session(session_id: str):
    """Get a tuning session.

    Args:
        session_id: Session ID

    Returns:
        JSON response with session details
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    session = auto_tuner.get_session(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    return jsonify(session.to_dict()), 200


@tuning_bp.route("/sessions/<session_id>/analyze", methods=["POST"])
def analyze_and_recommend(session_id: str):
    """Analyze metrics and get tuning recommendations.

    Request body:
        {
            "metrics": {"metric_name": value, ...}
        }

    Args:
        session_id: Session ID

    Returns:
        JSON response with recommended adjustments
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    session = auto_tuner.get_session(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    if session.status != "active":
        return (
            jsonify({"error": f"Session is not active (status: {session.status})"}),
            400,
        )

    data = request.get_json()

    if not data or "metrics" not in data:
        return jsonify({"error": "Missing required field: metrics"}), 400

    metrics = data["metrics"]

    adjustments = auto_tuner.analyze_and_recommend(session_id, metrics)

    return (
        jsonify(
            {
                "session_id": session_id,
                "iteration": session.iterations,
                "status": session.status,
                "adjustments": [a.to_dict() for a in adjustments],
                "best_config": session.best_config,
                "best_metric_value": session.best_metric_value,
            }
        ),
        200,
    )


@tuning_bp.route("/sessions/<session_id>/apply", methods=["POST"])
def apply_adjustment(session_id: str):
    """Mark an adjustment as applied.

    Request body:
        {
            "parameter": str,
            "old_value": any,
            "new_value": any,
            "reason": str (optional)
        }

    Args:
        session_id: Session ID

    Returns:
        JSON response confirming application
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    session = auto_tuner.get_session(session_id)

    if not session:
        return jsonify({"error": "Session not found"}), 404

    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body required"}), 400

    required = ["parameter", "old_value", "new_value"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    from spark_optimizer.tuning.auto_tuner import TuningAdjustment

    adjustment = TuningAdjustment(
        parameter=data["parameter"],
        old_value=data["old_value"],
        new_value=data["new_value"],
        reason=data.get("reason", ""),
    )

    if auto_tuner.apply_adjustment(session_id, adjustment):
        return (
            jsonify(
                {
                    "status": "applied",
                    "adjustment": adjustment.to_dict(),
                    "current_config": session.current_config,
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "Failed to apply adjustment"}), 500


@tuning_bp.route("/sessions/<session_id>/pause", methods=["POST"])
def pause_session(session_id: str):
    """Pause a tuning session.

    Args:
        session_id: Session ID

    Returns:
        JSON response confirming pause
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    if auto_tuner.pause_session(session_id):
        return (
            jsonify(
                {
                    "status": "paused",
                    "session_id": session_id,
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "Session not found or not active"}), 404


@tuning_bp.route("/sessions/<session_id>/resume", methods=["POST"])
def resume_session(session_id: str):
    """Resume a paused tuning session.

    Args:
        session_id: Session ID

    Returns:
        JSON response confirming resume
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    if auto_tuner.resume_session(session_id):
        return (
            jsonify(
                {
                    "status": "resumed",
                    "session_id": session_id,
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "Session not found or not paused"}), 404


@tuning_bp.route("/sessions/<session_id>/end", methods=["POST"])
def end_session(session_id: str):
    """End a tuning session.

    Request body:
        {
            "status": str (optional, default: "completed")
        }

    Args:
        session_id: Session ID

    Returns:
        JSON response with final session state
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    data = request.get_json() or {}
    status = data.get("status", "completed")

    if auto_tuner.end_session(session_id, status):
        session = auto_tuner.get_session(session_id)
        return (
            jsonify(
                {
                    "status": "ended",
                    "session": session.to_dict() if session else None,
                }
            ),
            200,
        )
    else:
        return jsonify({"error": "Session not found or already ended"}), 404


@tuning_bp.route("/sessions/<session_id>/best-config", methods=["GET"])
def get_best_config(session_id: str):
    """Get the best configuration found in a session.

    Args:
        session_id: Session ID

    Returns:
        JSON response with best configuration
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    config = auto_tuner.get_best_config(session_id)

    if config is None:
        return jsonify({"error": "Session not found"}), 404

    session = auto_tuner.get_session(session_id)

    return (
        jsonify(
            {
                "session_id": session_id,
                "best_config": config,
                "best_metric_value": session.best_metric_value if session else None,
                "iterations": session.iterations if session else 0,
            }
        ),
        200,
    )


@tuning_bp.route("/parameters", methods=["GET"])
def list_tunable_parameters():
    """List all tunable parameters.

    Returns:
        JSON response with tunable parameters
    """
    auto_tuner = get_auto_tuner()

    if not auto_tuner:
        return jsonify({"error": "Auto-tuner not initialized"}), 503

    params = auto_tuner.get_tunable_parameters()

    return (
        jsonify(
            {
                "parameters": {
                    name: config.to_dict() for name, config in params.items()
                },
                "count": len(params),
            }
        ),
        200,
    )


@tuning_bp.route("/validate", methods=["POST"])
def validate_config():
    """Validate a Spark configuration.

    Request body:
        {
            "config": {"param": value, ...}
        }

    Returns:
        JSON response with validation results
    """
    data = request.get_json()

    if not data or "config" not in data:
        return jsonify({"error": "Missing required field: config"}), 400

    config = data["config"]
    adjuster = ConfigAdjuster()

    results = {}
    all_valid = True

    for param, value in config.items():
        valid = adjuster.validate_change(param, value)
        results[param] = {
            "value": value,
            "valid": valid,
        }
        if not valid:
            all_valid = False

    return (
        jsonify(
            {
                "valid": all_valid,
                "results": results,
            }
        ),
        200,
    )


@tuning_bp.route("/feedback", methods=["POST"])
def submit_tuning_feedback():
    """Submit feedback on a tuning recommendation.

    Request body:
        {
            "session_id": str,
            "app_id": str,
            "config_applied": dict,
            "metric_name": str,
            "metric_before": float,
            "metric_after": float,
            "expected_improvement": float (optional),
            "notes": str (optional)
        }

    Returns:
        JSON response confirming feedback recorded
    """
    feedback_loop = get_feedback_loop()

    if not feedback_loop:
        return jsonify({"error": "Feedback loop not initialized"}), 503

    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body required"}), 400

    required = [
        "session_id",
        "app_id",
        "config_applied",
        "metric_name",
        "metric_before",
        "metric_after",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    feedback = feedback_loop.record_feedback(
        session_id=data["session_id"],
        app_id=data["app_id"],
        config_applied=data["config_applied"],
        metric_name=data["metric_name"],
        metric_before=data["metric_before"],
        metric_after=data["metric_after"],
        expected_improvement=data.get("expected_improvement", 0.0),
        notes=data.get("notes", ""),
    )

    return (
        jsonify(
            {
                "status": "recorded",
                "feedback": feedback.to_dict(),
            }
        ),
        201,
    )


@tuning_bp.route("/feedback", methods=["GET"])
def get_tuning_feedback():
    """Get tuning feedback records.

    Query parameters:
        - session_id: Filter by session ID (optional)
        - app_id: Filter by application ID (optional)
        - limit: Maximum records to return (default: 100)

    Returns:
        JSON response with feedback records
    """
    feedback_loop = get_feedback_loop()

    if not feedback_loop:
        return jsonify({"error": "Feedback loop not initialized"}), 503

    session_id = request.args.get("session_id")
    app_id = request.args.get("app_id")
    limit = request.args.get("limit", 100, type=int)

    feedback = feedback_loop.get_feedback(
        session_id=session_id,
        app_id=app_id,
        limit=limit,
    )

    return (
        jsonify(
            {
                "feedback": [f.to_dict() for f in feedback],
                "count": len(feedback),
            }
        ),
        200,
    )


@tuning_bp.route("/feedback/statistics", methods=["GET"])
def get_feedback_statistics():
    """Get feedback loop statistics.

    Returns:
        JSON response with statistics
    """
    feedback_loop = get_feedback_loop()

    if not feedback_loop:
        return jsonify({"error": "Feedback loop not initialized"}), 503

    stats = feedback_loop.get_statistics()

    return jsonify(stats), 200


@tuning_bp.route("/patterns", methods=["GET"])
def get_learned_patterns():
    """Get learned tuning patterns.

    Query parameters:
        - min_samples: Minimum sample size (default: 5)
        - min_confidence: Minimum confidence (optional)

    Returns:
        JSON response with learned patterns
    """
    feedback_loop = get_feedback_loop()

    if not feedback_loop:
        return jsonify({"error": "Feedback loop not initialized"}), 503

    min_samples = request.args.get("min_samples", 5, type=int)
    min_confidence = request.args.get("min_confidence", type=float)

    patterns = feedback_loop.get_patterns(
        min_samples=min_samples,
        min_confidence=min_confidence,
    )

    return (
        jsonify(
            {
                "patterns": [p.to_dict() for p in patterns],
                "count": len(patterns),
            }
        ),
        200,
    )


@tuning_bp.route("/recommend", methods=["POST"])
def get_learned_recommendation():
    """Get a recommendation based on learned patterns.

    Request body:
        {
            "metrics": {"metric_name": value, ...},
            "current_config": dict
        }

    Returns:
        JSON response with recommendation
    """
    feedback_loop = get_feedback_loop()

    if not feedback_loop:
        return jsonify({"error": "Feedback loop not initialized"}), 503

    data = request.get_json()

    if not data or "metrics" not in data or "current_config" not in data:
        return (
            jsonify({"error": "Missing required fields: metrics, current_config"}),
            400,
        )

    result = feedback_loop.get_recommendation(
        metrics=data["metrics"],
        current_config=data["current_config"],
    )

    if result:
        recommended_changes, confidence = result
        return (
            jsonify(
                {
                    "has_recommendation": True,
                    "recommended_changes": recommended_changes,
                    "confidence": confidence,
                }
            ),
            200,
        )
    else:
        return (
            jsonify(
                {
                    "has_recommendation": False,
                    "message": "No matching pattern found with sufficient confidence",
                }
            ),
            200,
        )
