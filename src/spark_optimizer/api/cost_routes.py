"""API routes for advanced cost modeling."""

from flask import Blueprint, request, jsonify
from typing import Optional
import logging

from spark_optimizer.cost import (
    CostModel,
    CostOptimizer,
    CloudPricing,
    CostComparison,
)
from spark_optimizer.cost.cost_model import InstanceType
from spark_optimizer.cost.cost_optimizer import OptimizationGoal

logger = logging.getLogger(__name__)

cost_bp = Blueprint("cost", __name__, url_prefix="/cost")

# Global instances
_cost_model: Optional[CostModel] = None
_cost_optimizer: Optional[CostOptimizer] = None
_cloud_pricing: Optional[CloudPricing] = None
_cost_comparison: Optional[CostComparison] = None


def init_cost_modeling() -> None:
    """Initialize cost modeling components."""
    global _cost_model, _cost_optimizer, _cloud_pricing, _cost_comparison

    _cloud_pricing = CloudPricing()
    _cost_model = CostModel()
    _cost_optimizer = CostOptimizer(
        cost_model=_cost_model, cloud_pricing=_cloud_pricing
    )
    _cost_comparison = CostComparison(cloud_pricing=_cloud_pricing)


def get_cost_model() -> Optional[CostModel]:
    """Get the cost model instance."""
    return _cost_model


def get_cost_optimizer() -> Optional[CostOptimizer]:
    """Get the cost optimizer instance."""
    return _cost_optimizer


def get_cloud_pricing() -> Optional[CloudPricing]:
    """Get the cloud pricing instance."""
    return _cloud_pricing


def get_cost_comparison() -> Optional[CostComparison]:
    """Get the cost comparison instance."""
    return _cost_comparison


@cost_bp.route("/status", methods=["GET"])
def get_cost_status():
    """Get cost modeling service status.

    Returns:
        JSON response with cost service status
    """
    return (
        jsonify(
            {
                "cost_model_available": _cost_model is not None,
                "cost_optimizer_available": _cost_optimizer is not None,
                "cloud_pricing_available": _cloud_pricing is not None,
                "supported_providers": ["aws", "gcp", "azure"],
            }
        ),
        200,
    )


@cost_bp.route("/estimate", methods=["POST"])
def estimate_cost():
    """Estimate cost for a Spark job configuration.

    Request body:
        {
            "num_executors": int,
            "executor_cores": int,
            "executor_memory_mb": int,
            "driver_memory_mb": int (optional),
            "duration_hours": float,
            "instance_type": str (optional, default: "on_demand"),
            "provider": str (optional, default: "generic"),
            "shuffle_bytes": int (optional),
            "input_bytes": int (optional),
            "output_bytes": int (optional)
        }

    Returns:
        JSON response with cost estimate
    """
    cost_model = get_cost_model()

    if not cost_model:
        return jsonify({"error": "Cost model not initialized"}), 503

    data = request.get_json()

    if not data:
        return jsonify({"error": "Request body required"}), 400

    required = [
        "num_executors",
        "executor_cores",
        "executor_memory_mb",
        "duration_hours",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    # Parse instance type
    instance_type_str = data.get("instance_type", "on_demand").lower()
    try:
        instance_type = InstanceType(instance_type_str)
    except ValueError:
        return (
            jsonify(
                {
                    "error": f"Invalid instance_type: {instance_type_str}. "
                    f"Valid options: on_demand, spot, preemptible, reserved, savings_plan"
                }
            ),
            400,
        )

    # Set provider if specified
    provider = data.get("provider", "generic")
    cost_model.set_provider(provider)

    estimate = cost_model.estimate_job_cost(
        num_executors=data["num_executors"],
        executor_cores=data["executor_cores"],
        executor_memory_mb=data["executor_memory_mb"],
        driver_memory_mb=data.get("driver_memory_mb", 2048),
        duration_hours=data["duration_hours"],
        instance_type=instance_type,
        shuffle_bytes=data.get("shuffle_bytes", 0),
        input_bytes=data.get("input_bytes", 0),
        output_bytes=data.get("output_bytes", 0),
    )

    return jsonify(estimate.to_dict()), 200


@cost_bp.route("/estimate/config", methods=["POST"])
def estimate_from_config():
    """Estimate cost from a Spark configuration dictionary.

    Request body:
        {
            "config": dict (Spark configuration),
            "duration_hours": float,
            "data_size_gb": float (optional)
        }

    Returns:
        JSON response with cost estimate
    """
    cost_model = get_cost_model()

    if not cost_model:
        return jsonify({"error": "Cost model not initialized"}), 503

    data = request.get_json()

    if not data or "config" not in data or "duration_hours" not in data:
        return (
            jsonify({"error": "Missing required fields: config, duration_hours"}),
            400,
        )

    estimate = cost_model.estimate_from_config(
        config=data["config"],
        estimated_duration_hours=data["duration_hours"],
        data_size_gb=data.get("data_size_gb", 0),
    )

    return jsonify(estimate.to_dict()), 200


@cost_bp.route("/optimize", methods=["POST"])
def optimize_cost():
    """Optimize a Spark configuration for cost.

    Request body:
        {
            "config": dict (current Spark configuration),
            "duration_hours": float,
            "goal": str (optional, default: "balance"),
            "budget": float (optional),
            "provider": str (optional, default: "generic"),
            "constraints": dict (optional)
        }

    Returns:
        JSON response with optimization result
    """
    optimizer = get_cost_optimizer()

    if not optimizer:
        return jsonify({"error": "Cost optimizer not initialized"}), 503

    data = request.get_json()

    if not data or "config" not in data or "duration_hours" not in data:
        return (
            jsonify({"error": "Missing required fields: config, duration_hours"}),
            400,
        )

    # Parse goal
    goal_str = data.get("goal", "balance").lower()
    try:
        goal = OptimizationGoal(goal_str)
    except ValueError:
        return (
            jsonify(
                {
                    "error": f"Invalid goal: {goal_str}. "
                    f"Valid options: minimize_cost, minimize_duration, balance, budget_constraint"
                }
            ),
            400,
        )

    result = optimizer.optimize(
        current_config=data["config"],
        estimated_duration_hours=data["duration_hours"],
        goal=goal,
        budget=data.get("budget"),
        provider=data.get("provider", "generic"),
        constraints=data.get("constraints"),
    )

    return jsonify(result.to_dict()), 200


@cost_bp.route("/optimize/budget", methods=["POST"])
def optimize_for_budget():
    """Optimize configuration to stay within budget.

    Request body:
        {
            "config": dict,
            "duration_hours": float,
            "budget": float,
            "provider": str (optional)
        }

    Returns:
        JSON response with optimization result
    """
    optimizer = get_cost_optimizer()

    if not optimizer:
        return jsonify({"error": "Cost optimizer not initialized"}), 503

    data = request.get_json()

    required = ["config", "duration_hours", "budget"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    result = optimizer.optimize_for_budget(
        current_config=data["config"],
        estimated_duration_hours=data["duration_hours"],
        budget=data["budget"],
        provider=data.get("provider", "generic"),
    )

    return jsonify(result.to_dict()), 200


@cost_bp.route("/optimize/frontier", methods=["POST"])
def get_cost_duration_frontier():
    """Get the cost-duration trade-off frontier.

    Request body:
        {
            "config": dict,
            "base_duration_hours": float,
            "num_points": int (optional, default: 5)
        }

    Returns:
        JSON response with frontier points
    """
    optimizer = get_cost_optimizer()

    if not optimizer:
        return jsonify({"error": "Cost optimizer not initialized"}), 503

    data = request.get_json()

    if not data or "config" not in data or "base_duration_hours" not in data:
        return (
            jsonify({"error": "Missing required fields: config, base_duration_hours"}),
            400,
        )

    frontier = optimizer.find_cost_duration_frontier(
        current_config=data["config"],
        base_duration_hours=data["base_duration_hours"],
        num_points=data.get("num_points", 5),
    )

    return (
        jsonify(
            {
                "frontier": [
                    {
                        "config": config,
                        "cost": round(cost, 4),
                        "estimated_duration_hours": round(duration, 2),
                    }
                    for config, cost, duration in frontier
                ],
                "points": len(frontier),
            }
        ),
        200,
    )


@cost_bp.route("/compare/providers", methods=["POST"])
def compare_providers():
    """Compare costs across cloud providers.

    Request body:
        {
            "config": dict,
            "duration_hours": float,
            "include_regions": bool (optional, default: false)
        }

    Returns:
        JSON response with provider comparison
    """
    comparison = get_cost_comparison()

    if not comparison:
        return jsonify({"error": "Cost comparison not initialized"}), 503

    data = request.get_json()

    if not data or "config" not in data or "duration_hours" not in data:
        return (
            jsonify({"error": "Missing required fields: config, duration_hours"}),
            400,
        )

    result = comparison.compare_providers(
        config=data["config"],
        duration_hours=data["duration_hours"],
        include_regions=data.get("include_regions", False),
    )

    return jsonify(result.to_dict()), 200


@cost_bp.route("/compare/regions", methods=["POST"])
def compare_regions():
    """Compare costs across regions for a provider.

    Request body:
        {
            "config": dict,
            "duration_hours": float,
            "provider": str
        }

    Returns:
        JSON response with regional comparison
    """
    comparison = get_cost_comparison()

    if not comparison:
        return jsonify({"error": "Cost comparison not initialized"}), 503

    data = request.get_json()

    required = ["config", "duration_hours", "provider"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    costs = comparison.compare_regions(
        config=data["config"],
        duration_hours=data["duration_hours"],
        provider=data["provider"],
    )

    sorted_costs = sorted(costs.items(), key=lambda x: x[1])

    return (
        jsonify(
            {
                "provider": data["provider"],
                "regions": {r: c for r, c in sorted_costs},
                "cheapest_region": sorted_costs[0][0] if sorted_costs else None,
                "cheapest_cost": sorted_costs[0][1] if sorted_costs else None,
            }
        ),
        200,
    )


@cost_bp.route("/compare/cheapest", methods=["POST"])
def find_cheapest_option():
    """Find the cheapest provider/region combination.

    Request body:
        {
            "config": dict,
            "duration_hours": float,
            "include_spot": bool (optional, default: true)
        }

    Returns:
        JSON response with cheapest options
    """
    comparison = get_cost_comparison()

    if not comparison:
        return jsonify({"error": "Cost comparison not initialized"}), 503

    data = request.get_json()

    if not data or "config" not in data or "duration_hours" not in data:
        return (
            jsonify({"error": "Missing required fields: config, duration_hours"}),
            400,
        )

    result = comparison.find_cheapest_option(
        config=data["config"],
        duration_hours=data["duration_hours"],
        include_spot=data.get("include_spot", True),
    )

    return jsonify(result), 200


@cost_bp.route("/compare/migration", methods=["POST"])
def estimate_migration():
    """Estimate cost of migrating between providers.

    Request body:
        {
            "config": dict,
            "duration_hours": float,
            "from_provider": str,
            "to_provider": str,
            "data_size_gb": float
        }

    Returns:
        JSON response with migration analysis
    """
    comparison = get_cost_comparison()

    if not comparison:
        return jsonify({"error": "Cost comparison not initialized"}), 503

    data = request.get_json()

    required = [
        "config",
        "duration_hours",
        "from_provider",
        "to_provider",
        "data_size_gb",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    result = comparison.estimate_migration_cost(
        config=data["config"],
        duration_hours=data["duration_hours"],
        from_provider=data["from_provider"],
        to_provider=data["to_provider"],
        data_size_gb=data["data_size_gb"],
    )

    return jsonify(result), 200


@cost_bp.route("/instances", methods=["GET"])
def list_instances():
    """List available instance types.

    Query parameters:
        - provider: Filter by provider (optional)
        - min_vcpus: Minimum vCPUs (optional)
        - min_memory_gb: Minimum memory in GB (optional)

    Returns:
        JSON response with instance list
    """
    pricing = get_cloud_pricing()

    if not pricing:
        return jsonify({"error": "Cloud pricing not initialized"}), 503

    provider = request.args.get("provider")
    min_vcpus = request.args.get("min_vcpus", 0, type=int)
    min_memory_gb = request.args.get("min_memory_gb", 0, type=float)

    instances = pricing.list_instances(
        provider=provider,
        min_vcpus=min_vcpus,
        min_memory_gb=min_memory_gb,
    )

    return (
        jsonify(
            {
                "instances": [i.to_dict() for i in instances],
                "count": len(instances),
            }
        ),
        200,
    )


@cost_bp.route("/instances/<provider>/<instance_type>", methods=["GET"])
def get_instance_pricing(provider: str, instance_type: str):
    """Get pricing for a specific instance type.

    Args:
        provider: Cloud provider
        instance_type: Instance type name

    Query parameters:
        - tier: Pricing tier (optional, default: on_demand)
        - region: Region (optional)

    Returns:
        JSON response with instance pricing
    """
    pricing = get_cloud_pricing()

    if not pricing:
        return jsonify({"error": "Cloud pricing not initialized"}), 503

    tier_str = request.args.get("tier", "on_demand")
    region = request.args.get("region")

    from spark_optimizer.cost.cloud_pricing import PricingTier

    try:
        tier = PricingTier(tier_str)
    except ValueError:
        return jsonify({"error": f"Invalid tier: {tier_str}"}), 400

    instance = pricing.get_instance(instance_type, provider, tier, region)

    if not instance:
        return jsonify({"error": "Instance type not found"}), 404

    return jsonify(instance.to_dict()), 200


@cost_bp.route("/instances/best", methods=["POST"])
def find_best_instance():
    """Find the best instances matching requirements.

    Request body:
        {
            "min_vcpus": int,
            "min_memory_gb": float,
            "provider": str (optional),
            "tier": str (optional, default: "on_demand"),
            "prefer_local_storage": bool (optional, default: false)
        }

    Returns:
        JSON response with matching instances
    """
    pricing = get_cloud_pricing()

    if not pricing:
        return jsonify({"error": "Cloud pricing not initialized"}), 503

    data = request.get_json()

    if not data or "min_vcpus" not in data or "min_memory_gb" not in data:
        return (
            jsonify({"error": "Missing required fields: min_vcpus, min_memory_gb"}),
            400,
        )

    from spark_optimizer.cost.cloud_pricing import PricingTier

    tier_str = data.get("tier", "on_demand")
    try:
        tier = PricingTier(tier_str)
    except ValueError:
        return jsonify({"error": f"Invalid tier: {tier_str}"}), 400

    instances = pricing.find_best_instance(
        min_vcpus=data["min_vcpus"],
        min_memory_gb=data["min_memory_gb"],
        provider=data.get("provider"),
        tier=tier,
        prefer_local_storage=data.get("prefer_local_storage", False),
    )

    return (
        jsonify(
            {
                "instances": [i.to_dict() for i in instances[:10]],
                "count": len(instances),
            }
        ),
        200,
    )


@cost_bp.route("/spot/recommend", methods=["POST"])
def recommend_spot_strategy():
    """Get spot instance strategy recommendation.

    Request body:
        {
            "config": dict,
            "duration_hours": float,
            "provider": str,
            "fault_tolerance": float (optional, default: 0.8)
        }

    Returns:
        JSON response with spot strategy recommendation
    """
    optimizer = get_cost_optimizer()

    if not optimizer:
        return jsonify({"error": "Cost optimizer not initialized"}), 503

    data = request.get_json()

    required = ["config", "duration_hours", "provider"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    result = optimizer.recommend_spot_strategy(
        config=data["config"],
        duration_hours=data["duration_hours"],
        provider=data["provider"],
        fault_tolerance=data.get("fault_tolerance", 0.8),
    )

    # Filter out None values from recommendations
    result["recommendations"] = [r for r in result.get("recommendations", []) if r]

    return jsonify(result), 200


@cost_bp.route("/providers/<provider>", methods=["GET"])
def get_provider_summary(provider: str):
    """Get a summary of a cloud provider's offerings.

    Args:
        provider: Cloud provider name

    Returns:
        JSON response with provider summary
    """
    comparison = get_cost_comparison()

    if not comparison:
        return jsonify({"error": "Cost comparison not initialized"}), 503

    summary = comparison.get_provider_summary(provider)

    return jsonify(summary), 200


@cost_bp.route("/providers/recommend", methods=["POST"])
def recommend_provider():
    """Recommend a cloud provider based on requirements.

    Request body:
        {
            "requirements": {
                "executors": int,
                "vcpus": int,
                "memory_mb": int,
                "driver_memory_mb": int (optional),
                "duration_hours": float
            },
            "priorities": {
                "cost": float (optional),
                "performance": float (optional),
                "reliability": float (optional)
            } (optional)
        }

    Returns:
        JSON response with provider recommendation
    """
    comparison = get_cost_comparison()

    if not comparison:
        return jsonify({"error": "Cost comparison not initialized"}), 503

    data = request.get_json()

    if not data or "requirements" not in data:
        return jsonify({"error": "Missing required field: requirements"}), 400

    result = comparison.recommend_provider(
        requirements=data["requirements"],
        priorities=data.get("priorities"),
    )

    return jsonify(result), 200
