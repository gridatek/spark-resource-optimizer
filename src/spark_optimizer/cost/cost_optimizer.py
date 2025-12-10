"""Cost optimization for Spark configurations."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .cost_model import CostModel, CostEstimate, InstanceType
from .cloud_pricing import CloudPricing, PricingTier, InstancePricing

logger = logging.getLogger(__name__)


class OptimizationGoal(Enum):
    """Cost optimization goals."""

    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_DURATION = "minimize_duration"
    BALANCE = "balance"  # Balance cost and performance
    BUDGET_CONSTRAINT = "budget_constraint"  # Stay under budget


@dataclass
class OptimizationResult:
    """Result of a cost optimization analysis."""

    original_config: Dict
    optimized_config: Dict
    original_cost: float
    optimized_cost: float
    savings: float
    savings_percent: float
    original_instance: Optional[str] = None
    recommended_instance: Optional[str] = None
    provider: str = "generic"
    region: str = "us-east-1"
    recommendations: List[str] = field(default_factory=list)
    trade_offs: List[str] = field(default_factory=list)
    confidence: float = 0.8
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "original_config": self.original_config,
            "optimized_config": self.optimized_config,
            "original_cost": round(self.original_cost, 4),
            "optimized_cost": round(self.optimized_cost, 4),
            "savings": round(self.savings, 4),
            "savings_percent": round(self.savings_percent, 2),
            "original_instance": self.original_instance,
            "recommended_instance": self.recommended_instance,
            "provider": self.provider,
            "region": self.region,
            "recommendations": self.recommendations,
            "trade_offs": self.trade_offs,
            "confidence": self.confidence,
            "calculated_at": self.calculated_at.isoformat(),
        }


class CostOptimizer:
    """Optimizes Spark configurations for cost.

    Provides:
    - Configuration optimization for cost
    - Instance type recommendations
    - Trade-off analysis
    - Budget-constrained optimization
    """

    def __init__(
        self,
        cost_model: Optional[CostModel] = None,
        cloud_pricing: Optional[CloudPricing] = None,
    ):
        """Initialize the cost optimizer.

        Args:
            cost_model: Cost model to use
            cloud_pricing: Cloud pricing data
        """
        self._cost_model = cost_model or CostModel()
        self._cloud_pricing = cloud_pricing or CloudPricing()

    def optimize(
        self,
        current_config: Dict,
        estimated_duration_hours: float,
        goal: OptimizationGoal = OptimizationGoal.BALANCE,
        budget: Optional[float] = None,
        provider: str = "generic",
        constraints: Optional[Dict] = None,
    ) -> OptimizationResult:
        """Optimize a Spark configuration for cost.

        Args:
            current_config: Current Spark configuration
            estimated_duration_hours: Estimated job duration
            goal: Optimization goal
            budget: Optional budget constraint
            provider: Cloud provider
            constraints: Optional constraints (min_executors, max_executors, etc.)

        Returns:
            OptimizationResult with recommended configuration
        """
        constraints = constraints or {}

        # Calculate current cost
        current_estimate = self._cost_model.estimate_from_config(
            current_config, estimated_duration_hours
        )
        current_cost = current_estimate.total_cost

        # Generate optimization candidates
        candidates = self._generate_candidates(current_config, goal, constraints)

        # Evaluate candidates
        best_config = current_config.copy()
        best_cost = current_cost
        best_estimate = current_estimate

        for candidate in candidates:
            estimate = self._cost_model.estimate_from_config(
                candidate, estimated_duration_hours
            )

            # Check budget constraint
            if budget and estimate.total_cost > budget:
                continue

            # Evaluate based on goal
            if self._is_better(estimate, best_estimate, goal):
                best_config = candidate
                best_cost = estimate.total_cost
                best_estimate = estimate

        # Calculate savings
        savings = current_cost - best_cost
        savings_percent = (savings / current_cost * 100) if current_cost > 0 else 0

        # Find best instance type
        recommended_instance = self._find_best_instance(best_config, provider)

        # Generate recommendations and trade-offs
        recommendations = self._generate_recommendations(
            current_config, best_config, goal, savings
        )
        trade_offs = self._identify_trade_offs(current_config, best_config)

        return OptimizationResult(
            original_config=current_config,
            optimized_config=best_config,
            original_cost=current_cost,
            optimized_cost=best_cost,
            savings=savings,
            savings_percent=savings_percent,
            recommended_instance=recommended_instance,
            provider=provider,
            recommendations=recommendations,
            trade_offs=trade_offs,
        )

    def optimize_for_budget(
        self,
        current_config: Dict,
        estimated_duration_hours: float,
        budget: float,
        provider: str = "generic",
    ) -> OptimizationResult:
        """Optimize configuration to stay within budget.

        Args:
            current_config: Current configuration
            estimated_duration_hours: Estimated duration
            budget: Maximum budget
            provider: Cloud provider

        Returns:
            OptimizationResult
        """
        return self.optimize(
            current_config=current_config,
            estimated_duration_hours=estimated_duration_hours,
            goal=OptimizationGoal.BUDGET_CONSTRAINT,
            budget=budget,
            provider=provider,
        )

    def find_cost_duration_frontier(
        self,
        current_config: Dict,
        base_duration_hours: float,
        num_points: int = 5,
    ) -> List[Tuple[Dict, float, float]]:
        """Find the cost-duration trade-off frontier.

        Args:
            current_config: Current configuration
            base_duration_hours: Base duration estimate
            num_points: Number of points on the frontier

        Returns:
            List of (config, cost, estimated_duration) tuples
        """
        frontier = []

        # Scale from 0.5x to 2x resources
        scales = [0.5 + i * 0.375 for i in range(num_points)]

        for scale in scales:
            config = current_config.copy()

            # Scale executors
            base_executors = config.get("spark.executor.instances", 2)
            config["spark.executor.instances"] = max(1, int(base_executors * scale))

            # Estimate duration (inversely proportional to resources, with diminishing returns)
            duration_factor = 1 / (0.5 + 0.5 * scale)  # Diminishing returns
            duration = base_duration_hours * duration_factor

            # Calculate cost
            estimate = self._cost_model.estimate_from_config(config, duration)

            frontier.append((config, estimate.total_cost, duration))

        return sorted(frontier, key=lambda x: x[1])  # Sort by cost

    def compare_instance_types(
        self,
        config: Dict,
        duration_hours: float,
        provider: str,
    ) -> List[Dict]:
        """Compare costs across instance types.

        Args:
            config: Spark configuration
            duration_hours: Estimated duration
            provider: Cloud provider

        Returns:
            List of instance comparisons
        """
        comparisons = []

        vcpus = config.get("spark.executor.cores", 4)
        memory_gb = config.get("spark.executor.memory", 4096) / 1024

        # Find matching instances
        instances = self._cloud_pricing.find_best_instance(
            min_vcpus=vcpus,
            min_memory_gb=memory_gb,
            provider=provider,
        )

        for instance in instances[:10]:  # Top 10
            num_executors = config.get("spark.executor.instances", 2)
            total_cost = instance.hourly_price * num_executors * duration_hours

            comparisons.append(
                {
                    "instance_type": instance.instance_type,
                    "vcpus": instance.vcpus,
                    "memory_gb": instance.memory_gb,
                    "hourly_price": instance.hourly_price,
                    "total_cost": round(total_cost, 4),
                    "tier": instance.tier.value,
                }
            )

        return comparisons

    def recommend_spot_strategy(
        self,
        config: Dict,
        duration_hours: float,
        provider: str,
        fault_tolerance: float = 0.8,
    ) -> Dict:
        """Recommend a spot instance strategy.

        Args:
            config: Spark configuration
            duration_hours: Estimated duration
            provider: Cloud provider
            fault_tolerance: Job's tolerance to interruption (0-1)

        Returns:
            Spot strategy recommendation
        """
        # Calculate on-demand cost
        on_demand_estimate = self._cost_model.estimate_from_config(
            config, duration_hours
        )

        # Estimate spot cost (typically 60-90% savings)
        spot_model = CostModel(cloud_provider=provider)
        spot_estimate = spot_model.estimate_job_cost(
            num_executors=config.get("spark.executor.instances", 2),
            executor_cores=config.get("spark.executor.cores", 4),
            executor_memory_mb=config.get("spark.executor.memory", 4096),
            driver_memory_mb=config.get("spark.driver.memory", 2048),
            duration_hours=duration_hours,
            instance_type=InstanceType.SPOT,
        )

        # Calculate expected cost including interruption risk
        interruption_probability = 0.05  # 5% per hour typical
        total_interruption_prob = 1 - (1 - interruption_probability) ** duration_hours

        # If interrupted, we restart and pay again
        expected_spot_cost = spot_estimate.total_cost * (
            1 + total_interruption_prob * 0.5
        )

        # Determine recommendation
        savings = on_demand_estimate.total_cost - expected_spot_cost

        if fault_tolerance >= 0.7 and savings > 0:
            strategy = "full_spot"
            message = "Use spot instances for all executors"
        elif fault_tolerance >= 0.4:
            strategy = "mixed"
            message = "Use spot for workers, on-demand for driver"
        else:
            strategy = "on_demand"
            message = "Use on-demand instances for reliability"

        return {
            "strategy": strategy,
            "message": message,
            "on_demand_cost": round(on_demand_estimate.total_cost, 4),
            "spot_cost": round(spot_estimate.total_cost, 4),
            "expected_cost": round(expected_spot_cost, 4),
            "expected_savings": round(savings, 4),
            "interruption_probability": round(total_interruption_prob * 100, 1),
            "recommendations": [
                (
                    "Enable checkpointing to handle interruptions"
                    if strategy != "on_demand"
                    else None
                ),
                (
                    "Set spark.dynamicAllocation.enabled=true for flexibility"
                    if strategy == "mixed"
                    else None
                ),
                (
                    "Consider using Spot Fleet for better availability"
                    if strategy == "full_spot"
                    else None
                ),
            ],
        }

    def _generate_candidates(
        self,
        current_config: Dict,
        goal: OptimizationGoal,
        constraints: Dict,
    ) -> List[Dict]:
        """Generate optimization candidates.

        Args:
            current_config: Current configuration
            goal: Optimization goal
            constraints: Constraints on configuration

        Returns:
            List of candidate configurations
        """
        candidates = []

        # Get current values
        executors = current_config.get("spark.executor.instances", 2)
        cores = current_config.get("spark.executor.cores", 4)
        memory = current_config.get("spark.executor.memory", 4096)

        # Define ranges based on goal
        if goal == OptimizationGoal.MINIMIZE_COST:
            executor_range = [max(1, executors - 2), executors, executors + 1]
            memory_range = [max(1024, memory - 2048), memory]
        elif goal == OptimizationGoal.MINIMIZE_DURATION:
            executor_range = [executors, executors + 2, executors + 4]
            memory_range = [memory, memory + 2048]
        else:  # BALANCE or BUDGET_CONSTRAINT
            executor_range = [max(1, executors - 1), executors, executors + 1]
            memory_range = [max(1024, memory - 1024), memory, memory + 1024]

        # Apply constraints
        min_executors = constraints.get("min_executors", 1)
        max_executors = constraints.get("max_executors", 100)
        min_memory = constraints.get("min_memory", 1024)
        max_memory = constraints.get("max_memory", 32768)

        # Generate combinations
        for e in executor_range:
            if e < min_executors or e > max_executors:
                continue

            for m in memory_range:
                if m < min_memory or m > max_memory:
                    continue

                config = current_config.copy()
                config["spark.executor.instances"] = e
                config["spark.executor.memory"] = m
                candidates.append(config)

        return candidates

    def _is_better(
        self,
        new_estimate: CostEstimate,
        current_estimate: CostEstimate,
        goal: OptimizationGoal,
    ) -> bool:
        """Check if new estimate is better than current.

        Args:
            new_estimate: New cost estimate
            current_estimate: Current cost estimate
            goal: Optimization goal

        Returns:
            True if new is better
        """
        if goal == OptimizationGoal.MINIMIZE_COST:
            return new_estimate.total_cost < current_estimate.total_cost

        elif goal == OptimizationGoal.MINIMIZE_DURATION:
            # For duration, we assume more resources = faster
            # This is a simplification; in reality it depends on the workload
            return new_estimate.cost_per_hour > current_estimate.cost_per_hour

        elif goal == OptimizationGoal.BALANCE:
            # Balance: prefer lower cost with acceptable performance
            # Use a weighted score
            cost_score = new_estimate.total_cost / max(
                current_estimate.total_cost, 0.01
            )
            perf_score = current_estimate.cost_per_hour / max(
                new_estimate.cost_per_hour, 0.01
            )
            return (cost_score * 0.6 + perf_score * 0.4) < 1.0

        else:  # BUDGET_CONSTRAINT
            return new_estimate.total_cost < current_estimate.total_cost

    def _find_best_instance(self, config: Dict, provider: str) -> Optional[str]:
        """Find the best instance type for a configuration.

        Args:
            config: Spark configuration
            provider: Cloud provider

        Returns:
            Recommended instance type or None
        """
        vcpus = config.get("spark.executor.cores", 4)
        memory_gb = config.get("spark.executor.memory", 4096) / 1024

        instances = self._cloud_pricing.find_best_instance(
            min_vcpus=vcpus,
            min_memory_gb=memory_gb,
            provider=provider,
        )

        return instances[0].instance_type if instances else None

    def _generate_recommendations(
        self,
        original: Dict,
        optimized: Dict,
        goal: OptimizationGoal,
        savings: float,
    ) -> List[str]:
        """Generate optimization recommendations.

        Args:
            original: Original configuration
            optimized: Optimized configuration
            goal: Optimization goal
            savings: Cost savings

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if savings > 0:
            recommendations.append(f"Potential savings: ${savings:.2f}")

        # Compare executor counts
        orig_exec = original.get("spark.executor.instances", 2)
        opt_exec = optimized.get("spark.executor.instances", 2)

        if opt_exec < orig_exec:
            recommendations.append(f"Reduce executors from {orig_exec} to {opt_exec}")
        elif opt_exec > orig_exec:
            recommendations.append(
                f"Increase executors from {orig_exec} to {opt_exec} for better performance"
            )

        # Compare memory
        orig_mem = original.get("spark.executor.memory", 4096)
        opt_mem = optimized.get("spark.executor.memory", 4096)

        if opt_mem < orig_mem:
            recommendations.append(
                f"Reduce executor memory from {orig_mem}MB to {opt_mem}MB"
            )
        elif opt_mem > orig_mem:
            recommendations.append(
                f"Increase executor memory from {orig_mem}MB to {opt_mem}MB"
            )

        # Goal-specific recommendations
        if goal == OptimizationGoal.MINIMIZE_COST:
            recommendations.append("Consider spot instances for additional savings")
        elif goal == OptimizationGoal.MINIMIZE_DURATION:
            recommendations.append("Enable dynamic allocation for peak efficiency")

        return recommendations

    def _identify_trade_offs(self, original: Dict, optimized: Dict) -> List[str]:
        """Identify trade-offs in the optimization.

        Args:
            original: Original configuration
            optimized: Optimized configuration

        Returns:
            List of trade-off descriptions
        """
        trade_offs = []

        orig_exec = original.get("spark.executor.instances", 2)
        opt_exec = optimized.get("spark.executor.instances", 2)

        if opt_exec < orig_exec:
            trade_offs.append(f"Fewer executors may increase job duration")

        orig_mem = original.get("spark.executor.memory", 4096)
        opt_mem = optimized.get("spark.executor.memory", 4096)

        if opt_mem < orig_mem:
            trade_offs.append(
                f"Less memory may cause increased spilling or GC pressure"
            )

        return trade_offs
