"""Cost optimization example for Spark Resource Optimizer.

This example demonstrates the cost optimization functionality:
1. Estimating costs for Spark configurations
2. Comparing costs across cloud providers
3. Optimizing configurations for different goals
4. Analyzing spot instance strategies

To run this example:
    python examples/cost_optimization_example.py

Note: This example uses built-in pricing data. For production,
consider integrating with real-time cloud pricing APIs.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spark_optimizer.cost.cost_model import (
    CostModel,
    CostEstimate,
    InstanceType,
)
from spark_optimizer.cost.cost_optimizer import (
    CostOptimizer,
    OptimizationGoal,
    OptimizationResult,
)
from spark_optimizer.cost.cloud_pricing import (
    CloudPricing,
    PricingTier,
)


def format_cost(cost: float) -> str:
    """Format cost as currency."""
    return f"${cost:,.2f}"


def display_estimate(estimate: CostEstimate, title: str = "Cost Estimate"):
    """Display a cost estimate nicely."""
    print(f"\n  {title}:")
    print(f"    Total Cost: {format_cost(estimate.total_cost)}")
    print(f"    Duration: {estimate.duration_hours:.1f} hours")
    print(f"    Provider: {estimate.cloud_provider}")

    if estimate.breakdown:
        print("    Breakdown:")
        for item in estimate.breakdown:
            print(f"      - {item['resource_type']}: {format_cost(item['cost'])}")


def main():
    """Demonstrate cost optimization capabilities."""
    print("=" * 60)
    print("Spark Resource Optimizer - Cost Optimization Example")
    print("=" * 60)

    # 1. Define a Spark configuration to analyze
    print("\nStep 1: Setting up Spark configuration...")

    spark_config = {
        "spark.executor.instances": 20,
        "spark.executor.cores": 4,
        "spark.executor.memory": 16384,  # 16 GB in MB
        "spark.driver.memory": 8192,     # 8 GB in MB
    }

    estimated_duration = 2.0  # hours

    print(f"\n  Configuration:")
    print(f"    Executors: {spark_config['spark.executor.instances']}")
    print(f"    Cores per executor: {spark_config['spark.executor.cores']}")
    print(f"    Memory per executor: {spark_config['spark.executor.memory']} MB")
    print(f"    Driver memory: {spark_config['spark.driver.memory']} MB")
    print(f"    Estimated duration: {estimated_duration} hours")

    # 2. Initialize cost model and optimizer
    print("\nStep 2: Initializing cost optimizer...")

    cost_model = CostModel(cloud_provider="aws")
    optimizer = CostOptimizer(cost_model=cost_model)
    print("  CostOptimizer initialized for AWS")

    # 3. Get baseline cost estimate
    print("\nStep 3: Calculating baseline cost estimate...")

    baseline_estimate = cost_model.estimate(
        config=spark_config,
        duration_hours=estimated_duration,
    )

    display_estimate(baseline_estimate, "Baseline Cost Estimate")

    # 4. Optimize for minimum cost
    print("\nStep 4: Optimizing for minimum cost...")

    cost_result = optimizer.optimize(
        current_config=spark_config,
        estimated_duration_hours=estimated_duration,
        goal=OptimizationGoal.MINIMIZE_COST,
    )

    print(f"\n  Optimization Result (MINIMIZE_COST):")
    print(f"    Original cost: {format_cost(cost_result.original_cost)}")
    print(f"    Optimized cost: {format_cost(cost_result.optimized_cost)}")
    print(f"    Savings: {format_cost(cost_result.savings)} ({cost_result.savings_percent:.1f}%)")

    print(f"\n  Optimized Configuration:")
    print(f"    Executors: {cost_result.optimized_config.get('spark.executor.instances')}")
    print(f"    Cores: {cost_result.optimized_config.get('spark.executor.cores')}")
    print(f"    Memory: {cost_result.optimized_config.get('spark.executor.memory')} MB")

    if cost_result.recommendations:
        print(f"\n  Recommendations:")
        for rec in cost_result.recommendations[:3]:
            print(f"    - {rec}")

    if cost_result.trade_offs:
        print(f"\n  Trade-offs to consider:")
        for trade in cost_result.trade_offs[:3]:
            print(f"    - {trade}")

    # 5. Optimize for minimum duration
    print("\nStep 5: Optimizing for minimum duration...")

    duration_result = optimizer.optimize(
        current_config=spark_config,
        estimated_duration_hours=estimated_duration,
        goal=OptimizationGoal.MINIMIZE_DURATION,
    )

    print(f"\n  Optimization Result (MINIMIZE_DURATION):")
    print(f"    Original cost: {format_cost(duration_result.original_cost)}")
    print(f"    Optimized cost: {format_cost(duration_result.optimized_cost)}")

    print(f"\n  Optimized Configuration:")
    print(f"    Executors: {duration_result.optimized_config.get('spark.executor.instances')}")
    print(f"    Cores: {duration_result.optimized_config.get('spark.executor.cores')}")
    print(f"    Memory: {duration_result.optimized_config.get('spark.executor.memory')} MB")

    # 6. Budget-constrained optimization
    print("\nStep 6: Optimizing with budget constraint...")

    budget = 5.0  # $5 budget

    budget_result = optimizer.optimize(
        current_config=spark_config,
        estimated_duration_hours=estimated_duration,
        goal=OptimizationGoal.BUDGET_CONSTRAINT,
        budget=budget,
    )

    print(f"\n  Budget: {format_cost(budget)}")
    print(f"  Optimized cost: {format_cost(budget_result.optimized_cost)}")
    print(f"  Within budget: {'Yes' if budget_result.optimized_cost <= budget else 'No'}")

    print(f"\n  Configuration to fit budget:")
    print(f"    Executors: {budget_result.optimized_config.get('spark.executor.instances')}")
    print(f"    Cores: {budget_result.optimized_config.get('spark.executor.cores')}")
    print(f"    Memory: {budget_result.optimized_config.get('spark.executor.memory')} MB")

    # 7. Compare instance types
    print("\nStep 7: Comparing instance types...")

    comparisons = optimizer.compare_instance_types(
        config=spark_config,
        duration_hours=estimated_duration,
        provider="aws",
    )

    print(f"\n  Instance Type Comparison (AWS):")
    print(f"  {'Instance':<20} {'vCPUs':<8} {'Memory':<10} {'Hourly':<10} {'Total Cost':<12}")
    print("  " + "-" * 60)

    for comp in comparisons[:5]:  # Show top 5
        print(f"  {comp['instance_type']:<20} "
              f"{comp['vcpus']:<8} "
              f"{comp['memory_gb']:<10} "
              f"${comp['hourly_price']:<9.4f} "
              f"${comp['total_cost']:<11.2f}")

    # 8. Cloud provider comparison
    print("\nStep 8: Comparing cloud providers...")

    providers = ["aws", "gcp", "azure"]
    provider_costs = []

    for provider in providers:
        model = CostModel(cloud_provider=provider)
        estimate = model.estimate(
            config=spark_config,
            duration_hours=estimated_duration,
        )
        provider_costs.append({
            "provider": provider,
            "total_cost": estimate.total_cost,
        })

    print(f"\n  Cloud Provider Cost Comparison:")
    print(f"  {'Provider':<15} {'Total Cost':<15} {'Difference':<15}")
    print("  " + "-" * 45)

    min_cost = min(p["total_cost"] for p in provider_costs)
    for p in sorted(provider_costs, key=lambda x: x["total_cost"]):
        diff = p["total_cost"] - min_cost
        marker = " (cheapest)" if diff == 0 else ""
        print(f"  {p['provider'].upper():<15} "
              f"{format_cost(p['total_cost']):<15} "
              f"+{format_cost(diff)}{marker}")

    # 9. Spot instance strategy
    print("\nStep 9: Analyzing spot instance strategy...")

    spot_strategy = optimizer.recommend_spot_strategy(
        config=spark_config,
        duration_hours=estimated_duration,
        provider="aws",
        fault_tolerance=0.8,  # 80% fault tolerant
    )

    print(f"\n  Spot Instance Analysis:")
    print(f"    Recommended strategy: {spot_strategy['strategy']}")
    print(f"    On-demand cost: {format_cost(spot_strategy['on_demand_cost'])}")
    print(f"    Spot cost (expected): {format_cost(spot_strategy['spot_cost'])}")
    print(f"    Expected savings: {format_cost(spot_strategy['expected_savings'])}")
    print(f"    Interruption probability: {spot_strategy['interruption_probability']:.1%}")

    # 10. Cost-duration frontier
    print("\nStep 10: Finding cost-duration trade-off frontier...")

    frontier = optimizer.find_cost_duration_frontier(
        current_config=spark_config,
        base_duration_hours=estimated_duration,
        num_points=5,
    )

    print(f"\n  Cost-Duration Frontier:")
    print(f"  {'Option':<10} {'Cost':<12} {'Est. Duration':<15} {'Trade-off':<20}")
    print("  " + "-" * 57)

    for i, (config, cost, duration) in enumerate(frontier):
        trade_off = "Baseline" if i == 2 else ("Cheaper/Slower" if i < 2 else "Faster/Costlier")
        print(f"  {i + 1:<10} "
              f"{format_cost(cost):<12} "
              f"{duration:.1f} hours{'':<7} "
              f"{trade_off}")

    # 11. Monthly projection
    print("\nStep 11: Monthly cost projection...")

    # Assume job runs 4 times daily
    daily_runs = 4
    monthly_runs = daily_runs * 30
    monthly_on_demand = baseline_estimate.total_cost * monthly_runs
    monthly_spot = monthly_on_demand * 0.3  # ~70% savings with spot
    monthly_reserved = monthly_on_demand * 0.6  # ~40% savings with reserved

    print(f"\n  Monthly Projection (4 runs/day):")
    print(f"    On-demand: {format_cost(monthly_on_demand)}/month")
    print(f"    With spot instances: {format_cost(monthly_spot)}/month (save {format_cost(monthly_on_demand - monthly_spot)})")
    print(f"    With reserved: {format_cost(monthly_reserved)}/month (save {format_cost(monthly_on_demand - monthly_reserved)})")

    # 12. Production configuration
    print("\nStep 12: Production configuration example:")
    print("-" * 60)
    config_example = """
# Cost optimization for production:

from spark_optimizer.cost.cost_optimizer import CostOptimizer, OptimizationGoal
from spark_optimizer.cost.cost_model import CostModel

# Initialize with your cloud provider
optimizer = CostOptimizer(
    cost_model=CostModel(cloud_provider="aws"),
)

# Optimize with constraints
result = optimizer.optimize(
    current_config=your_spark_config,
    estimated_duration_hours=2.0,
    goal=OptimizationGoal.BALANCE,  # Balance cost and performance
    constraints={
        "min_executors": 5,
        "max_executors": 50,
        "min_memory": 4096,
        "max_memory": 32768,
    },
)

# Apply optimized config
optimized_config = result.optimized_config
print(f"Estimated savings: ${result.savings:.2f} ({result.savings_percent:.1f}%)")
"""
    print(config_example)
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Cost optimization example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
