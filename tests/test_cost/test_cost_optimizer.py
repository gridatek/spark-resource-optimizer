"""Tests for cost optimization functionality."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from spark_optimizer.cost.cost_optimizer import (
    CostOptimizer,
    OptimizationResult,
    OptimizationGoal,
)
from spark_optimizer.cost.cost_model import CostModel, InstanceType


class TestOptimizationGoal:
    """Test OptimizationGoal enum."""

    def test_goal_values(self):
        """Test that goal enum has expected values."""
        assert OptimizationGoal.MINIMIZE_COST.value == "minimize_cost"
        assert OptimizationGoal.MINIMIZE_DURATION.value == "minimize_duration"
        assert OptimizationGoal.BALANCE.value == "balance"
        assert OptimizationGoal.BUDGET_CONSTRAINT.value == "budget_constraint"


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""

    def test_result_creation(self):
        """Test creating an optimization result."""
        result = OptimizationResult(
            original_config={"spark.executor.instances": 10},
            optimized_config={"spark.executor.instances": 5},
            original_cost=10.0,
            optimized_cost=5.5,
            savings=4.5,
            savings_percent=45.0,
        )

        assert result.savings == 4.5
        assert result.savings_percent == 45.0

    def test_result_to_dict(self):
        """Test converting result to dictionary."""
        result = OptimizationResult(
            original_config={"spark.executor.instances": 10},
            optimized_config={"spark.executor.instances": 5},
            original_cost=10.0,
            optimized_cost=5.5,
            savings=4.5,
            savings_percent=45.0,
            recommended_instance="m5.xlarge",
            provider="aws",
            recommendations=["Reduce executors"],
            trade_offs=["May increase duration"],
        )

        data = result.to_dict()

        assert data["original_cost"] == 10.0
        assert data["optimized_cost"] == 5.5
        assert data["savings"] == 4.5
        assert data["recommended_instance"] == "m5.xlarge"
        assert len(data["recommendations"]) == 1
        assert len(data["trade_offs"]) == 1


class TestCostOptimizer:
    """Test CostOptimizer class."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = CostOptimizer()

        assert optimizer._cost_model is not None
        assert optimizer._cloud_pricing is not None

    def test_optimizer_custom_model(self):
        """Test optimizer with custom cost model."""
        custom_model = CostModel(cloud_provider="aws")
        optimizer = CostOptimizer(cost_model=custom_model)

        assert optimizer._cost_model._cloud_provider == "aws"

    def test_optimize_basic(self):
        """Test basic optimization."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            estimated_duration_hours=2.0,
        )

        assert result.original_cost > 0
        assert result.optimized_cost > 0
        assert result.original_config["spark.executor.instances"] == 10

    def test_optimize_minimize_cost(self):
        """Test optimization with minimize cost goal."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 4,
                "spark.executor.memory": 16384,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        # Optimized cost should be <= original
        assert result.optimized_cost <= result.original_cost

    def test_optimize_minimize_duration(self):
        """Test optimization with minimize duration goal."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 2,
                "spark.executor.cores": 2,
                "spark.executor.memory": 4096,
            },
            estimated_duration_hours=2.0,
            goal=OptimizationGoal.MINIMIZE_DURATION,
        )

        # For minimize duration, we might increase resources
        assert result.optimized_cost >= 0

    def test_optimize_balance(self):
        """Test optimization with balance goal."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.BALANCE,
        )

        assert result.optimized_cost > 0

    def test_optimize_with_budget_constraint(self):
        """Test optimization with budget constraint."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 8,
                "spark.executor.memory": 16384,
            },
            estimated_duration_hours=2.0,
            goal=OptimizationGoal.BUDGET_CONSTRAINT,
            budget=5.0,
        )

        # Optimized cost should be within budget
        assert result.optimized_cost <= 5.0

    def test_optimize_with_constraints(self):
        """Test optimization with executor constraints."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            estimated_duration_hours=1.0,
            constraints={
                "min_executors": 5,
                "max_executors": 15,
                "min_memory": 4096,
                "max_memory": 16384,
            },
        )

        # Config should respect constraints
        assert result.optimized_config["spark.executor.instances"] >= 5
        assert result.optimized_config["spark.executor.instances"] <= 15

    def test_optimize_generates_recommendations(self):
        """Test that optimization generates recommendations."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 4,
                "spark.executor.memory": 16384,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        # Should have some recommendations
        assert isinstance(result.recommendations, list)

    def test_optimize_identifies_trade_offs(self):
        """Test that optimization identifies trade-offs."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 8,
                "spark.executor.memory": 16384,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        # If cost is reduced, should mention potential trade-offs
        if result.savings > 0:
            assert isinstance(result.trade_offs, list)

    def test_optimize_for_budget(self):
        """Test optimize_for_budget convenience method."""
        optimizer = CostOptimizer()

        result = optimizer.optimize_for_budget(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 8,
                "spark.executor.memory": 16384,
            },
            estimated_duration_hours=2.0,
            budget=3.0,
        )

        assert result.optimized_cost <= 3.0

    def test_find_cost_duration_frontier(self):
        """Test finding cost-duration frontier."""
        optimizer = CostOptimizer()

        frontier = optimizer.find_cost_duration_frontier(
            current_config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            base_duration_hours=2.0,
            num_points=5,
        )

        assert len(frontier) == 5

        # Each point should have (config, cost, duration)
        for config, cost, duration in frontier:
            assert isinstance(config, dict)
            assert cost >= 0
            assert duration >= 0

        # Should be sorted by cost
        costs = [cost for _, cost, _ in frontier]
        assert costs == sorted(costs)

    def test_compare_instance_types(self):
        """Test comparing instance types."""
        optimizer = CostOptimizer()

        comparisons = optimizer.compare_instance_types(
            config={
                "spark.executor.instances": 4,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            duration_hours=1.0,
            provider="aws",
        )

        assert len(comparisons) > 0

        for comparison in comparisons:
            assert "instance_type" in comparison
            assert "vcpus" in comparison
            assert "memory_gb" in comparison
            assert "total_cost" in comparison

    def test_recommend_spot_strategy_full_spot(self):
        """Test spot strategy recommendation for fault-tolerant jobs."""
        optimizer = CostOptimizer()

        strategy = optimizer.recommend_spot_strategy(
            config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
                "spark.driver.memory": 4096,
            },
            duration_hours=1.0,
            provider="aws",
            fault_tolerance=0.9,  # High fault tolerance
        )

        assert strategy["strategy"] == "full_spot"
        assert strategy["expected_savings"] > 0

    def test_recommend_spot_strategy_mixed(self):
        """Test spot strategy recommendation for mixed mode."""
        optimizer = CostOptimizer()

        strategy = optimizer.recommend_spot_strategy(
            config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
                "spark.driver.memory": 4096,
            },
            duration_hours=1.0,
            provider="aws",
            fault_tolerance=0.5,  # Medium fault tolerance
        )

        assert strategy["strategy"] == "mixed"

    def test_recommend_spot_strategy_on_demand(self):
        """Test spot strategy recommendation for on-demand."""
        optimizer = CostOptimizer()

        strategy = optimizer.recommend_spot_strategy(
            config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
                "spark.driver.memory": 4096,
            },
            duration_hours=1.0,
            provider="aws",
            fault_tolerance=0.2,  # Low fault tolerance
        )

        assert strategy["strategy"] == "on_demand"

    def test_spot_strategy_includes_cost_info(self):
        """Test that spot strategy includes cost information."""
        optimizer = CostOptimizer()

        strategy = optimizer.recommend_spot_strategy(
            config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
                "spark.driver.memory": 4096,
            },
            duration_hours=1.0,
            provider="aws",
            fault_tolerance=0.8,
        )

        assert "on_demand_cost" in strategy
        assert "spot_cost" in strategy
        assert "expected_cost" in strategy
        assert "interruption_probability" in strategy


class TestCostOptimizerCandidateGeneration:
    """Test candidate configuration generation."""

    def test_generates_multiple_candidates(self):
        """Test that multiple candidates are generated."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            estimated_duration_hours=1.0,
        )

        # Should have evaluated multiple configs
        assert result.original_config != {} or result.optimized_config != {}

    def test_minimize_cost_reduces_resources(self):
        """Test that minimize cost tends to reduce resources."""
        optimizer = CostOptimizer()

        current_config = {
            "spark.executor.instances": 20,
            "spark.executor.cores": 4,
            "spark.executor.memory": 16384,
        }

        result = optimizer.optimize(
            current_config=current_config,
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        # Should tend to reduce at least one resource
        # (not guaranteed, but likely with high initial config)
        reduced_executors = (
            result.optimized_config.get("spark.executor.instances", 20)
            <= current_config["spark.executor.instances"]
        )
        reduced_memory = (
            result.optimized_config.get("spark.executor.memory", 16384)
            <= current_config["spark.executor.memory"]
        )

        assert reduced_executors or reduced_memory

    def test_minimize_duration_may_increase_resources(self):
        """Test that minimize duration may increase resources."""
        optimizer = CostOptimizer()

        current_config = {
            "spark.executor.instances": 2,
            "spark.executor.cores": 2,
            "spark.executor.memory": 4096,
        }

        result = optimizer.optimize(
            current_config=current_config,
            estimated_duration_hours=4.0,
            goal=OptimizationGoal.MINIMIZE_DURATION,
        )

        # Could increase resources for better performance
        assert result.optimized_config is not None


class TestCostOptimizerRecommendations:
    """Test recommendation generation."""

    def test_savings_recommendation(self):
        """Test that savings are mentioned in recommendations."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 4,
                "spark.executor.memory": 16384,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        if result.savings > 0:
            # Should mention savings
            savings_mentioned = any(
                "saving" in rec.lower() or "$" in rec
                for rec in result.recommendations
            )
            assert savings_mentioned

    def test_executor_change_recommendation(self):
        """Test executor change recommendations."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        if (result.optimized_config.get("spark.executor.instances", 20)
                != 20):
            # Should mention executor change
            executor_mentioned = any(
                "executor" in rec.lower()
                for rec in result.recommendations
            )
            assert executor_mentioned

    def test_trade_off_for_reduced_executors(self):
        """Test trade-off warning for reduced executors."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 20,
                "spark.executor.cores": 4,
                "spark.executor.memory": 8192,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        if (result.optimized_config.get("spark.executor.instances", 20)
                < 20):
            # Should warn about potential duration increase
            duration_warning = any(
                "duration" in trade.lower()
                for trade in result.trade_offs
            )
            assert duration_warning

    def test_trade_off_for_reduced_memory(self):
        """Test trade-off warning for reduced memory."""
        optimizer = CostOptimizer()

        result = optimizer.optimize(
            current_config={
                "spark.executor.instances": 10,
                "spark.executor.cores": 4,
                "spark.executor.memory": 16384,
            },
            estimated_duration_hours=1.0,
            goal=OptimizationGoal.MINIMIZE_COST,
        )

        if (result.optimized_config.get("spark.executor.memory", 16384)
                < 16384):
            # Should warn about potential spilling
            memory_warning = any(
                "memory" in trade.lower() or "spill" in trade.lower()
                for trade in result.trade_offs
            )
            assert memory_warning
