"""Tests for cost modeling functionality."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from spark_optimizer.cost.cost_model import (
    CostModel,
    CostEstimate,
    ResourceCost,
    InstanceType,
)


class TestInstanceType:
    """Test InstanceType enum."""

    def test_instance_type_values(self):
        """Test that instance type enum has expected values."""
        assert InstanceType.ON_DEMAND.value == "on_demand"
        assert InstanceType.SPOT.value == "spot"
        assert InstanceType.PREEMPTIBLE.value == "preemptible"
        assert InstanceType.RESERVED.value == "reserved"
        assert InstanceType.SAVINGS_PLAN.value == "savings_plan"


class TestResourceCost:
    """Test ResourceCost dataclass."""

    def test_resource_cost_creation(self):
        """Test creating a resource cost."""
        cost = ResourceCost(
            resource_type="compute",
            quantity=16,
            unit="vCPU-hours",
            unit_price=0.05,
            total_cost=0.80,
            duration_hours=1.0,
        )

        assert cost.resource_type == "compute"
        assert cost.quantity == 16
        assert cost.unit_price == 0.05
        assert cost.total_cost == 0.80

    def test_resource_cost_to_dict(self):
        """Test converting resource cost to dictionary."""
        cost = ResourceCost(
            resource_type="memory",
            quantity=64,
            unit="GB-hours",
            unit_price=0.005,
            total_cost=0.32,
            duration_hours=1.0,
        )

        result = cost.to_dict()

        assert result["resource_type"] == "memory"
        assert result["quantity"] == 64
        assert result["unit_price"] == 0.005
        assert result["total_cost"] == 0.32


class TestCostEstimate:
    """Test CostEstimate dataclass."""

    def test_estimate_creation(self):
        """Test creating a cost estimate."""
        estimate = CostEstimate(
            job_id="job-123",
            total_cost=5.50,
            breakdown=[],
            instance_type=InstanceType.ON_DEMAND,
            cloud_provider="aws",
            region="us-east-1",
            estimated_duration_hours=2.0,
        )

        assert estimate.job_id == "job-123"
        assert estimate.total_cost == 5.50
        assert estimate.cloud_provider == "aws"

    def test_estimate_to_dict(self):
        """Test converting estimate to dictionary."""
        breakdown = [
            ResourceCost("compute", 16, "vCPU-hours", 0.05, 0.80, 1.0),
            ResourceCost("memory", 64, "GB-hours", 0.005, 0.32, 1.0),
        ]

        estimate = CostEstimate(
            job_id="job-456",
            total_cost=1.12,
            breakdown=breakdown,
            instance_type=InstanceType.SPOT,
            cloud_provider="gcp",
            region="us-central1",
            estimated_duration_hours=1.5,
            spot_savings=0.5,
            recommendations=["Use spot instances"],
        )

        result = estimate.to_dict()

        assert result["job_id"] == "job-456"
        assert result["total_cost"] == 1.12
        assert result["instance_type"] == "spot"
        assert len(result["breakdown"]) == 2
        assert result["spot_savings"] == 0.5
        assert len(result["recommendations"]) == 1


class TestCostModel:
    """Test CostModel class."""

    def test_model_initialization(self):
        """Test cost model initialization."""
        model = CostModel()

        assert model._cloud_provider == "generic"
        assert model._region == "us-east-1"

    def test_model_custom_initialization(self):
        """Test cost model with custom settings."""
        model = CostModel(
            cloud_provider="aws",
            region="eu-west-1",
        )

        assert model._cloud_provider == "aws"
        assert model._region == "eu-west-1"

    def test_estimate_job_cost_basic(self):
        """Test basic job cost estimation."""
        model = CostModel()

        estimate = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
        )

        assert estimate.total_cost > 0
        assert estimate.estimated_duration_hours == 1.0
        assert len(estimate.breakdown) >= 2  # At least compute and memory

    def test_estimate_job_cost_with_data(self):
        """Test job cost estimation with data transfer."""
        model = CostModel()

        estimate = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            shuffle_bytes=10 * 1024**3,  # 10GB shuffle
            input_bytes=5 * 1024**3,
            output_bytes=2 * 1024**3,
        )

        # Should have storage and network costs
        resource_types = [r.resource_type for r in estimate.breakdown]
        assert "storage" in resource_types
        assert "network" in resource_types

    def test_estimate_job_cost_spot_instance(self):
        """Test cost estimation with spot instances."""
        model = CostModel()

        on_demand = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.ON_DEMAND,
        )

        spot = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.SPOT,
        )

        # Spot should be cheaper
        assert spot.total_cost < on_demand.total_cost

    def test_estimate_job_cost_reserved_instance(self):
        """Test cost estimation with reserved instances."""
        model = CostModel()

        on_demand = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.ON_DEMAND,
        )

        reserved = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.RESERVED,
        )

        # Reserved should be cheaper than on-demand
        assert reserved.total_cost < on_demand.total_cost

    def test_estimate_job_cost_spot_savings(self):
        """Test that spot savings are calculated."""
        model = CostModel()

        estimate = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.ON_DEMAND,
        )

        assert estimate.spot_savings > 0

    def test_estimate_from_config(self):
        """Test cost estimation from Spark config."""
        model = CostModel()

        config = {
            "spark.executor.instances": 4,
            "spark.executor.cores": 4,
            "spark.executor.memory": 8192,
            "spark.driver.memory": 4096,
        }

        estimate = model.estimate_from_config(
            config=config,
            estimated_duration_hours=2.0,
            data_size_gb=10.0,
        )

        assert estimate.total_cost > 0
        assert estimate.estimated_duration_hours == 2.0

    def test_estimate_from_config_defaults(self):
        """Test cost estimation with default config values."""
        model = CostModel()

        estimate = model.estimate_from_config(
            config={},
            estimated_duration_hours=1.0,
        )

        # Should use defaults
        assert estimate.total_cost > 0

    def test_estimate_from_historical(self):
        """Test cost estimation from historical job data."""
        model = CostModel()

        historical = {
            "num_executors": 4,
            "executor_cores": 4,
            "executor_memory_mb": 8192,
            "driver_memory_mb": 4096,
            "duration_ms": 3600000,  # 1 hour
            "shuffle_write_bytes": 1 * 1024**3,
            "input_bytes": 5 * 1024**3,
            "output_bytes": 2 * 1024**3,
        }

        estimate = model.estimate_from_historical(historical)

        assert estimate.total_cost > 0
        assert estimate.estimated_duration_hours == pytest.approx(1.0, rel=0.01)

    def test_estimate_from_historical_scaled(self):
        """Test scaled cost estimation from historical job."""
        model = CostModel()

        historical = {
            "num_executors": 4,
            "executor_cores": 4,
            "executor_memory_mb": 8192,
            "driver_memory_mb": 4096,
            "duration_ms": 3600000,
        }

        estimate_1x = model.estimate_from_historical(historical, scale_factor=1.0)
        estimate_2x = model.estimate_from_historical(historical, scale_factor=2.0)

        # 2x should have higher cost (longer duration)
        assert estimate_2x.total_cost > estimate_1x.total_cost

    def test_recommendations_spot_savings(self):
        """Test that spot savings recommendation is generated."""
        model = CostModel()

        estimate = model.estimate_job_cost(
            num_executors=10,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=2.0,
            instance_type=InstanceType.ON_DEMAND,
        )

        # Should recommend spot instances
        has_spot_rec = any("spot" in rec.lower() for rec in estimate.recommendations)
        assert has_spot_rec

    def test_recommendations_shuffle_optimization(self):
        """Test shuffle optimization recommendation."""
        model = CostModel()

        estimate = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            shuffle_bytes=50 * 1024**3,  # 50GB - high shuffle
        )

        # Should recommend shuffle optimization
        has_shuffle_rec = any(
            "shuffle" in rec.lower() for rec in estimate.recommendations
        )
        assert has_shuffle_rec

    def test_set_pricing(self):
        """Test updating pricing configuration."""
        model = CostModel()

        original_compute_price = model._pricing["compute"]["on_demand"]

        model.set_pricing(
            {
                "compute": {"on_demand": 0.10},
            }
        )

        assert model._pricing["compute"]["on_demand"] == 0.10

    def test_set_provider(self):
        """Test setting cloud provider."""
        model = CostModel()

        model.set_provider("aws", "us-west-2")

        assert model._cloud_provider == "aws"
        assert model._region == "us-west-2"


class TestCostModelCloudProviders:
    """Test cloud provider-specific behavior."""

    def test_aws_spot_discount(self):
        """Test AWS spot discount is applied."""
        model = CostModel(cloud_provider="aws")

        on_demand = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.ON_DEMAND,
        )

        spot = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.SPOT,
        )

        # AWS spot discount is 70%
        expected_discount = 0.3
        compute_ratio = spot.breakdown[0].total_cost / on_demand.breakdown[0].total_cost

        assert compute_ratio == pytest.approx(expected_discount, rel=0.1)

    def test_gcp_preemptible_discount(self):
        """Test GCP preemptible discount is applied."""
        model = CostModel(cloud_provider="gcp")

        on_demand = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.ON_DEMAND,
        )

        preemptible = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.PREEMPTIBLE,
        )

        # GCP preemptible discount is 80%
        assert preemptible.total_cost < on_demand.total_cost * 0.3

    def test_azure_spot_discount(self):
        """Test Azure spot discount is applied."""
        model = CostModel(cloud_provider="azure")

        on_demand = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.ON_DEMAND,
        )

        spot = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
            instance_type=InstanceType.SPOT,
        )

        # Azure spot discount is 65%
        assert spot.total_cost < on_demand.total_cost * 0.5


class TestCostModelCalculations:
    """Test internal calculation methods."""

    def test_compute_cost_scales_with_cores(self):
        """Test that compute cost scales with number of cores."""
        model = CostModel()

        estimate_4_cores = model.estimate_job_cost(
            num_executors=1,
            executor_cores=4,
            executor_memory_mb=4096,
            driver_memory_mb=2048,
            duration_hours=1.0,
        )

        estimate_8_cores = model.estimate_job_cost(
            num_executors=1,
            executor_cores=8,
            executor_memory_mb=4096,
            driver_memory_mb=2048,
            duration_hours=1.0,
        )

        # 8 cores should cost more than 4 cores
        compute_4 = next(
            r.total_cost
            for r in estimate_4_cores.breakdown
            if r.resource_type == "compute"
        )
        compute_8 = next(
            r.total_cost
            for r in estimate_8_cores.breakdown
            if r.resource_type == "compute"
        )

        assert compute_8 > compute_4

    def test_memory_cost_scales_with_memory(self):
        """Test that memory cost scales with memory allocation."""
        model = CostModel()

        estimate_4gb = model.estimate_job_cost(
            num_executors=1,
            executor_cores=4,
            executor_memory_mb=4096,
            driver_memory_mb=2048,
            duration_hours=1.0,
        )

        estimate_8gb = model.estimate_job_cost(
            num_executors=1,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=2048,
            duration_hours=1.0,
        )

        memory_4 = next(
            r.total_cost for r in estimate_4gb.breakdown if r.resource_type == "memory"
        )
        memory_8 = next(
            r.total_cost for r in estimate_8gb.breakdown if r.resource_type == "memory"
        )

        assert memory_8 > memory_4

    def test_cost_scales_with_duration(self):
        """Test that cost scales with job duration."""
        model = CostModel()

        estimate_1h = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
        )

        estimate_2h = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=2.0,
        )

        # 2 hours should cost about 2x
        assert estimate_2h.total_cost == pytest.approx(
            estimate_1h.total_cost * 2, rel=0.1
        )

    def test_cost_scales_with_executors(self):
        """Test that cost scales with number of executors."""
        model = CostModel()

        estimate_2_exec = model.estimate_job_cost(
            num_executors=2,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
        )

        estimate_4_exec = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=1.0,
        )

        # 4 executors should cost about 2x
        # (not exactly 2x due to driver costs being constant)
        assert estimate_4_exec.total_cost > estimate_2_exec.total_cost * 1.5

    def test_cost_per_hour(self):
        """Test cost per hour calculation."""
        model = CostModel()

        estimate = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=2.0,
        )

        expected_cost_per_hour = estimate.total_cost / 2.0
        assert estimate.cost_per_hour == pytest.approx(expected_cost_per_hour, rel=0.01)

    def test_zero_duration_handling(self):
        """Test handling of zero duration jobs."""
        model = CostModel()

        estimate = model.estimate_job_cost(
            num_executors=4,
            executor_cores=4,
            executor_memory_mb=8192,
            driver_memory_mb=4096,
            duration_hours=0.0,
        )

        assert estimate.total_cost == 0
        assert estimate.cost_per_hour == 0
