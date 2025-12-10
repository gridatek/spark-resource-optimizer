"""Tests for cloud pricing functionality."""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch

from spark_optimizer.cost.cloud_pricing import (
    CloudPricing,
    InstancePricing,
    PricingTier,
)


class TestPricingTier:
    """Test PricingTier enum."""

    def test_tier_values(self):
        """Test that tier enum has expected values."""
        assert PricingTier.ON_DEMAND.value == "on_demand"
        assert PricingTier.SPOT.value == "spot"
        assert PricingTier.PREEMPTIBLE.value == "preemptible"
        assert PricingTier.RESERVED_1Y.value == "reserved_1y"
        assert PricingTier.RESERVED_3Y.value == "reserved_3y"
        assert PricingTier.SAVINGS_PLAN.value == "savings_plan"
        assert PricingTier.COMMITTED_USE.value == "committed_use"


class TestInstancePricing:
    """Test InstancePricing dataclass."""

    def test_instance_pricing_creation(self):
        """Test creating instance pricing."""
        pricing = InstancePricing(
            instance_type="m5.xlarge",
            provider="aws",
            region="us-east-1",
            vcpus=4,
            memory_gb=16,
            hourly_price=0.192,
            tier=PricingTier.ON_DEMAND,
        )

        assert pricing.instance_type == "m5.xlarge"
        assert pricing.provider == "aws"
        assert pricing.vcpus == 4
        assert pricing.memory_gb == 16
        assert pricing.hourly_price == 0.192

    def test_instance_pricing_to_dict(self):
        """Test converting instance pricing to dictionary."""
        pricing = InstancePricing(
            instance_type="n1-standard-4",
            provider="gcp",
            region="us-central1",
            vcpus=4,
            memory_gb=15,
            hourly_price=0.190,
            tier=PricingTier.ON_DEMAND,
            gpu_count=1,
            gpu_type="nvidia-t4",
            local_storage_gb=375,
        )

        result = pricing.to_dict()

        assert result["instance_type"] == "n1-standard-4"
        assert result["provider"] == "gcp"
        assert result["vcpus"] == 4
        assert result["gpu_count"] == 1
        assert result["gpu_type"] == "nvidia-t4"
        assert result["local_storage_gb"] == 375

    def test_price_per_vcpu_hour(self):
        """Test price per vCPU hour calculation."""
        pricing = InstancePricing(
            instance_type="test",
            provider="test",
            region="test",
            vcpus=4,
            memory_gb=16,
            hourly_price=0.40,
            tier=PricingTier.ON_DEMAND,
        )

        assert pricing.price_per_vcpu_hour == 0.10

    def test_price_per_gb_hour(self):
        """Test price per GB memory hour calculation."""
        pricing = InstancePricing(
            instance_type="test",
            provider="test",
            region="test",
            vcpus=4,
            memory_gb=16,
            hourly_price=0.32,
            tier=PricingTier.ON_DEMAND,
        )

        assert pricing.price_per_gb_hour == 0.02

    def test_zero_vcpu_handling(self):
        """Test handling of zero vCPUs."""
        pricing = InstancePricing(
            instance_type="test",
            provider="test",
            region="test",
            vcpus=0,
            memory_gb=16,
            hourly_price=0.32,
            tier=PricingTier.ON_DEMAND,
        )

        assert pricing.price_per_vcpu_hour == 0


class TestCloudPricing:
    """Test CloudPricing class."""

    def test_pricing_initialization(self):
        """Test cloud pricing initialization."""
        pricing = CloudPricing()

        assert "aws" in pricing._instances
        assert "gcp" in pricing._instances
        assert "azure" in pricing._instances

    def test_get_instance_aws(self):
        """Test getting AWS instance pricing."""
        pricing = CloudPricing()

        instance = pricing.get_instance("m5.xlarge", "aws")

        assert instance is not None
        assert instance.instance_type == "m5.xlarge"
        assert instance.provider == "aws"
        assert instance.vcpus == 4
        assert instance.memory_gb == 16

    def test_get_instance_gcp(self):
        """Test getting GCP instance pricing."""
        pricing = CloudPricing()

        instance = pricing.get_instance("n1-standard-4", "gcp")

        assert instance is not None
        assert instance.instance_type == "n1-standard-4"
        assert instance.provider == "gcp"
        assert instance.vcpus == 4

    def test_get_instance_azure(self):
        """Test getting Azure instance pricing."""
        pricing = CloudPricing()

        instance = pricing.get_instance("Standard_D4s_v3", "azure")

        assert instance is not None
        assert instance.instance_type == "Standard_D4s_v3"
        assert instance.provider == "azure"

    def test_get_instance_not_found(self):
        """Test getting non-existent instance."""
        pricing = CloudPricing()

        instance = pricing.get_instance("nonexistent", "aws")

        assert instance is None

    def test_get_instance_unknown_provider(self):
        """Test getting instance from unknown provider."""
        pricing = CloudPricing()

        instance = pricing.get_instance("m5.xlarge", "unknown")

        assert instance is None

    def test_get_instance_spot_tier(self):
        """Test getting spot instance pricing."""
        pricing = CloudPricing()

        on_demand = pricing.get_instance("m5.xlarge", "aws", PricingTier.ON_DEMAND)
        spot = pricing.get_instance("m5.xlarge", "aws", PricingTier.SPOT)

        assert spot.hourly_price < on_demand.hourly_price
        assert spot.tier == PricingTier.SPOT

    def test_get_instance_reserved_tier(self):
        """Test getting reserved instance pricing."""
        pricing = CloudPricing()

        on_demand = pricing.get_instance("m5.xlarge", "aws", PricingTier.ON_DEMAND)
        reserved_1y = pricing.get_instance("m5.xlarge", "aws", PricingTier.RESERVED_1Y)
        reserved_3y = pricing.get_instance("m5.xlarge", "aws", PricingTier.RESERVED_3Y)

        assert reserved_1y.hourly_price < on_demand.hourly_price
        assert reserved_3y.hourly_price < reserved_1y.hourly_price

    def test_get_instance_regional_adjustment(self):
        """Test regional price adjustment."""
        pricing = CloudPricing()

        us_east = pricing.get_instance("m5.xlarge", "aws", region="us-east-1")
        eu_west = pricing.get_instance("m5.xlarge", "aws", region="eu-west-1")

        # EU should be more expensive
        assert eu_west.hourly_price > us_east.hourly_price

    def test_find_best_instance_basic(self):
        """Test finding best instances."""
        pricing = CloudPricing()

        instances = pricing.find_best_instance(
            min_vcpus=4,
            min_memory_gb=16,
        )

        assert len(instances) > 0

        for instance in instances:
            assert instance.vcpus >= 4
            assert instance.memory_gb >= 16

    def test_find_best_instance_by_provider(self):
        """Test finding best instances by provider."""
        pricing = CloudPricing()

        instances = pricing.find_best_instance(
            min_vcpus=4,
            min_memory_gb=8,
            provider="aws",
        )

        assert len(instances) > 0

        for instance in instances:
            assert instance.provider == "aws"

    def test_find_best_instance_sorted_by_price(self):
        """Test that results are sorted by price."""
        pricing = CloudPricing()

        instances = pricing.find_best_instance(
            min_vcpus=4,
            min_memory_gb=8,
        )

        prices = [i.hourly_price for i in instances]
        assert prices == sorted(prices)

    def test_find_best_instance_prefer_local_storage(self):
        """Test preferring instances with local storage."""
        pricing = CloudPricing()

        instances = pricing.find_best_instance(
            min_vcpus=4,
            min_memory_gb=16,
            provider="aws",
            prefer_local_storage=True,
        )

        # i3 instances should come before m5/r5 if prefer_local_storage
        if len(instances) > 1:
            has_storage = [i for i in instances if i.local_storage_gb > 0]
            no_storage = [i for i in instances if i.local_storage_gb == 0]

            if has_storage and no_storage:
                # Instances with storage should come first
                first_with_storage = instances.index(has_storage[0])
                first_without_storage = instances.index(no_storage[0])
                assert first_with_storage < first_without_storage

    def test_find_best_instance_spot_tier(self):
        """Test finding best instances with spot pricing."""
        pricing = CloudPricing()

        on_demand = pricing.find_best_instance(
            min_vcpus=4,
            min_memory_gb=8,
            tier=PricingTier.ON_DEMAND,
        )

        spot = pricing.find_best_instance(
            min_vcpus=4,
            min_memory_gb=8,
            tier=PricingTier.SPOT,
        )

        # Spot prices should be lower
        assert spot[0].hourly_price < on_demand[0].hourly_price

    def test_compare_providers(self):
        """Test comparing providers."""
        pricing = CloudPricing()

        comparison = pricing.compare_providers(
            vcpus=4,
            memory_gb=16,
        )

        assert "aws" in comparison
        assert "gcp" in comparison
        assert "azure" in comparison

        for provider, instance in comparison.items():
            if instance:
                assert instance.vcpus >= 4
                assert instance.memory_gb >= 16

    def test_add_custom_pricing(self):
        """Test adding custom instance pricing."""
        pricing = CloudPricing()

        custom = InstancePricing(
            instance_type="custom.xlarge",
            provider="custom_provider",
            region="custom-region",
            vcpus=8,
            memory_gb=32,
            hourly_price=0.50,
            tier=PricingTier.ON_DEMAND,
        )

        pricing.add_custom_pricing(custom)

        # Should be retrievable
        retrieved = pricing.get_instance("custom.xlarge", "custom_provider")
        assert retrieved is not None
        assert retrieved.vcpus == 8

    def test_list_instances_all(self):
        """Test listing all instances."""
        pricing = CloudPricing()

        instances = pricing.list_instances()

        assert len(instances) > 0

        # Should include instances from multiple providers
        providers = set(i.provider for i in instances)
        assert len(providers) >= 3

    def test_list_instances_by_provider(self):
        """Test listing instances by provider."""
        pricing = CloudPricing()

        instances = pricing.list_instances(provider="aws")

        for instance in instances:
            assert instance.provider == "aws"

    def test_list_instances_with_filters(self):
        """Test listing instances with filters."""
        pricing = CloudPricing()

        instances = pricing.list_instances(
            min_vcpus=8,
            min_memory_gb=32,
        )

        for instance in instances:
            assert instance.vcpus >= 8
            assert instance.memory_gb >= 32

    def test_get_regional_prices(self):
        """Test getting regional prices."""
        pricing = CloudPricing()

        regional_prices = pricing.get_regional_prices("m5.xlarge", "aws")

        assert len(regional_prices) > 0
        assert "us-east-1" in regional_prices
        assert "eu-west-1" in regional_prices

        # EU should be more expensive than US
        assert regional_prices["eu-west-1"] > regional_prices["us-east-1"]

    def test_get_regional_prices_not_found(self):
        """Test getting regional prices for non-existent instance."""
        pricing = CloudPricing()

        regional_prices = pricing.get_regional_prices("nonexistent", "aws")

        assert regional_prices == {}

    def test_estimate_monthly_cost(self):
        """Test estimating monthly cost."""
        pricing = CloudPricing()

        monthly_cost = pricing.estimate_monthly_cost(
            instance_type="m5.xlarge",
            provider="aws",
            count=5,
            hours_per_day=8,
            days_per_month=22,
        )

        # 5 instances * 8 hours * 22 days * ~$0.192/hour
        expected = 5 * 8 * 22 * 0.192
        assert monthly_cost == pytest.approx(expected, rel=0.01)

    def test_estimate_monthly_cost_full_time(self):
        """Test estimating monthly cost for full-time usage."""
        pricing = CloudPricing()

        monthly_cost = pricing.estimate_monthly_cost(
            instance_type="m5.xlarge",
            provider="aws",
            count=1,
            hours_per_day=24,
            days_per_month=30,
        )

        # 1 instance * 24 hours * 30 days * ~$0.192/hour
        expected = 1 * 24 * 30 * 0.192
        assert monthly_cost == pytest.approx(expected, rel=0.01)

    def test_estimate_monthly_cost_not_found(self):
        """Test estimating monthly cost for non-existent instance."""
        pricing = CloudPricing()

        monthly_cost = pricing.estimate_monthly_cost(
            instance_type="nonexistent",
            provider="aws",
        )

        assert monthly_cost == 0.0

    def test_export_pricing(self):
        """Test exporting pricing data."""
        pricing = CloudPricing()

        exported = pricing.export_pricing()

        assert isinstance(exported, str)

        data = json.loads(exported)
        assert "aws" in data
        assert "gcp" in data
        assert "azure" in data

    def test_import_pricing(self):
        """Test importing pricing data."""
        pricing = CloudPricing()

        # Create some custom data
        custom_data = {
            "custom_provider": {
                "custom.small": {
                    "instance_type": "custom.small",
                    "provider": "custom_provider",
                    "region": "custom-region",
                    "vcpus": 2,
                    "memory_gb": 4,
                    "hourly_price": 0.10,
                    "tier": "on_demand",
                },
                "custom.medium": {
                    "instance_type": "custom.medium",
                    "provider": "custom_provider",
                    "region": "custom-region",
                    "vcpus": 4,
                    "memory_gb": 8,
                    "hourly_price": 0.20,
                    "tier": "on_demand",
                },
            }
        }

        count = pricing.import_pricing(json.dumps(custom_data))

        assert count == 2
        assert "custom_provider" in pricing._instances

        instance = pricing.get_instance("custom.small", "custom_provider")
        assert instance is not None
        assert instance.vcpus == 2


class TestCloudPricingRegionalAdjustments:
    """Test regional price adjustments."""

    def test_aws_regional_adjustments(self):
        """Test AWS regional adjustments."""
        pricing = CloudPricing()

        base = pricing.get_instance("m5.xlarge", "aws", region="us-east-1")
        tokyo = pricing.get_instance("m5.xlarge", "aws", region="ap-northeast-1")

        # Tokyo should be more expensive
        assert tokyo.hourly_price > base.hourly_price

    def test_gcp_regional_adjustments(self):
        """Test GCP regional adjustments."""
        pricing = CloudPricing()

        base = pricing.get_instance("n1-standard-4", "gcp", region="us-central1")
        europe = pricing.get_instance("n1-standard-4", "gcp", region="europe-west1")

        # Europe should be more expensive
        assert europe.hourly_price > base.hourly_price

    def test_azure_regional_adjustments(self):
        """Test Azure regional adjustments."""
        pricing = CloudPricing()

        base = pricing.get_instance("Standard_D4s_v3", "azure", region="eastus")
        asia = pricing.get_instance("Standard_D4s_v3", "azure", region="southeastasia")

        # Asia should be more expensive
        assert asia.hourly_price > base.hourly_price


class TestCloudPricingInstanceTypes:
    """Test different instance type families."""

    def test_aws_compute_optimized(self):
        """Test AWS compute-optimized instances."""
        pricing = CloudPricing()

        c5 = pricing.get_instance("c5.xlarge", "aws")

        assert c5 is not None
        assert c5.vcpus == 4
        # c5 has lower memory per CPU
        assert c5.memory_gb < 16

    def test_aws_memory_optimized(self):
        """Test AWS memory-optimized instances."""
        pricing = CloudPricing()

        r5 = pricing.get_instance("r5.xlarge", "aws")

        assert r5 is not None
        assert r5.vcpus == 4
        # r5 has higher memory per CPU
        assert r5.memory_gb >= 32

    def test_aws_storage_optimized(self):
        """Test AWS storage-optimized instances."""
        pricing = CloudPricing()

        i3 = pricing.get_instance("i3.xlarge", "aws")

        assert i3 is not None
        assert i3.local_storage_gb > 0

    def test_gcp_high_memory(self):
        """Test GCP high-memory instances."""
        pricing = CloudPricing()

        highmem = pricing.get_instance("n1-highmem-4", "gcp")

        assert highmem is not None
        # High memory instances have more RAM per CPU
        assert highmem.memory_gb / highmem.vcpus > 3

    def test_gcp_high_cpu(self):
        """Test GCP high-CPU instances."""
        pricing = CloudPricing()

        highcpu = pricing.get_instance("n1-highcpu-4", "gcp")

        assert highcpu is not None
        # High CPU instances have less RAM per CPU
        assert highcpu.memory_gb / highcpu.vcpus < 2
