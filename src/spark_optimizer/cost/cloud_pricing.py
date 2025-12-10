"""Cloud provider pricing data and management."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

logger = logging.getLogger(__name__)


class PricingTier(Enum):
    """Pricing tier types."""

    ON_DEMAND = "on_demand"
    SPOT = "spot"
    PREEMPTIBLE = "preemptible"
    RESERVED_1Y = "reserved_1y"
    RESERVED_3Y = "reserved_3y"
    SAVINGS_PLAN = "savings_plan"
    COMMITTED_USE = "committed_use"


@dataclass
class InstancePricing:
    """Pricing information for an instance type."""

    instance_type: str
    provider: str
    region: str
    vcpus: int
    memory_gb: float
    hourly_price: float
    tier: PricingTier
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    local_storage_gb: int = 0
    network_performance: str = "moderate"
    valid_from: datetime = field(default_factory=datetime.utcnow)
    valid_until: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "instance_type": self.instance_type,
            "provider": self.provider,
            "region": self.region,
            "vcpus": self.vcpus,
            "memory_gb": self.memory_gb,
            "hourly_price": self.hourly_price,
            "tier": self.tier.value,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "local_storage_gb": self.local_storage_gb,
            "network_performance": self.network_performance,
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
        }

    @property
    def price_per_vcpu_hour(self) -> float:
        """Calculate price per vCPU hour."""
        return self.hourly_price / self.vcpus if self.vcpus > 0 else 0

    @property
    def price_per_gb_hour(self) -> float:
        """Calculate price per GB memory hour."""
        return self.hourly_price / self.memory_gb if self.memory_gb > 0 else 0


class CloudPricing:
    """Cloud pricing database and utilities.

    Provides:
    - Pricing data for AWS, GCP, Azure
    - Instance type lookup
    - Price comparison across providers
    - Regional pricing variations
    """

    # Sample pricing data (representative, not exhaustive)
    AWS_INSTANCES = {
        "m5.large": InstancePricing(
            "m5.large", "aws", "us-east-1", 2, 8, 0.096, PricingTier.ON_DEMAND
        ),
        "m5.xlarge": InstancePricing(
            "m5.xlarge", "aws", "us-east-1", 4, 16, 0.192, PricingTier.ON_DEMAND
        ),
        "m5.2xlarge": InstancePricing(
            "m5.2xlarge", "aws", "us-east-1", 8, 32, 0.384, PricingTier.ON_DEMAND
        ),
        "m5.4xlarge": InstancePricing(
            "m5.4xlarge", "aws", "us-east-1", 16, 64, 0.768, PricingTier.ON_DEMAND
        ),
        "r5.large": InstancePricing(
            "r5.large", "aws", "us-east-1", 2, 16, 0.126, PricingTier.ON_DEMAND
        ),
        "r5.xlarge": InstancePricing(
            "r5.xlarge", "aws", "us-east-1", 4, 32, 0.252, PricingTier.ON_DEMAND
        ),
        "r5.2xlarge": InstancePricing(
            "r5.2xlarge", "aws", "us-east-1", 8, 64, 0.504, PricingTier.ON_DEMAND
        ),
        "c5.large": InstancePricing(
            "c5.large", "aws", "us-east-1", 2, 4, 0.085, PricingTier.ON_DEMAND
        ),
        "c5.xlarge": InstancePricing(
            "c5.xlarge", "aws", "us-east-1", 4, 8, 0.170, PricingTier.ON_DEMAND
        ),
        "c5.2xlarge": InstancePricing(
            "c5.2xlarge", "aws", "us-east-1", 8, 16, 0.340, PricingTier.ON_DEMAND
        ),
        "i3.xlarge": InstancePricing(
            "i3.xlarge",
            "aws",
            "us-east-1",
            4,
            30.5,
            0.312,
            PricingTier.ON_DEMAND,
            local_storage_gb=950,
        ),
        "i3.2xlarge": InstancePricing(
            "i3.2xlarge",
            "aws",
            "us-east-1",
            8,
            61,
            0.624,
            PricingTier.ON_DEMAND,
            local_storage_gb=1900,
        ),
    }

    GCP_INSTANCES = {
        "n1-standard-2": InstancePricing(
            "n1-standard-2", "gcp", "us-central1", 2, 7.5, 0.095, PricingTier.ON_DEMAND
        ),
        "n1-standard-4": InstancePricing(
            "n1-standard-4", "gcp", "us-central1", 4, 15, 0.190, PricingTier.ON_DEMAND
        ),
        "n1-standard-8": InstancePricing(
            "n1-standard-8", "gcp", "us-central1", 8, 30, 0.380, PricingTier.ON_DEMAND
        ),
        "n1-standard-16": InstancePricing(
            "n1-standard-16", "gcp", "us-central1", 16, 60, 0.760, PricingTier.ON_DEMAND
        ),
        "n1-highmem-2": InstancePricing(
            "n1-highmem-2", "gcp", "us-central1", 2, 13, 0.118, PricingTier.ON_DEMAND
        ),
        "n1-highmem-4": InstancePricing(
            "n1-highmem-4", "gcp", "us-central1", 4, 26, 0.237, PricingTier.ON_DEMAND
        ),
        "n1-highmem-8": InstancePricing(
            "n1-highmem-8", "gcp", "us-central1", 8, 52, 0.473, PricingTier.ON_DEMAND
        ),
        "n1-highcpu-4": InstancePricing(
            "n1-highcpu-4", "gcp", "us-central1", 4, 3.6, 0.142, PricingTier.ON_DEMAND
        ),
        "n1-highcpu-8": InstancePricing(
            "n1-highcpu-8", "gcp", "us-central1", 8, 7.2, 0.284, PricingTier.ON_DEMAND
        ),
        "n2-standard-4": InstancePricing(
            "n2-standard-4", "gcp", "us-central1", 4, 16, 0.194, PricingTier.ON_DEMAND
        ),
        "n2-standard-8": InstancePricing(
            "n2-standard-8", "gcp", "us-central1", 8, 32, 0.388, PricingTier.ON_DEMAND
        ),
    }

    AZURE_INSTANCES = {
        "Standard_D2s_v3": InstancePricing(
            "Standard_D2s_v3", "azure", "eastus", 2, 8, 0.096, PricingTier.ON_DEMAND
        ),
        "Standard_D4s_v3": InstancePricing(
            "Standard_D4s_v3", "azure", "eastus", 4, 16, 0.192, PricingTier.ON_DEMAND
        ),
        "Standard_D8s_v3": InstancePricing(
            "Standard_D8s_v3", "azure", "eastus", 8, 32, 0.384, PricingTier.ON_DEMAND
        ),
        "Standard_D16s_v3": InstancePricing(
            "Standard_D16s_v3", "azure", "eastus", 16, 64, 0.768, PricingTier.ON_DEMAND
        ),
        "Standard_E2s_v3": InstancePricing(
            "Standard_E2s_v3", "azure", "eastus", 2, 16, 0.126, PricingTier.ON_DEMAND
        ),
        "Standard_E4s_v3": InstancePricing(
            "Standard_E4s_v3", "azure", "eastus", 4, 32, 0.252, PricingTier.ON_DEMAND
        ),
        "Standard_E8s_v3": InstancePricing(
            "Standard_E8s_v3", "azure", "eastus", 8, 64, 0.504, PricingTier.ON_DEMAND
        ),
        "Standard_F4s_v2": InstancePricing(
            "Standard_F4s_v2", "azure", "eastus", 4, 8, 0.169, PricingTier.ON_DEMAND
        ),
        "Standard_F8s_v2": InstancePricing(
            "Standard_F8s_v2", "azure", "eastus", 8, 16, 0.338, PricingTier.ON_DEMAND
        ),
    }

    # Spot pricing multipliers (approximate)
    SPOT_MULTIPLIERS = {
        "aws": 0.3,
        "gcp": 0.2,
        "azure": 0.35,
    }

    # Regional price adjustments (relative to base region)
    REGIONAL_ADJUSTMENTS = {
        "aws": {
            "us-east-1": 1.0,
            "us-west-2": 1.0,
            "eu-west-1": 1.05,
            "eu-central-1": 1.08,
            "ap-northeast-1": 1.15,
            "ap-southeast-1": 1.10,
            "sa-east-1": 1.25,
        },
        "gcp": {
            "us-central1": 1.0,
            "us-east1": 1.0,
            "europe-west1": 1.05,
            "europe-west4": 1.08,
            "asia-northeast1": 1.15,
            "asia-southeast1": 1.10,
        },
        "azure": {
            "eastus": 1.0,
            "westus2": 1.0,
            "northeurope": 1.05,
            "westeurope": 1.08,
            "japaneast": 1.15,
            "southeastasia": 1.10,
        },
    }

    def __init__(self):
        """Initialize the cloud pricing database."""
        self._instances: Dict[str, Dict[str, InstancePricing]] = {
            "aws": dict(self.AWS_INSTANCES),
            "gcp": dict(self.GCP_INSTANCES),
            "azure": dict(self.AZURE_INSTANCES),
        }
        self._custom_pricing: Dict[str, InstancePricing] = {}

    def get_instance(
        self,
        instance_type: str,
        provider: str,
        tier: PricingTier = PricingTier.ON_DEMAND,
        region: Optional[str] = None,
    ) -> Optional[InstancePricing]:
        """Get pricing for an instance type.

        Args:
            instance_type: Instance type name
            provider: Cloud provider
            tier: Pricing tier
            region: Optional region for adjusted pricing

        Returns:
            InstancePricing or None if not found
        """
        provider = provider.lower()

        # Try to get from standard pricing first
        instance = None
        if provider in self._instances:
            instance = self._instances[provider].get(instance_type)

        if not instance:
            # Check custom pricing
            key = f"{provider}:{instance_type}"
            instance = self._custom_pricing.get(key)

        if not instance:
            return None

        # Clone the instance for modifications
        adjusted = InstancePricing(
            instance_type=instance.instance_type,
            provider=provider,
            region=region or instance.region,
            vcpus=instance.vcpus,
            memory_gb=instance.memory_gb,
            hourly_price=instance.hourly_price,
            tier=tier,
            gpu_count=instance.gpu_count,
            gpu_type=instance.gpu_type,
            local_storage_gb=instance.local_storage_gb,
            network_performance=instance.network_performance,
        )

        # Apply tier adjustment
        if tier in [PricingTier.SPOT, PricingTier.PREEMPTIBLE]:
            adjusted.hourly_price *= self.SPOT_MULTIPLIERS.get(provider, 0.3)
        elif tier == PricingTier.RESERVED_1Y:
            adjusted.hourly_price *= 0.6
        elif tier == PricingTier.RESERVED_3Y:
            adjusted.hourly_price *= 0.4

        # Apply regional adjustment
        if region:
            regional_adj = self.REGIONAL_ADJUSTMENTS.get(provider, {})
            adjustment = regional_adj.get(region, 1.0)
            adjusted.hourly_price *= adjustment

        return adjusted

    def find_best_instance(
        self,
        min_vcpus: int,
        min_memory_gb: float,
        provider: Optional[str] = None,
        tier: PricingTier = PricingTier.ON_DEMAND,
        prefer_local_storage: bool = False,
    ) -> List[InstancePricing]:
        """Find best instances matching requirements.

        Args:
            min_vcpus: Minimum vCPUs required
            min_memory_gb: Minimum memory in GB
            provider: Optional provider to filter by
            tier: Pricing tier
            prefer_local_storage: Prefer instances with local storage

        Returns:
            List of matching instances sorted by price
        """
        matches = []
        providers = [provider] if provider else list(self._instances.keys())

        for p in providers:
            for instance_type, instance in self._instances.get(p, {}).items():
                if instance.vcpus >= min_vcpus and instance.memory_gb >= min_memory_gb:
                    adjusted = self.get_instance(instance_type, p, tier)
                    if adjusted:
                        matches.append(adjusted)

        # Sort by price, preferring local storage if requested
        if prefer_local_storage:
            matches.sort(key=lambda i: (i.local_storage_gb == 0, i.hourly_price))
        else:
            matches.sort(key=lambda i: i.hourly_price)

        return matches

    def compare_providers(
        self,
        vcpus: int,
        memory_gb: float,
        tier: PricingTier = PricingTier.ON_DEMAND,
    ) -> Dict[str, Optional[InstancePricing]]:
        """Compare pricing across providers for similar requirements.

        Args:
            vcpus: Required vCPUs
            memory_gb: Required memory in GB
            tier: Pricing tier

        Returns:
            Dictionary of provider to best matching instance
        """
        comparison = {}

        for provider in self._instances.keys():
            matches = self.find_best_instance(
                min_vcpus=vcpus,
                min_memory_gb=memory_gb,
                provider=provider,
                tier=tier,
            )
            comparison[provider] = matches[0] if matches else None

        return comparison

    def add_custom_pricing(self, pricing: InstancePricing) -> None:
        """Add custom instance pricing.

        Args:
            pricing: Custom pricing data
        """
        key = f"{pricing.provider}:{pricing.instance_type}"
        self._custom_pricing[key] = pricing

    def list_instances(
        self,
        provider: Optional[str] = None,
        min_vcpus: int = 0,
        min_memory_gb: float = 0,
    ) -> List[InstancePricing]:
        """List available instances.

        Args:
            provider: Optional provider to filter by
            min_vcpus: Minimum vCPUs filter
            min_memory_gb: Minimum memory filter

        Returns:
            List of matching instances
        """
        instances = []
        providers = [provider] if provider else list(self._instances.keys())

        for p in providers:
            for instance in self._instances.get(p, {}).values():
                if instance.vcpus >= min_vcpus and instance.memory_gb >= min_memory_gb:
                    instances.append(instance)

        return sorted(instances, key=lambda i: (i.provider, i.hourly_price))

    def get_regional_prices(
        self,
        instance_type: str,
        provider: str,
        tier: PricingTier = PricingTier.ON_DEMAND,
    ) -> Dict[str, float]:
        """Get prices for an instance across regions.

        Args:
            instance_type: Instance type name
            provider: Cloud provider
            tier: Pricing tier

        Returns:
            Dictionary of region to hourly price
        """
        base = self.get_instance(instance_type, provider, tier)
        if not base:
            return {}

        regional_adj = self.REGIONAL_ADJUSTMENTS.get(provider.lower(), {})
        prices = {}

        for region, adjustment in regional_adj.items():
            price = base.hourly_price * adjustment
            prices[region] = round(price, 4)

        return prices

    def estimate_monthly_cost(
        self,
        instance_type: str,
        provider: str,
        count: int = 1,
        hours_per_day: float = 24,
        days_per_month: int = 30,
        tier: PricingTier = PricingTier.ON_DEMAND,
    ) -> float:
        """Estimate monthly cost for instances.

        Args:
            instance_type: Instance type name
            provider: Cloud provider
            count: Number of instances
            hours_per_day: Hours running per day
            days_per_month: Days per month
            tier: Pricing tier

        Returns:
            Estimated monthly cost in USD
        """
        instance = self.get_instance(instance_type, provider, tier)
        if not instance:
            return 0.0

        total_hours = hours_per_day * days_per_month
        return instance.hourly_price * count * total_hours

    def export_pricing(self) -> str:
        """Export all pricing data as JSON.

        Returns:
            JSON string of pricing data
        """
        data = {}
        for provider, instances in self._instances.items():
            data[provider] = {
                name: instance.to_dict() for name, instance in instances.items()
            }
        return json.dumps(data, indent=2)

    def import_pricing(self, pricing_json: str) -> int:
        """Import pricing data from JSON.

        Args:
            pricing_json: JSON string of pricing data

        Returns:
            Number of instances imported
        """
        data = json.loads(pricing_json)
        count = 0

        for provider, instances in data.items():
            if provider not in self._instances:
                self._instances[provider] = {}

            for name, instance_data in instances.items():
                try:
                    pricing = InstancePricing(
                        instance_type=instance_data["instance_type"],
                        provider=instance_data["provider"],
                        region=instance_data["region"],
                        vcpus=instance_data["vcpus"],
                        memory_gb=instance_data["memory_gb"],
                        hourly_price=instance_data["hourly_price"],
                        tier=PricingTier(instance_data["tier"]),
                        gpu_count=instance_data.get("gpu_count", 0),
                        gpu_type=instance_data.get("gpu_type"),
                        local_storage_gb=instance_data.get("local_storage_gb", 0),
                        network_performance=instance_data.get(
                            "network_performance", "moderate"
                        ),
                    )
                    self._instances[provider][name] = pricing
                    count += 1
                except Exception as e:
                    logger.warning(f"Failed to import instance {name}: {e}")

        return count
