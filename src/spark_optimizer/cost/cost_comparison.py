"""Multi-cloud cost comparison capabilities."""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .cost_model import CostModel, CostEstimate, InstanceType
from .cloud_pricing import CloudPricing, PricingTier, InstancePricing

logger = logging.getLogger(__name__)


@dataclass
class CloudComparison:
    """Cost comparison across cloud providers."""

    config: Dict
    duration_hours: float
    comparisons: Dict[str, CostEstimate] = field(default_factory=dict)
    cheapest_provider: Optional[str] = None
    cheapest_cost: float = 0.0
    most_expensive_provider: Optional[str] = None
    most_expensive_cost: float = 0.0
    savings_vs_most_expensive: float = 0.0
    provider_rankings: List[Tuple[str, float]] = field(default_factory=list)
    regional_analysis: Dict[str, Dict] = field(default_factory=dict)
    calculated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "config": self.config,
            "duration_hours": self.duration_hours,
            "comparisons": {p: e.to_dict() for p, e in self.comparisons.items()},
            "cheapest_provider": self.cheapest_provider,
            "cheapest_cost": round(self.cheapest_cost, 4),
            "most_expensive_provider": self.most_expensive_provider,
            "most_expensive_cost": round(self.most_expensive_cost, 4),
            "savings_vs_most_expensive": round(self.savings_vs_most_expensive, 4),
            "provider_rankings": [
                {"provider": p, "cost": round(c, 4)} for p, c in self.provider_rankings
            ],
            "regional_analysis": self.regional_analysis,
            "calculated_at": self.calculated_at.isoformat(),
        }


class CostComparison:
    """Multi-cloud cost comparison and analysis.

    Provides:
    - Comparison across AWS, GCP, Azure
    - Regional cost analysis
    - Instance type recommendations per provider
    - Migration cost estimates
    """

    PROVIDERS = ["aws", "gcp", "azure"]

    PROVIDER_REGIONS = {
        "aws": ["us-east-1", "us-west-2", "eu-west-1", "ap-northeast-1"],
        "gcp": ["us-central1", "us-east1", "europe-west1", "asia-northeast1"],
        "azure": ["eastus", "westus2", "northeurope", "japaneast"],
    }

    def __init__(self, cloud_pricing: Optional[CloudPricing] = None):
        """Initialize the cost comparison engine.

        Args:
            cloud_pricing: Cloud pricing data
        """
        self._cloud_pricing = cloud_pricing or CloudPricing()
        self._cost_models: Dict[str, CostModel] = {
            provider: CostModel(cloud_provider=provider) for provider in self.PROVIDERS
        }

    def compare_providers(
        self,
        config: Dict,
        duration_hours: float,
        include_regions: bool = False,
    ) -> CloudComparison:
        """Compare costs across cloud providers.

        Args:
            config: Spark configuration
            duration_hours: Estimated job duration
            include_regions: Whether to include regional analysis

        Returns:
            CloudComparison with detailed analysis
        """
        comparisons = {}

        for provider in self.PROVIDERS:
            estimate = self._cost_models[provider].estimate_from_config(
                config, duration_hours
            )
            comparisons[provider] = estimate

        # Find cheapest and most expensive
        costs = [(p, e.total_cost) for p, e in comparisons.items()]
        costs.sort(key=lambda x: x[1])

        cheapest = costs[0]
        most_expensive = costs[-1]

        savings = most_expensive[1] - cheapest[1]

        # Regional analysis if requested
        regional_analysis = {}
        if include_regions:
            for provider in self.PROVIDERS:
                regional_analysis[provider] = self._analyze_regions(
                    config, duration_hours, provider
                )

        return CloudComparison(
            config=config,
            duration_hours=duration_hours,
            comparisons=comparisons,
            cheapest_provider=cheapest[0],
            cheapest_cost=cheapest[1],
            most_expensive_provider=most_expensive[0],
            most_expensive_cost=most_expensive[1],
            savings_vs_most_expensive=savings,
            provider_rankings=costs,
            regional_analysis=regional_analysis,
        )

    def compare_regions(
        self,
        config: Dict,
        duration_hours: float,
        provider: str,
    ) -> Dict[str, float]:
        """Compare costs across regions for a provider.

        Args:
            config: Spark configuration
            duration_hours: Estimated job duration
            provider: Cloud provider

        Returns:
            Dictionary of region to cost
        """
        return self._analyze_regions(config, duration_hours, provider)

    def find_cheapest_option(
        self,
        config: Dict,
        duration_hours: float,
        include_spot: bool = True,
    ) -> Dict:
        """Find the cheapest provider/region combination.

        Args:
            config: Spark configuration
            duration_hours: Estimated job duration
            include_spot: Whether to include spot pricing

        Returns:
            Dictionary with cheapest option details
        """
        options = []

        for provider in self.PROVIDERS:
            for region in self.PROVIDER_REGIONS.get(provider, []):
                # On-demand cost
                model = CostModel(cloud_provider=provider, region=region)
                estimate = model.estimate_from_config(config, duration_hours)

                options.append(
                    {
                        "provider": provider,
                        "region": region,
                        "tier": "on_demand",
                        "cost": estimate.total_cost,
                    }
                )

                # Spot cost if requested
                if include_spot:
                    spot_estimate = model.estimate_job_cost(
                        num_executors=config.get("spark.executor.instances", 2),
                        executor_cores=config.get("spark.executor.cores", 4),
                        executor_memory_mb=config.get("spark.executor.memory", 4096),
                        driver_memory_mb=config.get("spark.driver.memory", 2048),
                        duration_hours=duration_hours,
                        instance_type=InstanceType.SPOT,
                    )

                    options.append(
                        {
                            "provider": provider,
                            "region": region,
                            "tier": "spot",
                            "cost": spot_estimate.total_cost,
                        }
                    )

        # Sort by cost
        options.sort(key=lambda x: float(x["cost"]))  # type: ignore[arg-type]

        return {
            "cheapest": options[0] if options else None,
            "all_options": options[:10],  # Top 10
            "total_options_analyzed": len(options),
        }

    def estimate_migration_cost(
        self,
        config: Dict,
        duration_hours: float,
        from_provider: str,
        to_provider: str,
        data_size_gb: float,
    ) -> Dict:
        """Estimate cost of migrating between providers.

        Args:
            config: Spark configuration
            duration_hours: Estimated job duration
            from_provider: Current provider
            to_provider: Target provider
            data_size_gb: Data to migrate in GB

        Returns:
            Migration cost analysis
        """
        # Current provider cost
        from_model = self._cost_models.get(from_provider, CostModel())
        from_estimate = from_model.estimate_from_config(config, duration_hours)

        # Target provider cost
        to_model = self._cost_models.get(to_provider, CostModel())
        to_estimate = to_model.estimate_from_config(config, duration_hours)

        # Egress cost (typically $0.09/GB for internet egress)
        egress_cost = data_size_gb * 0.09

        # Ingress is usually free
        ingress_cost = 0.0

        # One-time migration costs
        migration_overhead = egress_cost + ingress_cost

        # Calculate ongoing savings
        monthly_savings = (
            (from_estimate.total_cost - to_estimate.total_cost) * 720 / duration_hours
        )
        # Assumes 720 hours/month of job runtime

        # Payback period
        if monthly_savings > 0:
            payback_months = migration_overhead / monthly_savings
        else:
            payback_months = float("inf")

        return {
            "from_provider": from_provider,
            "to_provider": to_provider,
            "current_job_cost": round(from_estimate.total_cost, 4),
            "new_job_cost": round(to_estimate.total_cost, 4),
            "job_savings": round(from_estimate.total_cost - to_estimate.total_cost, 4),
            "egress_cost": round(egress_cost, 4),
            "ingress_cost": round(ingress_cost, 4),
            "total_migration_cost": round(migration_overhead, 4),
            "estimated_monthly_savings": round(monthly_savings, 4),
            "payback_months": (
                round(payback_months, 1) if payback_months != float("inf") else None
            ),
            "recommendation": self._migration_recommendation(
                monthly_savings, migration_overhead, payback_months
            ),
        }

    def get_provider_summary(self, provider: str) -> Dict:
        """Get a summary of a cloud provider's offerings.

        Args:
            provider: Cloud provider name

        Returns:
            Provider summary
        """
        instances = self._cloud_pricing.list_instances(provider=provider)

        if not instances:
            return {
                "provider": provider,
                "available": False,
            }

        # Calculate statistics
        prices = [i.hourly_price for i in instances]
        vcpus = [i.vcpus for i in instances]
        memory = [i.memory_gb for i in instances]

        return {
            "provider": provider,
            "available": True,
            "instance_count": len(instances),
            "regions": self.PROVIDER_REGIONS.get(provider, []),
            "price_range": {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices),
            },
            "vcpu_range": {
                "min": min(vcpus),
                "max": max(vcpus),
            },
            "memory_range": {
                "min": min(memory),
                "max": max(memory),
            },
            "spot_discount": f"{(1 - CostModel.SPOT_DISCOUNTS.get(provider, 0.3)) * 100:.0f}%",
        }

    def recommend_provider(
        self,
        requirements: Dict,
        priorities: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Recommend a cloud provider based on requirements.

        Args:
            requirements: Requirements dictionary (vcpus, memory, duration, etc.)
            priorities: Priority weights for different factors

        Returns:
            Provider recommendation
        """
        priorities = priorities or {
            "cost": 0.4,
            "performance": 0.3,
            "reliability": 0.3,
        }

        scores = {}

        config = {
            "spark.executor.instances": requirements.get("executors", 2),
            "spark.executor.cores": requirements.get("vcpus", 4),
            "spark.executor.memory": requirements.get("memory_mb", 4096),
            "spark.driver.memory": requirements.get("driver_memory_mb", 2048),
        }
        duration = requirements.get("duration_hours", 1.0)

        for provider in self.PROVIDERS:
            # Cost score (lower is better)
            estimate = self._cost_models[provider].estimate_from_config(
                config, duration
            )
            cost_score = 1 / (1 + estimate.total_cost)  # Normalize

            # Performance score (based on spot availability and instance variety)
            instances = self._cloud_pricing.list_instances(provider=provider)
            perf_score = min(1.0, len(instances) / 20)  # More options = better

            # Reliability score (simplified - could be based on SLA data)
            reliability_scores = {"aws": 0.95, "gcp": 0.93, "azure": 0.94}
            reliability_score = reliability_scores.get(provider, 0.9)

            # Weighted score
            total_score = (
                priorities["cost"] * cost_score
                + priorities["performance"] * perf_score
                + priorities["reliability"] * reliability_score
            )

            scores[provider] = {
                "total_score": total_score,
                "cost_score": cost_score,
                "performance_score": perf_score,
                "reliability_score": reliability_score,
                "estimated_cost": estimate.total_cost,
            }

        # Rank providers
        ranked = sorted(scores.items(), key=lambda x: x[1]["total_score"], reverse=True)

        return {
            "recommended_provider": ranked[0][0],
            "scores": {p: s for p, s in ranked},
            "requirements": requirements,
            "priorities": priorities,
        }

    def _analyze_regions(
        self,
        config: Dict,
        duration_hours: float,
        provider: str,
    ) -> Dict[str, float]:
        """Analyze costs across regions for a provider.

        Args:
            config: Spark configuration
            duration_hours: Estimated job duration
            provider: Cloud provider

        Returns:
            Dictionary of region to cost
        """
        regions = self.PROVIDER_REGIONS.get(provider, [])
        costs = {}

        for region in regions:
            model = CostModel(cloud_provider=provider, region=region)
            estimate = model.estimate_from_config(config, duration_hours)
            costs[region] = round(estimate.total_cost, 4)

        return costs

    def _migration_recommendation(
        self,
        monthly_savings: float,
        migration_cost: float,
        payback_months: float,
    ) -> str:
        """Generate migration recommendation.

        Args:
            monthly_savings: Monthly savings after migration
            migration_cost: One-time migration cost
            payback_months: Months to recover migration cost

        Returns:
            Recommendation string
        """
        if monthly_savings <= 0:
            return "Migration not recommended - no cost savings expected"

        if payback_months < 3:
            return "Strongly recommended - quick payback period"
        elif payback_months < 6:
            return "Recommended - reasonable payback period"
        elif payback_months < 12:
            return "Consider carefully - long payback period"
        else:
            return "Not recommended - payback period exceeds 1 year"
