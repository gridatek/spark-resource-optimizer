"""Cost modeling for Spark job execution."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InstanceType(Enum):
    """Instance type categories."""

    ON_DEMAND = "on_demand"
    SPOT = "spot"
    PREEMPTIBLE = "preemptible"
    RESERVED = "reserved"
    SAVINGS_PLAN = "savings_plan"


@dataclass
class ResourceCost:
    """Cost breakdown for a single resource type."""

    resource_type: str  # compute, memory, storage, network
    quantity: float
    unit: str
    unit_price: float
    total_cost: float
    duration_hours: float

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "resource_type": self.resource_type,
            "quantity": self.quantity,
            "unit": self.unit,
            "unit_price": self.unit_price,
            "total_cost": self.total_cost,
            "duration_hours": self.duration_hours,
        }


@dataclass
class CostEstimate:
    """Complete cost estimate for a Spark job."""

    job_id: str
    total_cost: float
    currency: str = "USD"
    breakdown: List[ResourceCost] = field(default_factory=list)
    instance_type: InstanceType = InstanceType.ON_DEMAND
    cloud_provider: str = "generic"
    region: str = "us-east-1"
    estimated_duration_hours: float = 0.0
    cost_per_hour: float = 0.0
    spot_savings: float = 0.0  # Potential savings with spot instances
    reserved_savings: float = 0.0  # Potential savings with reserved instances
    recommendations: List[str] = field(default_factory=list)
    calculated_at: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.8

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "total_cost": round(self.total_cost, 4),
            "currency": self.currency,
            "breakdown": [b.to_dict() for b in self.breakdown],
            "instance_type": self.instance_type.value,
            "cloud_provider": self.cloud_provider,
            "region": self.region,
            "estimated_duration_hours": round(self.estimated_duration_hours, 2),
            "cost_per_hour": round(self.cost_per_hour, 4),
            "spot_savings": round(self.spot_savings, 4),
            "reserved_savings": round(self.reserved_savings, 4),
            "recommendations": self.recommendations,
            "calculated_at": self.calculated_at.isoformat(),
            "confidence": self.confidence,
        }


class CostModel:
    """Advanced cost model for Spark job estimation.

    Supports:
    - Multiple cloud providers (AWS, GCP, Azure)
    - Different instance types (on-demand, spot, reserved)
    - Resource-based cost breakdown
    - Cost optimization recommendations
    """

    # Default pricing (USD per hour per unit)
    DEFAULT_PRICING = {
        "compute": {  # Per vCPU hour
            "on_demand": 0.05,
            "spot": 0.015,
            "reserved": 0.03,
        },
        "memory": {  # Per GB hour
            "on_demand": 0.005,
            "spot": 0.0015,
            "reserved": 0.003,
        },
        "storage": {  # Per GB hour
            "ssd": 0.0001,
            "hdd": 0.00005,
        },
        "network": {  # Per GB transferred
            "intra_region": 0.01,
            "inter_region": 0.02,
            "internet": 0.09,
        },
    }

    # Spot instance discount factors by provider
    SPOT_DISCOUNTS = {
        "aws": 0.3,  # 70% discount
        "gcp": 0.2,  # 80% discount (preemptible)
        "azure": 0.35,  # 65% discount
        "generic": 0.3,
    }

    # Reserved instance discount factors (1 year)
    RESERVED_DISCOUNTS = {
        "aws": 0.6,  # 40% discount
        "gcp": 0.57,  # 43% discount (committed use)
        "azure": 0.58,  # 42% discount
        "generic": 0.6,
    }

    def __init__(
        self,
        pricing: Optional[Dict] = None,
        cloud_provider: str = "generic",
        region: str = "us-east-1",
    ):
        """Initialize the cost model.

        Args:
            pricing: Custom pricing dictionary
            cloud_provider: Cloud provider name
            region: Region for pricing
        """
        self._pricing = pricing or self.DEFAULT_PRICING
        self._cloud_provider = cloud_provider.lower()
        self._region = region

    def estimate_job_cost(
        self,
        num_executors: int,
        executor_cores: int,
        executor_memory_mb: int,
        driver_memory_mb: int,
        duration_hours: float,
        instance_type: InstanceType = InstanceType.ON_DEMAND,
        shuffle_bytes: int = 0,
        input_bytes: int = 0,
        output_bytes: int = 0,
        _skip_comparisons: bool = False,
    ) -> CostEstimate:
        """Estimate the cost of a Spark job.

        Args:
            num_executors: Number of executor instances
            executor_cores: Cores per executor
            executor_memory_mb: Memory per executor in MB
            driver_memory_mb: Driver memory in MB
            duration_hours: Estimated duration in hours
            instance_type: Type of instances to use
            shuffle_bytes: Total shuffle data
            input_bytes: Total input data
            output_bytes: Total output data

        Returns:
            CostEstimate with detailed breakdown
        """
        breakdown = []
        total_cost = 0.0

        # Calculate compute cost
        total_cores = (num_executors * executor_cores) + 1  # +1 for driver
        compute_cost = self._calculate_compute_cost(
            total_cores, duration_hours, instance_type
        )
        breakdown.append(
            ResourceCost(
                resource_type="compute",
                quantity=total_cores,
                unit="vCPU-hours",
                unit_price=(
                    compute_cost / (total_cores * duration_hours)
                    if duration_hours > 0
                    else 0
                ),
                total_cost=compute_cost,
                duration_hours=duration_hours,
            )
        )
        total_cost += compute_cost

        # Calculate memory cost
        total_memory_gb = (
            (num_executors * executor_memory_mb) + driver_memory_mb
        ) / 1024
        memory_cost = self._calculate_memory_cost(
            total_memory_gb, duration_hours, instance_type
        )
        breakdown.append(
            ResourceCost(
                resource_type="memory",
                quantity=total_memory_gb,
                unit="GB-hours",
                unit_price=(
                    memory_cost / (total_memory_gb * duration_hours)
                    if duration_hours > 0
                    else 0
                ),
                total_cost=memory_cost,
                duration_hours=duration_hours,
            )
        )
        total_cost += memory_cost

        # Calculate storage cost (ephemeral for shuffle)
        shuffle_gb = shuffle_bytes / (1024**3)
        if shuffle_gb > 0:
            storage_cost = self._calculate_storage_cost(shuffle_gb, duration_hours)
            breakdown.append(
                ResourceCost(
                    resource_type="storage",
                    quantity=shuffle_gb,
                    unit="GB-hours",
                    unit_price=(
                        storage_cost / (shuffle_gb * duration_hours)
                        if duration_hours > 0
                        else 0
                    ),
                    total_cost=storage_cost,
                    duration_hours=duration_hours,
                )
            )
            total_cost += storage_cost

        # Calculate network cost
        total_network_gb = (input_bytes + output_bytes + shuffle_bytes) / (1024**3)
        if total_network_gb > 0:
            network_cost = self._calculate_network_cost(total_network_gb)
            breakdown.append(
                ResourceCost(
                    resource_type="network",
                    quantity=total_network_gb,
                    unit="GB",
                    unit_price=(
                        network_cost / total_network_gb if total_network_gb > 0 else 0
                    ),
                    total_cost=network_cost,
                    duration_hours=0,  # Network is not time-based
                )
            )
            total_cost += network_cost

        # Calculate potential savings (only if not already calculating comparisons)
        if _skip_comparisons:
            spot_cost = total_cost
            reserved_cost = total_cost
        else:
            spot_cost = (
                self.estimate_job_cost(
                    num_executors,
                    executor_cores,
                    executor_memory_mb,
                    driver_memory_mb,
                    duration_hours,
                    InstanceType.SPOT,
                    shuffle_bytes,
                    input_bytes,
                    output_bytes,
                    _skip_comparisons=True,
                ).total_cost
                if instance_type != InstanceType.SPOT
                else total_cost
            )

            reserved_cost = (
                self.estimate_job_cost(
                    num_executors,
                    executor_cores,
                    executor_memory_mb,
                    driver_memory_mb,
                    duration_hours,
                    InstanceType.RESERVED,
                    shuffle_bytes,
                    input_bytes,
                    output_bytes,
                    _skip_comparisons=True,
                ).total_cost
                if instance_type != InstanceType.RESERVED
                else total_cost
            )

        spot_savings = (
            total_cost - spot_cost if instance_type == InstanceType.ON_DEMAND else 0
        )
        reserved_savings = (
            total_cost - reserved_cost if instance_type == InstanceType.ON_DEMAND else 0
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            total_cost,
            spot_savings,
            reserved_savings,
            duration_hours,
            num_executors,
            executor_memory_mb,
            shuffle_bytes,
        )

        return CostEstimate(
            job_id=f"estimate-{int(datetime.utcnow().timestamp())}",
            total_cost=total_cost,
            breakdown=breakdown,
            instance_type=instance_type,
            cloud_provider=self._cloud_provider,
            region=self._region,
            estimated_duration_hours=duration_hours,
            cost_per_hour=total_cost / duration_hours if duration_hours > 0 else 0,
            spot_savings=spot_savings,
            reserved_savings=reserved_savings,
            recommendations=recommendations,
        )

    def estimate_from_config(
        self,
        config: Dict,
        estimated_duration_hours: float,
        data_size_gb: float = 0,
    ) -> CostEstimate:
        """Estimate cost from a Spark configuration.

        Args:
            config: Spark configuration dictionary
            estimated_duration_hours: Estimated job duration
            data_size_gb: Estimated data size in GB

        Returns:
            CostEstimate
        """
        num_executors = config.get("spark.executor.instances", 2)
        executor_cores = config.get("spark.executor.cores", 4)
        executor_memory_mb = config.get("spark.executor.memory", 4096)
        driver_memory_mb = config.get("spark.driver.memory", 2048)

        # Estimate shuffle as 2x input for typical jobs
        shuffle_bytes = int(data_size_gb * 2 * 1024**3)
        input_bytes = int(data_size_gb * 1024**3)
        output_bytes = int(data_size_gb * 0.5 * 1024**3)  # Typical output ratio

        return self.estimate_job_cost(
            num_executors=num_executors,
            executor_cores=executor_cores,
            executor_memory_mb=executor_memory_mb,
            driver_memory_mb=driver_memory_mb,
            duration_hours=estimated_duration_hours,
            shuffle_bytes=shuffle_bytes,
            input_bytes=input_bytes,
            output_bytes=output_bytes,
        )

    def estimate_from_historical(
        self,
        historical_job: Dict,
        scale_factor: float = 1.0,
    ) -> CostEstimate:
        """Estimate cost based on a historical job.

        Args:
            historical_job: Historical job data dictionary
            scale_factor: Factor to scale the estimate by

        Returns:
            CostEstimate
        """
        duration_ms = historical_job.get("duration_ms", 0)
        duration_hours = (duration_ms / 1000 / 3600) * scale_factor

        return self.estimate_job_cost(
            num_executors=historical_job.get("num_executors", 2),
            executor_cores=historical_job.get("executor_cores", 4),
            executor_memory_mb=historical_job.get("executor_memory_mb", 4096),
            driver_memory_mb=historical_job.get("driver_memory_mb", 2048),
            duration_hours=duration_hours,
            shuffle_bytes=int(
                historical_job.get("shuffle_write_bytes", 0) * scale_factor
            ),
            input_bytes=int(historical_job.get("input_bytes", 0) * scale_factor),
            output_bytes=int(historical_job.get("output_bytes", 0) * scale_factor),
        )

    def _calculate_compute_cost(
        self,
        cores: int,
        hours: float,
        instance_type: InstanceType,
    ) -> float:
        """Calculate compute cost.

        Args:
            cores: Number of vCPUs
            hours: Duration in hours
            instance_type: Instance type

        Returns:
            Cost in USD
        """
        base_price = self._pricing["compute"]["on_demand"]

        if (
            instance_type == InstanceType.SPOT
            or instance_type == InstanceType.PREEMPTIBLE
        ):
            price = base_price * self.SPOT_DISCOUNTS.get(self._cloud_provider, 0.3)
        elif (
            instance_type == InstanceType.RESERVED
            or instance_type == InstanceType.SAVINGS_PLAN
        ):
            price = base_price * self.RESERVED_DISCOUNTS.get(self._cloud_provider, 0.6)
        else:
            price = base_price

        return cores * hours * price

    def _calculate_memory_cost(
        self,
        memory_gb: float,
        hours: float,
        instance_type: InstanceType,
    ) -> float:
        """Calculate memory cost.

        Args:
            memory_gb: Memory in GB
            hours: Duration in hours
            instance_type: Instance type

        Returns:
            Cost in USD
        """
        base_price = self._pricing["memory"]["on_demand"]

        if (
            instance_type == InstanceType.SPOT
            or instance_type == InstanceType.PREEMPTIBLE
        ):
            price = base_price * self.SPOT_DISCOUNTS.get(self._cloud_provider, 0.3)
        elif (
            instance_type == InstanceType.RESERVED
            or instance_type == InstanceType.SAVINGS_PLAN
        ):
            price = base_price * self.RESERVED_DISCOUNTS.get(self._cloud_provider, 0.6)
        else:
            price = base_price

        return memory_gb * hours * price

    def _calculate_storage_cost(
        self,
        storage_gb: float,
        hours: float,
        storage_type: str = "ssd",
    ) -> float:
        """Calculate storage cost.

        Args:
            storage_gb: Storage in GB
            hours: Duration in hours
            storage_type: Type of storage

        Returns:
            Cost in USD
        """
        price = self._pricing["storage"].get(storage_type, 0.0001)
        return storage_gb * hours * price

    def _calculate_network_cost(
        self,
        data_gb: float,
        transfer_type: str = "intra_region",
    ) -> float:
        """Calculate network transfer cost.

        Args:
            data_gb: Data transferred in GB
            transfer_type: Type of transfer

        Returns:
            Cost in USD
        """
        price = self._pricing["network"].get(transfer_type, 0.01)
        return data_gb * price

    def _generate_recommendations(
        self,
        total_cost: float,
        spot_savings: float,
        reserved_savings: float,
        duration_hours: float,
        num_executors: int,
        executor_memory_mb: int,
        shuffle_bytes: int,
    ) -> List[str]:
        """Generate cost optimization recommendations.

        Args:
            total_cost: Total estimated cost
            spot_savings: Potential spot instance savings
            reserved_savings: Potential reserved instance savings
            duration_hours: Job duration
            num_executors: Number of executors
            executor_memory_mb: Memory per executor
            shuffle_bytes: Shuffle data size

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Spot instance recommendation
        if spot_savings > 0.10:  # At least 10 cents savings
            savings_pct = (spot_savings / total_cost) * 100 if total_cost > 0 else 0
            recommendations.append(
                f"Consider using spot instances to save ${spot_savings:.2f} ({savings_pct:.0f}%)"
            )

        # Reserved instance recommendation for long jobs
        if duration_hours > 2 and reserved_savings > 0.05:
            recommendations.append(
                f"For recurring jobs, reserved instances could save ${reserved_savings:.2f}"
            )

        # Memory optimization
        if executor_memory_mb > 8192 and num_executors > 5:
            recommendations.append(
                "Consider reducing executor memory if GC time is low"
            )

        # Shuffle optimization
        shuffle_gb = shuffle_bytes / (1024**3)
        if shuffle_gb > 10:
            recommendations.append(
                f"High shuffle volume ({shuffle_gb:.1f} GB). Consider optimizing partitioning"
            )

        # Executor count optimization
        if num_executors > 20 and duration_hours < 0.5:
            recommendations.append(
                "Job is short with many executors. Consider reducing executor count"
            )

        return recommendations

    def set_pricing(self, pricing: Dict) -> None:
        """Update pricing configuration.

        Args:
            pricing: New pricing dictionary
        """
        self._pricing.update(pricing)

    def set_provider(self, provider: str, region: str = "us-east-1") -> None:
        """Set cloud provider and region.

        Args:
            provider: Cloud provider name
            region: Region name
        """
        self._cloud_provider = provider.lower()
        self._region = region
