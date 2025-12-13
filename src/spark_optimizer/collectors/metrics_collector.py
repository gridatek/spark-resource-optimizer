"""Collector for Spark metrics and monitoring systems.

TODO: This collector is reserved for future use.
Currently not implemented as Prometheus is a CLIENT of our application
(consumes metrics FROM the app), not a DATA SOURCE (provides data TO the app).

Potential future uses:
- Enrich Event Log data with real-time metrics from monitoring systems
- Supplement History Server data with additional monitoring metrics
- Integration with custom metrics exporters

For now, use EventLogCollector or HistoryServerCollector for data collection.
"""

from typing import Dict, List, Optional
import logging

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)


class MetricsCollector(BaseCollector):
    """Reserved for future metrics system integration.

    Currently not implemented. Prometheus is used as a monitoring client
    for the spark-optimizer application itself, not as a data source.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Metrics collector (placeholder).

        Args:
            config: Optional configuration dictionary
        """
        super().__init__(config)
        logger.warning(
            "MetricsCollector is not currently implemented. "
            "Use EventLogCollector or HistoryServerCollector instead."
        )

    def collect(self) -> List[Dict]:
        """Collect job data from metrics system.

        Returns:
            Empty list (not implemented)
        """
        logger.warning("MetricsCollector.collect() is not implemented")
        return []

    def validate_config(self) -> bool:
        """Validate the collector configuration.

        Returns:
            False (not implemented)
        """
        return False
