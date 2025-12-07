"""Data collectors for Spark applications from various sources."""

from .base_collector import BaseCollector
from .history_server_collector import HistoryServerCollector
from .event_log_collector import EventLogCollector
from .metrics_collector import MetricsCollector

try:
    from .emr_collector import EMRCollector

    __all__ = [
        "BaseCollector",
        "HistoryServerCollector",
        "EventLogCollector",
        "MetricsCollector",
        "EMRCollector",
    ]
except ImportError:
    # boto3 not installed, EMR collector not available
    __all__ = [
        "BaseCollector",
        "HistoryServerCollector",
        "EventLogCollector",
        "MetricsCollector",
    ]
