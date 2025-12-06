"""Configuration management for the optimizer."""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration manager for the Spark Resource Optimizer."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.

        Args:
            config_path: Path to configuration file (YAML)
        """
        self.config_path = config_path
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from file and environment variables."""
        # Load from file if provided
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                self._config = yaml.safe_load(f) or {}

        # Override with environment variables
        self._load_env_variables()

        # Set defaults
        self._set_defaults()

    def _load_env_variables(self):
        """Load configuration from environment variables."""
        # Database
        if os.getenv("SPARK_OPTIMIZER_DB_URL"):
            self._config.setdefault("database", {})
            self._config["database"]["url"] = os.getenv("SPARK_OPTIMIZER_DB_URL")

        # API
        if os.getenv("SPARK_OPTIMIZER_API_HOST"):
            self._config.setdefault("api", {})
            self._config["api"]["host"] = os.getenv("SPARK_OPTIMIZER_API_HOST")

        if os.getenv("SPARK_OPTIMIZER_API_PORT"):
            self._config.setdefault("api", {})
            self._config["api"]["port"] = int(os.getenv("SPARK_OPTIMIZER_API_PORT"))

        # Logging
        if os.getenv("SPARK_OPTIMIZER_LOG_LEVEL"):
            self._config.setdefault("logging", {})
            self._config["logging"]["level"] = os.getenv("SPARK_OPTIMIZER_LOG_LEVEL")

    def _set_defaults(self):
        """Set default configuration values."""
        # Database defaults
        self._config.setdefault("database", {})
        self._config["database"].setdefault("url", "sqlite:///spark_optimizer.db")

        # API defaults
        self._config.setdefault("api", {})
        self._config["api"].setdefault("host", "0.0.0.0")
        self._config["api"].setdefault("port", 8080)
        self._config["api"].setdefault("debug", False)

        # Logging defaults
        self._config.setdefault("logging", {})
        self._config["logging"].setdefault("level", "INFO")
        self._config["logging"].setdefault(
            "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Recommender defaults
        self._config.setdefault("recommender", {})
        self._config["recommender"].setdefault("default_method", "similarity")
        self._config["recommender"].setdefault("min_similarity", 0.7)

        # Collector defaults
        self._config.setdefault("collector", {})
        self._config["collector"].setdefault("batch_size", 100)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.

        Args:
            key: Configuration key (supports dot notation, e.g., 'database.url')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def set(self, key: str, value: Any):
        """Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            config = config.setdefault(k, {})

        config[keys[-1]] = value

    def save(self, path: Optional[str] = None):
        """Save configuration to file.

        Args:
            path: Path to save configuration (uses config_path if not provided)
        """
        save_path = path or self.config_path

        if not save_path:
            raise ValueError("No path provided for saving configuration")

        with open(save_path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False)

    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary.

        Returns:
            Configuration dictionary
        """
        return self._config.copy()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or defaults.

    Args:
        config_path: Path to configuration file

    Returns:
        Config instance
    """
    return Config(config_path)
