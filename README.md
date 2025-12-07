# Spark Resource Optimizer

An open-source tool that analyzes historical Spark job runs to recommend optimal resource configurations for future executions.

## Features

- **Multiple Data Collection Methods**
  - Parse Spark event logs
  - Query Spark History Server API
  - Integrate with cloud provider APIs (AWS EMR, Databricks, GCP Dataproc)

- **Intelligent Recommendations**
  - Similarity-based matching with historical jobs
  - ML-powered predictions for runtime and resource needs
  - Rule-based optimization for common anti-patterns

- **Cost Optimization**
  - Balance performance vs. cost trade-offs
  - Support for spot/preemptible instances
  - Multi-cloud cost comparison

- **Easy Integration**
  - REST API for programmatic access
  - CLI for manual operations
  - Python SDK for custom workflows

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gridatek/spark-resource-optimizer.git
cd spark-resource-optimizer

# Install dependencies
pip install -e .

# Setup database
python scripts/setup_db.py
```

### Basic Usage

```bash
# Collect data from Spark event logs
spark-optimizer collect --event-log-dir /path/to/spark/logs

# Collect data from Spark History Server
spark-optimizer collect-from-history-server --history-server-url http://localhost:18080

# Get recommendations for a new job
spark-optimizer recommend --input-size 10GB --job-type etl

# Start the API server
spark-optimizer serve --port 8080
```

## Architecture

### Data Collection Layer
Collects metrics from various Spark deployment platforms and stores them in a normalized format.

### Analysis Layer
Extracts features from historical jobs and identifies patterns in resource usage and performance.

### Recommendation Layer
Uses similarity matching, ML models, and heuristics to suggest optimal configurations.

### API Layer
Provides REST endpoints and CLI commands for accessing recommendations.

## Project Structure

```
spark-resource-optimizer/
├── README.md
├── LICENSE
├── setup.py
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── docker-compose.yml
├── docs/
│   ├── architecture.md
│   ├── data-collection.md
│   ├── recommendation-engine.md
│   └── api-reference.md
├── src/
│   └── spark_optimizer/
│       ├── __init__.py
│       ├── collectors/
│       ├── storage/
│       ├── analyzer/
│       ├── recommender/
│       ├── api/
│       ├── cli/
│       └── utils/
├── tests/
├── examples/
└── scripts/
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=spark_optimizer
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Community

- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and general discussion
- Slack: [Join our community](link-to-slack)

## Citation

If you use this tool in your research or production systems, please cite:

```bibtex
@software{spark_resource_optimizer,
  title = {Spark Resource Optimizer},
  author = {Gridatek},
  year = {2024},
  url = {https://github.com/gridatek/spark-resource-optimizer}
}
```

## Roadmap

- [x] Basic event log parsing
- [x] SQLite storage backend
- [x] Similarity-based recommendations
- [ ] Integration with Spark History Server
- [ ] ML-based prediction models
- [ ] Cloud provider integrations (AWS EMR, Databricks)
- [ ] Web UI dashboard
- [ ] Real-time monitoring and alerts
- [ ] Auto-tuning capabilities
