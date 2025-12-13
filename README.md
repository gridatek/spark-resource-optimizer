# Spark Optimizer

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
git clone https://github.com/gridatek/spark-optimizer.git
cd spark-optimizer

# Install dependencies
pip install -e .

# Optional: Install with cloud provider support
pip install -e ".[aws]"          # For AWS EMR integration
pip install -e ".[gcp]"          # For GCP Dataproc integration

# Setup database
python scripts/setup_db.py
```

### Basic Usage

```bash
# Collect data from Spark event logs
spark-optimizer collect --event-log-dir /path/to/spark/logs

# Collect data from Spark History Server
spark-optimizer collect-from-history-server --history-server-url http://localhost:18080

# Collect data from Prometheus/Grafana metrics
spark-optimizer collect-from-metrics --metrics-endpoint http://localhost:9090

# Collect data from AWS EMR
pip install -e ".[aws]"
spark-optimizer collect-from-emr --region us-west-2

# Collect data from Databricks (uses core dependencies)
spark-optimizer collect-from-databricks --workspace-url https://dbc-xxx.cloud.databricks.com

# Collect data from GCP Dataproc
pip install -e ".[gcp]"
spark-optimizer collect-from-dataproc --project my-project --region us-central1

# Get recommendations for a new job
spark-optimizer recommend --input-size 10GB --job-type etl

# Start the API server
spark-optimizer serve --port 8080
```

## Architecture

### Overview

The Spark Resource Optimizer is designed with a modular, layered architecture that separates concerns and allows for easy extension and maintenance.

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                     │
│          (CLI, REST API Clients, Web Dashboard)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │  REST API   │  │     CLI     │  │   WebSocket  │         │
│  │   Routes    │  │  Commands   │  │  (Future)    │         │
│  └─────────────┘  └─────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Business Logic Layer                       │
│  ┌──────────────────────┐  ┌─────────────────────┐          │
│  │   Recommender        │  │   Analyzer          │          │
│  │  - Similarity        │  │  - Job Analysis     │          │
│  │  - ML-based          │  │  - Similarity       │          │
│  │  - Rule-based        │  │  - Features         │          │
│  └──────────────────────┘  └─────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Access Layer                         │
│  ┌──────────────────────────────────────────────┐           │
│  │          Repository Pattern                  │           │
│  │  - SparkApplicationRepository                │           │
│  │  - JobRecommendationRepository               │           │
│  └──────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐         │
│  │   SQLite    │  │  PostgreSQL │  │    MySQL     │         │
│  │  (Default)  │  │  (Optional) │  │  (Optional)  │         │
│  └─────────────┘  └─────────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Collection Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Event Log   │  │   History    │  │   Metrics    │       │
│  │  Collector   │  │    Server    │  │  Collector   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Spark Event  │  │    Spark     │  │  Prometheus  │       │
│  │    Logs      │  │   History    │  │   /Grafana   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Data Collection Layer

**Purpose**: Gather Spark job metrics from various sources

**Components**:
- `BaseCollector`: Abstract interface for all collectors
- `EventLogCollector`: Parse Spark event log files
- `HistoryServerCollector`: Query Spark History Server API
- `MetricsCollector`: Integrate with monitoring systems

**Key Features**:
- Pluggable collector architecture
- Batch processing support
- Error handling and retry logic
- Data normalization

#### 2. Storage Layer

**Purpose**: Persist job data and recommendations

**Components**:
- `Database`: Connection management and session handling
- `Models`: SQLAlchemy ORM models
  - `SparkApplication`: Job metadata and metrics
  - `SparkStage`: Stage-level details
  - `JobRecommendation`: Historical recommendations
- `Repository`: Data access abstraction

**Key Features**:
- Database-agnostic design (SQLAlchemy)
- Transaction management
- Query optimization
- Migration support (Alembic)

#### 3. Analysis Layer

**Purpose**: Analyze job characteristics and extract insights

**Components**:
- `JobAnalyzer`: Performance analysis and bottleneck detection
- `JobSimilarityCalculator`: Calculate job similarity scores
- `FeatureExtractor`: Extract ML features from job data

**Key Features**:
- Resource efficiency metrics
- Bottleneck identification (CPU, memory, I/O)
- Issue detection (data skew, spills, failures)
- Similarity-based job matching

#### 4. Recommendation Layer

**Purpose**: Generate optimal resource configurations

**Components**:
- `BaseRecommender`: Abstract recommender interface
- `SimilarityRecommender`: History-based recommendations
- `MLRecommender`: ML model predictions
- `RuleBasedRecommender`: Heuristic-based suggestions

**Key Features**:
- Multiple recommendation strategies
- Confidence scoring
- Cost-performance trade-offs
- Feedback loop integration

#### 5. API Layer

**Purpose**: Expose functionality to clients

**Components**:
- REST API (Flask)
- CLI interface (Click)
- WebSocket support (future)

**Endpoints**:
- `/recommend`: Get resource recommendations
- `/jobs`: List and query historical jobs
- `/analyze`: Analyze specific jobs
- `/feedback`: Submit recommendation feedback

### Data Flow

#### Collection Flow
```
Event Logs → Collector → Parser → Normalizer → Repository → Database
```

#### Recommendation Flow
```
User Request → API → Recommender → Analyzer → Repository → Database
                ↓
          Recommendation ← Model/Rules ← Historical Data
```

#### Analysis Flow
```
Job ID → Repository → Job Data → Analyzer → Insights
                                      ↓
                               Feature Extraction
                                      ↓
                               Similarity Matching
```

For more detailed architecture information, see [docs/architecture.md](docs/architecture.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this tool in your research or production systems, please cite:

```bibtex
@software{spark_resource_optimizer,
  title = {Spark Resource Optimizer},
  author = {Gridatek},
  year = {2024},
  url = {https://github.com/gridatek/spark-optimizer}
}
```
