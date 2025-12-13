# Architecture

## Overview

The Spark Resource Optimizer is designed with a modular, layered architecture that separates concerns and allows for easy extension and maintenance.

## System Architecture

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
│  │  Event Log   │  │   History    │  │  Cloud APIs  │       │
│  │  Collector   │  │    Server    │  │  Collector   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Sources                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Spark Event  │  │    Spark     │  │  Cloud APIs  │       │
│  │    Logs      │  │   History    │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Data Collection Layer

**Purpose**: Gather Spark job metrics from various sources

**Components**:
- `BaseCollector`: Abstract interface for all collectors
- `EventLogCollector`: Parse Spark event log files
- `HistoryServerCollector`: Query Spark History Server API
- Cloud API Collectors: EMR, Databricks, Dataproc integrations

**Key Features**:
- Pluggable collector architecture
- Batch processing support
- Error handling and retry logic
- Data normalization

### 2. Storage Layer

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

### 3. Analysis Layer

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

### 4. Recommendation Layer

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

### 5. API Layer

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

## Data Flow

### Collection Flow
```
Event Logs → Collector → Parser → Normalizer → Repository → Database
```

### Recommendation Flow
```
User Request → API → Recommender → Analyzer → Repository → Database
                ↓
          Recommendation ← Model/Rules ← Historical Data
```

### Analysis Flow
```
Job ID → Repository → Job Data → Analyzer → Insights
                                      ↓
                               Feature Extraction
                                      ↓
                               Similarity Matching
```

## Design Patterns

### 1. Repository Pattern
- Abstracts data access logic
- Provides clean interface for CRUD operations
- Enables easy testing with mocks

### 2. Strategy Pattern
- Multiple recommender implementations
- Runtime selection of recommendation strategy
- Easy addition of new strategies

### 3. Factory Pattern
- Collector creation based on source type
- Recommender instantiation based on method
- Configuration-driven component creation

### 4. Template Method Pattern
- BaseCollector defines collection workflow
- Subclasses implement specific steps
- Consistent behavior across collectors

## Configuration Management

The system uses a hierarchical configuration approach:

1. **Default Values**: Hardcoded defaults in `config.py`
2. **Configuration File**: YAML file for persistent settings
3. **Environment Variables**: Override for deployment-specific values
4. **Runtime Arguments**: CLI/API parameters take precedence

Priority: Runtime > Environment > Config File > Defaults

## Extension Points

### Adding New Collectors
1. Extend `BaseCollector`
2. Implement `collect()` and `validate_config()`
3. Register in factory/configuration

### Adding New Recommenders
1. Extend `BaseRecommender`
2. Implement `recommend()` and `train()`
3. Add to recommender registry

### Adding New Data Sources
1. Define new collector class
2. Add connection configuration
3. Implement data normalization

### Adding New Features
1. Update `FeatureExtractor`
2. Retrain ML models
3. Update similarity calculations

## Scalability Considerations

### Horizontal Scaling
- Stateless API servers
- Load balancer distribution
- Database connection pooling

### Data Volume
- Partitioned database tables
- Time-based data retention
- Background aggregation jobs

### Performance Optimization
- Caching frequently accessed data
- Async processing for long operations
- Batch operations for bulk imports

## Security

### Authentication & Authorization
- API key authentication (future)
- Role-based access control (future)
- Rate limiting per client

### Data Protection
- Sensitive data encryption
- Secure credential storage
- Audit logging

### Input Validation
- Request parameter validation
- SQL injection prevention (ORM)
- XSS protection in API responses

## Monitoring & Observability

### Logging
- Structured logging with loguru
- Log levels: DEBUG, INFO, WARNING, ERROR
- Correlation IDs for request tracing

### Metrics
- API request latency
- Recommendation accuracy
- Database query performance
- Collection throughput

### Health Checks
- API endpoint availability
- Database connectivity
- External service status

## Deployment Architecture

### Development
```
Single machine → SQLite → Local file system
```

### Production
```
Load Balancer → API Servers → PostgreSQL
                     ↓
            Message Queue (Celery)
                     ↓
            Background Workers
```

## Future Enhancements

1. **Web Dashboard**: React-based UI for visualization
2. **Real-time Monitoring**: WebSocket streaming of job metrics
3. **Auto-tuning**: Automatic resource adjustment
4. **Multi-cloud Support**: AWS EMR, Databricks, GCP Dataproc
5. **Cost Optimization**: Spot instance recommendations
6. **Alerting**: Proactive issue detection and notifications
