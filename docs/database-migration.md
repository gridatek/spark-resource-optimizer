# Database Migration Guide

This guide covers database setup, migrations, and maintenance for the Spark Resource Optimizer.

## Overview

The Spark Resource Optimizer uses SQLAlchemy ORM with support for multiple database backends:
- **SQLite** (default, for development)
- **PostgreSQL** (recommended for production)
- **MySQL/MariaDB** (supported)

## Database Schema

### Core Tables

#### `spark_applications`
Stores metadata and metrics for Spark applications.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| app_id | String | Spark application ID (unique) |
| app_name | String | Application name |
| user | String | User who submitted the job |
| submit_time | DateTime | Job submission time |
| start_time | DateTime | Job start time |
| end_time | DateTime | Job completion time |
| duration_ms | Integer | Total job duration |
| status | String | Job status (completed, failed, running) |
| spark_version | String | Spark version used |
| executor_cores | Integer | Cores per executor |
| executor_memory_mb | Integer | Memory per executor (MB) |
| num_executors | Integer | Number of executors |
| driver_memory_mb | Integer | Driver memory (MB) |
| total_tasks | Integer | Total tasks executed |
| failed_tasks | Integer | Number of failed tasks |
| total_stages | Integer | Total stages |
| failed_stages | Integer | Number of failed stages |
| input_bytes | Integer | Total input data size |
| output_bytes | Integer | Total output data size |
| shuffle_read_bytes | Integer | Shuffle read bytes |
| shuffle_write_bytes | Integer | Shuffle write bytes |
| cpu_time_ms | Integer | Total CPU time |
| memory_spilled_bytes | Integer | Memory spilled to disk |
| disk_spilled_bytes | Integer | Disk spill bytes |
| executor_run_time_ms | Integer | Executor run time |
| executor_cpu_time_ms | Integer | Executor CPU time |
| jvm_gc_time_ms | Integer | JVM garbage collection time |
| peak_memory_usage | Integer | Peak memory usage |
| cluster_type | String | Cluster type (EMR, Dataproc, etc.) |
| instance_type | String | Instance type used |
| estimated_cost | Float | Estimated job cost |
| tags | JSON | Custom tags |
| environment | JSON | Environment variables |
| spark_configs | JSON | Spark configuration options |
| created_at | DateTime | Record creation time |
| updated_at | DateTime | Record update time |

#### `spark_stages`
Stores per-stage metrics for Spark applications.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| application_id | Integer | Foreign key to spark_applications |
| stage_id | Integer | Spark stage ID |
| stage_name | String | Stage name |
| num_tasks | Integer | Number of tasks in stage |
| submission_time | DateTime | Stage submission time |
| completion_time | DateTime | Stage completion time |
| duration_ms | Integer | Stage duration |
| executor_run_time_ms | Integer | Executor run time |
| executor_cpu_time_ms | Integer | Executor CPU time |
| memory_bytes_spilled | Integer | Memory spilled |
| disk_bytes_spilled | Integer | Disk spilled |
| input_bytes | Integer | Stage input bytes |
| output_bytes | Integer | Stage output bytes |
| shuffle_read_bytes | Integer | Shuffle read bytes |
| shuffle_write_bytes | Integer | Shuffle write bytes |
| created_at | DateTime | Record creation time |

#### `job_recommendations`
Stores generated recommendations for future reference.

| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| job_signature | String | Hash of job characteristics |
| recommended_executor_cores | Integer | Recommended cores |
| recommended_executor_memory_mb | Integer | Recommended memory |
| recommended_num_executors | Integer | Recommended executor count |
| recommended_driver_memory_mb | Integer | Recommended driver memory |
| predicted_duration_ms | Integer | Predicted duration |
| predicted_cost | Float | Predicted cost |
| confidence_score | Float | Recommendation confidence |
| recommendation_method | String | Method used (similarity, ml, rule-based) |
| similar_jobs | JSON | Similar jobs used for recommendation |
| created_at | DateTime | Record creation time |
| used_at | DateTime | When recommendation was used |
| feedback_score | Float | User feedback score |

## Initial Setup

### SQLite (Development)

```python
from spark_optimizer.storage.database import Database

# Default SQLite database
db = Database("sqlite:///spark_optimizer.db")
db.create_tables()
```

### PostgreSQL (Production)

```python
from spark_optimizer.storage.database import Database

# PostgreSQL connection
db = Database("postgresql://user:password@localhost:5432/spark_optimizer")
db.create_tables()
```

### Environment Variable Configuration

```bash
# Set database URL via environment variable
export SPARK_OPTIMIZER_DB_URL="postgresql://user:password@localhost:5432/spark_optimizer"
```

```python
import os
from spark_optimizer.storage.database import Database

db = Database(os.environ.get("SPARK_OPTIMIZER_DB_URL", "sqlite:///spark_optimizer.db"))
```

## Migration Procedures

### From SQLite to PostgreSQL

1. **Export data from SQLite**:

```bash
# Using the CLI
spark-optimizer export-data --format json --output data_export.json

# Or using Python
python -c "
from spark_optimizer.storage.database import Database
from spark_optimizer.storage.repository import SparkApplicationRepository
import json

db = Database('sqlite:///spark_optimizer.db')
with db.get_session() as session:
    repo = SparkApplicationRepository(session)
    apps = repo.get_all()
    data = [app.to_dict() for app in apps]

with open('data_export.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)
"
```

2. **Create PostgreSQL database**:

```sql
CREATE DATABASE spark_optimizer;
CREATE USER spark_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE spark_optimizer TO spark_user;
```

3. **Initialize new database**:

```python
from spark_optimizer.storage.database import Database

db = Database("postgresql://spark_user:your_password@localhost:5432/spark_optimizer")
db.create_tables()
```

4. **Import data**:

```python
import json
from spark_optimizer.storage.database import Database
from spark_optimizer.storage.repository import SparkApplicationRepository

db = Database("postgresql://spark_user:your_password@localhost:5432/spark_optimizer")

with open('data_export.json', 'r') as f:
    data = json.load(f)

with db.get_session() as session:
    repo = SparkApplicationRepository(session)
    for app_data in data:
        repo.create(app_data)
    session.commit()

print(f"Imported {len(data)} applications")
```

### Schema Updates

When upgrading to a new version with schema changes:

1. **Backup your database**:

```bash
# PostgreSQL
pg_dump spark_optimizer > backup_$(date +%Y%m%d).sql

# SQLite
cp spark_optimizer.db spark_optimizer_backup_$(date +%Y%m%d).db
```

2. **Apply migrations using Alembic** (if configured):

```bash
# Generate migration
alembic revision --autogenerate -m "Add new columns for v0.2.0"

# Apply migration
alembic upgrade head
```

3. **Manual migration (alternative)**:

```python
from sqlalchemy import text
from spark_optimizer.storage.database import Database

db = Database("your_database_url")

# Example: Adding new columns for monitoring features
with db.get_session() as session:
    # Check if column exists before adding
    try:
        session.execute(text("""
            ALTER TABLE spark_applications
            ADD COLUMN IF NOT EXISTS monitoring_enabled BOOLEAN DEFAULT FALSE
        """))
        session.execute(text("""
            ALTER TABLE spark_applications
            ADD COLUMN IF NOT EXISTS tuning_session_id VARCHAR(255)
        """))
        session.commit()
        print("Migration completed successfully")
    except Exception as e:
        session.rollback()
        print(f"Migration failed: {e}")
```

## Version 0.2.0 Migration

Version 0.2.0 introduces new tables for monitoring, tuning, and cost features. If upgrading from v0.1.x:

### New Tables Required

```sql
-- Monitoring alerts table
CREATE TABLE IF NOT EXISTS monitoring_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(255) UNIQUE NOT NULL,
    app_id VARCHAR(255) NOT NULL,
    severity VARCHAR(50) NOT NULL,
    title VARCHAR(255) NOT NULL,
    message TEXT,
    metric_name VARCHAR(255),
    metric_value FLOAT,
    threshold FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE
);

-- Tuning sessions table
CREATE TABLE IF NOT EXISTS tuning_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    app_id VARCHAR(255) NOT NULL,
    app_name VARCHAR(255),
    strategy VARCHAR(50) NOT NULL,
    target_metric VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    iterations INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    initial_config JSON,
    current_config JSON,
    best_config JSON,
    best_metric_value FLOAT
);

-- Tuning adjustments table
CREATE TABLE IF NOT EXISTS tuning_adjustments (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    parameter VARCHAR(255) NOT NULL,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    applied BOOLEAN DEFAULT FALSE,
    result TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES tuning_sessions(session_id)
);

-- Cost estimates table
CREATE TABLE IF NOT EXISTS cost_estimates (
    id SERIAL PRIMARY KEY,
    app_id VARCHAR(255),
    config JSON NOT NULL,
    duration_hours FLOAT NOT NULL,
    total_cost FLOAT NOT NULL,
    breakdown JSON,
    cloud_provider VARCHAR(50),
    region VARCHAR(50),
    instance_type VARCHAR(100),
    pricing_tier VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_alerts_app_id ON monitoring_alerts(app_id);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON monitoring_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_sessions_app_id ON tuning_sessions(app_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON tuning_sessions(status);
CREATE INDEX IF NOT EXISTS idx_costs_app_id ON cost_estimates(app_id);
```

### Python Migration Script

```python
"""Migration script for v0.1.x to v0.2.0"""

from sqlalchemy import text
from spark_optimizer.storage.database import Database

def migrate_to_v020(db_url: str):
    """Migrate database schema to v0.2.0."""
    db = Database(db_url)

    migrations = [
        # Monitoring alerts
        """
        CREATE TABLE IF NOT EXISTS monitoring_alerts (
            id SERIAL PRIMARY KEY,
            alert_id VARCHAR(255) UNIQUE NOT NULL,
            app_id VARCHAR(255) NOT NULL,
            severity VARCHAR(50) NOT NULL,
            title VARCHAR(255) NOT NULL,
            message TEXT,
            metric_name VARCHAR(255),
            metric_value FLOAT,
            threshold FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            acknowledged_at TIMESTAMP,
            acknowledged BOOLEAN DEFAULT FALSE
        )
        """,

        # Tuning sessions
        """
        CREATE TABLE IF NOT EXISTS tuning_sessions (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) UNIQUE NOT NULL,
            app_id VARCHAR(255) NOT NULL,
            app_name VARCHAR(255),
            strategy VARCHAR(50) NOT NULL,
            target_metric VARCHAR(50) NOT NULL,
            status VARCHAR(50) DEFAULT 'active',
            iterations INTEGER DEFAULT 0,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            initial_config JSON,
            current_config JSON,
            best_config JSON,
            best_metric_value FLOAT
        )
        """,

        # Tuning adjustments
        """
        CREATE TABLE IF NOT EXISTS tuning_adjustments (
            id SERIAL PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL,
            parameter VARCHAR(255) NOT NULL,
            old_value TEXT,
            new_value TEXT,
            reason TEXT,
            applied BOOLEAN DEFAULT FALSE,
            result TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,

        # Cost estimates
        """
        CREATE TABLE IF NOT EXISTS cost_estimates (
            id SERIAL PRIMARY KEY,
            app_id VARCHAR(255),
            config JSON NOT NULL,
            duration_hours FLOAT NOT NULL,
            total_cost FLOAT NOT NULL,
            breakdown JSON,
            cloud_provider VARCHAR(50),
            region VARCHAR(50),
            instance_type VARCHAR(100),
            pricing_tier VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,

        # Indexes
        "CREATE INDEX IF NOT EXISTS idx_alerts_app_id ON monitoring_alerts(app_id)",
        "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON monitoring_alerts(severity)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_app_id ON tuning_sessions(app_id)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_status ON tuning_sessions(status)",
        "CREATE INDEX IF NOT EXISTS idx_costs_app_id ON cost_estimates(app_id)",
    ]

    with db.get_session() as session:
        for migration in migrations:
            try:
                session.execute(text(migration))
                print(f"Executed: {migration[:50]}...")
            except Exception as e:
                print(f"Skipped (may already exist): {e}")
        session.commit()

    print("Migration to v0.2.0 completed!")

if __name__ == "__main__":
    import sys
    db_url = sys.argv[1] if len(sys.argv) > 1 else "sqlite:///spark_optimizer.db"
    migrate_to_v020(db_url)
```

## Backup and Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup_db.sh

DB_URL="${SPARK_OPTIMIZER_DB_URL:-sqlite:///spark_optimizer.db}"
BACKUP_DIR="/var/backups/spark_optimizer"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

if [[ $DB_URL == sqlite* ]]; then
    # SQLite backup
    DB_FILE=$(echo $DB_URL | sed 's/sqlite:\/\/\///')
    cp "$DB_FILE" "$BACKUP_DIR/spark_optimizer_$DATE.db"
elif [[ $DB_URL == postgresql* ]]; then
    # PostgreSQL backup
    pg_dump "$DB_URL" > "$BACKUP_DIR/spark_optimizer_$DATE.sql"
fi

# Keep only last 7 days of backups
find $BACKUP_DIR -type f -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/spark_optimizer_$DATE.*"
```

### Restore from Backup

```bash
# PostgreSQL restore
psql spark_optimizer < backup_file.sql

# SQLite restore
cp backup_file.db spark_optimizer.db
```

## Performance Tuning

### PostgreSQL Recommendations

```sql
-- Add recommended indexes for common queries
CREATE INDEX IF NOT EXISTS idx_apps_name ON spark_applications(app_name);
CREATE INDEX IF NOT EXISTS idx_apps_user ON spark_applications(user);
CREATE INDEX IF NOT EXISTS idx_apps_start_time ON spark_applications(start_time);
CREATE INDEX IF NOT EXISTS idx_apps_status ON spark_applications(status);

-- Composite index for common filters
CREATE INDEX IF NOT EXISTS idx_apps_user_time
ON spark_applications(user, start_time DESC);

-- Index for recommendation lookups
CREATE INDEX IF NOT EXISTS idx_recs_signature
ON job_recommendations(job_signature);
```

### Connection Pooling

```python
from spark_optimizer.storage.database import Database

# Configure connection pool for production
db = Database(
    "postgresql://user:password@localhost:5432/spark_optimizer",
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
)
```

## Troubleshooting

### Common Issues

**Issue**: "Table already exists" error during migration
```python
# Solution: Use IF NOT EXISTS in CREATE statements
session.execute(text("CREATE TABLE IF NOT EXISTS ..."))
```

**Issue**: Foreign key constraint failures
```python
# Solution: Disable foreign key checks during migration (PostgreSQL)
session.execute(text("SET session_replication_role = replica"))
# Run migrations
session.execute(text("SET session_replication_role = DEFAULT"))
```

**Issue**: Slow queries on large datasets
```sql
-- Solution: Analyze tables and update statistics
ANALYZE spark_applications;
ANALYZE spark_stages;
```

### Health Check Query

```python
from spark_optimizer.storage.database import Database
from sqlalchemy import text

def check_database_health(db_url: str) -> dict:
    """Check database health and statistics."""
    db = Database(db_url)

    with db.get_session() as session:
        stats = {}

        # Count records
        result = session.execute(text("SELECT COUNT(*) FROM spark_applications"))
        stats["applications"] = result.scalar()

        result = session.execute(text("SELECT COUNT(*) FROM spark_stages"))
        stats["stages"] = result.scalar()

        result = session.execute(text("SELECT COUNT(*) FROM job_recommendations"))
        stats["recommendations"] = result.scalar()

        # Check for v0.2.0 tables
        try:
            result = session.execute(text("SELECT COUNT(*) FROM monitoring_alerts"))
            stats["alerts"] = result.scalar()
            stats["v020_tables"] = True
        except:
            stats["v020_tables"] = False

        return stats

# Usage
stats = check_database_health("your_database_url")
print(f"Database Health: {stats}")
```

## Support

For database-related issues:
- GitHub Issues: https://github.com/gridatek/spark-resource-optimizer/issues
- Documentation: https://spark-optimizer.readthedocs.io/database
