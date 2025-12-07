#!/bin/bash
# Quick demo of the Spark Resource Optimizer CLI

set -e  # Exit on error

echo "=========================================="
echo "Spark Resource Optimizer - Quick Demo"
echo "=========================================="
echo

# Activate virtual environment
source venv/bin/activate

# 1. Check version
echo "1. Checking version..."
spark-optimizer --version
echo

# 2. Initialize database
echo "2. Initializing database..."
python -c "
from spark_optimizer.storage.database import Database
from spark_optimizer.storage.models import SparkApplication
db = Database('sqlite:///demo.db')
db.create_tables()
print('✓ Database initialized: demo.db')
"
echo

# 3. View database stats (will be empty initially)
echo "3. Checking database stats..."
spark-optimizer stats --db-url sqlite:///demo.db
echo

# 4. Get a recommendation
echo "4. Getting resource recommendation for 50GB ETL job..."
spark-optimizer recommend \
  --input-size 50GB \
  --job-type etl \
  --priority balanced \
  --db-url sqlite:///demo.db \
  --format table
echo

# 5. Try another recommendation with different parameters
echo "5. Getting cost-optimized recommendation for 10GB job..."
spark-optimizer recommend \
  --input-size 10GB \
  --job-type ml \
  --priority cost \
  --db-url sqlite:///demo.db \
  --format json | head -20
echo

# 6. Show spark-submit format
echo "6. Getting recommendation in spark-submit format..."
spark-optimizer recommend \
  --input-size 100GB \
  --job-type etl \
  --priority performance \
  --db-url sqlite:///demo.db \
  --format spark-submit
echo

echo "=========================================="
echo "Demo complete!"
echo "=========================================="
echo
echo "Note: Recommendations use fallback logic since similarity"
echo "      matching is not yet implemented (shows as 50% confidence)"
echo
echo "To test with real data:"
echo "  1. Get Spark event logs from your cluster"
echo "  2. Run: spark-optimizer collect --event-log-dir /path/to/logs"
echo "  3. Then recommendations will be based on historical data"
echo
echo "To start the API server:"
echo "  spark-optimizer serve --port 8080"
echo
echo "Clean up:"
echo "  rm -f demo.db"
