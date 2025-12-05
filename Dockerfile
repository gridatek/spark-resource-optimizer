# Multi-stage build for smaller final image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY setup.py README.md LICENSE ./
COPY scripts/ ./scripts/
COPY examples/ ./examples/

# Install the application
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/event_logs /app/data /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SPARK_OPTIMIZER_DB_URL=sqlite:////app/data/spark_optimizer.db
ENV SPARK_OPTIMIZER_LOG_LEVEL=INFO

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Create non-root user
RUN useradd -m -u 1000 sparkopt && \
    chown -R sparkopt:sparkopt /app
USER sparkopt

# Default command
CMD ["spark-optimizer", "serve", "--host", "0.0.0.0", "--port", "8080"]
