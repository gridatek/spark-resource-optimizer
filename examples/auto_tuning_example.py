"""Auto-tuning example for Spark Resource Optimizer.

This example demonstrates the auto-tuning functionality:
1. Creating a tuning session
2. Configuring tuning strategy and target metrics
3. Running the auto-tuner with simulated metrics
4. Reviewing adjustments and improvements

To run this example:
    python examples/auto_tuning_example.py

Note: This example uses simulated data. In production, the auto-tuner
integrates with real Spark metrics from the monitoring module.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spark_optimizer.tuning.auto_tuner import (
    AutoTuner,
    TuningSession,
    TuningStrategy,
    TuningConfig,
    TuningAdjustment,
)
from spark_optimizer.tuning.config_adjuster import (
    ConfigAdjuster,
    ConfigChange,
    AdjustmentAction,
)
from spark_optimizer.tuning.feedback_loop import (
    FeedbackLoop,
    TuningFeedback,
)


def display_config(config: dict, title: str = "Configuration"):
    """Display a Spark configuration nicely."""
    print(f"\n  {title}:")
    for key, value in sorted(config.items()):
        print(f"    {key}: {value}")


def main():
    """Demonstrate auto-tuning capabilities."""
    print("=" * 60)
    print("Spark Resource Optimizer - Auto-Tuning Example")
    print("=" * 60)

    # 1. Define initial Spark configuration
    print("\nStep 1: Setting up initial configuration...")

    initial_config = {
        "spark.executor.instances": 10,
        "spark.executor.cores": 4,
        "spark.executor.memory": "8g",
        "spark.driver.memory": "4g",
        "spark.sql.shuffle.partitions": 200,
        "spark.dynamicAllocation.enabled": "false",
        "spark.memory.fraction": 0.6,
        "spark.memory.storageFraction": 0.5,
    }

    display_config(initial_config, "Initial Configuration")

    # 2. Initialize the AutoTuner
    print("\nStep 2: Initializing AutoTuner...")

    tuner = AutoTuner(
        strategy=TuningStrategy.MODERATE,
        target_metric="duration",  # Optimize for job duration
        max_iterations=10,
        convergence_threshold=0.05,  # 5% improvement threshold
    )
    print(f"  Strategy: {tuner.strategy.value}")
    print(f"  Target metric: {tuner.target_metric}")
    print(f"  Max iterations: {tuner.max_iterations}")

    # 3. Start a tuning session
    print("\nStep 3: Starting tuning session...")

    session = tuner.start_session(
        app_id="app-20241210-tuning-demo",
        app_name="ETL Pipeline - Auto-Tuning Demo",
        initial_config=initial_config,
    )

    print(f"  Session ID: {session.session_id}")
    print(f"  Status: {session.status}")

    # 4. Initialize the ConfigAdjuster
    print("\nStep 4: Initializing ConfigAdjuster...")

    adjuster = ConfigAdjuster(
        min_executors=2,
        max_executors=20,
        min_memory_mb=2048,
        max_memory_mb=32768,
        min_cores=1,
        max_cores=8,
    )
    print("  ConfigAdjuster initialized with boundaries")

    # 5. Simulate tuning iterations
    print("\nStep 5: Running tuning iterations...")

    # Simulated metrics that would come from monitoring
    simulated_metrics_sequence = [
        # Iteration 1: High GC, memory pressure
        {
            "duration_seconds": 3600,
            "gc_time_percent": 15.0,
            "memory_spill_mb": 4096,
            "cpu_utilization": 45.0,
            "shuffle_spill_mb": 2048,
        },
        # Iteration 2: After memory increase
        {
            "duration_seconds": 3200,
            "gc_time_percent": 8.0,
            "memory_spill_mb": 1024,
            "cpu_utilization": 55.0,
            "shuffle_spill_mb": 1024,
        },
        # Iteration 3: After partition adjustment
        {
            "duration_seconds": 2900,
            "gc_time_percent": 6.0,
            "memory_spill_mb": 256,
            "cpu_utilization": 70.0,
            "shuffle_spill_mb": 256,
        },
        # Iteration 4: Near optimal
        {
            "duration_seconds": 2700,
            "gc_time_percent": 5.0,
            "memory_spill_mb": 0,
            "cpu_utilization": 75.0,
            "shuffle_spill_mb": 128,
        },
    ]

    current_config = initial_config.copy()

    for i, metrics in enumerate(simulated_metrics_sequence):
        print(f"\n  --- Iteration {i + 1}/{len(simulated_metrics_sequence)} ---")
        print(f"  Metrics: duration={metrics['duration_seconds']}s, "
              f"GC={metrics['gc_time_percent']}%, "
              f"CPU={metrics['cpu_utilization']}%")

        # Analyze metrics and get adjustments
        changes = adjuster.analyze_and_adjust(
            current_config=current_config,
            metrics=metrics,
            strategy=TuningStrategy.MODERATE,
        )

        if changes:
            print(f"  Proposed adjustments ({len(changes)}):")
            for change in changes:
                print(f"    - {change.parameter}: {change.old_value} -> {change.new_value}")
                print(f"      Reason: {change.reason}")

                # Apply the change
                current_config[change.parameter] = change.new_value

                # Record adjustment in session
                adjustment = TuningAdjustment(
                    parameter=change.parameter,
                    old_value=change.old_value,
                    new_value=change.new_value,
                    reason=change.reason,
                    applied=True,
                )
                session.adjustments.append(adjustment)
        else:
            print("  No adjustments needed")

        # Update session metrics
        session.iterations += 1
        session.metrics_history.append(metrics)
        session.current_config = current_config.copy()

        # Track best configuration
        if (session.best_metric_value is None or
                metrics["duration_seconds"] < session.best_metric_value):
            session.best_metric_value = metrics["duration_seconds"]
            session.best_config = current_config.copy()

        time.sleep(0.5)  # Brief pause for visibility

    # 6. Initialize feedback loop for learning
    print("\nStep 6: Recording feedback for learning...")

    feedback_loop = FeedbackLoop()

    # Calculate improvement
    initial_duration = simulated_metrics_sequence[0]["duration_seconds"]
    final_duration = simulated_metrics_sequence[-1]["duration_seconds"]
    improvement_percent = ((initial_duration - final_duration) / initial_duration) * 100

    feedback = TuningFeedback(
        session_id=session.session_id,
        app_id=session.app_id,
        initial_metrics=simulated_metrics_sequence[0],
        final_metrics=simulated_metrics_sequence[-1],
        improvement_percent=improvement_percent,
        adjustments_applied=len(session.adjustments),
        success=True,
    )

    feedback_loop.record_feedback(feedback)
    print(f"  Feedback recorded")
    print(f"  Improvement: {improvement_percent:.1f}%")

    # 7. End the tuning session
    print("\nStep 7: Completing tuning session...")

    session.status = "completed"
    session.ended_at = datetime.utcnow()

    print(f"  Session Status: {session.status}")
    print(f"  Total Iterations: {session.iterations}")
    print(f"  Adjustments Made: {len(session.adjustments)}")
    print(f"  Best Duration: {session.best_metric_value}s")

    display_config(session.best_config, "Optimized Configuration")

    # 8. Show improvement summary
    print("\nStep 8: Improvement Summary")
    print("-" * 60)
    print(f"  Initial Duration:  {initial_duration:,} seconds ({initial_duration/60:.1f} min)")
    print(f"  Final Duration:    {final_duration:,} seconds ({final_duration/60:.1f} min)")
    print(f"  Time Saved:        {initial_duration - final_duration:,} seconds")
    print(f"  Improvement:       {improvement_percent:.1f}%")
    print("-" * 60)

    # 9. Show key changes
    print("\nStep 9: Key configuration changes:")
    for adj in session.adjustments[:5]:  # Show up to 5
        print(f"  - {adj.parameter}")
        print(f"    From: {adj.old_value}")
        print(f"    To:   {adj.new_value}")
        print(f"    Why:  {adj.reason}")
        print()

    # 10. Production usage example
    print("\nStep 10: Production configuration example:")
    print("-" * 60)
    config_example = """
# Auto-tuner configuration for production:

from spark_optimizer.tuning.auto_tuner import AutoTuner, TuningStrategy
from spark_optimizer.monitoring.monitor import SparkMonitor

# Initialize with monitoring integration
monitor = SparkMonitor(
    metrics_endpoint="http://spark-driver:4040/metrics/json"
)

tuner = AutoTuner(
    strategy=TuningStrategy.MODERATE,  # or CONSERVATIVE, AGGRESSIVE
    target_metric="duration",  # or "cost", "throughput"
    max_iterations=20,
    convergence_threshold=0.03,  # 3% improvement threshold
)

# Start session with auto-monitoring
session = tuner.start_session(
    app_id="your-app-id",
    app_name="Your Spark Application",
    initial_config=your_spark_config,
    monitor=monitor,  # Integrate with monitoring
)

# Let the tuner run automatically
tuner.run_auto_tuning(session)
"""
    print(config_example)
    print("-" * 60)

    print("\n" + "=" * 60)
    print("Auto-tuning example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
