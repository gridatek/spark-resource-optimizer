#!/usr/bin/env python3
"""
CPU-Intensive Spark Job
A job designed to maximize CPU usage to test core allocation recommendations.
This job performs complex calculations and transformations.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, explode, array, lit
from pyspark.sql.types import DoubleType, StringType
import sys
import math


def main():
    # Create Spark session with limited cores
    spark = SparkSession.builder \
        .appName("CPU-Intensive Job") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "file:///spark-events") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "1") \
        .config("spark.sql.shuffle.partitions", "16") \
        .getOrCreate()

    try:
        print("Starting CPU-Intensive Job...")

        # Define CPU-intensive UDFs
        @udf(returnType=DoubleType())
        def compute_intensive_function(x):
            """Perform CPU-intensive mathematical operations."""
            result = 0.0
            for i in range(1000):
                result += math.sqrt(abs(x + i)) * math.log(abs(x + i + 1))
                result += math.sin(x + i) * math.cos(x + i)
            return result

        @udf(returnType=StringType())
        def string_intensive_function(x):
            """Perform CPU-intensive string operations."""
            text = str(x)
            result = text
            for i in range(100):
                result = result[::-1]  # Reverse string
                result = result.upper() if i % 2 == 0 else result.lower()
                result = result + text
            return result[:100]  # Limit result size

        # Generate dataset
        print("Generating dataset...")
        df = spark.range(0, 1000000)

        # Apply CPU-intensive transformations
        print("\nApplying CPU-intensive transformations...")

        # Transformation 1: Mathematical computations
        df = df.withColumn("computed_value", compute_intensive_function(col("id")))

        # Transformation 2: String manipulations
        df = df.withColumn("string_result", string_intensive_function(col("id")))

        # Force execution with multiple actions
        print("\nExecuting transformations...")

        # Action 1: Count
        count = df.count()
        print(f"Processed {count:,} rows")

        # Action 2: Compute statistics
        print("\nComputing statistics...")
        stats = df.select("computed_value").describe()
        stats.show()

        # Action 3: Sample and show
        print("\nSample results:")
        df.select("id", "computed_value", "string_result").show(10, truncate=True)

        # Perform a shuffle-intensive operation
        print("\nPerforming shuffle operations...")
        grouped = df.groupBy((col("id") % 100).alias("partition")) \
            .count() \
            .orderBy("partition")

        grouped.show()

        spark.stop()
        print("\nJob completed successfully!")
        print("This job demonstrates:")
        print("  - CPU-intensive mathematical computations")
        print("  - Complex string manipulations")
        print("  - UDF-based transformations")
        print("  - Would benefit from more executor cores")
        sys.exit(0)

    except Exception as e:
        print(f"Job failed with error: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
