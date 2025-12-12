#!/usr/bin/env python3
"""
Memory-Intensive Spark Job
A job designed to use significant memory to test memory optimization recommendations.
This job creates large datasets and performs operations that cache data in memory.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, array, lit, rand, sum as _sum
import sys


def main():
    # Create Spark session with moderate memory
    spark = SparkSession.builder \
        .appName("Memory-Intensive Job") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "file:///spark-events") \
        .config("spark.executor.memory", "1g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    try:
        print("Starting Memory-Intensive Job...")

        # Generate a large dataset that will be cached
        print("Generating large dataset...")

        # Create initial dataset
        df = spark.range(0, 10000000)

        # Add multiple columns with random data
        for i in range(20):
            df = df.withColumn(f"data_{i}", rand() * 1000000)

        # Cache the dataframe (this will use significant memory)
        df.cache()

        # Force materialization
        row_count = df.count()
        print(f"Generated dataset with {row_count:,} rows")

        # Perform multiple aggregations (will use cached data)
        print("\nPerforming aggregations...")

        # Aggregation 1: Sum of all data columns
        agg_exprs = [_sum(f"data_{i}").alias(f"sum_{i}") for i in range(20)]
        result1 = df.agg(*agg_exprs)
        result1.show()

        # Aggregation 2: Group by ranges
        print("\nGrouping by ID ranges...")
        result2 = df.withColumn("id_range", (col("id") / 1000000).cast("int")) \
            .groupBy("id_range") \
            .agg(
                _sum("data_0").alias("sum_data_0"),
                _sum("data_1").alias("sum_data_1"),
                _sum("data_2").alias("sum_data_2")
            ) \
            .orderBy("id_range")

        result2.show()

        # Create a second large dataset and join
        print("\nCreating second dataset and joining...")
        df2 = spark.range(0, 5000000).withColumn("value", rand() * 100)
        df2.cache()

        joined = df.join(df2, df.id == df2.id, "inner")
        join_count = joined.count()
        print(f"Join result: {join_count:,} rows")

        # Cleanup
        df.unpersist()
        df2.unpersist()

        spark.stop()
        print("\nJob completed successfully!")
        print("This job demonstrates:")
        print("  - Large dataset caching in memory")
        print("  - Multiple aggregations on cached data")
        print("  - Memory-intensive join operations")
        sys.exit(0)

    except Exception as e:
        print(f"Job failed with error: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
