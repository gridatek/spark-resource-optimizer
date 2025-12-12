#!/usr/bin/env python3
"""
Skewed Data Spark Job
A job with intentionally skewed data distribution to test skew detection.
This job creates datasets with uneven partition sizes.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, rand, count as _count, sum as _sum
import sys
import random


def main():
    # Create Spark session
    spark = SparkSession.builder \
        .appName("Skewed Data Job") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "file:///spark-events") \
        .config("spark.executor.memory", "2g") \
        .config("spark.executor.cores", "2") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    try:
        print("Starting Skewed Data Job...")

        # Generate dataset with intentional skew
        print("Generating skewed dataset...")

        # Create a dataset where 80% of records have the same key
        # This simulates a common data skew scenario
        df = spark.range(0, 100000)

        # Assign keys with heavy skew - 80% get key "A", rest distributed
        df = df.withColumn(
            "skewed_key",
            when(col("id") % 100 < 80, "A")  # 80% get key A
            .when(col("id") % 100 < 90, "B")  # 10% get key B
            .when(col("id") % 100 < 95, "C")  # 5% get key C
            .when(col("id") % 100 < 98, "D")  # 3% get key D
            .otherwise("E")  # 2% get key E
        )

        # Add some data columns
        df = df.withColumn("value1", rand() * 1000)
        df = df.withColumn("value2", rand() * 100)

        print("Dataset created with skewed distribution")

        # Show distribution
        print("\nKey distribution (this will show the skew):")
        df.groupBy("skewed_key").count().orderBy(col("count").desc()).show()

        # Perform operations that will be affected by skew
        print("\nPerforming group-by aggregation (will show skew effects)...")

        # This groupBy will have very uneven partition sizes
        result = df.groupBy("skewed_key").agg(
            _count("*").alias("count"),
            _sum("value1").alias("sum_value1"),
            _sum("value2").alias("sum_value2")
        ).orderBy(col("count").desc())

        print("\nAggregation results:")
        result.show()

        # Perform a join that will also be affected by skew
        print("\nCreating second skewed dataset for join...")
        df2 = spark.range(0, 10000).withColumn(
            "skewed_key",
            when(col("id") % 100 < 80, "A")
            .otherwise("OTHER")
        ).withColumn("join_value", rand() * 50)

        print("\nPerforming skewed join...")
        joined = df.join(df2, "skewed_key", "inner")

        join_count = joined.count()
        print(f"Join produced {join_count:,} rows")

        # Show sample of join results
        print("\nSample join results:")
        joined.select("skewed_key", "value1", "join_value").show(10)

        spark.stop()
        print("\nJob completed (with data skew issues)!")
        print("This job demonstrates:")
        print("  - Heavily skewed data distribution (80% in one key)")
        print("  - Uneven partition sizes during aggregation")
        print("  - Skewed join operations")
        print("  - Would benefit from skew optimization techniques")
        sys.exit(0)

    except Exception as e:
        print(f"Job failed with error: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
