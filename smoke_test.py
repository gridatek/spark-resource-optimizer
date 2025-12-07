#!/usr/bin/env python
"""
Quick smoke test to verify the Spark Resource Optimizer is working correctly.
Run this to test your local setup.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("1. Testing imports...")
    try:
        from spark_optimizer.storage.database import Database
        from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender
        from spark_optimizer.collectors.event_log_collector import EventLogCollector
        from spark_optimizer.cli.commands import cli
        print("   ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"   ✗ Import failed: {e}")
        return False


def test_database():
    """Test database creation and tables."""
    print("\n2. Testing database...")
    try:
        from spark_optimizer.storage.database import Database
        # Import models to register them with Base
        from spark_optimizer.storage.models import SparkApplication

        db = Database("sqlite:///test_smoke.db")
        db.create_tables()

        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()

        if tables:
            print(f"   ✓ Database created with {len(tables)} tables")
            return True
        else:
            print("   ✗ No tables created")
            return False
    except Exception as e:
        print(f"   ✗ Database test failed: {e}")
        return False
    finally:
        # Cleanup
        import os
        if os.path.exists("test_smoke.db"):
            os.remove("test_smoke.db")


def test_recommender():
    """Test the recommendation engine."""
    print("\n3. Testing recommender...")
    try:
        from spark_optimizer.storage.database import Database
        from spark_optimizer.recommender.similarity_recommender import SimilarityRecommender

        db = Database("sqlite:///test_smoke.db")
        db.create_tables()

        recommender = SimilarityRecommender(db)

        # Get a recommendation
        rec = recommender.recommend(
            input_size_bytes=10 * 1024**3,  # 10GB
            job_type="etl",
            priority="balanced"
        )

        # Verify response structure
        if "configuration" in rec and "confidence" in rec:
            config = rec["configuration"]
            print(f"   ✓ Recommender works (method: {rec['metadata'].get('method', 'unknown')})")
            print(f"     - Executors: {config['num_executors']}")
            print(f"     - Cores: {config['executor_cores']}")
            print(f"     - Memory: {config['executor_memory_mb']} MB")
            print(f"     - Confidence: {rec['confidence']:.0%}")
            return True
        else:
            print("   ✗ Invalid response structure")
            return False
    except Exception as e:
        print(f"   ✗ Recommender test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        import os
        if os.path.exists("test_smoke.db"):
            os.remove("test_smoke.db")


def test_cli():
    """Test CLI is accessible."""
    print("\n4. Testing CLI...")
    try:
        import subprocess
        result = subprocess.run(
            ["spark-optimizer", "--version"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"   ✓ CLI working: {version}")
            return True
        else:
            print(f"   ✗ CLI failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("   ✗ spark-optimizer command not found")
        print("     Try: pip install -e .")
        return False
    except Exception as e:
        print(f"   ✗ CLI test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("Spark Resource Optimizer - Smoke Test")
    print("=" * 60)

    tests = [
        test_imports,
        test_database,
        test_recommender,
        test_cli,
    ]

    results = []
    for test in tests:
        results.append(test())

    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)

    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
        print("\nYour environment is set up correctly!")
        print("\nNext steps:")
        print("  - See TESTING.md for detailed testing instructions")
        print("  - Try: spark-optimizer --help")
        print("  - Try: spark-optimizer serve --port 8080")
        return 0
    else:
        print(f"✗ SOME TESTS FAILED ({passed}/{total} passed)")
        print("\nPlease check the errors above and:")
        print("  1. Make sure you're in the virtual environment")
        print("  2. Run: pip install -e .")
        print("  3. Check TESTING.md for setup instructions")
        return 1


if __name__ == "__main__":
    sys.exit(main())
