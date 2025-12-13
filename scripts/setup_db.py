#!/usr/bin/env python3
"""Database setup script for Spark Optimizer.

This script initializes the database schema and creates all necessary tables.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add src to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spark_optimizer.storage.database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_database_url(custom_url: str = None) -> str:
    """Get database connection URL.

    Args:
        custom_url: Custom database URL provided by user

    Returns:
        Database connection string
    """
    if custom_url:
        return custom_url

    # Check environment variable
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        logger.info("Using DATABASE_URL from environment")
        return env_url

    # Default to SQLite in project root
    project_root = Path(__file__).parent.parent
    default_db_path = project_root / "spark_optimizer.db"
    logger.info(f"Using default SQLite database at {default_db_path}")
    return f"sqlite:///{default_db_path}"


def setup_database(database_url: str, drop_existing: bool = False) -> None:
    """Set up the database schema.

    Args:
        database_url: Database connection string
        drop_existing: Whether to drop existing tables first
    """
    try:
        logger.info("Connecting to database...")
        db = Database(database_url)

        if drop_existing:
            logger.warning("Dropping existing tables...")
            db.drop_tables()
            logger.info("Existing tables dropped")

        logger.info("Creating database tables...")
        db.create_tables()
        logger.info("Database setup completed successfully!")

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise


def main():
    """Main entry point for the setup script."""
    parser = argparse.ArgumentParser(
        description="Initialize Spark Optimizer database schema"
    )
    parser.add_argument(
        "--database-url",
        help="Database connection string (default: sqlite:///spark_optimizer.db)"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing tables before creating new ones (WARNING: deletes all data)"
    )

    args = parser.parse_args()

    # Confirm if dropping existing tables
    if args.drop_existing:
        response = input(
            "WARNING: This will delete all existing data. Are you sure? (yes/no): "
        )
        if response.lower() != "yes":
            logger.info("Aborted by user")
            return

    database_url = get_database_url(args.database_url)
    setup_database(database_url, args.drop_existing)


if __name__ == "__main__":
    main()
