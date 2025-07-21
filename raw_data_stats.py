#!/usr/bin/env python3
"""
Raw Data Statistics for MIND Dataset
====================================

This script analyzes the raw TSV files from the MIND dataset to provide
statistics before any processing is applied.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def collect_raw_data_statistics(dataset_path: Path) -> dict:
    """Collect statistics from raw TSV files before processing."""
    logger.info(f"Collecting raw data statistics from: {dataset_path}")
    
    raw_stats = {
        "dataset_path": str(dataset_path),
        "raw_train_behaviors_count": 0,
        "raw_test_behaviors_count": 0,  # Note: original 'valid' becomes our 'test'
        "raw_train_unique_users": 0,
        "raw_test_unique_users": 0,  # Note: original 'valid' becomes our 'test'
        "raw_total_unique_users": 0,
        "raw_train_news_count": 0,
        "raw_test_news_count": 0,  # Note: original 'valid' becomes our 'test'
        "raw_total_news_count": 0,
        "raw_user_id_ranges": {},
        "raw_user_overlaps": {},
        "raw_validation_note": "Validation split is artificially created from train data",
    }

    try:
        # Analyze train behaviors (this is the original train data)
        train_behaviors_path = dataset_path / "train" / "behaviors.tsv"
        if train_behaviors_path.exists():
            logger.info(f"Reading train behaviors from: {train_behaviors_path}")
            train_df = pd.read_csv(
                train_behaviors_path,
                sep="\t",
                header=None,
                names=["impression_id", "user_id", "time", "history", "impressions"],
            )

            raw_stats["raw_train_behaviors_count"] = len(train_df)
            raw_stats["raw_train_unique_users"] = len(train_df["user_id"].unique())

            # Convert user IDs to integers like processing does
            train_user_ids_int = [
                int(str(uid).split("U")[1]) for uid in train_df["user_id"].unique()
            ]
            raw_stats["raw_user_id_ranges"]["train"] = {
                "min": min(train_user_ids_int),
                "max": max(train_user_ids_int),
                "density": len(train_user_ids_int)
                / (max(train_user_ids_int) - min(train_user_ids_int) + 1),
            }

            logger.info(
                f"Raw train: {raw_stats['raw_train_behaviors_count']:,} behaviors, {raw_stats['raw_train_unique_users']:,} unique users"
            )
        else:
            logger.warning(f"Train behaviors file not found: {train_behaviors_path}")

        # Analyze test behaviors (this is the original 'valid' folder)
        test_behaviors_path = dataset_path / "valid" / "behaviors.tsv"
        if test_behaviors_path.exists():
            logger.info(f"Reading test behaviors from: {test_behaviors_path}")
            test_df = pd.read_csv(
                test_behaviors_path,
                sep="\t",
                header=None,
                names=["impression_id", "user_id", "time", "history", "impressions"],
            )

            raw_stats["raw_test_behaviors_count"] = len(test_df)
            raw_stats["raw_test_unique_users"] = len(test_df["user_id"].unique())

            # Convert user IDs to integers
            test_user_ids_int = [int(str(uid).split("U")[1]) for uid in test_df["user_id"].unique()]
            raw_stats["raw_user_id_ranges"]["test"] = {
                "min": min(test_user_ids_int),
                "max": max(test_user_ids_int),
                "density": len(test_user_ids_int)
                / (max(test_user_ids_int) - min(test_user_ids_int) + 1),
            }

            logger.info(
                f"Raw test (original 'valid'): {raw_stats['raw_test_behaviors_count']:,} behaviors, {raw_stats['raw_test_unique_users']:,} unique users"
            )
        else:
            logger.warning(f"Test behaviors file not found: {test_behaviors_path}")

        # Analyze news data
        train_news_path = dataset_path / "train" / "news.tsv"
        test_news_path = dataset_path / "valid" / "news.tsv"  # Note: original 'valid' becomes our 'test'

        if train_news_path.exists():
            logger.info(f"Reading train news from: {train_news_path}")
            train_news_df = pd.read_csv(
                train_news_path,
                sep="\t",
                header=None,
                names=[
                    "id",
                    "category",
                    "subcategory",
                    "title",
                    "abstract",
                    "url",
                    "title_entities",
                    "abstract_entities",
                ],
            )
            raw_stats["raw_train_news_count"] = len(train_news_df)
            logger.info(f"Raw train news articles: {raw_stats['raw_train_news_count']:,}")
        else:
            logger.warning(f"Train news file not found: {train_news_path}")

        if test_news_path.exists():
            logger.info(f"Reading test news from: {test_news_path}")
            test_news_df = pd.read_csv(
                test_news_path,
                sep="\t",
                header=None,
                names=[
                    "id",
                    "category",
                    "subcategory",
                    "title",
                    "abstract",
                    "url",
                    "title_entities",
                    "abstract_entities",
                ],
            )
            raw_stats["raw_test_news_count"] = len(test_news_df)
            logger.info(f"Raw test news articles: {raw_stats['raw_test_news_count']:,}")
        else:
            logger.warning(f"Test news file not found: {test_news_path}")

        raw_stats["raw_total_news_count"] = (
            raw_stats["raw_train_news_count"] + raw_stats["raw_test_news_count"]
        )

        # Calculate overlaps and total unique users
        if train_behaviors_path.exists() and test_behaviors_path.exists():
            train_unique = set(train_df["user_id"].unique())
            test_unique = set(test_df["user_id"].unique())
            overlap = train_unique.intersection(test_unique)

            raw_stats["raw_user_overlaps"]["train_test_overlap"] = len(overlap)
            raw_stats["raw_total_unique_users"] = len(train_unique.union(test_unique))

            logger.info(f"Raw user overlap (train-test): {len(overlap):,}")
            logger.info(f"Raw total unique users: {raw_stats['raw_total_unique_users']:,}")

        logger.info("Raw data statistics collection completed")
        logger.info(
            "NOTE: Validation split is artificially created from train data during processing"
        )

    except Exception as e:
        logger.error(f"Error collecting raw data statistics: {e}")
        raise

    return raw_stats


def log_raw_data_statistics(raw_stats: dict) -> None:
    """Log raw data statistics to console."""
    logger.info("=" * 60)
    logger.info("RAW DATA STATISTICS (Before Processing)")
    logger.info("=" * 60)
    logger.info(f"Dataset path: {raw_stats.get('dataset_path', 'N/A')}")
    logger.info("")
    
    # Behavior statistics
    logger.info("BEHAVIOR STATISTICS:")
    logger.info(f"  Train behaviors: {raw_stats.get('raw_train_behaviors_count', 'N/A'):,}")
    logger.info(
        f"  Test behaviors (original 'valid'): {raw_stats.get('raw_test_behaviors_count', 'N/A'):,}"
    )
    logger.info(f"  Total behaviors: {raw_stats.get('raw_train_behaviors_count', 0) + raw_stats.get('raw_test_behaviors_count', 0):,}")
    logger.info("")
    
    # User statistics
    logger.info("USER STATISTICS:")
    logger.info(f"  Train unique users: {raw_stats.get('raw_train_unique_users', 'N/A'):,}")
    logger.info(
        f"  Test unique users (original 'valid'): {raw_stats.get('raw_test_unique_users', 'N/A'):,}"
    )
    logger.info(f"  Total unique users: {raw_stats.get('raw_total_unique_users', 'N/A'):,}")
    logger.info(
        f"  User overlap (train-test): {raw_stats.get('raw_user_overlaps', {}).get('train_test_overlap', 'N/A'):,}"
    )
    logger.info("")
    
    # News statistics
    logger.info("NEWS STATISTICS:")
    logger.info(f"  Train news articles: {raw_stats.get('raw_train_news_count', 'N/A'):,}")
    logger.info(
        f"  Test news articles (original 'valid'): {raw_stats.get('raw_test_news_count', 'N/A'):,}"
    )
    logger.info(f"  Total news articles: {raw_stats.get('raw_total_news_count', 'N/A'):,}")
    logger.info("")
    
    # User ID ranges
    logger.info("USER ID RANGES:")
    for split, range_info in raw_stats.get("raw_user_id_ranges", {}).items():
        split_name = "test (original 'valid')" if split == "test" else split
        logger.info(
            f"  {split_name.capitalize()}: {range_info['min']} to {range_info['max']} (density: {range_info['density']:.2%})"
        )
    logger.info("")
    
    # Notes
    logger.info("NOTES:")
    logger.info(
        f"  {raw_stats.get('raw_validation_note', 'Validation split is artificially created from train data')}"
    )
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate raw data statistics for MIND dataset")
    parser.add_argument(
        "dataset_path", 
        type=str, 
        default="data/mind/small/",
        help="Path to the MIND dataset directory"
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        return 1
    
    try:
        # Collect raw data statistics
        raw_stats = collect_raw_data_statistics(dataset_path)
        
        # Log statistics to console
        log_raw_data_statistics(raw_stats)
        
        # Save to file if requested
        if args.output:
            import json
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(raw_stats, f, indent=2)
            logger.info(f"Statistics saved to: {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to generate statistics: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 