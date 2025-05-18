#!/usr/bin/env python3

import pandas as pd
import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object."""
    try:
        return datetime.strptime(timestamp_str, "%m/%d/%Y %I:%M:%S %p")
    except ValueError as e:
        logger.warning(f"Invalid timestamp format: {timestamp_str}. Error: {e}")
        return None


def extract_news_ids(history_str, impressions_str):
    """Extract news IDs from history and impressions strings."""
    news_ids = set()

    # Extract from history
    if history_str and history_str.strip():
        news_ids.update(history_str.split())

    # Extract from impressions
    if impressions_str and impressions_str.strip():
        for item in impressions_str.split():
            if "-" in item:
                news_id, _ = item.split("-")
                news_ids.add(news_id)

    return news_ids


def get_timestamp_range(timestamps_dict):
    """Get the earliest and latest timestamps from a dictionary of timestamps."""
    all_timestamps = []
    for timestamps in timestamps_dict.values():
        all_timestamps.extend(timestamps)
    if all_timestamps:
        return min(all_timestamps), max(all_timestamps)
    return None, None


def process_behaviors_file(file_path):
    """
    Process a single behaviors file and return news timestamps.

    Args:
        file_path (str): Path to behaviors.tsv file

    Returns:
        tuple: (news_timestamps dict, total_rows, processed_rows)
    """
    news_timestamps = defaultdict(list)
    total_rows = 0
    processed_rows = 0

    logger.info(f"Reading behaviors file from: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            total_rows += 1
            try:
                # Split line into components
                impression_id, user_id, timestamp_str, history, impressions = line.strip().split(
                    "\t"
                )

                # Parse timestamp
                timestamp = parse_timestamp(timestamp_str)
                if timestamp is None:
                    continue

                # Extract news IDs
                news_ids = extract_news_ids(history, impressions)

                # Add timestamps for each news ID
                for news_id in news_ids:
                    news_timestamps[news_id].append(timestamp)

                processed_rows += 1

            except ValueError as e:
                logger.warning(f"Error processing line {total_rows} in {file_path}: {e}")
                continue

    # Get timestamp range for this dataset
    first_ts, last_ts = get_timestamp_range(news_timestamps)
    if first_ts and last_ts:
        logger.info(f"Dataset timestamp range: {first_ts} to {last_ts}")

    logger.info(f"Processed {processed_rows} out of {total_rows} rows from {file_path}")
    return news_timestamps, total_rows, processed_rows


def infer_news_dates(train_path, valid_path):
    """
    Infer publication dates for news articles from both train and validation behaviors data.

    Args:
        train_path (str): Path to training behaviors.tsv file
        valid_path (str): Path to validation behaviors.tsv file

    Returns:
        dict: Dictionary mapping news_ids to their earliest appearance timestamp
    """
    # Process both files
    train_timestamps, train_total, train_processed = process_behaviors_file(train_path)
    valid_timestamps, valid_total, valid_processed = process_behaviors_file(valid_path)

    # Merge timestamps from both files
    all_timestamps = defaultdict(list)
    for news_id, timestamps in train_timestamps.items():
        all_timestamps[news_id].extend(timestamps)
    for news_id, timestamps in valid_timestamps.items():
        all_timestamps[news_id].extend(timestamps)

    # Get timestamp range for merged dataset
    first_ts, last_ts = get_timestamp_range(all_timestamps)
    if first_ts and last_ts:
        logger.info(f"Merged dataset timestamp range: {first_ts} to {last_ts}")

    # Calculate earliest timestamp for each news ID
    news_pub_dates = {news_id: min(timestamps) for news_id, timestamps in all_timestamps.items()}

    logger.info(f"Total rows processed: {train_processed + valid_processed}")
    logger.info(f"Inferred dates for {len(news_pub_dates)} unique news articles")
    return news_pub_dates


def save_dates_to_csv(news_pub_dates, output_path):
    """Save inferred dates to CSV file."""
    df = pd.DataFrame(
        [{"news_id": k, "publication_date": v.isoformat()} for k, v in news_pub_dates.items()]
    )

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Saved publication dates to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Infer news publication dates from MIND dataset behaviors"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to training behaviors.tsv file",
        default="./data/mind/small/train/behaviors.tsv",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        help="Path to validation behaviors.tsv file",
        default="./data/mind/small/valid/behaviors.tsv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the output CSV file",
        default="./data/mind/small/scripts_output/news_publication_dates.csv",
    )

    args = parser.parse_args()

    # Infer publication dates from both train and validation data
    news_pub_dates = infer_news_dates(args.train_path, args.valid_path)

    # Save results
    save_dates_to_csv(news_pub_dates, args.output_path)

    # Print some statistics
    if news_pub_dates:
        dates = list(news_pub_dates.values())
        logger.info(f"Final publication date range: {min(dates)} to {max(dates)}")


if __name__ == "__main__":
    main()
