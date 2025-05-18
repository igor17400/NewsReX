#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def combine_news_with_dates(news_path, dates_path, output_path):
    """
    Combine non-overlapping news dataset with publication dates.
    
    Args:
        news_path (str): Path to non-overlapping news dataset
        dates_path (str): Path to publication dates CSV
        output_path (str): Path to save the combined dataset
    """
    logger.info(f"Reading news dataset from: {news_path}")
    # Read the news dataset without column names to preserve exact format
    df_news = pd.read_table(
        news_path,
        header=None,
    )
    
    # Assign column names for internal processing
    df_news.columns = [
        "id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entities",
        "abstract_entities",
    ]
    
    logger.info(f"Reading publication dates from: {dates_path}")
    df_dates = pd.read_csv(dates_path)
    
    # Merge the datasets on news_id
    df_combined = pd.merge(
        df_news,
        df_dates,
        left_on="id",
        right_on="news_id",
        how="left"
    )
    
    # Drop the redundant news_id column
    df_combined = df_combined.drop(columns=["news_id"])
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the combined dataset without headers and with tab separator
    df_combined.to_csv(
        output_path,
        sep="\t",
        index=False,
        header=False,
        na_rep=""
    )
    logger.info(f"Combined dataset saved to: {output_path}")
    
    # Print some statistics
    logger.info(f"Total news articles: {len(df_combined)}")
    logger.info(f"Articles with publication dates: {df_combined['publication_date'].notna().sum()}")
    logger.info(f"Articles without publication dates: {df_combined['publication_date'].isna().sum()}")
    
    if df_combined['publication_date'].notna().any():
        min_date = pd.to_datetime(df_combined['publication_date'].min())
        max_date = pd.to_datetime(df_combined['publication_date'].max())
        logger.info(f"Publication date range: {min_date} to {max_date}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine non-overlapping news dataset with publication dates"
    )
    parser.add_argument(
        "--news_path",
        type=str,
        help="Path to non-overlapping news dataset",
        default="./data/mind/small/scripts_output/non_overlapping_news.tsv",
    )
    parser.add_argument(
        "--dates_path",
        type=str,
        help="Path to publication dates CSV",
        default="./data/mind/small/scripts_output/news_publication_dates.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the combined dataset",
        default="./data/mind/small/scripts_output/news_with_dates.tsv",
    )
    
    args = parser.parse_args()
    
    combine_news_with_dates(args.news_path, args.dates_path, args.output_path)

if __name__ == "__main__":
    main() 