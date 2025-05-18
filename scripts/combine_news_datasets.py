#!/usr/bin/env python3

import pandas as pd
import argparse
from pathlib import Path


def combine_news_datasets(train_path, valid_path, output_path):
    """
    Combine news datasets from train and validation sets, removing any overlapping IDs.

    Args:
        train_path (str): Path to training news dataset
        valid_path (str): Path to validation news dataset
        output_path (str): Path to save the combined dataset
    """
    # Read the datasets
    df_news_train = pd.read_table(
        train_path,
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

    df_news_valid = pd.read_table(
        valid_path,
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

    # Get the IDs that are only in validation set (not in training)
    valid_only_ids = set(df_news_valid["id"]) - set(df_news_train["id"])

    # Filter validation dataframe to keep only non-overlapping IDs
    df_news_valid_unique = df_news_valid[df_news_valid["id"].isin(valid_only_ids)]

    # Combine the dataframes
    df_news_combined = pd.concat([df_news_train, df_news_valid_unique], ignore_index=True)

    # Print information about the result
    print("Original training set size:", len(df_news_train))
    print("Original validation set size:", len(df_news_valid))
    print("Final combined set size:", len(df_news_combined))
    print("Number of overlapping IDs removed:", len(df_news_valid) - len(df_news_valid_unique))

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the combined dataset
    df_news_combined.to_csv(output_path, sep="\t", index=False, header=False)
    print(f"Combined dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combine news datasets while removing overlapping IDs"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        help="Path to training news dataset",
        default="./data/mind/small/train/news.tsv",
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        help="Path to validation news dataset",
        default="./data/mind/small/valid/news.tsv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the combined dataset",
        default="./data/mind/small/scripts_output/non_overlapping_news.tsv",
    )

    args = parser.parse_args()

    combine_news_datasets(args.train_path, args.valid_path, args.output_path)


if __name__ == "__main__":
    main()
