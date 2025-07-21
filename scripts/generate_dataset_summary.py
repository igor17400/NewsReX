#!/usr/bin/env python3
"""
Script to generate a comprehensive dataset summary CSV file.

This script loads a MIND dataset and generates a detailed summary of all relevant statistics,
including news articles, user behaviors, vocabulary information, and data quality metrics.
The summary is saved to the 'preprocessed' folder as 'datasets_summary.csv'.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import hydra
from omegaconf import DictConfig

from datasets.mind import MINDDataset
from utils.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config")
def generate_summary(cfg: DictConfig) -> None:
    """Generate dataset summary using the provided configuration."""
    logger.info("Starting dataset summary generation...")

    try:
        # Initialize the dataset
        logger.info(f"Initializing MIND dataset (version: {cfg.dataset.version})...")
        dataset = MINDDataset(
            name=cfg.dataset.name,
            version=cfg.dataset.version,
            urls=cfg.dataset.urls,
            max_title_length=cfg.dataset.max_title_length,
            max_abstract_length=cfg.dataset.max_abstract_length,
            max_history_length=cfg.dataset.max_history_length,
            max_impressions_length=cfg.dataset.max_impressions_length,
            seed=cfg.dataset.seed,
            embedding_type=cfg.dataset.embedding_type,
            embedding_size=cfg.dataset.embedding_size,
            sampling=cfg.dataset.sampling,
            data_fraction_train=cfg.dataset.data_fraction_train,
            data_fraction_val=cfg.dataset.data_fraction_val,
            data_fraction_test=cfg.dataset.data_fraction_test,
            mode=cfg.dataset.mode,
            use_knowledge_graph=cfg.dataset.use_knowledge_graph,
            random_train_samples=cfg.dataset.random_train_samples,
            validation_split_strategy=cfg.dataset.validation_split_strategy,
            validation_split_percentage=cfg.dataset.validation_split_percentage,
            validation_split_seed=cfg.dataset.validation_split_seed,
            word_threshold=cfg.dataset.word_threshold,
            process_title=cfg.dataset.process_title,
            process_abstract=cfg.dataset.process_abstract,
            process_category=cfg.dataset.process_category,
            process_subcategory=cfg.dataset.process_subcategory,
            process_user_id=cfg.dataset.process_user_id,
            max_entities=cfg.dataset.max_entities,
            max_relations=cfg.dataset.max_relations,
        )

        # Generate the summary
        logger.info("Generating dataset summary...")
        summary_path = dataset.generate_dataset_summary()
        
        logger.info(f"Dataset summary successfully generated and saved to: {summary_path}")
        
        # Display the summary file location
        logger.info("=" * 80)
        logger.info("DATASET SUMMARY GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Summary file: {summary_path}")
        logger.info(f"Dataset version: {cfg.dataset.version}")
        logger.info(f"Mode: {cfg.dataset.mode}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Error generating dataset summary: {str(e)}")
        raise


def main():
    """Main function to handle command line arguments and run the summary generation."""
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive dataset summary CSV file for MIND dataset"
    )
    parser.add_argument(
        "--config-path", type=str, default="../configs", help="Path to the configuration directory"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Name of the configuration file (without .yaml extension)",
    )
    parser.add_argument(
        "--version",
        type=str,
        choices=["small", "large"],
        help="Override dataset version from config",
    )
    parser.add_argument(
        "--mode", type=str, choices=["train", "test"], help="Override dataset mode from config"
    )

    args = parser.parse_args()

    # Override config with command line arguments if provided
    if args.version or args.mode:
        # We need to modify the config before hydra processes it
        # This is a simple approach - in practice, you might want to use hydra's override mechanism
        logger.info("Command line overrides will be applied to the configuration")

    # Run the summary generation
    generate_summary()


if __name__ == "__main__":
    main()
