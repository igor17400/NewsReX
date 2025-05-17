import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from nltk.tokenize import TreebankWordTokenizer, TreebankWordDetokenizer
import tensorflow as tf

logger = logging.getLogger(__name__)


def display_statistics(data_dict: Dict[str, Dict], mode: str = "train") -> None:
    """Display statistics about the dataset."""
    logger.info("Displaying dataset statistics...")

    if mode == "train":
        num_news = len(data_dict["news"]["news_ids"])
        num_train_behaviors = len(data_dict["train_behaviors"]["histories_news_ids"])
        num_val_behaviors = len(data_dict["val_behaviors"]["histories_news_ids"])

        logger.info(f"Number of news articles: {num_news}")
        logger.info(f"Number of training behaviors: {num_train_behaviors}")
        logger.info(f"Number of validation behaviors: {num_val_behaviors}")

        # Additional statistics using NumPy
        avg_history_length = np.mean(
            [len(history) for history in data_dict["train_behaviors"]["history_news_tokens"]]
        )
        avg_impressions_length = np.mean(
            [len(impression) for impression in data_dict["train_behaviors"]["candidate_news_tokens"]]
        )
        avg_history_length_val = np.mean(
            [len(history) for history in data_dict["val_behaviors"]["history_news_tokens"]]
        )
        avg_impressions_length_val = np.mean(
            [len(impression) for impression in data_dict["val_behaviors"]["candidate_news_tokens"]]
        )

        logger.info(f"Average history length: {avg_history_length:.2f}")
        logger.info(f"Average impressions length: {avg_impressions_length:.2f}")
        logger.info(f"Average history length (validation): {avg_history_length_val:.2f}")
        logger.info(f"Average impressions length (validation): {avg_impressions_length_val:.2f}")
    else:
        num_test_news = len(data_dict["news"]["news_ids"])
        num_test_behaviors = len(data_dict["test_behaviors"]["histories_news_ids"])

        logger.info(f"Number of news articles: {num_test_news}")
        logger.info(f"Number of test behaviors: {num_test_behaviors}")

        avg_history_length_test = np.mean(
            [len(history) for history in data_dict["test_behaviors"]["history_news_tokens"]]
        )
        avg_impressions_length_test = np.mean(
            [len(impression) for impression in data_dict["test_behaviors"]["candidate_news_tokens"]]
        )

        logger.info(f"Average history length (test): {avg_history_length_test:.2f}")
        logger.info(f"Average impressions length (test): {avg_impressions_length_test:.2f}")


def apply_data_fraction(data_dict: Dict[str, np.ndarray], data_fraction: float) -> None:
    """Reduce the dataset size based on the data_fraction parameter."""
    for key in data_dict:
        data_dict[key] = data_dict[key][: int(len(data_dict[key]) * data_fraction)]

    return data_dict


def apply_data_fraction(data_dict: Dict[str, np.ndarray], fraction: float) -> Dict[str, np.ndarray]:
    """Reduce the dataset size based on the fraction parameter."""
    if fraction < 1.0:
        logger.info(f"Using {fraction * 100:.0f}% of the dataset")
        return {k: v[: int(len(v) * fraction)] for k, v in data_dict.items()}
    return data_dict
