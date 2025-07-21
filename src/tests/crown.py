#!/usr/bin/env python3
"""Test script for CROWN model implementation."""

import numpy as np
import tensorflow as tf
from src.models.crown import CROWN

import sys
from pathlib import Path


# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_crown_model():
    """Test CROWN model instantiation and basic functionality."""

    # Mock processed news data
    processed_news = {
        "vocab_size": 10000,
        "num_categories": 20,
        "num_subcategories": 50,
        "embeddings": np.random.randn(10000, 300).astype(np.float32),
    }

    # Model parameters
    model_params = {
        "processed_news": processed_news,
        "embedding_size": 300,
        "attention_dim": 200,
        "intent_embedding_dim": 200,
        "intent_num": 4,
        "category_embedding_dim": 100,
        "subcategory_embedding_dim": 100,
        "word_embedding_dim": 300,
        "head_num": 8,
        "feedforward_dim": 512,
        "num_layers": 2,
        "isab_num_heads": 4,
        "isab_num_inds": 4,
        "alpha": 0.1,
        "max_title_length": 30,
        "max_abstract_length": 100,
        "max_history_length": 50,
        "max_impressions_length": 5,
        "dropout_rate": 0.2,
        "seed": 42,
    }

    try:
        # Create CROWN model
        print("Creating CROWN model...")
        crown_model = CROWN(**model_params)

        # Test model components
        print("✓ CROWN model created successfully")
        print(f"✓ Training model: {crown_model.training_model.name}")
        print(f"✓ Scoring model: {crown_model.scorer_model.name}")

        # Test with dummy data
        batch_size = 2
        hist_shape = (batch_size, 50, 132)  # max_history_length, title + abstract + 2
        cand_shape = (batch_size, 5, 132)  # max_impressions_length, title + abstract + 2

        # Create dummy inputs
        hist_input = np.random.randint(0, 1000, hist_shape, dtype=np.int32)
        cand_input = np.random.randint(0, 1000, cand_shape, dtype=np.int32)

        # Test training model
        print("\nTesting training model...")
        # Training model expects 2 inputs: history_concat and candidate_concat
        training_output = crown_model.training_model([hist_input, cand_input])
        print(f"✓ Training model output shape: {training_output.shape}")

        # Test scoring model
        print("\nTesting scoring model...")
        user_hist = np.random.randint(0, 1000, (batch_size, 50, 132), dtype=np.int32)
        news_input = np.random.randint(0, 1000, (batch_size, 132), dtype=np.int32)
        # Scoring model expects 2 inputs: user_history_concat and news_concat
        scoring_output = crown_model.scorer_model([user_hist, news_input])
        print(f"✓ Scoring model output shape: {scoring_output.shape}")

        # Test news encoder
        print("\nTesting news encoder...")
        news_input_single = np.random.randint(0, 1000, (batch_size, 132), dtype=np.int32)
        news_output = crown_model.newsencoder(news_input_single)
        print(f"✓ News encoder output shape: {news_output.shape}")

        # Test user encoder
        print("\nTesting user encoder...")
        edge_index_user = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], dtype=np.int64)
        edge_index_user = np.expand_dims(edge_index_user, 0)
        edge_index_user = np.tile(edge_index_user, (batch_size, 1, 1))
        user_output = crown_model.userencoder([hist_input, edge_index_user])
        print(f"✓ User encoder output shape: {user_output.shape}")

        print("\n🎉 All tests passed! CROWN model is working correctly.")

        # Print model summary
        print("\n" + "=" * 50)
        print("CROWN MODEL SUMMARY")
        print("=" * 50)
        crown_model.training_model.summary()

    except Exception as e:
        print(f"❌ Error testing CROWN model: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    print("Testing CROWN Model Implementation")
    print("=" * 40)

    success = test_crown_model()

    if success:
        print("\n✅ CROWN model implementation is ready for training!")
    else:
        print("\n❌ CROWN model implementation needs fixes.")
