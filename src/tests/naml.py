#!/usr/bin/env python3
"""Test script for NAML model implementation."""

import os
import sys
from pathlib import Path

# Set JAX as Keras backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import keras
from src.models.naml import NAML

def test_naml_model():
    """Test NAML model instantiation and basic functionality."""
    
    # Mock processed news data
    processed_news = {
        "vocab_size": 10000,
        "num_categories": 20,
        "num_subcategories": 50,
        "embeddings": np.random.randn(10000, 300).astype(np.float32)
    }
    
    # Model parameters
    model_params = {
        "processed_news": processed_news,
        "max_title_length": 30,
        "max_abstract_length": 100,
        "embedding_size": 300,
        "category_embedding_dim": 100,
        "subcategory_embedding_dim": 100,
        "cnn_filter_num": 400,
        "cnn_kernel_size": 3,
        "word_attention_query_dim": 200,
        "view_attention_query_dim": 200,
        "user_attention_query_dim": 200,
        "dropout_rate": 0.2,
        "activation": "relu",
        "max_history_length": 50,
        "max_impressions_length": 5,
        "seed": 42
    }
    
    try:
        # Create NAML model
        print("Creating NAML model...")
        naml_model = NAML(**model_params)
        
        # Test model components
        print("✓ NAML model created successfully")
        print(f"✓ Training model: {naml_model.training_model.name}")
        print(f"✓ Scoring model: {naml_model.scorer_model.name}")
        
        # Test with dummy data
        batch_size = 2
        hist_shape = (batch_size, 50, 132)  # max_history_length, title + abstract + 2
        cand_shape = (batch_size, 5, 132)   # max_impressions_length, title + abstract + 2
        
        # Create dummy inputs
        hist_input = np.random.randint(0, 1000, hist_shape, dtype=np.int32)
        cand_input = np.random.randint(0, 1000, cand_shape, dtype=np.int32)
        
        # Test training model
        print("\nTesting training model...")
        training_output = naml_model.training_model([hist_input, cand_input])
        print(f"✓ Training model output shape: {training_output.shape}")
        
        # Test scoring model
        print("\nTesting scoring model...")
        user_hist = np.random.randint(0, 1000, (batch_size, 50, 132), dtype=np.int32)
        news_input = np.random.randint(0, 1000, (batch_size, 132), dtype=np.int32)
        
        scoring_output = naml_model.scorer_model([user_hist, news_input])
        print(f"✓ Scoring model output shape: {scoring_output.shape}")
        
        # Test news encoder
        print("\nTesting news encoder...")
        news_input_single = np.random.randint(0, 1000, (batch_size, 132), dtype=np.int32)
        news_output = naml_model.newsencoder(news_input_single)
        print(f"✓ News encoder output shape: {news_output.shape}")
        
        # Test user encoder
        print("\nTesting user encoder...")
        user_output = naml_model.userencoder(hist_input)
        print(f"✓ User encoder output shape: {user_output.shape}")
        
        print("\n🎉 All tests passed! NAML model is working correctly.")
        
        # Print model summary
        print("\n" + "="*50)
        print("NAML MODEL SUMMARY")
        print("="*50)
        naml_model.training_model.summary()
        
    except Exception as e:
        print(f"❌ Error testing NAML model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    keras.utils.set_random_seed(42)
    
    print("Testing NAML Model Implementation")
    print(f"Backend: {keras.backend.backend()}")
    print("="*40)
    
    success = test_naml_model()
    
    if success:
        print("\n✅ NAML model implementation is ready for training!")
    else:
        print("\n❌ NAML model implementation needs fixes.") 