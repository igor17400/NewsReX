#!/usr/bin/env python3
"""Test script for LSTUR model implementation."""

import sys
from pathlib import Path

# Add the project root to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import tensorflow as tf
from src.models.lstur import LSTUR

def test_lstur_model():
    """Test LSTUR model instantiation and basic functionality."""
    
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
        "embedding_size": 300,
        "cnn_filter_num": 400,
        "cnn_kernel_size": 3,
        "attention_hidden_dim": 200,
        "dropout_rate": 0.2,
        "cnn_activation": "relu",
        "max_title_length": 30,
        "max_history_length": 50,
        "max_impressions_length": 5,
        "num_users": 100000,
        "user_representation_type": "lstm",
        "user_combination_type": "ini",
        "process_user_id": True,
        "category_embedding_dim": 100,
        "subcategory_embedding_dim": 100,
        "use_cat_subcat_encoder": True,
        "seed": 42
    }
    
    try:
        # Create LSTUR model
        print("Creating LSTUR model...")
        lstur_model = LSTUR(**model_params)
        
        # Test model components
        print("✓ LSTUR model created successfully")
        print(f"✓ Training model: {lstur_model.training_model.name}")
        print(f"✓ Scoring model: {lstur_model.scorer_model.name}")
        
        # Test with dummy data
        batch_size = 2
        hist_shape = (batch_size, 50, 32)  # max_history_length, title + category + subcategory
        cand_shape = (batch_size, 5, 32)   # max_impressions_length, title + category + subcategory
        user_ids_shape = (batch_size, 1)   # user IDs
        
        # Create dummy inputs
        hist_input = np.random.randint(0, 1000, hist_shape, dtype=np.int32)
        cand_input = np.random.randint(0, 1000, cand_shape, dtype=np.int32)
        user_ids = np.random.randint(0, 100000, user_ids_shape, dtype=np.int32)
        
        # Test training model
        print("\nTesting training model...")
        training_output = lstur_model.training_model([hist_input, user_ids, cand_input])
        print(f"✓ Training model output shape: {training_output.shape}")
        
        # Test scoring model
        print("\nTesting scoring model...")
        user_hist = np.random.randint(0, 1000, (batch_size, 50, 32), dtype=np.int32)
        news_input = np.random.randint(0, 1000, (batch_size, 32), dtype=np.int32)
        user_ids_score = np.random.randint(0, 100000, (batch_size, 1), dtype=np.int32)
        
        scoring_output = lstur_model.scorer_model([user_hist, user_ids_score, news_input])
        print(f"✓ Scoring model output shape: {scoring_output.shape}")
        
        # Test news encoder
        print("\nTesting news encoder...")
        news_input_single = np.random.randint(0, 1000, (batch_size, 32), dtype=np.int32)
        news_output = lstur_model.newsencoder(news_input_single)
        print(f"✓ News encoder output shape: {news_output.shape}")
        
        # Test user encoder
        print("\nTesting user encoder...")
        user_output = lstur_model.userencoder([hist_input, user_ids])
        print(f"✓ User encoder output shape: {user_output.shape}")
        
        print("\n🎉 All tests passed! LSTUR model is working correctly.")
        
        # Print model summary
        print("\n" + "="*50)
        print("LSTUR MODEL SUMMARY")
        print("="*50)
        lstur_model.training_model.summary()
        
    except Exception as e:
        print(f"❌ Error testing LSTUR model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("Testing LSTUR Model Implementation")
    print("="*40)
    
    success = test_lstur_model()
    
    if success:
        print("\n✅ LSTUR model implementation is ready for training!")
    else:
        print("\n❌ LSTUR model implementation needs fixes.") 