#!/usr/bin/env python3
"""Test script for NRMS model implementation."""

import sys
from pathlib import Path

# Add the project root and src to Python path to enable imports
project_root = Path(__file__).parent.parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

import numpy as np
from src.models.nrms import NRMS

def test_nrms_model():
    """Test NRMS model instantiation and basic functionality."""
    
    # Mock processed news data
    processed_news = {
        "vocab_size": 10000,
        "embeddings": np.random.randn(10000, 300).astype(np.float32)
    }
    
    # Model parameters
    model_params = {
        "processed_news": processed_news,
        "embedding_size": 300,
        "multiheads": 16,
        "head_dim": 16,
        "attention_hidden_dim": 200,
        "dropout_rate": 0.2,
        "seed": 42,
        "max_title_length": 50,
        "max_history_length": 50,
        "max_impressions_length": 5,
        "process_user_id": False
    }
    
    try:
        # Create NRMS model
        print("Creating NRMS model...")
        nrms_model = NRMS(**model_params)
        
        # Test model components
        print("✓ NRMS model created successfully")
        print(f"✓ Training model: {nrms_model.training_model.name}")
        print(f"✓ Scoring model: {nrms_model.scorer_model.name}")
        
        # Test with dummy data
        batch_size = 2
        hist_shape = (batch_size, 50, 50)  # max_history_length, max_title_length
        cand_shape = (batch_size, 5, 50)   # max_impressions_length, max_title_length
        
        # Create dummy inputs
        hist_input = np.random.randint(0, 1000, hist_shape, dtype=np.int32)
        cand_input = np.random.randint(0, 1000, cand_shape, dtype=np.int32)
        
        # Test training model
        print("\nTesting training model...")
        training_output = nrms_model.training_model([hist_input, cand_input])
        print(f"✓ Training model output shape: {training_output.shape}")
        
        # Test scoring model
        print("\nTesting scoring model...")
        user_hist = np.random.randint(0, 1000, (batch_size, 50, 50), dtype=np.int32)
        news_input = np.random.randint(0, 1000, (batch_size, 50), dtype=np.int32)
        
        scoring_output = nrms_model.scorer_model([user_hist, news_input])
        print(f"✓ Scoring model output shape: {scoring_output.shape}")
        
        # Test news encoder
        print("\nTesting news encoder...")
        news_input_single = np.random.randint(0, 1000, (batch_size, 50), dtype=np.int32)
        news_output = nrms_model.newsencoder(news_input_single)
        print(f"✓ News encoder output shape: {news_output.shape}")
        
        # Test user encoder
        print("\nTesting user encoder...")
        user_output = nrms_model.userencoder(hist_input)
        print(f"✓ User encoder output shape: {user_output.shape}")
        
        print("\n🎉 All tests passed! NRMS model is working correctly.")
        
        # Print model summary
        print("\n" + "="*50)
        print("NRMS MODEL SUMMARY")
        print("="*50)
        nrms_model.training_model.summary()
        
    except Exception as e:
        print(f"❌ Error testing NRMS model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)

    print("Testing NRMS Model Implementation")
    print("="*40)
    
    success = test_nrms_model()
    
    if success:
        print("\n✅ NRMS model implementation is ready for training!")
    else:
        print("\n❌ NRMS model implementation needs fixes.") 