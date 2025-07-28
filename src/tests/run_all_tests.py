#!/usr/bin/env python3
"""Run all model tests with Keras 3 + JAX backend."""

import os
import sys
import subprocess
from pathlib import Path

# Set JAX as Keras backend
os.environ["KERAS_BACKEND"] = "jax"

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import keras


def run_test(test_file: str) -> bool:
    """Run a single test file and return success status."""
    print(f"\n{'='*60}")
    print(f"Running {test_file}")
    print('='*60)
    
    test_path = Path(__file__).parent / test_file
    
    try:
        result = subprocess.run(
            [sys.executable, str(test_path)],
            capture_output=True,
            text=True,
            env={**os.environ, "KERAS_BACKEND": "jax"}
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running {test_file}: {e}")
        return False


def main():
    """Run all tests and report results."""
    print("BTC Test Suite - Keras 3 + JAX Backend")
    print(f"Backend: {keras.backend.backend()}")
    print("="*60)
    
    # Check JAX GPU availability first
    print("\n🔍 Checking JAX GPU availability...")
    gpu_result = subprocess.run(
        [sys.executable, str(Path(__file__).parent / "jax_gpu.py")],
        capture_output=True,
        text=True
    )
    print(f"JAX Platform: {gpu_result.stdout.strip()}")
    
    # List of test files to run
    test_files = [
        "nrms.py",
        "naml.py", 
        "lstur.py",
    ]
    
    results = {}
    
    # Run each test
    for test_file in test_files:
        success = run_test(test_file)
        results[test_file] = success
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_file, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_file:<20} {status}")
        if not success:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())