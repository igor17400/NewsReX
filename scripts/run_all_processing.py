#!/usr/bin/env python3

import subprocess
import logging
from pathlib import Path
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_script(script_name, description):
    """Run a Python script and handle any errors."""
    logger.info(f"Starting {description}...")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully completed {description}")
        # Print the script's output
        for line in result.stdout.splitlines():
            logger.info(f"  {line}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {description}:")
        logger.error(f"  Exit code: {e.returncode}")
        logger.error(f"  Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error running {description}: {str(e)}")
        return False

def main():
    # Define the scripts to run in sequence
    scripts = [
        ("scripts/combine_news_datasets.py", "combining news datasets"),
        ("scripts/infer_news_dates.py", "inferring publication dates"),
        ("scripts/combine_news_with_dates.py", "combining news with dates")
    ]
    
    # Create scripts_output directory if it doesn't exist
    output_dir = Path("./data/mind/small/scripts_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run each script in sequence
    for script_path, description in scripts:
        if not run_script(script_path, description):
            logger.error(f"Pipeline failed at {description}")
            sys.exit(1)
    
    logger.info("All processing completed successfully!")

if __name__ == "__main__":
    main() 