#!/usr/bin/env python
import sys
import warnings
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from personalized_narratives.crew import PersonalizedNarratives
from personalized_narratives.batch_utils import execute_batch_processing

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def setup_logging(news_id: str) -> logging.Logger:
    """Set up logging for a specific news article."""
    output_dir = Path(f"output/news_{news_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"news_{news_id}")
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(output_dir / "processing.log")
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def process_article(row: pd.Series, logger: logging.Logger) -> None:
    """Process a single news article."""
    news_id = row['id']
    logger.info(f"Processing article {news_id}")
    
    try:
        # Prepare inputs
        inputs = {
            "title": row['title'],
            "abstract": row['abstract'],
            "publication_date": row['publication_date']
        }
        
        # Run the crew
        PersonalizedNarratives().crew().kickoff(inputs=inputs)
        logger.info(f"Completed processing article {news_id}")
        
    except Exception as e:
        logger.error(f"Error processing article {news_id}: {str(e)}")
        raise

def run():
    """
    Run the crew on a single article (original functionality).
    """
    inputs = {
        "title": "Fearing US abandonment, Kurds kept back channels wide open",
        "abstract": "When Syria's Kurdish fighters, America's longtime battlefield allies against the Islamic State, announced over the weekend that they were switching sides and joining up with Damascus and Moscow, it seemed like a moment of geopolitical whiplash. But in fact, the move had been in the works for more than a year.",
        "publication_date": "2019-11-11",
    }

    try:
        PersonalizedNarratives().crew().kickoff(inputs=inputs)
        print("Single article processing complete. Outputs should be in the default crew output location (likely CWD or a configured log path).")
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

def run_batch():
    """
    Run the crew on a batch of news articles.
    """
    # Load the dataset
    print("Loading dataset...")
    df_news_with_dates = pd.read_table(
        "../data/mind/small/scripts_output/news_with_dates.tsv",
        header=None,
        names=[
            "id", "category", "subcategory", "title", "abstract",
            "url", "title_entities", "abstract_entities", "publication_date"
        ]
    )
    
    # Create main output directory
    Path("output").mkdir(exist_ok=True)
    
    # Process articles in batches
    batch_size = 5000
    total_articles = len(df_news_with_dates)
    
    for batch_start in range(0, total_articles, batch_size):
        batch_end = min(batch_start + batch_size, total_articles)
        print(f"\nProcessing batch {batch_start//batch_size + 1} (articles {batch_start+1} to {batch_end})")
        
        for idx, row in df_news_with_dates.iloc[batch_start:batch_end].iterrows():
            logger = setup_logging(row['id'])
            process_article(row, logger)
        
        # Ask for user input after each batch
        if batch_end < total_articles:
            while True:
                user_input = input("\nContinue to next batch? (y/n): ").lower()
                if user_input in ['y', 'n']:
                    break
                print("Please enter 'y' or 'n'")
            
            if user_input == 'n':
                print("Processing paused. You can resume later.")
                break
    
    print("\nProcessing completed!")

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}
    try:
        PersonalizedNarratives().crew().train(
            n_iterations=int(sys.argv[2]), filename=sys.argv[3], inputs=inputs
        )

    except IndexError:
        print("Usage: main.py train <n_iterations> <filename>")
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        PersonalizedNarratives().crew().replay(task_id=sys.argv[2])

    except IndexError:
        print("Usage: main.py replay <task_id>")
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}

    try:
        PersonalizedNarratives().crew().test(
            n_iterations=int(sys.argv[2]), eval_llm=sys.argv[3], inputs=inputs
        )

    except IndexError:
        print("Usage: main.py test <n_iterations> <eval_llm_name>")
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "batch":
            execute_batch_processing()
        elif command == "train":
            train()
        elif command == "replay":
            replay()
        elif command == "test":
            test()
        elif command == "run":
            run()
        else:
            print(f"Unknown command: {command}")
            print("Usage: main.py [run|batch|train|replay|test] [options]")
    else:
        print("No command specified, running single article processing by default.")
        run()
