import pandas as pd
from pathlib import Path
import logging
import sys
from contextlib import contextmanager
from personalized_narratives.crew import PersonalizedNarratives

@contextmanager
def redirect_crew_logs(file_path: Path):
    """Temporarily redirect stdout and stderr to a file."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            sys.stdout = f
            sys.stderr = f
            yield
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

def _configure_logger_handlers(logger: logging.Logger):
    """Configures file and console handlers for the given logger."""
    news_id_from_logger_name = logger.name.split('news_')[-1]
    output_dir = Path(f"output/news_{news_id_from_logger_name}")
    
    # File handler
    script_log_path = output_dir / "processing.log"
    fh = logging.FileHandler(script_log_path)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)

def setup_logging(news_id: str) -> logging.Logger:
    """Set up logging for a specific news article."""
    output_dir = Path(f"output/news_{news_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"news_{news_id}")
    # Prevent duplicate handlers if called multiple times for the same logger
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        _configure_logger_handlers(logger)
    
    return logger

def _process_single_article(row: pd.Series, logger: logging.Logger) -> None:
    """Internal function to process a single news article with its dedicated logger."""
    news_id = row['id']
    logger.info(f"Processing article {news_id} - Title: {row.get('title', 'N/A')}")
    
    article_output_dir = Path(f"output/news_{news_id}")
    crew_verbose_log_path = article_output_dir / "crew_verbose.log"

    try:
        inputs = {
            "title": row['title'],
            "abstract": row['abstract'],
            "publication_date": row['publication_date']
        }
        
        # Initialize the crew instance with the article-specific output path
        pn_crew_instance = PersonalizedNarratives(article_output_path=article_output_dir)

        logger.info(f"Running CrewAI kickoff for {news_id}. Crew verbose logs will be in {crew_verbose_log_path}")
        with redirect_crew_logs(crew_verbose_log_path):
            pn_crew_instance.crew().kickoff(inputs=inputs)
        
        logger.info(f"Completed CrewAI processing for article {news_id}")
        
    except Exception as e:
        logger.error(f"Error processing article {news_id}: {str(e)}", exc_info=True)
        # Optionally re-raise or handle more gracefully depending on desired batch behavior
        # raise

def execute_batch_processing():
    """
    Loads the dataset and runs the PersonalizedNarratives crew on a batch of news articles.
    Handles batching and user prompts for continuation.
    """
    # Load the dataset
    print("Loading dataset...")
    try:
        df_news_with_dates = pd.read_table(
            "../../data/mind/small/scripts_output/news_with_dates.tsv",
            header=None,
            names=[
                "id", "category", "subcategory", "title", "abstract",
                "url", "title_entities", "abstract_entities", "publication_date"
            ]
        )
    except FileNotFoundError:
        print("Error: The data file news_with_dates.tsv was not found.")
        print("Expected path: ../../data/mind/small/scripts_output/news_with_dates.tsv relative to where main.py is run (e.g., from project root or src/ folder)")
        return
    
    # Create main output directory if it doesn't exist
    Path("output").mkdir(exist_ok=True)
    
    batch_size = 5000
    total_articles = len(df_news_with_dates)
    
    processed_count = 0
    for batch_start in range(0, total_articles, batch_size):
        batch_end = min(batch_start + batch_size, total_articles)
        current_batch_number = batch_start // batch_size + 1
        print(f"\nProcessing batch {current_batch_number} (articles {batch_start + 1} to {batch_end} of {total_articles})")
        
        batch_df = df_news_with_dates.iloc[batch_start:batch_end]
        
        for idx, row in batch_df.iterrows():
            processed_count += 1
            article_logger = setup_logging(row['id'])
            _process_single_article(row, article_logger)
            if processed_count % 10 == 0:
                print(f"  Processed {processed_count}/{total_articles} articles...")
        
        print(f"Completed batch {current_batch_number}.")

        if batch_end < total_articles:
            while True:
                user_input = input("Continue to next batch? (y/n): ").strip().lower()
                if user_input in ['y', 'n']:
                    break
                print("Invalid input. Please enter 'y' or 'n'.")
            
            if user_input == 'n':
                print("Processing paused by user. You can resume later by re-running the batch command.")
                break
    
    print(f"\nBatch processing finished. Total articles processed in this run: {processed_count}.") 