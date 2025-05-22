import pandas as pd
from pathlib import Path
import logging
import sys
from contextlib import contextmanager
from tqdm import tqdm
from personalized_narratives.crew import PersonalizedNarratives

# Helper function to find already processed news IDs
def get_processed_news_ids(output_base_dir: Path) -> set[str]:
    """Scans the output directory to find IDs of already processed news articles."""
    processed_ids = set()
    if not output_base_dir.is_dir():
        # This is not an error, just means no output dir exists yet.
        # print(f"Info: Output directory '{output_base_dir}' not found. Assuming no articles processed yet.")
        return processed_ids

    for item in output_base_dir.iterdir():
        if item.is_dir() and item.name.startswith("news_"):
            parts = item.name.split("_", 1)
            if len(parts) > 1 and parts[1]:
                processed_ids.add(parts[1])
    return processed_ids

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
    Handles batching, skipping processed articles, and user prompts for continuation.
    Provides detailed progress information.
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
        print("Please ensure the path '../../data/mind/small/scripts_output/news_with_dates.tsv' is correct relative to your execution directory.")
        return
    
    total_articles_in_dataset = len(df_news_with_dates)
    output_dir_base = Path("output")
    output_dir_base.mkdir(parents=True, exist_ok=True)
    
    print("\nIdentifying already processed articles by scanning existing output folders...")
    globally_processed_ids = get_processed_news_ids(output_dir_base)
    num_already_processed_globally = len(globally_processed_ids)

    # Calculate how many articles in the current dataset are NOT yet processed
    # This is important if globally_processed_ids contains IDs not in the current df_news_with_dates
    ids_in_current_dataset = set(df_news_with_dates['id'])
    ids_to_potentially_process_in_this_run = ids_in_current_dataset - globally_processed_ids
    num_to_potentially_process_this_run = len(ids_to_potentially_process_in_this_run)

    print(f"Dataset contains {total_articles_in_dataset} articles.")
    if num_already_processed_globally > 0:
        print(f"{num_already_processed_globally} articles appear to be already processed (output folders exist).")
    print(f"This session will attempt to process up to {num_to_potentially_process_this_run} new articles.")
        
    batch_size = 5000 
    
    overall_articles_processed_this_run = 0
    overall_articles_skipped_this_run = 0 # Articles skipped because they were in globally_processed_ids
    
    for batch_start in range(0, total_articles_in_dataset, batch_size):
        batch_end = min(batch_start + batch_size, total_articles_in_dataset)
        current_batch_number = batch_start // batch_size + 1
        
        batch_df = df_news_with_dates.iloc[batch_start:batch_end]
        num_articles_in_batch = len(batch_df)

        # Calculate how many in this specific batch are already processed vs. to be processed
        batch_ids_to_skip = {news_id for news_id in batch_df['id'] if news_id in globally_processed_ids}
        num_to_skip_in_batch = len(batch_ids_to_skip)
        num_to_process_in_batch = num_articles_in_batch - num_to_skip_in_batch

        print(f"\n--- Processing Batch {current_batch_number} (Articles {batch_start + 1} to {batch_end} of {total_articles_in_dataset}) ---")
        print(f"This batch contains {num_articles_in_batch} articles: {num_to_process_in_batch} to process, {num_to_skip_in_batch} will be skipped (already processed).")
        
        batch_articles_processed_count = 0
        batch_articles_skipped_count = 0
        
        with tqdm(total=num_articles_in_batch, desc=f"Batch {current_batch_number}", unit="article") as pbar:
            for idx, row in batch_df.iterrows():
                news_id = row['id']
                if news_id in globally_processed_ids:
                    # tqdm.write(f"Skipping article {news_id}: Already processed.") # Optional: for very verbose logging
                    overall_articles_skipped_this_run += 1
                    batch_articles_skipped_count += 1
                    pbar.update(1)
                    pbar.set_postfix_str(f"Processed: {batch_articles_processed_count}, Skipped: {batch_articles_skipped_count}")
                    continue

                article_logger = setup_logging(news_id) 
                _process_single_article(row, article_logger) 
                overall_articles_processed_this_run += 1
                batch_articles_processed_count +=1
                pbar.update(1)
                pbar.set_postfix_str(f"Processed: {batch_articles_processed_count}, Skipped: {batch_articles_skipped_count}")
        
        print(f"--- Completed Batch {current_batch_number} ---")
        print(f"In this batch: Processed {batch_articles_processed_count} new articles, Skipped {batch_articles_skipped_count} (already processed).")

        if batch_end < total_articles_in_dataset:
            while True:
                user_input = input(f"\nSession so far: {overall_articles_processed_this_run} processed, {overall_articles_skipped_this_run} skipped. Continue to next batch? (y/n): ").strip().lower()
                if user_input in ['y', 'n']:
                    break
                print("Invalid input. Please enter 'y' or 'n'.")
            
            if user_input == 'n':
                print("\nProcessing paused by user. You can resume later by re-running the batch command.")
                break
        else:
            print("\nAll batches have been processed.") 
    
    print(f"\n--- Batch Processing Session Finished ---")
    print(f"Total new articles processed in this run: {overall_articles_processed_this_run}")
    print(f"Total articles skipped in this run (found as already processed): {overall_articles_skipped_this_run}")
    remaining_to_process = num_to_potentially_process_this_run - overall_articles_processed_this_run
    if remaining_to_process > 0 and batch_end < total_articles_in_dataset : # only if paused early
        print(f"There are approximately {remaining_to_process} articles from the initial set that were not processed in this session.") 