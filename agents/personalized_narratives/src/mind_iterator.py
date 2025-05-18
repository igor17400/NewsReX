from typing import Iterator, Dict, Any, Optional, List
import pandas as pd
import os 
from datetime import datetime

# Constants for MIND dataset paths
# Path to the MIND data, relative to the execution directory 
# (expected to be 'agents/narrative_generator/')
# e.g., <BTC_PROJECT_ROOT>/agents/narrative_generator/
MIND_DATA_BASE_PATH = "../../data/mind/small/" 
# This path should point to <BTC_PROJECT_ROOT>/data/mind/small/

MIND_TRAIN_DIR_NAME = "train"
MIND_TEST_DIR_NAME = "val"  # As per BTC project, 'val' is used for test set news/behaviors
NEWS_TSV_FILE = "news.tsv"
BEHAVIORS_TSV_FILE = "behaviors.tsv"


def load_news_corpus(
    data_base_path: str, train_dir: str, test_dir: str, news_file: str
) -> Dict[str, str]:
    """
    Loads news articles from both train and test MIND directories,
    combines them, and returns a dictionary mapping NewsID to content (Title + Abstract).
    """
    news_dfs = []
    for data_split_dir in [train_dir, test_dir]:
        news_path = os.path.join(data_base_path, data_split_dir, news_file)
        if not os.path.exists(news_path):
            print(f"Warning: News file not found at {news_path}. Skipping.")
            continue
        try:
            df = pd.read_csv(
                news_path,
                sep="\t",
                header=None,
                names=[
                    "NewsID",
                    "Category",
                    "SubCategory",
                    "Title",
                    "Abstract",
                    "URL",
                    "TitleEntities",
                    "AbstractEntities",
                ],
            )
            news_dfs.append(df)
        except Exception as e:
            print(f"Error reading news file {news_path}: {e}")

    if not news_dfs:
        print("Error: No news data could be loaded. Returning empty corpus.")
        return {}

    combined_news_df = pd.concat(news_dfs).drop_duplicates(subset=["NewsID"]).set_index("NewsID")

    news_corpus = {}
    for news_id, row in combined_news_df.iterrows():
        title = str(row.get("Title", ""))
        abstract = str(row.get("Abstract", ""))
        content = f"{title}. {abstract}".strip()
        if not content or content == ".":
            content = title if title else "No Title Available"  # Fallback if abstract is also empty
        news_corpus[str(news_id)] = content

    print(f"Loaded a unified news corpus with {len(news_corpus)} unique articles.")
    return news_corpus


def get_mind_articles_iterator(
    data_base_path: str = MIND_DATA_BASE_PATH,
    train_dir: str = MIND_TRAIN_DIR_NAME,
    test_dir: str = MIND_TEST_DIR_NAME,
    news_file: str = NEWS_TSV_FILE,
    behaviors_file: str = BEHAVIORS_TSV_FILE,
    process_train_behaviors: bool = True,
    process_test_behaviors: bool = True,
    max_articles_to_yield_total: Optional[int] = None,  # For limiting output during testing
) -> Iterator[Dict[str, Any]]:
    """
    Iterates through MIND dataset behaviors, extracts article IDs and interaction timestamps,
    looks up article content from a unified news corpus, and yields a dictionary
    for each article impression.

    The 'publication_date_str' is derived from the user interaction timestamp in behaviors.tsv.
    """
    news_corpus = load_news_corpus(data_base_path, train_dir, test_dir, news_file)
    if not news_corpus:
        print("Critical: News corpus is empty. Cannot proceed with behavior iteration.")
        return

    behavior_dirs_to_process = []
    if process_train_behaviors:
        behavior_dirs_to_process.append(train_dir)
    if process_test_behaviors:
        behavior_dirs_to_process.append(test_dir)

    if not behavior_dirs_to_process:
        print("No behavior sets selected for processing (train/test).")
        return

    yielded_articles_count = 0

    for data_split_dir in behavior_dirs_to_process:
        behaviors_path = os.path.join(data_base_path, data_split_dir, behaviors_file)
        if not os.path.exists(behaviors_path):
            print(
                f"Warning: Behaviors file not found at {behaviors_path} for split '{data_split_dir}'. Skipping."
            )
            continue

        print(f"Processing behaviors from: {behaviors_path}")
        try:
            # Reading behaviors.tsv: UserID, Time, History, Impressions
            # Using chunksize for potentially large behavior files
            for chunk_df in pd.read_csv(
                behaviors_path,
                sep="\t",
                header=None,
                names=["ImpressionID", "UserID", "Time", "History", "Impressions"],
                usecols=["Time", "Impressions"],  # Only need these columns
                chunksize=10000,  # Process in chunks
            ):
                for _, row in chunk_df.iterrows():
                    try:
                        timestamp_str = str(row.get("Time"))
                        # Example timestamp: "11/11/2019 9:02:04 AM"
                        # Convert to "YYYY-MM-DD"
                        dt_object = datetime.strptime(timestamp_str, "%m/%d/%Y %I:%M:%S %p")
                        reference_date_str = dt_object.strftime("%Y-%m-%d")
                    except ValueError as ve:
                        print(
                            f"Warning: Could not parse timestamp '{timestamp_str}': {ve}. Skipping row."
                        )
                        continue
                    except Exception as e_ts:
                        print(
                            f"Warning: Error processing timestamp '{timestamp_str}': {e_ts}. Skipping row."
                        )
                        continue

                    impressions_str = str(row.get("Impressions", ""))
                    article_ids_in_impression = [
                        imp.split("-")[0]
                        for imp in impressions_str.split(" ")
                        if imp and "-" in imp
                    ]

                    for news_id in article_ids_in_impression:
                        if news_id in news_corpus:
                            article_content = news_corpus[news_id]
            yield {
                                "news_id": news_id,
                                "content": article_content,
                                "publication_date_str": reference_date_str,  # This is the interaction date
                            }
                            yielded_articles_count += 1
                            if (
                                max_articles_to_yield_total is not None
                                and yielded_articles_count >= max_articles_to_yield_total
                            ):
                                print(
                                    f"Reached max_articles_to_yield_total ({max_articles_to_yield_total}). Stopping iteration."
                                )
                                return
                        # else:
                        #     print(f"Warning: NewsID {news_id} from behavior log not found in news corpus. Skipping.")
                if (
                    max_articles_to_yield_total is not None
                    and yielded_articles_count >= max_articles_to_yield_total
                ):
                    break  # Break from outer chunk loop as well

    except FileNotFoundError:
            print(f"Error: Behaviors file not found during pandas read at {behaviors_path}.")
    except Exception as e:
            print(f"Error processing behaviors file {behaviors_path}: {e}")


if __name__ == "__main__":
    print(
        f"Testing MIND Article Iterator. Using base data path: '{MIND_DATA_BASE_PATH}'"
    )
    resolved_base_path = os.path.abspath(MIND_DATA_BASE_PATH)
    print(
        f"This path, when resolved from CWD ({os.getcwd()}), points to: {resolved_base_path}"
    )
    print(
        f"Expected train data in: {os.path.join(resolved_base_path, MIND_TRAIN_DIR_NAME)}"
    )
    print(
        f"Expected test data in: {os.path.join(resolved_base_path, MIND_TEST_DIR_NAME)}"
    )

    # Check if base data path exists
    if not os.path.exists(resolved_base_path): # Check the resolved path
        print(
            f"CRITICAL: Base MIND data path does not exist: {resolved_base_path}"
        )
        print("Iterator test will likely fail or yield no data. Please check the path and CWD.")
        # Attempt to create a more helpful message about where it's looking from
        abs_path_file = os.path.abspath(__file__)
        print(f"This script (__file__) is located at: {abs_path_file}")
        # Corrected diagnostic for expected BTC root based on script location
        expected_btc_root_from_script_location = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../../") 
        )
        print(
            f"For reference, BTC project root based on script location is likely: {expected_btc_root_from_script_location}"
        )
        print(
            f"And thus, data path based on script location would be: {os.path.join(expected_btc_root_from_script_location, 'data/mind/small/')}"
        )
        print(f"Ensure your CWD is 'agents/narrative_generator' and '{MIND_DATA_BASE_PATH}' is correct relative to it.")

    # Example: Process only train behaviors and limit to 5 articles for testing
    iterator = get_mind_articles_iterator(
        process_train_behaviors=True,
        process_test_behaviors=False,  # Switch to True to include test behaviors
        max_articles_to_yield_total=5,
    )

    count = 0
    for article_data in iterator:
        count += 1
        print(f"\n--- Article {count} (from {article_data.get('source_split', 'N/A')}) ---")
        print(f"  ID: {article_data['news_id']}")
        print(f"  Content: {article_data['content'][:100]}...")  # Print first 100 chars
        print(f"  Interaction Date: {article_data['publication_date_str']}")

    if count == 0:
        print("No articles were yielded. Check file paths, data integrity, and iterator logic.")
    else:
        print(f"\nSuccessfully yielded {count} articles.")

print(
    f"If you see 'CRITICAL: Base MIND data path does not exist', ensure that the path '{MIND_DATA_BASE_PATH}' is correct relative to your CWD."
)
print(f"When run, it tried to resolve to: {os.path.abspath(MIND_DATA_BASE_PATH)}")
print(
    "This script expects 'train' and 'val' subdirectories with news.tsv and behaviors.tsv within that path."
)
