#!/usr/bin/env python
import sys
import warnings
# pandas and Path likely not needed here anymore if only used by run_batch
# logging might also not be needed if its setup is fully in batch_utils
from datetime import datetime

from personalized_narratives.crew import PersonalizedNarratives
# The main batch processing logic is now in batch_utils
from personalized_narratives.batch_utils import execute_batch_processing 

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

# Removed get_processed_news_ids, setup_logging, and process_article functions
# as their logic is now handled within or by batch_utils.py

def run():
    """
    Run the crew on a single article (original functionality).
    """
    inputs = {
        "title": "12 Big Signs of Happy, Healthy Relationships",
        "abstract": "If you nod your head when reading this list, that's very good news.",
        "publication_date": "2019-11-11",
    }

    try:
        # Assuming PersonalizedNarratives() doesn't require article_output_path for single runs
        # or handles its absence gracefully.
        PersonalizedNarratives().crew().kickoff(inputs=inputs)
        print("Single article processing complete. Outputs should be in the default crew output location (likely CWD or a configured log path).")
    except Exception as e:
        # It's good practice to log the exception or print its details.
        # print(f"Error during single article run: {e}", file=sys.stderr)
        raise Exception(f"An error occurred while running the crew: {e}")

# Removed run_batch function as its functionality is now in batch_utils.execute_batch_processing

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
            execute_batch_processing() # Calls the function from batch_utils
        elif command == "train":
            train()
        elif command == "replay":
            replay()
        elif command == "test":
            test()
        elif command == "run": # Kept single run functionality
            run()
        else:
            print(f"Unknown command: {command}")
            print("Usage: main.py [run|batch|train|replay|test] [options]")
    else:
        print("No command specified, running single article processing by default.")
        run() # Default action is to run single processing
