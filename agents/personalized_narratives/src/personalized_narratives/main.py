#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from personalized_narratives.crew import PersonalizedNarratives

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """
    Run the crew.
    """
    # inputs = {
    #     "title": "How to Cook the Juiciest Instant Pot Turkey Breast EVER",
    #     "abstract": "For a cozy, hearty meal, look no further than this juicy Instant Pot turkey breast. Pass the gravy, please! The post How to Cook the Juiciest Instant Pot Turkey Breast Ever appeared first on Taste of Home.",
    #     "publication_date": "2019-11-11",
    # }
    
    # inputs = {
    #     "title": "Former Monessen School Bus Driver Pleads Guilty To Possessing Child Pornography",
    #     "abstract": "A former Monessen school bus driver is facing charges of possessing child pornography.",
    #     "publication_date": "2019-11-11",
    # }
    
    inputs = {
        "title": "Vetiquette program teaches kids about character through stories of war heroes' sacrifice",
        "abstract": "The Rockland nonprofit connects students with the stories of sacrifice and perseverance.",
        "publication_date": "2019-11-11",
    }

    try:
        PersonalizedNarratives().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}
    try:
        PersonalizedNarratives().crew().train(
            n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        PersonalizedNarratives().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {"topic": "AI LLMs", "current_year": str(datetime.now().year)}

    try:
        PersonalizedNarratives().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
