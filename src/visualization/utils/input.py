import pandas as pd
import re


def parse_recommendation_stream(filepath):
    """
        Parses a recommendation log file by treating it as a single stream
        of text. It finds all list blocks "[...]" and groups them by three,
        making it robust against word wrapping and inconsistent newlines.

        Args:
            filepath (str): The full path to the .txt log file.

        Returns:
            pd.DataFrame: A clean DataFrame with the parsed data, or None on error.
        """
    print(f"Reading file '{filepath}' as a continuous stream...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Read the entire file content into a single string
            file_content = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{filepath}' was not found.")
        return None

    # This regex find the content inside every square bracket.
    list_finder_regex = re.compile(r'\[(.*?)\]', re.DOTALL)

    all_list_contents = list_finder_regex.findall(file_content)

    # Ignore the header
    num_lists_found = len(all_list_contents)
    print(f"Found {num_lists_found} total list blocks in the file.")

    if num_lists_found % 3 != 0:
        print(
            f"WARNING: The number of lists ({num_lists_found}) is not a multiple of 3."
            f"The las record might be incomplete."
        )
        num_lists_found = (num_lists_found // 3) * 3

    impressions_list = []
    ground_truth_list = []
    scores_list = []

    # Iterate through the flat list of contents in chunks of 3
    for i in range(0, num_lists_found, 3):
        record_num = (i // 3) + 1
        try:
            impression_str = all_list_contents[i]
            ground_truth_str = all_list_contents[i + 1]
            scores_str = all_list_contents[i + 2]

            # Parse the content of each string
            impression_ids = [int(x) for x in impression_str.split()]
            ground_truths = [float(x) for x in ground_truth_str.split(',')]
            scores = [float(x) for x in scores_str.split(',')]

            # Final validation check
            if len(impression_ids) == len(ground_truths) == len(scores):
                impressions_list.append(impression_ids)
                ground_truth_list.append(ground_truths)
                scores_list.append(scores)
            else:
                print(f"WARNING: Mismatched list lengths in record #{record_num}. "
                      f"Skipping.")
        except Exception as e:
            print(f"ERROR: {e}."
                  f"Failed to parse record #{record_num}."
                  f"Skipping.")

    # Parse final dataframe
    df = pd.DataFrame({
        "ImpressionIDs": impressions_list,
        "GroundTruths": ground_truth_list,
        "PredictionScores": scores_list
    })

    print(f"\n✅ Parsing complete. Successfully loaded {len(df)} records.")
    return df
