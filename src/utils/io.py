import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.console import Console

def save_predictions_to_file_fn(
    predictions_dict: Dict[str, Tuple[List, List]],
    output_dir: Path,
    epoch_idx: Optional[int] = None,
    mode: str = "val"
) -> None:
    console = Console()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{mode}_predictions"
    if epoch_idx is not None:
        filename += f"_epoch_{epoch_idx + 1}"
    filename += ".txt"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        f.write("ImpressionID\tGroundTruth\tPredictionScores\n")
        for imp_id, (gt, pred_scores) in predictions_dict.items():
            gt_str = json.dumps(gt)
            pred_scores_str = json.dumps(pred_scores)
            f.write(f"{imp_id}\t{gt_str}\t{pred_scores_str}\n")
    console.log(f"Saved {mode} predictions to {filepath}") 