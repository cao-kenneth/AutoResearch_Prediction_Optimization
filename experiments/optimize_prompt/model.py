import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from baseline import get_model_response, parse_response

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
TRAIN_PATH = REPO_ROOT / "data" / "train.csv"
PROMPT_PATH = EXPERIMENT_DIR / "prompt.txt"
OUTPUT_DIR = EXPERIMENT_DIR / "outputs"


def get_next_run_id():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing = [
        path.name for path in OUTPUT_DIR.iterdir()
        if path.is_dir() and path.name.startswith("run_")
    ]
    if not existing:
        return "run_001"

    nums = [int(d.split("_")[1]) for d in existing]
    return f"run_{max(nums)+1:03d}"


def run_model():
    df = pd.read_csv(TRAIN_PATH)

    template = PROMPT_PATH.read_text(encoding="utf-8")

    probs = []
    reasons = []
    raw_outputs = []
    parse_status = []

    for i, row in df.iterrows():
        prompt = template.format(
            question=row["Question"],
            forecast_date=row["forecast_date_formatted"]
        )

        print(f"Processing row {i + 1} / {len(df)}")

        try:
            response = get_model_response(prompt)
            prob, reasoning = parse_response(response)

            probs.append(prob / 100.0)
            reasons.append(reasoning)
            raw_outputs.append(response)
            parse_status.append("ok")

        except Exception as e:
            print(f"Failed on row {i + 1}: {e}")
            probs.append(None)
            reasons.append("")
            raw_outputs.append(str(e))
            parse_status.append("failed")

        print(f"{i + 1}/{len(df)} questions done", flush=True)

    df["llm_prob"] = probs
    df["llm_reasoning"] = reasons
    df["raw_model_output"] = raw_outputs
    df["parse_status"] = parse_status

    valid_df = df[df["llm_prob"].notna()].copy()

    if len(valid_df) == 0:
        raise RuntimeError("No valid parsed predictions.")

    valid_df["brier"] = (
        valid_df["llm_prob"] - valid_df["resolution_binary"]
    ) ** 2

    brier = valid_df["brier"].mean()

    # ---- SAVE RUN ----
    run_id = get_next_run_id()
    run_path = OUTPUT_DIR / run_id
    run_path.mkdir()

    # save predictions
    df.to_csv(run_path / "predictions.csv", index=False)

    # save prompt used
    (run_path / "prompt.txt").write_text(template, encoding="utf-8")

    # save metrics
    with (run_path / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "brier": brier,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"Brier: {brier}")
    print(f"Saved to {run_path}")

    return brier


if __name__ == "__main__":
    run_model()
