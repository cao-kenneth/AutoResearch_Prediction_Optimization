import os
import json
import pandas as pd
from datetime import datetime
from baseline import get_model_response, parse_response

OUTPUT_DIR = "outputs"

def get_next_run_id():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    existing = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("run_")]
    if not existing:
        return "run_001"

    nums = [int(d.split("_")[1]) for d in existing]
    return f"run_{max(nums)+1:03d}"


def run_model():
    df = pd.read_csv("data/train.csv")

    with open("prompt.txt", "r") as f:
        template = f.read()

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
    run_path = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_path)

    # save predictions
    df.to_csv(os.path.join(run_path, "predictions.csv"), index=False)

    # save prompt used
    with open(os.path.join(run_path, "prompt.txt"), "w") as f:
        f.write(template)

    # save metrics
    with open(os.path.join(run_path, "metrics.json"), "w") as f:
        json.dump({
            "brier": brier,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    print(f"Brier: {brier}")
    print(f"Saved to {run_path}")

    return brier


if __name__ == "__main__":
    run_model()
