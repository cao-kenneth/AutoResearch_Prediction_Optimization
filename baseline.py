import pandas as pd
import re
import subprocess
import os

CODEX_TIMEOUT_SECONDS = 180
TRAIN_PATH = "data/train.csv"
OUTPUT_DIR = "outputs"
TRAIN_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "train_predictions.csv")
TRAIN_VALID_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "train_valid_predictions.csv")

def build_prompt(question, forecast_date):
    return f"""
You are making a probabilistic forecast.

Do not use any information after {forecast_date}.

For the question below, return:
1. A probability from 0 to 100 that the question resolves YES.
2. One paragraph of reasoning.

Question: {question}

Return EXACTLY in this format:

Probability: <number>
Reasoning: <one paragraph>

Do not include any other text.
""".strip()


def parse_response(text):
    prob_match = re.search(r"Probability:\s*([0-9]+(?:\.[0-9]+)?)", text)
    reason_match = re.search(r"Reasoning:\s*(.*)", text, re.DOTALL)

    if not prob_match:
        raise ValueError(f"Could not parse probability from:\n{text}")

    prob = float(prob_match.group(1))
    if prob < 0 or prob > 100:
        raise ValueError(f"Probability out of range: {prob}")

    reasoning = reason_match.group(1).strip() if reason_match else ""
    return prob, reasoning


def get_model_response(prompt):
    try:
        result = subprocess.run(
            ["codex", "exec", "-m", "gpt-5.4", prompt],
            input="",
            capture_output=True,
            text=True,
            timeout=CODEX_TIMEOUT_SECONDS
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex timed out after {CODEX_TIMEOUT_SECONDS} seconds.")

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex command failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH)

    llm_probs = []
    llm_reasons = []
    raw_outputs = []
    parse_status = []

    for i, row in df.iterrows():
        question = row["Question"]
        forecast_date = row["forecast_date_formatted"]
        prompt = build_prompt(question, forecast_date)

        print(f"Processing row {i + 1} / {len(df)}")

        try:
            response = get_model_response(prompt)
            prob, reasoning = parse_response(response)

            llm_probs.append(prob / 100.0)
            llm_reasons.append(reasoning)
            raw_outputs.append(response)
            parse_status.append("ok")

        except Exception as e:
            print(f"Failed on row {i + 1}: {e}")

            llm_probs.append(None)
            llm_reasons.append("")
            raw_outputs.append(str(e))
            parse_status.append("failed")

    df["llm_prob"] = llm_probs
    df["llm_reasoning"] = llm_reasons
    df["raw_model_output"] = raw_outputs
    df["parse_status"] = parse_status

    valid_df = df[df["llm_prob"].notna()].copy()

    if len(valid_df) == 0:
        raise RuntimeError("No valid parsed predictions.")

    valid_df["llm_brier_row"] = (
        valid_df["llm_prob"] - valid_df["resolution_binary"]
    ) ** 2
    valid_df["community_brier_row"] = (
        valid_df["community_prob"] - valid_df["resolution_binary"]
    ) ** 2

    llm_brier = valid_df["llm_brier_row"].mean()
    community_brier = valid_df["community_brier_row"].mean()

    df.to_csv(TRAIN_PREDICTIONS_PATH, index=False)
    valid_df.to_csv(TRAIN_VALID_PREDICTIONS_PATH, index=False)

    print(f"\nValid parsed rows: {len(valid_df)} / {len(df)}")
    print(f"LLM Brier Score: {llm_brier}")
    print(f"Community Brier Score: {community_brier}")


if __name__ == "__main__":
    main()
