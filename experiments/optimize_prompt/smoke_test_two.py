import re
import subprocess
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
RAW_DATA_PATH = REPO_ROOT / "data" / "RawData.csv"


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
    reasoning = reason_match.group(1).strip() if reason_match else ""

    return prob, reasoning


def get_model_response(prompt):
    result = subprocess.run(
        ["codex", "exec", prompt],
        input="",
        capture_output=True,
        text=True,
        timeout=180
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    # take first 2 rows
    test_rows = df.iloc[:2]

    for i, row in test_rows.iterrows():
        question = row["Question"]

        # clean date format (important!)
        forecast_date = pd.to_datetime(row["Forecast_Date"]).strftime("%B %d, %Y")

        prompt = build_prompt(question, forecast_date)

        print(f"\n=== TEST {i+1} ===")
        print("Question:", question)
        print("Forecast Date:", forecast_date)

        response = get_model_response(prompt)

        print("\n--- RAW OUTPUT ---")
        print(response)

        try:
            prob, reasoning = parse_response(response)

            print("\n--- PARSED ---")
            print("Probability:", prob)
            print("Reasoning:", reasoning[:150], "...")  # truncate for readability

        except Exception as e:
            print("\nPARSE FAILED")
            print(e)


if __name__ == "__main__":
    main()
