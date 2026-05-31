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
    if prob < 0 or prob > 100:
        raise ValueError(f"Probability out of range: {prob}")

    reasoning = reason_match.group(1).strip() if reason_match else ""
    return prob, reasoning


def get_model_response(prompt):
    result = subprocess.run(
        ["codex", "exec", prompt],
        input="",
        capture_output=True,
        text=True,
        timeout=180,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex command failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def main():
    df = pd.read_csv(RAW_DATA_PATH)

    row = df.iloc[0]
    question = row["Question"]
    forecast_date = row["Forecast_Date"]

    prompt = build_prompt(question, forecast_date)

    print("=== TEST QUESTION ===")
    print(question)
    print()
    print("=== FORECAST DATE ===")
    print(forecast_date)
    print()

    response = get_model_response(prompt)

    print("=== RAW MODEL OUTPUT ===")
    print(response)
    print()

    prob, reasoning = parse_response(response)

    print("=== PARSED OUTPUT ===")
    print("Probability:", prob)
    print("Reasoning:", reasoning)


if __name__ == "__main__":
    main()
