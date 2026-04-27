import os
import json
import subprocess
import time

import pandas as pd


PROMPT_FILE = "prompt.txt"
BASELINE_PREDICTIONS_PATH = "outputs/train_valid_predictions.csv"
START_RUN_DIR = "outputs/run_001"
OUTPUT_DIR = "outputs"
COMPLETED_OPTIMIZATION_ITERATIONS = 1
ADDITIONAL_OPTIMIZATION_ITERATIONS = 2
CODEX_TIMEOUT_SECONDS = 180
CODEX_MODEL = "gpt-5.4"


def score_predictions(df):
    if "llm_brier_row" in df.columns:
        return df["llm_brier_row"].mean()
    if "brier" in df.columns:
        return df["brier"].mean()
    return ((df["llm_prob"] - df["resolution_binary"]) ** 2).mean()


def summarize_predictions(path, max_rows=8):
    df = pd.read_csv(path)
    valid_df = df[df["llm_prob"].notna()].copy()

    if len(valid_df) == 0:
        raise RuntimeError(f"No valid predictions found in {path}.")

    if "llm_brier_row" not in valid_df.columns:
        valid_df["llm_brier_row"] = (
            valid_df["llm_prob"] - valid_df["resolution_binary"]
        ) ** 2

    score = score_predictions(valid_df)
    worst_rows = valid_df.sort_values("llm_brier_row", ascending=False).head(max_rows)

    lines = [
        f"Prediction file: {path}",
        f"Valid rows: {len(valid_df)} / {len(df)}",
        f"Brier score: {score:.6f}",
        "",
        "Worst forecast rows:",
    ]

    for _, row in worst_rows.iterrows():
        lines.extend([
            f"- Question: {row['Question']}",
            f"  Forecast date: {row['forecast_date_formatted']}",
            f"  Resolution: {row['Resolution']} ({row['resolution_binary']})",
            f"  Predicted probability: {row['llm_prob']:.3f}",
            f"  Row Brier: {row['llm_brier_row']:.6f}",
            f"  Reasoning: {str(row.get('llm_reasoning', ''))[:700]}",
        ])

    return score, "\n".join(lines)


def load_starting_point():
    run_predictions_path = os.path.join(START_RUN_DIR, "predictions.csv")
    run_prompt_path = os.path.join(START_RUN_DIR, "prompt.txt")
    run_metrics_path = os.path.join(START_RUN_DIR, "metrics.json")

    if os.path.exists(run_predictions_path) and os.path.exists(run_prompt_path):
        with open(run_prompt_path, "r") as f:
            best_prompt = f.read()

        if os.path.exists(run_metrics_path):
            with open(run_metrics_path, "r") as f:
                best_score = float(json.load(f)["brier"])
        else:
            best_score, _ = summarize_predictions(run_predictions_path)

        return best_prompt, best_score, run_predictions_path, START_RUN_DIR

    if not os.path.exists(BASELINE_PREDICTIONS_PATH):
        raise FileNotFoundError(
            f"Missing baseline predictions: {BASELINE_PREDICTIONS_PATH}. "
            "Run baseline.py first."
        )

    with open(PROMPT_FILE, "r") as f:
        best_prompt = f.read()

    best_score, _ = summarize_predictions(BASELINE_PREDICTIONS_PATH)
    return best_prompt, best_score, BASELINE_PREDICTIONS_PATH, "baseline"


def strip_code_fence(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def optimize_prompt_with_codex(current_prompt, predictions_summary, iteration):
    optimizer_prompt = f"""
You are optimizing a forecasting prompt for a local experiment.

The model is evaluated with Brier score, so lower is better. Review the
prediction summary and rewrite the prompt to improve calibration.

Constraints:
- Return only the revised prompt text.
- Preserve the placeholders {{forecast_date}} and {{question}}.
- Require this exact output format from the forecasting model:
  Probability: <number from 0 to 100, with no percent sign>
  Reasoning: <one paragraph>
- Make clear that probabilities must be reported on a 0-to-100 scale,
  not as 0-to-1 decimals.
- Do not add external browsing, tools, APIs, or data retrieval.
- Keep the prompt concise.

Optimization iteration: {iteration}

Current prompt:
{current_prompt}

Latest prediction summary:
{predictions_summary}
""".strip()

    result = subprocess.run(
        ["codex", "exec", "-m", CODEX_MODEL, optimizer_prompt],
        input="",
        capture_output=True,
        text=True,
        timeout=CODEX_TIMEOUT_SECONDS,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Codex prompt optimization failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    new_prompt = strip_code_fence(result.stdout)

    required_strings = ["{forecast_date}", "{question}", "Probability:", "Reasoning:"]
    missing = [value for value in required_strings if value not in new_prompt]
    if missing:
        raise RuntimeError(f"Optimized prompt is missing required text: {missing}")

    return new_prompt


def run_model():
    process = subprocess.Popen(
        ["python", "model.py"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    score = None
    run_path = None

    for line in process.stdout:
        print(line, end="", flush=True)

        if line.startswith("Brier:"):
            score = float(line.split(":", 1)[1].strip())
        elif line.startswith("Saved to "):
            run_path = line.removeprefix("Saved to ").strip()

    returncode = process.wait()

    if returncode != 0:
        return None, None

    predictions_path = os.path.join(run_path, "predictions.csv") if run_path else None
    return score, predictions_path


def main():
    best_prompt, best_score, latest_predictions_path, start_label = load_starting_point()
    with open(PROMPT_FILE, "w") as f:
        f.write(best_prompt)

    total_iterations = (
        COMPLETED_OPTIMIZATION_ITERATIONS + ADDITIONAL_OPTIMIZATION_ITERATIONS
    )
    print(f"Starting from {start_label} score: {best_score:.6f}")

    start_iteration = COMPLETED_OPTIMIZATION_ITERATIONS + 1
    stop_iteration = total_iterations + 1

    for iteration in range(start_iteration, stop_iteration):
        print(f"\n=== Optimization iteration {iteration} / {total_iterations} ===")

        _, predictions_summary = summarize_predictions(latest_predictions_path)
        previous_prompt = best_prompt

        try:
            new_prompt = optimize_prompt_with_codex(
                best_prompt,
                predictions_summary,
                iteration,
            )

            with open(PROMPT_FILE, "w") as f:
                f.write(new_prompt)

            new_score, new_predictions_path = run_model()

            if new_score is not None and new_score < best_score:
                print(f"Iteration {iteration} improved: {new_score:.6f}")
                best_score = new_score
                best_prompt = new_prompt
            else:
                print(f"Iteration {iteration} rejected: {new_score}")
                with open(PROMPT_FILE, "w") as f:
                    f.write(previous_prompt)

            if new_predictions_path:
                latest_predictions_path = new_predictions_path

        except Exception as e:
            print(f"Iteration {iteration} crashed: {e}")
            with open(PROMPT_FILE, "w") as f:
                f.write(previous_prompt)

        time.sleep(1)

    print(f"\nBest score after {total_iterations} iterations: {best_score:.6f}")


if __name__ == "__main__":
    main()
