from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
TRAIN_PATH = REPO_ROOT / "data" / "train.csv"
DEFAULT_SAMPLE_PATH = EXPERIMENT_DIR / "sample_20.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SAMPLE_SEED = 390
DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class ExperimentConfig:
    order: int
    experiment_id: str
    condition: str
    upper_thresholds: list[float]
    lower_thresholds: list[float]

    @property
    def upper_path(self) -> list[float]:
        return [50.0] + self.upper_thresholds

    @property
    def lower_path(self) -> list[float]:
        return [50.0] + self.lower_thresholds

    @property
    def threshold_label(self) -> str:
        pairs = []
        for lower, upper in zip(self.lower_thresholds, self.upper_thresholds):
            pairs.append(f"{format_probability(lower)}/{format_probability(upper)}")
        if pairs:
            return "50, " + ", ".join(pairs)
        return "50"


EXPERIMENTS = [
    ExperimentConfig(
        order=1,
        experiment_id="exp_01_50",
        condition="50 only",
        upper_thresholds=[],
        lower_thresholds=[],
    ),
    ExperimentConfig(
        order=2,
        experiment_id="exp_02_25_75",
        condition="50 plus 25/75 buckets",
        upper_thresholds=[75.0],
        lower_thresholds=[25.0],
    ),
    ExperimentConfig(
        order=3,
        experiment_id="exp_03_10_90",
        condition="50 plus 25/75 and 10/90 buckets",
        upper_thresholds=[75.0, 90.0],
        lower_thresholds=[25.0, 10.0],
    ),
    ExperimentConfig(
        order=4,
        experiment_id="exp_04_5_95",
        condition="50 plus 25/75, 10/90, and 5/95 buckets",
        upper_thresholds=[75.0, 90.0, 95.0],
        lower_thresholds=[25.0, 10.0, 5.0],
    ),
    ExperimentConfig(
        order=5,
        experiment_id="exp_05_1_99",
        condition="50 plus 25/75, 10/90, 5/95, and 1/99 buckets",
        upper_thresholds=[75.0, 90.0, 95.0, 99.0],
        lower_thresholds=[25.0, 10.0, 5.0, 1.0],
    ),
]


def format_probability(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def get_model_response(
    prompt: str,
    model: str,
    timeout_seconds: int,
) -> str:
    try:
        result = subprocess.run(
            ["codex", "exec", "-m", model, prompt],
            input="",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Codex timed out after {timeout_seconds} seconds.") from exc

    if result.returncode != 0:
        raise RuntimeError(
            "Codex command failed.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return strip_code_fence(result.stdout)


def parse_direction(text: str) -> tuple[str, str]:
    direction_match = re.search(r"Direction:\s*(Higher|Lower)\b", text, re.IGNORECASE)
    reason_match = re.search(r"Reasoning:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    if not direction_match:
        raise ValueError(f"Could not parse higher/lower direction from:\n{text}")

    direction = direction_match.group(1).lower()
    reasoning = reason_match.group(1).strip() if reason_match else ""
    return direction, reasoning


def parse_probability(text: str, lower_bound: float, upper_bound: float) -> tuple[float, str]:
    prob_match = re.search(r"Probability:\s*([0-9]+(?:\.[0-9]+)?)", text)
    reason_match = re.search(r"Reasoning:\s*(.*)", text, re.IGNORECASE | re.DOTALL)

    if not prob_match:
        raise ValueError(f"Could not parse probability from:\n{text}")

    probability = float(prob_match.group(1))
    if probability < 0 or probability > 100:
        raise ValueError(f"Probability out of 0-100 range: {probability}")

    tolerance = 1e-9
    if probability < lower_bound - tolerance or probability > upper_bound + tolerance:
        raise ValueError(
            "Probability is outside the discovered interval: "
            f"{probability} not in [{lower_bound}, {upper_bound}]"
        )

    reasoning = reason_match.group(1).strip() if reason_match else ""
    return probability, reasoning


def build_threshold_prompt(
    question: str,
    forecast_date: str,
    threshold: float,
    lower_bound: float,
    upper_bound: float,
    trace: list[dict[str, Any]],
) -> str:
    prior_checks = format_trace_for_prompt(trace)
    threshold_text = format_probability(threshold)
    lower_text = format_probability(lower_bound)
    upper_text = format_probability(upper_bound)

    return f"""
You are making a calibrated probabilistic forecast.

Pretend today's date is {forecast_date}. Use only information available on or before this date.

Question: {question}

Current probability interval from previous checks: {lower_text}% to {upper_text}%.

Previous threshold checks:
{prior_checks}

Decide whether the probability that the question resolves YES is higher or lower than {threshold_text}%.

Return exactly:
Direction: <Higher or Lower>
Reasoning: <one paragraph>

Do not return a final probability yet. Do not include any other text.
""".strip()


def build_final_probability_prompt(
    question: str,
    forecast_date: str,
    lower_bound: float,
    upper_bound: float,
    trace: list[dict[str, Any]],
) -> str:
    prior_checks = format_trace_for_prompt(trace)
    lower_text = format_probability(lower_bound)
    upper_text = format_probability(upper_bound)

    return f"""
You are making a calibrated probabilistic forecast.

Pretend today's date is {forecast_date}. Use only information available on or before this date.

Question: {question}

The threshold checks place the probability that this resolves YES between {lower_text}% and {upper_text}%.

Threshold checks:
{prior_checks}

Estimate the final probability within this interval. Report the probability as a 0-to-100 number with no percent sign. For example, write 63 for 63%, not 0.63 and not 63%.

Return exactly:
Probability: <number from {lower_text} to {upper_text}, no percent sign>
Reasoning: <one paragraph>

Do not include any other text.
""".strip()


def format_trace_for_prompt(trace: list[dict[str, Any]]) -> str:
    if not trace:
        return "- None yet."

    lines = []
    for item in trace:
        threshold = format_probability(float(item["threshold"]))
        direction = str(item["direction"]).capitalize()
        reasoning = str(item.get("reasoning", "")).strip()
        if reasoning:
            lines.append(f"- {threshold}%: {direction}. Reasoning: {reasoning}")
        else:
            lines.append(f"- {threshold}%: {direction}.")
    return "\n".join(lines)


def load_or_create_sample(
    train_path: Path,
    sample_path: Path,
    sample_size: int,
    sample_seed: int,
    resample: bool,
) -> pd.DataFrame:
    if sample_path.exists() and not resample:
        return pd.read_csv(sample_path)

    df = pd.read_csv(train_path)
    if len(df) < sample_size:
        raise ValueError(
            f"Cannot sample {sample_size} rows from {train_path}; only {len(df)} rows exist."
        )

    sample_df = (
        df.sample(n=sample_size, random_state=sample_seed)
        .reset_index(drop=True)
        .copy()
    )
    sample_df.insert(0, "sample_index", range(1, sample_size + 1))
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    return sample_df


def write_experiment_conditions(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for config in EXPERIMENTS:
        rows.append(
            {
                "experiment_order": config.order,
                "experiment_id": config.experiment_id,
                "controlled_variable_changed": "maximum symmetric bucket depth",
                "condition": config.condition,
                "threshold_label": config.threshold_label,
                "upper_path": " -> ".join(format_probability(v) for v in config.upper_path),
                "lower_path": " -> ".join(format_probability(v) for v in config.lower_path),
                "fixed_sample_size": DEFAULT_SAMPLE_SIZE,
                "fixed_model": DEFAULT_CODEX_MODEL,
                "fixed_metric": "Brier score",
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "experiment_conditions.csv", index=False)


def choose_experiments(selected_ids: list[str] | None) -> list[ExperimentConfig]:
    if not selected_ids:
        return EXPERIMENTS

    experiments_by_id = {config.experiment_id: config for config in EXPERIMENTS}
    unknown = [experiment_id for experiment_id in selected_ids if experiment_id not in experiments_by_id]
    if unknown:
        known = ", ".join(experiments_by_id)
        raise ValueError(f"Unknown experiment id(s): {unknown}. Known ids: {known}")

    return [experiments_by_id[experiment_id] for experiment_id in selected_ids]


def ask_threshold(
    row: pd.Series,
    threshold: float,
    lower_bound: float,
    upper_bound: float,
    trace: list[dict[str, Any]],
    model: str,
    timeout_seconds: int,
    mock_codex: bool,
) -> dict[str, Any]:
    prompt = build_threshold_prompt(
        question=row["Question"],
        forecast_date=row["forecast_date_formatted"],
        threshold=threshold,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trace=trace,
    )

    if mock_codex:
        community_probability = float(row["community_prob"]) * 100.0
        direction = "higher" if community_probability >= threshold else "lower"
        raw_output = (
            f"Direction: {direction.capitalize()}\n"
            "Reasoning: Mock response based on the row's community probability."
        )
        reasoning = "Mock response based on the row's community probability."
    else:
        raw_output = get_model_response(prompt, model=model, timeout_seconds=timeout_seconds)
        direction, reasoning = parse_direction(raw_output)

    return {
        "threshold": threshold,
        "direction": direction,
        "reasoning": reasoning,
        "raw_output": raw_output,
        "prompt": prompt,
        "lower_bound_before": lower_bound,
        "upper_bound_before": upper_bound,
    }


def ask_final_probability(
    row: pd.Series,
    lower_bound: float,
    upper_bound: float,
    trace: list[dict[str, Any]],
    model: str,
    timeout_seconds: int,
    mock_codex: bool,
) -> dict[str, Any]:
    prompt = build_final_probability_prompt(
        question=row["Question"],
        forecast_date=row["forecast_date_formatted"],
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trace=trace,
    )

    if mock_codex:
        community_probability = float(row["community_prob"]) * 100.0
        probability = min(max(community_probability, lower_bound), upper_bound)
        raw_output = (
            f"Probability: {probability:.2f}\n"
            "Reasoning: Mock response clipped to the discovered interval."
        )
        reasoning = "Mock response clipped to the discovered interval."
    else:
        raw_output = get_model_response(prompt, model=model, timeout_seconds=timeout_seconds)
        probability, reasoning = parse_probability(raw_output, lower_bound, upper_bound)

    return {
        "probability_percent": probability,
        "reasoning": reasoning,
        "raw_output": raw_output,
        "prompt": prompt,
    }


def discover_interval(
    row: pd.Series,
    config: ExperimentConfig,
    model: str,
    timeout_seconds: int,
    mock_codex: bool,
) -> tuple[float, float, list[dict[str, Any]]]:
    lower_bound = 0.0
    upper_bound = 100.0
    trace: list[dict[str, Any]] = []

    first_decision = ask_threshold(
        row=row,
        threshold=50.0,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        trace=trace,
        model=model,
        timeout_seconds=timeout_seconds,
        mock_codex=mock_codex,
    )
    trace.append(first_decision)

    if first_decision["direction"] == "higher":
        lower_bound = 50.0
        first_decision["lower_bound_after"] = lower_bound
        first_decision["upper_bound_after"] = upper_bound

        for threshold in config.upper_thresholds:
            decision = ask_threshold(
                row=row,
                threshold=threshold,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                trace=trace,
                model=model,
                timeout_seconds=timeout_seconds,
                mock_codex=mock_codex,
            )
            trace.append(decision)

            if decision["direction"] == "higher":
                lower_bound = threshold
                decision["lower_bound_after"] = lower_bound
                decision["upper_bound_after"] = upper_bound
            else:
                upper_bound = threshold
                decision["lower_bound_after"] = lower_bound
                decision["upper_bound_after"] = upper_bound
                break
    else:
        upper_bound = 50.0
        first_decision["lower_bound_after"] = lower_bound
        first_decision["upper_bound_after"] = upper_bound

        for threshold in config.lower_thresholds:
            decision = ask_threshold(
                row=row,
                threshold=threshold,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                trace=trace,
                model=model,
                timeout_seconds=timeout_seconds,
                mock_codex=mock_codex,
            )
            trace.append(decision)

            if decision["direction"] == "lower":
                upper_bound = threshold
                decision["lower_bound_after"] = lower_bound
                decision["upper_bound_after"] = upper_bound
            else:
                lower_bound = threshold
                decision["lower_bound_after"] = lower_bound
                decision["upper_bound_after"] = upper_bound
                break

    return lower_bound, upper_bound, trace


def prediction_record(
    row: pd.Series,
    config: ExperimentConfig,
    model: str,
    timeout_seconds: int,
    mock_codex: bool,
) -> dict[str, Any]:
    record = row.to_dict()
    record.update(
        {
            "experiment_id": config.experiment_id,
            "experiment_order": config.order,
            "experiment_condition": config.condition,
            "threshold_label": config.threshold_label,
            "llm_prob": None,
            "llm_probability_percent": None,
            "llm_reasoning": "",
            "raw_model_output": "",
            "threshold_trace_json": "[]",
            "final_interval_lower": None,
            "final_interval_upper": None,
            "threshold_calls": 0,
            "parse_status": "failed",
            "error": "",
        }
    )

    try:
        lower_bound, upper_bound, trace = discover_interval(
            row=row,
            config=config,
            model=model,
            timeout_seconds=timeout_seconds,
            mock_codex=mock_codex,
        )
        final_response = ask_final_probability(
            row=row,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            trace=trace,
            model=model,
            timeout_seconds=timeout_seconds,
            mock_codex=mock_codex,
        )
        probability_percent = float(final_response["probability_percent"])

        record.update(
            {
                "llm_prob": probability_percent / 100.0,
                "llm_probability_percent": probability_percent,
                "llm_reasoning": final_response["reasoning"],
                "raw_model_output": final_response["raw_output"],
                "threshold_trace_json": json.dumps(trace, default=json_default),
                "final_interval_lower": lower_bound,
                "final_interval_upper": upper_bound,
                "threshold_calls": len(trace),
                "parse_status": "ok",
            }
        )
    except Exception as exc:
        record["error"] = str(exc)

    return record


def run_experiment(
    sample_df: pd.DataFrame,
    config: ExperimentConfig,
    output_dir: Path,
    model: str,
    timeout_seconds: int,
    force: bool,
    mock_codex: bool,
) -> dict[str, Any]:
    run_dir = output_dir / "runs" / config.experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / f"experiment_{config.order}.csv"
    metrics_path = run_dir / "metrics.json"
    condition_path = run_dir / "condition.json"

    with open(condition_path, "w") as f:
        json.dump(asdict(config), f, indent=2, default=json_default)

    records: list[dict[str, Any]] = []
    completed_sample_ids: set[str] = set()
    if predictions_path.exists() and not force:
        existing_df = pd.read_csv(predictions_path)
        records, completed_sample_ids = reusable_existing_records(
            existing_df=existing_df,
            sample_df=sample_df,
        )
        if len(completed_sample_ids) >= len(sample_df):
            print(f"Skipping {config.experiment_id}; experiment_{config.order}.csv is complete.")
            return score_and_write_metrics(
                predictions_df=pd.DataFrame(records),
                config=config,
                run_dir=run_dir,
                predictions_path=predictions_path,
                metrics_path=metrics_path,
                model=model,
                mock_codex=mock_codex,
            )
        print(
            f"Resuming {config.experiment_id}; "
            f"{len(completed_sample_ids)} / {len(sample_df)} sampled rows already saved."
        )

    total = len(sample_df)
    for index, row in sample_df.iterrows():
        sample_id = str(row["sample_index"])
        if sample_id in completed_sample_ids:
            continue

        print(
            f"[{config.experiment_id}] Processing sampled row {index + 1} / {total}",
            flush=True,
        )
        records.append(
            prediction_record(
                row=row,
                config=config,
                model=model,
                timeout_seconds=timeout_seconds,
                mock_codex=mock_codex,
            )
        )
        pd.DataFrame(records).to_csv(predictions_path, index=False)

    predictions_df = pd.DataFrame(records)
    return score_and_write_metrics(
        predictions_df=predictions_df,
        config=config,
        run_dir=run_dir,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        model=model,
        mock_codex=mock_codex,
    )


def reusable_existing_records(
    existing_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> tuple[list[dict[str, Any]], set[str]]:
    if "sample_index" not in existing_df.columns:
        return [], set()

    sample_ids = set(sample_df["sample_index"].astype(str))
    existing_df = existing_df.copy()
    existing_df["sample_index_for_resume"] = existing_df["sample_index"].astype(str)
    existing_df = existing_df[existing_df["sample_index_for_resume"].isin(sample_ids)]
    existing_df = existing_df.drop_duplicates("sample_index_for_resume", keep="last")
    completed_sample_ids = set(existing_df["sample_index_for_resume"])
    existing_df = existing_df.drop(columns=["sample_index_for_resume"])
    return existing_df.to_dict("records"), completed_sample_ids


def score_and_write_metrics(
    predictions_df: pd.DataFrame,
    config: ExperimentConfig,
    run_dir: Path,
    predictions_path: Path,
    metrics_path: Path,
    model: str,
    mock_codex: bool,
) -> dict[str, Any]:
    numeric_df = predictions_df.copy()
    numeric_df["llm_prob"] = pd.to_numeric(numeric_df["llm_prob"], errors="coerce")
    numeric_df["resolution_binary"] = pd.to_numeric(
        numeric_df["resolution_binary"], errors="coerce"
    )
    numeric_df["community_prob"] = pd.to_numeric(
        numeric_df["community_prob"], errors="coerce"
    )
    numeric_df["llm_brier_row"] = (
        numeric_df["llm_prob"] - numeric_df["resolution_binary"]
    ) ** 2
    numeric_df["community_brier_row"] = (
        numeric_df["community_prob"] - numeric_df["resolution_binary"]
    ) ** 2

    valid_df = numeric_df[numeric_df["llm_prob"].notna()].copy()
    if len(valid_df) > 0:
        brier = float(valid_df["llm_brier_row"].mean())
        community_brier = float(valid_df["community_brier_row"].mean())
        mean_llm_prob = float(valid_df["llm_prob"].mean())
        mean_threshold_calls = float(valid_df["threshold_calls"].mean())
    else:
        brier = None
        community_brier = None
        mean_llm_prob = None
        mean_threshold_calls = None

    if "sample_index" in numeric_df.columns:
        numeric_df = numeric_df.sort_values("sample_index").reset_index(drop=True)

    failed_predictions = int(numeric_df["llm_prob"].isna().sum())
    metrics = {
        "experiment_order": config.order,
        "experiment_id": config.experiment_id,
        "condition": config.condition,
        "threshold_label": config.threshold_label,
        "upper_path": config.upper_path,
        "lower_path": config.lower_path,
        "model": model,
        "mock_codex": mock_codex,
        "questions_total": int(len(numeric_df)),
        "valid_predictions": int(len(valid_df)),
        "failed_predictions": failed_predictions,
        "brier": brier,
        "community_brier_on_valid_rows": community_brier,
        "mean_llm_prob": mean_llm_prob,
        "mean_threshold_calls": mean_threshold_calls,
        "predictions_path": str(predictions_path),
        "run_dir": str(run_dir),
        "timestamp": datetime.now().isoformat(),
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=json_default)
    numeric_df.to_csv(predictions_path, index=False)

    return metrics


def write_result_matrix(summaries: list[dict[str, Any]], output_dir: Path) -> pd.DataFrame:
    result_df = pd.DataFrame(summaries).sort_values("experiment_order").reset_index(drop=True)
    result_path = output_dir / "result_matrix.csv"
    markdown_path = output_dir / "result_matrix.md"
    brier_scores_path = output_dir / "brier_scores.csv"

    result_df.to_csv(result_path, index=False)
    result_df[
        [
            "experiment_order",
            "experiment_id",
            "threshold_label",
            "valid_predictions",
            "failed_predictions",
            "brier",
        ]
    ].to_csv(brier_scores_path, index=False)
    markdown_path.write_text(build_markdown_table(result_df), encoding="utf-8")
    return result_df


def build_markdown_table(result_df: pd.DataFrame) -> str:
    columns = [
        "experiment_order",
        "experiment_id",
        "threshold_label",
        "valid_predictions",
        "failed_predictions",
        "brier",
        "community_brier_on_valid_rows",
        "mean_threshold_calls",
    ]
    headers = [
        "Order",
        "Experiment",
        "Thresholds",
        "Valid",
        "Failed",
        "Brier",
        "Community Brier",
        "Mean Threshold Calls",
    ]

    lines = [
        "# Experiment-Result Matrix",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in result_df.iterrows():
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                values.append("" if pd.isna(value) else f"{value:.6f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(values) + " |")

    lines.append("")
    return "\n".join(lines)


def write_metric_plot(result_df: pd.DataFrame, output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    plot_df = result_df.copy()
    plot_df["brier"] = pd.to_numeric(plot_df["brier"], errors="coerce")
    plot_df = plot_df[plot_df["brier"].notna()]

    if plot_df.empty:
        raise RuntimeError("No valid Brier scores were available for plotting.")

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        plot_df["experiment_order"],
        plot_df["brier"],
        marker="o",
        linewidth=2,
        label="LLM Brier",
    )

    community_values = pd.to_numeric(
        result_df["community_brier_on_valid_rows"], errors="coerce"
    ).dropna()
    if len(community_values) > 0:
        axis.axhline(
            community_values.mean(),
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="Mean community Brier on valid rows",
        )

    axis.set_title("Brier Score Across Architecture Experiments")
    axis.set_xlabel("Experiment run")
    axis.set_ylabel("Brier score, lower is better")
    axis.set_xticks(plot_df["experiment_order"])
    axis.set_xticklabels(plot_df["experiment_id"], rotation=25, ha="right")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    png_path = output_dir / "metric_over_time.png"
    figure.savefig(png_path, dpi=200)
    plt.close(figure)
    return png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run symmetric probability bucket architecture experiments."
    )
    parser.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    parser.add_argument("--sample-path", type=Path, default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--model", default=DEFAULT_CODEX_MODEL)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_CODEX_TIMEOUT_SECONDS)
    parser.add_argument(
        "--experiments",
        nargs="+",
        help="Experiment ids to run. Defaults to all experiments.",
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Create a new deterministic sample using --sample-seed even if sample_20.csv exists.",
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Create or reuse sample_20.csv, then exit without calling Codex.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even when predictions.csv already exists.",
    )
    parser.add_argument(
        "--mock-codex",
        action="store_true",
        help="Use deterministic local mock responses for testing only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(REPO_ROOT)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = load_or_create_sample(
        train_path=args.train_path,
        sample_path=args.sample_path,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        resample=args.resample,
    )
    print(f"Sample rows: {len(sample_df)}")
    print(f"Sample path: {args.sample_path}")

    write_experiment_conditions(args.output_dir)

    if args.sample_only:
        print("Sample-only mode complete. Codex was not called.")
        return

    selected_experiments = choose_experiments(args.experiments)
    summaries = []
    for config in selected_experiments:
        print(f"\n=== Running {config.experiment_id}: {config.condition} ===", flush=True)
        summaries.append(
            run_experiment(
                sample_df=sample_df,
                config=config,
                output_dir=args.output_dir,
                model=args.model,
                timeout_seconds=args.timeout_seconds,
                force=args.force,
                mock_codex=args.mock_codex,
            )
        )

    result_df = write_result_matrix(summaries, args.output_dir)
    plot_path = write_metric_plot(result_df, args.output_dir)

    print(f"\nResult matrix: {args.output_dir / 'result_matrix.csv'}")
    print(f"Metric plot: {plot_path}")


if __name__ == "__main__":
    main()
