from __future__ import annotations

import argparse
import asyncio
import os
import time
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import scaffold_optimizer as optimizer
import scaffold_runner as runner


DEFAULT_OUTPUT_DIR = runner.REPO_ROOT / "outputs" / "final_test_sensitivity"
TEST_PATH = runner.REPO_ROOT / "data" / "test_40_unused.csv"
SELECTED_OPTIMIZER_CONFIG = "p1_sk_pm0_agg_mean_ref_none"
RUNNER_CONFIGS = [
    "baseline_one_shot",
    "premortem_one_shot",
    "self_critique",
    "three_perspectives_judge",
]


def temperature_label(value: float) -> str:
    return str(value).replace(".", "_")


def parse_probability(value: Any) -> float:
    if pd.isna(value):
        raise ValueError("Missing probability.")
    text = str(value).strip()
    if text.endswith("%"):
        return float(text[:-1]) / 100
    probability = float(text)
    if probability > 1:
        probability /= 100
    return probability


def load_test_split(limit: int | None) -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_PATH}")
    df = pd.read_csv(TEST_PATH)
    df["forecast_date_formatted"] = pd.to_datetime(
        df["Forecast_Date"],
        format="%m/%d/%y",
    ).dt.strftime("%B %d, %Y")
    df["resolution_binary"] = (
        df["Resolution"].str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
    )
    df["community_prob"] = df["ForecastDate_Probability"].apply(parse_probability)
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def fixed_baseline_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        question_id = runner.safe_id(row.get("Question ID"), f"row_{int(row.name)}")
        outcome = int(row["resolution_binary"])
        for config, probability, reasoning in [
            ("community", float(row["community_prob"]), "Community forecast from source data."),
            ("guess_50_50", 0.5, "Fixed 50/50 baseline."),
        ]:
            rows.append(
                {
                    "split": "test",
                    "temperature": "fixed",
                    "config": config,
                    "question_id": question_id,
                    "question": row["Question"],
                    "forecast_date": runner.forecast_date(row),
                    "resolution": row["Resolution"],
                    "resolution_binary": outcome,
                    "final_probability": probability,
                    "final_reasoning": reasoning,
                    "brier": (probability - outcome) ** 2,
                    "status": "ok",
                    "error": "",
                    "calls_per_question": 0,
                    "description": reasoning,
                    "stage_paths": "{}",
                    "stage_outputs": "{}",
                }
            )
    return rows


def make_client(args: argparse.Namespace) -> runner.ResponsesClient | None:
    runner.load_dotenv(runner.REPO_ROOT / ".env")
    if args.mock:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment or .env, or use --mock.")
    return runner.ResponsesClient(
        api_key,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
    )


def make_stage_args(args: argparse.Namespace, temperature: float) -> Namespace:
    label = temperature_label(temperature)
    return Namespace(
        output_dir=args.output_dir / f"temp_{label}",
        model=args.model,
        temperature=temperature,
        max_output_tokens=args.max_output_tokens,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        force=args.force,
        mock=args.mock,
        verbose=args.verbose,
    )


async def evaluate_temperature(
    *,
    client: runner.ResponsesClient | None,
    args: argparse.Namespace,
    df: pd.DataFrame,
    temperature: float,
) -> list[dict[str, Any]]:
    stage_args = make_stage_args(args, temperature)
    temp_text = str(temperature)
    selected = optimizer.CANDIDATES[SELECTED_OPTIMIZER_CONFIG]
    tasks = []

    for _, row in df.iterrows():
        for name in RUNNER_CONFIGS:
            tasks.append(
                runner.run_config_for_row(
                    client=client,
                    args=stage_args,
                    row=row,
                    split="test",
                    config=runner.CONFIGS[name],
                )
            )
        tasks.append(
            optimizer.run_candidate_for_row(
                client=client,
                args=stage_args,
                row=row,
                split="test",
                candidate=selected,
            )
        )

    results: list[dict[str, Any]] = []
    for index, task in enumerate(asyncio.as_completed(tasks), start=1):
        result = await task
        result["temperature"] = temp_text
        if result["config"] == SELECTED_OPTIMIZER_CONFIG:
            result["description"] = selected.description
        results.append(result)
        if args.verbose:
            print(
                f"[temp={temperature} {index}/{len(tasks)}] "
                f"{result['config']} {result['question_id']} {result['status']}",
                flush=True,
            )
    return results


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (temperature, config), group in predictions.groupby(["temperature", "config"], sort=False):
        valid = group[group["status"].eq("ok") & group["brier"].notna()]
        rows.append(
            {
                "split": "test",
                "temperature": temperature,
                "config": config,
                "rows": len(group),
                "valid_rows": len(valid),
                "failures": len(group) - len(valid),
                "brier": valid["brier"].mean() if len(valid) else None,
                "mean_probability": valid["final_probability"].mean() if len(valid) else None,
                "calls_per_question": group["calls_per_question"].iloc[0]
                if "calls_per_question" in group.columns and len(group)
                else None,
                "description": group["description"].iloc[0]
                if "description" in group.columns and len(group)
                else "",
            }
        )
    summary = pd.DataFrame(rows)
    order = {
        "community": 0,
        "guess_50_50": 1,
        "baseline_one_shot": 2,
        "premortem_one_shot": 3,
        "self_critique": 4,
        "three_perspectives_judge": 5,
        SELECTED_OPTIMIZER_CONFIG: 6,
    }
    temp_order = {"fixed": -1, "0.0": 0, "0": 0, "0.7": 1}
    summary["config_order"] = summary["config"].map(order).fillna(99)
    summary["temp_order"] = summary["temperature"].map(temp_order).fillna(50)
    return summary.sort_values(["temp_order", "config_order"]).drop(
        columns=["temp_order", "config_order"]
    )


def write_chart(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    chart_data = summary[summary["brier"].notna()].copy()
    if chart_data.empty:
        return None
    mpl_config_dir = output_dir.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    chart_data["label"] = chart_data.apply(
        lambda row: row["config"]
        if row["temperature"] == "fixed"
        else f"{row['config']}\\ntemp={row['temperature']}",
        axis=1,
    )
    chart_data = chart_data.sort_values("brier", ascending=True)
    colors = [
        "#9aa0a6"
        if temperature == "fixed"
        else ("#4c78a8" if temperature in {"0", "0.0"} else "#f58518")
        for temperature in chart_data["temperature"]
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(chart_data["label"], chart_data["brier"], color=colors)
    ax.set_ylabel("Test Brier score")
    ax.set_title("Final Test Brier with Temperature Sensitivity")
    ax.set_ylim(0, max(chart_data["brier"]) * 1.25)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, chart_data["brier"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    path = output_dir / "test_sensitivity_brier.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(summary: pd.DataFrame, output_dir: Path, temperatures: list[float], mock: bool) -> Path:
    best = summary[summary["brier"].notna()].sort_values("brier").iloc[0]
    lines = [
        "# Final Test Sensitivity Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Mode: {'mock' if mock else 'OpenAI API'}",
        f"Temperatures tested for LLM configs: {', '.join(str(t) for t in temperatures)}",
        "",
        "This is a post-hoc sensitivity analysis and was not used to select the final scaffold.",
        "",
        (
            f"Best test result in this analysis: `{best['config']}` at temperature "
            f"`{best['temperature']}` with Brier {best['brier']:.6f}."
        ),
        "",
        "| Temperature | Config | Valid rows | Failures | Brier | Calls/question |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        brier = "" if pd.isna(row["brier"]) else f"{row['brier']:.6f}"
        calls = "" if pd.isna(row["calls_per_question"]) else int(row["calls_per_question"])
        lines.append(
            f"| {row['temperature']} | `{row['config']}` | {int(row['valid_rows'])} | "
            f"{int(row['failures'])} | {brier} | {calls} |"
        )

    path = output_dir / "test_sensitivity_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_outputs(
    *,
    records: list[dict[str, Any]],
    output_dir: Path,
    temperatures: list[float],
    mock: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(records)
    summary = summarize(predictions)
    predictions_path = output_dir / "test_sensitivity_predictions.csv"
    results_path = output_dir / "test_sensitivity_results.csv"
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(results_path, index=False)
    chart_path = write_chart(summary, output_dir)
    report_path = write_report(summary, output_dir, temperatures, mock)

    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote results: {results_path}")
    print(f"Wrote report: {report_path}")
    if chart_path:
        print(f"Wrote chart: {chart_path}")
    else:
        print("Chart not written; no validation results or matplotlib is unavailable.")


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_test_split(args.limit)
    client = make_client(args)
    records = fixed_baseline_rows(df)

    started = time.time()
    for temperature in args.temperatures:
        records.extend(
            await evaluate_temperature(
                client=client,
                args=args,
                df=df,
                temperature=temperature,
            )
        )
    if args.verbose:
        print(f"Completed final sensitivity run in {time.time() - started:.1f}s")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run final held-out test plus post-hoc temperature sensitivity."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
    parser.add_argument("--model", default=runner.DEFAULT_MODEL)
    parser.add_argument("--max-output-tokens", type=int, default=runner.DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=runner.DEFAULT_RETRIES)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Ignore cached raw responses and rerun calls.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses; no API calls.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    records = asyncio.run(run(args))
    write_outputs(
        records=records,
        output_dir=args.output_dir,
        temperatures=args.temperatures,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
