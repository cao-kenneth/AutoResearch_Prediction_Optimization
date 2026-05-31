from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"

RUNS = [
    ("initial_prompt", "train_predictions.csv"),
    ("run_001", "run_001/predictions.csv"),
    ("run_002", "run_002/predictions.csv"),
    ("run_003", "run_003/predictions.csv"),
]


def score_predictions(
    experiment_order: int,
    experiment_id: str,
    predictions_path: Path,
) -> dict[str, Any]:
    df = pd.read_csv(predictions_path)
    numeric_df = df.copy()
    numeric_df["llm_prob"] = pd.to_numeric(numeric_df["llm_prob"], errors="coerce")
    numeric_df["resolution_binary"] = pd.to_numeric(
        numeric_df["resolution_binary"], errors="coerce"
    )
    numeric_df["community_prob"] = pd.to_numeric(
        numeric_df["community_prob"], errors="coerce"
    )

    valid_df = numeric_df[
        numeric_df["llm_prob"].notna() & numeric_df["resolution_binary"].notna()
    ].copy()
    failed_predictions = int(len(numeric_df) - len(valid_df))

    if valid_df.empty:
        brier = None
        community_brier = None
        fifty_fifty_brier = None
    else:
        brier = float(((valid_df["llm_prob"] - valid_df["resolution_binary"]) ** 2).mean())
        community_brier = float(
            ((valid_df["community_prob"] - valid_df["resolution_binary"]) ** 2).mean()
        )
        fifty_fifty_brier = float(((0.5 - valid_df["resolution_binary"]) ** 2).mean())

    return {
        "experiment_order": experiment_order,
        "experiment_id": experiment_id,
        "valid_predictions": int(len(valid_df)),
        "failed_predictions": failed_predictions,
        "brier": brier,
        "community_brier_on_valid_rows": community_brier,
        "fifty_fifty_brier_on_valid_rows": fifty_fifty_brier,
        "predictions_path": str(predictions_path),
        "metrics_path": str(predictions_path.parent / "metrics.json")
        if (predictions_path.parent / "metrics.json").exists()
        else "",
        "timestamp": datetime.now().isoformat(),
    }


def load_result_matrix(output_dir: Path) -> pd.DataFrame:
    rows = []
    for experiment_order, (experiment_id, relative_path) in enumerate(RUNS):
        predictions_path = output_dir / relative_path
        if not predictions_path.exists():
            continue
        rows.append(
            score_predictions(
                experiment_order=experiment_order,
                experiment_id=experiment_id,
                predictions_path=predictions_path,
            )
        )

    if not rows:
        raise RuntimeError(f"No optimize-prompt prediction files found in {output_dir}")

    return pd.DataFrame(rows).sort_values("experiment_order").reset_index(drop=True)


def build_markdown_table(result_df: pd.DataFrame) -> str:
    columns = [
        "experiment_order",
        "experiment_id",
        "valid_predictions",
        "failed_predictions",
        "brier",
        "community_brier_on_valid_rows",
        "fifty_fifty_brier_on_valid_rows",
    ]
    headers = [
        "Order",
        "Run",
        "Valid",
        "Failed",
        "Brier",
        "Community Brier",
        "50/50 Brier",
    ]

    lines = [
        "# Optimize-Prompt Result Matrix",
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


def write_result_files(result_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_dir / "result_matrix.csv", index=False)
    result_df[
        [
            "experiment_order",
            "experiment_id",
            "valid_predictions",
            "failed_predictions",
            "brier",
        ]
    ].to_csv(output_dir / "brier_scores.csv", index=False)
    (output_dir / "result_matrix.md").write_text(
        build_markdown_table(result_df),
        encoding="utf-8",
    )


def write_metric_plot(result_df: pd.DataFrame, output_dir: Path) -> Path:
    matplotlib_config_dir = output_dir / ".matplotlib"
    matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config_dir))

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

    axis.set_title("Brier Score Across Prompt-Optimization Runs")
    axis.set_xlabel("Prompt-optimization run")
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
        description="Create result tables and a Brier-score chart for optimize_prompt outputs."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_df = load_result_matrix(args.output_dir)
    write_result_files(result_df, args.output_dir)
    plot_path = write_metric_plot(result_df, args.output_dir)

    print(json.dumps({"plot_path": str(plot_path)}, indent=2))


if __name__ == "__main__":
    main()
