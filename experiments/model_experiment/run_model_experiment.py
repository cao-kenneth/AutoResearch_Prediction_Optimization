from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scaffold_optimizer as optimizer
import scaffold_runner as runner


DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "outputs"
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 1400
DEFAULT_CONCURRENCY = 1
DEFAULT_CONFIG = "p1_sk_pm0_agg_mean_ref_none"
DEFAULT_SPLIT = "test"
TEST_PATH = REPO_ROOT / "data" / "test_40_unused.csv"

WEB_SEARCH_SYSTEM_PROMPT = runner.SYSTEM_PROMPT.replace(
    "Do not browse, call tools, or use post-forecast-date information.",
    (
        "You may use web search, but only to find information that would have "
        "been available on or before the forecast date. Ignore and do not use "
        "any source, snippet, title, or known fact after the forecast date, "
        "including actual resolution reporting. If a source is undated or "
        "appears to include later information, treat it cautiously or ignore it."
    ),
)


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


def load_split(
    split: str,
    limit: int | None,
    only_question_ids: list[str] | None = None,
) -> pd.DataFrame:
    if split == "test":
        path = TEST_PATH
    else:
        path = REPO_ROOT / "data" / f"{split}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_csv(path)
    if "forecast_date_formatted" not in df.columns:
        try:
            forecast_dates = pd.to_datetime(df["Forecast_Date"], format="%m/%d/%y")
        except ValueError:
            forecast_dates = pd.to_datetime(df["Forecast_Date"])
        df["forecast_date_formatted"] = forecast_dates.dt.strftime("%B %d, %Y")
    if "resolution_binary" not in df.columns:
        df["resolution_binary"] = (
            df["Resolution"].str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)
        )
    if "community_prob" not in df.columns and "ForecastDate_Probability" in df.columns:
        df["community_prob"] = df["ForecastDate_Probability"].apply(parse_probability)
    if "question_id_safe" not in df.columns:
        df["question_id_safe"] = [
            runner.safe_id(row.get("Question ID"), f"row_{index}")
            for index, row in df.iterrows()
        ]
    if only_question_ids:
        selected_ids = set(only_question_ids)
        df = df[df["question_id_safe"].astype(str).isin(selected_ids)]
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


def web_search_tool(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "type": "web_search",
        "search_context_size": args.search_context_size,
        "external_web_access": True,
        "user_location": {
            "type": "approximate",
            "country": "US",
            "timezone": "America/Chicago",
        },
    }


def build_web_search_payload(
    *,
    args: argparse.Namespace,
    schema_name: str,
    schema: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    payload = runner.build_payload(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        schema_name=schema_name,
        schema=schema,
        prompt=prompt,
    )
    if not args.send_temperature:
        payload.pop("temperature", None)
    payload["input"][0]["content"] = WEB_SEARCH_SYSTEM_PROMPT
    payload["tools"] = [web_search_tool(args)]
    payload["tool_choice"] = args.tool_choice
    payload["max_tool_calls"] = args.max_tool_calls
    if args.reasoning_effort:
        payload["reasoning"] = {"effort": args.reasoning_effort}
    return payload


def partial_response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for output in response.get("output", []):
        if output.get("type") != "message":
            continue
        for item in output.get("content", []):
            if item.get("type") == "output_text":
                parts.append(item.get("text", ""))
    return "".join(parts).strip()


def parse_response_with_probability_fallback(response: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    try:
        return runner.parse_response_json(response), None
    except Exception as exc:
        text = partial_response_text(response)
        match = re.search(r'"probability"\s*:\s*([0-9]*\.?[0-9]+)', text)
        if not match:
            raise
        probability = runner.validate_probability(match.group(1))
        return (
            {
                "probability": probability,
                "reasoning": (
                    "Recovered probability from incomplete structured output. "
                    f"Original parse error: {exc}"
                ),
            },
            str(exc),
        )


def cache_matches_args(cached: dict[str, Any], args: argparse.Namespace) -> bool:
    if bool(cached.get("mock")) != bool(args.mock):
        return False
    if cached.get("parsed") is None:
        return False

    request = cached.get("request", {})
    if request.get("model") != args.model:
        return False
    cached_max_output_tokens = request.get("max_output_tokens")
    if cached_max_output_tokens is None or cached_max_output_tokens > args.max_output_tokens:
        return False
    if cached.get("recovered_from_incomplete_output") and cached_max_output_tokens < args.max_output_tokens:
        return False
    if request.get("max_tool_calls") != args.max_tool_calls:
        return False
    if request.get("tool_choice") != args.tool_choice:
        return False
    if request.get("tools") != [web_search_tool(args)]:
        return False
    if args.send_temperature:
        return request.get("temperature") == args.temperature
    return "temperature" not in request


def extract_web_search_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    calls = []
    for output in response.get("output", []):
        if output.get("type") == "web_search_call":
            calls.append(
                {
                    "id": output.get("id"),
                    "status": output.get("status"),
                    "action": output.get("action"),
                }
            )
    return calls


async def call_web_search_stage(
    *,
    client: runner.ResponsesClient | None,
    args: argparse.Namespace,
    cache_path: Path,
    prompt: str,
) -> dict[str, Any]:
    if cache_path.exists() and not args.force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_matches_args(cached, args):
            return cached

    payload = build_web_search_payload(
        args=args,
        schema_name="perspective_forecast",
        schema=runner.FORECAST_SCHEMA,
        prompt=prompt,
    )

    started_at = datetime.now().isoformat()
    parse_error = None
    recovered_from_incomplete_output = False
    if args.mock:
        parsed = runner.mock_parsed_response("perspective_forecast", prompt)
        response = {
            "id": f"mock_{hashlib.sha1(prompt.encode('utf-8')).hexdigest()[:12]}",
            "model": args.model,
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": json.dumps(parsed)}],
                }
            ],
        }
    else:
        if client is None:
            raise RuntimeError("OpenAI client is required unless --mock is set.")
        response = await client.create(payload)
        try:
            parsed, parse_error = parse_response_with_probability_fallback(response)
        except Exception as exc:
            runner.atomic_write_json(
                cache_path,
                {
                    "mock": args.mock,
                    "schema_name": "perspective_forecast",
                    "request": payload,
                    "response": response,
                    "parsed": None,
                    "parse_error": str(exc),
                    "web_search_calls": extract_web_search_calls(response),
                    "started_at": started_at,
                    "completed_at": datetime.now().isoformat(),
                },
            )
            raise
        recovered_from_incomplete_output = parse_error is not None

    record = {
        "mock": args.mock,
        "schema_name": "perspective_forecast",
        "request": payload,
        "response": response,
        "parsed": parsed,
        "parse_error": parse_error,
        "recovered_from_incomplete_output": recovered_from_incomplete_output,
        "web_search_calls": extract_web_search_calls(response),
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
    }
    runner.atomic_write_json(cache_path, record)
    return record


async def run_row(
    *,
    client: runner.ResponsesClient | None,
    args: argparse.Namespace,
    row: pd.Series,
    split: str,
    candidate: optimizer.CandidateConfig,
) -> dict[str, Any]:
    question_id = runner.safe_id(row.get("Question ID"), f"row_{int(row.name)}")
    cache_dir = args.output_dir / "raw" / split / candidate.name / question_id
    cache_path = cache_dir / "skeptic_web_search.json"

    try:
        prompt = optimizer.perspective_prompt(row, candidate, "skeptic")
        stage = await call_web_search_stage(
            client=client,
            args=args,
            cache_path=cache_path,
            prompt=prompt,
        )
        final_probability = runner.validate_probability(stage["parsed"]["probability"])
        outcome = int(row["resolution_binary"])
        brier = (final_probability - outcome) ** 2
        status = "ok"
        error = ""
        final_reasoning = stage["parsed"]["reasoning"]
        web_search_calls = len(stage.get("web_search_calls", []))

    except Exception as exc:
        final_probability = None
        final_reasoning = ""
        brier = None
        status = "failed"
        error = str(exc)
        web_search_calls = 0

    return {
        "split": split,
        "model": args.model,
        "temperature": args.temperature if args.send_temperature else "omitted",
        "requested_temperature": args.temperature,
        "tool": "web_search",
        "tool_choice": args.tool_choice,
        "config": candidate.name,
        "question_id": question_id,
        "question": row["Question"],
        "forecast_date": runner.forecast_date(row),
        "resolution": row["Resolution"],
        "resolution_binary": row["resolution_binary"],
        "community_prob": row.get("community_prob"),
        "final_probability": final_probability,
        "final_reasoning": final_reasoning,
        "brier": brier,
        "status": status,
        "error": error,
        "calls_per_question": candidate.calls_per_question,
        "web_search_calls": web_search_calls,
        "description": candidate.description,
        "stage_paths": json.dumps({"skeptic": str(cache_path)}, ensure_ascii=False),
        "stage_outputs": json.dumps({}, ensure_ascii=False),
    }


def fixed_baseline_rows(df: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    rows = []
    for _, row in df.iterrows():
        question_id = runner.safe_id(row.get("Question ID"), f"row_{int(row.name)}")
        outcome = int(row["resolution_binary"])
        baselines = [("guess_50_50", 0.5, "Fixed 50/50 baseline.")]
        if "community_prob" in row and not pd.isna(row["community_prob"]):
            baselines.insert(
                0,
                ("community", float(row["community_prob"]), "Community forecast from source data."),
            )
        for config, probability, reasoning in baselines:
            rows.append(
                {
                    "split": split,
                    "model": "fixed",
                    "temperature": "fixed",
                    "requested_temperature": "fixed",
                    "tool": "none",
                    "tool_choice": "none",
                    "config": config,
                    "question_id": question_id,
                    "question": row["Question"],
                    "forecast_date": runner.forecast_date(row),
                    "resolution": row["Resolution"],
                    "resolution_binary": outcome,
                    "community_prob": row.get("community_prob"),
                    "final_probability": probability,
                    "final_reasoning": reasoning,
                    "brier": (probability - outcome) ** 2,
                    "status": "ok",
                    "error": "",
                    "calls_per_question": 0,
                    "web_search_calls": 0,
                    "description": reasoning,
                    "stage_paths": "{}",
                    "stage_outputs": "{}",
                }
            )
    return rows


def make_client(args: argparse.Namespace) -> runner.ResponsesClient | None:
    runner.load_dotenv(REPO_ROOT / ".env")
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


async def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = load_split(args.split, args.limit, args.only_question_ids)
    candidate = optimizer.CANDIDATES[args.config]

    if args.rebuild_from_cache:
        return rebuild_from_cache(args=args, df=df, split=args.split, candidate=candidate)

    client = make_client(args)

    tasks = [
        run_row(
            client=client,
            args=args,
            row=row,
            split=args.split,
            candidate=candidate,
        )
        for _, row in df.iterrows()
    ]

    started = time.time()
    records = fixed_baseline_rows(df, args.split)
    for index, task in enumerate(asyncio.as_completed(tasks), start=1):
        result = await task
        records.append(result)
        if args.verbose:
            print(
                f"[{index}/{len(tasks)}] {result['config']} "
                f"{result['question_id']} {result['status']}",
                flush=True,
            )
    if args.verbose:
        print(f"Completed {len(tasks)} web-search rows in {time.time() - started:.1f}s")

    if args.only_question_ids:
        if args.verbose:
            print("Rebuilding full output tables from cache after targeted retry.")
        full_df = load_split(args.split, None, None)
        return rebuild_from_cache(
            args=args,
            df=full_df,
            split=args.split,
            candidate=candidate,
        )

    return records


def rebuild_from_cache(
    *,
    args: argparse.Namespace,
    df: pd.DataFrame,
    split: str,
    candidate: optimizer.CandidateConfig,
) -> list[dict[str, Any]]:
    records = fixed_baseline_rows(df, split)

    for _, row in df.iterrows():
        question_id = runner.safe_id(row.get("Question ID"), f"row_{int(row.name)}")
        cache_path = (
            args.output_dir
            / "raw"
            / split
            / candidate.name
            / question_id
            / "skeptic_web_search.json"
        )
        outcome = int(row["resolution_binary"])
        final_probability = None
        final_reasoning = ""
        brier = None
        status = "failed"
        error = ""
        web_search_calls = 0
        stage_outputs: dict[str, Any] = {}

        if not cache_path.exists():
            error = "Missing cached response."
        else:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            web_search_calls = len(cached.get("web_search_calls", []))
            if cached.get("mock"):
                error = "Cached response is mock; real API response is not available."
            elif cached.get("parsed") is None:
                error = str(cached.get("parse_error") or "Cached response has no parsed output.")
            else:
                parsed = cached["parsed"]
                final_probability = runner.validate_probability(parsed["probability"])
                final_reasoning = parsed["reasoning"]
                brier = (final_probability - outcome) ** 2
                status = "ok"
                stage_outputs = {
                    "skeptic": parsed,
                    "recovered_from_incomplete_output": cached.get(
                        "recovered_from_incomplete_output", False
                    ),
                }

        records.append(
            {
                "split": split,
                "model": args.model,
                "temperature": args.temperature if args.send_temperature else "omitted",
                "requested_temperature": args.temperature,
                "tool": "web_search",
                "tool_choice": args.tool_choice,
                "config": candidate.name,
                "question_id": question_id,
                "question": row["Question"],
                "forecast_date": runner.forecast_date(row),
                "resolution": row["Resolution"],
                "resolution_binary": row["resolution_binary"],
                "community_prob": row.get("community_prob"),
                "final_probability": final_probability,
                "final_reasoning": final_reasoning,
                "brier": brier,
                "status": status,
                "error": error,
                "calls_per_question": candidate.calls_per_question,
                "web_search_calls": web_search_calls,
                "description": candidate.description,
                "stage_paths": json.dumps({"skeptic": str(cache_path)}, ensure_ascii=False),
                "stage_outputs": json.dumps(stage_outputs, ensure_ascii=False),
            }
        )

    return records


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for config, group in predictions.groupby("config", sort=False):
        valid = group[group["status"].eq("ok") & group["brier"].notna()]
        errors = group[group["status"].ne("ok")]["error"].fillna("")
        rows.append(
            {
                "split": group["split"].iloc[0],
                "config": config,
                "model": group["model"].iloc[0],
                "temperature": group["temperature"].iloc[0],
                "requested_temperature": group["requested_temperature"].iloc[0]
                if "requested_temperature" in group.columns
                else group["temperature"].iloc[0],
                "tool": group["tool"].iloc[0],
                "tool_choice": group["tool_choice"].iloc[0],
                "rows": len(group),
                "valid_rows": len(valid),
                "failures": len(group) - len(valid),
                "brier": valid["brier"].mean() if len(valid) else None,
                "mean_probability": valid["final_probability"].mean() if len(valid) else None,
                "calls_per_question": group["calls_per_question"].iloc[0],
                "mean_web_search_calls": valid["web_search_calls"].mean() if len(valid) else 0,
                "rate_limit_failures": int(errors.str.contains("HTTP 429").sum()),
                "parse_failures": int(
                    errors.str.contains("Unterminated string|output_text|JSON", regex=True).sum()
                ),
                "description": group["description"].iloc[0],
            }
        )
    order = {"community": 0, "guess_50_50": 1, DEFAULT_CONFIG: 2}
    summary = pd.DataFrame(rows)
    summary["config_order"] = summary["config"].map(order).fillna(99)
    return summary.sort_values("config_order").drop(columns=["config_order"])


def write_chart(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    chart_data = summary[summary["brier"].notna()].copy()
    if chart_data.empty:
        return None

    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    colors = [
        "#9aa0a6" if config in {"community", "guess_50_50"} else "#4c78a8"
        for config in chart_data["config"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(chart_data["config"], chart_data["brier"], color=colors)
    ax.set_ylabel("Brier score")
    ax.set_title("Model Experiment Brier Score")
    ax.set_ylim(0, max(chart_data["brier"]) * 1.25)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, chart_data["brier"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()

    path = output_dir / "model_experiment_brier.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_report(summary: pd.DataFrame, output_dir: Path, *, mock: bool) -> Path:
    best = summary[summary["brier"].notna()].sort_values("brier").head(1)
    best_line = "No successful model rows yet."
    if not best.empty:
        row = best.iloc[0]
        best_line = (
            f"Best result in this run: `{row['config']}` with Brier "
            f"{row['brier']:.6f} across {int(row['valid_rows'])} valid rows."
        )

    lines = [
        "# Model Experiment Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Mode: {'mock' if mock else 'OpenAI API'}",
        "",
        best_line,
        "",
        "Caveat: this is a post-hoc web-search diagnostic. Live search can leak information unavailable on the forecast date.",
        "",
        "| Config | Model | Temp sent | Requested temp | Tool | Valid rows | Failures | Rate-limit failures | Parse failures | Brier | Mean web searches |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        brier = "" if pd.isna(row["brier"]) else f"{row['brier']:.6f}"
        lines.append(
            f"| `{row['config']}` | `{row['model']}` | {row['temperature']} | "
            f"{row['requested_temperature']} | {row['tool']} | "
            f"{int(row['valid_rows'])} | {int(row['failures'])} | "
            f"{int(row['rate_limit_failures'])} | {int(row['parse_failures'])} | "
            f"{brier} | {row['mean_web_search_calls']:.2f} |"
        )

    path = output_dir / "model_experiment_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_outputs(records: list[dict[str, Any]], output_dir: Path, *, mock: bool) -> None:
    predictions = pd.DataFrame(records)
    predictions_path = output_dir / "model_experiment_predictions.csv"
    results_path = output_dir / "model_experiment_results.csv"

    predictions.to_csv(predictions_path, index=False)
    summary = summarize(predictions)
    summary.to_csv(results_path, index=False)
    chart_path = write_chart(summary, output_dir)
    report_path = write_report(summary, output_dir, mock=mock)

    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote results: {results_path}")
    print(f"Wrote report: {report_path}")
    if chart_path:
        print(f"Wrote chart: {chart_path}")
    else:
        print("Chart not written; no successful results or matplotlib is unavailable.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-5.5 + web_search on the selected forecasting scaffold."
    )
    parser.add_argument("--split", choices=["dev", "validation", "test"], default=DEFAULT_SPLIT)
    parser.add_argument("--config", choices=[DEFAULT_CONFIG], default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--send-temperature",
        action="store_true",
        help="Send the temperature field. Off by default because gpt-5.5 rejects it.",
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=runner.DEFAULT_RETRIES)
    parser.add_argument("--search-context-size", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--tool-choice", choices=["auto", "required"], default="required")
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default=None,
        help="Omit by default so gpt-5.5 uses the API default.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit rows for smoke tests.")
    parser.add_argument(
        "--only-question-ids",
        nargs="+",
        default=None,
        help="Evaluate only these safe question IDs, e.g. 22459 22465.",
    )
    parser.add_argument(
        "--rebuild-from-cache",
        action="store_true",
        help="Rebuild predictions/results/report from cached raw JSON without API calls.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore cached raw responses and rerun calls.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses; no API calls.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.mock and args.output_dir == DEFAULT_OUTPUT_DIR.resolve():
        args.output_dir = args.output_dir / "mock_smoke"
    records = asyncio.run(run(args))
    write_outputs(records, args.output_dir, mock=args.mock)


if __name__ == "__main__":
    main()
