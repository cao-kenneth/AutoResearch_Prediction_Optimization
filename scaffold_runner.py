from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_MODEL = "gpt-5.4-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_OUTPUT_TOKENS = 700
DEFAULT_CONCURRENCY = 8
DEFAULT_RETRIES = 3
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


@dataclass(frozen=True)
class ScaffoldConfig:
    name: str
    calls_per_question: int
    description: str


CONFIGS: dict[str, ScaffoldConfig] = {
    "baseline_one_shot": ScaffoldConfig(
        name="baseline_one_shot",
        calls_per_question=1,
        description="Single direct forecast.",
    ),
    "premortem_one_shot": ScaffoldConfig(
        name="premortem_one_shot",
        calls_per_question=1,
        description="Initial probability, likely failure modes, adjusted final probability.",
    ),
    "self_critique": ScaffoldConfig(
        name="self_critique",
        calls_per_question=2,
        description="Initial forecast followed by one global critique/revision call.",
    ),
    "three_perspectives_judge": ScaffoldConfig(
        name="three_perspectives_judge",
        calls_per_question=4,
        description="Base-rate, skeptic, and optimist forecasts followed by a judge.",
    ),
}


FORECAST_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["probability", "reasoning"],
    "properties": {
        "probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Probability that the question resolves YES, on a 0 to 1 scale.",
        },
        "reasoning": {
            "type": "string",
            "description": "Concise forecasting rationale.",
        },
    },
}

PREMORTEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "initial_probability",
        "ways_forecast_could_be_wrong",
        "final_probability",
        "reasoning",
    ],
    "properties": {
        "initial_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "ways_forecast_could_be_wrong": {
            "type": "array",
            "minItems": 2,
            "maxItems": 5,
            "items": {"type": "string"},
        },
        "final_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning": {"type": "string"},
    },
}

REVISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["critique", "final_probability", "reasoning"],
    "properties": {
        "critique": {"type": "string"},
        "final_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning": {"type": "string"},
    },
}

JUDGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["final_probability", "reasoning"],
    "properties": {
        "final_probability": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
        "reasoning": {"type": "string"},
    },
}


SYSTEM_PROMPT = (
    "You are a calibrated binary forecasting model. Use only information that "
    "would have been available on or before the forecast date. Do not browse, "
    "call tools, or use post-forecast-date information. Return valid JSON that "
    "matches the requested schema. Probabilities must be decimals from 0 to 1."
)


class ApiError(RuntimeError):
    pass


class ResponsesClient:
    def __init__(
        self,
        api_key: str,
        *,
        concurrency: int,
        timeout_seconds: int,
        retries: int,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.semaphore = asyncio.Semaphore(concurrency)

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self.semaphore:
            for attempt in range(self.retries + 1):
                try:
                    return await asyncio.to_thread(self._post, payload)
                except ApiError as exc:
                    if attempt >= self.retries or not is_retryable_error(str(exc)):
                        raise
                    await asyncio.sleep(2 ** attempt)
        raise ApiError("OpenAI request failed after retries.")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ApiError(f"HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"Network error: {exc.reason}") from exc

        return json.loads(body)


def is_retryable_error(message: str) -> bool:
    return any(token in message for token in ("HTTP 408", "HTTP 409", "HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "Network error"))


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def safe_id(value: Any, fallback: str) -> str:
    if value is None or pd.isna(value):
        raw = fallback
    elif isinstance(value, float) and value.is_integer():
        raw = str(int(value))
    else:
        raw = str(value)
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw)


def forecast_date(row: pd.Series) -> str:
    formatted = row.get("forecast_date_formatted")
    if isinstance(formatted, str) and formatted.strip():
        return formatted.strip()
    return pd.to_datetime(row["Forecast_Date"]).strftime("%B %d, %Y")


def question_context(row: pd.Series) -> str:
    tournament = str(row.get("Tournament", "") or "").strip()
    tournament_text = f"\nTournament/context: {tournament}" if tournament else ""
    return (
        f"Forecast date: {forecast_date(row)}\n"
        f"Question: {row['Question']}{tournament_text}\n\n"
        "Estimate the probability that the question resolves YES. Be literal "
        "about the resolution criteria and deadline."
    )


def build_payload(
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
    schema_name: str,
    schema: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    return {
        "model": model,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        "tools": [],
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    }


def response_text(response: dict[str, Any]) -> str:
    parts: list[str] = []
    for output in response.get("output", []):
        if output.get("type") != "message":
            continue
        for item in output.get("content", []):
            if item.get("type") == "refusal":
                raise ValueError(f"Model refusal: {item.get('refusal', '')}")
            if item.get("type") == "output_text":
                parts.append(item.get("text", ""))
    text = "".join(parts).strip()
    if not text:
        raise ValueError("Response did not contain output_text.")
    return text


def parse_response_json(response: dict[str, Any]) -> dict[str, Any]:
    text = response_text(response)
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Structured response was not a JSON object.")
    return parsed


def validate_probability(value: Any) -> float:
    probability = float(value)
    if probability < 0 or probability > 1:
        raise ValueError(f"Probability out of range [0, 1]: {probability}")
    return probability


def mock_parsed_response(schema_name: str, prompt: str) -> dict[str, Any]:
    digest = hashlib.sha256(f"{schema_name}:{prompt}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    base = round(0.12 + 0.76 * value, 4)
    adjusted = round(min(0.95, max(0.05, base + 0.04 - 0.08 * value)), 4)

    if schema_name == "premortem_forecast":
        return {
            "initial_probability": base,
            "ways_forecast_could_be_wrong": [
                "The resolution criteria may exclude a near miss.",
                "The available pre-date evidence may not reflect later conditions.",
            ],
            "final_probability": adjusted,
            "reasoning": "Mock response for local testing only.",
        }
    if schema_name in {"revision_forecast", "judge_forecast"}:
        return {
            "critique": "Mock critique for local testing only.",
            "final_probability": adjusted,
            "reasoning": "Mock response for local testing only.",
        } if schema_name == "revision_forecast" else {
            "final_probability": adjusted,
            "reasoning": "Mock response for local testing only.",
        }
    return {
        "probability": base,
        "reasoning": "Mock response for local testing only.",
    }


async def call_stage(
    *,
    client: ResponsesClient | None,
    args: argparse.Namespace,
    cache_path: Path,
    schema_name: str,
    schema: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    if cache_path.exists() and not args.force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    payload = build_payload(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        schema_name=schema_name,
        schema=schema,
        prompt=prompt,
    )

    started_at = datetime.now().isoformat()
    if args.mock:
        parsed = mock_parsed_response(schema_name, prompt)
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
        parsed = parse_response_json(response)

    record = {
        "schema_name": schema_name,
        "request": payload,
        "response": response,
        "parsed": parsed,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
    }
    atomic_write_json(cache_path, record)
    return record


def direct_prompt(row: pd.Series) -> str:
    return (
        f"{question_context(row)}\n\n"
        "Return one calibrated probability and concise reasoning. The probability "
        "must be a decimal from 0 to 1, not a percent."
    )


def premortem_prompt(row: pd.Series) -> str:
    return (
        f"{question_context(row)}\n\n"
        "First give an initial probability. Then list the most plausible ways "
        "your forecast could be wrong. Finally adjust to a final probability. "
        "All probabilities must be decimals from 0 to 1, not percents."
    )


def perspective_prompt(row: pd.Series, perspective: str) -> str:
    instructions = {
        "base_rate": "Focus on outside-view base rates, precedents, and class frequency.",
        "skeptic": "Focus on blockers, resolution details, and reasons YES may fail.",
        "optimist": "Focus on credible paths to YES and evidence that could make YES underpriced.",
    }
    return (
        f"{question_context(row)}\n\n"
        f"Perspective: {perspective}. {instructions[perspective]} "
        "Return a probability as a decimal from 0 to 1 and concise reasoning."
    )


def revision_prompt(row: pd.Series, initial: dict[str, Any]) -> str:
    return (
        f"{question_context(row)}\n\n"
        "Initial forecast JSON:\n"
        f"{json.dumps(initial, ensure_ascii=False)}\n\n"
        "Critique the initial forecast for calibration errors, missed resolution "
        "details, and overconfidence. Then return a revised final probability "
        "as a decimal from 0 to 1."
    )


def judge_prompt(row: pd.Series, perspective_outputs: dict[str, dict[str, Any]]) -> str:
    return (
        f"{question_context(row)}\n\n"
        "You are the final judge/AIA forecaster. Synthesize these independent "
        "perspective forecasts. Do not average mechanically if a perspective "
        "appears to misunderstand the question.\n\n"
        f"Perspective forecasts JSON:\n{json.dumps(perspective_outputs, ensure_ascii=False)}\n\n"
        "Return one final probability as a decimal from 0 to 1 and concise reasoning."
    )


async def run_config_for_row(
    *,
    client: ResponsesClient | None,
    args: argparse.Namespace,
    row: pd.Series,
    split: str,
    config: ScaffoldConfig,
) -> dict[str, Any]:
    question_id = safe_id(row.get("Question ID"), f"row_{int(row.name)}")
    cache_dir = args.output_dir / "raw" / split / config.name / question_id
    stage_paths: dict[str, str] = {}
    stage_outputs: dict[str, Any] = {}

    try:
        if config.name == "baseline_one_shot":
            stage = await call_stage(
                client=client,
                args=args,
                cache_path=cache_dir / "forecast.json",
                schema_name="direct_forecast",
                schema=FORECAST_SCHEMA,
                prompt=direct_prompt(row),
            )
            stage_paths["forecast"] = str(cache_dir / "forecast.json")
            stage_outputs["forecast"] = stage["parsed"]
            final_probability = validate_probability(stage["parsed"]["probability"])
            final_reasoning = stage["parsed"]["reasoning"]

        elif config.name == "premortem_one_shot":
            stage = await call_stage(
                client=client,
                args=args,
                cache_path=cache_dir / "premortem.json",
                schema_name="premortem_forecast",
                schema=PREMORTEM_SCHEMA,
                prompt=premortem_prompt(row),
            )
            stage_paths["premortem"] = str(cache_dir / "premortem.json")
            stage_outputs["premortem"] = stage["parsed"]
            final_probability = validate_probability(stage["parsed"]["final_probability"])
            final_reasoning = stage["parsed"]["reasoning"]

        elif config.name == "self_critique":
            initial = await call_stage(
                client=client,
                args=args,
                cache_path=cache_dir / "initial_forecast.json",
                schema_name="direct_forecast",
                schema=FORECAST_SCHEMA,
                prompt=direct_prompt(row),
            )
            revision = await call_stage(
                client=client,
                args=args,
                cache_path=cache_dir / "revision.json",
                schema_name="revision_forecast",
                schema=REVISION_SCHEMA,
                prompt=revision_prompt(row, initial["parsed"]),
            )
            stage_paths["initial_forecast"] = str(cache_dir / "initial_forecast.json")
            stage_paths["revision"] = str(cache_dir / "revision.json")
            stage_outputs["initial_forecast"] = initial["parsed"]
            stage_outputs["revision"] = revision["parsed"]
            final_probability = validate_probability(revision["parsed"]["final_probability"])
            final_reasoning = revision["parsed"]["reasoning"]

        elif config.name == "three_perspectives_judge":
            perspective_names = ["base_rate", "skeptic", "optimist"]
            perspective_records = await asyncio.gather(
                *[
                    call_stage(
                        client=client,
                        args=args,
                        cache_path=cache_dir / f"{name}.json",
                        schema_name="perspective_forecast",
                        schema=FORECAST_SCHEMA,
                        prompt=perspective_prompt(row, name),
                    )
                    for name in perspective_names
                ]
            )
            perspective_outputs = {
                name: record["parsed"]
                for name, record in zip(perspective_names, perspective_records)
            }
            judge = await call_stage(
                client=client,
                args=args,
                cache_path=cache_dir / "judge.json",
                schema_name="judge_forecast",
                schema=JUDGE_SCHEMA,
                prompt=judge_prompt(row, perspective_outputs),
            )
            for name in perspective_names:
                stage_paths[name] = str(cache_dir / f"{name}.json")
            stage_paths["judge"] = str(cache_dir / "judge.json")
            stage_outputs.update(perspective_outputs)
            stage_outputs["judge"] = judge["parsed"]
            final_probability = validate_probability(judge["parsed"]["final_probability"])
            final_reasoning = judge["parsed"]["reasoning"]

        else:
            raise ValueError(f"Unknown config: {config.name}")

        outcome = int(row["resolution_binary"])
        brier = (final_probability - outcome) ** 2
        status = "ok"
        error = ""

    except Exception as exc:
        final_probability = None
        final_reasoning = ""
        brier = None
        status = "failed"
        error = str(exc)

    return {
        "split": split,
        "config": config.name,
        "question_id": question_id,
        "question": row["Question"],
        "forecast_date": forecast_date(row),
        "resolution": row["Resolution"],
        "resolution_binary": row["resolution_binary"],
        "final_probability": final_probability,
        "final_reasoning": final_reasoning,
        "brier": brier,
        "status": status,
        "error": error,
        "calls_per_question": config.calls_per_question,
        "stage_paths": json.dumps(stage_paths, ensure_ascii=False),
        "stage_outputs": json.dumps(stage_outputs, ensure_ascii=False),
    }


def load_split(split: str, limit: int | None) -> pd.DataFrame:
    path = REPO_ROOT / "data" / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = pd.read_csv(path)
    if "forecast_date_formatted" not in df.columns:
        df["forecast_date_formatted"] = pd.to_datetime(df["Forecast_Date"]).dt.strftime("%B %d, %Y")
    if "resolution_binary" not in df.columns:
        df["resolution_binary"] = df["Resolution"].str.strip().str.lower().map({"yes": 1, "no": 0})
    if limit is not None:
        df = df.head(limit)
    return df.reset_index(drop=True)


async def run_all(args: argparse.Namespace) -> list[dict[str, Any]]:
    load_dotenv(REPO_ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    client: ResponsesClient | None = None
    if not args.mock:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY in the environment or .env, or use --mock.")
        client = ResponsesClient(
            api_key,
            concurrency=args.concurrency,
            timeout_seconds=args.timeout_seconds,
            retries=args.retries,
        )

    configs = [CONFIGS[name] for name in args.configs]
    for config in configs:
        if config.calls_per_question > 5:
            raise ValueError(f"{config.name} exceeds the call budget.")

    tasks = []
    for split in args.splits:
        df = load_split(split, args.limit)
        for config in configs:
            for _, row in df.iterrows():
                tasks.append(
                    run_config_for_row(
                        client=client,
                        args=args,
                        row=row,
                        split=split,
                        config=config,
                    )
                )

    started = time.time()
    results: list[dict[str, Any]] = []
    for index, task in enumerate(asyncio.as_completed(tasks), start=1):
        result = await task
        results.append(result)
        if args.verbose:
            print(
                f"[{index}/{len(tasks)}] {result['split']} "
                f"{result['config']} {result['question_id']} {result['status']}",
                flush=True,
            )

    if args.verbose:
        print(f"Completed {len(results)} rows in {time.time() - started:.1f}s")
    return results


def summarize_results(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (split, config), group in predictions.groupby(["split", "config"], sort=False):
        valid = group[group["status"].eq("ok") & group["brier"].notna()]
        rows.append(
            {
                "split": split,
                "config": config,
                "rows": len(group),
                "valid_rows": len(valid),
                "failures": len(group) - len(valid),
                "brier": valid["brier"].mean() if len(valid) else None,
                "mean_probability": valid["final_probability"].mean() if len(valid) else None,
                "calls_per_question": CONFIGS[config].calls_per_question,
                "description": CONFIGS[config].description,
            }
        )
    summary = pd.DataFrame(rows)
    order = {name: index for index, name in enumerate(CONFIGS)}
    summary["config_order"] = summary["config"].map(order)
    return summary.sort_values(["split", "config_order"]).drop(columns=["config_order"])


def write_chart(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    validation = summary[summary["split"].eq("validation") & summary["brier"].notna()]
    if validation.empty:
        return None
    mpl_config_dir = output_dir / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    chart_path = output_dir / "scaffold_validation_brier.png"
    labels = validation["config"].tolist()
    values = validation["brier"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_ylabel("Brier score")
    ax.set_title("Validation Brier by Scaffold")
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.tick_params(axis="x", rotation=20)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    return chart_path


def write_summary(summary: pd.DataFrame, output_dir: Path, *, mock: bool) -> Path:
    validation = summary[summary["split"].eq("validation") & summary["brier"].notna()]
    best_line = "No successful validation results yet."
    if not validation.empty:
        best = validation.sort_values("brier").iloc[0]
        best_line = (
            f"Best validation scaffold: `{best['config']}` with Brier "
            f"{best['brier']:.6f} across {int(best['valid_rows'])} valid rows."
        )

    lines = [
        "# Scaffold Comparison Summary",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Mode: {'mock' if mock else 'OpenAI API'}",
        "",
        best_line,
        "",
        "| Split | Config | Valid rows | Failures | Brier | Calls/question |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        brier = "" if pd.isna(row["brier"]) else f"{row['brier']:.6f}"
        lines.append(
            f"| {row['split']} | `{row['config']}` | {int(row['valid_rows'])} | "
            f"{int(row['failures'])} | {brier} | {int(row['calls_per_question'])} |"
        )

    path = output_dir / "scaffold_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_outputs(results: list[dict[str, Any]], output_dir: Path, *, mock: bool) -> None:
    predictions = pd.DataFrame(results)
    predictions_path = output_dir / "scaffold_predictions.csv"
    results_path = output_dir / "scaffold_results.csv"

    predictions.to_csv(predictions_path, index=False)
    summary = summarize_results(predictions)
    summary.to_csv(results_path, index=False)
    chart_path = write_chart(summary, output_dir)
    summary_path = write_summary(summary, output_dir, mock=mock)

    print(f"Wrote predictions: {predictions_path}")
    print(f"Wrote results: {results_path}")
    print(f"Wrote summary: {summary_path}")
    if chart_path:
        print(f"Wrote chart: {chart_path}")
    else:
        print("Chart not written; no validation results or matplotlib is unavailable.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OpenAI API scaffold comparison on dev/validation splits."
    )
    parser.add_argument("--splits", nargs="+", default=["dev", "validation"], choices=["dev", "validation"])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per split for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Ignore cached raw responses and rerun calls.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses; no API calls.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    results = asyncio.run(run_all(args))
    write_outputs(results, args.output_dir, mock=args.mock)


if __name__ == "__main__":
    main()
