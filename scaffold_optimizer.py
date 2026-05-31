from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import scaffold_runner as runner


DEFAULT_OUTPUT_DIR = runner.REPO_ROOT / "outputs" / "optimizer"
DEFAULT_SOURCE_RESULTS = runner.REPO_ROOT / "outputs" / "scaffold_predictions.csv"
DEFAULT_ACCEPT_THRESHOLD = 0.005
DEFAULT_COST_PENALTY = 0.002

PERSPECTIVES = {
    "base_rate": {
        "code": "br",
        "instruction": "Focus on outside-view base rates, precedents, and class frequency.",
    },
    "skeptic": {
        "code": "sk",
        "instruction": "Focus on blockers, deadline problems, near-misses, and reasons YES fails.",
    },
    "optimist": {
        "code": "op",
        "instruction": "Focus on credible paths to YES and evidence that could make YES underpriced.",
    },
    "domain_analyst": {
        "code": "da",
        "instruction": "Focus on domain-specific facts, institutions, actors, incentives, and timelines.",
    },
    "tail_risk": {
        "code": "tr",
        "instruction": "Focus on concrete rare-YES paths that may be underweighted.",
    },
}

PREMORTEM_OPTIONS = (False, True)
REFINEMENT_OPTIONS = ("none", "self_critique", "rare_yes_audit", "resolution_audit")
AGGREGATIONS = ("mean", "median", "judge")

REFINEMENT_CODES = {
    "none": "none",
    "self_critique": "self",
    "rare_yes_audit": "rare",
    "resolution_audit": "res",
}

SOURCE_EQUIVALENTS = {
    "p1_br_pm0_agg_mean_ref_none",
    "p1_br_pm1_agg_mean_ref_none",
    "p1_br_pm0_agg_mean_ref_self",
    "p3_br_sk_op_pm0_agg_judge_ref_none",
}


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    calls_per_question: int
    perspectives: tuple[str, ...]
    premortem: bool
    aggregation: str
    refinement: str
    description: str


def candidate_name(
    *,
    perspectives: tuple[str, ...],
    premortem: bool,
    aggregation: str,
    refinement: str,
) -> str:
    role_codes = "_".join(PERSPECTIVES[name]["code"] for name in perspectives)
    premortem_code = 1 if premortem else 0
    refinement_code = REFINEMENT_CODES[refinement]
    return f"p{len(perspectives)}_{role_codes}_pm{premortem_code}_agg_{aggregation}_ref_{refinement_code}"


def candidate_description(
    *,
    perspectives: tuple[str, ...],
    premortem: bool,
    aggregation: str,
    refinement: str,
) -> str:
    parts = [
        f"perspectives={list(perspectives)}",
        f"premortem={'on' if premortem else 'off'}",
        f"aggregation={aggregation}",
        f"refinement={refinement}",
    ]
    return "; ".join(parts)


def valid_aggregations(num_perspectives: int) -> tuple[str, ...]:
    if num_perspectives == 1:
        return ("mean", "judge")
    return AGGREGATIONS


def build_candidate_space(*, include_source_equivalents: bool = False) -> dict[str, CandidateConfig]:
    candidates: dict[str, CandidateConfig] = {}
    perspective_names = tuple(PERSPECTIVES)
    for num_perspectives in (1, 2, 3):
        for perspectives in itertools.combinations(perspective_names, num_perspectives):
            for premortem in PREMORTEM_OPTIONS:
                for aggregation in valid_aggregations(num_perspectives):
                    for refinement in REFINEMENT_OPTIONS:
                        calls = (
                            num_perspectives
                            + (1 if aggregation == "judge" else 0)
                            + (1 if refinement != "none" else 0)
                        )
                        if calls > 5:
                            continue
                        name = candidate_name(
                            perspectives=perspectives,
                            premortem=premortem,
                            aggregation=aggregation,
                            refinement=refinement,
                        )
                        if not include_source_equivalents and name in SOURCE_EQUIVALENTS:
                            continue
                        candidates[name] = CandidateConfig(
                            name=name,
                            calls_per_question=calls,
                            perspectives=perspectives,
                            premortem=premortem,
                            aggregation=aggregation,
                            refinement=refinement,
                            description=candidate_description(
                                perspectives=perspectives,
                                premortem=premortem,
                                aggregation=aggregation,
                                refinement=refinement,
                            ),
                        )
    return candidates


CANDIDATES = build_candidate_space()


def optimizer_selection_schema(max_items: int, candidate_names: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["selected_candidates", "rationale"],
        "properties": {
            "selected_candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_items,
                "items": {"type": "string", "enum": candidate_names},
            },
            "rationale": {"type": "string"},
        },
    }


def perspective_prompt(row: pd.Series, candidate: CandidateConfig, perspective: str) -> str:
    context = runner.question_context(row)
    premortem_text = ""
    if candidate.premortem:
        premortem_text = (
            " Before returning the probability, internally list the most plausible "
            "ways your forecast could be wrong, then adjust for those failure modes."
        )
    return (
        f"{context}\n\n"
        f"Perspective: {perspective}. {PERSPECTIVES[perspective]['instruction']}"
        f"{premortem_text} Return one probability as a decimal from 0 to 1 and concise reasoning."
    )


def aggregate_probabilities(probabilities: list[float], aggregation: str) -> float:
    if aggregation == "mean":
        return sum(probabilities) / len(probabilities)
    if aggregation == "median":
        ordered = sorted(probabilities)
        midpoint = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[midpoint]
        return (ordered[midpoint - 1] + ordered[midpoint]) / 2
    raise ValueError(f"Cannot mechanically aggregate with `{aggregation}`.")


def judge_prompt(
    row: pd.Series,
    candidate: CandidateConfig,
    perspective_outputs: dict[str, dict[str, Any]],
) -> str:
    premortem_text = ""
    if candidate.premortem:
        premortem_text = (
            " Each perspective was asked to account for ways its own forecast could be wrong."
        )
    return (
        f"{runner.question_context(row)}\n\n"
        "You are the final judge/AIA forecaster. Synthesize these independent "
        "perspective forecasts. Do not average mechanically if a perspective "
        "appears to misunderstand the question."
        f"{premortem_text}\n\n"
        f"Perspective forecasts JSON:\n{json.dumps(perspective_outputs, ensure_ascii=False)}\n\n"
        "Return one final probability as a decimal from 0 to 1 and concise reasoning."
    )


def refinement_prompt(
    row: pd.Series,
    candidate: CandidateConfig,
    initial: dict[str, Any],
    perspective_outputs: dict[str, dict[str, Any]],
) -> str:
    if candidate.refinement == "self_critique":
        instruction = (
            "Critique the initial forecast for calibration errors, missed resolution "
            "details, weak base-rate reasoning, and overconfidence. Revise only if warranted."
        )
    elif candidate.refinement == "rare_yes_audit":
        instruction = (
            "Audit the initial forecast for underprediction of rare-but-real YES "
            "outcomes. Ask whether a concrete path was dismissed because it seemed "
            "unusual, politically surprising, low-base-rate, or dependent on rapid events. "
            "Revise only if that path deserves more weight."
        )
    elif candidate.refinement == "resolution_audit":
        instruction = (
            "Audit the initial forecast for resolution-criteria and deadline errors. "
            "Check whether the event needs to be true by the deadline, announced by "
            "the deadline, or true at final resolution, and whether near-misses count. "
            "Revise only if the criteria change the probability."
        )
    else:
        raise ValueError(f"Unknown refinement type: {candidate.refinement}")

    return (
        f"{runner.question_context(row)}\n\n"
        f"Candidate config: {candidate.description}\n\n"
        f"Perspective forecasts JSON:\n{json.dumps(perspective_outputs, ensure_ascii=False)}\n\n"
        f"Initial final forecast JSON:\n{json.dumps(initial, ensure_ascii=False)}\n\n"
        f"{instruction}\n"
        "Return a revised final probability as a decimal from 0 to 1."
    )


async def call_stage(
    *,
    client: runner.ResponsesClient | None,
    args: argparse.Namespace,
    cache_path: Path,
    schema_name: str,
    schema: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    return await runner.call_stage(
        client=client,
        args=args,
        cache_path=cache_path,
        schema_name=schema_name,
        schema=schema,
        prompt=prompt,
    )


async def run_candidate_for_row(
    *,
    client: runner.ResponsesClient | None,
    args: argparse.Namespace,
    row: pd.Series,
    split: str,
    candidate: CandidateConfig,
) -> dict[str, Any]:
    question_id = runner.safe_id(row.get("Question ID"), f"row_{int(row.name)}")
    cache_dir = args.output_dir / "raw" / split / candidate.name / question_id
    stage_paths: dict[str, str] = {}
    stage_outputs: dict[str, Any] = {}

    try:
        perspective_records = await asyncio.gather(
            *[
                call_stage(
                    client=client,
                    args=args,
                    cache_path=cache_dir / f"{perspective}.json",
                    schema_name="perspective_forecast",
                    schema=runner.FORECAST_SCHEMA,
                    prompt=perspective_prompt(row, candidate, perspective),
                )
                for perspective in candidate.perspectives
            ]
        )
        perspective_outputs = {
            perspective: record["parsed"]
            for perspective, record in zip(candidate.perspectives, perspective_records)
        }
        for perspective in candidate.perspectives:
            stage_paths[perspective] = str(cache_dir / f"{perspective}.json")
        stage_outputs.update(perspective_outputs)

        if candidate.aggregation == "judge":
            judge = await call_stage(
                client=client,
                args=args,
                cache_path=cache_dir / "judge.json",
                schema_name="judge_forecast",
                schema=runner.JUDGE_SCHEMA,
                prompt=judge_prompt(row, candidate, perspective_outputs),
            )
            initial_probability = runner.validate_probability(judge["parsed"]["final_probability"])
            initial_reasoning = judge["parsed"]["reasoning"]
            initial_output = {
                "probability": initial_probability,
                "reasoning": initial_reasoning,
                "aggregation": "judge",
            }
            stage_paths["judge"] = str(cache_dir / "judge.json")
            stage_outputs["judge"] = judge["parsed"]
        else:
            probabilities = [
                runner.validate_probability(record["parsed"]["probability"])
                for record in perspective_records
            ]
            initial_probability = aggregate_probabilities(probabilities, candidate.aggregation)
            initial_reasoning = (
                f"{candidate.aggregation.title()} of perspective forecasts: "
                + ", ".join(
                    f"{perspective}={probability:.3f}"
                    for perspective, probability in zip(candidate.perspectives, probabilities)
                )
            )
            initial_output = {
                "probability": initial_probability,
                "reasoning": initial_reasoning,
                "aggregation": candidate.aggregation,
            }

        if candidate.refinement != "none":
            revision = await call_stage(
                client=client,
                args=args,
                cache_path=cache_dir / f"{candidate.refinement}.json",
                schema_name="revision_forecast",
                schema=runner.REVISION_SCHEMA,
                prompt=refinement_prompt(row, candidate, initial_output, perspective_outputs),
            )
            stage_paths[candidate.refinement] = str(cache_dir / f"{candidate.refinement}.json")
            stage_outputs[candidate.refinement] = revision["parsed"]
            final_probability = runner.validate_probability(revision["parsed"]["final_probability"])
            final_reasoning = revision["parsed"]["reasoning"]
        else:
            final_probability = initial_probability
            final_reasoning = initial_reasoning

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
        "config": candidate.name,
        "question_id": question_id,
        "question": row["Question"],
        "forecast_date": runner.forecast_date(row),
        "resolution": row["Resolution"],
        "resolution_binary": row["resolution_binary"],
        "final_probability": final_probability,
        "final_reasoning": final_reasoning,
        "brier": brier,
        "status": status,
        "error": error,
        "calls_per_question": candidate.calls_per_question,
        "description": candidate.description,
        "perspectives": ",".join(candidate.perspectives),
        "premortem": candidate.premortem,
        "aggregation": candidate.aggregation,
        "refinement": candidate.refinement,
        "stage_paths": json.dumps(stage_paths, ensure_ascii=False),
        "stage_outputs": json.dumps(stage_outputs, ensure_ascii=False),
    }


def load_existing_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing source scaffold predictions: {path}. Run scaffold_runner.py first."
        )
    df = pd.read_csv(path)
    required = {"split", "config", "question_id", "question", "brier", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Source predictions missing columns: {sorted(missing)}")
    return df


def load_prior_optimizer_predictions(args: argparse.Namespace) -> pd.DataFrame:
    if args.force:
        return pd.DataFrame()
    path = args.output_dir / "optimizer_predictions.csv"
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
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
                "mean_probability": valid["final_probability"].mean()
                if "final_probability" in valid.columns and len(valid)
                else None,
                "calls_per_question": group["calls_per_question"].iloc[0]
                if "calls_per_question" in group.columns and len(group)
                else None,
                "description": group["description"].iloc[0]
                if "description" in group.columns and len(group)
                else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["split", "brier"], na_position="last")


def best_validation(summary: pd.DataFrame) -> pd.Series:
    validation = summary[summary["split"].eq("validation") & summary["brier"].notna()]
    if validation.empty:
        raise ValueError("No valid validation rows found.")
    return validation.sort_values("brier").iloc[0]


def build_error_report(predictions: pd.DataFrame, *, config: str, max_rows: int = 10) -> str:
    data = predictions[
        predictions["split"].eq("dev")
        & predictions["config"].eq(config)
        & predictions["status"].eq("ok")
    ].copy()
    if data.empty:
        data = predictions[
            predictions["split"].eq("validation")
            & predictions["config"].eq(config)
            & predictions["status"].eq("ok")
        ].copy()
    data = data.sort_values("brier", ascending=False).head(max_rows)
    lines = [f"Error report for current best config `{config}`:"]
    for _, row in data.iterrows():
        lines.append(
            "- "
            f"Question ID {row['question_id']}: {row['question']} "
            f"Resolution={row['resolution']}; prediction={row['final_probability']}; "
            f"Brier={row['brier']:.4f}"
        )
    return "\n".join(lines)


def optimizer_prompt(
    *,
    summary: pd.DataFrame,
    error_report: str,
    available_candidates: list[str],
    candidates_per_iteration: int,
    cost_penalty: float,
) -> str:
    candidate_lines = [
        f"- {name}: calls={CANDIDATES[name].calls_per_question}; {CANDIDATES[name].description}"
        for name in available_candidates
    ]
    return (
        "You are selecting scaffold configs for a black-box LLM forecasting experiment.\n"
        "The candidate pool is a generated hyperparameter grid over perspectives, "
        "premortem on/off, aggregation, and refinement/audit type. Choose configs "
        "most likely to improve Brier score under the cost budget. Prefer cheaper "
        "configs unless extra calls are clearly justified.\n"
        "Known failure modes include underpredicting rare-but-real YES outcomes, "
        "short-window overconfidence, and resolution/deadline mistakes.\n\n"
        f"Cost-adjusted selection penalty: add {cost_penalty} Brier per extra call above 1.\n\n"
        "Current aggregate results:\n"
        f"{summary.to_string(index=False)}\n\n"
        f"{error_report}\n\n"
        "Available candidate configs:\n"
        + "\n".join(candidate_lines)
        + f"\n\nSelect up to {candidates_per_iteration} candidate names from the available list."
    )


async def select_candidates(
    *,
    client: runner.ResponsesClient | None,
    args: argparse.Namespace,
    iteration: int,
    source_predictions: pd.DataFrame,
    evaluated: set[str],
) -> tuple[list[str], str]:
    available = [name for name in CANDIDATES if name not in evaluated]
    if not available:
        return [], "No candidates left."

    if args.evaluate_all:
        return available, "Selected all remaining candidates because --evaluate-all was set."

    count = min(args.candidates_per_iteration, len(available))
    summary = summarize_predictions(source_predictions)
    best = best_validation(summary)
    error_report = build_error_report(source_predictions, config=str(best["config"]))

    if args.mock:
        return available[:count], "Mock selection: chose the first available candidates."

    if client is None:
        raise RuntimeError("OpenAI client is required unless --mock is set.")

    cache_path = args.output_dir / "optimizer_raw" / f"iteration_{iteration:02d}_selection.json"
    if cache_path.exists() and not args.force:
        record = json.loads(cache_path.read_text(encoding="utf-8"))
        parsed = record["parsed"]
    else:
        prompt = optimizer_prompt(
            summary=summary,
            error_report=error_report,
            available_candidates=available,
            candidates_per_iteration=count,
            cost_penalty=args.cost_penalty,
        )
        payload = runner.build_payload(
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_output_tokens,
            schema_name="optimizer_selection",
            schema=optimizer_selection_schema(count, available),
            prompt=prompt,
        )
        response = await client.create(payload)
        parsed = runner.parse_response_json(response)
        runner.atomic_write_json(
            cache_path,
            {
                "request": payload,
                "response": response,
                "parsed": parsed,
                "completed_at": datetime.now().isoformat(),
            },
        )

    selected = [
        name
        for name in parsed.get("selected_candidates", [])
        if name in available and CANDIDATES[name].calls_per_question <= 5
    ][:count]
    rationale = str(parsed.get("rationale", ""))
    if not selected:
        selected = available[:count]
        rationale = "Optimizer returned no valid candidates; fell back to first available."
    return selected, rationale


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


async def evaluate_candidates(
    *,
    client: runner.ResponsesClient | None,
    args: argparse.Namespace,
    candidates: list[str],
) -> list[dict[str, Any]]:
    tasks = []
    for split in ["dev", "validation"]:
        df = runner.load_split(split, args.limit)
        for name in candidates:
            candidate = CANDIDATES[name]
            if candidate.calls_per_question > 5:
                continue
            for _, row in df.iterrows():
                tasks.append(
                    run_candidate_for_row(
                        client=client,
                        args=args,
                        row=row,
                        split=split,
                        candidate=candidate,
                    )
                )

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
    return results


def cost_adjusted_score(row: pd.Series, cost_penalty: float) -> float:
    calls = row.get("calls_per_question")
    calls = 1 if pd.isna(calls) else int(calls)
    return float(row["brier"]) + cost_penalty * max(0, calls - 1)


def acceptance_decision(
    *,
    incumbent: pd.Series,
    challenger: pd.Series,
    cost_penalty: float,
    threshold: float,
) -> tuple[bool, str]:
    incumbent_score = cost_adjusted_score(incumbent, cost_penalty)
    challenger_score = cost_adjusted_score(challenger, cost_penalty)
    improvement = incumbent_score - challenger_score
    if improvement >= threshold:
        return True, (
            f"Accepted `{challenger['config']}`: cost-adjusted validation score "
            f"improved by {improvement:.6f}."
        )
    return False, (
        f"Rejected `{challenger['config']}`: cost-adjusted validation score "
        f"improved by only {improvement:.6f}; threshold is {threshold:.6f}."
    )


def format_selected_candidates(selected: list[str]) -> str:
    if len(selected) <= 20:
        return ", ".join(selected)
    preview = ", ".join(selected[:20])
    return f"{preview}, ... ({len(selected)} total)"


async def run_optimizer(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = make_client(args)
    source_predictions = load_existing_results(args.source_predictions)
    prior_predictions = load_prior_optimizer_predictions(args)
    evaluated: set[str] = set()
    candidate_records: list[dict[str, Any]] = []
    decisions: list[str] = []

    if prior_predictions.empty:
        all_predictions = source_predictions.copy()
    else:
        evaluated.update(str(name) for name in prior_predictions["config"].dropna().unique())
        candidate_records.extend(prior_predictions.to_dict("records"))
        all_predictions = pd.concat([source_predictions, prior_predictions], ignore_index=True)
        decisions.append(
            f"Loaded {len(prior_predictions)} prior optimizer prediction rows "
            f"for {len(evaluated)} already-evaluated configs."
        )

    decisions.append(
        f"Candidate search space: {len(CANDIDATES)} valid generated configs under "
        "the five-call budget."
    )

    for iteration in range(1, args.iterations + 1):
        pre_iteration_summary = summarize_predictions(all_predictions)
        incumbent = best_validation(pre_iteration_summary)
        selection_index = len(evaluated) + 1

        selected, rationale = await select_candidates(
            client=client,
            args=args,
            iteration=selection_index,
            source_predictions=all_predictions,
            evaluated=evaluated,
        )
        if not selected:
            decisions.append(f"Iteration {selection_index}: stopped because no candidates remain.")
            break

        decisions.append(
            f"Iteration {selection_index}: selected {format_selected_candidates(selected)}. "
            f"Rationale: {rationale}"
        )
        evaluated.update(selected)

        started = time.time()
        results = await evaluate_candidates(client=client, args=args, candidates=selected)
        decisions.append(
            f"Iteration {selection_index}: evaluated {len(results)} prediction rows in "
            f"{time.time() - started:.1f}s."
        )
        if results:
            new_predictions = pd.DataFrame(results)
            all_predictions = pd.concat([all_predictions, new_predictions], ignore_index=True)
            candidate_records.extend(results)

        summary = summarize_predictions(all_predictions)
        validation = summary[summary["split"].eq("validation") & summary["brier"].notna()]
        challenger_rows = validation[validation["config"].isin(selected)].sort_values("brier")
        if not challenger_rows.empty:
            challenger = challenger_rows.iloc[0]
            accepted, reason = acceptance_decision(
                incumbent=incumbent,
                challenger=challenger,
                cost_penalty=args.cost_penalty,
                threshold=args.accept_threshold,
            )
            decisions.append(reason)
            if accepted:
                decisions.append(f"Current best candidate: `{challenger['config']}`.")

        if len(evaluated) == len(CANDIDATES):
            decisions.append("All candidates have been evaluated.")
            break

    write_optimizer_outputs(
        candidate_records=candidate_records,
        combined_predictions=all_predictions,
        decisions=decisions,
        args=args,
    )


def write_optimizer_outputs(
    *,
    candidate_records: list[dict[str, Any]],
    combined_predictions: pd.DataFrame,
    decisions: list[str],
    args: argparse.Namespace,
) -> None:
    candidate_predictions = pd.DataFrame(candidate_records)
    candidate_predictions_path = args.output_dir / "optimizer_predictions.csv"
    combined_predictions_path = args.output_dir / "optimizer_all_predictions.csv"
    results_path = args.output_dir / "optimizer_results.csv"
    report_path = args.output_dir / "optimizer_report.md"

    if not candidate_predictions.empty:
        candidate_predictions.to_csv(candidate_predictions_path, index=False)
    else:
        candidate_predictions_path.write_text("", encoding="utf-8")

    combined_predictions.to_csv(combined_predictions_path, index=False)
    summary = summarize_predictions(combined_predictions)
    summary.to_csv(results_path, index=False)
    chart_path = write_optimizer_chart(summary, args.output_dir, args.cost_penalty)

    validation = summary[summary["split"].eq("validation") & summary["brier"].notna()].copy()
    if not validation.empty:
        validation["cost_adjusted_score"] = validation.apply(
            lambda row: cost_adjusted_score(row, args.cost_penalty),
            axis=1,
        )
        best = validation.sort_values("cost_adjusted_score").iloc[0]
        best_line = (
            f"Best cost-adjusted validation config: `{best['config']}` "
            f"(Brier {best['brier']:.6f}, calls {int(best['calls_per_question'])}, "
            f"score {best['cost_adjusted_score']:.6f})."
        )
    else:
        best_line = "No validation results available."

    lines = [
        "# Scaffold Optimizer Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        f"Mode: {'mock' if args.mock else 'OpenAI API'}",
        f"Generated candidate configs: {len(CANDIDATES)}",
        f"Acceptance threshold: {args.accept_threshold}",
        f"Cost penalty per extra call: {args.cost_penalty}",
        "",
        best_line,
        "",
        "## Decisions",
        "",
    ]
    lines.extend(f"- {line}" for line in decisions)
    lines.extend(
        [
            "",
            "## Validation Summary",
            "",
            "| Config | Brier | Calls/question | Cost-adjusted score |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    if not validation.empty:
        for _, row in validation.sort_values("cost_adjusted_score").iterrows():
            lines.append(
                f"| `{row['config']}` | {row['brier']:.6f} | "
                f"{int(row['calls_per_question'])} | {row['cost_adjusted_score']:.6f} |"
            )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote candidate predictions: {candidate_predictions_path}")
    print(f"Wrote combined predictions: {combined_predictions_path}")
    print(f"Wrote optimizer results: {results_path}")
    print(f"Wrote optimizer report: {report_path}")
    if chart_path:
        print(f"Wrote optimizer chart: {chart_path}")
    else:
        print("Optimizer chart not written; no validation results or matplotlib is unavailable.")


def write_optimizer_chart(
    summary: pd.DataFrame,
    output_dir: Path,
    cost_penalty: float,
) -> Path | None:
    validation = summary[summary["split"].eq("validation") & summary["brier"].notna()].copy()
    if validation.empty:
        return None

    mpl_config_dir = output_dir.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    validation["cost_adjusted_score"] = validation.apply(
        lambda row: cost_adjusted_score(row, cost_penalty),
        axis=1,
    )
    validation = validation.sort_values("cost_adjusted_score", ascending=True)

    labels = [
        f"{row['config']} ({int(row['calls_per_question'])}c)"
        for _, row in validation.iterrows()
    ]
    values = validation["brier"].tolist()
    colors = [
        "#9aa0a6" if str(config) in runner.CONFIGS else "#4c78a8"
        for config in validation["config"]
    ]
    baseline = validation[validation["config"].eq("baseline_one_shot")]
    baseline_brier = None if baseline.empty else float(baseline.iloc[0]["brier"])

    height = max(6, 0.42 * len(validation) + 1.6)
    fig, ax = plt.subplots(figsize=(12, height))
    bars = ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Validation Brier score")
    ax.set_title("Optimizer Validation Brier by Scaffold Config")
    ax.set_xlim(0, max(values) * 1.18 if values else 1)
    ax.grid(axis="x", alpha=0.2)

    if baseline_brier is not None:
        ax.axvline(
            baseline_brier,
            color="#d62728",
            linestyle="--",
            linewidth=1,
            label=f"baseline_one_shot {baseline_brier:.4f}",
        )
        ax.legend(loc="lower right")

    for bar, value in zip(bars, values):
        ax.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f" {value:.4f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    fig.tight_layout()
    chart_path = output_dir / "optimizer_validation_brier.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)
    return chart_path


def print_candidate_summary() -> None:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES.values():
        rows.append(
            {
                "calls": candidate.calls_per_question,
                "num_perspectives": len(candidate.perspectives),
                "premortem": candidate.premortem,
                "aggregation": candidate.aggregation,
                "refinement": candidate.refinement,
            }
        )
    df = pd.DataFrame(rows)
    print(f"Generated candidate configs: {len(CANDIDATES)}")
    print("\nBy calls/question:")
    print(df.groupby("calls").size().to_string())
    print("\nBy aggregation/refinement:")
    print(df.groupby(["aggregation", "refinement"]).size().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune generated scaffold candidates using dev/validation results."
    )
    parser.add_argument("--source-predictions", type=Path, default=DEFAULT_SOURCE_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--candidates-per-iteration", type=int, default=2)
    parser.add_argument("--accept-threshold", type=float, default=DEFAULT_ACCEPT_THRESHOLD)
    parser.add_argument("--cost-penalty", type=float, default=DEFAULT_COST_PENALTY)
    parser.add_argument("--evaluate-all", action="store_true")
    parser.add_argument(
        "--include-source-equivalents",
        action="store_true",
        help="Include generated configs equivalent to the four already-run scaffold baselines.",
    )
    parser.add_argument(
        "--list-candidates",
        action="store_true",
        help="Print the generated candidate-space size and exit without API calls.",
    )
    parser.add_argument("--model", default=runner.DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=runner.DEFAULT_TEMPERATURE)
    parser.add_argument("--max-output-tokens", type=int, default=runner.DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--concurrency", type=int, default=runner.DEFAULT_CONCURRENCY)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--retries", type=int, default=runner.DEFAULT_RETRIES)
    parser.add_argument("--limit", type=int, default=None, help="Limit rows per split for smoke tests.")
    parser.add_argument("--force", action="store_true", help="Ignore cached raw responses and rerun calls.")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock responses; no API calls.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()

    global CANDIDATES
    CANDIDATES = build_candidate_space(
        include_source_equivalents=args.include_source_equivalents
    )

    if args.list_candidates:
        print_candidate_summary()
        return

    asyncio.run(run_optimizer(args))


if __name__ == "__main__":
    main()
