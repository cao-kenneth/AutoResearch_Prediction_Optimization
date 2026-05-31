# Agent Build Brief

Build the active AutoResearch scaffold-comparison and scaffold-optimization
code for this repository.

## Objective

Treat the LLM as a black-box forecaster. Compare inference-time forecasting
scaffolds under a cost constraint, then tune a small pool of follow-up scaffold
variants using dev error analysis and validation Brier score.

## Constraints

- Use `data/dev.csv` and `data/validation.csv` for scaffold development and
  comparison.
- Do not use `data/test_40_unused.csv` until the final scaffold is selected.
- Keep `data/train.csv` unchanged because it is preserved for the legacy
  prompt-optimization experiment.
- New scaffold experiments should use the OpenAI API, not Codex CLI.
- Use model `gpt-5.4-mini`, temperature `0`, no tools, and no web search.
- Implement `baseline_one_shot` inside the new scaffold runner using the same
  OpenAI API settings as the other scaffolds.
- Use structured JSON outputs.
- Checkpoint raw and parsed outputs so interrupted runs can resume.
- Keep total model calls per question at or below five.

## Initial Scaffolds

Implement and evaluate these four configs first:

| Config | Calls/question | Behavior |
| --- | ---: | --- |
| `baseline_one_shot` | 1 | Single direct forecast. |
| `premortem_one_shot` | 1 | Give initial probability, list plausible ways the forecast could be wrong, then adjust final probability. |
| `self_critique` | 2 | Initial forecast, then one global critique/revision call. |
| `three_perspectives_judge` | 4 | Base-rate, skeptic, and optimist forecasts, then a judge/AIA forecaster chooses the final probability. |

Each config must output one final probability per question on a `0` to `1`
scale.

## Initial Comparison Outputs

The initial comparison phase should produce:

- A scaffold runner.
- A config definition file or clear in-code config registry.
- Per-question prediction records with raw model outputs and parsed final
  probabilities.
- A results CSV with Brier score by config and split.
- A simple chart comparing the four validation Brier scores.
- A short summary artifact describing which scaffold performed best.

## Optimizer Outputs

The optimizer phase should:

- Read the existing scaffold comparison predictions as incumbents.
- Generate the valid scaffold hyperparameter grid under the five-call budget.
- Use dev errors and aggregate results to choose 1-2 unevaluated candidate
  scaffold variants per iteration.
- Evaluate selected candidates on `dev.csv` and `validation.csv`.
- Reject candidates that exceed five calls per question.
- Select by validation Brier with an explicit cost penalty for extra calls.
- Write candidate predictions, combined predictions, aggregate results, raw
  checkpoints, and a short optimizer report.

## Implementation Notes

- Prefer async/concurrent API calls with a conservative concurrency limit.
- Retry transient API failures with backoff.
- Make runs resumable without paying twice for completed calls.
- Keep prompts compact to control cost and runtime.
- The generated optimizer pool should cover combinations of perspectives,
  premortem on/off, aggregation method, and refinement/audit type.

## Current Entry Point

Use `scaffold_runner.py` for the initial comparison and `scaffold_optimizer.py`
for the optimizer phase. Runtime artifacts should stay under the root
`outputs/` directory.
