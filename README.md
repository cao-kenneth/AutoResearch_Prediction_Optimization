# AutoResearch Prediction Optimization

This is an AutoResearch project for improving LLM-based probabilistic forecasts
on binary real-world questions.

The active direction is scaffold comparison: keep the model fixed, vary the
forecasting scaffold around it, and compare Brier scores under a call-budget
constraint. The selected scaffold is then checked on a held-out test set and in
post-hoc sensitivity experiments.

## Task

Given a question and a forecast date, predict the probability from `0` to `1`
that the event resolves Yes.

Performance is evaluated with Brier score:

```text
Brier score = mean((predicted_probability - outcome)^2)
```

Lower is better.

## Data

Source data is a hand-picked set of binary Metaculus-style forecasting
questions.

```text
data/
├── RawData.csv           # original dataset
├── train.csv             # original train split, kept for prompt optimization
├── dev.csv               # scaffold development split, seed 42
├── validation.csv        # scaffold validation split, seed 42
└── test_40_unused.csv    # final holdout
```

`dev.csv` and `validation.csv` are derived from `train.csv` with stratification
by `Resolution` and random seed `42`.

Current split sizes:

| Split | Rows | Yes | No | Use |
| --- | ---: | ---: | ---: | --- |
| `train.csv` | 67 | 22 | 45 | Preserved for prompt optimization. |
| `dev.csv` | 33 | 11 | 22 | Scaffold development and error analysis. |
| `validation.csv` | 34 | 11 | 23 | Scaffold comparison and selection. |
| `test_40_unused.csv` | 45 | 15 | 30 | Final held-out test after scaffold selection. |

## Project Structure

```text
.
├── README.md
├── program.md                         # agent-facing build brief
├── clean.py                           # creates train/dev/validation/test splits
├── baseline.py                        # legacy Codex CLI one-shot baseline
├── scaffold_runner.py                 # OpenAI API scaffold comparison runner
├── scaffold_optimizer.py              # scaffold tuning loop
├── final_test_sensitivity.py          # held-out test and temperature analysis
├── outputs/                           # general project outputs
├── data/
└── experiments/
    ├── README.md
    ├── buckets_experiment/            # legacy probability bucket experiment
    ├── model_experiment/              # post-hoc GPT-5.5 + web_search diagnostic
    └── optimize_prompt/               # legacy Codex CLI prompt optimization
```

## Main Scaffold Setup

The main scaffold experiments use:

- Model: `gpt-5.4-mini`
- Temperature: `0`
- Interface: OpenAI API
- Output format: structured JSON
- Tools/web search: disabled
- Runtime behavior: checkpoint and resume completed calls
- Call budget: at most five model calls per question

The initial comparison scores four configs:

| Config | Calls/question | Description |
| --- | ---: | --- |
| `baseline_one_shot` | 1 | Direct one-shot forecast. |
| `premortem_one_shot` | 1 | Forecast, list plausible failure modes, then adjust. |
| `self_critique` | 2 | Initial forecast followed by one global critique/revision call. |
| `three_perspectives_judge` | 4 | Three perspective forecasts followed by a judge/AIA forecaster. |

The root `baseline.py` is a simple one-shot baseline script kept at the top
level for convenience. The scaffold comparison should still implement
`baseline_one_shot` through the OpenAI API so all four scaffold configs use the
same model and API settings.

Run a local smoke test without API calls:

```bash
python scaffold_runner.py \
  --mock \
  --limit 2 \
  --splits validation \
  --output-dir outputs/api_smoke \
  --verbose
```

Run the initial scaffold comparison with the OpenAI API:

```bash
export OPENAI_API_KEY=...
python scaffold_runner.py --splits dev validation --verbose
```

The runner writes `scaffold_predictions.csv`, `scaffold_results.csv`,
`scaffold_summary.md`, `scaffold_validation_brier.png`, and raw checkpoint
files under `outputs/`.

After the initial comparison works, the next phase is a scaffold optimization
loop over a generated hyperparameter grid. The optimizer can choose from
combinations of:

- 1-3 perspectives: base-rate, skeptic, optimist, domain analyst, tail-risk
- Premortem on/off
- Aggregation: mean, median, judge
- Refinement: none, self-critique, rare-Yes audit, resolution audit

The default generated pool has 556 candidate configs under the five-call budget
because the four configs already evaluated by `scaffold_runner.py` are excluded.
Use this to inspect the pool without API calls:

```bash
python scaffold_optimizer.py --list-candidates
```

Run a no-cost optimizer smoke test:

```bash
python scaffold_optimizer.py \
  --mock \
  --limit 2 \
  --iterations 1 \
  --output-dir outputs/optimizer_smoke \
  --verbose
```

Run the optimizer candidate loop with the OpenAI API:

```bash
python scaffold_optimizer.py --iterations 5 --verbose
```

By default, each iteration lets the optimizer select up to two unevaluated
candidates. To evaluate every generated candidate, use `--evaluate-all`; that is
much more expensive because it is roughly 126,764 model calls on the current
dev+validation splits.

## Current Result Artifacts

Important outputs are kept in `outputs/`:

- Initial scaffold comparison:
  - `outputs/scaffold_predictions.csv`
  - `outputs/scaffold_results.csv`
  - `outputs/scaffold_summary.md`
  - `outputs/scaffold_validation_brier.png`
- Scaffold optimizer:
  - `outputs/optimizer/optimizer_predictions.csv`
  - `outputs/optimizer/optimizer_all_predictions.csv`
  - `outputs/optimizer/optimizer_results.csv`
  - `outputs/optimizer/optimizer_report.md`
  - `outputs/optimizer/optimizer_validation_brier.png`
- Final held-out sensitivity run:
  - `outputs/final_test_sensitivity/test_sensitivity_predictions.csv`
  - `outputs/final_test_sensitivity/test_sensitivity_results.csv`
  - `outputs/final_test_sensitivity/test_sensitivity_report.md`
  - `outputs/final_test_sensitivity/test_sensitivity_brier.png`

Result snapshot:

| Phase | Best/important result |
| --- | --- |
| Initial validation comparison | `self_critique`, Brier `0.147497`. |
| Optimizer validation selection | `p1_sk_pm0_agg_mean_ref_none`, Brier `0.131979`, one call/question. |
| Held-out test sensitivity | Community forecast Brier `0.144584`; selected scaffold at temperature `0.0` had Brier `0.209552` on 44 valid rows. |
| GPT-5.5 + web search diagnostic | Brier `0.157006` on 38 valid cached rows; this is post-hoc and not a fair no-leakage test. |

Raw API checkpoint folders are intentionally not part of the recommended GitHub
submission. They are useful locally for resume/cost control, but they are large
and not needed to review the project.

Run the held-out sensitivity script:

```bash
python final_test_sensitivity.py --verbose
```

Rebuild the GPT-5.5 web-search diagnostic report from cache without API calls:

```bash
python experiments/model_experiment/run_model_experiment.py --rebuild-from-cache --max-output-tokens 1400
```

## Legacy Experiments

Prompt optimization:

```bash
python experiments/optimize_prompt/baseline.py
python experiments/optimize_prompt/optimize_prompt.py
```

Architecture bucket experiment:

```bash
python experiments/buckets_experiment/run_architecture_experiments.py
```
