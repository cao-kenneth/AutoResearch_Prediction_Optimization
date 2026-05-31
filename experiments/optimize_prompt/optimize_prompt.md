# Prompt Optimization Experiment

This folder contains the first AutoResearch experiment: test whether a Codex
CLI loop can improve a single forecasting prompt on the training set.

This is now treated as a completed baseline experiment, separate from the newer
scaffold-optimization work. Keep the Codex CLI behavior here unchanged so the
original results remain interpretable.

## Question

Can an LLM improve its own probabilistic forecasting prompt enough to reduce
Brier score on binary forecasting questions?

## Fixed Setup

- Dataset: `../../data/train.csv`
- Forecasting runner: `model.py`
- Baseline runner: `baseline.py`
- Optimizer: `optimize_prompt.py`
- Prompt under optimization: `prompt.txt`
- Model interface: Codex CLI via `codex exec`
- Metric: Brier score, lower is better
- Output format required from the forecaster:

```text
Probability: <number>
Reasoning: <one paragraph>
```

The optimizer may modify only `prompt.txt`. It should not change the data,
evaluation logic, model runner, parser, tools, APIs, or retrieval behavior.

## Files

```text
experiments/optimize_prompt/
├── baseline.py              # creates baseline predictions with Codex CLI
├── model.py                 # evaluates the current prompt with Codex CLI
├── optimize_prompt.py       # prompt optimization loop
├── optimize_prompt.md       # this experiment note
├── prompt.txt               # current/best prompt from the experiment
├── smoke_test_single.py     # one-row Codex CLI smoke test
├── smoke_test_two.py        # two-row Codex CLI smoke test
├── weekly_report.md         # written summary of the experiment
└── outputs/
    ├── train_predictions.csv
    ├── run_001/
    ├── run_002/
    └── run_003/
```

## Run

From the repository root:

```bash
python experiments/optimize_prompt/baseline.py
python experiments/optimize_prompt/optimize_prompt.py
```

`baseline.py` writes:

- `experiments/optimize_prompt/outputs/train_predictions.csv`
- `experiments/optimize_prompt/outputs/train_valid_predictions.csv`

`optimize_prompt.py` starts from `outputs/run_001` if it exists. Otherwise, it
starts from `outputs/train_valid_predictions.csv`.

Each optimization run writes:

- `experiments/optimize_prompt/outputs/run_XXX/predictions.csv`
- `experiments/optimize_prompt/outputs/run_XXX/prompt.txt`
- `experiments/optimize_prompt/outputs/run_XXX/metrics.json`

## Original Run Budget

The original experiment used exactly three optimization iterations after the
baseline. Each iteration:

1. Summarized the latest prediction errors.
2. Asked Codex CLI to revise `prompt.txt`.
3. Ran a full evaluation with `model.py`.
4. Kept the revised prompt only if Brier score improved.
5. Reverted to the previous prompt if the run failed or worsened.

## Results

The logged dry-run results were:

| Run | Brier | Notes |
| --- | ---: | --- |
| Baseline | 0.158942 | Original one-shot prompt baseline. |
| `run_001` | 0.140731 | Best result; kept as the strongest prompt. |
| `run_002` | 0.151896 | Worse than `run_001`; rejected. |
| `run_003` | 0.141570 | Near `run_001`, but one parse failure; rejected. |

Main takeaway: basic prompt optimization helped, but later iterations tended to
overcorrect toward caution and did not improve on the first optimized prompt.
