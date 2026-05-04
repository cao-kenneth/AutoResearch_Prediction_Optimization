# Architecture Bucket Experiments

This folder contains the controlled architecture experiment for probability
bucket refinement. The data sample, model, scoring rule, and prompt structure
are fixed across runs. The only changed variable is the maximum bucket depth.

## Controlled Experiment Set

All experiments start with the same first threshold:

```text
Is the probability higher or lower than 50%?
```

If the model says higher, the runner walks outward through upper thresholds. If
the model says lower, it walks outward through lower thresholds. The run stops
when the model reverses into an interval or when the experiment has no finer
bucket left. A final Codex call then asks for a probability constrained to the
resulting interval.

| Experiment | Changed variable: bucket depth | Upper path | Lower path |
| --- | --- | --- | --- |
| `exp_01_50` | 50 only | 50 | 50 |
| `exp_02_25_75` | add 25/75 | 50, 75 | 50, 25 |
| `exp_03_10_90` | add 10/90 | 50, 75, 90 | 50, 25, 10 |
| `exp_04_5_95` | add 5/95 | 50, 75, 90, 95 | 50, 25, 10, 5 |
| `exp_05_1_99` | add 1/99 | 50, 75, 90, 95, 99 | 50, 25, 10, 5, 1 |

Fixed controls:

- Source data: `../data/train.csv`
- Sample size: 20 rows
- Default sample seed: `390`
- Model command: `codex exec -m gpt-5.4`
- Metric: Brier score, using `resolution_binary`
- Forecast date handling: prompts say to use only information available on or
  before `forecast_date_formatted`

## Run

From the repository root:

```bash
python experiments/run_architecture_experiments.py
```

The full run can make up to 400 Codex calls: 20 sampled questions times the
maximum threshold checks and final probability calls across the five
experiments.

If an experiment CSV already has all 20 sampled rows, the runner skips that
experiment. If an experiment CSV is partial, the runner resumes the missing
sampled rows.

This will create:

- `experiments/sample_20.csv`
- `experiments/outputs/experiment_conditions.csv`
- `experiments/outputs/experiment_1.csv`
- `experiments/outputs/experiment_2.csv`
- `experiments/outputs/experiment_3.csv`
- `experiments/outputs/experiment_4.csv`
- `experiments/outputs/experiment_5.csv`
- `experiments/outputs/runs/<experiment_id>/metrics.json`
- `experiments/outputs/brier_scores.csv`
- `experiments/outputs/result_matrix.csv`
- `experiments/outputs/result_matrix.md`
- `experiments/outputs/metric_over_time.png`

Useful options:

```bash
# Create or reuse the 20-row sample without calling Codex.
python experiments/run_architecture_experiments.py --sample-only

# Run one experiment.
python experiments/run_architecture_experiments.py --experiments exp_03_10_90

# Re-run existing experiment outputs.
python experiments/run_architecture_experiments.py --force

# Use deterministic mock responses for code testing only. This does not call Codex.
python experiments/run_architecture_experiments.py --mock-codex --force
```
