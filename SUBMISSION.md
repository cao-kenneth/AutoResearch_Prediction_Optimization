# Submission Notes

This repository contains the active scaffold-optimization project plus archived
experiments.

## Include in GitHub

Commit the source code, data splits, experiment notes, and compact result
artifacts:

- `README.md`, `program.md`, `SUBMISSION.md`, `requirements.txt`, `.gitignore`
- `clean.py`, `baseline.py`, `scaffold_runner.py`, `scaffold_optimizer.py`,
  `final_test_sensitivity.py`
- `data/RawData.csv`, `data/train.csv`, `data/dev.csv`,
  `data/validation.csv`, `data/test_40_unused.csv`
- `experiments/README.md`
- `experiments/buckets_experiment/`
- `experiments/optimize_prompt/`
- `experiments/model_experiment/README.md`
- `experiments/model_experiment/run_model_experiment.py`
- Compact output summaries and charts under `outputs/` and
  `experiments/*/outputs/`, such as result CSVs, reports, and PNG charts

## Do Not Commit

Keep these local:

- `.env` and any API keys
- `__pycache__/`, `.DS_Store`, `.matplotlib/`
- Raw API checkpoint folders such as `outputs/raw/`,
  `outputs/optimizer/raw/`, `outputs/optimizer/optimizer_raw/`,
  `outputs/final_test_sensitivity/temp_*/raw/`, and
  `experiments/model_experiment/outputs/raw/`
- One-off smoke-test folders such as `outputs/api_smoke/`

The raw checkpoint folders are useful locally for resuming expensive API runs,
but they are large and not necessary for reviewing the project.

## Suggested Git Commands

From the repository root:

```bash
git status --short
git add -A
git status --short
git commit -m "Reorganize forecasting scaffold experiments"
git branch -M main
git remote add origin git@github.com:<your-username>/<your-repo>.git
git push -u origin main
```

If the remote already exists, skip `git remote add origin ...` and run:

```bash
git remote -v
git push -u origin main
```
