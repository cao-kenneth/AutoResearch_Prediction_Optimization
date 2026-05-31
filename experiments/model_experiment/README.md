# Model Experiment

Post-hoc model/tooling experiment for the selected scaffold:

- Model: `gpt-5.5`
- Requested temperature: `0`
- Tooling: Responses API `web_search`
- Tool choice: `required`
- Scaffold: `p1_sk_pm0_agg_mean_ref_none`
- Meaning: one skeptical forecaster, no premortem, mean aggregation, no refinement
- Calls/question: 1 model call plus web search tool use

Note: the GPT-5.5 API rejects the `temperature` request field, so the script
records `requested_temperature = 0` but omits the unsupported field from the
actual API payload.

The scaffold itself is imported from `scaffold_optimizer.py`; the experiment
uses `optimizer.CANDIDATES["p1_sk_pm0_agg_mean_ref_none"]` and
`optimizer.perspective_prompt(...)`. The intended change is the model/tooling
surface: `gpt-5.5` plus hosted web search.

This is not part of scaffold selection. It is a model/tooling sensitivity check.

Important caveat: these are historical forecasting questions. Live web search can expose post-forecast-date information, including actual resolution evidence. The prompt tells the model to ignore sources or snippets after the forecast date, but that cannot fully prevent contamination. Treat this experiment as diagnostic, not as a fair replacement for the no-search final test.

## Run

```bash
python experiments/model_experiment/run_model_experiment.py
```

By default, this evaluates `data/test_40_unused.csv`. For a cheap smoke test:

```bash
python experiments/model_experiment/run_model_experiment.py --limit 2
```

For local verification without API calls:

```bash
python experiments/model_experiment/run_model_experiment.py --mock --limit 2
```

Mock runs use `outputs/mock_smoke/` by default so they do not overwrite the
real diagnostic outputs.

The default concurrency is `1` to avoid confusing model/tool failures with rate
limit pressure. You can raise it later with `--concurrency 2` or higher.

To rebuild the report from cached raw responses without spending API calls:

```bash
python experiments/model_experiment/run_model_experiment.py --rebuild-from-cache --max-output-tokens 1400
```

To retry only the currently failed real rows, use their question IDs. The script
will call the API for those rows and then rebuild the full output tables from
cache:

```bash
python experiments/model_experiment/run_model_experiment.py \
  --only-question-ids 22459 22465 41501 20789 40217 \
  --max-output-tokens 4000 \
  --verbose
```

If a smoke test overwrote a cached real row with mock output, that row is marked
as unavailable in the cache rebuild rather than mixed into the real result.

## Outputs

The script writes:

- `outputs/model_experiment_predictions.csv`
- `outputs/model_experiment_results.csv`
- `outputs/model_experiment_report.md`
- `outputs/model_experiment_brier.png`
- `outputs/raw/...` cached request/response JSON
