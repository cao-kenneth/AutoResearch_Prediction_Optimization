# Outputs

General project outputs are written here.

Compact CSV summaries, reports, and charts are intended to be shareable. Raw
checkpoint folders are local resume/cost-control artifacts and are ignored for
GitHub by default.

The scaffold runner writes:

- `scaffold_predictions.csv`
- `scaffold_results.csv`
- `scaffold_summary.md`
- `scaffold_validation_brier.png`
- `raw/` checkpoint files for resumable API calls

The scaffold optimizer writes under `optimizer/`:

- `optimizer_predictions.csv`
- `optimizer_all_predictions.csv`
- `optimizer_results.csv`
- `optimizer_report.md`
- `optimizer_validation_brier.png`
- `raw/` checkpoint files for evaluated candidate scaffolds
- `optimizer_raw/` checkpoint files for optimizer selection calls

The final held-out sensitivity run writes under `final_test_sensitivity/`:

- `test_sensitivity_predictions.csv`
- `test_sensitivity_results.csv`
- `test_sensitivity_report.md`
- `test_sensitivity_brier.png`
- `temp_*/raw/` checkpoint files separated by temperature
