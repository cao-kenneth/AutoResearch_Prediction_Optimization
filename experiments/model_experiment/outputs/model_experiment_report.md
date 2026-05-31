# Model Experiment Report

Generated: 2026-05-31T15:21:12.819184
Mode: OpenAI API

Best result in this run: `community` with Brier 0.144584 across 45 valid rows.

Caveat: this is a post-hoc web-search diagnostic. Live search can leak information unavailable on the forecast date.

| Config | Model | Temp sent | Requested temp | Tool | Valid rows | Failures | Rate-limit failures | Parse failures | Brier | Mean web searches |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `community` | `fixed` | fixed | fixed | none | 45 | 0 | 0 | 0 | 0.144584 | 0.00 |
| `guess_50_50` | `fixed` | fixed | fixed | none | 45 | 0 | 0 | 0 | 0.250000 | 0.00 |
| `p1_sk_pm0_agg_mean_ref_none` | `gpt-5.5` | omitted | 0.0 | web_search | 38 | 7 | 0 | 5 | 0.157006 | 1.47 |
