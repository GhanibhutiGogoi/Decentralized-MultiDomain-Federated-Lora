# Completion evidence report

Start with `aggregate.json` for all contexts, arm statistics, provenance and pairing exclusions. `seed_metrics.csv` retains each final seed result; `paired_differences.csv` records right minus left for every compatible arm pair. Positive accuracy differences and negative cost/gap differences have different interpretations. No winners or unfavorable runs were filtered.

`round_aggregate.csv` and `per_domain_aggregate.csv` supply chart-ready means and sample standard deviations; their matching raw CSVs retain every seed. PNG/SVG figures show mean ± sample SD. Separate context IDs indicate different protocols and must not be pooled. Partial runs and single-seed evidence remain labeled in `aggregate.json`.
