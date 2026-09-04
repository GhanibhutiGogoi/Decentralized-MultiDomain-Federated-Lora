# Experiment 2 Comparison Report

## Injection Point

Project 1 aggregation computes normalized client weights from `samples * q`.
Experiment 2 keeps that path unchanged when lambda is disabled. The optional
plug-in in `experiment/experiment2/lambda_aggregation.py` passes `q * lambda`
to the existing aggregator, producing `samples * q * lambda` without replacing
the aggregation algorithm.

The quality score q is computed as `1 / (1 + avg_train_loss)`. Because q already
contains local loss information, Experiment 2 does not include local loss or q
as lambda features.

## Experiment 1 Evidence

| predictor | pearson | spearman | n |
| --- | --- | --- | --- |
| js_to_global | -0.221937 | -0.233241 | 75 |
| kl_to_global | -0.214259 | -0.203182 | 75 |
| update_cosine_distance_to_mean | -0.0551151 | -0.0643024 | 75 |
| update_l2_distance_to_mean | 0.0954963 | 0.349593 | 75 |

Controlled regressions from Experiment 1:

| model_predictor | term | standardized_beta | standard_error | r_squared | n |
| --- | --- | --- | --- | --- | --- |
| js_to_global | js_to_global | -0.217617 | 0.116164 | 0.0551776 | 75 |
| js_to_global | adaptive_rank | -0.0179529 | 0.116205 | 0.0551776 | 75 |
| js_to_global | local_loss | -0.0756791 | 0.115523 | 0.0551776 | 75 |
| kl_to_global | kl_to_global | -0.209746 | 0.116991 | 0.0514195 | 75 |
| kl_to_global | adaptive_rank | -0.0112994 | 0.116999 | 0.0514195 | 75 |
| kl_to_global | local_loss | -0.0739765 | 0.115797 | 0.0514195 | 75 |
| update_cosine_distance_to_mean | update_cosine_distance_to_mean | -0.0361413 | 0.121355 | 0.00971271 | 75 |
| update_cosine_distance_to_mean | adaptive_rank | -0.0401174 | 0.118514 | 0.00971271 | 75 |
| update_cosine_distance_to_mean | local_loss | -0.075292 | 0.121215 | 0.00971271 | 75 |
| update_l2_distance_to_mean | update_l2_distance_to_mean | 0.0799145 | 0.120511 | 0.0145789 | 75 |
| update_l2_distance_to_mean | adaptive_rank | -0.037428 | 0.118173 | 0.0145789 | 75 |
| update_l2_distance_to_mean | local_loss | -0.0671087 | 0.120407 | 0.0145789 | 75 |

The evidence supports a multi-factor lambda: update L2 is the strongest positive
monotonic signal, JS divergence is a weak negative domain-distribution penalty,
and update cosine is weak enough to be shrinkage-controlled rather than dominant.

## Form A

Interpretable formula fitted with standardized OLS:

`score_A = beta_l2 * z(log(1 + update_l2_distance_to_mean)) + beta_js * z(js_to_global) + intercept`

`lambda_A` is obtained by exponentiating the centered score inside each
`(task, round)` aggregation context, clipping to `[0.5, 1.5]`, and renormalizing
to mean one inside the same context.

## Form B

Data-driven but interpretable ridge formula:

`score_B = X_standardized * beta_ridge`

Features: `log_update_l2, js_to_global, update_cosine_distance_to_mean, normalized_entropy, log_class_imbalance_ratio`.

Selected ridge alpha from leave-one-task-out RMSE: `100.0`.

## Normalization

For each candidate score s:

`lambda_i = exp(scale * (s_i - mean_context(s)))`

then clip to `[0.5, 1.5]` and renormalize so each `(task, round)` has mean
lambda equal to one. The fitted scale is `0.886394` and is chosen so
lambda's coefficient of variation is no more than half of q's coefficient of
variation, capped at 0.20. This keeps lambda positive and numerically stable
without allowing it to dominate q.

## Coefficients

| form | term | coefficient | abs_coefficient | ridge_alpha | feature_mean | feature_std |
| --- | --- | --- | --- | --- | --- | --- |
| form_a | intercept | -1.02558e-16 | 1.02558e-16 | nan |  |  |
| form_a | log_update_l2 | 0.102844 | 0.102844 | nan | 1.7562762023044465 | 0.6296896253909049 |
| form_a | js_to_global | -0.231414 | 0.231414 | nan | 0.10389204405189059 | 0.06137239129252118 |
| form_b | intercept | 0 | 0 | 100 |  |  |
| form_b | log_update_l2 | 0.0469097 | 0.0469097 | 100 | 1.7562762023044465 | 0.6296896253909049 |
| form_b | js_to_global | -0.0746838 | 0.0746838 | 100 | 0.10389204405189059 | 0.06137239129252118 |
| form_b | update_cosine_distance_to_mean | -0.0119129 | 0.0119129 | 100 | 0.1713374406436421 | 0.10745471690532812 |
| form_b | normalized_entropy | 0.0357781 | 0.0357781 | 100 | 0.777420395326653 | 0.22268256994878355 |
| form_b | log_class_imbalance_ratio | -0.0626901 | 0.0626901 | 100 | 3.806954209693196 | 2.1589594422348592 |

## Validation

| form | task | n | lambda_mean | lambda_std | lambda_min | lambda_max | lambda_cv | lambda_delta_pearson | lambda_delta_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| form_a | ALL | 75 | 1 | 0.192501 | 0.562033 | 1.28403 | 0.192501 | 0.225294 | 0.277934 |
| form_b | ALL | 75 | 1 | 0.113467 | 0.771519 | 1.18793 | 0.113467 | 0.218448 | 0.292189 |

## Orthogonality Against q

| form | task | n | lambda_quality_pearson | lambda_quality_spearman | quality_delta_pearson | quality_delta_spearman | mean_effective_quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| form_a | ALL | 75 | -0.115337 | -0.0624751 | 0.0380474 | 0.0781142 | 0.476418 |
| form_b | ALL | 75 | -0.136893 | -0.205747 | 0.0380474 | 0.0781142 | 0.477448 |

## Leave-One-Task-Out Summary

| form | ridge_alpha | rmse | mae | spearman |
| --- | --- | --- | --- | --- |
| form_a |  | 5.94 | 4.39503 | 0.349083 |
| form_b | 0.01 | 6.55814 | 5.38198 | 0.0637457 |
| form_b | 0.1 | 6.5273 | 5.35126 | 0.0666029 |
| form_b | 1.0 | 6.33479 | 5.14366 | 0.0773721 |
| form_b | 10.0 | 6.01459 | 4.69659 | 0.0838715 |
| form_b | 100.0 | 5.83565 | 4.3522 | 0.210225 |

## Recommendation For Experiment 3

Use Form B as the primary candidate because it retains interpretability while
shrinking weak signals. Use Form A as the ablation baseline because it directly
tests the Experiment 1 conclusion that update L2 reward plus JS penalty is the
minimal sensible domain-aware construction.
