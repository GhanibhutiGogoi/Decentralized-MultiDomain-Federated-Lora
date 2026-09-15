# Experiment 2 Evaluation Report

## Scope

This report is generated from Experiment 2 output tables. It reports regression,
ranking, and statistical evaluation metrics without selecting a preferred lambda
form or changing the calibration method.

## Inputs

Experiment 1 signal tables:

| predictor | is_synthetic | pearson | spearman | n |
| --- | --- | --- | --- | --- |
| js_to_global | False | 0.0681522 | 0.1391 | 75 |
| kl_to_global | False | 0.0778893 | 0.180395 | 75 |
| update_cosine_distance_to_mean | False | 0.0273695 | 0.0545232 | 75 |
| update_l2_distance_to_mean | False | 0.191474 | 0.502771 | 75 |

Experiment 1 controlled regressions:

| model_predictor | term | is_synthetic | standardized_beta | standard_error | r_squared | n |
| --- | --- | --- | --- | --- | --- | --- |
| js_to_global | js_to_global | False | 0.0663171 | 0.117113 | 0.034352 | 75 |
| js_to_global | adaptive_rank | False | 0.0137135 | 0.117153 | 0.034352 | 75 |
| js_to_global | local_loss | False | -0.171387 | 0.116706 | 0.034352 | 75 |
| kl_to_global | kl_to_global | False | 0.0785726 | 0.116828 | 0.0361313 | 75 |
| kl_to_global | adaptive_rank | False | 0.0135178 | 0.116887 | 0.0361313 | 75 |
| kl_to_global | local_loss | False | -0.172437 | 0.116574 | 0.0361313 | 75 |
| update_cosine_distance_to_mean | update_cosine_distance_to_mean | False | 0.0223963 | 0.117379 | 0.0304879 | 75 |
| update_cosine_distance_to_mean | adaptive_rank | False | 0.00572904 | 0.117399 | 0.0304879 | 75 |
| update_cosine_distance_to_mean | local_loss | False | -0.172244 | 0.116945 | 0.0304879 | 75 |
| update_l2_distance_to_mean | update_l2_distance_to_mean | False | 0.209898 | 0.11562 | 0.0730202 | 75 |
| update_l2_distance_to_mean | adaptive_rank | False | 0.0360309 | 0.115376 | 0.0730202 | 75 |
| update_l2_distance_to_mean | local_loss | False | -0.187007 | 0.11459 | 0.0730202 | 75 |

## Lambda Forms

Form A features: `log_update_l2, js_to_global`.

Form B features: `log_update_l2, js_to_global, update_cosine_distance_to_mean, normalized_entropy, log_class_imbalance_ratio`.

Form C label: `Form C - within-round relative-contribution calibration`.

Form C features: `log_update_l2, js_to_global, update_cosine_distance_to_mean, normalized_entropy, log_class_imbalance_ratio`.

Form C was introduced after observing cross-task target-scale mismatch in the
original A/B negative result. It is post-hoc and exploratory at calibration
time; Experiment 3 is required for independent confirmation if it is supported.

Selected Ridge alpha remains RMSE-based: `100.0`.

## Null-Model Validity Gate

Experiment 2 reports a fold-safe intercept-only null comparator. A lambda form is
supported only when its raw mean leave-one-task-out RMSE is strictly lower than
the fold-safe null by more than the documented numerical tolerance; Ridge forms
must also select an interior alpha.

| form | status | primary_metric_name | model_mean_rmse | null_mean_rmse | absolute_improvement | relative_improvement | selected_alpha | alpha_boundary_status | coefficient_available | gamma_lambda_available | rejection_reason | source_run_identity | null_improvement_tolerance | methodology_label | exploratory_status | feature_order | target_transformation | coefficient_norm | prediction_variance | zero_variance_feature_group_count | zero_variance_target_group_count | experiment3_eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| form_a | unsupported | mean_leave_one_task_out_rmse | 4.88218 | 4.38708 | -0.495106 | -0.112856 |  | not_applicable | False | False | model_mean_rmse_not_strictly_better_than_fold_safe_null | 49033dc9c5623f4edb0c64407249ab820b94922836b181683d601bbb721a01a4 | 1e-09 |  |  |  |  |  |  | 0 | 0 | False |
| form_b | unsupported | mean_leave_one_task_out_rmse | 4.57326 | 4.38708 | -0.186183 | -0.042439 | 100.0 | maximum | False | False | model_mean_rmse_not_strictly_better_than_fold_safe_null | 49033dc9c5623f4edb0c64407249ab820b94922836b181683d601bbb721a01a4 | 1e-09 |  |  |  |  |  |  | 0 | 0 | False |
| form_c | unsupported | mean_leave_one_task_out_rmse | 1.07847 | 0.978885 | -0.0995853 | -0.101733 | 100.0 | maximum | False | False | model_mean_rmse_not_strictly_better_than_fold_safe_null | 49033dc9c5623f4edb0c64407249ab820b94922836b181683d601bbb721a01a4 | 1e-09 | Form C - within-round relative-contribution calibration | post_hoc_exploratory | log_update_l2|js_to_global|update_cosine_distance_to_mean|normalized_entropy|log_class_imbalance_ratio | within_task_round_zscore_delta_accuracy | 0.12406791116646791 | 0.034703871870000194 | 0 | 1 | False |

## Gamma Calibration

_No rows._

## Coefficients

_No rows._

## Regression And Ranking Metrics

Global metrics:

_No rows._

Per-task metrics:

_No rows._

## Alpha Comparison

| form | ridge_alpha | rmse | mae | r_squared | pearson | spearman | pairwise_ranking_accuracy | kendall_tau | permutation_p_value | null_rmse | null_mae | null_spearman | coefficient_norm | prediction_variance | zero_variance_feature_group_count | zero_variance_target_group_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| form_a |  | 4.88218 | 3.91947 | -92.4636 | -0.191028 | -0.257431 | 0.406523 | -0.177804 | 0.840759 | nan | nan | nan | nan | nan | nan | nan |
| form_b | 0.01 | 6.14012 | 4.88126 | -283.31 | -0.28561 | -0.384815 | 0.350934 | -0.269886 | 1 | nan | nan | nan | nan | nan | nan | nan |
| form_b | 0.1 | 6.11334 | 4.86923 | -282.412 | -0.286231 | -0.381957 | 0.354744 | -0.262267 | 1 | nan | nan | nan | nan | nan | nan | nan |
| form_b | 1.0 | 5.96054 | 4.80042 | -274.377 | -0.297788 | -0.417073 | 0.335287 | -0.300744 | 1 | nan | nan | nan | nan | nan | nan | nan |
| form_b | 10.0 | 5.40554 | 4.40024 | -228.095 | -0.330808 | -0.370209 | 0.35174 | -0.277552 | 1 | nan | nan | nan | nan | nan | nan | nan |
| form_b | 100.0 | 4.57326 | 3.56702 | -130.138 | -0.172257 | -0.141827 | 0.43923 | -0.106595 | 0.656344 | nan | nan | nan | nan | nan | nan | nan |
| form_c | 0.01 | 1.84272 | 1.60589 | -3.11835 | -0.639388 | -0.620238 | 0.251289 | -0.475397 | 1 | 0.978885 | 0.861433 | 0 | 2.28953 | 1.71378 | 0 | 0.2 |
| form_c | 0.1 | 1.5425 | 1.35673 | -1.54915 | -0.644734 | -0.643809 | 0.245575 | -0.486825 | 1 | 0.978885 | 0.861433 | 0 | 1.56496 | 0.572664 | 0 | 0.2 |
| form_c | 1.0 | 1.29802 | 1.16617 | -0.80745 | -0.624757 | -0.59382 | 0.269349 | -0.446225 | 1 | 0.978885 | 0.861433 | 0 | 0.789638 | 0.212774 | 0 | 0.2 |
| form_c | 10.0 | 1.19627 | 1.07926 | -0.557964 | -0.377799 | -0.397792 | 0.346492 | -0.304917 | 0.821379 | 0.978885 | 0.861433 | 0 | 0.334633 | 0.149935 | 0 | 0.2 |
| form_c | 100.0 | 1.07847 | 0.963162 | -0.232651 | -0.341101 | -0.288287 | 0.389178 | -0.226339 | 0.69031 | 0.978885 | 0.861433 | 0 | 0.124068 | 0.0347039 | 0 | 0.2 |
| null |  | 4.38708 | 3.46878 | -62.631 | 0 | 0 | 0.5 | 0 | 1 | nan | nan | nan | nan | nan | nan | nan |

## Lambda Validation

_No rows._

## Orthogonality Against q

_No rows._

## Leave-One-Task-Out Details

| form | held_out_task | ridge_alpha | n_train | n_test | rmse | mae | r_squared | pearson | spearman | pairwise_ranking_accuracy | kendall_tau | permutation_p_value | null_rmse | null_mae | null_spearman | coefficient_norm | prediction_variance | zero_variance_feature_group_count | zero_variance_target_group_count | methodology_label | exploratory_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| null | AGNews-LSTM |  | 60 | 15 | 1.39586 | 1.38846 | -93.5406 | 0 | 0 | 0.5 | 0 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_a | AGNews-LSTM |  | 60 | 15 | 1.08584 | 1.00565 | -56.2096 | 0.220534 | 0.152605 | 0.56 | 0.117108 | 0.203796 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | AGNews-LSTM | 0.01 | 60 | 15 | 2.99628 | 1.85897 | -434.61 | -0.0829812 | -0.360867 | 0.35 | -0.29277 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | AGNews-LSTM | 0.1 | 60 | 15 | 2.98423 | 1.85175 | -431.114 | -0.0829375 | -0.360867 | 0.35 | -0.29277 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | AGNews-LSTM | 1.0 | 60 | 15 | 2.86986 | 1.78616 | -398.628 | -0.08182 | -0.330346 | 0.36 | -0.273252 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | AGNews-LSTM | 10.0 | 60 | 15 | 2.13189 | 1.38398 | -219.529 | -0.0616287 | -0.0233397 | 0.47 | -0.058554 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | AGNews-LSTM | 100.0 | 60 | 15 | 1.20706 | 1.12941 | -69.6954 | 0.0478141 | 0.159787 | 0.53 | 0.058554 | 0.257742 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_c | AGNews-LSTM | 0.01 | 60 | 15 | 1.41016 | 1.26087 | -1.48569 | -0.545069 | -0.619399 | 0.26 | -0.468432 | 1 | 0.894427 | 0.715859 | 0 | 0.829316 | 0.499461 | 0 | 1 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | AGNews-LSTM | 0.1 | 60 | 15 | 1.4096 | 1.26001 | -1.48373 | -0.547657 | -0.619399 | 0.26 | -0.468432 | 1 | 0.894427 | 0.715859 | 0 | 0.80744 | 0.496603 | 0 | 1 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | AGNews-LSTM | 1.0 | 60 | 15 | 1.40213 | 1.25004 | -1.45747 | -0.566684 | -0.658896 | 0.25 | -0.48795 | 1 | 0.894427 | 0.715859 | 0 | 0.667351 | 0.470578 | 0 | 1 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | AGNews-LSTM | 10.0 | 60 | 15 | 1.32203 | 1.1668 | -1.18471 | -0.626471 | -0.545789 | 0.31 | -0.370842 | 1 | 0.894427 | 0.715859 | 0 | 0.428356 | 0.3169 | 0 | 1 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | AGNews-LSTM | 100.0 | 60 | 15 | 1.07754 | 0.90904 | -0.451365 | -0.6777 | -0.500905 | 0.33 | -0.331806 | 1 | 0.894427 | 0.715859 | 0 | 0.180776 | 0.0611954 | 0 | 1 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| null | Audio-1DCNN |  | 60 | 15 | 1.31731 | 1.3143 | -217.837 | 0 | 0 | 0.5 | 0 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_a | Audio-1DCNN |  | 60 | 15 | 1.77757 | 1.75955 | -397.472 | -0.577112 | -0.644887 | 0.228916 | -0.482035 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Audio-1DCNN | 0.01 | 60 | 15 | 2.76524 | 2.73394 | -963.295 | -0.73527 | -0.711599 | 0.180723 | -0.56773 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Audio-1DCNN | 0.1 | 60 | 15 | 2.76411 | 2.73336 | -962.513 | -0.735807 | -0.711599 | 0.180723 | -0.56773 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Audio-1DCNN | 1.0 | 60 | 15 | 2.75552 | 2.72809 | -956.53 | -0.739377 | -0.731984 | 0.168675 | -0.589154 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Audio-1DCNN | 10.0 | 60 | 15 | 2.6904 | 2.66847 | -911.805 | -0.751677 | -0.752368 | 0.156627 | -0.610578 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Audio-1DCNN | 100.0 | 60 | 15 | 2.14351 | 2.13017 | -578.421 | -0.768258 | -0.752368 | 0.156627 | -0.610578 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_c | Audio-1DCNN | 0.01 | 60 | 15 | 3.38237 | 2.81626 | -10.4404 | -0.700091 | -0.710361 | 0.172414 | -0.596377 | 1 | 1 | 0.835401 | 0 | 5.79281 | 6.79149 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Audio-1DCNN | 0.1 | 60 | 15 | 1.97922 | 1.6503 | -2.91733 | -0.720472 | -0.710361 | 0.172414 | -0.596377 | 1 | 1 | 0.835401 | 0 | 2.51966 | 1.28433 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Audio-1DCNN | 1.0 | 60 | 15 | 1.18023 | 1.02299 | -0.392948 | -0.598463 | -0.429521 | 0.356322 | -0.261569 | 1 | 1 | 0.835401 | 0 | 0.460581 | 0.0719315 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Audio-1DCNN | 10.0 | 60 | 15 | 1.05578 | 0.921827 | -0.114672 | -0.244792 | -0.352427 | 0.344828 | -0.282494 | 1 | 1 | 0.835401 | 0 | 0.204554 | 0.0299477 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Audio-1DCNN | 100.0 | 60 | 15 | 1.00815 | 0.860138 | -0.0163629 | -0.109663 | 0.139502 | 0.528736 | 0.0523137 | 0.344655 | 1 | 0.835401 | 0 | 0.0609132 | 0.00346062 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| null | CIFAR-CNN |  | 60 | 15 | 1.71122 | 1.29414 | -0.00265543 | 0 | 0 | 0.5 | 0 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_a | CIFAR-CNN |  | 60 | 15 | 4.88965 | 4.57991 | -7.18648 | 0.106867 | -0.0214286 | 0.495238 | -0.00952381 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | CIFAR-CNN | 0.01 | 60 | 15 | 7.18088 | 6.66865 | -16.6562 | -0.152353 | -0.146429 | 0.466667 | -0.0666667 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | CIFAR-CNN | 0.1 | 60 | 15 | 7.14728 | 6.63936 | -16.4914 | -0.15164 | -0.146429 | 0.466667 | -0.0666667 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | CIFAR-CNN | 1.0 | 60 | 15 | 6.83204 | 6.356 | -14.9825 | -0.145368 | -0.160714 | 0.457143 | -0.0857143 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | CIFAR-CNN | 10.0 | 60 | 15 | 4.98199 | 4.59172 | -7.49862 | -0.0865063 | -0.160714 | 0.457143 | -0.0857143 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | CIFAR-CNN | 100.0 | 60 | 15 | 2.38417 | 1.96402 | -0.94633 | 0.635919 | 0.596429 | 0.714286 | 0.428571 | 0.023976 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_c | CIFAR-CNN | 0.01 | 60 | 15 | 1.16597 | 1.06107 | -0.359495 | -0.507193 | -0.575 | 0.285714 | -0.428571 | 1 | 1 | 0.922364 | 0 | 0.830859 | 0.077359 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | CIFAR-CNN | 0.1 | 60 | 15 | 1.16702 | 1.06197 | -0.361941 | -0.507701 | -0.692857 | 0.257143 | -0.485714 | 1 | 1 | 0.922364 | 0 | 0.802816 | 0.0781257 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | CIFAR-CNN | 1.0 | 60 | 15 | 1.17281 | 1.06705 | -0.375484 | -0.510995 | -0.785714 | 0.190476 | -0.619048 | 1 | 1 | 0.922364 | 0 | 0.619235 | 0.0822983 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | CIFAR-CNN | 10.0 | 60 | 15 | 1.15599 | 1.05595 | -0.336316 | -0.522195 | -0.785714 | 0.190476 | -0.619048 | 1 | 1 | 0.922364 | 0 | 0.327415 | 0.066663 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | CIFAR-CNN | 100.0 | 60 | 15 | 1.06344 | 0.981154 | -0.130898 | -0.54615 | -0.785714 | 0.190476 | -0.619048 | 1 | 1 | 0.922364 | 0 | 0.129039 | 0.0118738 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| null | Fashion-MLP |  | 60 | 15 | 9.06321 | 8.33008 | -1.45037 | 0 | 0 | 0.5 | 0 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_a | Fashion-MLP |  | 60 | 15 | 8.77067 | 7.99758 | -1.29474 | -0.516184 | -0.771429 | 0.219048 | -0.561905 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Fashion-MLP | 0.01 | 60 | 15 | 9.62284 | 8.62641 | -1.76232 | -0.10411 | -0.239286 | 0.419048 | -0.161905 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Fashion-MLP | 0.1 | 60 | 15 | 9.5308 | 8.56905 | -1.70973 | -0.117995 | -0.225 | 0.438095 | -0.12381 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Fashion-MLP | 1.0 | 60 | 15 | 9.14014 | 8.29858 | -1.49214 | -0.244472 | -0.396429 | 0.352381 | -0.295238 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Fashion-MLP | 10.0 | 60 | 15 | 8.87508 | 8.06184 | -1.3497 | -0.550528 | -0.771429 | 0.219048 | -0.561905 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Fashion-MLP | 100.0 | 60 | 15 | 8.92905 | 8.13461 | -1.37836 | -0.590156 | -0.646429 | 0.295238 | -0.409524 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_c | Fashion-MLP | 0.01 | 60 | 15 | 1.56286 | 1.43727 | -1.44252 | -0.874809 | -0.7 | 0.247619 | -0.504762 | 1 | 1 | 0.915992 | 0 | 1.00881 | 0.373395 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Fashion-MLP | 0.1 | 60 | 15 | 1.56226 | 1.43669 | -1.44066 | -0.875294 | -0.7 | 0.247619 | -0.504762 | 1 | 1 | 0.915992 | 0 | 0.974133 | 0.372388 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Fashion-MLP | 1.0 | 60 | 15 | 1.55642 | 1.43107 | -1.42246 | -0.878498 | -0.7 | 0.247619 | -0.504762 | 1 | 1 | 0.915992 | 0 | 0.744322 | 0.363356 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Fashion-MLP | 10.0 | 60 | 15 | 1.50559 | 1.38404 | -1.2668 | -0.884582 | -0.7 | 0.247619 | -0.504762 | 1 | 1 | 0.915992 | 0 | 0.389512 | 0.299156 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Fashion-MLP | 100.0 | 60 | 15 | 1.27748 | 1.17525 | -0.631957 | -0.889316 | -0.689286 | 0.257143 | -0.485714 | 1 | 1 | 0.915992 | 0 | 0.190611 | 0.0921199 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| null | Tabular-MLP |  | 60 | 15 | 8.4478 | 5.01695 | -0.324607 | 0 | 0 | 0.5 | 0 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_a | Tabular-MLP |  | 60 | 15 | 7.88718 | 4.25465 | -0.154634 | -0.189245 | -0.00201685 | 0.529412 | 0.0473381 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Tabular-MLP | 0.01 | 60 | 15 | 8.13535 | 4.51831 | -0.228437 | -0.353337 | -0.465892 | 0.338235 | -0.26036 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Tabular-MLP | 0.1 | 60 | 15 | 8.14027 | 4.55263 | -0.229924 | -0.342773 | -0.465892 | 0.338235 | -0.26036 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Tabular-MLP | 1.0 | 60 | 15 | 8.20513 | 4.83327 | -0.249602 | -0.277902 | -0.465892 | 0.338235 | -0.26036 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Tabular-MLP | 10.0 | 60 | 15 | 8.34832 | 5.29521 | -0.293597 | -0.203698 | -0.143196 | 0.455882 | -0.0710072 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_b | Tabular-MLP | 100.0 | 60 | 15 | 8.20252 | 4.4769 | -0.248806 | -0.186603 | -0.0665561 | 0.5 | 0 | 1 | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| form_c | Tabular-MLP | 0.01 | 60 | 15 | 1.69221 | 1.454 | -1.86358 | -0.569778 | -0.496429 | 0.290698 | -0.378842 | 1 | 1 | 0.917547 | 0 | 2.98583 | 0.827171 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Tabular-MLP | 0.1 | 60 | 15 | 1.5944 | 1.37465 | -1.54211 | -0.572546 | -0.496429 | 0.290698 | -0.378842 | 1 | 1 | 0.917547 | 0 | 2.72077 | 0.631871 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Tabular-MLP | 1.0 | 60 | 15 | 1.17851 | 1.05969 | -0.388897 | -0.569143 | -0.394969 | 0.302326 | -0.357796 | 1 | 1 | 0.917547 | 0 | 1.4567 | 0.0757042 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Tabular-MLP | 10.0 | 60 | 15 | 0.941977 | 0.867703 | 0.112679 | 0.389047 | 0.394969 | 0.639535 | 0.252562 | 0.106893 | 1 | 0.917547 | 0 | 0.32333 | 0.0370095 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |
| form_c | Tabular-MLP | 100.0 | 60 | 15 | 0.965748 | 0.890226 | 0.0673309 | 0.517322 | 0.394969 | 0.639535 | 0.252562 | 0.106893 | 1 | 0.917547 | 0 | 0.0590011 | 0.00486966 | 0 | 0 | Form C - within-round relative-contribution calibration | post_hoc_exploratory |

## Statistical Tests

Permutation p-values are computed as one-sided tests for positive Spearman rank
association using `1000` permutations and seed
`42`. Pooled evaluations stratify
permutations by the configured aggregation context columns when those columns
are available.

## Figures

_No rows._

## Decision Status

Forms A and B remain recorded under their original raw-RMSE/null evaluation.
Unsupported forms do not receive coefficients, gamma, lambda values, or
Experiment 3 treatment-arm eligibility. If the support table contains no
supported treatment form, Experiment 3 is blocked because no scientifically
runnable domain-aware calibration bundle is emitted.
