# Domain-Aware Aggregation Weight (λ) Formulation

## 1. Purpose

Experiment 2 introduces a calibrated Domain-Aware Aggregation Weight, denoted
\(\lambda\), to extend the Project 1 quality-weighted aggregation rule with a
domain-aware correction factor.

Project 1 aggregates clients using a weight proportional to the product of a
client aggregation weight and a client quality score:

$$
\text{Weight}_i = w_i \times q_i
$$

where \(w_i\) is the original aggregation weight component and \(q_i\) is the
quality score computed from local validation/training loss.

Experiment 2 extends this rule to:

$$
\text{Weight}_i = w_i \times q_i \times \lambda_i
$$

Historical regenerated-run values are recorded under
`docs/artifacts/p2-exp2-real-seed42/`. Earlier tables in this design note are
retained as historical context. The current conservative methodology supersedes
any unconditional form-selection language below: each candidate form must pass a
fold-safe intercept-only null gate using raw mean leave-one-task-out RMSE, and a
Ridge form must select an interior alpha. Unsupported forms are recorded as
diagnostics only and do not receive coefficients, gamma, lambda values, or
Experiment 3 treatment-arm eligibility.

## Regenerated real-data run (2026-09-12)

Experiment 1 used five real datasets, one seed (42), five rounds and 75
client-round observations. A prior Experiment 2 attempt selected ridge alpha
`1000.0` from the expanded grid. Under the current conservative methodology,
boundary-selected Ridge fits are not accepted as supported treatment arms. In
that historical attempt, Form A used gamma `2.444578`; Form B used gamma `5.0`.
Global
Spearman association with leave-one-client-out contribution was `0.3676` for
Form A and `0.3693` for Form B, with global R² `0.0788` and `0.0254`
respectively. Per-task behavior is heterogeneous, so this run does not claim
a universal preferred form or a validated allocator.

Use `docs/artifacts/p2-exp2-real-seed42/comparison_report.md` and its CSV
tables as the authoritative regenerated values.

The role of \(\lambda_i\) is to adjust the contribution of client \(i\) using
domain and update-space evidence measured in Experiment 1, while preserving the
Project 1 aggregation behavior whenever \(\lambda\) is disabled.

Experiment 2 has two distinct objectives:

- **Calibration objective:** estimate a mathematically grounded \(\lambda\)
  using the best available offline proxy for useful aggregation behavior,
  namely Leave-One-Client-Out contribution \((\Delta\mathrm{accuracy})\)
  measured in Experiment 1.
- **Research objective:** determine whether the calibrated \(\lambda\), when
  inserted into the federated aggregation rule, improves federated learning
  behavior. Experiment 2 does not answer this second question directly;
  Experiment 3 is required for that evaluation.

Thus, the regression model in Experiment 2 is used as a calibration mechanism
for constructing \(\lambda\). Its held-out prediction metrics are used to compare
candidate calibrations, not to claim that offline contribution prediction is the
final research endpoint.

## 2. Experiment 1 Motivation

Experiment 1 measured relationships between client contribution and several
domain or update-space signals. Its central finding was that no single signal
sufficiently explained client contribution.

The empirical evidence was:

- Jensen-Shannon divergence to the global label distribution showed a weak
  negative relationship with client contribution.
- KL divergence showed a similar weak negative relationship.
- Update L2 distance to the mean update showed the strongest positive monotonic
  relationship.
- Update cosine distance contributed little.
- Controlled regressions explained only a small portion of contribution
  variance.

Therefore, Experiment 2 does not define \(\lambda\) from a single domain signal.
Instead, it constructs a multi-factor, regularized, interpretable weight using
the signals that were empirically supported by Experiment 1.

## 3. Feature Definitions

The historical Ridge candidate is Form B. For each client-round observation
\(i\), Form B uses the following five features:

### Update L2 Feature

Let \(d^{\mathrm{L2}}_i\) be the Experiment 1 measurement
`update_l2_distance_to_mean`.

The implemented transformed feature is:

$$
x_{i,1} = \log(1 + \max(d^{\mathrm{L2}}_i, 0))
$$

This corresponds exactly to:

```python
np.log1p(update_l2_distance_to_mean.clip(lower=0.0))
```

### Jensen-Shannon Divergence Feature

Let \(d^{\mathrm{JS}}_i\) be the Experiment 1 measurement `js_to_global`.

The implemented feature is:

$$
x_{i,2} = d^{\mathrm{JS}}_i
$$

### Update Cosine Distance Feature

Let \(d^{\cos}_i\) be the Experiment 1 measurement
`update_cosine_distance_to_mean`.

The implemented feature is:

$$
x_{i,3} = d^{\cos}_i
$$

### Normalized Entropy Feature

Let \(H^{\mathrm{norm}}_i\) be the Experiment 1 measurement
`normalized_entropy`.

The implemented feature is:

$$
x_{i,4} = H^{\mathrm{norm}}_i
$$

### Class Imbalance Feature

Let \(r^{\mathrm{imb}}_i\) be the Experiment 1 measurement
`class_imbalance_ratio`.

Experiment 1 computes this ratio from the complete client class-count vector:

$$
\begin{aligned}
\mathcal{P}_i &= \{n_{i,c}: n_{i,c} > 0\} \\
r^{\mathrm{base}}_i
&= \frac{\max_{n \in \mathcal{P}_i} n}{\min_{n \in \mathcal{P}_i} n} \\
r^{\mathrm{imb}}_i
&= r^{\mathrm{base}}_i
\left(1 + \frac{|\{c: n_{i,c}=0\}|}{C}\right).
\end{aligned}
$$

Zero-count classes are excluded from the denominator by computing the base
ratio over positive class counts only. Missing classes are represented through
the finite multiplicative penalty
\(1 + \mathrm{zero\_class\_count}/\mathrm{num\_classes}\), so a one-class
client distribution is not treated as balanced. If a client has no samples, the
ratio is defined as \(0\) because no empirical class distribution exists.

This implemented measure remains finite, increases monotonically with observed
positive-count imbalance, and also increases monotonically as missing-class
severity grows.

The implemented transformed feature is:

$$
x_{i,5} = \log(1 + \max(r^{\mathrm{imb}}_i, 0))
$$

This corresponds exactly to:

```python
np.log1p(class_imbalance_ratio.clip(lower=0.0))
```

## 4. Feature Standardization

Each feature is standardized before applying the fitted coefficients. For
feature \(k\), the standardized value for client-round observation \(i\) is:

$$
z_{i,k} = \frac{x_{i,k} - \mu_k}{\sigma_k}
$$

where \(\mu_k\) and \(\sigma_k\) are computed over the Experiment 1 measurement
rows used for Experiment 2 calibration. The implementation uses population
standard deviation:

$$
\sigma_k = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_{i,k} - \mu_k)^2}
$$

If \(\sigma_k \le 10^{-12}\), the implementation replaces \(\sigma_k\) with
1.0.

The fitted means and standard deviations used by the final Form B are:

| Feature | Symbol | Mean \(\mu_k\) | Std. \(\sigma_k\) |
| --- | --- | ---: | ---: |
| `log_update_l2` | \(x_{i,1}\) | 1.8883672674985266 | 1.0210197854954943 |
| `js_to_global` | \(x_{i,2}\) | 0.10049756769211601 | 0.06212430993597597 |
| `update_cosine_distance_to_mean` | \(x_{i,3}\) | 0.5017276868494069 | 0.4161764311366351 |
| `normalized_entropy` | \(x_{i,4}\) | 0.7782734221962865 | 0.22432808028037712 |
| `log_class_imbalance_ratio` | \(x_{i,5}\) | 4.147386055045582 | 2.4874350966084178 |

Standardization is required because the raw signals are measured on different
scales. Without standardization, the fitted coefficient magnitudes would reflect
units of measurement rather than calibrated relative influence.

## 5. Final Mathematical Formulation

The historical Form B candidate is an interpretable ridge-calibrated linear
score. In the prior regenerated attempt, the ridge parameter selected by
leave-one-task-out validation was:

$$
\alpha_{\mathrm{ridge}} = 1000.0
$$

The fitted intercept is:

$$
\beta_0 = 0.0
$$

The final standardized score for client-round observation \(i\) is:

$$
\begin{aligned}
s_i
&= \beta_0
+ \beta_1 z_{i,1}
+ \beta_2 z_{i,2}
+ \beta_3 z_{i,3}
+ \beta_4 z_{i,4}
+ \beta_5 z_{i,5} \\
&= 0.0
+ 0.018210413515057963\,z_{i,1}
+ 0.006324207413926036\,z_{i,2}
+ 0.013998787520691848\,z_{i,3}
+ 0.004740807394714487\,z_{i,4}
+ 0.01754345937451533\,z_{i,5}.
\end{aligned}
$$

The variables are:

- \(z_{i,1}\): standardized `log_update_l2`
- \(z_{i,2}\): standardized `js_to_global`
- \(z_{i,3}\): standardized `update_cosine_distance_to_mean`
- \(z_{i,4}\): standardized `normalized_entropy`
- \(z_{i,5}\): standardized `log_class_imbalance_ratio`
- \(s_i\): raw standardized lambda score before positive mapping

This equation exactly matches the implemented Form B coefficients in
`outputs/exp2/fitted_coefficients.csv`.

## 6. Positive Mapping

Because aggregation weights must not become negative, the standardized score is
mapped through an exponential function.

For each aggregation context \(g = (\mathrm{task}, \mathrm{round})\), first
center the score:

$$
\tilde{s}_i = s_i - \frac{1}{|g|}\sum_{j \in g}s_j
$$

The implementation clips this centered score before exponentiation for
numerical stability:

$$
\tilde{s}^{\mathrm{clip20}}_i
= \min(\max(\tilde{s}_i, -20), 20)
$$

The raw positive lambda value is:

$$
\lambda^{\mathrm{raw}}_{i,f}
= \exp(\gamma_f \tilde{s}^{\mathrm{clip20}}_{i,f})
$$

with a form-specific implemented scale:

$$
\gamma_f,\quad f \in \{\mathrm{Form\ A}, \mathrm{Form\ B}\}.
$$

The exponential mapping ensures:

$$
\lambda^{\mathrm{raw}}_i > 0
$$

which is required because \(\lambda_i\) multiplicatively modifies the
aggregation weight.

### Origin of \(\gamma_f\)

The scale parameter \(\gamma_f\) is calibrated independently for each lambda
form. This controls the spread of each form's \(\lambda\) after exponential
mapping without pooling Form A and Form B score populations. For each form, the
Experiment 2 calibration code keeps the achieved coefficient of variation of
\(\lambda_f\) below the configured stability target:

$$
\mathrm{CV}(\lambda_f) \le \min(0.5\,\mathrm{CV}(q), 0.20).
$$

The implementation obtains \(\gamma_f\) by binary search over \([0, 5]\) using
only that form's fitted scores. After calibration, it records the target CV,
achieved CV, and \(\gamma_f\). If the achieved CV exceeds the target, the run
raises an exception instead of silently accepting an invalid calibration. The
scale is not an independently optimized scientific parameter and is not tuned
by rerunning federated learning. Its purpose is numerical: preserve enough
score variation for \(\lambda_f\) to express the calibrated evidence, while
preventing the exponential map from producing a factor that dominates the
existing quality score \(q\).

## 7. Clipping

After the positive mapping, the implementation enforces both the documented
bounds and the mean-one context invariant with an iterative clip-renormalize
procedure. Starting from \(\lambda^{(0)}\), each iteration clips to the
implemented range:

$$
\lambda^{(t,\mathrm{clip})}_i
= \min(\max(\lambda^{(t)}_i, 0.5), 1.5)
$$

and then renormalizes within aggregation context
\(g = (\mathrm{task}, \mathrm{round})\):

$$
\lambda^{(t+1)}_i
= \frac{\lambda^{(t,\mathrm{clip})}_i}
{\frac{1}{|g|}\sum_{j \in g}\lambda^{(t,\mathrm{clip})}_j}.
$$

Clipping prevents the domain-aware factor from overwhelming the original
quality-weighted aggregation term \(q_i\). It also improves numerical stability
and prevents isolated noisy domain measurements from producing extreme
aggregation weights.

## 8. Mean Normalization

The iterative procedure stops only after both invariants hold for the final
\(\lambda_i\):

$$
0.5 \le \lambda_i \le 1.5
$$

and

$$
\frac{1}{|g|}\sum_{i \in g}\lambda_i = 1
$$

If the implementation cannot satisfy both invariants, it raises an exception
instead of writing invalid lambda values.

Mean normalization is performed within each `(task, round)` because aggregation
occurs over the clients participating in a specific training round for a
specific task. This makes \(\lambda\) redistributive within an aggregation
context rather than globally increasing or decreasing the overall aggregation
scale.

## 9. Final Aggregation Equation

The final Experiment 2 aggregation rule is:

$$
\text{Weight}_i = w_i \times q_i \times \lambda_i
$$

where:

- \(w_i\) is the original Project 1 aggregation weight component for client
  \(i\), corresponding to client sample weighting before normalization.
- \(q_i\) is the Project 1 quality score:

  $$
  q_i = \frac{1}{1 + \mathrm{avg\_train\_loss}_i}
  $$

- \(\lambda_i\) is the final normalized Domain-Aware Aggregation Weight defined
  above.

In the implemented aggregation helper, the effective quality score is:

$$
q^{\mathrm{eff}}_i = q_i \lambda_i
$$

The existing Project 1 aggregator then receives \(q^{\mathrm{eff}}_i\), so its
existing weighting path becomes proportional to:

$$
w_i q^{\mathrm{eff}}_i
= w_i q_i \lambda_i.
$$

For normalized aggregation over clients in context \(g\), this is equivalent to:

$$
\bar{W}_i
= \frac{w_i q_i \lambda_i}
{\sum_{j \in g} w_j q_j \lambda_j}.
$$

## 10. Backward Compatibility

Experiment 2 implements \(\lambda\) as an optional extension.

If `lambda_weights=None`, then the implementation returns the original quality
scores unchanged:

$$
q^{\mathrm{eff}}_i = q_i
$$

and the aggregation rule is exactly the original Project 1 formulation:

$$
\text{Weight}_i = w_i \times q_i.
$$

If \(\lambda_i = 1\) for all clients, then:

$$
w_i q_i \lambda_i = w_i q_i,
$$

so the same Project 1 behavior is recovered.

## 11. Form Selection Gate

Experiment 2 constructed and compared two candidate lambda formulations.

Form A was a simple interpretable two-factor model:

$$
s^{A}_i
= 0.10284439088479502\,z(\log(1+d^{\mathrm{L2}}_i))
- 0.23141392711338707\,z(d^{\mathrm{JS}}_i).
$$

Form A directly tests the minimal hypothesis suggested by Experiment 1: reward
larger update L2 distance and penalize JS divergence from the global label
distribution.

Historically, Form B was treated as the primary formulation because it was the
most regularized candidate among the evaluated forms. Under the current
conservative methodology, that historical preference is not sufficient for
Experiment 3 eligibility. Form A and Form B are evaluated independently against
the fold-safe null comparator, and Form B must also select an interior Ridge
alpha. Its coefficients remain scientifically interpretable when the form is
supported:

- update L2 contributes positively;
- JS divergence contributes positively in this regenerated fit;
- update cosine distance contributes positively;
- normalized entropy contributes mildly positively;
- class imbalance contributes positively.

Leave-one-task-out validation selected the Form B ridge parameter
\(\alpha_{\mathrm{ridge}} = 1000.0\). The held-out \(R^2\) values remain
negative, and the predictive performance is weak. Form B should therefore be
interpreted as the least-bad calibrated formulation among the evaluated
candidates, not as a highly predictive contribution model.

Despite weak predictive performance, Form B produced lower mean RMSE and MAE
than Form A, applied ridge regularization, and yielded a more conservative
lambda distribution. These properties reduce the risk that \(\lambda\) overfits
the small Experiment 1 calibration set or dominates the existing Project 1
quality score \(q\).

Therefore, neither Form A nor Form B is carried forward automatically. Only a
form that passes the null-model gate and, for Ridge, the interior-alpha gate may
appear as a supported treatment arm in the Experiment 3 calibration bundle.

## 11A. Form C Relative-Contribution Candidate

Form C is a new post-hoc exploratory candidate, not a replacement for the
recorded Form A/Form B negative evidence. It was introduced after observing that
raw target scales differ strongly by task while lambda is applied within the
current task-round aggregation context.

Form C reuses the deployable Form B feature order:

$$
[\log ||\Delta_i||_2,\ JS(p_i || p_{\mathrm{global}}),\ d_{\cos}(\Delta_i,
\bar{\Delta}_g),\ H(p_i),\ \log(1+\mathrm{imbalance}_i)].
$$

For each task-round group \(g\), every predictor is population-standardized
within the group:

$$
\tilde{x}_{igk} =
\begin{cases}
0, & \sigma_{gk}=0 \\
\frac{x_{igk}-\mu_{gk}}{\sigma_{gk}}, & \sigma_{gk}>0
\end{cases}
$$

The calibration target is the within-group relative contribution:

$$
\tilde{y}_{ig} =
\begin{cases}
0, & \sigma_{yg}=0 \\
\frac{y_{ig}-\mu_{yg}}{\sigma_{yg}}, & \sigma_{yg}>0.
\end{cases}
$$

No rows are dropped for zero-variance predictors or targets. Form C fits Ridge
without an intercept because both predictors and target are group centered:

$$
s_i = \sum_k \beta_k \tilde{x}_{igk}.
$$

The fold-safe null prediction for Form C is zero. Support requires mean
group-normalized leave-one-task-out RMSE to beat that null by more than the
fixed tolerance, an interior selected Ridge alpha, finite calibration values,
and non-reversed predictive orientation. Ranking metrics and stratified
permutation tests remain secondary diagnostics.

If supported, Form C maps scores to lambda within the current aggregation
context:

$$
\lambda^{raw}_{ig} =
\mathrm{clip}\left(\exp(\gamma (s_{ig} - \bar{s}_g)),
\lambda_{\min}, \lambda_{\max}\right),
\qquad
\lambda_{ig} = \frac{\lambda^{raw}_{ig}}{\frac{1}{|g|}\sum_j \lambda^{raw}_{jg}}.
$$

Experiment 3 then applies the unchanged aggregation rule:

$$
p_i = \frac{w_i q_i \lambda_i}{\sum_j w_j q_j \lambda_j}.
$$

The true leave-one-out contribution is a training target only. Experiment 3
Form C inference uses only current task-round client features available before
aggregation.

## 12. Orthogonality Validation

Experiment 2 explicitly validated \(\lambda\) against the existing Project 1
quality score \(q\). The purpose of this validation was to check whether
\(\lambda\) contributes information distinct from \(q\), rather than duplicating
the same loss-derived quality signal.

The orthogonality analysis is reported in
`outputs/exp2/orthogonality_report.csv`. The overall correlations across all
75 Experiment 1 client-round measurements were:

| Form | Pearson corr. \((\lambda, q)\) | Spearman corr. \((\lambda, q)\) |
| --- | ---: | ---: |
| Form A | -0.11533724386672116 | -0.062475106685633 |
| Form B | -0.13689318370205766 | -0.20574679943100999 |

These weak overall correlations indicate that \(\lambda\) is globally distinct
from \(q\). This is consistent with the design goal: \(q\) measures client
quality through local loss, while \(\lambda\) models domain and update-space
structure.

However, the task-level analysis also shows that some datasets exhibit stronger
task-specific correlations. For example, CIFAR-CNN shows strong negative
task-level correlations between \(\lambda\) and \(q\). Therefore, Experiment 2
satisfies the global orthogonality objective, but task-level \(\lambda\)-\(q\)
behavior should continue to be monitored in Experiment 3.

The per-task \(\lambda\)-\(q\) correlations from
`outputs/exp2/orthogonality_report.csv` are:

| Form | Task | Pearson corr. \((\lambda, q)\) | Spearman corr. \((\lambda, q)\) |
| --- | --- | ---: | ---: |
| Form A | AGNews-LSTM | -0.4607450959405998 | -0.42499999999999993 |
| Form A | Audio-1DCNN | 0.22608755180938953 | 0.3 |
| Form A | CIFAR-CNN | -0.9421750624645309 | -0.8642857142857141 |
| Form A | Fashion-MLP | -0.6749107574704369 | -0.5714285714285713 |
| Form A | Tabular-MLP | 0.4929337929920553 | 0.5214285714285714 |
| Form B | AGNews-LSTM | -0.7638906372108178 | -0.4535714285714285 |
| Form B | Audio-1DCNN | 0.04104163294028534 | 0.25357142857142856 |
| Form B | CIFAR-CNN | -0.9144866080672905 | -0.825 |
| Form B | Fashion-MLP | -0.7980823743041282 | -0.5285714285714286 |
| Form B | Tabular-MLP | 0.5058887839884542 | 0.5714285714285713 |

The CIFAR-CNN exception is therefore quantitatively large: Form B has Pearson
correlation \(-0.9144866080672905\) and Spearman correlation \(-0.825\) between
\(\lambda\) and \(q\) on that task. This does not overturn the global
orthogonality result, but it is an important task-level limitation.

## 13. Held-Out Validation Summary

Experiment 2 used leave-one-task-out validation to evaluate the offline
calibration behavior of the fitted scores across the five benchmark tasks. The
validation results are reported in `outputs/exp2/cross_validation.csv` and summarized in
`outputs/exp2/comparison_report.md`.

The mean leave-one-task-out metrics were:

| Form | Ridge alpha | RMSE | MAE | Pearson | Spearman | \(R^2\) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Form A | n/a | 5.940004 | 4.395028 | 0.279438 | 0.349083 | -2.345528 |
| Form B | 0.01 | 6.558139 | 5.381980 | 0.064661 | 0.063746 | -2.507719 |
| Form B | 0.10 | 6.527303 | 5.351258 | 0.066049 | 0.066603 | -2.490587 |
| Form B | 1.00 | 6.334788 | 5.143657 | 0.077249 | 0.077372 | -2.404195 |
| Form B | 10.00 | 6.014593 | 4.696589 | 0.081332 | 0.083871 | -2.692933 |
| Form B | 100.00 | 5.835653 | 4.352204 | 0.140555 | 0.210225 | -4.065864 |

The negative \(R^2\) values mean that the held-out calibration models predict
Leave-One-Client-Out contribution worse than a constant-mean predictor on the
held-out task. This demonstrates weak predictive generalization of the offline
calibration model. It does not, by itself, prove that the aggregation
methodology fails, because the calibration model is only an intermediate step
used to estimate \(\lambda\).

Form B was not selected because it wins every validation metric. The regenerated
run shows similar global rank association for the two forms, while per-task
behavior varies substantially. Form B is retained as a documented candidate
because it uses all measured signals and explicit ridge regularization; this is
an exploratory calibration choice, not evidence of a validated allocator.

Thus, historical Form B selection is a documented diagnostic rather than an
automatic approval. Experiment 3 is required to determine whether any supported
calibrated \(\lambda\) formulation is beneficial when incorporated into the
federated aggregation process.

## 14. Interpretation of Predictive Performance

Leave-One-Client-Out contribution was used because it is the best available
offline proxy in Experiment 1 for whether a client update was useful to the
current aggregation round. The regression models in Experiment 2 therefore use
contribution prediction as a calibration mechanism: they translate observed
domain and update-space signals into a bounded multiplicative aggregation
factor.

Weak held-out predictive performance does not automatically invalidate
\(\lambda\). Aggregation weights do not necessarily require strong standalone
prediction of \(\Delta\mathrm{accuracy}\) to be useful, because the final
aggregation behavior depends on the interaction among \(w_i\), \(q_i\),
\(\lambda_i\), LoRA update geometry, client sampling, and the subsequent
training trajectory. Offline regression metrics only evaluate the calibration
proxy in isolation.

At the same time, the weak predictive performance is an important limitation.
It means \(\lambda\) should be interpreted cautiously as a conservative,
evidence-calibrated weighting factor, not as a precise contribution estimator.
The final effectiveness of

$$
\text{Weight}_i = w_i \times q_i \times \lambda_i
$$

must be evaluated inside the federated learning process. Experiment 3 is the
first experiment capable of answering whether the calibrated \(\lambda\)
improves federated learning performance.

## 15. Sample Count Discussion

The final \(\lambda\) formulation does not include sample count as a feature.
The implemented Form B feature set is:

$$
\{\log(1+d^{\mathrm{L2}}),\ d^{\mathrm{JS}},\ d^{\cos},\
H^{\mathrm{norm}},\ \log(1+r^{\mathrm{imb}})\}.
$$

There is no evidence in the Experiment 2 implementation that sample count was
explicitly evaluated during the original coefficient-selection step. This is a
scientific limitation of the current calibration record.

However, sample count is already represented in the aggregation rule through
the original Project 1 aggregation weight component \(w_i\). The implemented
optional aggregation wrapper computes effective quality as:

$$
q^{\mathrm{eff}}_i = q_i\lambda_i,
$$

and the existing aggregation path then weights clients proportionally to:

$$
w_i q^{\mathrm{eff}}_i = w_i q_i\lambda_i.
$$

Because \(w_i\) already contains the sample-weighting component, adding sample
count again inside \(\lambda_i\) could double-count client size. For this
reason, \(\lambda\) is defined as a domain-heterogeneity and update-space
correction rather than a dataset-size correction.

A supplementary sensitivity analysis was added in
`outputs/exp2/sample_count_sensitivity.csv` and
`outputs/exp2/sample_count_sensitivity.md`. In the existing Experiment 1
measurements, `partition_samples` has weak Pearson correlation with
`delta_accuracy` \((0.034332855764779756)\), moderate Pearson correlation with
`quality_score` \((0.38522876886650376)\), and modest Pearson correlation with
Form B \(\lambda\) \((0.24314621260356034)\). A simple regression
`delta_accuracy ~ partition_samples` gives \(R^2 = 0.0011787449849652853\).

These results support documenting sample count as a monitored confound rather
than changing the frozen \(\lambda\) formulation.

## 16. Leave-One-Task-Out vs. Task Fixed Effects

Task fixed effects were intentionally omitted from the final Experiment 2
formulation. The calibration objective is not to maximize within-task regression
fit. The research objective is to construct a dataset-agnostic \(\lambda\)
candidate for evaluation across different federated learning tasks.

Task fixed effects can improve within-task fit by allowing task-specific
offsets. However, such offsets are tied to the identities of the training tasks
and do not directly define a portable aggregation rule for new or held-out
tasks.

Leave-one-task-out validation instead asks how coefficients fitted on four
tasks behave on the fifth task. This directly evaluates the cross-task behavior
needed for a dataset-agnostic \(\lambda\) calibration. For that reason,
leave-one-task-out validation is more closely aligned with the purpose of
Experiment 2 than task fixed effects, even though the resulting predictive
performance remains weak.

## 17. Limitations

Experiment 2 is complete as a calibration and validation step, but the following
limitations should be carried into Experiment 3:

- Held-out predictive performance is weak. The leave-one-task-out \(R^2\) values
  are negative, meaning the offline calibration model predicts held-out
  contribution worse than a constant-mean predictor.
- Calibration is based on a single completed Experiment 1 output set.
- The calibration inherits the single random seed used by Experiment 1.
- The calibration inherits one Dirichlet partition setting from Experiment 1.
- Global \(\lambda\)-\(q\) orthogonality is weak, but CIFAR-CNN shows a strong
  task-level exception.
- The form-specific scale \(\gamma_f\) is empirically calibrated to control
  each \(\lambda_f\)'s spread; it is not independently optimized through
  federated learning runs.
- \(\lambda\) has been validated offline using Experiment 1 measurements only.
- Experiment 3 is required to determine whether the frozen
  \(w_i \times q_i \times \lambda_i\) aggregation rule improves federated
  learning performance.

## 18. Summary

Experiment 2 defines candidate positive, calibrated, multi-factor Domain-Aware
Aggregation Weights \(\lambda\). A final supported formulation exists only if
the candidate passes the null-model gate and, for Ridge, the interior-alpha
gate. Supported forms use standardized Experiment 1 signals, exponential
positive mapping, and iterative bounded mean-one normalization within each
`(task, round)` aggregation context.

The mathematical contribution of Experiment 2 is the construction and
validation of:

$$
\text{Weight}_i = w_i \times q_i \times \lambda_i
$$

as an optional extension of Project 1's original:

$$
\text{Weight}_i = w_i \times q_i.
$$

Experiment 2 calibrates eligible \(\lambda\) candidates, validates their
mathematical properties, validates boundedness and stability, validates global
orthogonality against \(q\), documents limitations, and prepares an Experiment 3
calibration bundle only when at least one treatment form is supported.

Experiment 2 does not prove that \(\lambda\) improves federated learning.
Experiment 3 is the first experiment capable of answering whether

$$
\text{Weight}_i = w_i \times q_i \times \lambda_i
$$

actually improves federated learning performance when used during aggregation.
