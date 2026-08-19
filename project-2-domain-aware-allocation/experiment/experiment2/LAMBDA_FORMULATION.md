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

The role of \(\lambda_i\) is to adjust the contribution of client \(i\) using
domain and update-space evidence measured in Experiment 1, while preserving the
Project 1 aggregation behavior whenever \(\lambda\) is disabled.

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

The final selected formulation is Form B. For each client-round observation
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
| `log_update_l2` | \(x_{i,1}\) | 1.7562762023044465 | 0.6296896253909049 |
| `js_to_global` | \(x_{i,2}\) | 0.10389204405189059 | 0.06137239129252118 |
| `update_cosine_distance_to_mean` | \(x_{i,3}\) | 0.1713374406436421 | 0.10745471690532812 |
| `normalized_entropy` | \(x_{i,4}\) | 0.777420395326653 | 0.22268256994878355 |
| `log_class_imbalance_ratio` | \(x_{i,5}\) | 3.806954209693196 | 2.1589594422348592 |

Standardization is required because the raw signals are measured on different
scales. Without standardization, the fitted coefficient magnitudes would reflect
units of measurement rather than calibrated relative influence.

## 5. Final Mathematical Formulation

The final selected formulation is Form B, an interpretable ridge-calibrated
linear score. The ridge parameter selected by leave-one-task-out validation is:

$$
\alpha_{\mathrm{ridge}} = 100.0
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
+ 0.04690965790495837\,z_{i,1}
- 0.07468377369916912\,z_{i,2}
- 0.01191288883367715\,z_{i,3}
+ 0.035778085198657245\,z_{i,4}
- 0.0626900688304372\,z_{i,5}.
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
\lambda^{\mathrm{raw}}_i
= \exp(\gamma \tilde{s}^{\mathrm{clip20}}_i)
$$

with the implemented scale:

$$
\gamma = 0.8863940762603306
$$

The exponential mapping ensures:

$$
\lambda^{\mathrm{raw}}_i > 0
$$

which is required because \(\lambda_i\) multiplicatively modifies the
aggregation weight.

## 7. Clipping

After the positive mapping, the implementation first normalizes raw lambda by
the raw context mean:

$$
\lambda^{\mathrm{mean-raw}}_i
= \frac{\lambda^{\mathrm{raw}}_i}
{\frac{1}{|g|}\sum_{j \in g}\lambda^{\mathrm{raw}}_j}
$$

Then it clips the result to the implemented range:

$$
\lambda^{\mathrm{clip}}_i
= \min(\max(\lambda^{\mathrm{mean-raw}}_i, 0.5), 1.5)
$$

The clipping range is therefore:

$$
0.5 \le \lambda^{\mathrm{clip}}_i \le 1.5
$$

Clipping prevents the domain-aware factor from overwhelming the original
quality-weighted aggregation term \(q_i\). It also improves numerical stability
and prevents isolated noisy domain measurements from producing extreme
aggregation weights.

## 8. Mean Normalization

After clipping, the implementation renormalizes lambda within each aggregation
context \(g = (\mathrm{task}, \mathrm{round})\):

$$
\lambda_i
= \frac{\lambda^{\mathrm{clip}}_i}
{\frac{1}{|g|}\sum_{j \in g}\lambda^{\mathrm{clip}}_j}
$$

This guarantees:

$$
\frac{1}{|g|}\sum_{i \in g}\lambda_i = 1
$$

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

## 11. Why Form B Was Selected

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

Form B was selected as the primary formulation because it preserves this
interpretable structure while using ridge regularization to shrink weak or
uncertain signals. Its coefficients remain scientifically interpretable:

- update L2 contributes positively;
- JS divergence contributes negatively;
- update cosine distance is shrunk close to zero;
- normalized entropy contributes mildly positively;
- class imbalance contributes negatively.

Leave-one-task-out validation selected the Form B ridge parameter
\(\alpha_{\mathrm{ridge}} = 100.0\). Form B also produced a more conservative
lambda distribution than Form A, reducing the risk that \(\lambda\) dominates
the existing Project 1 quality score \(q\).

Therefore, Form B is the primary \(\lambda\) formulation carried forward into
Experiment 3. Form A is retained as an interpretable ablation baseline.

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

## 13. Held-Out Validation Summary

Experiment 2 used leave-one-task-out validation to evaluate whether the fitted
scores generalize across the five benchmark tasks. The validation results are
reported in `outputs/exp2/cross_validation.csv` and summarized in
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

Form B was not selected because it wins every validation metric. Form A retains
stronger rank-order behavior, as shown by its higher mean Spearman correlation.
Form B was selected as the primary Experiment 3 formulation because the
\(\alpha_{\mathrm{ridge}} = 100.0\) model provides lower mean RMSE and MAE,
more conservative \(\lambda\) values, stronger numerical stability, explicit
ridge regularization, and lower risk of overfitting weak Experiment 1 signals.

Thus, the Form B selection is a scientific trade-off. Form B is the conservative
primary candidate, while Form A remains the interpretable ablation baseline.

## 14. Sample Count Discussion

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

## 15. Leave-One-Task-Out vs. Task Fixed Effects

Task fixed effects were intentionally omitted from the final Experiment 2
formulation. The goal of Experiment 2 is not to maximize within-task regression
fit. The goal is to construct a dataset-agnostic \(\lambda\) that can generalize
across different federated learning tasks.

Task fixed effects can improve within-task fit by allowing task-specific
offsets. However, such offsets are tied to the identities of the training tasks
and do not directly define a portable aggregation rule for new or held-out
tasks.

Leave-one-task-out validation instead asks whether coefficients fitted on four
tasks transfer to the fifth task. This directly evaluates the cross-task
generalization behavior needed for a dataset-agnostic \(\lambda\). For that
reason, leave-one-task-out validation is more closely aligned with the purpose
of Experiment 2 than task fixed effects.

## 16. Summary

Experiment 2 defines a positive, calibrated, multi-factor Domain-Aware
Aggregation Weight \(\lambda\). The final selected formulation is a
ridge-regularized linear score over standardized Experiment 1 signals, followed
by exponential positive mapping, clipping to \([0.5, 1.5]\), and mean
normalization within each `(task, round)` aggregation context.

The mathematical contribution of Experiment 2 is the construction and
validation of:

$$
\text{Weight}_i = w_i \times q_i \times \lambda_i
$$

as an optional extension of Project 1's original:

$$
\text{Weight}_i = w_i \times q_i.
$$

Experiment 2 constructs and validates \(\lambda\). Experiment 3 evaluates
whether this calibrated domain-aware factor improves federated learning
performance when used during aggregation.
