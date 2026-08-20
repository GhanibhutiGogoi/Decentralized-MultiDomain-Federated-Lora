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

### Origin of \(\gamma\)

The scale parameter \(\gamma = 0.8863940762603306\) is the implemented scaling
constant used to control the spread of \(\lambda\) after exponential mapping.
It was empirically selected by the Experiment 2 calibration code to keep the
coefficient of variation of \(\lambda\) below the configured stability target:

$$
\mathrm{CV}(\lambda) \le \min(0.5\,\mathrm{CV}(q), 0.20).
$$

The implementation obtains this value by binary search over the interval
\([0, 5]\) using the already fitted Form A and Form B scores. It is not an
independently optimized scientific parameter and was not tuned by rerunning
federated learning. Its purpose is numerical: preserve enough score variation
for \(\lambda\) to express the calibrated evidence, while preventing the
exponential map from producing a factor that dominates the existing quality
score \(q\).

This value is reasonable because the resulting Form B distribution remains
centered at mean 1, bounded by the implemented clipping and renormalization
steps, and substantially less variable than \(q\).

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

Form B was selected as the primary formulation because it is the most stable
and conservative calibrated formulation among the evaluated candidates. It
preserves the interpretable structure suggested by Experiment 1 while using
ridge regularization to shrink weak or uncertain signals. Its coefficients
remain scientifically interpretable:

- update L2 contributes positively;
- JS divergence contributes negatively;
- update cosine distance is shrunk close to zero;
- normalized entropy contributes mildly positively;
- class imbalance contributes negatively.

Leave-one-task-out validation selected the Form B ridge parameter
\(\alpha_{\mathrm{ridge}} = 100.0\). The held-out \(R^2\) values remain
negative, and the predictive performance is weak. Form B should therefore be
interpreted as the least-bad calibrated formulation among the evaluated
candidates, not as a highly predictive contribution model.

Despite weak predictive performance, Form B produced lower mean RMSE and MAE
than Form A, applied ridge regularization, and yielded a more conservative
lambda distribution. These properties reduce the risk that \(\lambda\) overfits
the small Experiment 1 calibration set or dominates the existing Project 1
quality score \(q\).

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

Form B was not selected because it wins every validation metric. Form A retains
stronger rank-order behavior, as shown by its higher mean Spearman correlation.
Form B was selected as the primary Experiment 3 formulation because the
\(\alpha_{\mathrm{ridge}} = 100.0\) model provides lower mean RMSE and MAE,
more conservative \(\lambda\) values, stronger numerical stability, explicit
ridge regularization, and lower risk of overfitting weak Experiment 1 signals.

Thus, the Form B selection is a scientific trade-off. Form B is the conservative
primary candidate, while Form A remains the interpretable ablation baseline.
Experiment 3 is required to determine whether either calibrated \(\lambda\)
formulation is beneficial when incorporated into the federated aggregation
process.

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
- The scale \(\gamma\) is empirically calibrated to control \(\lambda\)'s spread;
  it is not independently optimized through federated learning runs.
- \(\lambda\) has been validated offline using Experiment 1 measurements only.
- Experiment 3 is required to determine whether the frozen
  \(w_i \times q_i \times \lambda_i\) aggregation rule improves federated
  learning performance.

## 18. Summary

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

Experiment 2 calibrates \(\lambda\), validates its mathematical properties,
validates boundedness and stability, validates global orthogonality against
\(q\), documents limitations, and prepares a finalized \(\lambda\) candidate
for Experiment 3.

Experiment 2 does not prove that \(\lambda\) improves federated learning.
Experiment 3 is the first experiment capable of answering whether

$$
\text{Weight}_i = w_i \times q_i \times \lambda_i
$$

actually improves federated learning performance when used during aggregation.
