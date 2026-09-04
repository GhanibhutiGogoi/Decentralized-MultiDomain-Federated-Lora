# Sample Count Sensitivity Analysis

This supplementary analysis evaluates whether sample count behaves as a
meaningful confounding variable for Experiment 2's frozen \(\lambda\)
formulation. It does not refit \(\lambda\), change coefficients, change the
feature set, or alter aggregation.

## Source Data

The analysis uses existing Experiment 1 and Experiment 2 outputs:

- `outputs/exp1/per_round_client_measurements.csv`
- `outputs/exp2/lambda_values.csv`

Two equivalent sample-count definitions are reported:

- `partition_samples`: number of examples assigned to the client partition.
- `train_samples_seen`: number of local training examples processed. In the
  current Experiment 1 run, this is proportional to `partition_samples`.

Experiment 1 includes an extremely small Tabular client:

| Task | Round | Client | partition_samples | train_samples_seen | delta_accuracy | quality_score | Form A lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tabular-MLP | 1 | 1 | 11 | 33 | -1.666667 | 0.591927 | 0.562067 |

## Results

| Analysis | Form | Sample count | Target | n | Pearson | Spearman | Intercept | Slope | R² |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correlation | all | partition_samples | delta_accuracy | 75 | 0.034332855764779756 | 0.25442518521118757 |  |  |  |
| Correlation | all | partition_samples | quality_score | 75 | 0.38522876886650376 | 0.17975807309977426 |  |  |  |
| OLS | all | partition_samples | delta_accuracy | 75 |  |  | 1.3817584497429272 | 0.0000265254524431672 | 0.0011787449849652853 |
| Correlation | all | train_samples_seen | delta_accuracy | 75 | 0.034332855764779735 | 0.25442518521118757 |  |  |  |
| Correlation | all | train_samples_seen | quality_score | 75 | 0.3852287688665034 | 0.17975807309977426 |  |  |  |
| OLS | all | train_samples_seen | delta_accuracy | 75 |  |  | 1.381758449742927 | 0.000008841817481055754 | 0.0011787449849652853 |
| Correlation | Form A | partition_samples | lambda_weight | 75 | 0.23296531837518739 | 0.08638651252851959 |  |  |  |
| Correlation | Form B | partition_samples | lambda_weight | 75 | 0.24314621260356034 | 0.12573086476923148 |  |  |  |
| Correlation | Form A | train_samples_seen | lambda_weight | 75 | 0.23296531837518725 | 0.08638651252851959 |  |  |  |
| Correlation | Form B | train_samples_seen | lambda_weight | 75 | 0.24314621260356004 | 0.12573086476923148 |  |  |  |

## Interpretation

Sample count shows weak Pearson correlation with leave-one-client-out
contribution:

\[
\mathrm{corr}_{P}(\mathrm{partition\_samples}, \Delta\mathrm{accuracy})
= 0.034332855764779756.
\]

The simple regression

\[
\Delta\mathrm{accuracy}
= \beta_0 + \beta_1\,\mathrm{partition\_samples}
\]

has:

\[
R^2 = 0.0011787449849652853.
\]

This indicates that raw sample count alone explains very little variance in
the Experiment 1 contribution measurements.

Sample count has a moderate Pearson correlation with quality score:

\[
\mathrm{corr}_{P}(\mathrm{partition\_samples}, q)
= 0.38522876886650376.
\]

It also has modest Pearson correlation with the frozen Form B \(\lambda\):

\[
\mathrm{corr}_{P}(\mathrm{partition\_samples}, \lambda_B)
= 0.24314621260356034.
\]

These correlations indicate that sample count should be monitored as a
possible confound in Experiment 3, but they do not justify changing the frozen
Experiment 2 \(\lambda\) formulation.

## Why Sample Count Remains Outside λ

The aggregation rule already includes sample count through the original
aggregation weight \(w_i\). Experiment 2 modifies the existing quality factor
by using:

\[
q_i^{\mathrm{eff}} = q_i\lambda_i.
\]

The existing aggregation path then weights clients proportionally to:

\[
w_i q_i\lambda_i.
\]

Because sample count is already represented by \(w_i\), including sample count
again inside \(\lambda_i\) could double-count client size. The intended role of
\(\lambda\) is to model domain heterogeneity and update-space behavior, not to
replace or duplicate the original sample-size weighting.

## Conclusion

This supplementary analysis supports documenting sample count as a limitation
and monitoring it in Experiment 3. It does not support modifying the frozen
Experiment 2 \(\lambda\) formulation.
