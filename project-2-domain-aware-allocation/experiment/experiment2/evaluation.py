"""Unified evaluation metrics for Experiment 2 calibration outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from experiment2.lambda_calibration import (
    EPS,
    GROUP_COLS,
    TARGET,
    enforce_ridge_alpha_not_on_boundary,
    fit_form_a,
    fit_form_b,
    pearson,
    predict_delta_accuracy,
    predict_standardized_score,
    ridge_alpha_grid,
    spearman,
)
from experiment2.form_c import (
    FORM_C_EXPLORATORY_STATUS,
    FORM_C_FEATURES,
    FORM_C_METHOD_LABEL,
    FORM_C_TARGET_TRANSFORMATION,
    coefficient_norm,
    fit_form_c,
    predict_form_c_score,
    transform_form_c_context,
)


DEFAULT_PERMUTATION_SEED = 42
DEFAULT_PERMUTATIONS = 1000
NULL_IMPROVEMENT_TOLERANCE = 1e-9
STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for deterministic statistical evaluation."""

    permutation_seed: int = DEFAULT_PERMUTATION_SEED
    permutations: int = DEFAULT_PERMUTATIONS
    group_cols: tuple[str, ...] = tuple(GROUP_COLS)


@dataclass(frozen=True)
class FormSupportDecision:
    """Scientific support decision for a candidate lambda form."""

    form: str
    status: str
    primary_metric_name: str
    model_mean_rmse: float
    null_mean_rmse: float
    absolute_improvement: float
    relative_improvement: float
    selected_alpha: float | None = None
    alpha_boundary_status: str = "not_applicable"
    coefficient_available: bool = False
    gamma_lambda_available: bool = False
    rejection_reason: str = ""
    source_run_identity: str = ""
    methodology_label: str = ""
    exploratory_status: str = ""
    feature_order: tuple[str, ...] = ()
    target_transformation: str = ""
    coefficient_norm: float | None = None
    prediction_variance: float | None = None
    zero_variance_feature_group_count: int = 0
    zero_variance_target_group_count: int = 0
    experiment3_eligible: bool = False

    @property
    def supported(self) -> bool:
        return self.status == STATUS_SUPPORTED

    def to_record(self) -> dict[str, object]:
        return {
            "form": self.form,
            "status": self.status,
            "primary_metric_name": self.primary_metric_name,
            "model_mean_rmse": self.model_mean_rmse,
            "null_mean_rmse": self.null_mean_rmse,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "selected_alpha": "" if self.selected_alpha is None else self.selected_alpha,
            "alpha_boundary_status": self.alpha_boundary_status,
            "coefficient_available": self.coefficient_available,
            "gamma_lambda_available": self.gamma_lambda_available,
            "rejection_reason": self.rejection_reason,
            "source_run_identity": self.source_run_identity,
            "null_improvement_tolerance": NULL_IMPROVEMENT_TOLERANCE,
            "methodology_label": self.methodology_label,
            "exploratory_status": self.exploratory_status,
            "feature_order": "|".join(self.feature_order),
            "target_transformation": self.target_transformation,
            "coefficient_norm": "" if self.coefficient_norm is None else self.coefficient_norm,
            "prediction_variance": "" if self.prediction_variance is None else self.prediction_variance,
            "zero_variance_feature_group_count": self.zero_variance_feature_group_count,
            "zero_variance_target_group_count": self.zero_variance_target_group_count,
            "experiment3_eligible": self.experiment3_eligible,
        }


def form_support_status_table(
    decisions: Iterable[FormSupportDecision],
) -> pd.DataFrame:
    """Render support decisions as a stable CSV-friendly table."""
    return pd.DataFrame([decision.to_record() for decision in decisions])


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Return standard regression metrics without changing model fitting."""
    actual = np.asarray(list(y_true), dtype=float)
    predicted = np.asarray(list(y_pred), dtype=float)
    if actual.size == 0:
        return {"rmse": 0.0, "mae": 0.0, "r_squared": 0.0, "pearson": 0.0}

    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > EPS else 0.0
    return {
        "rmse": rmse,
        "mae": mae,
        "r_squared": float(r_squared),
        "pearson": pearson(predicted, actual),
    }


def pairwise_ranking_accuracy(
    y_true: Iterable[float],
    y_score: Iterable[float],
) -> float:
    """Fraction of comparable item pairs ordered the same by score and target."""
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    comparable = 0
    correct = 0.0
    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            actual_diff = actual[i] - actual[j]
            if abs(actual_diff) <= EPS:
                continue
            score_diff = score[i] - score[j]
            comparable += 1
            if abs(score_diff) <= EPS:
                correct += 0.5
            elif np.sign(actual_diff) == np.sign(score_diff):
                correct += 1.0
    return float(correct / comparable) if comparable else 0.0


def kendall_tau_b(
    y_true: Iterable[float],
    y_score: Iterable[float],
) -> float:
    """Compute Kendall tau-b with tie correction."""
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    concordant = 0
    discordant = 0
    ties_actual = 0
    ties_score = 0

    for i in range(len(actual)):
        for j in range(i + 1, len(actual)):
            actual_diff = actual[i] - actual[j]
            score_diff = score[i] - score[j]
            actual_tie = abs(actual_diff) <= EPS
            score_tie = abs(score_diff) <= EPS
            if actual_tie and score_tie:
                continue
            if actual_tie:
                ties_actual += 1
            elif score_tie:
                ties_score += 1
            elif np.sign(actual_diff) == np.sign(score_diff):
                concordant += 1
            else:
                discordant += 1

    denom = np.sqrt(
        (concordant + discordant + ties_actual)
        * (concordant + discordant + ties_score)
    )
    return float((concordant - discordant) / denom) if denom > EPS else 0.0


def _permutation_groups(
    df: pd.DataFrame,
    group_cols: tuple[str, ...],
) -> list[object] | None:
    """Return row-level permutation strata when group columns are available."""
    cols = list(group_cols)
    if not cols or not set(cols).issubset(df.columns):
        return None
    return [tuple(row) for row in df[cols].itertuples(index=False, name=None)]


def permutation_rank_p_value(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    groups: Iterable[object] | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> float:
    """One-sided permutation p-value for positive Spearman association.

    When ``groups`` are supplied, targets are permuted only within those strata.
    This preserves exchangeability for pooled evaluations over heterogeneous
    aggregation contexts such as ``(task, round)``.
    """
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    if actual.size < 2 or n_permutations <= 0:
        return 1.0
    if groups is None:
        group_values = np.zeros(actual.size, dtype=int)
    else:
        raw_groups = list(groups)
        group_values = np.empty(len(raw_groups), dtype=object)
        group_values[:] = raw_groups
        if group_values.shape[0] != actual.shape[0]:
            raise ValueError("Permutation groups must match the target length.")

    observed = spearman(score, actual)
    if observed <= 0:
        return 1.0

    rng = np.random.default_rng(seed)
    positions_by_group: dict[object, list[int]] = {}
    for idx, group in enumerate(group_values):
        positions_by_group.setdefault(group, []).append(idx)
    group_positions = [
        np.asarray(positions, dtype=int)
        for positions in positions_by_group.values()
    ]
    exceedances = 0
    for _ in range(n_permutations):
        permuted = actual.copy()
        for positions in group_positions:
            permuted[positions] = rng.permutation(permuted[positions])
        if spearman(score, permuted) >= observed - EPS:
            exceedances += 1
    return float((exceedances + 1) / (n_permutations + 1))


def ranking_metrics(
    y_true: Iterable[float],
    y_score: Iterable[float],
    *,
    groups: Iterable[object] | None = None,
    n_permutations: int = DEFAULT_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> dict[str, float]:
    """Return rank-based metrics and statistical significance."""
    actual = np.asarray(list(y_true), dtype=float)
    score = np.asarray(list(y_score), dtype=float)
    return {
        "spearman": spearman(score, actual),
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy(actual, score),
        "kendall_tau": kendall_tau_b(actual, score),
        "permutation_p_value": permutation_rank_p_value(
            actual,
            score,
            groups=groups,
            n_permutations=n_permutations,
            seed=seed,
        ),
    }


def evaluate_predictions(
    df: pd.DataFrame,
    *,
    score_col: str,
    prediction_col: str = "predicted_delta_accuracy",
    target_col: str = TARGET,
    config: EvaluationConfig | None = None,
) -> dict[str, float]:
    """Compute the full unified metric set for a prediction table."""
    cfg = config or EvaluationConfig()
    groups = _permutation_groups(df, cfg.group_cols)
    metrics = regression_metrics(df[target_col], df[prediction_col])
    metrics.update(
        ranking_metrics(
            df[target_col],
            df[score_col],
            groups=groups,
            n_permutations=cfg.permutations,
            seed=cfg.permutation_seed,
        )
    )
    return metrics


def _fold_result_row(
    *,
    form: str,
    held_out_task: str,
    ridge_alpha: float | str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    prediction: Iterable[float],
    score: Iterable[float],
    config: EvaluationConfig,
) -> dict[str, object]:
    metrics = regression_metrics(test[TARGET], prediction)
    metrics.update(
        ranking_metrics(
            test[TARGET],
            score,
            groups=_permutation_groups(test, config.group_cols),
            n_permutations=config.permutations,
            seed=config.permutation_seed,
        )
    )
    return {
        "form": form,
        "held_out_task": held_out_task,
        "ridge_alpha": ridge_alpha,
        "n_train": len(train),
        "n_test": len(test),
        **metrics,
    }


def _alpha_boundary_status(selected_alpha: float, alphas: list[float]) -> str:
    if selected_alpha == min(alphas):
        return "minimum"
    if selected_alpha == max(alphas):
        return "maximum"
    return "interior"


def _support_decision(
    *,
    form: str,
    model_mean_rmse: float,
    null_mean_rmse: float,
    selected_alpha: float | None = None,
    alpha_boundary_status: str = "not_applicable",
    source_run_identity: str = "",
    methodology_label: str = "",
    exploratory_status: str = "",
    feature_order: Sequence[str] = (),
    target_transformation: str = "",
    coefficient_norm_value: float | None = None,
    prediction_variance: float | None = None,
    zero_variance_feature_group_count: int = 0,
    zero_variance_target_group_count: int = 0,
    reversed_orientation: bool = False,
) -> FormSupportDecision:
    improvement = float(null_mean_rmse - model_mean_rmse)
    relative = improvement / max(abs(float(null_mean_rmse)), EPS)
    rejection_reason = ""
    if improvement <= NULL_IMPROVEMENT_TOLERANCE:
        rejection_reason = "model_mean_rmse_not_strictly_better_than_fold_safe_null"
    elif alpha_boundary_status in {"minimum", "maximum"}:
        rejection_reason = f"ridge_alpha_boundary_{alpha_boundary_status}"
    elif reversed_orientation:
        rejection_reason = "predictive_orientation_reversed"

    supported = rejection_reason == ""
    return FormSupportDecision(
        form=form,
        status=STATUS_SUPPORTED if supported else STATUS_UNSUPPORTED,
        primary_metric_name="mean_leave_one_task_out_rmse",
        model_mean_rmse=float(model_mean_rmse),
        null_mean_rmse=float(null_mean_rmse),
        absolute_improvement=improvement,
        relative_improvement=float(relative),
        selected_alpha=selected_alpha,
        alpha_boundary_status=alpha_boundary_status,
        coefficient_available=supported,
        gamma_lambda_available=supported,
        rejection_reason=rejection_reason,
        source_run_identity=source_run_identity,
        methodology_label=methodology_label,
        exploratory_status=exploratory_status,
        feature_order=tuple(feature_order),
        target_transformation=target_transformation,
        coefficient_norm=coefficient_norm_value,
        prediction_variance=prediction_variance,
        zero_variance_feature_group_count=int(zero_variance_feature_group_count),
        zero_variance_target_group_count=int(zero_variance_target_group_count),
        experiment3_eligible=supported,
    )


def support_decision_evaluation(
    df: pd.DataFrame,
    *,
    ridge_alphas: Iterable[float] | None = None,
    config: EvaluationConfig | None = None,
    source_run_identity: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FormSupportDecision]]:
    """Evaluate candidate forms against their fold-safe null validity gates.

    Form A and Form B use raw mean leave-one-task-out RMSE. Form C uses
    within-task-round normalized RMSE against the zero-null centered target.
    Unsupported forms remain in diagnostics but do not produce lambda
    parameters.
    """
    cfg = config or EvaluationConfig()
    rows = []
    alphas = ridge_alpha_grid(custom_alphas=ridge_alphas)

    for held_out in sorted(df["task"].unique()):
        train = df[df["task"] != held_out].reset_index(drop=True)
        test = df[df["task"] == held_out].reset_index(drop=True)

        null_prediction = np.full(len(test), float(train[TARGET].mean()), dtype=float)
        rows.append(
            _fold_result_row(
                form="null",
                held_out_task=held_out,
                ridge_alpha="",
                train=train,
                test=test,
                prediction=null_prediction,
                score=null_prediction,
                config=cfg,
            )
        )

        fit_a = fit_form_a(train)
        pred_a = predict_delta_accuracy(test, fit_a)
        score_a = predict_standardized_score(test, fit_a)
        rows.append(
            _fold_result_row(
                form="form_a",
                held_out_task=held_out,
                ridge_alpha="",
                train=train,
                test=test,
                prediction=pred_a,
                score=score_a,
                config=cfg,
            )
        )

        for alpha in alphas:
            fit_b = fit_form_b(train, alpha)
            pred_b = predict_delta_accuracy(test, fit_b)
            score_b = predict_standardized_score(test, fit_b)
            rows.append(
                _fold_result_row(
                    form="form_b",
                    held_out_task=held_out,
                    ridge_alpha=alpha,
                    train=train,
                    test=test,
                    prediction=pred_b,
                    score=score_b,
                    config=cfg,
                )
            )

        test_form_c = transform_form_c_context(test, include_target=True)
        y_c = test_form_c.frame[test_form_c.target_column].to_numpy(dtype=float)
        zero_prediction = np.zeros(len(test_form_c.frame), dtype=float)
        null_c_metrics = regression_metrics(y_c, zero_prediction)
        null_c_metrics.update(
            ranking_metrics(
                y_c,
                zero_prediction,
                groups=_permutation_groups(test_form_c.frame, cfg.group_cols),
                n_permutations=cfg.permutations,
                seed=cfg.permutation_seed,
            )
        )
        for alpha in alphas:
            fit_c = fit_form_c(train, alpha)
            score_c = predict_form_c_score(test, fit_c)
            metrics_c = regression_metrics(y_c, score_c)
            metrics_c.update(
                ranking_metrics(
                    y_c,
                    score_c,
                    groups=_permutation_groups(test_form_c.frame, cfg.group_cols),
                    n_permutations=cfg.permutations,
                    seed=cfg.permutation_seed,
                )
            )
            rows.append(
                {
                    "form": "form_c",
                    "held_out_task": held_out,
                    "ridge_alpha": alpha,
                    "n_train": len(train),
                    "n_test": len(test),
                    "null_rmse": null_c_metrics["rmse"],
                    "null_mae": null_c_metrics["mae"],
                    "null_spearman": null_c_metrics["spearman"],
                    "coefficient_norm": coefficient_norm(fit_c.coefficients),
                    "prediction_variance": float(np.var(score_c, ddof=0)),
                    "zero_variance_feature_group_count": int(
                        sum(test_form_c.feature_zero_variance_counts.values())
                    ),
                    "zero_variance_target_group_count": int(
                        test_form_c.target_zero_variance_group_count
                    ),
                    "methodology_label": FORM_C_METHOD_LABEL,
                    "exploratory_status": FORM_C_EXPLORATORY_STATUS,
                    **metrics_c,
                }
            )

    cv = pd.DataFrame(rows)
    null_mean_rmse = float(cv[cv["form"] == "null"]["rmse"].mean())
    form_a_mean_rmse = float(cv[cv["form"] == "form_a"]["rmse"].mean())

    ridge_rows = cv[cv["form"] == "form_b"].copy()
    mean_rmse = ridge_rows.groupby("ridge_alpha")["rmse"].mean()
    selected_alpha = float(mean_rmse.idxmin())
    selected_form_b_mean_rmse = float(mean_rmse.loc[selected_alpha])
    boundary_status = _alpha_boundary_status(selected_alpha, alphas)

    decisions = {
        "form_a": _support_decision(
            form="form_a",
            model_mean_rmse=form_a_mean_rmse,
            null_mean_rmse=null_mean_rmse,
            source_run_identity=source_run_identity,
        ),
        "form_b": _support_decision(
            form="form_b",
            model_mean_rmse=selected_form_b_mean_rmse,
            null_mean_rmse=null_mean_rmse,
            selected_alpha=selected_alpha,
            alpha_boundary_status=boundary_status,
            source_run_identity=source_run_identity,
        ),
    }
    form_c_rows = cv[cv["form"] == "form_c"].copy()
    form_c_mean_rmse = form_c_rows.groupby("ridge_alpha")["rmse"].mean()
    selected_form_c_alpha = float(form_c_mean_rmse.idxmin())
    selected_form_c_mean_rmse = float(form_c_mean_rmse.loc[selected_form_c_alpha])
    selected_form_c_rows = form_c_rows[
        form_c_rows["ridge_alpha"] == selected_form_c_alpha
    ]
    form_c_null_mean_rmse = float(selected_form_c_rows["null_rmse"].mean())
    form_c_boundary = _alpha_boundary_status(selected_form_c_alpha, alphas)
    form_c_spearman = float(selected_form_c_rows["spearman"].mean())
    form_c_reversed = form_c_spearman < -EPS
    decisions["form_c"] = _support_decision(
        form="form_c",
        model_mean_rmse=selected_form_c_mean_rmse,
        null_mean_rmse=form_c_null_mean_rmse,
        selected_alpha=selected_form_c_alpha,
        alpha_boundary_status=form_c_boundary,
        source_run_identity=source_run_identity,
        methodology_label=FORM_C_METHOD_LABEL,
        exploratory_status=FORM_C_EXPLORATORY_STATUS,
        feature_order=FORM_C_FEATURES,
        target_transformation=FORM_C_TARGET_TRANSFORMATION,
        coefficient_norm_value=float(selected_form_c_rows["coefficient_norm"].mean()),
        prediction_variance=float(selected_form_c_rows["prediction_variance"].mean()),
        zero_variance_feature_group_count=int(
            selected_form_c_rows["zero_variance_feature_group_count"].sum()
        ),
        zero_variance_target_group_count=int(
            selected_form_c_rows["zero_variance_target_group_count"].sum()
        ),
        reversed_orientation=form_c_reversed,
    )
    return cv, form_support_status_table(decisions.values()), decisions


def evaluation_table(
    lambda_values: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Evaluate each form globally, per task, and per aggregation context."""
    cfg = config or EvaluationConfig()
    rows = []
    for form, form_df in lambda_values.groupby("form"):
        scopes = [("all", "ALL", form_df)]
        scopes.extend(("task", task, group) for task, group in form_df.groupby("task"))
        context_cols = list(cfg.group_cols)
        if set(context_cols).issubset(form_df.columns):
            for context_key, group in form_df.groupby(context_cols):
                context_name = "|".join(str(part) for part in context_key)
                scopes.append(("aggregation_context", context_name, group))

        for scope, scope_value, group in scopes:
            target_col = (
                "relative_contribution_target"
                if form == "form_c" and "relative_contribution_target" in group.columns
                else TARGET
            )
            metrics = evaluate_predictions(
                group,
                score_col="raw_lambda_score",
                prediction_col="predicted_delta_accuracy",
                target_col=target_col,
                config=cfg,
            )
            rows.append(
                {
                    "form": form,
                    "scope": scope,
                    "scope_value": scope_value,
                    "n": len(group),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def leave_one_task_out_evaluation(
    df: pd.DataFrame,
    *,
    ridge_alphas: Iterable[float] | None = None,
    config: EvaluationConfig | None = None,
) -> tuple[pd.DataFrame, float]:
    """Evaluate held-out tasks while preserving RMSE alpha selection."""
    cfg = config or EvaluationConfig()
    rows = []
    alphas = ridge_alpha_grid(custom_alphas=ridge_alphas)

    for held_out in sorted(df["task"].unique()):
        train = df[df["task"] != held_out].reset_index(drop=True)
        test = df[df["task"] == held_out].reset_index(drop=True)

        fit_a = fit_form_a(train)
        pred_a = predict_delta_accuracy(test, fit_a)
        score_a = predict_standardized_score(test, fit_a)
        metrics_a = regression_metrics(test[TARGET], pred_a)
        metrics_a.update(
            ranking_metrics(
                test[TARGET],
                score_a,
                groups=_permutation_groups(test, cfg.group_cols),
                n_permutations=cfg.permutations,
                seed=cfg.permutation_seed,
            )
        )
        rows.append(
            {
                "form": "form_a",
                "held_out_task": held_out,
                "ridge_alpha": "",
                "n_train": len(train),
                "n_test": len(test),
                **metrics_a,
            }
        )

        for alpha in alphas:
            fit_b = fit_form_b(train, alpha)
            pred_b = predict_delta_accuracy(test, fit_b)
            score_b = predict_standardized_score(test, fit_b)
            metrics_b = regression_metrics(test[TARGET], pred_b)
            metrics_b.update(
                ranking_metrics(
                    test[TARGET],
                    score_b,
                    groups=_permutation_groups(test, cfg.group_cols),
                    n_permutations=cfg.permutations,
                    seed=cfg.permutation_seed,
                )
            )
            rows.append(
                {
                    "form": "form_b",
                    "held_out_task": held_out,
                    "ridge_alpha": alpha,
                    "n_train": len(train),
                    "n_test": len(test),
                    **metrics_b,
                }
            )

    cv = pd.DataFrame(rows)
    ridge_rows = cv[cv["form"] == "form_b"].copy()
    mean_rmse = ridge_rows.groupby("ridge_alpha")["rmse"].mean()
    selected_alpha = float(mean_rmse.idxmin())
    enforce_ridge_alpha_not_on_boundary(selected_alpha, alphas)
    return cv, selected_alpha


def alpha_evaluation_table(
    cv: pd.DataFrame,
    *,
    config: EvaluationConfig | None = None,
) -> pd.DataFrame:
    """Summarize held-out alpha candidates without changing selection logic."""
    del config
    numeric = [
        "rmse",
        "mae",
        "r_squared",
        "pearson",
        "spearman",
        "pairwise_ranking_accuracy",
        "kendall_tau",
        "permutation_p_value",
        "null_rmse",
        "null_mae",
        "null_spearman",
        "coefficient_norm",
        "prediction_variance",
        "zero_variance_feature_group_count",
        "zero_variance_target_group_count",
    ]
    available = [col for col in numeric if col in cv.columns]
    group_cols = ["form", "ridge_alpha"]
    rows = (
        cv.groupby(group_cols, dropna=False)[available]
        .mean()
        .reset_index()
    )
    rows["_ridge_alpha_sort"] = pd.to_numeric(
        rows["ridge_alpha"], errors="coerce"
    ).fillna(-1.0)
    rows = rows.sort_values(["form", "_ridge_alpha_sort"]).drop(
        columns=["_ridge_alpha_sort"]
    )
    return rows
