from collections import Counter
from typing import Any

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from core.utils.general_utils import create_logger

logger = create_logger(__name__)


def standardize_continuous_covariates(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    vars_to_standardize: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Standardize continuous covariates using training statistics.

    For training-only runs, pass the same dataframe for df_train and df_test.
    For cross-validation, pass the train and test dataframes to avoid leakage.

    Args:
        - df_train: Dataframe used to fit the scaler.
        - df_test: Dataframe to transform using the training scaler.
        - vars_to_standardize: Covariate names to be standardized.

    Returns:
        - df_train_res: Training dataframe with standardized covariates.
        - df_test_res: Test dataframe with standardized covariates.
    """
    logger.info("Standardizing continuous covariates.")
    if not vars_to_standardize:
        logger.info("No continuous covariates to standardize.")
        return df_train, df_test

    # Fit scaler on training data
    scaler = StandardScaler()
    train_data = df_train.select(vars_to_standardize).to_numpy()
    scaler.fit(train_data)

    def _scale_df(df: pl.DataFrame, fitted_scaler: StandardScaler) -> pl.DataFrame:
        """Scale the specified columns of a dataframe using a fitted scaler."""
        data_to_scale = df.select(vars_to_standardize).to_numpy()
        df_scaled = pl.DataFrame(
            fitted_scaler.transform(data_to_scale),
            schema=vars_to_standardize,
        )

        # Check for NaN and infinite values, report and replace with zeros
        # Any such values are likely due to floating point precision issues
        for col in vars_to_standardize:
            inf_sum = df_scaled.get_column(col).is_infinite().sum()
            nan_sum = df_scaled.get_column(col).is_nan().sum()
            if inf_sum > 0:
                logger.warning(
                    f"{inf_sum} infinite values found in {col}. Filling with 0."
                )
            if nan_sum > 0:
                logger.warning(f"{nan_sum} NaN values found in {col}. Filling with 0.")
        df_scaled = df_scaled.fill_nan(0)

        # Replace original columns with the standardized ones
        return pl.concat([df.drop(vars_to_standardize), df_scaled], how="horizontal")

    # Transform both dataframes using the fitted scaler
    df_train_res = _scale_df(df_train, scaler)
    df_test_res = _scale_df(df_test, scaler)

    logger.info("Finished standardizing continuous covariates.")

    return df_train_res, df_test_res


def validate_design_matrix_columns(
    df_train: pl.DataFrame,
    df_test: pl.DataFrame,
    x_vars: list[str],
) -> None:
    """
    Validate design matrix columns for duplicates and train/test consistency.

    For training-only runs, pass the same dataframe for df_train and df_test.
    """
    if not x_vars:
        raise ValueError("Design matrix columns are empty.")

    duplicates = [name for name, count in Counter(x_vars).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate columns in design matrix: {duplicates}")

    missing_train = [col for col in x_vars if col not in df_train.columns]
    missing_test = [col for col in x_vars if col not in df_test.columns]
    if missing_train:
        raise ValueError(f"Missing design matrix columns in train: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing design matrix columns in test: {missing_test}")

    train_cols = df_train.select(x_vars).columns
    test_cols = df_test.select(x_vars).columns
    if train_cols != x_vars:
        raise ValueError("Train design matrix columns do not match expected ordering.")
    if test_cols != x_vars:
        raise ValueError("Test design matrix columns do not match expected ordering.")
    if train_cols != test_cols:
        raise ValueError("Train and test design matrix columns do not match.")


def get_scope_counts(
    df: pl.DataFrame,
    diversity_type: str,
    *,
    site_key: str | None = None,
    check_key: str | None = None,
) -> dict[str, int]:
    """
    Count studies, observations, and sites or site pairs for a dataframe.

    Args:
        - df: Dataframe to summarize
        - diversity_type: Either "alpha" or "beta"
        - site_key: Optional column name for site or site-pair identifiers
        - check_key: Optional column name for a composite uniqueness key

    Returns:
        - counts: Dictionary of counts for studies, observations, and scope
    """
    counts = {
        "studies": df.get_column("SS").n_unique(),
        "observations": df.height,
    }

    if diversity_type == "alpha":
        key = site_key or "SSBS"
        if key not in df.columns:
            raise ValueError(f"Missing site key column: {key}")
        counts["sites"] = df.get_column(key).n_unique()
    elif diversity_type == "beta":
        if site_key and site_key in df.columns:
            counts["site_pairs"] = df.get_column(site_key).n_unique()
        elif {"SSBS", "Primary_minimal_site"}.issubset(df.columns):
            counts["site_pairs"] = df.select(
                ["SSBS", "Primary_minimal_site"]
            ).n_unique()
        else:
            raise ValueError("Missing columns for site-pair counting")
    else:
        raise ValueError(f"Unsupported diversity_type: {diversity_type}")

    if check_key:
        if check_key not in df.columns:
            raise ValueError(f"Missing check_key column: {check_key}")
        counts["check_key"] = df.get_column(check_key).n_unique()

    return counts


def calculate_performance_metrics(
    df: pl.DataFrame,
    model_type: str,
    mode: str,
    *,
    prediction_col: str | None = None,
) -> dict[str, float]:
    """
    Evaluate model performance and log results. Calculates R2, mean absolute
    error, Pearson correlation, and Spearman rank correlation. Metrics are
    calculated for the full dataset and for top and bottom quartiles of
    observed values.

    Args:
        - df: Dataframe containing 'Observed' and model predictions.
        - model_type: Type of model ('bayesian' or 'glmm').
        - mode: Mode of evaluation ('training' or 'crossval').
        - prediction_col: Optional column name to use for predictions.

    Returns:
        - performance_metrics: A dictionary containing the calculated metrics.
    """
    logger.info("Evaluating model performance metrics.")

    # Extract observed and predicted values
    y_true = df.get_column("Observed").to_numpy()
    y_pred_re: np.ndarray | None = None
    if prediction_col is not None:
        y_pred = df.get_column(prediction_col).to_numpy()
    elif model_type == "glmm" and mode == "training":
        y_pred = df.get_column("Predicted_RE").to_numpy()
    elif model_type == "glmm" and mode == "crossval":
        y_pred = df.get_column("Predicted_FE").to_numpy()
        y_pred_re = df.get_column("Predicted_RE").to_numpy()
    else:
        y_pred = df.get_column("Predicted").to_numpy()

    # Calculate quartiles for observed values using rank-based bins
    n_obs = y_true.size
    order = np.argsort(y_true)
    quartile_edges = np.linspace(0, n_obs, 5, dtype=int)
    bottom_indices = order[quartile_edges[0] : quartile_edges[1]]
    top_indices = order[quartile_edges[3] : quartile_edges[4]]
    if bottom_indices.size == 0:
        logger.warning("Bottom quartile empty (n_obs=%d).", n_obs)
    if top_indices.size == 0:
        logger.warning("Top quartile empty (n_obs=%d).", n_obs)

    # Calculate overall metrics for the general case
    r2_std = r2_score(y_true, y_pred)
    if model_type == "glmm" and mode == "crossval":
        if y_pred_re is None:
            y_pred_re = df.get_column("Predicted_RE").to_numpy()
        r2_var = np.var(y_pred) / (np.var(y_pred_re) + np.var(y_true - y_pred))
    else:
        r2_var = np.var(y_pred) / (np.var(y_pred) + np.var(y_true - y_pred))
    mean_abs_error = mean_absolute_error(y_true, y_pred)
    median_abs_error = median_absolute_error(y_true, y_pred)

    def _safe_corr(
        y_true_slice: np.ndarray,
        y_pred_slice: np.ndarray,
        label: str,
    ) -> tuple[float, float]:
        def _slice_summary(values: np.ndarray) -> str:
            if values.size == 0:
                return "n=0"
            return (
                f"n={values.size}, min={float(np.min(values)):.4g}, "
                f"max={float(np.max(values)):.4g}"
            )

        if y_true_slice.size == 0 or y_pred_slice.size == 0:
            logger.warning(
                "%s correlation skipped: empty slice (true: %s, pred: %s).",
                label,
                _slice_summary(y_true_slice),
                _slice_summary(y_pred_slice),
            )
            return np.nan, np.nan
        std_true = float(np.std(y_true_slice))
        std_pred = float(np.std(y_pred_slice))
        if np.isclose(std_true, 0.0) or np.isclose(std_pred, 0.0):
            logger.warning(
                "%s correlation skipped: constant input "
                "(true: %s, pred: %s, std_true=%.4g, std_pred=%.4g).",
                label,
                _slice_summary(y_true_slice),
                _slice_summary(y_pred_slice),
                std_true,
                std_pred,
            )
            return np.nan, np.nan
        pearson_corr, _ = pearsonr(y_true_slice, y_pred_slice)
        spearman_corr, _ = spearmanr(y_true_slice, y_pred_slice)
        return pearson_corr, spearman_corr

    pearson_corr, spearman_corr = _safe_corr(y_true, y_pred, "Overall")

    # Bias calculations for overall, bottom, and top quartiles
    bias_metrics = calculate_bias_metrics(y_true, y_pred, bottom_indices, top_indices)

    # Log overall metrics
    logger.info(
        "\nOverall model performance:\n"
        f" - R2 (standard): {r2_std:.3f}\n"
        f" - R2 (variance explained): {r2_var:.3f}\n"
        f" - Mean absolute error: {mean_abs_error:.3f}\n"
        f" - Median absolute error: {median_abs_error:.3f}\n"
        f" - Pearson correlation: {pearson_corr:.3f}\n"
        f" - Spearman rank correlation: {spearman_corr:.3f}\n"
        f" - Bias ratio (pred/obs): {bias_metrics['overall_bias_ratio']:.3f}\n"
    )

    # Metrics for bottom quartile
    bottom_quartile_true = y_true[bottom_indices]
    bottom_quartile_pred = y_pred[bottom_indices]

    r2_std_bottom = r2_score(bottom_quartile_true, bottom_quartile_pred)
    if model_type == "glmm" and mode == "crossval":
        if y_pred_re is None:
            y_pred_re = df.get_column("Predicted_RE").to_numpy()
        r2_var_bottom = np.var(bottom_quartile_pred) / (
            np.var(y_pred_re[bottom_indices])
            + np.var(bottom_quartile_true - bottom_quartile_pred)
        )
    else:
        r2_var_bottom = np.var(bottom_quartile_pred) / (
            np.var(bottom_quartile_pred)
            + np.var(bottom_quartile_true - bottom_quartile_pred)
        )

    mean_abs_error_bottom = mean_absolute_error(
        bottom_quartile_true, bottom_quartile_pred
    )
    median_abs_error_bottom = median_absolute_error(
        bottom_quartile_true, bottom_quartile_pred
    )
    pearson_corr_bottom, spearman_corr_bottom = _safe_corr(
        bottom_quartile_true, bottom_quartile_pred, "Bottom quartile"
    )

    # Metrics for top quartile
    top_quartile_true = y_true[top_indices]
    top_quartile_pred = y_pred[top_indices]

    r2_std_top = r2_score(top_quartile_true, top_quartile_pred)
    if model_type == "glmm" and mode == "crossval":
        if y_pred_re is None:
            y_pred_re = df.get_column("Predicted_RE").to_numpy()
        r2_var_top = np.var(top_quartile_pred) / (
            np.var(y_pred_re[top_indices])
            + np.var(top_quartile_true - top_quartile_pred)
        )
    else:
        r2_var_top = np.var(top_quartile_pred) / (
            np.var(top_quartile_pred) + np.var(top_quartile_true - top_quartile_pred)
        )

    mean_abs_error_top = mean_absolute_error(top_quartile_true, top_quartile_pred)
    median_abs_error_top = median_absolute_error(top_quartile_true, top_quartile_pred)
    pearson_corr_top, spearman_corr_top = _safe_corr(
        top_quartile_true, top_quartile_pred, "Top quartile"
    )

    # Return all metrics
    metrics = {
        "r2_std": r2_std,
        "r2_var": r2_var,
        "mean_abs_error": mean_abs_error,
        "median_abs_error": median_abs_error,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "r2_std_bottom": r2_std_bottom,
        "r2_var_bottom": r2_var_bottom,
        "mean_abs_error_bottom": mean_abs_error_bottom,
        "median_abs_error_bottom": median_abs_error_bottom,
        "pearson_corr_bottom": pearson_corr_bottom,
        "spearman_corr_bottom": spearman_corr_bottom,
        "r2_std_top": r2_std_top,
        "r2_var_top": r2_var_top,
        "mean_abs_error_top": mean_abs_error_top,
        "median_abs_error_top": median_abs_error_top,
        "pearson_corr_top": pearson_corr_top,
        "spearman_corr_top": spearman_corr_top,
        "bias_metrics": bias_metrics,
    }

    return metrics


def calculate_bias_metrics(
    y_true: np.array,
    y_pred: np.array,
    bottom_indices: np.array,
    top_indices: np.array,
) -> dict[str, Any]:
    """
    Calculate the bias (mean predicted / mean observed) for overall, bottom,
    and top quartiles.

    Args:
        - y_true: Array of observed values.
        - y_pred: Array of predicted values.

    Returns:
        - bias_dict: Dictionary containing overall bias, bias per decile,
          and quartile-specific biases.
    """

    def _safe_bias_ratio(
        y_true_slice: np.ndarray,
        y_pred_slice: np.ndarray,
        label: str,
    ) -> float:
        def _slice_summary(values: np.ndarray) -> str:
            if values.size == 0:
                return "n=0"
            min_val = float(np.min(values))
            max_val = float(np.max(values))
            zero_frac = float(np.isclose(values, 0.0).mean())
            one_frac = float(np.isclose(values, 1.0).mean())
            return (
                f"n={values.size}, min={min_val:.4g}, max={max_val:.4g}, "
                f"zero_frac={zero_frac:.3f}, one_frac={one_frac:.3f}"
            )

        if y_true_slice.size == 0 or y_pred_slice.size == 0:
            logger.warning(
                "Bias ratio '%s' skipped: empty slice (true: %s, pred: %s).",
                label,
                _slice_summary(y_true_slice),
                _slice_summary(y_pred_slice),
            )
            return np.nan
        mean_true = float(np.mean(y_true_slice))
        if np.isnan(mean_true) or np.isclose(mean_true, 0.0):
            logger.warning(
                "Bias ratio '%s' skipped: mean observed is %.4g (true: %s).",
                label,
                mean_true,
                _slice_summary(y_true_slice),
            )
            return np.nan
        return float(np.mean(y_pred_slice)) / mean_true

    # Overall bias
    overall_bias_ratio = _safe_bias_ratio(y_true, y_pred, "overall")

    # Bottom and top quartiles
    bias_bottom = _safe_bias_ratio(
        y_true[bottom_indices], y_pred[bottom_indices], "bottom_quartile"
    )
    bias_top = _safe_bias_ratio(
        y_true[top_indices], y_pred[top_indices], "top_quartile"
    )

    # Decile-based bias (rank-based to handle tied values)
    bias_per_decile = []
    order = np.argsort(y_true)
    n_obs = order.size
    if n_obs == 0:
        logger.warning("Decile bins skipped: no observations.")
        bias_per_decile = [np.nan] * 10
    else:
        edges = np.linspace(0, n_obs, 11, dtype=int)
        for idx in range(10):
            start = int(edges[idx])
            end = int(edges[idx + 1])
            bin_idx = order[start:end]
            if bin_idx.size == 0:
                logger.warning(
                    "Decile bin %d empty (range %d:%d).", idx + 1, start, end
                )
                bias_per_decile.append(np.nan)
                continue
            bias = _safe_bias_ratio(
                y_true[bin_idx],
                y_pred[bin_idx],
                f"decile_{idx + 1}",
            )
            bias_per_decile.append(bias)

    return {
        "overall_bias_ratio": overall_bias_ratio,
        "bias_bottom": bias_bottom,
        "bias_top": bias_top,
        "bias_per_decile": bias_per_decile,
    }
