from typing import Any

import numpy as np
import polars as pl
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

from core.utils.general_utils import create_logger

logger = create_logger(__name__)


def standardize_continuous_covariates(
    df: pl.DataFrame, vars_to_scale: list[str]
) -> pl.DataFrame:
    """
    Standardize the continuous covariates in the dataframe, by subtracting the
    mean and dividing by the standard devation: (x - mu_x) / sigma_x.

    Args:
        - df: Dataframe containing all covariates in the 'vars_to_scale' list.
        - vars_to_scale: List of names of covariates to be standardized.

    Returns:
        - df_res: Dataframe with standardized continuous covariates.

    """
    logger.info("Standardizing continuous covariates.")

    scaler = StandardScaler()
    data_to_scale = df.select(vars_to_scale).to_numpy()
    df_scaled = pl.DataFrame(scaler.fit_transform(data_to_scale), schema=vars_to_scale)

    # Check for NaN and infinite values, report and replace with zeros
    # Any such values are likely due to floating point precision issues
    for col in vars_to_scale:
        inf_sum = df_scaled.get_column(col).is_infinite().sum()
        nan_sum = df_scaled.get_column(col).is_nan().sum()
        if inf_sum > 0:
            logger.warning(f"{inf_sum} infinite values found in {col}. Filling with 0.")
        if nan_sum > 0:
            logger.warning(f"{nan_sum} NaN values found in {col}. Filling with 0.")
    df_scaled = df_scaled.fill_nan(0)  # Fill all invalid values with 0

    # Replace original columns with the standardized ones
    df_res = pl.concat([df.drop(vars_to_scale), df_scaled], how="horizontal")

    logger.info("Finished standardizing continuous covariates.")

    return df_res


def augment_prediction_dataframe(
    df: pl.DataFrame, model_type: str, response_transform: str
) -> pl.DataFrame:
    y_obs_trans = df.get_column("Observed")
    y_obs = _inverse_transform_response(y_obs_trans, response_transform)

    if model_type == "bayesian":
        y_pred_trans = np.clip(df.get_column("Predicted"), 0, 1)
        y_pred = _inverse_transform_response(y_pred_trans, response_transform)

    elif model_type == "lmm":
        y_pred_re_trans = np.clip(df.get_column("Predicted_RE"), 0, 1)
        y_pred_re = _inverse_transform_response(y_pred_re_trans, response_transform)

        y_pred_fe_trans = np.clip(df.get_column("Predicted_FE"), 0, 1)
        y_pred_fe = _inverse_transform_response(y_pred_fe_trans, response_transform)

    output_dict = {
        "SSBS": df.get_column("SSBS"),
        "Observed": y_obs,
        "Observed_transformed": y_obs_trans,
    }

    if model_type == "bayesian":
        output_dict["Predicted"] = y_pred
        output_dict["Residuals"] = y_pred - y_obs
        output_dict["Predicted_transformed"] = y_pred_trans

    elif model_type == "lmm":
        output_dict["Predicted_RE"] = y_pred_re
        output_dict["Residuals_RE"] = y_pred_re - y_obs
        output_dict["Predicted_RE_transformed"] = y_pred_re_trans

        output_dict["Predicted_FE"] = y_pred_fe
        output_dict["Residuals_FE"] = y_pred_fe - y_obs
        output_dict["Predicted_FE_transformed"] = y_pred_fe_trans

    df_pred = pl.DataFrame(output_dict)

    return df_pred


def approximate_change_predictions(
    df_pred: pl.DataFrame, df_site_info: pl.DataFrame, model_type: str, mode: str
) -> pl.DataFrame:
    """
    Approximate change predictions by calculating pairwise differences in predictions
    and observed values between each pair of sites within the same block.

    Args:
        - df_pred: Prediction dataframe containing predicted and observed values.
        - df_site_info: Site information dataframe with "SSBS" and "Block" columns.
        - model_type: Type of the model ('bayesian' or 'lmm').
        - mode: Mode of operation ('training' or 'crossval').

    Returns:
        - pl.DataFrame: Output dataframe with columns "Site_1", "Site_2",
          "Delta_predicted_re", "Delta_predicted_fe", "Delta_observed".
    """
    # Join the two dataframes on site id
    df_merged = df_pred.join(df_site_info, on="SSBS", how="inner")

    # Filter out cases where the study doesn't contain spatial blocks
    df_merged = df_merged.filter(~pl.col("Block").is_null())

    # Pairwise differences in predictions and observed within each group
    pairwise_results = []
    grouped = df_merged.group_by("SSB").agg(pl.col("*"))

    for group in grouped.iter_rows(named=True):
        group_sites = group["SSBS"]
        predominant_land_uses = group["Predominant_land_use"]
        observed = group["Observed"]

        if model_type == "bayesian":
            predicted = group["Predicted"]
        elif model_type == "lmm":
            predicted_re = group["Predicted_RE"]
            predicted_fe = group["Predicted_FE"]

        # Calculate pairwise differences
        n_sites = len(group_sites)
        seen_pairs = set()  # To keep track of processed site pairs

        for i in range(n_sites):
            for j in range(i + 1, n_sites):
                # Skip pairs with the same Predominant_land_use
                if predominant_land_uses[i] == predominant_land_uses[j]:
                    continue

                # Create a unique pair identifier
                pair_key = tuple(sorted((group_sites[i], group_sites[j])))

                # Skip if this pair was already processed
                if pair_key in seen_pairs:
                    continue

                seen_pairs.add(pair_key)

                result = {
                    "Site_1": group_sites[i],
                    "Site_2": group_sites[j],
                    "Delta_observed": observed[j] - observed[i],
                }

                if model_type == "bayesian":
                    result["Delta_predicted"] = predicted[j] - predicted[i]
                    result["Delta_residuals"] = (
                        result["Delta_predicted"] - result["Delta_observed"]
                    )
                elif model_type == "lmm":
                    result["Delta_predicted_RE"] = predicted_re[j] - predicted_re[i]
                    result["Delta_predicted_FE"] = predicted_fe[j] - predicted_fe[i]
                    result["Delta_residuals_RE"] = (
                        result["Delta_predicted_RE"] - result["Delta_observed"]
                    )
                    result["Delta_residuals_FE"] = (
                        result["Delta_predicted_FE"] - result["Delta_observed"]
                    )

                pairwise_results.append(result)

        output_df = pl.DataFrame(pairwise_results)

    return output_df


def calculate_performance_metrics(
    df: pl.DataFrame, model_type: str, mode: str, pred_type: str
) -> dict[str, float]:
    """
    Evaluate model performance and log results. Calculates R2, mean absolute
    error, Pearson correlation, and Spearman rank correlation. All metrics
    are calculated on back-transformed data, including for top and bottom
    quartiles of observed values.

    Args:
        - df: Dataframe containing 'Observed' and 'Predicted' columns.
        - model_type: Type of model ('bayesian' or 'lmm').
        - mode: Mode of evaluation ('training' or 'crossval').
        - response_transform: Method used to transform the response variable.

    Returns:
        - performance_metrics: A dictionary containing the calculated metrics.
    """
    logger.info("Evaluating model performance metrics.")

    # Extract observed and predicted values based on prediction type
    if pred_type == "state":
        # Observed values
        y_true = df.get_column("Observed").to_numpy()
        y_true_trans = df.get_column("Observed_transformed").to_numpy()

        # Predicted values
        if model_type == "bayesian":
            y_pred = df.get_column("Predicted").to_numpy()
            y_pred_trans = df.get_column("Predicted_transformed").to_numpy()
        elif model_type == "lmm" and mode == "training":
            y_pred = df.get_column("Predicted_RE").to_numpy()
            y_pred_trans = df.get_column("Predicted_RE_transformed").to_numpy()
        elif model_type == "lmm" and mode == "crossval":
            y_pred = df.get_column("Predicted_FE").to_numpy()
            y_pred_trans = df.get_column("Predicted_FE_transformed").to_numpy()
            y_pred_re = df.get_column("Predicted_RE").to_numpy()
            y_pred_trans_re = df.get_column("Predicted_RE_transformed").to_numpy()

    elif pred_type == "change":
        # Observed changes
        y_true = df.get_column("Delta_observed").to_numpy()

        # Predicted changes
        if model_type == "bayesian":
            y_pred = df.get_column("Delta_predicted").to_numpy()
        elif model_type == "lmm" and mode == "training":
            y_pred = df.get_column("Delta_predicted_RE").to_numpy()
        elif model_type == "lmm" and mode == "crossval":
            y_pred = df.get_column("Delta_predicted_FE").to_numpy()
            y_pred_re = df.get_column("Delta_predicted_RE").to_numpy()

    # Calculate quartiles for observed values
    q1 = np.percentile(y_true, 25)
    q3 = np.percentile(y_true, 75)
    bottom_indices = np.where(y_true <= q1)[0]
    top_indices = np.where(y_true >= q3)[0]

    # Bias calculations for overall, bottom, and top quartiles
    bias_metrics = calculate_bias_metrics(y_true, y_pred, bottom_indices, top_indices)

    # Calculate overall metrics
    r2_std = r2_score(y_true, y_pred)
    if pred_type == "state":
        r2_std_trans = r2_score(y_true_trans, y_pred_trans)

    if model_type == "lmm" and mode == "crossval":
        r2_var = np.var(y_pred) / (np.var(y_pred_re) + np.var(y_true - y_pred))
        if pred_type == "state":
            r2_var_trans = np.var(y_pred_trans) / (
                np.var(y_pred_trans_re) + np.var(y_true_trans - y_pred_trans)
            )
    else:
        r2_var = np.var(y_pred) / (np.var(y_pred) + np.var(y_true - y_pred))
        if pred_type == "state":
            r2_var_trans = np.var(y_pred_trans) / (
                np.var(y_pred_trans) + np.var(y_true_trans - y_pred_trans)
            )

    mean_abs_error = mean_absolute_error(y_true, y_pred)
    median_abs_error = median_absolute_error(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

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
    if pred_type == "state":
        logger.info(
            "\n"
            f" - R2 (std, not back-transformed): {r2_std_trans:.3f}\n"
            f" - R2 (var expl, not back-transformed): {r2_var_trans:.3f}\n"
        )

    # Metrics for bottom quartile
    bottom_quartile_true = y_true[bottom_indices]
    bottom_quartile_pred = y_pred[bottom_indices]

    r2_std_bottom = r2_score(bottom_quartile_true, bottom_quartile_pred)
    if model_type == "lmm" and mode == "crossval":
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
    pearson_corr_bottom, _ = pearsonr(bottom_quartile_true, bottom_quartile_pred)
    spearman_corr_bottom, _ = spearmanr(bottom_quartile_true, bottom_quartile_pred)

    # Metrics for top quartile
    top_quartile_true = y_true[top_indices]
    top_quartile_pred = y_pred[top_indices]

    r2_std_top = r2_score(top_quartile_true, top_quartile_pred)
    if model_type == "lmm" and mode == "crossval":
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
    pearson_corr_top, _ = pearsonr(top_quartile_true, top_quartile_pred)
    spearman_corr_top, _ = spearmanr(top_quartile_true, top_quartile_pred)

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
    if pred_type == "state":
        metrics.update(
            {
                "r2_std_trans": r2_std_trans,
                "r2_var_trans": r2_var_trans,
            }
        )

    return metrics


def calculate_bias_metrics(
    y_true: np.array, y_pred: np.array, bottom_indices: np.array, top_indices: np.array
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
    # Overall bias
    overall_bias_ratio = np.mean(y_pred) / np.mean(y_true)

    # Bottom and top quartiles
    bias_bottom = np.mean(y_pred[bottom_indices]) / np.mean(y_true[bottom_indices])
    bias_top = np.mean(y_pred[top_indices]) / np.mean(y_true[top_indices])

    # Decile-based bias
    deciles = np.percentile(y_true, np.arange(10, 100, 10))
    bias_per_decile = []
    indices_remaining = np.arange(len(y_true))

    for threshold in deciles:
        indices_below_threshold = indices_remaining[
            y_true[indices_remaining] < threshold
        ]
        bias = np.mean(y_pred[indices_below_threshold]) / np.mean(
            y_true[indices_below_threshold]
        )
        bias_per_decile.append(bias)
        indices_remaining = np.setdiff1d(indices_remaining, indices_below_threshold)

    # Final group for remaining values
    bias = np.mean(y_pred[indices_remaining]) / np.mean(y_true[indices_remaining])
    bias_per_decile.append(bias)

    return {
        "overall_bias_ratio": overall_bias_ratio,
        "bias_bottom": bias_bottom,
        "bias_top": bias_top,
        "bias_per_decile": bias_per_decile,
    }


def _inverse_transform_response(y: np.array, method: str) -> np.array:
    """
    Back-transform the response variable (predictions or observed values) to
    the original scale.

    Args:
        - y: The predicted / observed data from the model.
        - method: Method used to originally transform the response variable.
        - adjust: Adjustment value used during the transformation.

    Returns:
        - res: The back-transformed response variable.
    """
    if method == "logit":
        res = expit(y)  # Inverse logit transformation
    elif method == "sqrt":
        res = np.square(y)  # Inverse square-root transformation
    elif method == "adjust":
        adjust = 0.001
        res = np.where(
            y < adjust,  # If y is less than the adjustment threshold
            0,
            np.where(
                y > (1 - adjust), 1, y  # If y is greater than the upper threshold
            ),
        )
    else:
        res = y

    return res
