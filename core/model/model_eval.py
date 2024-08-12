from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from scipy.special import expit
from sklearn.metrics import mean_absolute_error, median_absolute_error

from core.utils.general_utils import create_logger

logger = create_logger(__name__)


def plot_prior_distribution(model: pm.Model) -> None:
    """
    Plot the results of forward sampling of the prior predictive distribution,
    and compare this with the observed data distribution.

    Args:
        model: PyMC model object used for sampling.
    """
    logger.info("Plotting prior predictive distribution and observed data.")

    def _generate_plot(
        prior_samples: az.InferenceData, category: str, variable: str
    ) -> None:
        """Generate one plot each for prior predictive and observed data."""
        plt.figure(figsize=(6, 3))

        if category == "prior_predictive":
            data = prior_samples.prior_predictive
        elif category == "constant_data":
            data = prior_samples.constant_data

        # Arviz distribution plot
        az.plot_dist(
            data[variable],
            kind="hist",
            color="C1",
            hist_kwargs=dict(alpha=0.6, bins=50),
        )

        # Formatting of the plot
        plt.title(f"{category}: {variable}", fontsize=12)
        plt.tick_params(axis="x", labelsize=10)
        plt.tick_params(axis="y", labelsize=10)
        max_ticks = 15
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        plt.xticks(rotation=45)
        plt.show(block=False)

    # Sample from the prior predictive distribution and call plotting function
    with model:
        prior_samples = pm.sample_prior_predictive(samples=1000)

    for cat, var in zip(["constant_data", "prior_predictive"], ["y_obs", "y_like"]):
        _generate_plot(prior_samples, category=cat, variable=var)

    logger.info("Finished plotting prior predictive distribution and observed data.")


def make_predictions(
    model: pm.Model,
    trace: az.InferenceData,
    mode: str,
    data: dict[str, Any],
) -> tuple[np.array, az.InferenceData]:
    """
    Sample from the posterior predictive distribution to make predictions. How
    this is done depends on if predictions are made on the training or test
    data.

    Args:
        model: PyMC model object used for training.
        trace: The trace object from the fitted model.
        mode: Whether to make predictions on the training or test data.
        data: The data to make predictions on. If mode is "train", there is no
            need to pass data for the predictions.

    Returns:
        y_pred: The predicted values from the model.
        trace: The updated trace object from the model, incl. predictions.
    """
    # If training mode, there is no need to maniulate data
    if mode == "train":
        with model:
            pred = pm.sample_posterior_predictive(trace, predictions=True)

    # If test mode, we need to add the test data to the model and make sure all
    # indices are correct
    elif mode == "test":
        train_len = len(trace.observed_data["idx"])
        test_len = len(data["x"])
        data["coords"]["idx"] = np.arange(train_len, train_len + test_len)

        with model:
            pm.set_data(
                {
                    "x_obs": data["x"],
                    "x_subset": data["x_subset"],
                    "y_obs": data["y"],
                    "biome_realm_idx": data["biome_realm_idx"],
                    "biome_realm_eco_idx": data["biome_realm_eco_idx"],
                    "site_idx": data["site_idx"],
                },
                coords=data["coords"],
            )
            pred = pm.sample_posterior_predictive(
                trace,
                var_names=[
                    "y_like",
                    "alpha_eco_site",
                    "alpha_realm_site",
                    "ref_pred_eco",
                    "ref_pred_realm",
                ],
                predictions=True,
                extend_inferencedata=True,  # Modify the existing trace
            )

    y_pred = pred.predictions["y_like"].mean(dim=["chain"]).values

    return y_pred, trace


def inverse_transform_response(y: np.array, method: str) -> np.array:
    """
    Back-transforms the response variable to the original scale.

    Args:
        y: The predicted data from the model.
        method: The method used to originally transform the response variable.

    Returns:
        res: The back-transformed response variable.
    """
    if method == "logit":
        res = expit(y)
    elif method == "sqrt":
        res = np.square(y)
    else:
        res = y

    return res


def evaluate_model_performance(
    y_pred: np.array,
    y_true: np.array,
    trace: az.InferenceData,
    response_transform: str,
) -> dict[str, tuple[float, float]]:
    """
    Evaluate model performance by calculating R2 (before back-transformation of
    predictions and observed values) and mean / median absolute error (after
    back-transformation). Each metric is calculated for each sample in the
    postierior predictive distribution.

    Args:
        y_pred: The predicted values from the model.
        y_true: The observed values.
        trace: The trace object from the fitted model.
        response_transform: The method used to transform the response variable,
            used for back-transformation.

    Returns:
        performance_metrics: A dictionary with mean and standard deviation of
            each metric.
    """
    logger.info("Evaluating model performance metrics.")

    # Back-transfsorm the response variable for accuracy metrics
    y_pred_trans = inverse_transform_response(y_pred, response_transform)
    y_true_trans = inverse_transform_response(y_true, response_transform)

    r2_values = []
    mean_abs_err = []
    median_abs_err = []

    # Iterate over samples to get distributions of performance metrics
    for s in range(y_pred.shape[0]):
        r2 = np.var(y_pred[s]) / (np.var(y_pred[s]) + np.var(y_true - y_pred[s]))
        r2_values.append(r2)

        mean = mean_absolute_error(y_true_trans, y_pred_trans[s])
        mean_abs_err.append(mean)
        median = median_absolute_error(y_true_trans, y_pred_trans[s])
        median_abs_err.append(median)

    performance_metrics = {
        "r2": (np.mean(r2_values), np.std(r2_values)),
        "mean_abs_err": (np.mean(mean_abs_err), np.std(mean_abs_err)),
        "median_abs_err": (np.mean(median_abs_err), np.std(median_abs_err)),
    }

    logger.info("Finished calculating model performance metrics.")

    return performance_metrics


def log_model_performance(metrics: dict[str, tuple[float, float]]) -> None:
    """Log the results of the model performance evaluation."""

    logger.info("Model performance results: \n")
    logger.info(f"R2: {metrics['r2'][0]:.3f} (mean) | {metrics['r2'][1]:.3f} (std)")
    logger.info(
        f"Mean abs error: "
        f"{metrics['mean_abs_err'][0]:.3f} (mean) | "
        f"{metrics['mean_abs_err'][1]:.3f} (std)"
    )
    logger.info(
        f"Median abs error: "
        f"{metrics['median_abs_err'][0]:.3f} (mean) | "
        f"{metrics['median_abs_err'][1]:.3f} (std)"
    )
