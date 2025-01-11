import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr
from scipy.special import expit


def plot_prior_and_posterior_distributions(
    y_obs: np.array,
    y_obs_trans: np.array,
    y_prior_pred: np.array,
    y_posterior_pred: np.array,
    titles: list[str],
) -> None:
    """
    Plot four distributions in a 2x2 grid: observed data, transformed observed data,
    prior predictive sample, and posterior predictive sample.

    Args:
        - y_obs: Observed data.
        - y_obs_trans: Transformed observed data.
        - y_prior_pred: Prior predictive sample.
        - y_posterior_pred: Posterior predictive sample.
        - titles: List of subplot titles.
        - figsize: Size of the figure.
    """

    # Create figure and 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    alpha = 0.6
    bins = 50

    # Plot observed data
    sns.histplot(y_obs, bins=bins, kde=False, alpha=alpha, ax=axes[0, 0])
    axes[0, 0].set_title(titles[0])
    axes[0, 0].set_xlabel("Relative abundance")
    axes[0, 0].set_ylabel("Frequency")

    # Plot transformed observed data
    sns.histplot(y_obs_trans, bins=bins, kde=False, alpha=alpha, ax=axes[0, 1])
    axes[0, 1].set_title(titles[1])
    axes[0, 1].set_xlabel("Relative abundance")
    axes[0, 1].set_ylabel("Frequency")

    # Plot prior predictive sample
    sns.histplot(y_prior_pred, bins=bins, kde=True, alpha=alpha, ax=axes[1, 0])
    axes[1, 0].set_title(titles[2])
    axes[1, 0].set_xlabel("Relative abundance")
    axes[1, 0].set_ylabel("Frequency")

    # Plot posterior predictive sample
    sns.histplot(y_posterior_pred, bins=bins, kde=True, alpha=alpha, ax=axes[1, 1])
    axes[1, 1].set_title(titles[3])
    axes[1, 1].set_xlabel("Relative abundance")
    axes[1, 1].set_ylabel("Frequency")

    # Adjust layout for clarity
    plt.tight_layout()
    plt.show()


def plot_prior_distribution(
    prior_samples: az.InferenceData, category: str, variable: str
) -> None:

    if category == "prior":
        data = prior_samples.prior
    elif category == "prior_predictive":
        data = prior_samples.prior_predictive
    else:
        data = prior_samples.observed_data

    az.plot_dist(
        data[variable],
        figsize=(6, 3),
        kind="hist",
        color="C1",
        hist_kwargs=dict(alpha=0.6, bins=50),
    )

    plt.title(f"{category}: {variable}", fontsize=12)

    plt.tick_params(axis="x", labelsize=10)
    plt.tick_params(axis="y", labelsize=10)

    max_ticks = 15
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    plt.xticks(rotation=45)

    plt.show()


def assign_legible_trace_variables(
    trace: az.InferenceData,
    level_1_mapping: dict[int, str],
    level_2_mapping: dict[int, str],
    level_3_mapping: dict[int, str],
    variable_mapping: dict[int, str],
) -> az.InferenceData:
    """Assign legible names to dimensions of the trace for better interpretability."""
    # Update the coordinates of the regression parameters for level 1
    trace.posterior = trace.posterior.assign_coords(
        alpha_level_1_dim_0=[
            level_1_mapping[idx] for idx in trace.posterior.alpha_level_1_dim_0.values
        ],
        beta_level_1_dim_0=[
            level_1_mapping[idx] for idx in trace.posterior.beta_level_1_dim_0.values
        ],
        beta_level_1_dim_1=[
            variable_mapping[idx] for idx in trace.posterior.beta_level_1_dim_1.values
        ],
    )

    # Update the coordinates for level 2
    if "alpha_level_2_dim_0" in trace.posterior.dims:
        trace.posterior = trace.posterior.assign_coords(
            alpha_level_2_dim_0=[
                level_2_mapping[idx]
                for idx in trace.posterior.alpha_level_2_dim_0.values
            ],
            beta_level_2_dim_0=[
                level_2_mapping[idx]
                for idx in trace.posterior.beta_level_2_dim_0.values
            ],
            beta_level_2_dim_1=[
                variable_mapping[idx]
                for idx in trace.posterior.beta_level_2_dim_1.values
            ],
        )

    # Update the coordinates for level 3
    if "alpha_level_3_dim_0" in trace.posterior.dims:
        trace.posterior = trace.posterior.assign_coords(
            alpha_level_3_dim_0=[
                level_3_mapping[idx]
                for idx in trace.posterior.alpha_level_3_dim_0.values
            ],
            beta_level_3_dim_0=[
                level_3_mapping[idx]
                for idx in trace.posterior.beta_level_3_dim_0.values
            ],
            beta_level_3_dim_1=[
                variable_mapping[idx]
                for idx in trace.posterior.beta_level_3_dim_1.values
            ],
        )

    return trace


def filter_trace_by_group(
    trace: az.InferenceData, level: str, group_name: str
) -> az.InferenceData:
    posterior = trace.posterior
    filtered_trace = az.InferenceData()

    if level == "biome":
        filtered_alpha = posterior["alpha_biome"].sel(alpha_biome_dim_0=group_name)
        filtered_beta = posterior["beta_biome"].sel(beta_biome_dim_0=group_name)
        dataset = xr.Dataset(
            {"alpha_biome": filtered_alpha, "beta_biome": filtered_beta}
        )

    elif level == "realm":
        filtered_alpha = posterior["alpha_realm"].sel(alpha_realm_dim_0=group_name)
        filtered_beta = posterior["beta_realm"].sel(beta_realm_dim_0=group_name)
        dataset = xr.Dataset(
            {"alpha_realm": filtered_alpha, "beta_realm": filtered_beta}
        )

    filtered_trace = az.InferenceData(posterior=dataset)

    return filtered_trace


def inverse_transform_trace_data(trace: az.InferenceData) -> az.InferenceData:
    posterior = trace.posterior

    # Directly transforming alphas
    for alpha in ["mu_a", "alpha_biome", "alpha_realm"]:
        transformed_data = expit(posterior[alpha].values)
        trace.posterior[alpha] = xr.DataArray(
            transformed_data,
            dims=posterior[alpha].dims,
            coords=posterior[alpha].coords,
        )

    # Handling interactions between alphas and betas
    for alpha, beta in [
        ("mu_a", "mu_b"),
        ("alpha_biome", "beta_biome"),
        ("alpha_realm", "beta_realm"),
    ]:
        alpha_vals = posterior[alpha].values
        beta_vals = posterior[beta].values

        # Determine the shape differences and expand dimensions accordingly
        shape_diff = len(beta_vals.shape) - len(alpha_vals.shape)
        for _ in range(shape_diff):
            alpha_vals = np.expand_dims(alpha_vals, axis=-1)  # Expand the last axis

        # Repeat alpha_vals to match the full shape of beta_vals if necessary
        repeats = [
            beta_vals.shape[dim] if dim >= len(alpha_vals.shape) else 1
            for dim in range(len(beta_vals.shape))
        ]
        alpha_vals = np.tile(alpha_vals, repeats)

        # Compute transformed data
        transformed_data = expit(alpha_vals + beta_vals) - expit(alpha_vals)
        trace.posterior[beta] = xr.DataArray(
            transformed_data,
            dims=posterior[beta].dims,
            coords=posterior[beta].coords,
        )

    return trace


def forest_plot(trace: az.InferenceData, var_names: list[str]) -> None:
    axes = az.plot_forest(
        data=trace,
        var_names=var_names,
        combined=True,
        hdi_prob=0.95,
    )

    ax = axes[0]
    labels = [item.get_text() for item in ax.get_yticklabels()]
    new_labels = []
    for label in labels:
        new_label = (
            label.replace("[", "")
            .replace("]", "")
            .replace(",", ":")
            .replace("beta", "")
        )
        new_labels.append(new_label)

    for label in ax.get_yticklabels():
        label.set_fontsize(10)
    for label in ax.get_xticklabels():
        label.set_fontsize(10)

    # Set the new labels to the y-axis
    ax.set_yticklabels(new_labels)

    plt.tight_layout()


def calculate_bayesian_r2(y_true: np.array, y_pred: np.array) -> list[float]:

    r2_values = []
    for s in range(y_pred.shape[0]):
        pred = y_pred[s]
        r2 = np.var(pred) / (np.var(pred) + np.var(y_true - pred))
        r2_values.append(r2)

    return r2_values


def plot_performance_distribution(values: np.array, metric: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, alpha=0.6, color="g", edgecolor="black")
    mean_val = np.mean(values)
    perc_2_5 = np.percentile(values, q=2.5)
    perc_97_5 = np.percentile(values, q=97.5)
    plt.axvline(
        mean_val,
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Mean: {mean_val:.3f}",
    )
    plt.axvline(
        perc_2_5,
        color="b",
        linestyle="dashed",
        linewidth=1,
        label=f"CI 2.5%: {perc_2_5:.3f}",
    )
    plt.axvline(
        perc_97_5,
        color="b",
        linestyle="dashed",
        linewidth=1,
        label=f"CI 97.5%: {perc_97_5:.3f}",
    )
    plt.title(f"Distribution of {metric}")
    plt.xlabel(f"{metric}")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
