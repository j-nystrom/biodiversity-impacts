from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score


def plot_observed_and_predicted_distributions(
    y_true: np.array,
    y_pred: np.array,
    titles: list = ["Observed data", "Predicted values"],
) -> None:
    """
    Plot two distributions side by side: observed data and predicted values.

    Args:
        - y_true: Observed data.
        - y_pred: Predicted data.
        - titles: List of subplot titles.
        - figsize: Size of the figure.
        - bins: Number of bins for the histograms.
        - alpha: Transparency level for the histograms.
    """
    # Create figure and subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    alpha = 0.6
    bins = 50

    # Plot observed data
    sns.histplot(y_true, bins=bins, kde=False, alpha=alpha, ax=axes[0])
    axes[0].set_title(titles[0])
    axes[0].set_xlabel("Relative abundance")
    axes[0].set_ylabel("Frequency")

    # Plot predicted data
    sns.histplot(y_pred, bins=bins, kde=False, alpha=alpha, ax=axes[1])
    axes[1].set_title(titles[1])
    axes[1].set_xlabel("Relative abundance")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_calibration_and_residuals(
    y_true: np.array,
    y_pred: np.array,
    y_residual: np.array,
    metrics: dict[str, Any],
) -> None:
    """
    Plot calibration and residual diagnostics in a 2x2 grid using provided
    residuals and bias metrics.

    Args:
        - y_true: Observed values (ground truth).
        - y_pred: Predicted values.
        - y_residual: Residuals (predicted - observed).
        - bias_metrics: Dictionary containing bias values (overall, per decile,
            quartiles).
    """
    # Plot parameters
    params = {
        "alpha_scatter": 0.2,
        "alpha_density": 0.1,
        "alpha_bar": 0.8,
        "scatter_size": 15,
        "line_style": "--",
        "line_width": 1.25,
        "line_color": "red",
    }

    # Extract data from metrics dictionary
    bias_metrics = metrics["bias_metrics"]
    bias_values = bias_metrics["bias_per_decile"]
    overall_bias_ratio = bias_metrics["overall_bias_ratio"]
    r2_std = metrics["r2_std"]
    r2_var = metrics["r2_var"]

    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Calibration plot (predicted vs observed)
    sns.scatterplot(
        ax=axes[0, 0],
        x=y_true,
        y=y_pred,
        alpha=params["alpha_scatter"],
        s=params["scatter_size"],
    )
    axes[0, 0].plot(
        [min(y_true), max(y_true)],
        [min(y_true), max(y_true)],
        linestyle=params["line_style"],
        color=params["line_color"],
        linewidth=params["line_width"],
    )
    axes[0, 0].set_title("Calibration plot (predicted vs observed)")
    axes[0, 0].set_xlabel("Observed")
    axes[0, 0].set_ylabel("Predicted")

    # Add the R^2 score text
    vertical_offset = 0.02  # Space between each line of text
    horizontal_offset = 0.02
    axes[0, 0].text(
        horizontal_offset,  # Horizontal offset (left aligned)
        1.0 - vertical_offset,  # Top position
        f"R² (standard def.)= {r2_std:.3f}",
        fontsize=11,
        transform=axes[0, 0].transAxes,  # Use axes coordinate system
        verticalalignment="top",  # Align text to top
        horizontalalignment="left",  # Align text to left
    )
    axes[0, 0].text(
        horizontal_offset,
        1.0 - 3 * vertical_offset,  # Adjust for the second line
        f"R² (var. explained)= {r2_var:.3f}",
        fontsize=11,
        transform=axes[0, 0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )

    # Residuals by observed values
    sns.scatterplot(
        ax=axes[0, 1],
        x=y_true,
        y=y_residual,
        alpha=params["alpha_scatter"],
        s=params["scatter_size"],
    )
    axes[0, 1].axhline(
        y=0,
        linestyle=params["line_style"],
        color=params["line_color"],
        linewidth=params["line_width"],
    )
    axes[0, 1].set_title("Residuals by observed values")
    axes[0, 1].set_xlabel("Observed")
    axes[0, 1].set_ylabel("Residuals")

    # Residual density plot
    sns.kdeplot(
        ax=axes[1, 0],
        data=y_residual,
        fill=True,
        alpha=params["alpha_density"],
    )
    axes[1, 0].set_title("Residual density plot")
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Density")

    # Bias plot
    bars = axes[1, 1].bar(
        x=np.arange(10) + 1,
        height=bias_values,
        alpha=params["alpha_bar"],
    )
    axes[1, 1].axhline(
        y=overall_bias_ratio,
        color=params["line_color"],
        linestyle=params["line_style"],
        linewidth=params["line_width"],
    )
    axes[1, 1].text(
        x=0.95,
        y=0.8,
        s=f"Overall bias ratio: {round(overall_bias_ratio, 3)}",
        color="black",
        va="top",
        ha="right",
        transform=axes[1, 1].transAxes,
        fontsize=11,
    )
    for bar in bars:
        yval = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            yval,
            round(yval, 2),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    axes[1, 1].set_title("Total bias by observed values")
    axes[1, 1].set_xlabel("Observed value deciles")
    axes[1, 1].set_ylabel("Bias")

    # Finalize layout
    plt.tight_layout()
    plt.show()


def plot_calibration_by_group(
    y_true: np.array,
    y_pred: np.array,
    group_idx: np.array,
    group_mapping: dict[int, str],
) -> None:
    """
    Plot calibration (Observed vs Predicted) for each group, sorted by the
    number of observations.

    Args:
        y_true: Array of true observed values.
        y_pred: Array of predicted values.
        group_idx: Array of group indices corresponding to each observation.
        group_mapping: Mapping of group indices to their names.
    """
    # Determine the number of unique groups and their sizes
    groups, counts = np.unique(group_idx, return_counts=True)

    # Sort groups by the number of observations, from largest to smallest
    sorted_groups = groups[np.argsort(-counts)]
    sorted_counts = counts[np.argsort(-counts)]

    # Log group sizes
    print("Groups sorted by size (largest to smallest):")
    for group, count in zip(sorted_groups, sorted_counts):
        print(f"{group_mapping.get(group, group)}: {count} observations")

    # Calculate the dimensions of the grid of subplots
    n_cols = 3
    n_rows = int(np.ceil(len(sorted_groups) / n_cols))

    # Set the figsize dynamically based on the number of groups (arbitrary
    # width and height per subplot)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        constrained_layout=True,
        sharey=True,
    )

    axes = axes.ravel()  # Flatten the axs array for easy iteration
    for idx, group in enumerate(sorted_groups):
        # Select the data for the current group
        mask = group_idx == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]

        # Calculate the R^2 scores for the current group
        r2_std = r2_score(y_true_group, y_pred_group)
        r2_var = np.var(y_pred_group) / (
            np.var(y_pred_group) + np.var(y_true_group - y_pred_group)
        )

        # Plot the actual vs predicted values for the current group
        axes[idx].scatter(y_true_group, y_pred_group, alpha=0.2)

        # Add the perfect fit line
        min_val, max_val = min(y_true_group), max(y_true_group)
        axes[idx].plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            linewidth=1.25,
        )

        # Add the R^2 score text
        axes[idx].text(
            0.05,
            0.95,
            f"R² (standard def.)= {r2_std:.3f}",
            fontsize=11,
            transform=axes[idx].transAxes,
        )
        axes[idx].text(
            0.05,
            0.9,
            f"R² (var. explained)= {r2_var:.3f}",
            fontsize=11,
            transform=axes[idx].transAxes,
        )

        # Set labels and title
        axes[idx].set_xlabel("Observed")
        axes[idx].set_ylabel("Predicted")
        axes[idx].set_title(f"{group_mapping.get(group, group)}")

        # Ensure the subplot axes do not overlap
        axes[idx].tick_params(axis="x")

    # Hide any empty subplots that aren't used (if the number of groups is not
    # a perfect square)
    for ax in axes[len(sorted_groups) :]:
        ax.set_visible(False)

    plt.show()


def plot_residuals_by_group(
    y_true: np.array,
    y_pred: np.array,
    group_idx: np.array,
    group_mapping: dict[int, str],
) -> None:
    # Determine the number of unique groups
    groups = np.unique(group_idx)
    n_groups = len(groups)

    # Calculate the dimensions of the grid of subplots
    n_cols = 3
    n_rows = int(np.ceil(n_groups / n_cols))

    # Set the figsize dynamically based on the number of groups (arbitrary
    # width and height per subplot)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), constrained_layout=True
    )

    # Flatten the axs array for easy iteration
    axs = axs.ravel()

    for idx, group in enumerate(groups):
        # Select the data for the current group
        mask = group_idx == group
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        residuals_group = y_pred_group - y_true_group

        # Plot the actual values and residuals for the current group
        axs[idx].scatter(y_true_group, residuals_group, alpha=0.5)
        axs[idx].axhline(y=0, color="r", linestyle="--", linewidth=2)

        # Set labels and title
        axs[idx].set_xlabel("Observed")
        axs[idx].set_ylabel("Residuals")
        axs[idx].set_title(f"{group_mapping.get(group, group)}")

        # Ensure the subplot axes do not overlap
        axs[idx].tick_params(axis="x", labelrotation=45)

    # Hide any empty subplots that aren't used (if the number of groups is not
    # a perfect square)
    for ax in axs[n_groups:]:
        ax.set_visible(False)

    plt.show()


def print_evaluation_metrics_train(metrics: dict[str, Any]) -> None:
    """Print model performance metrics in a structured format."""
    print("Model performance metrics")
    print("=" * 80)

    # Overall metrics
    print("Overall metrics:")
    print(f"  - R² (standard def): {metrics['r2_std']:.3f}")
    print(f"  - R² (variance explained): {metrics['r2_var']:.3f}")
    print(f"  - Mean absolute error: {metrics['mean_abs_error']:.3f}")
    print(f"  - Median absolute error: {metrics['median_abs_error']:.3f}")
    print(f"  - Pearson correlation: {metrics['pearson_corr']:.3f}")
    print(f"  - Spearman correlation: {metrics['spearman_corr']:.3f}")
    print(f"  - Bias ratio: {metrics['bias_metrics']['overall_bias_ratio']:.3f}")
    print()

    # Bottom quartile metrics
    print("Bottom quartile metrics:")
    print(f"  - R² (standard def): {metrics['r2_std_bottom']:.3f}")
    print(f"  - R² (variance explained): {metrics['r2_var_bottom']:.3f}")
    print(f"  - Mean absolute error: {metrics['mean_abs_error_bottom']:.3f}")
    print(f"  - Median absolute error: {metrics['median_abs_error_bottom']:.3f}")
    print(f"  - Pearson correlation: {metrics['pearson_corr_bottom']:.3f}")
    print(f"  - Spearman correlation: {metrics['spearman_corr_bottom']:.3f}")
    print(f"  - Bias ratio: {metrics['bias_metrics']['bias_bottom']:.3f}")
    print()

    # Top quartile metrics
    print("Top quartile metrics:")
    print(f"  - R² (standard def): {metrics['r2_std_top']:.3f}")
    print(f"  - R² (variance explained): {metrics['r2_var_top']:.3f}")
    print(f"  - Mean absolute error: {metrics['mean_abs_error_top']:.3f}")
    print(f"  - Median absolute error: {metrics['median_abs_error_top']:.3f}")
    print(f"  - Pearson correlation: {metrics['pearson_corr_top']:.3f}")
    print(f"  - Spearman correlation: {metrics['spearman_corr_top']:.3f}")
    print(f"  - Bias ratio: {metrics['bias_metrics']['bias_top']:.3f}")
    print()


def print_evaluation_metrics_crossval(metrics: list[dict[str, Any]]) -> None:
    """
    Print model performance metrics for cross-validation in a structured format.
    Shows mean, min, and max for each metric across all folds for test and train.
    """
    print("Cross-validation performance metrics")
    print("=" * 80)

    # Helper function to calculate mean, min, and max for a metric across folds
    def summarize_metric(
        metrics: list[dict[str, dict[str, float]]], mode: str, key: str
    ) -> tuple[float, float, float]:
        values = [fold[mode][key] for fold in metrics]
        return np.mean(values), np.min(values), np.max(values)

    # Sections to report metrics for
    sections = [
        (
            "Overall metrics",
            [
                ("R² (standard def.)", "r2_std"),
                ("R² (variance explained)", "r2_var"),
                ("Mean absolute error", "mean_abs_error"),
                ("Median absolute error", "median_abs_error"),
                ("Pearson correlation", "pearson_corr"),
                ("Spearman correlation", "spearman_corr"),
            ],
        ),
        (
            "Bottom quartile metrics",
            [
                ("R² (standard def.)", "r2_std_bottom"),
                ("R² (variance explained)", "r2_var_bottom"),
                ("Mean absolute error", "mean_abs_error_bottom"),
                ("Median absolute error", "median_abs_error_bottom"),
                ("Pearson correlation", "pearson_corr_bottom"),
                ("Spearman correlation", "spearman_corr_bottom"),
            ],
        ),
        (
            "Top quartile metrics",
            [
                ("R² (standard def.)", "r2_std_top"),
                ("R² (variance explained)", "r2_var_top"),
                ("Mean absolute error", "mean_abs_error_top"),
                ("Median absolute error", "median_abs_error_top"),
                ("Pearson correlation", "pearson_corr_top"),
                ("Spearman correlation", "spearman_corr_top"),
            ],
        ),
    ]

    # Print metrics for each section
    for section_name, metric_keys in sections:
        print(section_name)
        print("-" * 80)

        for display_name, key in metric_keys:
            test_mean, test_min, test_max = summarize_metric(metrics, "test", key)
            train_mean, train_min, train_max = summarize_metric(metrics, "train", key)

            print(f"  - {display_name}:")
            print(
                f"   Test:  mean: {test_mean:.3f} | min: {test_min:.3f} | "
                f"max: {test_max:.3f}"
            )
            print(
                f"   Train: mean: {train_mean:.3f} | min: {train_min:.3f} | "
                f"max: {train_max:.3f}"
            )

        # Add bias metrics for each section
        if "Overall" in section_name:
            test_bias = [
                fold["test"]["bias_metrics"]["overall_bias_ratio"] for fold in metrics
            ]
            train_bias = [
                fold["train"]["bias_metrics"]["overall_bias_ratio"] for fold in metrics
            ]
        elif "Bottom" in section_name:
            test_bias = [
                fold["test"]["bias_metrics"]["bias_bottom"] for fold in metrics
            ]
            train_bias = [
                fold["train"]["bias_metrics"]["bias_bottom"] for fold in metrics
            ]
        elif "Top" in section_name:
            test_bias = [fold["test"]["bias_metrics"]["bias_top"] for fold in metrics]
            train_bias = [fold["train"]["bias_metrics"]["bias_top"] for fold in metrics]

        print("  - Bias ratio:")
        print(
            f"   Test:  mean: {np.mean(test_bias):.3f} | min: {np.min(test_bias):.3f}"
            f" | max: {np.max(test_bias):.3f}"
        )
        print(
            f"   Train: mean: {np.mean(train_bias):.3f} | min: {np.min(train_bias):.3f}"
            f" | max: {np.max(train_bias):.3f}"
        )
        print()

    print("=" * 80)
