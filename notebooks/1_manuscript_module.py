import json
import subprocess
import tempfile
import textwrap
import warnings
from pathlib import Path

import geopandas as gpd
import gower
import jupyter_black
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from scipy.special import logit
from scipy.stats import pearsonr

jupyter_black.load()

# Set global Seaborn theme
sns.set_theme(
    style="white",  # White background
    context="notebook",  # Default context; adjust font sizes for notebooks
    rc={
        "axes.spines.top": False,  # Remove top spine
        "axes.spines.right": False,  # Remove right spine
        "axes.grid": False,  # Disable gridlines
        "xtick.bottom": True,  # Enable bottom ticks
        "ytick.left": True,  # Enable left ticks
        "xtick.major.size": 6,  # Length of major x-axis ticks
        "ytick.major.size": 6,  # Length of major y-axis ticks
        "axes.titlesize": 14,  # Font size for titles
        "axes.labelsize": 11,  # Font size for axis labels
        "legend.fontsize": 11,  # Font size for legends
    },
)

color_scheme = {  # https://matplotlib.org/stable/gallery/color/named_colors.html
    "calibration": "teal",
    "fixed_eff": "lightsteelblue",
    "random_eff": "peachpuff",
    "training": "darkgray",
    "standard_cv": "darkseagreen",
    "cross_study_cv": "mediumseagreen",
    "train_line": "royalblue",
    "test_line": "orange",
    "group_scatter": "darkolivegreen",
}

summary_colors = {
    "Training": color_scheme["training"],
    "Standard CV": color_scheme["standard_cv"],
    "Cross-study CV": color_scheme["cross_study_cv"],
}

# ALPHA diversity: Base model runs

lmm_alpha_folders = {
    "Training": "lmm_alpha_train",
    "Standard CV": "lmm_alpha_standard_cv",
    "Cross-study CV": "lmm_alpha_cross_study",
}

bhm_alpha_folders = {
    "Training": "bhm_alpha_train",
    "Standard CV": "bhm_alpha_standard_cv",
    "Cross-study CV": "bhm_alpha_cross_study",
}

# BETA diversity: Base model runs

lmm_beta_folders = {
    "Training": "lmm_beta_train",
    "Standard CV": "lmm_beta_standard_cv",
    "Cross-study CV": "lmm_beta_cross_study",
}

bhm_beta_folders = {
    "Training": "bhm_beta_train",
    "Standard CV": "bhm_beta_standard_cv",
    "Cross-study CV": "bhm_beta_cross_study",
}

# Base paths and file endings
base_path = "../../data/saved_runs"
key_output_path = "key_output"
site_info_filename = "site_info.parquet"
bhm_added_output_path = "additional_output"


def load_prediction_dataframes(
    mode_folders: dict[str, str], base_path: str = base_path
) -> dict[str, pl.DataFrame]:
    """
    Load prediction dataframes from run folders. Given {mode: run_folder},
    return {mode: df_predictions}.

    Training runs read key_output/train_predictions.parquet.
    Cross-validation runs read key_output/test_predictions_fold_*.parquet and
    concatenate the test folds (with fold and mode labels).
    """
    out = {}
    for mode, run_folder in mode_folders.items():
        run_path = Path(base_path) / run_folder
        key_output_dir = run_path / "key_output"
        if "training" in mode.lower():
            df = pl.read_parquet(run_path / "key_output" / "train_predictions.parquet")
            df = df.with_columns(pl.lit("Training").alias("mode"))
        else:
            test_files = sorted(
                key_output_dir.glob("test_predictions_fold_*.parquet"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            dfs = []
            for i, test_path in enumerate(test_files, start=1):
                df_fold = pl.read_parquet(test_path)
                df_fold = df_fold.with_columns(
                    [pl.lit(i).alias("fold"), pl.lit("test").alias("mode")]
                )
                dfs.append(df_fold)
            df = pl.concat(dfs, how="vertical", rechunk=True)
        out[mode] = df
    return out


def _infer_model_family(model_name: str) -> str:
    return "lmm" if "lmm" in model_name.lower() else "bhm"


def _pick_pred_col(mode: str, model_family: str) -> str:
    """Choose which prediction column name to use, since these are different
    between the two model types."""
    model_family = model_family.lower()
    if model_family == "bhm":
        name = "Predicted"
    elif "training" in mode.lower():
        name = "Predicted_RE"
    else:
        name = "Predicted_FE"
    return name


def build_model_performance_summary(
    dfs: dict[str, dict[str, pl.DataFrame]], true_col: str = "Observed"
) -> pl.DataFrame:
    """
    Build a performance summary table for a set of predictive models. For
    training, only the mean Pearson correlation is computed, while for test
    modes, fold-wise correlations, including min / max, are also calculated.
    """
    rows = []
    for model_name, modes in dfs.items():
        model_family = _infer_model_family(model_name)
        for mode_name, df in modes.items():
            pred_col = _pick_pred_col(mode_name, model_family)
            mean_r = float(df.select(pl.corr(true_col, pred_col)).item())
            row = {
                "Model": model_name,
                "Eval type": mode_name,
                "Mean": mean_r,
                "Min": float("nan"),
                "Max": float("nan"),
            }
            if "training" not in mode_name.lower() and "fold" in df.columns:
                per_fold_r = (
                    df.group_by("fold")
                    .agg(pl.corr(true_col, pred_col).alias("r"))
                    .sort("fold")
                )
                for fold_val, r_val in zip(
                    per_fold_r["fold"].to_list(), per_fold_r["r"].to_list()
                ):
                    row[f"r_fold_{fold_val}"] = float(r_val)
                stats = per_fold_r.select(
                    pl.col("r").min().alias("min"), pl.col("r").max().alias("max")
                ).row(0)
                row["Min"], row["Max"] = map(float, stats)
            rows.append(row)
    return pl.DataFrame(rows)


def model_comparison_barplot(
    summary_df: pl.DataFrame,
    figsize: tuple[int, int] = (6, 4),
    mode_color_scheme: dict[str, str] | None = None,
    show_bar_values: bool = True,
    show_axes_labels_values: bool = True,
    show_legend: bool = True,
    bar_number_size: int = 11,
    axes_label_size: int = 11,
    axes_number_size: int = 11,
) -> plt.Figure:
    df = summary_df.to_pandas()
    model_order = df["Model"].unique().tolist()
    mode_order = ["Training", "Standard CV", "Cross-study CV"]
    df_plot = df.copy()
    df_plot["Model"] = pd.Categorical(df_plot["Model"], categories=model_order)
    df_plot["Eval type"] = pd.Categorical(df_plot["Eval type"], categories=mode_order)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=df_plot,
        x="Model",
        y="Mean",
        hue="Eval type",
        order=model_order,
        hue_order=mode_order,
        palette=mode_color_scheme,
        errorbar=None,
        ax=ax,
        dodge=True,
        legend=show_legend,
    )
    for bar in ax.patches:
        bar.set_linewidth(0.8)
        bar.set_edgecolor("white")
    if show_bar_values:
        offset_mean = 0.06
        offset_range = 0.015
        for mode, container in zip(mode_order, ax.containers):
            mode_rows = df_plot[df_plot["Eval type"] == mode]
            for model, bar in zip(model_order, container.patches):
                row = mode_rows[mode_rows["Model"] == model]
                if row.empty:
                    continue
                mean_val = float(row["Mean"].iloc[0])
                min_val = float(row["Min"].iloc[0])
                max_val = float(row["Max"].iloc[0])
                x = bar.get_x() + bar.get_width() / 2
                if np.isfinite(min_val) and np.isfinite(max_val):
                    ax.text(
                        x,
                        mean_val + offset_range,
                        f"({min_val:.2f}–{max_val:.2f})",
                        ha="center",
                        va="bottom",
                        fontsize=bar_number_size,
                    )
                ax.text(
                    x,
                    mean_val + offset_mean,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=bar_number_size,
                    fontweight="bold",
                )
    ax.set_yticks([0, 1])
    if show_axes_labels_values:
        ax.set_xlabel("")
        ax.set_ylabel("Pearson's r", fontsize=axes_label_size)
        ax.set_yticklabels(["0", "1"], fontsize=axes_number_size)
        ax.tick_params(axis="x", labelsize=axes_label_size)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.tick_params(
            axis="both", which="both", labelbottom=False, labelleft=False, length=4
        )
    if show_legend:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.8, 1),
            frameon=False,
            fontsize=axes_label_size,
            title="",
        )
    fig.tight_layout()
    return fig


def read_site_info_parquet(
    run_folder: str,
    base_path: str = base_path,
    site_info_filename: str = site_info_filename,
) -> pl.DataFrame:
    """
    Helper function for loading the site_info.parquet file for a given run
    folder. Used in several functions in this notebook.
    """
    site_info_file = Path(base_path) / run_folder / site_info_filename
    df_site_info = pl.read_parquet(site_info_file)
    df_site_info = df_site_info.unique(subset=["SSBS"], keep="first")
    return df_site_info


def estimate_delta_predictions(
    state_pred_dfs: dict[str, dict[str, pl.DataFrame]],
    run_folders: dict[str, dict[str, str]],
    grouping_level: str = "SSB",
    reference_comparison_only: bool = False,
) -> dict[str, dict[str, pl.DataFrame]]:
    """
    Compute delta predictions (site_2 - site_1) within a grouping level (SS or
    SSB), for each (model, mode).

    If `Custom_taxonomic_group` exists, deltas are computed within matching
    taxonomic groups. Otherwise, site-level comparisons are used.
    """
    out = {}
    for model, modes in state_pred_dfs.items():
        model_family = _infer_model_family(model)
        out[model] = {}
        for mode, df_pred in modes.items():
            pred_col = _pick_pred_col(mode, model_family)
            has_taxa = "Custom_taxonomic_group" in df_pred.columns
            df_site_info = read_site_info_parquet(run_folders[model][mode])
            df_merged = df_pred.join(
                df_site_info.select(
                    ["SSBS", "SSB", "Predominant_land_use", "Use_intensity"]
                ),
                on="SSBS",
                how="left",
            )
            rows = []
            for df_group in df_merged.partition_by(grouping_level, as_dict=False):
                site_frames = {}
                site_taxa_frames = {}
                site_is_baseline = {}
                for df_site in df_group.partition_by("SSBS", as_dict=False):
                    site_id = df_site["SSBS"][0]
                    site_frames[site_id] = df_site
                    site_is_baseline[site_id] = bool(
                        (
                            (df_site["Predominant_land_use"] == "Primary vegetation")
                            & (df_site["Use_intensity"] == "Minimal use")
                        ).any()
                    )
                    if has_taxa:
                        site_taxa_frames[site_id] = {
                            tax: sub_df
                            for tax, sub_df in df_site.partition_by(
                                "Custom_taxonomic_group", as_dict=True
                            ).items()
                        }
                site_ids = list(site_frames.keys())
                if len(site_ids) < 2:
                    continue
                seen_pairs = set()
                for i in range(len(site_ids)):
                    for j in range(i + 1, len(site_ids)):
                        s1 = site_ids[i]
                        s2 = site_ids[j]
                        pair_key = tuple(sorted((s1, s2)))
                        if pair_key in seen_pairs:
                            continue
                        seen_pairs.add(pair_key)
                        if reference_comparison_only:
                            if not (
                                site_is_baseline[s1] and (not site_is_baseline[s2])
                            ):
                                continue
                        df_s1 = site_frames[s1]
                        df_s2 = site_frames[s2]
                        if has_taxa:
                            taxa1 = site_taxa_frames[s1]
                            taxa2 = site_taxa_frames[s2]
                            common_taxa = taxa1.keys() & taxa2.keys()
                            if not common_taxa:
                                continue
                            for taxon in common_taxa:
                                df1 = taxa1[taxon]
                                df2 = taxa2[taxon]
                                if df1.height != 1 or df2.height != 1:
                                    warnings.warn(
                                        f"Unexpected multiple rows for taxon-level comparison ({s1}, {s2}, taxon={taxon}); skipping."
                                    )
                                    continue
                                delta_obs = df2["Observed"][0] - df1["Observed"][0]
                                delta_pred = df2[pred_col][0] - df1[pred_col][0]
                                rows.append(
                                    {
                                        "site_1": s1,
                                        "site_2": s2,
                                        "Custom_taxonomic_group": taxon,
                                        "Observed": delta_obs,
                                        pred_col: delta_pred,
                                        "residual": delta_pred - delta_obs,
                                        "mode": df1["mode"][0],
                                        "lu_1": df1["Predominant_land_use"][0],
                                        "ui_1": df1["Use_intensity"][0],
                                        "lu_2": df2["Predominant_land_use"][0],
                                        "ui_2": df2["Use_intensity"][0],
                                    }
                                )
                        else:
                            if df_s1.height != 1 or df_s2.height != 1:
                                warnings.warn(
                                    f"Unexpected multiple rows for site-level comparison ({s1}, {s2}); skipping."
                                )
                                continue
                            delta_obs = df_s2["Observed"][0] - df_s1["Observed"][0]
                            delta_pred = df_s2[pred_col][0] - df_s1[pred_col][0]
                            rows.append(
                                {
                                    "site_1": s1,
                                    "site_2": s2,
                                    "Observed": delta_obs,
                                    pred_col: delta_pred,
                                    "residual": delta_pred - delta_obs,
                                    "mode": df_s1["mode"][0],
                                    "lu_1": df_s1["Predominant_land_use"][0],
                                    "ui_1": df_s1["Use_intensity"][0],
                                    "lu_2": df_s2["Predominant_land_use"][0],
                                    "ui_2": df_s2["Use_intensity"][0],
                                }
                            )
            out[model][mode] = pl.DataFrame(rows)
    return out


def basic_scatter_plot(
    df: pl.DataFrame,
    x_col: str,
    y_col: str,
    figsize: tuple[int, int],
    show_axes_labels_values: bool = True,
    show_metrics: bool = True,
    show_best_fit_line: bool = True,
    show_diagonal_ref_line: bool = True,
    show_zero_ref_line: bool = False,
    color: str = "steelblue",
    alpha: float = 0.4,
    point_size: int = 5,
    sample_frac: float = 1.0,
    poly_degree: int = 3,
    x_label: str = "Observed values",
    y_label: str = "Model predictions",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    axes_label_size: int = 11,
    axes_number_size: int = 11,
) -> plt.Figure:
    """
    Create a simple scatterplot comparing two arrays of variables, with
    optional sampling, polynomial best-fit line, and performance metrics.
    """
    x_array = df.get_column(x_col).to_numpy()
    y_array = df.get_column(y_col).to_numpy()
    mask = np.isfinite(x_array) & np.isfinite(y_array)
    x_valid = x_array[mask]
    y_valid = y_array[mask]
    pearson_r, _ = pearsonr(x_valid, y_valid)
    mae = float(np.mean(np.abs(x_valid - y_valid)))
    if sample_frac is not None and 0 < sample_frac < 1.0:
        n = len(x_valid)
        k = max(1, int(n * sample_frac))
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n, size=k, replace=False)
        x_plot = x_valid[idx]
        y_plot = y_valid[idx]
    else:
        x_plot, y_plot = (x_valid, y_valid)
    fig, ax = plt.subplots(figsize=figsize)
    sns.scatterplot(x=x_plot, y=y_plot, color=color, alpha=alpha, s=point_size, ax=ax)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if show_diagonal_ref_line:
        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            transform=ax.transAxes,
            color="black",
            linewidth=1.2,
            zorder=5,
        )
    if show_zero_ref_line:
        ax.axhline(y=0, linestyle=":", color="black", linewidth=1.2, zorder=4)
    if show_best_fit_line and len(x_plot) >= poly_degree + 1:
        coefs = np.polyfit(x_plot, y_plot, deg=poly_degree)
        x_vals = np.linspace(*sorted(ax.get_xlim()), 200)
        y_vals = np.polyval(coefs, x_vals)
        ax.plot(x_vals, y_vals, linestyle="-", color="firebrick", linewidth=2, zorder=6)
    if show_metrics:
        ax.text(
            0.03,
            0.97,
            f"r = {pearson_r:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=axes_label_size,
        )
    if show_axes_labels_values:
        ax.set_xlabel(x_label, fontsize=axes_label_size)
        ax.set_ylabel(y_label, fontsize=axes_label_size)
        ax.tick_params(axis="both", labelsize=axes_number_size)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
    fig.tight_layout()
    return fig


def load_glmm_phi(
    run_folder: str, base_path: str = base_path, filename: str = "train_phi.json"
) -> float:
    phi_path = Path(base_path) / run_folder / "key_output" / filename
    with open(phi_path) as f:
        out = json.load(f)
    return float(out["phi"])


def calculate_r2_var_explained(
    df: pl.DataFrame,
    distribution: str = "gaussian",
    phi: float | None = None,
    eps: float = 1e-06,
) -> tuple[float, float]:
    """
    Compute marginal/conditional R2 for Gaussian and Beta mixed models.
    Uses latent-scale decomposition for Beta.
    """
    y = df.get_column("Observed").to_numpy()
    y_fe = df.get_column("Predicted_FE").to_numpy()
    y_re = df.get_column("Predicted_RE").to_numpy()
    if distribution == "gaussian":
        var_fe = np.var(y_fe, ddof=1)
        var_re = np.var(y_re - y_fe, ddof=1)
        var_residual = np.var(y - y_re, ddof=1)
    elif distribution == "beta":
        mu_fe = np.clip(y_fe, eps, 1 - eps)
        mu_re = np.clip(y_re, eps, 1 - eps)
        eta_fe = logit(mu_fe)
        eta_re = logit(mu_re)
        var_fe = np.var(eta_fe, ddof=1)
        var_re = np.var(eta_re - eta_fe, ddof=1)
        var_residual = np.mean(1.0 / ((1.0 + phi) * mu_re * (1.0 - mu_re)))
    denominator = var_fe + var_re + var_residual
    var_expl_fe = var_fe / denominator
    var_expl_re = (var_fe + var_re) / denominator
    return (var_expl_fe, var_expl_re)


def compute_r2_by_run(
    run_folders: dict[str, str],
    distribution: str = "beta",
    base_path: str = base_path,
    key_output_dirname: str = key_output_path,
    predictions_filename: str = "train_predictions.parquet",
) -> dict[str, tuple[float, float]]:
    results = {}
    base = Path(base_path)
    for run_name, folder in run_folders.items():
        df = pl.read_parquet(base / folder / key_output_dirname / predictions_filename)
        phi = (
            load_glmm_phi(folder, base_path=base_path)
            if distribution == "beta"
            else None
        )
        results[run_name] = calculate_r2_var_explained(
            df, distribution=distribution, phi=phi
        )
    return results


def r2_stacked_barplot(
    r2_dictionary: dict[str, tuple[float, float]],
    figsize: tuple[int, int] = (6, 4),
    show_axes_labels: bool = True,
    show_bar_values: bool = True,
    show_legend: bool = True,
    colors: tuple[str, str] = ("steelblue", "yellowgreen"),
    alpha: float = 0.8,
    bar_number_size: int = 11,
    axes_label_size: int = 11,
    axes_number_size: int = 11,
    wrap_labels_at: int | None = 12,
) -> plt.Figure:
    """
    Plot marginal and conditional R² as stacked bars. Marginal R² is the
    fixed-effects portion; the additional bar height shows variance explained
    by random effects.
    """
    models = list(r2_dictionary.keys())
    marginal_r2 = [r2_dictionary[m][0] for m in models]
    conditional_r2 = [r2_dictionary[m][1] for m in models]
    random_effect_r2 = [c - m for m, c in zip(marginal_r2, conditional_r2)]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(
        x,
        height=marginal_r2,
        width=0.5,
        label="Marginal R²",
        color=colors[0],
        alpha=alpha,
    )
    ax.bar(
        x,
        height=random_effect_r2,
        width=0.5,
        bottom=marginal_r2,
        label="Additional R² from random effects",
        color=colors[1],
        alpha=alpha,
    )
    if show_bar_values:
        for i in range(len(models)):
            ax.text(
                x[i],
                marginal_r2[i] / 2,
                f"{marginal_r2[i]:.2f}",
                ha="center",
                va="center",
                fontsize=bar_number_size,
            )
            ax.text(
                x[i],
                marginal_r2[i] + random_effect_r2[i] / 2,
                f"{random_effect_r2[i]:.2f}",
                ha="center",
                va="center",
                fontsize=bar_number_size,
            )
            ax.text(
                x[i],
                conditional_r2[i] + 0.015,
                f"{conditional_r2[i]:.2f}",
                ha="center",
                va="bottom",
                fontsize=bar_number_size,
            )
    if wrap_labels_at is not None:
        xticklabels = [
            textwrap.fill(label, width=wrap_labels_at, break_long_words=False)
            for label in models
        ]
    else:
        xticklabels = models
    ax.set_yticks([0, 1])
    if show_axes_labels:
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=axes_label_size)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 1.0)
        ax.set_yticklabels(["0", "1"], fontsize=axes_number_size)
        ax.tick_params(axis="y", length=5)
        ax.set_ylabel("Var. explained (R²)", fontsize=axes_label_size)
        ax.yaxis.set_label_coords(-0.07, 0.5)
    else:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False)
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        cond_patch = Patch(
            facecolor="white",
            edgecolor="grey",
            label="= Conditional R² (total)",
            linewidth=1,
        )
        handles.append(cond_patch)
        labels.append("= Conditional R² (total)")
        ax.legend(
            handles=handles,
            labels=labels,
            fontsize=axes_label_size,
            edgecolor="none",
            loc="upper left",
            bbox_to_anchor=(0, 1),
            frameon=False,
        )
    fig.tight_layout()
    return fig


def load_glmm_effects(
    run_folder: str, base_path: str = base_path, filename: str = "train_effects.json"
) -> dict[str, dict[str, float]]:
    effects_path = Path(base_path) / run_folder / "key_output" / filename
    with open(effects_path) as f:
        return json.load(f)


def extract_glmm_effects(
    model_path: str, re_lower_perc: float = 5, re_upper_perc: float = 95
) -> dict[str, dict[str, float]]:
    """
    Load a glmmTMB model (.rds) and extract fixed effects plus optional
    study-level random slope variation summaries.

    Effects are returned on the response scale as delta(mu) from baseline.
    """
    model_path = str(model_path)
    r_code = '\n      args <- commandArgs(trailingOnly = TRUE)\n      model_path <- args[[1]]\n      out_path <- args[[2]]\n      re_lower <- as.numeric(args[[3]])\n      re_upper <- as.numeric(args[[4]])\n\n      suppressPackageStartupMessages({\n        library(glmmTMB)\n        library(jsonlite)\n      })\n\n      model <- readRDS(model_path)\n      coef_tab <- summary(model)$coefficients$cond\n      fixef_cond <- fixef(model)$cond\n\n      intercept <- if ("(Intercept)" %in% names(fixef_cond)) {\n        as.numeric(fixef_cond[["(Intercept)"]])\n      } else {\n        0\n      }\n\n      linkinv_func <- model$family$linkinv\n      if (!is.function(linkinv_func)) {\n        # Fallback for beta/logit models\n        linkinv_func <- make.link("logit")$linkinv\n      }\n\n      to_response_delta <- function(delta_eta) {\n        as.numeric(linkinv_func(intercept + delta_eta) - linkinv_func(intercept))\n      }\n\n      # Try model-based CIs first; fall back to Wald CIs.\n      ci_mat <- tryCatch(\n        suppressMessages(confint(model, parm = "beta_", level = 0.95)),\n        error = function(e) NULL\n      )\n\n      get_ci <- function(term) {\n        est <- as.numeric(coef_tab[term, "Estimate"])\n        se <- as.numeric(coef_tab[term, "Std. Error"])\n        ci_low <- est - 1.96 * se\n        ci_up <- est + 1.96 * se\n\n        if (!is.null(ci_mat)) {\n          rn <- rownames(ci_mat)\n          idx <- which(rn %in% c(paste0("cond.", term), term))\n          if (length(idx) > 0) {\n            ci_low <- as.numeric(ci_mat[idx[1], 1])\n            ci_up <- as.numeric(ci_mat[idx[1], 2])\n          }\n        }\n        c(ci_low, ci_up)\n      }\n\n      effect_dict <- list()\n\n      re_cond <- tryCatch(ranef(model)$cond, error = function(e) NULL)\n      re_study <- NULL\n      if (!is.null(re_cond) && "SS" %in% names(re_cond)) {\n        re_study <- as.data.frame(re_cond$SS)\n      }\n\n      for (term in names(fixef_cond)) {\n        if (term == "(Intercept)") next\n\n        est_eta <- as.numeric(fixef_cond[[term]])\n        ci_eta <- get_ci(term)\n\n        ci_resp <- sort(c(\n          to_response_delta(ci_eta[1]),\n          to_response_delta(ci_eta[2])\n        ))\n\n        info <- list(\n          mean = to_response_delta(est_eta),\n          ci_lower_2_5 = as.numeric(ci_resp[1]),\n          ci_upper_97_5 = as.numeric(ci_resp[2])\n        )\n\n        if (!is.null(re_study) && term %in% colnames(re_study)) {\n          deviations <- re_study[[term]]\n          deviations <- deviations[!is.na(deviations)]\n          if (length(deviations) > 1) {\n            lower_eta <- est_eta + as.numeric(quantile(deviations, probs = re_lower / 100))\n            upper_eta <- est_eta + as.numeric(quantile(deviations, probs = re_upper / 100))\n            rs_resp <- sort(c(\n              to_response_delta(lower_eta),\n              to_response_delta(upper_eta)\n            ))\n            info$random_slope_lower <- as.numeric(rs_resp[1])\n            info$random_slope_upper <- as.numeric(rs_resp[2])\n          }\n        }\n\n        effect_dict[[term]] <- info\n      }\n\n      write_json(effect_dict, out_path, auto_unbox = TRUE)\n      '
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        r_script_path = tmp_dir / "extract_glmmtmb_effects.R"
        out_json_path = tmp_dir / "effects.json"
        r_script_path.write_text(r_code)
        subprocess.run(
            [
                "Rscript",
                str(r_script_path),
                model_path,
                str(out_json_path),
                str(re_lower_perc),
                str(re_upper_perc),
            ],
            check=True,
        )
        effect_dict = json.loads(out_json_path.read_text())
    return effect_dict


def lmm_effects_forest_plot(
    effect_dict: dict[str, dict[str, float]],
    re_percentiles: tuple[float, float] = (5, 95),
    figsize: tuple[int, int] = (6, 4),
    show_mean_values: bool = True,
    show_axes_labels_values: bool = True,
    show_legend: bool = True,
    colors: tuple[str, str] = ("steelblue", "yellowgreen"),
    alpha: float = 0.8,
    mean_label_size: int = 11,
    axes_label_size: int = 11,
    axes_number_size: int = 11,
    ci_method: str = "Wald",
) -> plt.Figure:
    """
    Forest plot of LMM fixed effects with 95% CI, optionally showing random
    slope percentile ranges if available.
    """
    param_names = list(effect_dict.keys())
    y_pos = np.arange(len(param_names))
    means = [effect_dict[p]["mean"] for p in param_names]
    ci_lows = [effect_dict[p]["ci_lower_2_5"] for p in param_names]
    ci_highs = [effect_dict[p]["ci_upper_97_5"] for p in param_names]
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    for i, p in enumerate(param_names):
        eff = effect_dict[p]
        if "random_slope_lower" in eff and "random_slope_upper" in eff:
            ax.hlines(
                y=y_pos[i],
                xmin=eff["random_slope_lower"],
                xmax=eff["random_slope_upper"],
                color=colors[1],
                linewidth=3,
                alpha=alpha,
                label=(
                    f"Study slope range ({re_percentiles[0]}/{re_percentiles[1]})"
                    if i == 0
                    else ""
                ),
            )
    ax.hlines(
        y=y_pos,
        xmin=ci_lows,
        xmax=ci_highs,
        color=colors[0],
        linewidth=5,
        alpha=alpha,
        label=f"95% CI ({ci_method})",
    )
    ax.plot(
        means,
        y_pos,
        "o",
        markersize=8,
        color=colors[0],
        alpha=alpha,
        label="Fixed effect mean",
    )
    if show_mean_values:
        for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs)):
            star = "*" if lo > 0 or hi < 0 else ""
            ax.text(
                m,
                y_pos[i] - 0.1,
                f"{m:.2f}{star}",
                ha="center",
                va="bottom",
                fontsize=mean_label_size,
            )
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    if show_axes_labels_values:
        ax.set_yticklabels(param_names, fontsize=axes_label_size)
        ax.tick_params(axis="x", labelsize=axes_number_size)
    else:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        unique = []
        for h, l in zip(handles, labels):
            if l not in seen and l != "":
                unique.append((h, l))
                seen.add(l)
        if unique:
            ax.legend(
                [h for h, _ in unique],
                [l for _, l in unique],
                fontsize=axes_number_size,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False,
            )
    fig.tight_layout()
    return fig


def load_all_test_fold_dfs(run_folder: str, n_folds: int = 5) -> pl.DataFrame:
    """Load all test dataframes, in order to get covariate information."""
    dfs = []
    for i in range(1, n_folds + 1):
        file_path = Path(base_path) / run_folder / f"test_fold_{i}.parquet"
        df = pl.read_parquet(file_path)
        dfs.append(df)
    return pl.concat(dfs, how="vertical")


def gower_knn_distances(
    query_x: np.ndarray,
    reference_x: np.ndarray,
    k: int = 5,
    max_reference_points: int | None = None,
    seed: int = 42,
    same_array: bool = False,
    selection: str = "nearest",
    metric: str = "median",
) -> np.ndarray:
    """
    Compute per-row Gower distance summaries to either:
    - k nearest reference points, or
    - k random reference points.
    """
    rng = np.random.default_rng(seed=seed)
    if max_reference_points is not None and len(reference_x) > max_reference_points:
        subset_idx = rng.choice(
            len(reference_x), size=max_reference_points, replace=False
        )
        ref = reference_x[subset_idx]
    else:
        ref = reference_x
    dist_matrix = gower.gower_matrix(query_x, ref)
    is_square = dist_matrix.shape[0] == dist_matrix.shape[1]
    if same_array and is_square:
        np.fill_diagonal(dist_matrix, np.nan if selection == "random" else np.inf)
    n_ref = dist_matrix.shape[1]
    k_eff = min(k, n_ref - 1 if same_array and is_square else n_ref)
    if selection == "nearest":
        selected = np.sort(dist_matrix, axis=1)[:, :k_eff]
    elif selection == "random":
        idx = rng.choice(n_ref, size=k_eff, replace=False)
        selected = dist_matrix[:, idx]
    if metric == "median":
        aggregate = np.nanmedian(selected, axis=1)
    elif metric == "mean":
        aggregate = np.nanmean(selected, axis=1)
    return aggregate


def compute_covariate_gower_per_group(
    df: pl.DataFrame,
    group_col: str,
    covariate_cols: list[str],
    k_neighbors: int = 10,
    selection: str = "random",
    metric: str = "median",
) -> pl.DataFrame:
    """Compute Gower homogeneity metrics (mean/median/std) for each group,
    based on the specified covariate columns.
    """
    groups = df[group_col].unique().to_list()
    records = []
    for g in groups:
        df_group = df.filter(pl.col(group_col) == g)
        x_group = df_group.select(covariate_cols).to_numpy()
        knn_vals = gower_knn_distances(
            query_x=x_group,
            reference_x=x_group,
            k=k_neighbors,
            same_array=True,
            max_reference_points=None,
            selection=selection,
            metric=metric,
        )
        records.append(
            {
                group_col: g,
                "gower_mean": float(np.mean(knn_vals)),
                "gower_median": float(np.median(knn_vals)),
                "gower_std": float(np.std(knn_vals)),
            }
        )
    return pl.DataFrame(records)


def compute_outlier_proportions_per_group(
    df_joined: pl.DataFrame, group_col: str, true_col: str
) -> pl.DataFrame:
    """Compute proportion of outliers per group using IQR method."""
    df_iqr = df_joined.group_by(group_col).agg(
        [
            pl.col(true_col).quantile(0.25).alias("q1"),
            pl.col(true_col).quantile(0.75).alias("q3"),
        ]
    )
    df_iqr = df_iqr.with_columns((pl.col("q3") - pl.col("q1")).alias("iqr"))
    df_iqr = df_iqr.with_columns(
        [
            (pl.col("q1") - pl.col("iqr")).alias("lower_1_threshold"),
            (pl.col("q3") + pl.col("iqr")).alias("upper_1_threshold"),
            (pl.col("q1") - 1.5 * pl.col("iqr")).alias("lower_1_5_threshold"),
            (pl.col("q3") + 1.5 * pl.col("iqr")).alias("upper_1_5_threshold"),
        ]
    )
    df_with_thresholds = df_joined.join(df_iqr, on=group_col, how="left")
    df_with_flags = df_with_thresholds.with_columns(
        [
            (
                (pl.col(true_col) < pl.col("lower_1_5_threshold"))
                | (pl.col(true_col) > pl.col("upper_1_5_threshold"))
            ).alias("is_outlier_1_5"),
            (
                (pl.col(true_col) < pl.col("lower_1_threshold"))
                | (pl.col(true_col) > pl.col("upper_1_threshold"))
            ).alias("is_outlier_1"),
        ]
    )
    df_outliers = df_with_flags.group_by(group_col).agg(
        [
            pl.col("is_outlier_1_5").mean().alias("prop_outliers_1_5"),
            pl.col("is_outlier_1_5").sum().alias("n_outliers_1_5"),
            pl.col("is_outlier_1").mean().alias("prop_outliers_1"),
            pl.col("is_outlier_1").sum().alias("n_outliers_1"),
        ]
    )
    return df_outliers


def create_hierarchical_groups_in_predicts_data(
    df_predicts: pl.DataFrame,
) -> pl.DataFrame:
    """
    Add species group to the original PREDICTS data and combine this with
    biomes, to recreate the groupings used in the model. This is required for
    the next step.
    """
    df_predicts = df_predicts.with_columns(
        [
            pl.when(pl.col("Phylum") == "Arthropoda")
            .then(
                pl.when(pl.col("Class") == "Insecta")
                .then(pl.lit("Insecta"))
                .otherwise(pl.lit("Other Arthropoda"))
            )
            .when(pl.col("Phylum") == "Chordata")
            .then(
                pl.when(pl.col("Class").is_in(["Aves", "Mammalia"]))
                .then(pl.col("Class"))
                .when(pl.col("Class").is_in(["Amphibia", "Reptilia"]))
                .then(pl.lit("Amphibia_Reptilia"))
                .otherwise(pl.lit("Other Chordata"))
            )
            .when(pl.col("Phylum") == "Tracheophyta")
            .then(pl.lit("Tracheophyta"))
            .when(pl.col("Kingdom") == "Fungi")
            .then(pl.lit("Fungi"))
            .otherwise(pl.lit("Other ") + pl.col("Kingdom"))
            .alias("Species group")
        ]
    )
    return df_predicts


def compute_n_taxa_per_group(df_joined: pl.DataFrame, group_col: str) -> pl.DataFrame:
    """Compute number of unique taxa per group at the specified taxonomic rank."""
    df_predicts = pl.read_parquet("../../data/PREDICTS/merged_data.parquet")
    df_predicts = create_hierarchical_groups_in_predicts_data(df_predicts)
    df_predicts = df_predicts.join(
        df_joined.select(["SSBS", group_col]), on="SSBS", how="left"
    )
    df_taxon_counts = df_predicts.group_by(group_col).agg(
        [
            pl.col("Order").n_unique().alias("n_orders"),
            pl.col("Family").n_unique().alias("n_families"),
            pl.col("Genus").n_unique().alias("n_genera"),
            pl.col("Species").n_unique().alias("n_species"),
        ]
    )
    return df_taxon_counts


def compute_deepdive_summary_per_group(
    df_pred: pl.DataFrame,
    df_covars: pl.DataFrame,
    run_folder: str,
    group_col: str,
    pred_col: str,
    covariate_cols: list[str],
    true_col: str = "Observed",
    k_neighbors: int = 10,
    gower_selection: str = "nearest",
    gower_metric: str = "median",
) -> pl.DataFrame:
    """
    Join predictions with site info (on SSBS), then per stratum compute various
    metrics for analysis.
    """
    df_site_info = read_site_info_parquet(run_folder)
    df_joined = df_pred.join(df_site_info, on="SSBS", how="left")
    if group_col == "Final_hierarchical_group":
        df_joined = df_joined.filter(pl.col("Final_hierarchical_level") != "Population")
    df_stats = df_joined.group_by(group_col).agg(
        [
            pl.col("SSBS").n_unique().alias("n_sites"),
            pl.col("SSBS").len().alias("n_observations"),
            pl.col("SS").n_unique().alias("n_studies"),
            pl.col(true_col).std().alias("response_std"),
            pl.corr(true_col, pred_col).alias("Pearson_r"),
        ]
    )
    df_stats = df_stats.with_columns(
        [
            pl.col("n_sites").log().alias("n_sites_log"),
            pl.col("n_observations").log().alias("n_observations_log"),
            pl.col("n_studies").log().alias("n_studies_log"),
        ]
    )
    df_gower = compute_covariate_gower_per_group(
        df=df_covars,
        group_col=group_col,
        covariate_cols=covariate_cols,
        k_neighbors=k_neighbors,
        selection=gower_selection,
        metric=gower_metric,
    )
    df_outliers = compute_outlier_proportions_per_group(
        df_joined, group_col=group_col, true_col=true_col
    )
    df_taxa = compute_n_taxa_per_group(df_joined, group_col=group_col)
    df_out = (
        df_stats.join(df_outliers, on=group_col, how="left")
        .join(df_gower, on=group_col, how="left")
        .join(df_taxa, on=group_col, how="left")
        .sort(group_col)
    )
    return df_out


def plot_histogram(
    df: pl.DataFrame,
    data_col: str,
    bins: int = 50,
    alpha: float = 0.7,
    color: str = "steelblue",
    figsize: tuple = (5, 4),
    axes_numbers_size: int = 12,
    axes_label_size: int = 12,
    show_axes_labels: bool = False,
    show_axes_numbers: bool = True,
) -> plt.Figure:
    data_array = df.select(pl.col(data_col)).to_series().to_numpy()
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data_array, bins=bins, kde=False, ax=ax, color=color, alpha=alpha)
    if not show_axes_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")
    else:
        ax.set_xlabel(data_col, fontsize=axes_label_size)
        ax.set_ylabel("Frequency", fontsize=axes_label_size)
    if show_axes_numbers:
        ax.tick_params(axis="x", labelsize=axes_numbers_size)
        ax.tick_params(axis="y", labelsize=axes_numbers_size)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    plt.tight_layout()
    return fig


def ols_forest_plot(
    model,
    var_names,
    include_intercept: bool = False,
    figsize: tuple[int, int] = (6, 4),
    show_mean_values: bool = True,
    show_axes_labels_values: bool = True,
    show_legend: bool = True,
    color: str = "steelblue",
    alpha: float = 0.8,
    mean_label_size: int = 11,
    axes_label_size: int = 11,
    axes_number_size: int = 11,
    ci_method: str = "Wald",
    ci_alpha: float = 0.05,
) -> plt.Figure:
    """
    Forest plot of standardized OLS effects with (1-ci_alpha)% CI.
    Assumes model was fit with sm.add_constant(Xz) and var_names matches Xz
    column order.
    """
    names = ["Intercept"] + list(var_names)
    params = np.asarray(model.params)
    ci = np.asarray(model.conf_int(alpha=ci_alpha))
    rows = []
    for name, est, (lo, hi) in zip(names, params, ci):
        if name == "Intercept" and (not include_intercept):
            continue
        rows.append((name, float(est), float(lo), float(hi)))
    rows.sort(key=lambda r: r[1], reverse=False)
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    ci_lows = np.array([r[2] for r in rows])
    ci_highs = np.array([r[3] for r in rows])
    y_pos = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0)
    ax.hlines(
        y=y_pos,
        xmin=ci_lows,
        xmax=ci_highs,
        color=color,
        linewidth=5,
        alpha=alpha,
        label=f"{int(round((1 - ci_alpha) * 100))}% CI ({ci_method})",
    )
    ax.plot(
        means,
        y_pos,
        "o",
        markersize=8,
        color=color,
        alpha=alpha,
        label="Effect estimate",
    )
    if show_mean_values:
        for i, (m, lo, hi) in enumerate(zip(means, ci_lows, ci_highs)):
            star = "*" if lo > 0 or hi < 0 else ""
            ax.text(
                m,
                y_pos[i] - 0.1,
                f"{m:.2f}{star}",
                ha="center",
                va="bottom",
                fontsize=mean_label_size,
            )
    ax.invert_yaxis()
    ax.set_yticks(y_pos)
    if show_axes_labels_values:
        ax.set_yticklabels(labels, fontsize=axes_label_size)
        ax.tick_params(axis="x", labelsize=axes_number_size)
    else:
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
    if show_legend:
        handles, lab = ax.get_legend_handles_labels()
        seen, uniq = (set(), [])
        for h, l in zip(handles, lab):
            if l and l not in seen:
                uniq.append((h, l))
                seen.add(l)
        if uniq:
            ax.legend(
                [h for h, _ in uniq],
                [l for _, l in uniq],
                fontsize=axes_number_size,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=False,
            )
    fig.tight_layout()
    return fig


def get_train_test_data_per_fold(
    run_folder: str, base_path: str = base_path, key_output_dirname: str = "key_output"
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Load CV fold prediction files and extract observed values per fold.

    Expects files:
      <run_folder>/<key_output_dirname>/train_predictions_fold_*.parquet
      <run_folder>/<key_output_dirname>/test_predictions_fold_*.parquet
    """
    key_output_dir = Path(base_path) / run_folder / key_output_dirname
    train_files = sorted(
        key_output_dir.glob("train_predictions_fold_*.parquet"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    test_files = sorted(
        key_output_dir.glob("test_predictions_fold_*.parquet"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    train_y_list: list[np.ndarray] = []
    test_y_list: list[np.ndarray] = []
    for train_path, test_path in zip(train_files, test_files):
        df_train = pl.read_parquet(train_path)
        df_test = pl.read_parquet(test_path)
        train_y_list.append(df_train.get_column("Observed").to_numpy())
        test_y_list.append(df_test.get_column("Observed").to_numpy())
    return (train_y_list, test_y_list)


def foldwise_kde_subplots(
    train_data_list: list[np.ndarray],
    test_data_list: list[np.ndarray],
    test_r_list: list[float],
    figsize: tuple[int, int] = (4, 3),
    show_axes_labels_values: bool = True,
    show_fold_labels: bool = True,
    show_r_values: bool = False,
    show_legend: bool = True,
    xlabel: str = "",
    colors: dict[str, str] = {"train": "steelblue", "test": "darkorange"},
    alpha: float = 0.8,
    r_label_size: int = 11,
    axes_label_size: int = 11,
    axes_number_size: int = 11,
    xlim: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot fold-wise KDE curves for train and test data distributions using
    stacked subplots.
    """
    num_folds = len(train_data_list)
    fig, axs = plt.subplots(
        nrows=num_folds,
        ncols=1,
        figsize=(figsize[0], figsize[1] * num_folds),
        sharex=True,
        gridspec_kw={"hspace": 0.1},
        constrained_layout=True,
    )
    all_values = np.concatenate(train_data_list + test_data_list)
    vmin, vmax = (all_values.min(), all_values.max())
    x_pad = 0.05 * (vmax - vmin)
    x_range = (vmin - x_pad, vmax + x_pad)
    for i in range(num_folds):
        ax = axs[i]
        sns.kdeplot(
            train_data_list[i],
            ax=ax,
            color=colors["train"],
            fill=False,
            alpha=alpha,
            linewidth=2,
            bw_adjust=1.2,
        )
        sns.kdeplot(
            test_data_list[i],
            ax=ax,
            color=colors["test"],
            fill=False,
            alpha=alpha,
            linewidth=2,
            bw_adjust=1.2,
        )
        if xlim is not None:
            ax.set_xlim(*xlim)
        else:
            ax.set_xlim(*x_range)
        if show_fold_labels:
            ax.set_ylabel(
                f"Fold {i + 1}", fontsize=axes_label_size, rotation=0, labelpad=30
            )
        else:
            ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_yticklabels([])
        if test_r_list is not None and show_r_values:
            ax.text(
                0.95,
                0.5,
                f"r = {test_r_list[i]:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=r_label_size,
            )
        if i == num_folds - 1 and show_axes_labels_values:
            ax.set_xlabel(xlabel, fontsize=axes_label_size)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis="x", labelbottom=False)
        ax.tick_params(axis="x", labelsize=axes_number_size)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
    if show_legend:
        legend_handles = [
            Line2D([0], [0], color=colors["train"], lw=2, label="Train"),
            Line2D([0], [0], color=colors["test"], lw=2, label="Test"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=axes_label_size,
            frameon=False,
            bbox_to_anchor=(1.12, 1.05),
        )
    return fig


def compute_gower_distances_train_test(
    run_folder: str,
    covariate_cols: list[str],
    base_path: str = base_path,
    k_neighbors: int = 5,
    max_train_points: int = 1000,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute fold-wise train/test kNN Gower distances from saved CV folds."""
    run_dir = Path(base_path) / run_folder
    train_files = sorted(
        run_dir.glob("train_fold_*.parquet"), key=lambda p: int(p.stem.split("_")[-1])
    )
    test_files = sorted(
        run_dir.glob("test_fold_*.parquet"), key=lambda p: int(p.stem.split("_")[-1])
    )
    train_dist_list = []
    test_dist_list = []
    for train_path, test_path in zip(train_files, test_files):
        df_train = pl.read_parquet(train_path)
        df_test = pl.read_parquet(test_path)
        train_x = df_train.select(covariate_cols).to_numpy()
        test_x = df_test.select(covariate_cols).to_numpy()
        train_dist_list.append(
            gower_knn_distances(
                query_x=train_x,
                reference_x=train_x,
                k=k_neighbors,
                max_reference_points=max_train_points,
                same_array=True,
                selection="nearest",
                metric="median",
            )
        )
        test_dist_list.append(
            gower_knn_distances(
                query_x=test_x,
                reference_x=train_x,
                k=k_neighbors,
                max_reference_points=max_train_points,
                same_array=False,
                selection="nearest",
                metric="median",
            )
        )
    return (train_dist_list, test_dist_list)


def build_country_lookup_table() -> pd.DataFrame:
    """
    Load Natural Earth country table with ADMIN and ISO_A3 information. Returns
    a pandas DataFrame with columns ['country_std', 'iso_a3'] that can be
    joined with model output data.
    """
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)[["ADMIN", "ISO_A3"]]
    world = world.rename(columns={"ADMIN": "country_std", "ISO_A3": "iso_a3"})
    return world


def add_iso_codes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add Natural Earth `iso_a3` and `country_std` codes to a DataFrame with
    model output data.

    Policy:
    - Use Natural Earth country definitions for mapping
    - Harmonize common country-name mismatches
    - Explicitly patch known Natural Earth ISO gaps (-99 or null)
    - Non-sovereign territories are either mapped to parent states or left
      missing by design (see overrides below)

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with added `country_std` and `iso_a3` columns.
        Rows with unresolved ISO codes will not appear in maps unless
        handled downstream.
    """
    df_pd = df.to_pandas()
    name_overrides = {
        "United States": "United States of America",
        "Czech Republic": "Czechia",
        "Korea, Republic of": "South Korea",
        "United Republic of Tanzania": "Tanzania",
        "Cote d'Ivoire": "Côte d'Ivoire",
        "Sao Tome and Principe": "São Tomé and Príncipe",
        "Viet Nam": "Vietnam",
        "Lao People's Democratic Republic": "Laos",
        "Iran (Islamic Republic of)": "Iran",
        "Syrian Arab Republic": "Syria",
        "Côte d'Ivoire": "Ivory Coast",
        "Serbia": "Republic of Serbia",
        "Comoros": "Union of the Comoros",
    }
    df_pd["Country"] = df_pd["Country"].replace(name_overrides)
    country_lookup = build_country_lookup_table()
    df_out = df_pd.merge(
        country_lookup, left_on="Country", right_on="country_std", how="left"
    )
    iso_overrides = {
        "France": "FRA",
        "Norway": "NOR",
        "Hong Kong": "CHN",
        "French Guiana": "FRA",
        "Puerto Rico": "USA",
        "São Tomé and Príncipe": "STP",
        "Tanzania": "TZA",
        "Côte d'Ivoire": "CIV",
        "Union of the Comoros": "COM",
    }
    df_out["iso_a3"] = df_out["iso_a3"].where(
        df_out["iso_a3"].notna() & (df_out["iso_a3"] != "-99"),
        df_out["Country"].map(iso_overrides),
    )
    return pl.DataFrame(df_out)


def calculate_country_stats(
    df_pred: pl.DataFrame,
    run_folder: str,
    pred_col: str,
    true_col: str = "Observed",
    clip: tuple[float, float] | None = None,
) -> pl.DataFrame:
    """
    Return one per-country dataframe containing:
      - number of unique studies
      - number of unique sites
      - Pearson's r between predictions and observations
      - ISO country codes (for plotting)
    """
    df_site_info = read_site_info_parquet(run_folder).select(["SS", "SSBS", "Country"])
    df_merged = df_pred.join(df_site_info, on="SSBS", how="inner")
    df_counts = df_merged.group_by("Country").agg(
        [
            pl.col("SS").n_unique().alias("Nb_of_studies").cast(pl.Int32),
            pl.col("SSBS").n_unique().alias("Nb_of_sites").cast(pl.Int32),
        ]
    )
    corr_expr = pl.corr(true_col, pred_col)
    if clip is not None:
        corr_expr = corr_expr.clip(clip[0], clip[1])
    df_accuracy = df_merged.group_by("Country").agg(corr_expr.alias("Average_r"))
    df_out = df_counts.join(df_accuracy, on="Country", how="left").sort(
        "Nb_of_sites", descending=True
    )
    df_out = add_iso_codes(df_out)
    return df_out


def calculate_per_country_accuracy(
    df_pred: pl.DataFrame,
    run_folder: str,
    pred_col: str,
    clip: tuple[float, float] = (0, 1),
    true_col: str = "Observed",
) -> pl.DataFrame:
    """Compute Pearson's r per country."""
    df_site_info = read_site_info_parquet(run_folder)
    site_country = df_site_info.select(["SSBS", "Country"])
    df_merged = df_pred.join(site_country, on="SSBS", how="inner")
    df_accuracy = (
        df_merged.group_by("Country")
        .agg(pl.corr(true_col, pred_col).clip(clip[0], clip[1]).alias("Average_r"))
        .sort("Country")
    )
    df_accuracy = add_iso_codes(df_accuracy)
    return df_accuracy


def plot_world_heatmap(
    df: pl.DataFrame,
    value_col: str,
    min_max: tuple[float, float] | None = None,
    figsize: tuple[int, int] = (12, 6),
    cmap: str = "PuOr_r",
    show_legend: bool = True,
    legend_label: str | None = None,
    legend_decimals: bool = True,
    iso_col: str = "iso_a3",
) -> plt.Figure:
    """
    Plot a choropleth heatmap of country-level values using ISO-A3 codes.
    Countries without data are shown in light gray.
    """
    url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(url)[["ADMIN", "ISO_A3", "geometry"]]
    world = world[world["ADMIN"] != "Antarctica"]
    df = df.to_pandas()
    merged = world.merge(
        df[[iso_col, value_col]], left_on="ISO_A3", right_on=iso_col, how="left"
    )
    vmin, vmax = (min_max[0], min_max[1] if min_max is not None else (None, None))
    if vmin is None or vmax is None:
        vals = merged[value_col].dropna().to_numpy()
        vmin, vmax = (float(vals.min()), float(vals.max()))
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)
    merged.plot(
        column=value_col,
        cmap=cmap,
        norm=norm,
        ax=ax,
        linewidth=0.3,
        edgecolor="0.7",
        legend=show_legend,
        legend_kwds=(
            {"label": legend_label or value_col, "shrink": 0.6} if show_legend else None
        ),
        missing_kwds={"color": "lightgray", "edgecolor": "white"},
    )
    if show_legend:
        cax = fig.axes[-1]
        if not legend_decimals:
            cax.yaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_axis_off()
    fig.tight_layout()
    return fig
