# flake8: noqa
# mypy: ignore-errors
import json
from pathlib import Path

import geopandas as gpd
import jupyter_black
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import statsmodels.api as sm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from shapely.geometry import LineString

jupyter_black.load()

# Make summary and sample-size tables readable without unlimited dataframe output.
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(80)
pl.Config.set_tbl_width_chars(200)

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
    "fixed_eff": "cadetblue",
    "random_eff": "peachpuff",
    "training": "gray",
    "standard_cv": "darkseagreen",
    "cross_study_cv": "steelblue",
    "sbm": "cadetblue",
    "brm": "darksalmon",
    "btm": "palevioletred",
    "train_test_line": "black",
    "zero_line": "black",
    "value_text": "black",
    "metric_separator": "0.72",
    "panel_separator": "0.45",
    "legend_edge": "none",
    "map_edge": "white",
    "map_missing": "0.88",
    "globe_outline": "0.35",
}

# Run-folder mode names are part of the experiment folder naming convention.
evaluation_run_modes = {
    "Training": "training",
    "Standard CV": "standard_cv",
    "Cross-study CV": "cross_study",
}
evaluation_mode_order = list(evaluation_run_modes.keys())
cv_evaluation_mode_order = [
    mode for mode in evaluation_mode_order if mode != "Training"
]
evaluation_mode_color_keys = {
    "Training": "training",
    "Standard CV": "standard_cv",
    "Cross-study CV": "cross_study_cv",
}

model_order = ["SBM", "BRM", "BTM"]
performance_metric_order = ["Spearman", "R2", "MAE"]


def find_project_root(start: Path = Path.cwd()) -> Path:
    """
    Return the nearest parent directory that contains a .git folder. The
    notebook uses this to build stable relative paths no matter where the
    kernel was started from.
    """
    for path in [start, *start.parents]:
        if (path / "core").exists() and (path / "notebooks").exists():
            return path


# Locate the repository root once so all run-folder paths are stable.
project_root = find_project_root()
base_path = project_root.parent / "data" / "runs_revision"
key_output_path = "key_output"
site_info_filename = "site_info.parquet"
brm_added_output_path = "additional_output"
current_run_folder_suffix = "main"

spread_interval = "p2_5_97_5"  # "all", "iqr", "p2_5_97_5", "p5_95", or "p1_99"
effect_size_scale = "response"  # "latent" or "response"
effect_intervals = {
    "all": (0.0, 1.0),
    "iqr": (0.25, 0.75),
    "p2_5_97_5": (0.025, 0.975),
    "p5_95": (0.05, 0.95),
    "p1_99": (0.01, 0.99),
}

beta_training_folders = {}


def infer_run_folders(
    model_key: str, experiment_suffix: str = "main", run_root: Path | str | None = None
) -> dict[str, str]:
    """
    Infer the three evaluation-mode run folders for one experiment family.

    Run directories are assumed to follow the experiment-name convention
    `run_folder_<date>_<time>_<model_key>_<mode>_<experiment_suffix>`,
    where the timestamp is the only part that changes between launches. The
    returned dictionary uses the notebook display labels expected by the
    loading and plotting code: Training, Standard CV, and Cross-study CV.
    """
    run_root = Path(base_path if run_root is None else run_root)
    experiment_folders = {
        folder.name.split("_", 4)[4]: folder.name
        for folder in run_root.glob("run_folder_*")
    }
    experiment_names = {
        label: experiment_folders[f"{model_key}_{mode}_{experiment_suffix}"]
        for label, mode in evaluation_run_modes.items()
    }
    return experiment_names


def load_prediction_dataframes(
    model_folders: dict[str, str], base_path: str = base_path
) -> dict[str, pl.DataFrame]:
    """
    Load prediction tables for one model across manuscript evaluation modes.

    Input is a mapping from display mode to run folder. Training mode returns
    the full training predictions. Cross-validation modes concatenate the
    held-out test predictions across folds and add fold/mode labels. Each
    output row is one prediction for one observation, or one taxon-site
    prediction for taxonomic models.
    """
    out = {}
    for mode, run_folder in model_folders.items():
        run_path = Path(base_path) / run_folder
        key_output_dir = run_path / "key_output"
        if "training" in mode.lower():
            df = pl.read_parquet(key_output_dir / "train_predictions.parquet")
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


def print_prediction_shapes_and_studies(
    model_label: str, results: dict[str, pl.DataFrame], folders: dict[str, str]
) -> None:
    """Print prediction table shapes and unique study counts by mode."""
    print(f"{model_label} shapes and studies:")
    for mode, df in results.items():
        if "SS" in df.columns:
            n_studies = df.get_column("SS").n_unique()
        else:
            site_info_path = Path(base_path) / folders[mode] / site_info_filename
            site_info = pl.read_parquet(site_info_path, columns=["SSBS", "SS"])
            prediction_sites = df.select("SSBS").unique()
            prediction_studies = prediction_sites.join(site_info, on="SSBS", how="left")
            n_studies = prediction_studies.get_column("SS").n_unique()
        print(f"{mode}, {df.shape}, studies={n_studies}")


def read_site_info_parquet(
    run_folder: str,
    base_path: str = base_path,
    site_info_filename: str = site_info_filename,
) -> pl.DataFrame:
    """
    Load the site metadata table for a run folder.

    The metadata are used to add land-use, study, biome, realm, and taxonomic
    group labels to prediction outputs before derived summaries are calculated.
    """
    site_info_file = Path(base_path) / run_folder / site_info_filename
    df_site_info = pl.read_parquet(site_info_file)
    return df_site_info


def _pick_pred_col(df: pl.DataFrame, mode: str) -> str:
    """
    Return the prediction column used by the current output table. Depending on
    the model and evaluation mode, predictions may be in a generic "Predicted"
    column or split into "Predicted_RE" and "Predicted_FE", for predictions with
    random effects and fixed effects only, respectively.
    """
    if "Predicted" in df.columns:
        return "Predicted"
    if mode == "Training":
        return "Predicted_RE"
    return "Predicted_FE"


def _compute_performance_metrics(
    df: pl.DataFrame, true_col: str, pred_col: str
) -> dict[str, float]:
    """
    Compute the three manuscript performance metrics for one evaluated slice.

    Rows are the datapoints being evaluated: observations for base predictions,
    site-pair deltas for Fig 2c-d, or group/fold subsets when called from
    Fig 5 driver. Spearman measures rank correlation. MAE is the mean absolute
    prediction error on the evaluated rows. R2 is 1 - SSE/SST for the same rows.
    """
    y_true = df.get_column(true_col).to_numpy()
    y_pred = df.get_column(pred_col).to_numpy()
    residual_sse = np.sum((y_true - y_pred) ** 2)
    residual_sst = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - residual_sse / residual_sst)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    spearman = float(spearmanr(y_true, y_pred).statistic)
    metric_values = {"Spearman": spearman, "MAE": mae, "R2": r2}
    return metric_values


def build_model_performance_summary(
    dfs: dict[str, dict[str, pl.DataFrame]], true_col: str = "Observed"
) -> pl.DataFrame:
    """
    Build the tidy performance table used by Fig 2 bar plots.

    For each model and evaluation mode, the function returns one pooled
    all-observation summary row per metric. For CV modes it also returns one row
    per fold for diagnostics and tables. Fig 2 displays the pooled CV value.
    `Metric` stores the display name directly: Spearman, MAE, or R2.
    """
    rows = []
    for model_name, modes in dfs.items():
        for mode_name, df in modes.items():
            pred_col = _pick_pred_col(df, mode_name)
            metrics = _compute_performance_metrics(
                df=df, true_col=true_col, pred_col=pred_col
            )
            for metric_name, value in metrics.items():
                rows.append(
                    {
                        "Model": model_name,
                        "Eval type": mode_name,
                        "Metric": metric_name,
                        "Summary type": "All observations",
                        "Fold": None,
                        "Value": value,
                        "N": df.height,
                    }
                )
            if "training" not in mode_name.lower() and "fold" in df.columns:
                for fold_value, df_fold in df.partition_by(
                    "fold", as_dict=True
                ).items():
                    fold_id = (
                        fold_value[0] if isinstance(fold_value, tuple) else fold_value
                    )
                    fold_metrics = _compute_performance_metrics(
                        df=df_fold, true_col=true_col, pred_col=pred_col
                    )
                    for metric_name, value in fold_metrics.items():
                        rows.append(
                            {
                                "Model": model_name,
                                "Eval type": mode_name,
                                "Metric": metric_name,
                                "Summary type": "Fold",
                                "Fold": int(fold_id),
                                "Value": value,
                                "N": df_fold.height,
                            }
                        )
    df_out = pl.DataFrame(rows)
    return df_out


def effect_interval(values: list[float], interval: str) -> tuple[float, float]:
    """
    Return the selected interval for posterior-mean effect values.

    Each input value is one study-level slope for the SBM or one final
    ecological-group slope for the BRM/BTM. The configured manuscript spread
    is the full minimum-maximum range.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if interval == "all":
        low = float(arr.min())
        high = float(arr.max())
    else:
        q_low, q_high = effect_intervals[interval]
        low = float(np.quantile(arr, q_low))
        high = float(np.quantile(arr, q_high))
    return (low, high)


def filter_ecological_rows_to_active_term(
    df_term: pl.DataFrame, train_df: pl.DataFrame, term: str
) -> pl.DataFrame:
    """
    Keep ecological parameter rows whose final prediction group contains a term.

    BRM/BTM ecological ranges are intended to describe the group-level effects
    actually informed by a land-use category. For binary land-use indicators,
    this drops final rolled-up groups with no training observations in that
    category. Continuous terms are returned unchanged.
    """
    if term not in train_df.columns:
        return df_term
    term_values = (
        train_df.select(pl.col(term).drop_nulls().unique()).get_column(term).to_list()
    )
    is_binary_indicator = bool(term_values) and set(term_values).issubset(
        {0, 1, 0.0, 1.0}
    )
    if not is_binary_indicator:
        return df_term
    active_rows = train_df.filter(pl.col(term) == 1)
    if active_rows.is_empty():
        return df_term.head(0)
    active_pairs = active_rows.select(
        [
            pl.col("Final_hierarchical_level").alias("level"),
            pl.col("Final_hierarchical_group").alias("group"),
        ]
    ).unique()
    non_population = df_term.filter(pl.col("level") != "population").join(
        active_pairs, on=["level", "group"], how="inner"
    )
    has_population_support = (
        active_pairs.filter(pl.col("level") == "Population").height > 0
    )
    population = df_term.filter(pl.col("level") == "population")
    if not has_population_support:
        population = population.head(0)
    return pl.concat([non_population, population], how="vertical_relaxed")


def prepare_effect_panel_inputs(
    training_folders: dict[str, str],
    diversity_name: str,
    interval: str = spread_interval,
    effect_scale: str = effect_size_scale,
) -> tuple[dict[str, dict], list[str], tuple[float, float]]:
    """
    Load Fig 3 effect summaries from current training-run parameter outputs.

    Each training run must contain `key_output/parameter_summary.parquet`, which
    stores population, study, and ecological-group parameter summaries on both
    latent and response scales. `mean` is the population-level fixed slope, SBM
    ranges are active study-level total slopes, and BRM/BTM ranges are active
    final ecological-group total slopes from `rolled_up_hierarchy_mapping.json`.
    For binary land-use indicators, active means the study or final prediction
    group has at least one training row with that category present. Continuous
    covariates are not filtered by active support.
    """
    value_col = {"latent": "mean", "response": "response_mean"}[effect_scale]
    summaries = {}
    train_dataframes = {}
    effect_summaries = {model_name: {} for model_name in training_folders}
    for model_name in model_order:
        if model_name not in training_folders:
            continue
        run_folder = training_folders[model_name]
        run_path = Path(base_path) / run_folder / key_output_path
        train_path = (
            Path(base_path)
            / run_folder
            / brm_added_output_path
            / "train_dataframe.parquet"
        )
        parameter_path = run_path / "parameter_summary.parquet"
        if not parameter_path.exists():
            raise FileNotFoundError(
                f"Missing required parameter summary for {model_name}: {parameter_path}. The training run likely did not finish the current output-writing step."
            )
        if not train_path.exists():
            raise FileNotFoundError(
                f"Missing training dataframe for {model_name}: {train_path}. Fig 3 active land-use filtering requires this file."
            )
        summary = pl.read_parquet(parameter_path)
        summary = summary.with_columns(pl.col("covariate").cast(pl.Utf8).alias("term"))
        summaries[model_name] = summary
        train_dataframes[model_name] = pl.read_parquet(train_path)
        means = summary.filter(
            (pl.col("parameter") == "mu_beta") & pl.col("term").is_not_null()
        ).select(["term", value_col])
        for term, value in means.iter_rows():
            effect_summaries[model_name][term] = {"mean": float(value)}
    effect_order = (
        summaries["SBM"]
        .filter((pl.col("parameter") == "mu_beta") & pl.col("term").is_not_null())
        .get_column("term")
        .unique(maintain_order=True)
        .to_list()
    )
    sbm_study = summaries["SBM"].filter(
        (pl.col("parameter") == "beta_study") & pl.col("term").is_not_null()
    )
    for term_key, df_term in sbm_study.partition_by("term", as_dict=True).items():
        term = term_key[0] if isinstance(term_key, tuple) else term_key
        if term not in effect_order:
            continue
        active_studies = None
        if term in train_dataframes["SBM"].columns:
            term_values = (
                train_dataframes["SBM"]
                .select(pl.col(term).drop_nulls().unique())
                .get_column(term)
                .to_list()
            )
            is_binary_indicator = bool(term_values) and set(term_values).issubset(
                {0, 1, 0.0, 1.0}
            )
            if is_binary_indicator:
                active_studies = (
                    train_dataframes["SBM"]
                    .filter(pl.col(term) == 1)
                    .get_column("SS")
                    .unique()
                    .to_list()
                )
        if active_studies is not None:
            df_term = df_term.filter(
                pl.col("group").is_in([str(study) for study in active_studies])
            )
        if df_term.is_empty():
            continue
        low, high = effect_interval(df_term.get_column(value_col).to_list(), interval)
        effect_summaries["SBM"][term]["random_slope_lower"] = low
        effect_summaries["SBM"][term]["random_slope_upper"] = high
        effect_summaries["SBM"][term]["spread_n"] = df_term.height
    ecological_models = [
        model_name
        for model_name in model_order
        if model_name != "SBM" and model_name in training_folders
    ]
    for model_name in ecological_models:
        run_folder = training_folders[model_name]
        with open(
            Path(base_path) / run_folder / "rolled_up_hierarchy_mapping.json"
        ) as f:
            hierarchy = json.load(f)
        final_rows = []
        for level in hierarchy["column_names"]:
            if not level.startswith("level_"):
                continue
            groups = list(hierarchy.get(level, {}))
            if not groups:
                continue
            parameter = f"beta_{level.split('_')[-1]}"
            final_rows.append(
                summaries[model_name].filter(
                    (pl.col("parameter") == parameter)
                    & (pl.col("level") == level)
                    & pl.col("group").is_in(groups)
                    & pl.col("term").is_not_null()
                )
            )
        if "Population" in hierarchy:
            final_rows.append(
                summaries[model_name].filter(
                    (pl.col("parameter") == "mu_beta")
                    & (pl.col("level") == "population")
                    & pl.col("term").is_not_null()
                )
            )
        if not final_rows:
            continue
        final_effects = pl.concat(final_rows, how="vertical")
        for term_key, df_term in final_effects.partition_by(
            "term", as_dict=True
        ).items():
            term = term_key[0] if isinstance(term_key, tuple) else term_key
            if term not in effect_order or term not in effect_summaries[model_name]:
                continue
            df_term = filter_ecological_rows_to_active_term(
                df_term, train_dataframes[model_name], term
            )
            if df_term.is_empty():
                continue
            low, high = effect_interval(
                df_term.get_column(value_col).to_list(), interval
            )
            effect_summaries[model_name][term]["ecological_slope_lower"] = low
            effect_summaries[model_name][term]["ecological_slope_upper"] = high
            effect_summaries[model_name][term]["spread_n"] = df_term.height
    limit_values = []
    for effect_dict in effect_summaries.values():
        for term in effect_order:
            values = effect_dict.get(term, {})
            for key in [
                "mean",
                "random_slope_lower",
                "random_slope_upper",
                "ecological_slope_lower",
                "ecological_slope_upper",
            ]:
                if values.get(key) is not None:
                    limit_values.append(values[key])
    limit_values = np.asarray(limit_values, dtype=float)
    padding = 0.08 * (limit_values.max() - limit_values.min())
    effect_x_limits = (
        min(limit_values.min(), 0) - padding,
        max(limit_values.max(), 0) + padding,
    )
    return (effect_summaries, effect_order, effect_x_limits)


def plot_effect_panel(
    effect_summary: dict[str, dict[str, float]],
    show_axes_labels_values: bool = True,
    show_legend: bool = False,
    show_mean_values: bool = True,
    show_ecological_spread: bool = False,
    figsize: tuple[float, float] = (3.8, 5.4),
    axes_label_size: int = 9,
    axes_number_size: int = 9,
    ecological_line_width: float = 2.0,
    plot_effect_order: list[str] | None = None,
    plot_effect_x_limits: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot one Fig 3 effect-spread panel.

    Rows are covariates. Circles show population fixed-effect slopes. When present,
    peach bars show the configured range of SBM study-level slopes. Blue bars show
    the configured range of BRM/BTM final ecological-group slopes. Covariate
    labels are raw output names so final figure labels can be added separately.
    """
    if plot_effect_order is None:
        plot_effect_order = effect_order
    if plot_effect_x_limits is None:
        plot_effect_x_limits = effect_x_limits
    terms = [term for term in plot_effect_order if term in effect_summary]
    y_pos = np.arange(len(terms))
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(
        0, color=color_scheme["zero_line"], linestyle="--", linewidth=1.0, zorder=1
    )
    showed_study_label = False
    showed_ecological_label = False
    for i, term in enumerate(terms):
        values = effect_summary[term]
        if values.get("random_slope_lower") is not None:
            ax.hlines(
                y=y_pos[i],
                xmin=values["random_slope_lower"],
                xmax=values["random_slope_upper"],
                color=color_scheme["random_eff"],
                linewidth=5,
                label="" if showed_study_label else "Study heterogeneity",
                zorder=2,
            )
            showed_study_label = True
        if show_ecological_spread and values.get("ecological_slope_lower") is not None:
            ax.hlines(
                y=y_pos[i],
                xmin=values["ecological_slope_lower"],
                xmax=values["ecological_slope_upper"],
                color=color_scheme["fixed_eff"],
                linewidth=ecological_line_width,
                label="" if showed_ecological_label else "Ecological group spread",
                zorder=3,
            )
            showed_ecological_label = True
    means = np.asarray([effect_summary[term]["mean"] for term in terms])
    ax.plot(
        means,
        y_pos,
        "o",
        markersize=6,
        color=color_scheme["fixed_eff"],
        label="Population fixed-effect mean",
        zorder=5,
    )
    if show_mean_values:
        x_offset = 6 * np.sign(means)
        x_offset[x_offset == 0] = 6
        for y, mean, offset in zip(y_pos, means, x_offset):
            ax.annotate(
                f"{mean:.2f}",
                xy=(mean, y),
                xytext=(offset, 0),
                textcoords="offset points",
                ha="left" if offset > 0 else "right",
                va="center",
                fontsize=axes_number_size,
            )
    ax.set_yticks(y_pos)
    if show_axes_labels_values:
        ax.set_yticklabels(terms, fontsize=axes_label_size)
    else:
        ax.set_yticklabels([])
    ax.set_ylim(len(terms) - 0.5, -0.5)
    ax.set_xlim(*plot_effect_x_limits)
    ax.tick_params(
        axis="x", labelsize=axes_number_size, labelbottom=show_axes_labels_values
    )
    ax.tick_params(axis="y", length=0, labelleft=show_axes_labels_values)
    if show_legend:
        ax.legend(
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
            fontsize=axes_label_size,
        )
    fig.tight_layout()
    return fig


def _load_glmm_parameter_values(path: Path, value_col: str) -> pl.DataFrame:
    """
    Load GLMM fixed-effect estimates from train_effects JSON output.

    The returned table has one row per covariate and is used as either the
    full-training baseline or a fold-specific estimate. Covariate names are used
    exactly as written in the JSON output.
    """
    with open(path) as in_stream:
        effects = json.load(in_stream)
    rows = []
    for term, values in effects.items():
        if "mean" not in values:
            continue
        rows.append({"covariate": str(term), value_col: float(values["mean"])})
    parameter_values = pl.DataFrame(rows)
    return parameter_values


def _load_bhm_parameter_values(
    path: Path, model_name: str, value_col: str, train_path: Path
) -> pl.DataFrame:
    """
    Load BHM parameter posterior means for the component shown in Fig 4c-e.

    SBM uses population fixed-effect means. BRM and BTM use the ecological
    parameter rows listed in `rolled_up_hierarchy_mapping.json`, i.e. the same
    hierarchy levels that can be used for rolled-up out-of-sample prediction.
    """
    summary = pl.read_parquet(path)
    if model_name == "SBM":
        parameter_values = summary.filter(
            (pl.col("parameter") == "mu_beta")
            & (pl.col("level") == "population")
            & pl.col("covariate").is_not_null()
        ).select(["covariate", pl.col("response_mean").alias(value_col)])
        return parameter_values
    with open(train_path / "rolled_up_hierarchy_mapping.json") as in_stream:
        hierarchy = json.load(in_stream)
    final_rows = []
    for level in hierarchy["column_names"]:
        if not level.startswith("level_"):
            continue
        groups = list(hierarchy.get(level, {}))
        if not groups:
            continue
        parameter = f"beta_{level.split('_')[-1]}"
        level_rows = summary.filter(
            (pl.col("parameter") == parameter)
            & (pl.col("level") == level)
            & pl.col("group").is_in(groups)
            & pl.col("covariate").is_not_null()
        ).select(["group", "covariate", pl.col("response_mean").alias(value_col)])
        if level_rows.height:
            final_rows.append(level_rows)
    if "Population" in hierarchy:
        population_rows = summary.filter(
            (pl.col("parameter") == "mu_beta")
            & (pl.col("level") == "population")
            & pl.col("covariate").is_not_null()
        ).select(
            [
                pl.lit("Population").alias("group"),
                "covariate",
                pl.col("response_mean").alias(value_col),
            ]
        )
        if population_rows.height:
            final_rows.append(population_rows)
    if not final_rows:
        return pl.DataFrame({"group": [], "covariate": [], value_col: []})
    parameter_values = pl.concat(final_rows, how="vertical")
    return parameter_values


def build_parameter_spread_table(
    cv_folders: dict[str, dict[str, str]],
    training_folders: dict[str, str] = beta_training_folders,
) -> pl.DataFrame:
    """
    Build Fig 4c-e conditional-shift parameter-change table.

    For each model, CV mode, fold, covariate, and final group where applicable, the datapoint is the fold-training estimate minus the full-training estimate. BHM outputs are read from parameter_summary parquet files; GLMM outputs are read from train_effects JSON files. The plot later summarizes these datapoints with the configured spread interval.
    """
    rows = []
    for cv_mode, model_folders in cv_folders.items():
        for model_name, run_folder in model_folders.items():
            train_path = Path(base_path) / training_folders[model_name]
            cv_path = Path(base_path) / run_folder
            bhm_train_path = train_path / key_output_path / "parameter_summary.parquet"
            glmm_train_path = train_path / key_output_path / "train_effects.json"
            if bhm_train_path.exists():
                baseline = _load_bhm_parameter_values(
                    path=bhm_train_path,
                    model_name=model_name,
                    value_col="baseline",
                    train_path=train_path,
                )
                cv_files = sorted(
                    (cv_path / key_output_path).glob(
                        "parameter_summary_fold_*.parquet"
                    ),
                    key=lambda path: int(path.stem.split("_")[-1]),
                )
                for path in cv_files:
                    fold = int(path.stem.split("_")[-1])
                    values = _load_bhm_parameter_values(
                        path=path,
                        model_name=model_name,
                        value_col="value",
                        train_path=train_path,
                    )
                    join_cols = (
                        ["covariate"] if model_name == "SBM" else ["group", "covariate"]
                    )
                    rows.append(
                        values.join(baseline, on=join_cols, how="inner").with_columns(
                            [
                                (pl.col("value") - pl.col("baseline")).alias("change"),
                                pl.lit(cv_mode).alias("CV mode"),
                                pl.lit(model_name).alias("Model"),
                                pl.lit(fold).alias("fold"),
                            ]
                        )
                    )
                continue
            baseline = _load_glmm_parameter_values(glmm_train_path, "baseline")
            cv_files = sorted(
                (cv_path / key_output_path).glob("train_effects_fold_*.json"),
                key=lambda path: int(path.stem.split("_")[-1]),
            )
            for path in cv_files:
                fold = int(path.stem.split("_")[-1])
                values = _load_glmm_parameter_values(path, "value")
                rows.append(
                    values.join(baseline, on="covariate", how="inner").with_columns(
                        [
                            (pl.col("value") - pl.col("baseline")).alias("change"),
                            pl.lit(cv_mode).alias("CV mode"),
                            pl.lit(model_name).alias("Model"),
                            pl.lit(fold).alias("fold"),
                        ]
                    )
                )
    parameter_spread_table = pl.concat(rows, how="diagonal_relaxed")
    return parameter_spread_table


def plot_parameter_spread_panel(
    parameter_table: pl.DataFrame,
    model_name: str,
    effect_order: list[str],
    figsize: tuple[float, float] = (4.8, 5.2),
    show_axes_labels_values: bool = True,
    show_legend: bool = True,
    interval: str = spread_interval,
    x_limits: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Plot one Fig 4c-e parameter-spread panel.

    Rows are the selected diversity metric's covariates. Each horizontal segment summarizes
    fold-training parameter changes from the full-training fit, with standard
    and cross-study CV overlaid. The interval argument uses the same configured
    spread interval as the effect panels. Labels use raw output covariate names.
    """
    df = parameter_table.filter(pl.col("Model") == model_name).to_pandas()
    terms = [term for term in effect_order if term in set(df["covariate"])]
    y_lookup = {term: i for i, term in enumerate(terms)}
    offsets = {"Standard CV": -0.16, "Cross-study CV": 0.16}
    colors = {
        cv_mode: color_scheme[evaluation_mode_color_keys[cv_mode]]
        for cv_mode in cv_evaluation_mode_order
    }
    summary = (
        df.groupby(["CV mode", "covariate"], as_index=False)["change"]
        .agg(
            low=lambda values: effect_interval(values.tolist(), interval)[0],
            high=lambda values: effect_interval(values.tolist(), interval)[1],
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.axvline(
        0, color=color_scheme["zero_line"], linestyle="--", linewidth=1, zorder=1
    )
    for cv_mode, color in colors.items():
        cv_summary = summary.loc[summary["CV mode"] == cv_mode]
        cv_summary = cv_summary.loc[cv_summary["covariate"].isin(terms)]
        y_summary = np.asarray(
            [y_lookup[term] + offsets[cv_mode] for term in cv_summary["covariate"]]
        )
        ax.hlines(
            y=y_summary,
            xmin=cv_summary["low"],
            xmax=cv_summary["high"],
            color=color,
            linewidth=3,
            alpha=0.95,
            zorder=2,
        )
    ax.set_yticks(np.arange(len(terms)))
    if show_axes_labels_values:
        ax.set_yticklabels(terms)
    else:
        ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel(
        "Parameter change from training fit" if show_axes_labels_values else ""
    )
    ax.tick_params(axis="x", labelbottom=show_axes_labels_values)
    ax.tick_params(axis="y", length=0, labelleft=show_axes_labels_values)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if show_legend:
        handles = [
            Line2D([0], [0], color=color, linewidth=3, label=cv_mode)
            for cv_mode, color in colors.items()
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            borderaxespad=0,
        )
    fig.tight_layout()
    return fig
