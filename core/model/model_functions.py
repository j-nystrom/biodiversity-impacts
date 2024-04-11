from typing import Union

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler

from core.utils.general_utils import create_logger

logger = create_logger(__name__)


def filter_data_scope(
    df: pl.DataFrame,
    taxonomic_resolution: str,
    geographic_scope: dict[str, list[str]],
    species_scope: list[str],
) -> pl.DataFrame:
    """
    Filters the data so that only the biomes / geographic regions and species
    groups specified in the config are included.

    Args:
        taxonomic_resolution: Level in the taxonomy used in this model.
        geographic_scope: Dictionary with biomes and geographic regions to
            include.
        species_scope: List of species groups to include for this level.

    Returns:
        df: Dataframe with rows removed based on the specified scope.
    """
    logger.info("Filtering data based on specified scope.")

    for col in geographic_scope.keys():
        df = df.filter(pl.col(col).is_in(geographic_scope[col]))

    if species_scope:  # Check if the list is not empty
        df = df.filter(pl.col(taxonomic_resolution).is_in(species_scope))

    logger.info("Finished filtering data based on specified scope.")

    return df


def standardize_continuous_covariates(
    df: pl.DataFrame, vars_to_scale: list[str]
) -> pl.DataFrame:
    """
    Standardizes the continuous covariates in the dataframe.

    Args:
        df: Dataframe containing the covariates.
        vars_to_scale: List of column names of the continuous covariates to be
            standardized.

    Returns:
        df_res: Dataframe with the standardized continuous covariates.

    """
    logger.info("Standardizing continuous covariates.")

    scaler = StandardScaler()
    data_to_scale = df.select(vars_to_scale).to_numpy()
    df_scaled = pl.DataFrame(scaler.fit_transform(data_to_scale), schema=vars_to_scale)

    # Check for NaN and infinite values
    for col in vars_to_scale:
        inf_sum = df_scaled.get_column(col).is_infinite().sum()
        nan_sum = df_scaled.get_column(col).is_nan().sum()
        if inf_sum > 0:
            logger.warning(f"{inf_sum} infinite values found in {col}. Filling with 0.")
        if nan_sum > 0:
            logger.warning(f"{nan_sum} NaN values found in {col}. Filling with 0.")
    df_scaled = df_scaled.fill_nan(0)

    df_res = pl.concat([df.drop(vars_to_scale), df_scaled], how="horizontal")

    logger.info("Finished standardizing continuous covariates.")

    return df_res


def create_interaction_terms(
    df: pl.DataFrame,
    categorical_vars: list[str],
    continuous_vars: list[str],
) -> pl.DataFrame:
    """
    Creates interaction terms between land-use related columns and population
    and road density at different resolutions.

    Args:
        df: Dataframe containing population and road density and the dummy
            columns for land-use and land-use intensity.

    Returns:
        df_res: Updated df with interaction terms added.

    """
    logger.info("Creating specified interaction terms.")

    for col_1 in continuous_vars:
        for col_2 in categorical_vars:
            df = df.with_columns(
                (pl.col(col_1) * pl.col(col_2)).alias(f"{col_2} x {col_1}")
            )

    logger.info("Finished creating interaction terms.")

    return df


def transform_response_variable(df: pl.DataFrame, method: str) -> pl.DataFrame:
    """
    Makes adjustments and transformations to the response variable (max scaled
    abundance). The first adjustment is to avoid 0 and 1 values, which are not
    supported by the Beta distribution. The second transformation is a square
    root transformation. The third transformation is a logit transformation.

    Args:
        df: Dataframe containing the response variable.

    Returns:
        df: Updated df with the transformed response variable as new columns.
    """
    logger.info("Transforming response variable.")

    adjust = 0.001
    original_col_name = "Max_scaled_abundance"
    transformed_col_name = original_col_name

    # Small adjustment to align with support for Beta distribution
    if method == "adjust" or method == "logit":
        df = df.with_columns(
            pl.when(pl.col(original_col_name) == 0)
            .then(adjust)
            .when(pl.col(original_col_name) == 1)
            .then(1 - adjust)
            .otherwise(pl.col(original_col_name))
            .alias(original_col_name)
        )
        if method == "logit":
            transformed_col_name += "_logit"
            df = df.with_columns(
                pl.col(original_col_name)
                .map_elements(lambda x: logit(x))
                .alias(transformed_col_name)
            )

    # Square root transformation
    elif method == "sqrt":
        transformed_col_name += "_sqrt"
        df = df.with_columns(
            pl.col(original_col_name).sqrt().alias(transformed_col_name)
        )

    # Replace original column with transformed one, if the name has changed
    if transformed_col_name != original_col_name:
        original_col_index = df.columns.index(original_col_name)
        new_col = df.get_column(transformed_col_name)
        df = df.drop([original_col_name, transformed_col_name])
        df = df.insert_column(index=original_col_index, column=new_col)

    logger.info("Finished transforming response variable.")

    return df


def add_intercept(df: pl.DataFrame, response_var: str) -> pl.DataFrame:
    """
    Adds an intercept column to the dataframe.

    Args:
        df: Dataframe to which the intercept column should be added.

    Returns:
        df: Updated df with the intercept column added.
    """
    logger.info("Adding intercept column to the dataframe.")

    response_var_idx = next(
        (i for i, col in enumerate(df.columns) if response_var in col), None
    )
    if response_var_idx is not None:
        df = df.insert_column(
            index=response_var_idx + 1, column=pl.Series("Intercept", [1] * df.shape[0])
        )
    else:
        logger.warning("Response variable not found in dataframe.")

    logger.info("Intercept column added.")

    return df


def format_data_for_model(
    df_scaled: pl.DataFrame,
    group_vars: list[str],
    response_var: str,
    response_var_transform: str,
    categorical_vars: list[str],
    continuous_vars: list[str],
    interaction_terms: list[str],
) -> tuple[
    np.array,
    np.array,
    dict[str, np.array],
    np.array,
    dict[str, int],
    np.array,
    np.array,
]:
    """
    Takes the dataframe and formats it in a way that can be used by the PyMC
    model. This includes creating a design matrix, a vector of output values,
    observation indices on group, block and study levels, and a coordinate
    dictionary.

    Args:
        df_scaled: Dataframe with the scaled covariates and response variable.
        group_vars: List of group variables.
        response_var: Name of the response variable.
        response_var_transform: Transformation to apply to response variable.
        categorical_vars: List of categorical covariates.
        continuous_vars: List of continuous covariates.
        interaction_terms: List of interaction terms to add to design matrix.

    Returns:
        x: Design matrix.
        y: Vector of output values.
        coords: Coordinate dictionary.
        group_idx: Group indices.
        group_code_map: Mapping dictionary for group indices.
        study_idx: Study indices.
        block_idx: Block indices.
    """
    logger.info("Formatting data for PyMC model.")

    # Sort dataframe for consistent operations below
    df_scaled.sort(["SS", "SSB", "SSBS"])

    # Extract all unique taxonomic groups and create a mapping dictionary
    # Use the mapping dictionary to create an array with numerical group id
    if group_vars[-1] != "SSBS":
        taxa = df_scaled.get_column(group_vars[-1]).unique().to_list()
        taxa_code_map = {taxon: code for code, taxon in enumerate(taxa)}
        taxon_idx = (
            df_scaled.get_column(group_vars[-1])
            .map_elements(lambda x: taxa_code_map[x])
            .to_numpy()
        )
    else:
        taxa, taxa_code_map, taxon_idx = np.zeros(1), np.zeros(1), np.zeros(1)

    # Do the same for the study ID and block ID
    studies = df_scaled.get_column("SS").unique().to_list()
    study_code_map = {study: code for code, study in enumerate(studies)}
    study_idx = (
        df_scaled.get_column("SS").map_elements(lambda x: study_code_map[x]).to_numpy()
    )

    blocks = df_scaled.get_column("SSB").unique().to_list()
    block_code_map = {block: code for code, block in enumerate(blocks)}
    block_idx = (
        df_scaled.get_column("SSB").map_elements(lambda x: block_code_map[x]).to_numpy()
    )

    # Create an array with block-to-study indices
    block_to_study_idx = (
        df_scaled.select(["SS", "SSB"])
        .unique()
        .with_columns(
            pl.col("SS").apply(lambda x: study_code_map[x]).alias("Study_idx")
        )
        .get_column("Study_idx")
        .to_numpy()
    )

    # Get a vector of output values
    if response_var_transform:
        y_col = response_var + "_" + response_var_transform
    else:
        y_col = response_var
    y = df_scaled.get_column(y_col).to_numpy().flatten()

    # Create the fixed effects design matrix
    x = df_scaled.select(
        categorical_vars + continuous_vars + interaction_terms
    ).to_numpy()

    # Create design matrix for study and block random effects
    z_s = df_scaled.select(categorical_vars + continuous_vars).to_numpy()

    z_b = df_scaled.select(categorical_vars + continuous_vars).to_numpy()

    # Create coordinate dictionary
    idx = np.arange(x.shape[0])
    coords = {
        "idx": idx,
        "x_var": categorical_vars + continuous_vars + interaction_terms,
        "z_s_var": categorical_vars + continuous_vars,
        "z_b_var": categorical_vars + continuous_vars,
        "studies": studies,
        "blocks": blocks,
        "taxa": taxa,
    }

    # Convert numpy array to the precision actually needed, and no more,
    # to increase sampling speed
    y = y.astype(np.float32)
    x = x.astype(np.float32)
    z_s = z_s.astype(np.float32)
    z_b = z_b.astype(np.float32)
    if taxon_idx:
        taxon_idx = taxon_idx.astype(np.uint16)
    study_idx = study_idx.astype(np.uint16)
    block_idx = block_idx.astype(np.uint16)
    block_to_study_idx.astype(np.uint16)
    idx = idx.astype(np.uint32)

    logger.info("Data formatted for PyMC model.")

    return (
        y,
        x,
        z_s,
        z_b,
        coords,
        study_idx,
        block_idx,
        block_to_study_idx,
        taxon_idx,
        taxa_code_map,
    )


def run_sampling(
    model: pm.Model, sampler_settings: dict[str, Union[str, int, float]]
) -> az.InferenceData:
    """
    Runs the NUTS sampler for the current model.

    Args:
        model: PyMC model object.
        sampler_settings: Dictionary with settings for the NUTS sampler.

    Returns:
        trace: PyMC trace with posterior distributions information appended.
    """
    with model:
        trace = pm.sample(
            draws=sampler_settings["draws"],
            tune=sampler_settings["tune"],
            cores=sampler_settings["cores"],
            chains=sampler_settings["chains"],
            target_accept=sampler_settings["target_accept"],
            nuts_sampler=sampler_settings["nuts_sampler"],
        )

    return trace


def summarize_sampling_statistics(
    trace: az.InferenceData, var_names: list[str]
) -> None:
    """
    Calcualtes and logs sampling statistics for the model to evaluate the
    convergence of the sampling chains. This includes divergences, acceptance
    rate, R-hat statistics and ESS statistics.

    Args:
        trace: The trace from the NUTS MCMC sampling.
        var_names: List of variable names to summarize (the main parameters in
            the model).
    """
    idata = az.convert_to_dataset(trace)

    # Divergences
    divergences = np.sum(trace.sample_stats["diverging"].values)
    logger.info(f"There are {divergences} divergences in the sampling chains.")

    # Acceptance rate
    accept_rate = np.mean(trace.sample_stats["acceptance_rate"].values)
    logger.info(f"The mean acceptance rate was {round(accept_rate, 3)}")

    # R-hat statistics
    for var in var_names:
        try:
            r_hat = az.summary(idata, var_names=var, round_to=2)["r_hat"]
            mean_r_hat = round(np.mean(r_hat), 3)
            min_r_hat = round(np.min(r_hat), 3)
            max_r_hat = round(np.max(r_hat), 3)
            logger.info(
                f"R-hat for {var} are: {mean_r_hat} (mean) | {min_r_hat} (min) | \
                    {max_r_hat} (max)"
            )
        except KeyError:
            continue

    # ESS statistics
    for var in var_names:
        try:
            ess = az.summary(idata, var_names=var_names, round_to=2)["ess_bulk"]
            mean_ess = round(np.mean(ess), 0)
            min_ess = round(np.min(ess), 0)
            max_ess = round(np.max(ess), 0)
            logger.info(
                f"ESS for {var} are: {int(mean_ess)} (mean) | {int(min_ess)} (min) | \
                    {int(max_ess)} (max)"
            )
        except KeyError:
            continue


def make_in_sample_predictions(
    model: pm.Model,
    trace: az.InferenceData,
    y: np.array,
    x: np.array,
    z_s: np.array,
    z_b: np.array,
    idx: np.array,
) -> tuple[az.InferenceData, np.array]:
    """
    Makes in-sample predictions using the posterior distributions from the
    trace generated by the MCMC sampling.

    Args:
        model: PyMC model object.
        trace: PyMC trace from sampling.
        x: Design matrix.
        y: Vector of output values, only used as a placeholder.
        idx: Index vector.

    Returns:
        trace: Updated PyMC trace with predicted values appended.
        p_pred: Predicted values.

    """
    data_dict = {"x": x, "y": y}
    # if z_s:
    # data_dict["z_s"] = z_s
    # if z_b:
    # data_dict["z_b"] = z_b

    with model:
        pm.set_data(
            data_dict,
            coords={"idx": idx},
        )
        trace.extend(pm.sample_posterior_predictive(trace, predictions=False))

    p_pred = trace.posterior_predictive["y_like"].mean(dim=["draw", "chain"]).values

    return trace, p_pred


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
