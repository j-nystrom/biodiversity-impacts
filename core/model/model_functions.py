import time
from datetime import timedelta
from typing import Any

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
from scipy.special import logit
from sklearn.model_selection import KFold, StratifiedKFold
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
        taxonomic_resolution: Level of taxonomic granularity used in the model.
        geographic_scope: Dictionary that contains which biomes and geographic
            regions to include.
        species_scope: List of species groups to include for this level.

    Returns:
        df: Dataframe with rows removed based on the specified scope.
    """
    logger.info("Filtering data based on specified scope.")

    # For each column in the geographic scope dictionary (Biome, UN_region),
    # filter dataframe to only contain rows that are included in that list
    for col in geographic_scope.keys():
        df = df.filter(pl.col(col).is_in(geographic_scope[col]))

    if species_scope:  # Check if the list is not empty (e.g. for All_species)
        df = df.filter(pl.col(taxonomic_resolution).is_in(species_scope))

    logger.info("Finished filtering data based on specified scope.")

    return df


def filter_out_small_groups(df: pl.DataFrame, threshold: int) -> pl.DataFrame:
    """
    Filters out biogeographic groups that are too small to be included in the
    model. This is done for the biome-realm and ecoregion levels.

    Args:
        df: Dataframe with the species abundance data.

    Returns:
        df: Dataframe with the small groups removed.
    """
    logger.info(f"Filtering out groups with <{threshold} observations.")
    original_len = len(df)

    # Get list of biome-realm combinations below threshold and use as filter
    df_realm_count = df.group_by("Biome_Realm").agg(pl.col("SSBS").count())
    biome_realms = (
        df_realm_count.filter(pl.col("SSBS") < threshold)
        .get_column("Biome_Realm")
        .to_list()
    )
    df = df.filter(~pl.col("Biome_Realm").is_in(biome_realms))

    # Do the same operation for ecoregions
    df_eco_count = df.group_by("Ecoregion").agg(pl.col("SSBS").count())
    ecoregions = (
        df_eco_count.filter(pl.col("SSBS") < threshold)
        .get_column("Ecoregion")
        .to_list()
    )
    df = df.filter(~pl.col("Ecoregion").is_in(ecoregions))

    diff = original_len - len(df)
    logger.info(f"Filtered out {diff} observations based on group size.")

    return df


def standardize_continuous_covariates(
    df: pl.DataFrame, vars_to_scale: list[str]
) -> pl.DataFrame:
    """
    Standardizes the continuous covariates in the dataframe, by subtracting the
    mean and dividing by the standard devation.

    Args:
        df: Dataframe containing the covariates.
        vars_to_scale: List of column names of the continuous covariates to be
            standardized.

    Returns:
        df_res: Original dataframe with the standardized continuous covariates.

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
    df_scaled = df_scaled.fill_nan(0)

    df_res = pl.concat([df.drop(vars_to_scale), df_scaled], how="horizontal")

    logger.info("Finished standardizing continuous covariates.")

    return df_res


def create_interaction_terms(
    df: pl.DataFrame,
    categorical_vars: list[str],
    continuous_vars: list[str],
) -> tuple[pl.DataFrame, list]:
    """
    Creates interaction terms between land-use related (categorical) columns
    and population and road density (continuous) at different resolutions.

    Args:
        df: Dataframe containing population and road density and the dummy
            columns for land-use and land-use intensity.

    Returns:
        df_res: Updated df with interaction terms added.
        new_cols: List of the names of the new interaction columns.
    """
    logger.info("Creating specified interaction terms.")

    new_cols = []
    for col_1 in continuous_vars:
        for col_2 in categorical_vars:
            df = df.with_columns(
                (pl.col(col_1) * pl.col(col_2)).alias(f"{col_2} x {col_1}")
            )
            new_cols.append(f"{col_2} x {col_1}")

    logger.info("Finished creating interaction terms.")

    return df, new_cols


def transform_response_variable(
    df: pl.DataFrame, response_var: str, method: str
) -> pl.DataFrame:
    """
    Makes adjustments and transformations to the response variable (max scaled
    abundance). The first adjustment is to avoid 0 and 1 values, which are not
    supported by the logit transformation (or the Beta distribution). The
    second transformation is a square root transformation. The third
    transformation is a logit transformation.

    Args:
        df: Dataframe containing the response variable.
        response_var: Name of the response variable.
        method: Transformation method to apply.

    Returns:
        df: Updated df with the transformed response variable as new column.
    """
    logger.info("Transforming response variable.")

    # Small adjustment to align with support for logit / Beta distribution
    adjust = 0.001
    if method == "adjust" or method == "logit":
        df = df.with_columns(
            pl.when(pl.col(response_var) < adjust)
            .then(adjust - pl.col(response_var))  # Add to reach 0.001
            .when(pl.col(response_var) > (1 - adjust))
            .then(1 - adjust)  # Subtract to reach 0.999
            .otherwise(pl.col(response_var))
            .alias(response_var)
        )
        if method == "logit":
            transformed_col_name = response_var + "_logit"
            df = df.with_columns(
                pl.col(response_var)
                .map_elements(lambda x: logit(x))
                .alias(transformed_col_name)
            )

    # Square root transformation
    elif method == "sqrt":
        transformed_col_name = response_var + "_sqrt"
        df = df.with_columns(pl.col(response_var).sqrt().alias(transformed_col_name))

    else:
        transformed_col_name = response_var

    # Replace original column with transformed one, if the name has changed
    if transformed_col_name != response_var:
        original_col_index = df.columns.index(response_var)
        new_col = df.get_column(transformed_col_name)
        df = df.drop([response_var, transformed_col_name])
        df = df.insert_column(index=original_col_index, column=new_col)

    logger.info("Finished transforming response variable.")

    return df


def format_data_for_model(
    df: pl.DataFrame,
    response_var: str,
    categorical_vars: list[str],
    continuous_vars: list[str],
    interaction_terms: list[str],
    site_name_to_idx: dict[str, int],
) -> dict[str, Any]:
    """
    Takes the dataframe and formats it in a way that can be used by the PyMC
    model. This includes creating a design matrix, a vector of output values,
    indices for different groups and a coordinate dictionary. The output
    supports a model hierarchy that is either based on the studies and spatial
    blocks in the PREDICTS data, or one based on biomes, realms and ecoregions.

    Args:
        df: Dataframe with the scaled covariates and response variable.
        response_var: Name of the response variable.
        categorical_vars: List of categorical covariates.
        continuous_vars: List of continuous covariates.
        interaction_terms: List of interaction terms.
        site_name_to_idx: Mapping of sites to their indices in dataframe.

    Returns:
        output_dict: Dictionary that contains all the formatted data for the
            PyMC model.
    """
    logger.info("Formatting data for PyMC model.")

    # Sort dataframe for consistent operations below
    df = df.sort(["SS", "SSB", "SSBS"])

    # Extract studies and blocks as indices
    studies = df.get_column("SS").unique().to_list()
    study_idx = df.get_column("SS").cast(pl.Categorical).to_physical().to_numpy()
    blocks = df.get_column("SSB").unique().to_list()
    block_idx = df.get_column("SSB").cast(pl.Categorical).to_physical().to_numpy()

    # Create an array with block-to-study indices
    block_to_study_idx = (
        df.select(["SS", "SSB"])
        .unique()
        .sort(["SS", "SSB"])
        .get_column("SS")
        .cast(pl.Categorical)
        .to_physical()
        .to_numpy()
    )

    # Do the same as above but for biomes, realms and ecoregions
    df = df.sort(["Biome", "Biome_Realm", "Biome_Realm_Ecoregion"])

    biomes = df.get_column("Biome").unique().to_list()
    biome_idx = df.get_column("Biome").cast(pl.Categorical).to_physical().to_numpy()
    biome_realm = df.get_column("Biome_Realm").unique().to_list()
    biome_realm_idx = (
        df.get_column("Biome_Realm").cast(pl.Categorical).to_physical().to_numpy()
    )

    biome_realm_eco = df.get_column("Biome_Realm_Ecoregion").unique().to_list()
    biome_realm_eco_idx = (
        df.get_column("Biome_Realm_Ecoregion")
        .cast(pl.Categorical)
        .to_physical()
        .to_numpy()
    )

    realm_to_biome_idx = (
        df.select(["Biome", "Biome_Realm"])
        .unique()
        .sort(["Biome", "Biome_Realm"])
        .get_column("Biome")
        .cast(pl.Categorical)
        .to_physical()
        .to_numpy()
    )

    eco_to_realm_idx = (
        df.select(["Biome_Realm", "Biome_Realm_Ecoregion"])
        .unique()
        .sort(["Biome_Realm", "Biome_Realm_Ecoregion"])
        .get_column("Biome_Realm")
        .cast(pl.Categorical)
        .to_physical()
        .to_numpy()
    )

    # Get a vector of output values
    y = (
        df.select([col for col in df.columns if response_var in col])
        .to_numpy()
        .flatten()
    )

    # Create main design matrix that includes all covariates
    x = df.select(categorical_vars + continuous_vars + interaction_terms).to_numpy()

    # Coordinate dictionary for PyMC model
    idx = np.arange(len(y))
    site_names = df.get_column("SSBS").to_list()
    site_idx = np.array([site_name_to_idx[site] for site in site_names])
    coords = {
        "idx": idx,
        "x_vars": categorical_vars + continuous_vars + interaction_terms,
        "x_vars_int": ["Intercept"]
        + categorical_vars
        + continuous_vars
        + interaction_terms,
        "biomes": biomes,
        "biome_realm": biome_realm,
        "biome_realm_eco": biome_realm_eco,
        "studies": studies,
        "blocks": blocks,
    }

    output_dict = {
        "coords": coords,
        "y": y,
        "x": x,
        "biome_idx": biome_idx,
        "biome_realm_idx": biome_realm_idx,
        "realm_to_biome_idx": realm_to_biome_idx,
        "biome_realm_eco_idx": biome_realm_eco_idx,
        "eco_to_realm_idx": eco_to_realm_idx,
        "study_idx": study_idx,
        "block_idx": block_idx,
        "block_to_study_idx": block_to_study_idx,
        "site_idx": site_idx,
    }

    logger.info("Data formatted for PyMC model.")

    return output_dict


def create_stratification_column(
    df: pl.DataFrame, stratify_groups: list[str]
) -> pl.DataFrame:
    """
    Create a new column for stratification by concatenating the specified group
    columns.

    Args:
        df: Dataframe with the data to stratify.
        stratify_groups: List of column names to concatenate into one
            stratification column.
    Returns:
        df: Dataframe including the new stratification column.
    """
    logger.info("Creating stratification column for k-folds.")

    if len(stratify_groups) > 1:
        df = df.with_columns(
            pl.concat_str(stratify_groups, separator="_").alias("Stratify_group")
        )
    else:
        df = df.with_columns(df.get_column(stratify_groups[0]).alias("Stratify_group"))

    logger.info("Finished creating stratification column.")

    return df


def generate_kfolds(
    df: pl.DataFrame,
    k: int = 5,
    stratify: bool = False,
) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
    """
    Create indices corresponding to the train and test sets for k-fold cross-
    validation, and split the data accordingly. Stratification on a specific
    column is optional.

    Args:
        df: Dataframe with the data to split.
        k: Number of folds for the cross-validation.
        stratify: Whether to stratify the data based on a specific column.

    Returns:
        df_train_list: List of dataframes for the training sets.
        df_test_list: List of dataframes for the test sets.
    """
    logger.info("Generating k-folds for cross-validation.")

    # Covert polars dataframe to pandas, since sklearn K-fold is not compatible
    # with polars dataframes
    df = df.to_pandas()

    # Lists for storing the train and test datasets
    df_train_list = []
    df_test_list = []

    # Set up stratified k-fold sampler object and sample using the
    # stratify code (as the "y class label") for stratification
    if stratify:
        kfold = StratifiedKFold(n_splits=k, shuffle=True)
        strat_col = df["Stratify_group"]
    else:
        kfold = KFold(n_splits=k, shuffle=True)
        strat_col = None

    for train_index, test_index in kfold.split(X=df, y=strat_col):
        df_train, df_test = df.iloc[train_index], df.iloc[test_index]

        # Reconversion to polars dataframes
        df_train = pl.DataFrame(df_train)
        df_test = pl.DataFrame(df_test)

        # Store the data for this fold
        df_train_list.append(df_train)
        df_test_list.append(df_test)

    logger.info("Finished generating k-folds for cross-validation.")

    return df_train_list, df_test_list


def run_sampling(
    model: pm.Model, sampler_settings: dict[str, str | int | float]
) -> az.InferenceData:
    """
    Runs the NUTS sampler for the current model.

    Args:
        model: PyMC model object.
        sampler_settings: Dictionary with settings for the NUTS sampler.

    Returns:
        trace: PyMC trace with posterior distributions information appended.
    """
    logger.info("Running NUTS sampler.")
    start = time.time()

    with model:
        trace = pm.sample(
            draws=sampler_settings["draws"],
            tune=sampler_settings["tune"],
            cores=sampler_settings["cores"],
            chains=sampler_settings["chains"],
            target_accept=sampler_settings["target_accept"],
            nuts_sampler=sampler_settings["nuts_sampler"],
            idata_kwargs={"log_likelihood": True},  # Compute log likelihood
        )

    runtime = str(timedelta(seconds=int(time.time() - start)))
    logger.info(f"Finished sampling in {runtime}.")

    return trace


def summarize_sampling_statistics(trace: az.InferenceData) -> None:
    """
    Calculates sampling statistics for the model to evaluate the convergence of
    the sampling chains. This includes divergences, acceptance rate, R-hat
    statistics and ESS statistics.

    Args:
        trace: The trace from the NUTS MCMC sampling.
    """
    var_names = list(trace.posterior.data_vars)
    idata = az.convert_to_dataset(trace)  # Avoid doing conversion twice

    # Divergences
    divergences = np.sum(trace.sample_stats["diverging"].values)
    logger.warning(f"There are {divergences} divergences in the sampling chains.")

    # Acceptance rate
    accept_rate = np.mean(trace.sample_stats["acceptance_rate"].values)
    logger.warning(f"The mean acceptance rate was {accept_rate:.3f}")

    # R-hat statistics
    for var in var_names:
        try:
            r_hat = az.summary(idata, var_names=var, round_to=2)["r_hat"]
            mean_r_hat = np.mean(r_hat)
            min_r_hat = np.min(r_hat)
            max_r_hat = np.max(r_hat)
            logger.info(
                f"R-hat for {var} are: {mean_r_hat:.3f} (mean) | "
                f"{min_r_hat:.3f} (min) | {max_r_hat:.3f} (max)"
            )
        except KeyError:
            continue

    # ESS statistics
    for var in var_names:
        try:
            ess = az.summary(idata, var_names=var, round_to=2)["ess_bulk"]
            mean_ess = np.mean(ess)
            min_ess = np.min(ess)
            max_ess = np.max(ess)
            logger.info(
                f"ESS for {var} are: {int(mean_ess)} (mean) | {int(min_ess)} "
                f"(min) | {int(max_ess)} (max)"
            )
        except KeyError:
            continue
