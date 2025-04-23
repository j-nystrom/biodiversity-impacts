import json
import os
import time
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from box import Box
from scipy.special import logit
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from core.tests.shared.validate_shared import (
    get_unique_value_count,
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "model_configs.yaml"))

# This task also needs to access data paths from 'feature_configs.yaml'
project_dir = os.path.dirname(script_dir)  # Navigate up to the project root
feature_config_path = os.path.join(project_dir, "features", "feature_configs.yaml")
feature_configs = Box.from_yaml(filename=feature_config_path)

logger = create_logger(__name__)


class ModelDataTask:
    """
    Task to prepare data in a format suitable for model training or cross-
    validation. This step precedes ModelTrainingTask and CrossValidationTask.
    Consequently, there are two data preparation modes, 'training' and
    'crossval'. The majority of processing steps are the same for both modes,
    but the latter includes an additional step to generate cross-validation
    folds and saving these as separate dataframes.
    """

    def __init__(self, run_folder_path: str, mode: str) -> None:
        """
        Attributes:

        General:
            - run_folder_path: Path to folder where all outputs are stored.
            - mode: Either 'training' or 'crossval'.
            - site_info_vars: List of variables to include in the site info
                dataframe that is saved for later analysis.

        Configs related to scope and data resolution
            - diversity_type: Diversity metric to be used ('alpha' or 'beta').
            - sample_fraction: Fraction of data to include in the run, used for
                scaling tests.
            - biogeographic_scope: Biogeographic scope in modeling, either
                global or a subset based on biomes, realms, or ecoregions.
            - taxonomic_scope: Taxonomic scope in modeling, either global or
                a subset based on taxonomic groups, matching the taxonomic
                resolution below.
            - taxonomic_resolution: Taxonomic resolution in modeling, either
                all species present per site, or more granular taxonomic
                breakdown per site.
            - input_data_path: Path to the input data matching the diversity
                type and taxonomic resolution.
            - group_vars: List of variables to group by in the model.
            - group_size_threshold: Minimum number of observations per group.
            - threshold_on_groups: List of categories to apply the group size
                threshold to, e.g. Biomes and Realms.

        # Model specific configs:
            - model_type: Type of model to be trained or cross-validated. One
                of 'bayesian', 'lmm', 'random_forest'.
            - model_vars: List of covariates to be used in the model, used to
                access specific sets of coviariates further down.
            - response_var: Name of the response variable.
            - response_var_transform: Transformation to apply to the response
                variable, if applicable. One of 'adjust', 'sqrt', 'logit' or
                None. 'adjust' is a small adjustment to avoid exact 0 and 1
                values, required for the beta likelihood function.
            - requires_intensity_data: Boolean indicating whether the model
                requires land-use intensity data.
            - categorical_vars: List of categorical land use variables.
            - continuous_vars: List of continuous covariates, that include e.g.
                population and road density, bioclimatic and topographic
                variables, etc.
            - interaction_cols: List of continuous covariates to interact with
                categorical land use variables.
            - hierarchy: Dictionary indicating which columns should be used to
                group the data into hierarchical levels (up to 3). Each level
                is a list of column names.

        # If running cross-validation
            - cv_settings: Dictionary containing cross-validation settings,
                including the random seed, number of folds, stratification
                variables, if data should be split on sites or studies, etc.
        """
        # General
        self.run_folder_path: str = run_folder_path
        self.mode: str = mode
        self.site_info_vars: list[str] = configs.site_info_vars

        # Configs related to scope and data resolution
        self.diversity_type: str = configs.data_scope.diversity_type
        self.taxonomic_resolution: str = configs.data_scope.taxonomic.resolution
        self.sample_fraction: float = configs.data_scope.sample_fraction
        self.biogeographic_scope: dict[str, Any] = configs.data_scope.biogeographic
        self.taxonomic_scope: dict[str, Any] = configs.data_scope.taxonomic
        self.input_data_path: str = feature_configs.diversity_metrics[
            self.diversity_type
        ].output_data_paths[self.taxonomic_resolution]
        self.group_vars: list[str] = (
            configs.group_vars.basic
            + configs.group_vars.taxonomic[self.taxonomic_resolution]
            + configs.group_vars.biogeographic
        )
        self.group_size_threshold: int = configs.data_scope.group_size_threshold
        self.threshold_on_groups: list[str] = configs.data_scope.threshold_on_groups

        # Model specific configs
        self.model_type = configs.run_settings.model_type
        model_vars = configs.model_variables[configs.run_settings.model_variables]
        self.response_var: str = model_vars.response_var
        self.response_var_transform: str = model_vars.response_var_transform
        self.requires_intensity_data: bool = model_vars.requires_intensity_data
        self.categorical_vars: list[str] = model_vars.categorical_vars
        self.continuous_vars: list[str] = model_vars.continuous_vars
        self.interaction_cols: list[str] = model_vars.interaction_cols

        if (
            self.model_type == "bayesian" or self.model_type == "random_forest"
        ):  # To create mapping for the hierarchy
            self.hierarchy: dict[str, list[str]] = configs.run_settings[
                self.model_type
            ].hierarchy

        # If running cross-validation
        if self.mode == "crossval":
            self.cv_settings: dict[str, Any] = configs.cv_settings

    def run_task(self) -> None:
        """
        Generate one or several dataframes for model training or cross-validation.
        The mode is either 'training', in which one dataframe is created, or
        'crossval', in which one dataframe per train and test fold is created.
        """
        logger.info(
            f"Initiating model data preparation for mode '{self.mode}' "
            f"and diversity type '{self.diversity_type}'."
        )
        start = time.time()

        # Load the input data
        validate_input_files(file_paths=[self.input_data_path])
        df = pl.read_parquet(self.input_data_path)

        # If there are NaNs, nulls or inf remaining after interpolation in the
        # feature generation task, those rows need to be dropped
        df = self.drop_invalid_rows(df)

        # If required, filter data based on biogeographic and species scope
        if self.biogeographic_scope["include_all"]:
            logger.info("Complete biogeographic scope, no filtering done.")
        else:
            df = self.filter_biogeographic_scope(df)

        if self.taxonomic_scope["include_all"]:
            logger.info("Complete taxonomic scope, no filtering on species groups.")
        else:
            df = self.filter_taxonomic_scope(df)

        # Filter out cases where land use info is missing (type and intensity)
        df = self.filter_out_unknown_lui(df)

        # Filter out small groups if a threshold has been specified
        if self.group_size_threshold > 0:
            df = self.filter_out_small_groups(df)

        # Share of all data to include in training / cross-validation
        # This is usually 1, but can be set to a smaller value for testing
        df = df.sample(fraction=self.sample_fraction, shuffle=True)

        # If specified in the config, the response variable is transformed
        if self.response_var_transform:  # If not None (null)
            df, self.transformed_response_var = self.transform_response_variable(df)
        else:
            self.transformed_response_var = self.response_var

        # Create interaction terms between categorical and continuous vars
        # TODO: Should be moved to the feature generation pipeline
        if self.interaction_cols:  # If list of cols is not empty
            df, interaction_terms = self.create_interaction_terms(df)
        else:
            interaction_terms = []

        # Define the columns to keep for model training and cross-validation
        all_model_vars = (
            self.group_vars
            + [self.transformed_response_var]
            + self.categorical_vars
            + self.continuous_vars
            + interaction_terms
        )

        # Save interaction terms to a JSON file, since they are created on the fly
        # TODO: Obsolete when moved to the feature generation pipeline
        interaction_terms_path = os.path.join(
            self.run_folder_path, "interaction_terms.json"
        )
        with open(interaction_terms_path, "w") as f:
            json.dump(interaction_terms, f)

        # For Bayesian and random forest models, create a fixed, consistent
        # index-name mapping for all hierarchical levels and save this
        if self.model_type == "bayesian" or self.model_type == "random_forest":
            df, hierarchy_cols, hierarchy_mapping = (
                self.generate_global_hierarchy_mapping(df)
            )
            hierarchy_mapping_path = os.path.join(
                self.run_folder_path, "hierarchy_mapping.json"
            )
            all_model_vars = list(set(all_model_vars + hierarchy_cols))
            with open(hierarchy_mapping_path, "w") as f:
                json.dump(hierarchy_mapping, f)

        # For this Bayesian hierarchical model, we additionally need a mapping
        # between site names and indices
        if self.model_type == "bayesian":
            site_names = df["SSBS"].unique().to_list()
            site_name_to_idx = {
                site_name: idx for idx, site_name in enumerate(site_names)
            }
            site_mapping_path = os.path.join(self.run_folder_path, "site_mapping.json")
            with open(site_mapping_path, "w") as f:
                json.dump(site_name_to_idx, f)

        # If the goal is to generate training data, we are done here
        if self.mode == "training":
            df_model = df.select(all_model_vars)
            traning_path = os.path.join(self.run_folder_path, "training_data.parquet")
            validate_output_files(
                file_paths=[traning_path], files=[df_model], allow_overwrite=False
            )
            df_model.write_parquet(traning_path)

        # Extra processing to generate cross-validation folds
        elif self.mode == "crossval":
            df_train_list, df_test_list, df_strata = self.generate_cv_folds(df)
            for fold_idx, (df_train, df_test) in enumerate(
                zip(df_train_list, df_test_list)
            ):
                train_path = os.path.join(
                    self.run_folder_path, f"train_fold_{fold_idx + 1}.parquet"
                )
                test_path = os.path.join(
                    self.run_folder_path, f"test_fold_{fold_idx + 1}.parquet"
                )
                df_train = df_train.select(all_model_vars)
                df_test = df_test.select(all_model_vars)

                validate_output_files(
                    file_paths=[train_path, test_path],
                    files=[df_train, df_test],
                    allow_overwrite=False,
                )
                df_train.write_parquet(train_path)
                df_test.write_parquet(test_path)

        # For both train and cross-validation models, save site-level
        # information to a separate file, for later analysis
        all_site_info_vars = self.group_vars + self.site_info_vars
        df_site_info = df.select(all_site_info_vars)

        # For cross-validation, also add the stratification information
        if self.mode == "crossval":
            df_site_info = df_site_info.join(df_strata, on="SSBS")

        site_info_path = os.path.join(self.run_folder_path, "site_info.parquet")
        validate_output_files(
            file_paths=[site_info_path], files=[df_site_info], allow_overwrite=False
        )
        df_site_info.write_parquet(site_info_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Model data preparation finished in {runtime}.")

    def drop_invalid_rows(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Drop rows containing NaN, NULL or inf values in the specified columns.

        Parameters:
        - df: Input Polars DataFrame containing all columns to be checked.

        Returns:
        - A Polars DataFrame with rows containing NaN, NULL, inf values removed.
        """
        logger.info("Dropping rows with NaN, NULL or inf values.")
        nb_sites_before = get_unique_value_count(df, column="SSBS")

        # Combine all columns to check
        columns_to_check = (
            [self.response_var] + self.categorical_vars + self.continuous_vars
        )

        # Filter columns to ensure only those present in the DataFrame are used
        columns_to_check = [col for col in columns_to_check if col in df.columns]

        # Drop rows with NULLs in the specified columns
        df_cleaned = df.drop_nulls(subset=columns_to_check)

        # Drop rows with NaNs in the specified columns
        df_cleaned = df_cleaned.drop_nans(subset=columns_to_check)

        # Drop rows with infinite values
        for col in columns_to_check:
            df_cleaned = df_cleaned.filter(~pl.col(col).is_infinite())

        nb_sites_after = get_unique_value_count(df, column="SSBS")
        logger.info(f"Number of sites dropped: {nb_sites_before - nb_sites_after}.")

        return df_cleaned

    def filter_biogeographic_scope(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter the DataFrame based on the defined biogeographic scope."""
        logger.info("Filtering data based on biogeographic scope.")
        nb_sites_before = get_unique_value_count(df, column="SSBS")

        # Retrieve filtering parameters
        filter_logic = self.biogeographic_scope[
            "filtering_logic"
        ]  # "include" or "exclude"
        filtering_dicts = self.biogeographic_scope["filtering_dicts"]

        # Apply filters dynamically for all specified columns, iterating over
        # different biogeographic levels
        df_filtered = df
        for filter_col, filter_values in filtering_dicts.items():
            if filter_values:  # Only apply filter if values are specified
                df_filtered = self._filter_data_scope(
                    df_filtered,
                    filtering_logic=filter_logic,
                    filtering_column=filter_col,
                    filtering_values=filter_values,
                )
                nb_sites_after = get_unique_value_count(df_filtered, column="SSBS")
                logger.info(
                    f"Filtering based on '{filter_col}' column completed. "
                    f"Sites dropped: {nb_sites_before - nb_sites_after}."
                )
                nb_sites_before = nb_sites_after  # Update count for subsequent filters

        return df_filtered

    def filter_taxonomic_scope(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter the DataFrame based on the defined taxonomic scope."""
        logger.info("Filtering data based on taxonomic scope.")
        nb_sites_before = get_unique_value_count(df, column="SSBS")

        # Retrieve filtering configuration
        filter_logic = self.taxonomic_scope["filtering_logic"]
        filtering_dicts = self.taxonomic_scope["filtering_dicts"]
        resolution = self.taxonomic_scope["resolution"]

        # If resolution is 'All_species', skip filtering entirely
        if resolution == "All_species":
            logger.info("No taxonomic filtering applied (resolution is 'All_species').")
            return df

        # Get the hierarchy order from config keys
        taxonomic_hierarchy = list(filtering_dicts.keys())
        resolution_index = taxonomic_hierarchy.index(resolution)
        allowed_filter_cols = taxonomic_hierarchy[: resolution_index + 1]

        df_filtered = df
        for filter_col, filter_values in filtering_dicts.items():
            if filter_values and filter_col in allowed_filter_cols:
                df_filtered = self._filter_data_scope(
                    df_filtered,
                    filtering_logic=filter_logic,
                    filtering_column=filter_col,
                    filtering_values=filter_values,
                )
                nb_sites_after = get_unique_value_count(df_filtered, column="SSBS")
                logger.info(
                    f"Filtering based on '{filter_col}' column completed. "
                    f"Sites dropped: {nb_sites_before - nb_sites_after}."
                )
                nb_sites_before = nb_sites_after  # Update count for subsequent filters

            elif filter_values:
                logger.warning(
                    f"Skipping filter on {filter_col}, "
                    f"not allowed below resolution {resolution}."
                )

        return df_filtered

    @staticmethod
    def _filter_data_scope(
        df: pl.DataFrame,
        filtering_logic: str,
        filtering_column: str,
        filtering_values: list[str],
    ) -> pl.DataFrame:
        """
        Filter dataframe based on specific column-value inclusion or exclusion
        logic. Specifically, it is used to filter data based on biogeographic
        and taxonomic scope.

        Args:
            - df: The input dataframe to filter.
            - filtering_logic: Filtering approach to use, either "include" or
                "exclude".
            - filtering_column: The column to filter on.
            - filtering_values: Values in that column to include or exclude.

        Returns:
            - df_res: A filtered dataframe based on specified scope and logic.

        Raises:
            - ValueError: If `filtering_logic` is not 'include' or 'exclude'.
        """
        if filtering_logic == "include":
            df_res = df.filter(pl.col(filtering_column).is_in(filtering_values))
        elif filtering_logic == "exclude":
            df_res = df.filter(~pl.col(filtering_column).is_in(filtering_values))
        else:
            raise ValueError(
                f"Invalid filtering logic: {filtering_logic}. "
                "Must be 'include' or 'exclude'."
            )

        return df_res

    def filter_out_unknown_lui(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out observations where land-use type or intensity is unknown.
        Unknown land use type is always filtered out, while the intensity can
        be optionally filtered out.

        Args:
            df: Dataframe with combined data from PREDICTS and other sources.

        Returns:
            df_res: Filtered version of the same dataframe.
        """
        logger.info("Filtering out cases where land use is unknown.")
        nb_sites_before = get_unique_value_count(df, column="SSBS")

        # Always remove cases where the land use type is not known
        df_res = df.filter(pl.col("Predominant_land_use") != "Cannot decide")

        # Optionally remove cases where the land use intensity is not known
        if self.requires_intensity_data:
            logger.info("Filtering out cases where land use intensity is unknown.")
            df_res = df_res.filter(pl.col("Use_intensity") != "Cannot decide")

        nb_sites_after = get_unique_value_count(df_res, column="SSBS")
        logger.info(f"Number of sites dropped: {nb_sites_before - nb_sites_after}.")

        return df_res

    def filter_out_small_groups(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out biogeographic or other groups that are too small (in terms)
        of number of sites to be included in the model. The filtering is
        applied to all columns specified in the `self.threshold_on_groups`
        attribute.

        Args:
            df: DataFrame containing the grouping columns used to filter on.

        Returns:
            df_res: DataFrame with small groups removed.
        """
        threshold = self.group_size_threshold
        groups = self.threshold_on_groups
        logger.info(f"Filtering out groups [{groups}] with <{threshold} observations.")
        nb_sites_before = get_unique_value_count(df, column="SSBS")

        for col in groups:
            logger.info(f"Applying threshold filter on '{col}'.")
            # Group by column and count occurrences, get groups below threshold
            df_group_count = df.group_by(col).agg(pl.col("SSBS").count())
            small_groups = (
                df_group_count.filter(pl.col("SSBS") < threshold)
                .get_column(col)
                .unique()
                .to_list()
            )

            # Filter out the small groups
            df_res = df.filter(~pl.col(col).is_in(small_groups))

        nb_sites_after = get_unique_value_count(df_res, column="SSBS")
        logger.info(
            f"Filtering out small groups completed. "
            f"Number of sites dropped: {nb_sites_before - nb_sites_after}."
        )

        return df_res

    def transform_response_variable(self, df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
        """
        Makes adjustments and transformations to the chosen response variable.
        The first adjustment is to avoid exact 0 and 1 values, which are not
        supported by the logit function (or the Beta distribution if that is
        used). The second possible transformation is the square root. The third
        possible transformation is a logit transformation.

        Args:
            - df: Dataframe containing the response variable.
            - response_var: Name of the response variable.
            - method: Transformation method to apply (if any).

        Returns:
            - df_res: Updated df with the transformed response variable as a
                new column.
        """
        response_var = self.response_var
        method = self.response_var_transform
        logger.info(f"Transforming response variable {response_var}.")

        # Small adjustment to align with support for logit / Beta distribution
        adjust = 0.001
        if method == "adjust" or method == "logit":
            df_res = df.with_columns(
                pl.when(pl.col(response_var) < adjust)
                .then(adjust)  # Replace with 0.001
                .when(pl.col(response_var) > (1 - adjust))
                .then(1 - adjust)  # Replace with 0.999
                .otherwise(pl.col(response_var))
                .alias(response_var)
            )

            if method == "adjust":
                transformed_col_name = response_var + "_adjust"
                df_res = df_res.with_columns(
                    pl.col(response_var).alias(transformed_col_name)
                )

            if method == "logit":
                transformed_col_name = response_var + "_logit"
                df_res = df_res.with_columns(
                    pl.col(response_var)
                    .map_elements(lambda x: logit(x))
                    .alias(transformed_col_name)
                )

        # Square root transformation
        elif method == "sqrt":
            transformed_col_name = response_var + "_sqrt"
            df_res = df.with_columns(
                pl.col(response_var).sqrt().alias(transformed_col_name)
            )

        # Replace original column with transformed one, if the name has changed
        if transformed_col_name != response_var:
            original_col_index = df.columns.index(response_var)  # Find position
            new_col = df_res.get_column(transformed_col_name)
            df_res = df_res.drop([response_var, transformed_col_name])
            df_res = df_res.insert_column(index=original_col_index, column=new_col)

        logger.info("Finished transforming response variable.")

        return df_res, transformed_col_name

    def create_interaction_terms(
        self,
        df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list]:
        """
        Creates interaction terms between land-use related (categorical)
        columns and continuous covariates at different resolutions. This is
        based on the BII immplementation in De Palma 2021.
        # TODO: This should be moved to the feature generation pipeline.

        Args:
            - df: Dataframe containing dummy columns for land-use and land-use
            intensity as well the continuous variables to interact with.

        Returns:
            - df_res: Updated df with interaction terms added.
            - new_cols: List of the names of the new interaction columns.
        """
        logger.info("Creating specified interaction terms.")

        new_cols = []
        for col_1 in self.interaction_cols:
            for col_2 in self.categorical_vars:
                df = df.with_columns(
                    (pl.col(col_1) * pl.col(col_2)).alias(f"{col_2} x {col_1}")
                )
                new_cols.append(f"{col_2} x {col_1}")

        logger.info(f"Finished creating interaction terms: {new_cols}.")

        return df, new_cols

    def generate_global_hierarchy_mapping(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, list[str], dict[str, dict[str, int]]]:
        """
        Generate a global mapping for hierarchical levels from a dataframe. The
        method can handle up to 3 hiearchical levels. This provides a consistent
        mapping that is used for all modeling-related tasks (training and
        prediction).

        Args:
            df: The input DataFrame containing the hierarchical data columns.

        Returns:
            mapping: A dictionary where keys are the hierarchy levels and
                values map unique combined categorical values to indices. Also
                includes column names for each level.
        """
        logger.info("Generating global hierarchy mapping.")

        mapping = {"column_names": {}}  # type: ignore
        all_cols = []  # Should hold all hierarchical columns across all levels
        new_cols = []  # Columns added to the modeling dataframe
        # Check which levels contain hierarchical columns to use
        valid_levels = {level: cols for level, cols in self.hierarchy.items() if cols}

        # Iterate through each level in the specified hierarchy
        for level, cols in valid_levels.items():
            all_cols.extend(cols)
            # Name is either a combination of columns, or a single column
            col_name = "_".join(all_cols) if len(all_cols) > 1 else all_cols[0]
            new_cols.append(col_name)

            # Add the (combined) column for this level, if not already present
            if col_name not in df.columns:
                df = df.with_columns(
                    pl.concat_str(
                        [pl.col(col) for col in all_cols], separator="_"
                    ).alias(col_name)
                )

            # Save the (combined) column name for this level
            mapping["column_names"][level] = col_name

            # Create mapping from unique combined values to indices
            unique_values = df.get_column(col_name).unique().to_list()
            mapping[level] = {value: idx for idx, value in enumerate(unique_values)}

            # Create explicit parent-child relationships between levels
            if level == "level_2":
                level_1_col = mapping["column_names"]["level_1"]
                level_2_col = mapping["column_names"]["level_2"]
                level_1_list = (
                    df.select([level_1_col, level_2_col])
                    .unique()
                    .get_column(level_1_col)
                    .to_list()
                )
                level_2_list = (
                    df.select([level_1_col, level_2_col])
                    .unique()
                    .get_column(level_2_col)
                    .to_list()
                )
                mapping["level_2_parents"] = dict(zip(level_2_list, level_1_list))

            if level == "level_3":
                level_2_col = mapping["column_names"]["level_2"]
                level_3_col = mapping["column_names"]["level_3"]
                level_2_list = (
                    df.select([level_2_col, level_3_col])
                    .unique()
                    .get_column(level_2_col)
                    .to_list()
                )
                level_3_list = (
                    df.select([level_2_col, level_3_col])
                    .unique()
                    .get_column(level_3_col)
                    .to_list()
                )
                mapping["level_3_parents"] = dict(zip(level_3_list, level_2_list))

        return df, new_cols, mapping

    def generate_cv_folds(
        self, df: pl.DataFrame
    ) -> tuple[list[pl.DataFrame], list[pl.DataFrame], pl.DataFrame]:
        """
        Create indices corresponding to the train and test sets for k-fold
        cross-validation. The function supports random, spatial, and
        environmental cross-validation strategies. Clustering is used to create
        the spatial and environmental folds.

        TODO: Check if there are libraries that can be used for the spatial
        and environmental CV.

        NOTE: In the current implementation, splitting at the study level
        results in studies appearing in multiple folds, in case a study spans
        multiple strata. This is a simplification that should be evaluated.

        Args:
            - df: DataFrame with the model data to split.

        Returns:
            - df_train_list: List of DataFrames for the training sets.
            - df_test_list: List of DataFrames for the test sets.
        """
        # Unpack settings
        seed = self.cv_settings["random_seed"]
        k = self.cv_settings["k"]  # Number of folds to generate
        split_level = self.cv_settings["split_level"]  # Split by study or site
        strategy = self.cv_settings["strategy"]  # Random, spatial, or environmental
        # Groups to use for stratification. Train-test splits are done in each
        # strata. There can be a minimum number of sites / studies per strata
        strata = [col for col in self.cv_settings["strata"] if col is not None]
        min_studies_per_stratum = self.cv_settings["min_studies_per_stratum"]

        # Variables used for spatial or environmental clustering
        if strategy != "random":
            clustering_vars = self.cv_settings["clustering_vars"][strategy]

            # For spatial and environmental CV with study splits, use
            # study-level mean values of variables for clustering.
            if split_level == "study":
                study_clustering_agg = self.cv_settings["study_clustering_agg"]

        logger.info(
            f"Generating {k} cross-validation folds using strategy '{strategy}' "
            f"and stratified on '{strata}'."
        )

        original_site_count = get_unique_value_count(df, column="SSBS")
        # Compatibility with sklearn methods
        df: pd.DataFrame = df.to_pandas()  # type: ignore
        df["Fold_id"] = np.nan  # Initialize fold ID column, 1 through K

        # Start by creating the strata in which sampling or clustering is done
        if not any(strata):  # If strata columns are empty (although unlikely)
            logger.warning(
                "No stratification groups provided. Treating dataset as one group."
            )
            df["Stratum"] = "all"
        else:
            # Create initial stratification column, will be used as is if there
            # are enough studies in that grouping
            df = self._create_stratification_column(df, strata)

            # Ensure there are enough studies in each stratification group,
            # otherwise roll up to higher levels, to meet the minimum threshold
            df = self._rollup_small_groups(df, strata, min_studies_per_stratum)

        # Standard (random) cross-validation at study or site level
        if strategy == "random":
            logger.info("Generating random (standard) CV folds.")

            if split_level == "study":
                # Study-level splits uses GroupKFold to avoid study overlaps
                # between train and test folds
                group_kfold = GroupKFold(n_splits=k, shuffle=True, random_state=seed)

                for group in df["Stratum"].unique():
                    df_group = df[df["Stratum"] == group]

                    if len(df_group["SS"].unique()) < k:
                        logger.warning(
                            f"Stratum '{group}' has fewer than {k} studies. Skipping."
                        )
                        continue
                    # Get the global indices for later mapping
                    global_idx = df_group.index

                    # Do GroupKFold splitting at the study level
                    for i, (_, local_test_idx) in enumerate(
                        group_kfold.split(X=df_group, groups=df_group["SS"])
                    ):
                        # Map local indices to global indices to assign the
                        # right fold IDs to the original dataframe, which is at
                        # the site level
                        global_test_idx = global_idx[local_test_idx]
                        df.loc[global_test_idx, "Fold_id"] = i

            else:  # Splitting at the site level
                for group in df["Stratum"].unique():
                    df_group = df[df["Stratum"] == group]

                    if len(df_group["SSBS"].unique()) < k:
                        logger.warning(
                            f"Stratum '{group}' has fewer than {k} sites. Skipping."
                        )

                    # StratifiedKFold creates balanced folds across strata
                    # If there are no strata, this is equivalent to the
                    # standard Kfold method
                    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                    for i, (_, test_idx) in enumerate(
                        kfold.split(X=df, y=df["Stratum"])
                    ):
                        df.loc[test_idx, "Fold_id"] = i

        # For spatial and environmental CV, perform clustering
        if strategy in ["spatial", "environmental"]:
            logger.info(f"Performing clustering based on {strategy} variables.")

            # For study-level clustering, aggregate the clustering variables
            # at the study + strata level
            if split_level == "study":
                df_site_level = df.drop("Fold_id", axis=1)  # Keep a copy of original df
                df = df.groupby(["SS"] + strata, as_index=False).agg(
                    study_clustering_agg
                )

            # Perform clustering within each stratify group, study or site
            # This also covers the case with only one stratum ('all')
            for group in df["Stratum"].unique():
                df_group = df[df["Stratum"] == group]

                # Ensure that groupby and agg operations results in unique groups
                duplicates = df_group[df_group.duplicated()]
                if not duplicates.empty:
                    logger.warning(f"Found {len(duplicates)} duplicate rows.")

                # Validate the number of samples in the group
                if df_group.shape[0] < k:
                    logger.warning(
                        f"Group '{group}' has fewer samples ({df_group.shape[0]}) "
                        f"than the number of folds ({k})."
                    )

                # Run the clustering
                clustered_group = self._perform_clustering(
                    df_group,
                    n_clusters=k,
                    features=clustering_vars,
                    seed=seed,
                )
                # Assign cluster labels
                df.loc[df["Stratum"] == group, "Fold_id"] = clustered_group[
                    "Fold_id"
                ].values

            # If applicable, go from study level fold assignments to site level
            if split_level == "study":
                df_study_fold = df[["SS"] + strata + ["Fold_id"]]
                df_merged = df_site_level.merge(
                    df_study_fold, on=["SS"] + strata, how="left"
                )
                df = df_merged

        # Generate train/test splits using cluster IDs
        df_train_list, df_test_list = [], []
        for fold_id in sorted(df["Fold_id"].dropna().unique()):
            df_test = df[df["Fold_id"] == fold_id]
            df_train = df[df["Fold_id"] != fold_id]

            # Convert back to Polars
            df_train_list.append(pl.DataFrame(df_train))
            df_test_list.append(pl.DataFrame(df_test))

        # Check for overlaps, duplicates, and other inconsistencies
        self._validate_folds(
            df_train_list, df_test_list, split_level, original_site_count
        )

        # Create dataframe mapping sites to strata for later analysis
        df_strata = pl.DataFrame(df[["SSBS", "Stratum"]])

        logger.info("Finished generating k-folds for cross-validation.")

        return df_train_list, df_test_list, df_strata

    @staticmethod
    def _create_stratification_column(
        df: pd.DataFrame, strata: list[str]
    ) -> pd.DataFrame:
        """
        Create column for stratification by concatenating specified columns.

        Args:
            - df: DataFrame with the data to stratify.
            - stratify_groups: List of column names to concatenate into one
            stratification column.

        Returns:
            - DataFrame including the new stratification column.
        """
        logger.info("Creating stratification column for k-folds.")

        if len(strata) == 1:
            df["Stratum"] = df[strata[0]]
        else:
            df["Stratum"] = df[strata].astype(str).agg("_".join, axis=1)

        logger.info("Finished creating stratification column.")

        return df

    @staticmethod
    def _rollup_small_groups(
        df: pd.DataFrame, strata: list[str], min_studies: int
    ) -> pd.DataFrame:
        """
        Rolls up hierarchical strata to ensure all groups meet a minimum study
        count threshold. The threshold is at the study level, even if splits
        are done at the site level.
        """
        nb_initial_groups = df["Stratum"].nunique()
        logger.info(f"Initial number of strata: {nb_initial_groups}.")

        def _build_label(row: pd.Series, level: int) -> str:
            """Constructs a label based on the hierarchy up to the given level."""
            return "_".join(str(row[g]) for g in strata[:level])

        def _rollup_row(row: pd.Series, small_groups: pd.Index, level: int) -> str:
            """Roll up a row until its group is no longer 'small'."""
            if row["Stratum"] in small_groups:
                if level != 1:
                    label = _build_label(row, level)
                    return label
                else:
                    return "rest"
            else:
                return row["Stratum"]

        # Start from the most detailed level to the top
        for level in range(len(strata), 0, -1):
            # Count unique studies per group
            group_counts = df.groupby("Stratum")["SS"].nunique()
            small_groups = group_counts[group_counts < min_studies].index

            # If no small groups remain, we're done
            if len(small_groups) == 0:
                break

            # Apply the roll-up logic
            df["Stratum"] = df.apply(
                lambda row: _rollup_row(row, small_groups, level), axis=1
            )

        # Final check: ensure all groups meet the threshold
        group_counts = df.groupby("Stratum")["SS"].nunique()
        smallest_group_count = group_counts.min()
        if smallest_group_count < min_studies:
            logger.warning(
                f"Smallest group(s) still too small ({smallest_group_count})."
            )

        nb_final_groups = df["Stratum"].nunique()
        logger.info(f"Final number of strata: {nb_final_groups}.")

        return df

    @staticmethod
    def _perform_clustering(
        df: pd.DataFrame,
        n_clusters: int,
        features: list[str],
        seed: int,
    ) -> pd.DataFrame:
        """
        Perform clustering on the specified features and assign cluster labels.

        Args:
            - df: DataFrame containing data to be clustered.
            - n_clusters: Number of clusters.
            - features: List of column names to use as features for clustering.
            - seed: Random seed for reproducibility.

        Returns:
            - df: DataFrame with an added "Fold_id" column for cluster labels.
        """
        # Extract feature data for clustering and handle scaling
        feature_data = df[features].to_numpy()
        feature_data = StandardScaler().fit_transform(feature_data)

        # Initialize the clustering model
        cluster_model = KMeans(n_clusters=n_clusters, random_state=seed)

        # Fit the model and assign cluster labels
        df.loc[:, "Fold_id"] = cluster_model.fit_predict(feature_data).astype(int)

        return df

    def _validate_folds(
        self,
        df_train_list: list[pl.DataFrame],
        df_test_list: list[pl.DataFrame],
        split_level: str,
        original_site_count: int,
    ) -> None:
        """
        Validate the generated folds for overlaps, duplicates, and inconsistencies.
        """
        # Check for overlaps between train and test sets
        for i, (df_train, df_test) in enumerate(zip(df_train_list, df_test_list)):
            # Calculate number of unique studies and sites
            train_studies = df_train.get_column("SS").n_unique()
            train_sites = df_train.get_column("SSBS").n_unique()
            test_studies = df_test.get_column("SS").n_unique()
            test_sites = df_test.get_column("SSBS").n_unique()
            logger.info(
                f"Fold {i + 1}: \n"
                f"Train: {train_studies} studies, {train_sites} sites. \n"
                f"Test: {test_studies} studies, {test_sites} sites. \n"
                f"Study test / train ratio: {(test_studies / train_studies):.3f}. \n"
                f"Site test / train ratio: {(test_sites / train_sites):.3f}."
            )

            # Check for overlaps at the site level, which should never occur
            site_overlap = df_train.filter(pl.col("SSBS").is_in(df_test["SSBS"]))
            if site_overlap.shape[0] > 0:
                logger.warning(f"Train-test overlap in sites detected in fold {i + 1}.")

            # If splitting on study-level, check for corresponding overlaps
            if split_level == "study":
                study_overlap = df_train.filter(pl.col("SS").is_in(df_test["SS"]))
                if study_overlap.shape[0] > 0:
                    logger.warning(
                        f"Train-test overlap in studies detected in fold {i}."
                    )

        # Do the same overlap checks but between all test folds
        for i, df_test_1 in enumerate(df_test_list):
            for j, df_test_2 in enumerate(df_test_list):
                if i >= j:
                    continue  # Skip same test sets or already compared pairs
                site_overlap = df_test_1.filter(pl.col("SSBS").is_in(df_test_2["SSBS"]))
                if site_overlap.shape[0] > 0:
                    logger.warning(
                        f"Duplicate sites found between test folds {i} and {j}."
                    )
                if split_level == "study":
                    study_overlap = df_test_1.filter(
                        pl.col("SS").is_in(df_test_2["SS"])
                    )
                    if study_overlap.shape[0] > 0:
                        logger.warning(
                            f"Duplicate studies found between test folds {i} and {j}."
                        )

        # Check for duplicates in each test fold
        for i, df_test in enumerate(df_test_list):
            duplicates = df_test.filter(pl.col("SSBS").is_duplicated())
            if duplicates.shape[0] > 0:
                logger.warning(f"Duplicate sites detected in test fold {i}.")

        # Check for duplicates in each train fold
        for i, df_train in enumerate(df_train_list):
            duplicates = df_train.filter(pl.col("SSBS").is_duplicated())
            if duplicates.shape[0] > 0:
                logger.warning(f"Duplicate sites detected in train fold {i}.")

        all_sites = pl.concat(df_train_list + df_test_list).get_column("SSBS").unique()
        site_diff = len(all_sites) - original_site_count
        if len(all_sites) != original_site_count:
            logger.warning(f"{site_diff} sites were not assigned to any fold.")
