import json
import os
import time
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from box import Box
from sklearn.model_selection import StratifiedGroupKFold

from core.model.model_utils import get_scope_counts
from core.tests.shared.validate_shared import (
    get_unique_value_count,
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
model_config_path = os.environ.get(
    "MODEL_CONFIG_PATH", os.path.join(script_dir, "model_configs.yaml")
)
configs = Box.from_yaml(filename=model_config_path)

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
            - exclude_outliers: Boolean indicating whether to exclude outliers
                using the IQR / Tukey rule.
            - iqr_multiplier: Multiplier for the IQR to define outliers.
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
            - min_sites_per_study: Minimum number of sites per study for the
                study to be included in the model.
            - scope_taxonomic_resolution: Taxonomic resolution used to define
                final scope when running all-species data.

        # Model specific configs:
            - model_type: Type of model to be trained or cross-validated. One
                of 'bayesian' or 'glmm'.
            - model_vars: List of covariates to be used in the model, used to
                access specific sets of coviariates further down.
            - response_var: Name of the response variable.
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
        self.random_seed = configs.random_seed
        self.diversity_type: str = configs.data_scope.diversity_type
        self.taxonomic_resolution: str = configs.data_scope.taxonomic.resolution
        self.sub_sampling_settings: dict[str, Any] = configs.data_scope.sub_sampling
        self.min_sites_per_study: int = configs.data_scope.min_sites_per_study
        self.min_ref_sites_for_beta: int = configs.data_scope.min_ref_sites_for_beta
        self.exclude_outliers: bool = configs.data_scope.exclude_outliers
        self.iqr_multiplier: float = configs.data_scope.iqr_multiplier
        self.biogeographic_scope: dict[str, Any] = configs.data_scope.biogeographic
        self.taxonomic_scope: dict[str, Any] = configs.data_scope.taxonomic
        self.taxonomic_filtering_scope: str = getattr(
            configs.data_scope.taxonomic, "filtering_scope", "Custom"
        )

        self.all_species_data_path: str = feature_configs.diversity_metrics[
            self.diversity_type
        ].output_data_paths["All_species"]
        self.input_data_path: str = feature_configs.diversity_metrics[
            self.diversity_type
        ].output_data_paths[self.taxonomic_resolution]

        self.group_vars: list[str] = (
            configs.group_vars.basic
            + configs.group_vars.taxonomic[self.taxonomic_resolution]
            + configs.group_vars.biogeographic
        )

        # Model specific configs
        self.model_type = configs.run_settings.model_type
        model_vars = configs.model_variables[configs.run_settings.model_variables]
        self.response_var: str = model_vars.response_var
        self.requires_intensity_data: bool = model_vars.requires_intensity_data
        self.categorical_vars: list[str] = model_vars.categorical_vars
        self.continuous_vars: list[str] = model_vars.continuous_vars
        self.interaction_cols: list[str] = model_vars.interaction_cols

        if self.model_type == "bayesian":  # To create mapping for the hierarchy
            model_run_settings = configs.run_settings[self.model_type]
            self.rolled_up_predictions: bool = model_run_settings[
                "rolled_up_predictions"
            ]
            self.min_studies_per_group: int = model_run_settings[
                "min_studies_per_group"
            ]
            self.hierarchy: dict[str, list[str]] = model_run_settings["hierarchy"]

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

        # Load data used for filtering and final scope determination
        validate_input_files(file_paths=[self.all_species_data_path])
        df_all_species = pl.read_parquet(self.all_species_data_path)
        df_filtering = df_all_species.clone()

        if self.taxonomic_resolution == "All_species":
            df = df_all_species.clone()
        else:
            validate_input_files(file_paths=[self.input_data_path])
            df = pl.read_parquet(self.input_data_path)

        if self.taxonomic_resolution == "All_species":
            scope_resolution = self.taxonomic_filtering_scope
            logger.info(
                "Using scope taxonomic resolution "
                f"'{scope_resolution}' to finalize scope."
            )
            if scope_resolution == "All_species":
                df_scope = df_all_species.clone()
            else:
                scope_paths = feature_configs.diversity_metrics[
                    self.diversity_type
                ].output_data_paths
                if scope_resolution not in scope_paths:
                    raise ValueError(
                        "Scope taxonomic resolution is not available in "
                        "feature configs: "
                        f"{scope_resolution}."
                    )
                scope_path = scope_paths[scope_resolution]
                validate_input_files(file_paths=[scope_path])
                df_scope = pl.read_parquet(scope_path)
        else:
            df_scope = df

        # If there are NaNs, nulls or inf remaining after interpolation in the
        # feature generation task, those rows need to be dropped
        df_filtering = self.drop_invalid_rows(df_filtering)

        # If required, filter data based on biogeographic and species scope
        if self.biogeographic_scope["include_all"]:
            logger.info("Complete biogeographic scope, no filtering done.")
        else:
            df_filtering = self.filter_biogeographic_scope(df_filtering)

        if self.taxonomic_scope["include_all"]:
            logger.info("Complete taxonomic scope, no filtering on species groups.")
        else:
            df_filtering = self.filter_taxonomic_scope(df_filtering)

        # Filter out cases where land use info is missing (type and intensity)
        df_filtering = self.filter_out_unknown_lui(df_filtering)

        # Filter outliers
        if self.exclude_outliers:
            df_filtering = self.filter_outliers(df_filtering)

        # Filter out small studies if a threshold has been specified
        if self.min_sites_per_study > 0:
            df_filtering = self.filter_out_small_studies(df_filtering)

        # For beta, we have a specific filter on the nb of reference sites
        if self.diversity_type == "beta" and self.min_ref_sites_for_beta > 0:
            df_filtering = self.filter_out_few_reference_sites(df_filtering)

        # Share of all data to include in training / cross-validation
        # This is usually 1, but can be set to a smaller value for testing
        if self.sub_sampling_settings["fraction"] < 1.0:
            if self.diversity_type == "alpha":
                df_filtering = self.sample_observations_or_studies(df_filtering)

            elif self.diversity_type == "beta":
                df_filtering = self.subsample_beta_pairs_balanced(df_filtering)

        # Build a scope manifest from the filtered all-species data
        if self.diversity_type == "alpha":
            scope_cols = ["SSBS"]
            scope_label = "sites"
        else:
            scope_cols = ["SSBS", "Primary_minimal_site"]
            scope_label = "site pairs"

        scope_manifest = df_filtering.select(scope_cols).unique()
        logger.info(
            f"Scope manifest includes {scope_manifest.height} {scope_label} "
            "from filtered all-species data."
        )
        scope_manifest_path = os.path.join(
            self.run_folder_path, "scope_manifest.parquet"
        )
        validate_output_files(
            file_paths=[scope_manifest_path],
            files=[scope_manifest],
            allow_overwrite=False,
        )
        scope_manifest.write_parquet(scope_manifest_path)

        # Filter the scope dataframe using the scope manifest
        df_scope = df_scope.join(scope_manifest, on=scope_cols, how="semi")
        df_scope = self.drop_invalid_rows(df_scope)

        final_scope_manifest = df_scope.select(scope_cols).unique()
        missing_scopes = scope_manifest.join(
            final_scope_manifest, on=scope_cols, how="anti"
        )
        if missing_scopes.height > 0:
            logger.warning(
                f"{missing_scopes.height} {scope_label} dropped after applying "
                "the scope manifest and invalid-value filtering."
            )
        final_scope_manifest_path = os.path.join(
            self.run_folder_path, "final_scope_manifest.parquet"
        )
        validate_output_files(
            file_paths=[final_scope_manifest_path],
            files=[final_scope_manifest],
            allow_overwrite=False,
        )
        final_scope_manifest.write_parquet(final_scope_manifest_path)

        if self.taxonomic_resolution == "All_species":
            df = df_filtering.join(final_scope_manifest, on=scope_cols, how="semi")
        else:
            df = df_scope

        # Create interaction terms between categorical and continuous vars
        if self.interaction_cols:  # If list of cols is not empty
            df, interaction_terms = self.create_interaction_terms(df)
        else:
            interaction_terms = []

        # Define the columns to keep for model training and cross-validation
        all_model_vars = (
            self.group_vars
            + [self.response_var]
            + self.categorical_vars
            + self.continuous_vars
            + interaction_terms
        )

        # Save interaction terms to a JSON file, since they are created on the fly
        interaction_terms_path = os.path.join(
            self.run_folder_path, "interaction_terms.json"
        )
        with open(interaction_terms_path, "w") as f:
            json.dump(interaction_terms, f)

        # For Bayesian models, create a fixed, consistent index-name mapping
        # for all hierarchical levels and save this
        hierarchy_cols, rolled_up_cols = None, None  # Initialize to None
        if self.model_type == "bayesian":
            (
                df,
                hierarchy_mapping,
                hierarchy_cols,
                levels,
                label_cols,
                study_counts,
            ) = self.generate_global_hierarchy_mapping(df)

            # Save the mapping to a JSON file
            hierarchy_mapping_path = os.path.join(
                self.run_folder_path, "complete_hierarchy_mapping.json"
            )
            with open(hierarchy_mapping_path, "w") as f:
                json.dump(hierarchy_mapping, f)

            # Update the list of model variables to include the hierarchy cols
            all_model_vars = list(set(all_model_vars + hierarchy_cols))

            # If specified, roll up small hierarchical groups to the next level
            # and save these mapping like above. These will be used to create
            # CV folds and for predictions
            if self.rolled_up_predictions:
                df, rolled_up_mapping, rolled_up_cols = self.apply_hierarchical_rollup(
                    df,
                    levels,
                    label_cols,
                    study_counts,
                )
                rolled_up_mapping_path = os.path.join(
                    self.run_folder_path, "rolled_up_hierarchy_mapping.json"
                )
                with open(rolled_up_mapping_path, "w") as f:
                    json.dump(rolled_up_mapping, f)

                all_model_vars = list(set(all_model_vars + rolled_up_cols))

        # For the Bayesian hierarchical model, we additionally need a fixed
        # mapping between site names and index numbers
        if self.model_type == "bayesian":
            site_names = df.get_column("SSBS").unique().to_list()
            site_name_to_idx = {
                site_name: idx for idx, site_name in enumerate(site_names)
            }
            site_mapping_path = os.path.join(self.run_folder_path, "site_mapping.json")
            with open(site_mapping_path, "w") as f:
                json.dump(site_name_to_idx, f)

            # For Bayesian hierarchical models with taxonomic groupings,
            # also need a mapping between taxon names and index numbers
            if self.taxonomic_resolution != "All_species":
                if self.taxonomic_resolution == "Custom":
                    taxon_names = (
                        df.get_column("Custom_taxonomic_group").unique().to_list()
                    )

                elif self.taxonomic_resolution != "All_species":
                    taxon_names = (
                        df.get_column(self.taxonomic_resolution).unique().to_list()
                    )

                # Generate the mapping and save to JSON
                taxon_name_to_idx = {
                    taxon_name: idx for idx, taxon_name in enumerate(taxon_names)
                }
                taxon_mapping_path = os.path.join(
                    self.run_folder_path, "taxon_mapping.json"
                )
                with open(taxon_mapping_path, "w") as f:
                    json.dump(taxon_name_to_idx, f)

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

        # Add hierarchical columns if previously generated
        if hierarchy_cols:
            all_site_info_vars = list(set(all_site_info_vars + hierarchy_cols))
        if rolled_up_cols:
            all_site_info_vars = list(set(all_site_info_vars + rolled_up_cols))

        # Create dataframe with auxiliary site info
        # Create dataframe with auxiliary site info
        df_site_info = df.select(all_site_info_vars)
        if self.taxonomic_resolution != "All_species":
            taxonomic_col = (
                self.taxonomic_resolution
                if self.taxonomic_resolution != "Custom"
                else "Custom_taxonomic_group"
            )
            df_site_info = df_site_info.unique(
                subset=["SSBS", taxonomic_col], keep="first"
            )
        else:
            df_site_info = df_site_info.unique(subset=["SSBS"], keep="first")

        # For cross-validation, also add the stratification information
        # (even if this overlaps with the final rolled up hierarchical column)
        if self.mode == "crossval":
            df_site_info = df_site_info.join(df_strata, on="SSBS", how="left")

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

        Args:
            - df: Input Polars DataFrame containing all columns to be checked.

        Returns:
            - Dataframe with rows containing NaN, NULL, inf values removed.
        """
        logger.info("Dropping rows with NaN, NULL or inf values.")
        counts_before = get_scope_counts(df, self.diversity_type)

        # Combine all columns to check
        columns_to_check = (
            [self.response_var] + self.categorical_vars + self.continuous_vars
        )

        # Filter columns to ensure only those present in the DataFrame are used
        columns_to_check = [col for col in columns_to_check if col in df.columns]

        # Diagnostic: summarize invalid values per column
        diagnostic_rows = []
        for col in columns_to_check:
            if col not in df.columns:
                continue

            n_null = df.select(pl.col(col).is_null().sum()).item()
            n_nan = df.select(pl.col(col).is_nan().sum()).item()
            n_inf = df.select(pl.col(col).is_infinite().sum()).item()

            n_invalid = n_null + n_nan + n_inf
            if n_invalid > 0:
                diagnostic_rows.append(
                    {
                        "column": col,
                        "n_null": n_null,
                        "n_nan": n_nan,
                        "n_inf": n_inf,
                        "n_invalid": n_invalid,
                    }
                )

        if diagnostic_rows:
            df_diag = pl.DataFrame(diagnostic_rows).sort("n_invalid", descending=True)

            logger.warning(
                "Invalid values detected in response/covariates "
                "(counts per column):\n"
                f"{df_diag}"
            )

        # Drop rows with NULLs in the specified columns
        df_cleaned = df.drop_nulls(subset=columns_to_check)

        # Drop rows with NaNs in the specified columns
        for col in columns_to_check:
            df_cleaned = df_cleaned.filter(~pl.col(col).is_nan())

        # Drop rows with infinite values
        for col in columns_to_check:
            df_cleaned = df_cleaned.filter(~pl.col(col).is_infinite())

        counts_after = get_scope_counts(df_cleaned, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

        return df_cleaned

    def filter_biogeographic_scope(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter the DataFrame based on the defined biogeographic scope."""
        logger.info("Filtering data based on biogeographic scope.")
        counts_before = get_scope_counts(df, self.diversity_type)

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
                nb_studies_before = counts_before["studies"]
                nb_studies_after = get_scope_counts(df_filtered, self.diversity_type)[
                    "studies"
                ]
                logger.info(
                    f"Filtering based on '{filter_col}' column completed. "
                    f"Studies dropped: {nb_studies_before - nb_studies_after}."
                )
                nb_studies_before = nb_studies_after  # Update count

        counts_after = get_scope_counts(df_filtered, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

        return df_filtered

    def filter_taxonomic_scope(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter the DataFrame based on the defined taxonomic scope."""
        logger.info("Filtering data based on taxonomic scope.")
        counts_before = get_scope_counts(df, self.diversity_type)

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
                nb_studies_before = counts_before["studies"]
                nb_studies_after = get_scope_counts(df_filtered, self.diversity_type)[
                    "studies"
                ]
                logger.info(
                    f"Filtering based on '{filter_col}' column completed. "
                    f"Studies dropped: {nb_studies_before - nb_studies_after}."
                )
                nb_studies_before = nb_studies_after  # Update count

            elif filter_values:
                logger.warning(
                    f"Skipping filter on {filter_col}, "
                    f"not allowed below resolution {resolution}."
                )

        counts_after = get_scope_counts(df_filtered, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

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
        counts_before = get_scope_counts(df, self.diversity_type)

        # Always remove cases where the land use type is not known
        df_res = df.filter(pl.col("Predominant_land_use") != "Cannot decide")

        counts_after = get_scope_counts(df_res, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

        # Optionally remove cases where the land use intensity is not known
        if self.requires_intensity_data:
            logger.info("Filtering out cases where land use intensity is unknown.")
            counts_before = get_scope_counts(df_res, self.diversity_type)

            df_res = df_res.filter(pl.col("Use_intensity") != "Cannot decide")

            counts_after = get_scope_counts(df_res, self.diversity_type)
            self._log_scope_change(counts_before, counts_after)

        return df_res

    def filter_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out outliers based on the IQR (Interquartile Range) method.
        This is done to avoid that extreme values do not skew the model
        training or cross-validation results. Only high values are filtered
        since the data is lower-bounded by zero.

        Args:
            df: DataFrame containing the data to be filtered.

        Returns:
            df_res: DataFrame with outliers removed.
        """
        response = self.response_var
        multiplier = self.iqr_multiplier
        logger.info(
            f"Filtering out outliers using the IQR method with multiplier {multiplier}."
        )
        counts_before = get_scope_counts(df, self.diversity_type)

        # Calculate IQR and upper bound for each study on all species data
        df_iqr = (
            df.group_by("SS")
            .agg(
                pl.col(response).quantile(0.25).alias("q1"),
                pl.col(response).quantile(0.75).alias("q3"),
            )
            .with_columns((pl.col("q3") - pl.col("q1")).alias("iqr"))
            .with_columns(
                (pl.col("q3") + multiplier * pl.col("iqr")).alias("upper"),
            )
            .select("SS", "upper")
        )

        # Filter using the upper bound and get remaining sites
        df_filtered = df.join(df_iqr, on="SS", how="left").filter(
            pl.col(response) <= pl.col("upper")
        )

        # Filter the original dataframe based on remaining sites or site pairs
        if self.diversity_type == "alpha":
            remaining_sites = df_filtered.get_column("SSBS").unique()
            df_res = df.filter(pl.col("SSBS").is_in(remaining_sites))
        elif self.diversity_type == "beta":
            beta_pair_keys = ["SSBS", "Primary_minimal_site"]
            df_res = df.join(
                df_filtered.select(beta_pair_keys).unique(),
                on=beta_pair_keys,
                how="semi",
            )

        counts_after = get_scope_counts(df_res, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

        return df_res

    def filter_out_small_studies(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out studies that are considered too small in terms of number of
        sites. The size of the study impacts its overall representativeness,
        and has direct implications for normalized biodiversity numbers that
        are pooled when training the model.
        """
        threshold = self.min_sites_per_study
        logger.info(f"Filtering out studies with less than {threshold} sites.")
        counts_before = get_scope_counts(df, self.diversity_type)

        small_studies = (
            df.group_by("SS")
            .agg(pl.count("SSBS").alias("n_sites"))
            .filter(pl.col("n_sites") < threshold)
            .get_column("SS")
            .to_list()
        )

        # Filter out the small groups
        df_res = df.filter(~pl.col("SS").is_in(small_studies))

        counts_after = get_scope_counts(df_res, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

        return df_res

    def filter_out_few_reference_sites(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out studies with too few reference sites for beta diversity.

        Reference sites are rows where 'Primary vegetation_Minimal use' equals 1.
        Only studies with at least `min_ref_sites_for_beta` unique reference
        sites are kept.
        """
        threshold = self.min_ref_sites_for_beta
        logger.info(
            f"Filtering out studies with less than {threshold} reference sites."
        )
        counts_before = get_scope_counts(df, self.diversity_type)

        # Reduce dataframe to only reference sites
        df_ref = df.filter(pl.col("Primary vegetation_Minimal use") == 1)

        # Count number of reference sites per study
        site_counts = df_ref.group_by("SS").agg(
            pl.col("SSBS").n_unique().alias("n_sites")
        )

        # Filter studies with enough reference sites
        eligible_studies = (
            site_counts.filter(pl.col("n_sites") >= threshold)
            .get_column("SS")
            .to_list()
        )

        # Filter original df to include only eligible SS
        df_res = df.filter(pl.col("SS").is_in(eligible_studies))

        counts_after = get_scope_counts(df_res, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

        return df_res

    def sample_observations_or_studies(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Randomly sample a fraction of studies from the DataFrame.

        Args:
            - df: DataFrame containing the data to be sampled.
            - fraction: Fraction of studies to sample (between 0 and 1).

        Returns:
            - df_sampled: Subsampled DataFrame.
        """
        fraction = self.sub_sampling_settings["fraction"]
        sampling_level = self.sub_sampling_settings["level"]
        logger.info(f"Subsampling {fraction:.0%} of the data for model training.")
        counts_before = get_scope_counts(df, self.diversity_type)

        # Sampling done at the study or site level
        if sampling_level == "studies":
            logger.info(f"Sampling {fraction:.0%} of studies.")
            eligible_pool = df.get_column("SS").unique().sort()

        elif sampling_level == "sites":
            logger.info(f"Sampling {fraction:.0%} of sites.")
            eligible_pool = df.get_column("SSBS").unique().sort()

        # Generate indices to keep
        rng = np.random.default_rng(self.random_seed)
        n_keep = int(len(eligible_pool) * fraction)
        keep_ids = rng.choice(eligible_pool, size=n_keep, replace=False)

        if sampling_level == "studies":
            df_sample = df.filter(pl.col("SS").is_in(keep_ids))
        elif sampling_level == "sites":
            df_sample = df.filter(pl.col("SSBS").is_in(keep_ids))
        else:
            raise ValueError(
                "Unsupported sub_sampling level. " "Expected 'studies' or 'sites'."
            )

        counts_after = get_scope_counts(df_sample, self.diversity_type)
        self._log_scope_change(counts_before, counts_after)

        return df_sample

    def subsample_beta_pairs_balanced(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Subsample beta-diversity pairs in a study-balanced, sublinear way.

        - Keeps all studies
        - Larger studies contribute more pairs, sublinearly
        - Controls total number of pairs (~fraction of df)
        """
        fraction = self.sub_sampling_settings["fraction"]
        k_min = self.sub_sampling_settings["min_k_per_study"]
        k_max = self.sub_sampling_settings["max_k_per_study"]

        logger.info(f"Subsampling {fraction:.0%} of the data for model training.")
        counts_before = get_scope_counts(df, self.diversity_type)
        nb_ref_sites_before = get_unique_value_count(df, column="Primary_minimal_site")

        rng = np.random.default_rng(self.random_seed)

        # Compute study-level number of sites, reference sites and weights
        df_stats = (
            df.group_by("SS")
            .agg(
                pl.col("SSBS").n_unique().alias("n_sites"),
                pl.col("Primary_minimal_site").n_unique().alias("n_refs"),
            )
            .with_columns(
                (pl.col("n_sites") * pl.col("n_refs")).alias("potential_pairs")
            )
            .with_columns(pl.col("potential_pairs").sqrt().alias("weight"))
            .sort("SS")
        )

        # Parameters for normalizing weights to target total number of pairs
        total_weight = df_stats.get_column("weight").sum()
        nb_site_pairs_before = df.select(["SSBS", "Primary_minimal_site"]).n_unique()
        target_pairs = int(nb_site_pairs_before * fraction)
        logger.info(f"Number of site-pairs before subsampling: {nb_site_pairs_before}.")
        logger.info(f"Target total number of pairs after subsampling: {target_pairs}.")

        sampled_parts: list[pl.DataFrame] = []

        # Iterate over all studies
        for row in df_stats.iter_rows(named=True):
            study = row["SS"]
            weight = row["weight"]

            # Target number of pairs for this study
            k_study = int(target_pairs * weight / total_weight)
            k_study = max(k_min, min(k_study, k_max))

            # Get study-level data from original dataframe
            df_study = df.filter(pl.col("SS") == study)
            sites = df_study.get_column("SSBS").unique().sort()
            ref_sites = df_study.get_column("Primary_minimal_site").unique().sort()

            n_sites = len(sites)
            n_refs = len(ref_sites)

            # Determine how many sites and reference sites to keep
            n_sites_keep = min(n_sites, max(1, int(round(np.sqrt(k_study)))))
            n_refs_keep = min(n_refs, max(1, int(np.ceil(k_study / n_sites_keep))))

            # Randomly choose sites and reference sites to keep
            chosen_sites = rng.choice(sites, size=n_sites_keep, replace=False)
            chosen_ref_sites = rng.choice(ref_sites, size=n_refs_keep, replace=False)

            df_sub = df_study.filter(
                pl.col("SSBS").is_in(chosen_sites)
                & pl.col("Primary_minimal_site").is_in(chosen_ref_sites)
            )

            # If we still have too many pairs, subsample down to k_study
            if df_sub.height > k_study:
                df_sub = df_sub.sample(
                    n=k_study,
                    with_replacement=False,
                    seed=self.random_seed,
                )

            sampled_parts.append(df_sub)
            del df_sub, df_study  # Free memory

        df_sampled = pl.concat(sampled_parts)

        counts_after = get_scope_counts(df_sampled, self.diversity_type)
        nb_ref_sites_after = get_unique_value_count(
            df_sampled, column="Primary_minimal_site"
        )
        nb_site_pairs_after = df_sampled.select(
            ["SSBS", "Primary_minimal_site"]
        ).n_unique()
        self._log_scope_change(counts_before, counts_after)
        logger.info(
            f"Number of reference sites dropped: "
            f"{nb_ref_sites_before - nb_ref_sites_after}. "
            f"Before: {nb_ref_sites_before}, after: {nb_ref_sites_after}.\n"
            f"Number of pairs / target pairs: "
            f"{nb_site_pairs_after / target_pairs:.3f}."
        )

        return df_sampled

    def create_interaction_terms(
        self,
        df: pl.DataFrame,
    ) -> tuple[pl.DataFrame, list]:
        """
        Creates interaction terms between land-use related (categorical)
        columns and continuous covariates at different resolutions. This is
        based on the BII immplementation in De Palma 2021.

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
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Build hierarchy labels, global group indices, and parent mappings.

        Returns:
            - df: Input dataframe with any newly created hierarchy label columns.
            - mapping: Dictionary with global group indices, parent mappings,
                hierarchy column names, and per-group study counts.
            - new_cols: New hierarchy label columns added to df.
            - levels: Hierarchy levels that were actually used.
            - label_cols: Mapping from level name to corresponding label column.
            - study_counts: Per-level dataframe with number of studies per group.
        """
        logger.info("Defining mappings between hierarchical groups.")
        hierarchy = self.hierarchy

        # Construct full labels for each hierarchy level
        all_cols = []
        new_cols = []
        label_cols = {}
        valid_levels = {level: cols for level, cols in hierarchy.items() if cols}
        levels = list(valid_levels.keys())

        for level, cols in valid_levels.items():
            all_cols += cols
            col_name = "_".join(all_cols)
            label_cols[level] = col_name
            if col_name not in df.columns:
                new_cols.append(col_name)
                df = df.with_columns(
                    pl.concat_str([pl.col(c) for c in all_cols], separator="_").alias(
                        col_name
                    )
                )

        # Map each hierarchical group to a unique, global index
        mapping: dict[str, Any] = {"column_names": {}}
        study_counts = {}
        for level in levels:
            col_name = label_cols[level]

            # Drop rows with null or NaN for this hierarchy column
            before_rows = df.height
            df = df.filter(
                pl.col(col_name).is_not_null()
                & (pl.col(col_name).str.strip_chars().str.len_chars() > 0)
                & (pl.col(col_name).str.to_lowercase() != "null")
            )
            after_rows = df.height
            dropped = before_rows - after_rows
            if dropped > 0:
                logger.warning(
                    "Filtered out %s rows with null/NaN values in '%s' " "for %s.",
                    dropped,
                    col_name,
                    level,
                )

            mapping["column_names"][level] = col_name
            unique_values = df.get_column(col_name).unique().to_list()
            mapping[level] = {value: idx for idx, value in enumerate(unique_values)}

            # Count number of studies per group, for use in priors and roll up
            group_study_counts = (
                df.select([pl.col(col_name), pl.col("SS")])
                .group_by(col_name)
                .agg(pl.col("SS").n_unique().alias("n_studies"))
            )
            study_counts[level] = group_study_counts
            mapping[f"{level}_n_studies"] = dict(
                zip(
                    group_study_counts.get_column(col_name),
                    group_study_counts.get_column("n_studies"),
                )
            )

            # Create parent-child mappings for levels 2 and 3
            if level == "level_2":
                pairs = df.select(
                    [label_cols["level_1"], label_cols["level_2"]]
                ).unique()
                mapping["level_2_parents"] = dict(
                    zip(
                        pairs.get_column(label_cols["level_2"]),
                        pairs.get_column(label_cols["level_1"]),
                    )
                )
            if level == "level_3":
                pairs = df.select(
                    [label_cols["level_2"], label_cols["level_3"]]
                ).unique()
                mapping["level_3_parents"] = dict(
                    zip(
                        pairs.get_column(label_cols["level_3"]),
                        pairs.get_column(label_cols["level_2"]),
                    )
                )

        return df, mapping, new_cols, levels, label_cols, study_counts

    def apply_hierarchical_rollup(
        self,
        df: pl.DataFrame,
        levels: list[str],
        label_cols: dict[str, str],
        study_counts: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Roll small hierarchy groups up to broader levels for prediction.

        Groups are assigned from the most specific level upwards based on
        `min_studies_per_group`, with a final population-level fallback for any
        still-unassigned rows.
        """
        if self.rolled_up_predictions and self.min_studies_per_group < 2:
            raise ValueError(
                "Hierarchical roll-up is enabled, but min_studies_per_group "
                "is set to less than 2. This is not allowed."
            )

        logger.info("Applying hierarchical roll-up to small groups.")
        nb_initial_groups = df.get_column(label_cols[levels[-1]]).unique().len()
        logger.info(f"Initial number of groups: {nb_initial_groups}.")
        min_studies = self.min_studies_per_group

        # Step 1: Initialize Final group and level
        df = df.with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("Final_hierarchical_group"),
                pl.lit(None, dtype=pl.Utf8).alias("Final_hierarchical_level"),
            ]
        )

        # Step 2: Roll-up loop (deepest to shallowest)
        for level in reversed(levels):
            label_name = label_cols[level]
            group_study_counts = study_counts[level]
            df = df.join(group_study_counts, on=label_name, how="left")

            # Assign group if unassigned and above threshold
            mask = pl.col("Final_hierarchical_group").is_null() & (
                pl.col("n_studies") >= min_studies
            )
            df = df.with_columns(
                [
                    pl.when(mask)
                    .then(pl.col(label_name))
                    .otherwise(pl.col("Final_hierarchical_group"))
                    .alias("Final_hierarchical_group"),
                    pl.when(mask)
                    .then(pl.lit(level))
                    .otherwise(pl.col("Final_hierarchical_level"))
                    .alias("Final_hierarchical_level"),
                ]
            ).drop("n_studies")

        # Step 3: Fallback to population-level
        group_study_counts = (
            df.select(["Final_hierarchical_group", "SS"])
            .unique()
            .group_by("Final_hierarchical_group")
            .agg(pl.count("SS").alias("actual_study_count"))
        )
        logger.info("Applying population-level fallback for groups below threshold.")
        df = df.join(group_study_counts, on="Final_hierarchical_group", how="left")
        df = df.with_columns(
            [
                pl.when(
                    pl.col("Final_hierarchical_group").is_null()
                    | (pl.col("actual_study_count") < min_studies)
                )
                .then(pl.lit("Population"))
                .otherwise(pl.col("Final_hierarchical_group"))
                .alias("Final_hierarchical_group"),
                pl.when(
                    pl.col("Final_hierarchical_level").is_null()
                    | (pl.col("actual_study_count") < min_studies)
                )
                .then(pl.lit("Population"))
                .otherwise(pl.col("Final_hierarchical_level"))
                .alias("Final_hierarchical_level"),
            ]
        ).drop("actual_study_count")
        post_counts = (
            df.select(["Final_hierarchical_group", "SS"])
            .unique()
            .group_by("Final_hierarchical_group")
            .agg(pl.count("SS").alias("n_studies"))
            .sort("n_studies")
        )
        low_post = post_counts.filter(pl.col("n_studies") < min_studies)
        if low_post.height > 0:
            logger.warning(
                "Hierarchical roll-up warning: groups below threshold remain.\n"
                f"{low_post}"
            )

        null_group = df.filter(pl.col("Final_hierarchical_group").is_null())
        if null_group.height > 0:
            logger.warning(
                "Hierarchical roll-up warning: Final_hierarchical_group contains nulls."
            )
        null_level = df.filter(pl.col("Final_hierarchical_level").is_null())
        if null_level.height > 0:
            logger.warning(
                "Hierarchical roll-up warning: Final_hierarchical_level contains nulls."
            )

        # Step 4: Flag rolled-up rows
        most_specific = levels[-1]
        df = df.with_columns(
            (pl.col("Final_hierarchical_level") != most_specific)
            .cast(pl.Int8)
            .alias("Rolled_up")
        )

        nb_final_groups = df.get_column("Final_hierarchical_group").unique().len()
        logger.info(f"Final number of groups: {nb_final_groups}.")

        # Generate global indices and mappings for final rolled up groups
        mapping = self._generate_mapping_from_final_groups(
            df, levels=levels, label_cols=label_cols
        )

        new_cols = ["Final_hierarchical_group", "Final_hierarchical_level", "Rolled_up"]

        return df, mapping, new_cols

    @staticmethod
    def _generate_mapping_from_final_groups(
        df: pl.DataFrame,
        levels: list[str],
        label_cols: dict[str, str],
    ) -> dict:
        """
        Create hierarchy mappings from rolled-up final groups.

        This includes per-level group indices, population fallback mapping, and
        parent-child mappings across hierarchy levels.
        """
        logger.info("Generating mapping from final rolled-up groups.")
        mapping: dict[str, Any] = {"column_names": {}}

        # Step 1: Base group mappings for each level
        for level in levels:
            groups = (
                df.filter(pl.col("Final_hierarchical_level") == level)
                .get_column("Final_hierarchical_group")
                .unique()
                .to_list()
            )
            mapping[level] = {g: i for i, g in enumerate(sorted(groups))}
            mapping["column_names"][level] = label_cols[level]

        # Step 2: Add population if present
        if "Population" in df.get_column("Final_hierarchical_level").unique():
            mapping["Population"] = {"None": 0}
            mapping["column_names"]["Population"] = "Population"

        # Step 3: Create parent mappings and ensure parent groups exist
        if "level_2" in mapping and "level_1" in mapping:
            l2_to_l1 = (
                df.filter(pl.col("Final_hierarchical_level") == "level_2")
                .select(["Final_hierarchical_group", label_cols["level_1"]])
                .unique()
            )
            parents = l2_to_l1.get_column(label_cols["level_1"])
            children = l2_to_l1.get_column("Final_hierarchical_group")

            # Inject missing parents into level_1 mapping with unique indices
            used_indices = set(mapping["level_1"].values())
            next_idx = max(used_indices, default=-1) + 1

            for parent in parents:
                if parent not in mapping["level_1"]:
                    logger.warning(
                        f"Missing L1 parent {parent}. Injecting into mapping."
                    )
                    mapping["level_1"][parent] = next_idx
                    next_idx += 1

            mapping["level_2_parents"] = dict(zip(children, parents))

        if "level_3" in mapping and "level_2" in mapping:
            l3_to_l2 = (
                df.filter(pl.col("Final_hierarchical_level") == "level_3")
                .select(["Final_hierarchical_group", label_cols["level_2"]])
                .unique()
            )
            parents = l3_to_l2.get_column(label_cols["level_2"])
            children = l3_to_l2.get_column("Final_hierarchical_group")

            # Inject missing parents into level_2 mapping with unique indices
            used_indices = set(mapping["level_2"].values())
            next_idx = max(used_indices, default=-1) + 1

            for parent in parents:
                if parent not in mapping["level_2"]:
                    logger.warning(
                        f"Missing L2 parent {parent}. Injecting into mapping."
                    )
                    mapping["level_2"][parent] = next_idx
                    next_idx += 1

            mapping["level_3_parents"] = dict(zip(children, parents))
            print(mapping)

        return mapping

    def generate_cv_folds(
        self, df: pl.DataFrame
    ) -> tuple[list[pl.DataFrame], list[pl.DataFrame], pl.DataFrame]:
        """
        Create indices corresponding to the train and test sets for k-fold
        cross-validation using stratified splits.

        NOTE: In the current implementation, splitting at the study level
        results in studies appearing in multiple folds, in case a study spans
        multiple strata.

        Args:
            - df: DataFrame with the model data to split.

        Returns:
            - df_train_list: List of DataFrames for the training sets.
            - df_test_list: List of DataFrames for the test sets.
        """
        # Unpack settings
        seed = self.random_seed
        k = self.cv_settings["k"]  # Number of folds to generate
        split_level = self.cv_settings["split_level"]  # Split by study or site
        # Groups to use for stratification
        stratification_cols = [
            col for col in self.cv_settings["stratification_cols"] if col is not None
        ]
        logger.info(f"Generating {k} cross-validation folds.")

        # Key columns for validation of input and output
        is_alpha = self.diversity_type == "alpha"
        site_key, site_label, site_count_key = (
            ("SSBS", "sites", "sites")
            if is_alpha
            else ("site_pair", "site pairs", "site_pairs")
        )
        df_key_base = df
        if not is_alpha:
            df_key_base = df.with_columns(
                pl.concat_str(
                    [pl.col("SSBS"), pl.col("Primary_minimal_site")], separator="_"
                ).alias("site_pair")
            )

        taxon_cols = self._resolve_taxon_cols(df_key_base)
        key_expr = pl.concat_str(
            [pl.col(site_key)] + [pl.col(col) for col in taxon_cols],
            separator="_",
        )
        df_keyed = df_key_base.with_columns(key_expr.alias("check_key"))
        expected_unique = {
            "SS": set(df_keyed.get_column("SS").unique().to_list()),
            "site_key": set(df_keyed.get_column(site_key).unique().to_list()),
            "check_key": set(df_keyed.get_column("check_key").unique().to_list()),
        }
        expected_counts = get_scope_counts(
            df_keyed,
            self.diversity_type,
            site_key=site_key,
            check_key="check_key",
        )
        logger.info(
            "Input counts: "
            f"{expected_counts['studies']} studies, "
            f"{expected_counts[site_count_key]} {site_label}, "
            f"{expected_counts['check_key']} check keys, "
            f"{expected_counts['observations']} observations."
        )
        # Compatibility with sklearn methods
        df: pd.DataFrame = df.to_pandas()  # type: ignore
        df["Fold_id"] = np.nan  # Initialize fold ID column

        # Create stratification column and generate splits at study or site level
        df = self._create_stratification_column(df, strata=stratification_cols)
        if stratification_cols:
            logger.info("Stratification keys: %s", ", ".join(stratification_cols))
        else:
            logger.info("Stratification keys: none (single stratum)")

        # Check strata sizes against k to flag undersized groups
        group_col = "SS" if split_level == "study" else "SSBS"
        group_label = "studies" if split_level == "study" else "sites"
        non_null_strata = df.loc[df["Stratum"].notna()]
        stratum_counts = non_null_strata.groupby("Stratum")[group_col].nunique()
        num_small = int((stratum_counts < k).sum())
        if num_small > 0:
            logger.warning(
                "Strata with fewer than %s %s: %s.", k, group_label, num_small
            )

        if split_level == "study":
            logger.info("Splitting on study level.")
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            split_gen = splitter.split(X=df, y=df["Stratum"], groups=df["SS"])
        else:
            logger.info("Splitting on site level.")
            splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
            split_gen = splitter.split(X=df, y=df["Stratum"], groups=df["SSBS"])

        # Assign fold IDs
        for fold_id, (_, test_idx) in enumerate(split_gen):
            df.loc[test_idx, "Fold_id"] = fold_id

        # Final check: Stratum should not contain null/NaN values
        stratum_nulls = df["Stratum"].isna()
        if stratum_nulls.any():
            null_rows = df.loc[
                stratum_nulls, ["SS", "SSBS", "Stratum"]
            ].drop_duplicates()
            logger.warning(
                "Stratum contains null/NaN values after fold assignment. "
                f"Rows affected: {len(null_rows)}."
            )
            logger.warning(
                f"Rows with null Stratum:\n{null_rows.to_string(index=False)}"
            )

        fold_id_nulls = df["Fold_id"].isna()
        if fold_id_nulls.any():
            null_count = int(fold_id_nulls.sum())
            logger.warning(
                f"{null_count} rows have null Fold_id values after fold assignment."
            )

        # Generate train/test splits using fold IDs
        df_train_list, df_test_list = [], []
        for fold_id in sorted(df["Fold_id"].dropna().unique()):
            df_test = df[df["Fold_id"] == fold_id]
            df_train = df[df["Fold_id"] != fold_id]

            # Convert back to Polars
            df_train_list.append(pl.DataFrame(df_train))
            df_test_list.append(pl.DataFrame(df_test))

        # Check for overlaps, duplicates, and other inconsistencies
        self._validate_folds(
            df_train_list,
            df_test_list,
            split_level,
            self.diversity_type,
            expected_unique,
            expected_counts,
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

        if not strata:
            df["Stratum"] = "all"
        elif len(strata) == 1:
            df["Stratum"] = df[strata[0]]
        else:
            df["Stratum"] = df[strata].astype(str).agg("_".join, axis=1)

        logger.info("Finished creating stratification column.")

        return df

    def _validate_folds(
        self,
        df_train_list: list[pl.DataFrame],
        df_test_list: list[pl.DataFrame],
        split_level: str,
        diversity_type: str,
        expected_unique: dict[str, set],
        expected_counts: dict[str, int],
    ) -> None:
        """
        Validate folds for overlaps, duplicates, and completeness.
        """
        if not df_train_list or not df_test_list:
            logger.warning("No folds available to validate.")
            return

        sample_df = df_train_list[0] if df_train_list else df_test_list[0]
        taxon_cols = self._resolve_taxon_cols(sample_df)

        if diversity_type == "alpha":
            site_key = "SSBS"
            site_label = "sites"
        else:
            site_key = "site_pair"
            site_label = "site pairs"
            df_train_list = [
                df.with_columns(
                    pl.concat_str(
                        [pl.col("SSBS"), pl.col("Primary_minimal_site")], separator="_"
                    ).alias("site_pair")
                )
                for df in df_train_list
            ]
            df_test_list = [
                df.with_columns(
                    pl.concat_str(
                        [pl.col("SSBS"), pl.col("Primary_minimal_site")], separator="_"
                    ).alias("site_pair")
                )
                for df in df_test_list
            ]

        check_key_cols = [site_key] + taxon_cols
        check_key_label = " + ".join(check_key_cols)
        logger.info(f"Validating folds using check key: {check_key_label}.")

        def add_check_key(df: pl.DataFrame) -> pl.DataFrame:
            if len(check_key_cols) == 1:
                key_expr = pl.col(check_key_cols[0])
            else:
                key_expr = pl.concat_str(
                    [pl.col(col) for col in check_key_cols], separator="_"
                )
            return df.with_columns(key_expr.alias("check_key"))

        df_train_list = [add_check_key(df) for df in df_train_list]
        df_test_list = [add_check_key(df) for df in df_test_list]

        def overlap_count(
            df_left: pl.DataFrame, df_right: pl.DataFrame, key: str
        ) -> int:
            return (
                df_left.select(key)
                .unique()
                .filter(pl.col(key).is_in(df_right.get_column(key)))
                .height
            )

        def _null_or_nan_count(df: pl.DataFrame, col: str) -> int:
            mask = pl.col(col).is_null()
            dtype = df.schema.get(col)
            if dtype in (pl.Float32, pl.Float64):
                mask = mask | pl.col(col).is_nan()
            return df.filter(mask).height

        def _empty_string_count(df: pl.DataFrame, col: str) -> int:
            dtype = df.schema.get(col)
            if dtype != pl.Utf8:
                return 0
            return df.filter(pl.col(col).str.strip_chars() == "").height

        site_count_key = "sites" if site_label == "sites" else "site_pairs"
        for i, (df_train, df_test) in enumerate(zip(df_train_list, df_test_list)):
            train_counts = get_scope_counts(
                df_train,
                diversity_type,
                site_key=site_key,
                check_key="check_key",
            )
            test_counts = get_scope_counts(
                df_test,
                diversity_type,
                site_key=site_key,
                check_key="check_key",
            )
            train_studies = train_counts["studies"]
            test_studies = test_counts["studies"]
            train_sites = train_counts[site_count_key]
            test_sites = test_counts[site_count_key]
            train_obs = train_counts["observations"]
            test_obs = test_counts["observations"]

            logger.info(
                f"Fold {i + 1}: \n"
                f"Train: {train_studies} studies, {train_sites} {site_label}, "
                f"{train_obs} observations. \n"
                f"Test: {test_studies} studies, {test_sites} {site_label}, "
                f"{test_obs} observations. \n"
                f"Study test / train ratio: {(test_studies / train_studies):.3f}. \n"
                f"{site_label.capitalize()} test / train ratio: "
                f"{(test_sites / train_sites):.3f}. \n"
                f"Observation test / train ratio: {(test_obs / train_obs):.3f}."
            )

            check_key_overlap = overlap_count(df_train, df_test, "check_key")
            if check_key_overlap > 0:
                logger.warning(
                    f"Train-test overlap in {check_key_label} in fold {i + 1} "
                    f"({check_key_overlap} keys)."
                )

            if split_level == "study":
                study_overlap = overlap_count(df_train, df_test, "SS")
                if study_overlap > 0:
                    logger.warning(
                        f"Train-test overlap in studies in fold {i + 1} "
                        f"({study_overlap} studies)."
                    )

            for label, df in [("train", df_train), ("test", df_test)]:
                dup_count = df.select("check_key").is_duplicated().sum()
                if dup_count > 0:
                    logger.warning(
                        f"{dup_count} duplicate {check_key_label} keys in "
                        f"{label} fold {i + 1}."
                    )

        df_all_tests = pl.concat(df_test_list)

        def warn_multi_fold_keys(df_all: pl.DataFrame, key: str, label: str) -> None:
            key_fold_counts = (
                df_all.select([key, "Fold_id"])
                .drop_nulls()
                .unique()
                .group_by(key)
                .agg(pl.col("Fold_id").n_unique().alias("fold_count"))
                .filter(pl.col("fold_count") > 1)
            )
            if key_fold_counts.height > 0:
                logger.warning(
                    f"{key_fold_counts.height} {label} appear in multiple test folds."
                )

        warn_multi_fold_keys(df_all_tests, "check_key", f"{check_key_label} keys")
        warn_multi_fold_keys(df_all_tests, site_key, site_label)
        if site_key != "SSBS":
            warn_multi_fold_keys(df_all_tests, "SSBS", "sites")
        if split_level == "study":
            warn_multi_fold_keys(df_all_tests, "SS", "studies")

        observed_counts = get_scope_counts(
            df_all_tests,
            diversity_type,
            site_key=site_key,
            check_key="check_key",
        )
        logger.info(
            "Final counts: "
            f"{observed_counts['studies']} studies, "
            f"{observed_counts[site_count_key]} {site_label}, "
            f"{observed_counts['check_key']} check keys, "
            f"{observed_counts['observations']} observations."
        )
        for key, expected_value in expected_counts.items():
            observed_value = observed_counts.get(key)
            if observed_value is None:
                continue
            if observed_value != expected_value:
                logger.warning(
                    f"Count mismatch for {key}: expected {expected_value}, "
                    f"observed {observed_value}."
                )

        observed_unique = {
            "SS": set(df_all_tests.get_column("SS").unique().to_list()),
            "site_key": set(df_all_tests.get_column(site_key).unique().to_list()),
            "check_key": set(df_all_tests.get_column("check_key").unique().to_list()),
        }
        label_map = {
            "SS": "studies",
            "site_key": site_label,
            "check_key": f"{check_key_label} keys",
        }
        for key_name, expected in expected_unique.items():
            observed = observed_unique.get(key_name, set())
            missing = expected - observed
            if missing:
                logger.warning(
                    f"{len(missing)} {label_map.get(key_name, key_name)} "
                    "were not assigned to any fold."
                )

        check_columns = ["SS", site_key, "check_key"]
        if diversity_type != "alpha":
            check_columns.extend(["SSBS", "Primary_minimal_site"])
        check_columns.extend(taxon_cols)
        for col in check_columns:
            if col not in df_all_tests.columns:
                continue
            null_or_nan = _null_or_nan_count(df_all_tests, col)
            if null_or_nan > 0:
                logger.warning(f"{null_or_nan} rows have null/NaN values in {col}.")
            empty_strings = _empty_string_count(df_all_tests, col)
            if empty_strings > 0:
                logger.warning(
                    f"{empty_strings} rows have empty-string values in {col}."
                )

        logger.info("Finished validating k-folds for cross-validation.")

    def _resolve_taxon_cols(self, df: pl.DataFrame) -> list[str]:
        """
        Resolve which taxonomic identifier column(s) should be used as keys.

        Uses the active taxonomic resolution, except when resolution is
        'All_species', where `taxonomic_filtering_scope` determines whether a
        taxon column still needs to be included for validation/keying.
        """
        if self.taxonomic_resolution == "All_species":
            scope_resolution = self.taxonomic_filtering_scope
        else:
            scope_resolution = self.taxonomic_resolution
        if scope_resolution == "All_species":
            return []
        if scope_resolution == "Custom":
            col = "Custom_taxonomic_group"
        else:
            col = scope_resolution
        return [col] if col in df.columns else []

    def _log_scope_change(self, before: dict[str, int], after: dict[str, int]) -> None:
        """
        Log changes in studies, observations, and sites or site pairs.
        """
        logger.info(
            f"Number of studies dropped: {before['studies'] - after['studies']}. "
            f"Before: {before['studies']}, after: {after['studies']}.\n"
            f"Number of observations dropped: "
            f"{before['observations'] - after['observations']}. "
            f"Before: {before['observations']}, "
            f"after: {after['observations']}."
        )
        if self.diversity_type == "alpha":
            logger.info(
                f"Number of sites dropped: {before['sites'] - after['sites']}. "
                f"Before: {before['sites']}, after: {after['sites']}."
            )
        else:
            logger.info(
                "Number of site-pairs dropped: "
                f"{before['site_pairs'] - after['site_pairs']}. "
                f"Before: {before['site_pairs']}, "
                f"after: {after['site_pairs']}."
            )
