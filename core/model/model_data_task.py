import json
import os
import time
from datetime import timedelta
from typing import Any

import pandas as pd
import polars as pl
from box import Box
from scipy.special import logit
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

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
    folds before saving the output dataframes.
    """

    def __init__(self, run_folder_path: str, mode: str) -> None:
        """
        Attributes:
            - run_folder_path: Path to folder where all outputs are stored.
        """
        # General
        self.run_folder_path: str = run_folder_path
        self.mode: str = mode
        self.site_info_vars: list[str] = configs.site_info_vars

        # Configs related to scope and data resolution
        self.diversity_type: str = configs.diversity_type
        self.taxonomic_resolution: str = configs.run_settings[
            configs.run_settings.model_type
        ].taxonomic_resolution
        self.input_data_path: str = feature_configs.diversity_metrics.output_data_paths[
            self.diversity_type
        ][self.taxonomic_resolution]
        self.group_vars: list[str] = (
            configs.group_vars.basic
            + configs.group_vars.taxonomic[self.taxonomic_resolution]
            + configs.group_vars.biogeographic
        )
        self.group_size_threshold: int = configs.run_settings.group_size_threshold
        self.threshold_on_groups: list[str] = configs.run_settings.threshold_on_groups
        self.biogeographic_scope: dict[str, Any] = configs.data_scope.biogeographic
        self.taxonomic_scope: dict[str, Any] = configs.data_scope.taxonomic

        # Model specific configs
        model_vars = configs.model_variables[configs.run_settings.model_variables]
        self.response_var: str = model_vars.response_var
        self.response_var_transform: str = model_vars.response_var_transform
        self.requires_intensity_data: bool = model_vars.requires_intensity_data
        self.categorical_vars: list[str] = model_vars.categorical_vars
        self.continuous_vars: list[str] = model_vars.continuous_vars
        self.interaction_cols: list[str] = model_vars.interaction_cols

        # If running cross-validation
        if self.mode == "crossval":
            self.cv_settings: dict[str, Any] = configs.cv_settings

    def run_task(self) -> None:
        """
        Generate one or several dataframes for model training or cross-validation.
        The mode is either 'training', in which one dataframe is created, or
        'crossval', in which one dataframe per fold is created.
        """
        logger.info(
            f"Initiating model data preparation for mode '{self.mode}' "
            f"and diversity type '{self.diversity_type}'."
        )
        start = time.time()

        # Load the input data
        validate_input_files(file_paths=[self.input_data_path])
        df = pl.read_parquet(self.input_data_path)

        # If required, filter data based on biogeographic and species scope
        if self.biogeographic_scope["include_all"]:
            logger.info("Complete biogeographic scope; no filtering on biomes.")
        else:
            df = self.filter_biogeographic_scope(df)

        if self.taxonomic_scope["include_all"]:
            logger.info("Complete taxonomic scope; no filtering on species groups.")
        else:
            df = self.filter_taxonomic_scope(df)

        # Filter out cases where land use info is missing (type and intensity)
        df = self.filter_out_unknown_lui(df)

        # Filter out small groups if a threshold has been specified
        if self.group_size_threshold > 0:
            df = self.filter_out_small_groups(df)

        # If specified in the config, the response variable is transformed
        if self.response_var_transform is not None:
            df, self.transformed_response_var = self.transform_response_variable(df)
        else:
            self.transformed_response_var = self.response_var  # type: ignore

        # Create interaction terms between categorical and continuous vars
        # NOTE: Should also interact population and road density
        if self.interaction_cols is not None:  # If list of cols is not empty
            df, interaction_terms = self.create_interaction_terms(df)

        # Define the columns to keep for model training and cross-validation
        all_model_vars = (
            self.group_vars
            + [self.transformed_response_var]
            + self.categorical_vars
            + self.continuous_vars
            + interaction_terms
        )

        # Save site-level information to a separate file, for later analysis
        all_site_info_vars = self.group_vars + self.site_info_vars
        df_site_info = df.select(all_site_info_vars)
        site_info_path = os.path.join(self.run_folder_path, "site_info.parquet")
        validate_output_files(
            file_paths=[site_info_path], files=[df_site_info], allow_overwrite=False
        )
        df_site_info.write_parquet(site_info_path)

        # Save interaction terms to a JSON file for later use
        interaction_terms_path = os.path.join(
            self.run_folder_path, "interaction_terms.json"
        )
        with open(interaction_terms_path, "w") as f:
            json.dump(interaction_terms, f)

        # Create mapping between site names and indices for later use
        site_names = df["SSBS"].unique().to_list()
        site_name_to_idx = {site_name: idx for idx, site_name in enumerate(site_names)}
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
            df_train_list, df_test_list = self.generate_cv_folds(df)
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

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Model data preparation finished in {runtime}.")

    def filter_biogeographic_scope(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter the DataFrame based on the defined biogeographic scope."""
        logger.info("Filtering data based on biogeographic scope.")
        nb_sites_before = get_unique_value_count(df, column="SSBS")

        filter = self.biogeographic_scope["filtering_logic"]  # include or exclude
        filter_col = "Biome"
        filter_values = self.biogeographic_scope["filtering_dicts"][filter_col]
        df_res = ModelDataTask._filter_data_scope(
            df,
            filtering_logic=filter,
            filtering_column=filter_col,
            filtering_values=filter_values,
        )

        nb_sites_after = get_unique_value_count(df_res, column="SSBS")
        logger.info(
            f"Filtering based on '{filter_col}' column completed | "
            f"Number of sites dropped: {nb_sites_before - nb_sites_after}."
        )

        return df_res

    def filter_taxonomic_scope(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter the DataFrame based on the defined taxonomic scope."""
        logger.info("Filtering data based on taxonomic scope.")
        nb_sites_before = get_unique_value_count(df, column="SSBS")

        filter = self.biogeographic_scope["filtering_logic"]  # include or exclude
        filter_col = self.taxonomic_resolution
        filter_values = self.biogeographic_scope["filtering_dicts"][filter_col]
        df_res = ModelDataTask._filter_data_scope(
            df,
            filtering_logic=filter,
            filtering_column=filter_col,
            filtering_values=filter_values,
        )

        nb_sites_after = get_unique_value_count(df_res, column="SSBS")
        logger.info(
            f"Filtering based on '{filter_col}' column completed | "
            f"Number of sites dropped: {nb_sites_before - nb_sites_after}."
        )

        return df_res

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
        logger.info(
            f"Filtering for unknown land use completed | "
            f"Number of sites dropped: {nb_sites_before - nb_sites_after}."
        )

        return df_res

    def filter_out_small_groups(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter out biogeographic or other groups that are too small to be
        included in the model. The filtering is applied to all columns
        specified in the `self.threshold_on_groups` attribute.

        Args:
            df: DataFrame with the species abundance data.

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
            f"Filtering out small groups completed | "
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
            - method: Transformation method to apply.

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
            original_col_index = df.columns.index(response_var)
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
        columns and continuous covariates at different resolutions.

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

    def generate_cv_folds(
        self, df: pl.DataFrame
    ) -> tuple[list[pl.DataFrame], list[pl.DataFrame]]:
        """
        Create indices corresponding to the train and test sets for k-fold
        cross-validation. The function supports random, spatial, and
        environmental cross-validation strategies. Clustering is used to create
        the spatial and environmental folds.

        Args:
            - df: DataFrame with the model data to split.

        Returns:
            - df_train_list: List of DataFrames for the training sets.
            - df_test_list: List of DataFrames for the test sets.
        """
        # Unpack settings
        seed = self.cv_settings["random_seed"]
        k = self.cv_settings["k"]
        strategy = self.cv_settings["strategy"]
        stratify_groups = self.cv_settings["stratify_groups"]
        clustering_method = self.cv_settings["clustering_method"]
        if strategy != "random":
            clustering_vars = self.cv_settings["clustering_vars"][strategy]

        logger.info(
            f"Generating {k} CV folds using strategy '{strategy}' "
            f"and stratified on '{stratify_groups}'."
        )

        # Convert Polars DataFrame to pandas for processing
        df = df.to_pandas()

        # Create stratification column
        df = ModelDataTask._create_stratification_column(df, stratify_groups)
        df["Cluster"] = None

        # For spatial and environmental CV, do clustering using k-means or GM
        if strategy in ["spatial", "environmental"]:
            logger.info(
                f"Performing clustering based on {strategy} variables "
                f"using method '{clustering_method}'."
            )
            # Perform clustering within each stratify group (e.g. Biome)
            for group in df["Stratify_group"].unique():
                df_group = df[df["Stratify_group"] == group]
                clustered_group = ModelDataTask._perform_clustering(
                    df_group,
                    method=clustering_method,
                    n_clusters=k,
                    features=clustering_vars,
                    seed=seed,
                    group=group,
                )
                df.loc[df["Stratify_group"] == group, "Cluster"] = clustered_group[
                    "Cluster"
                ]

        # For random CV, create "Cluster" column based on stratified K-Fold
        elif strategy == "random":
            kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
            for i, (_, test_idx) in enumerate(
                kfold.split(X=df, y=df["Stratify_group"])
            ):
                df.loc[test_idx, "Cluster"] = i

        # Generate train/test splits using cluster IDs
        df_train_list, df_test_list = [], []
        for cluster_id in sorted(df["Cluster"].unique()):
            df_test = df[df["Cluster"] == cluster_id]
            df_train = df[df["Cluster"] != cluster_id]

            # Convert back to Polars
            df_train_list.append(pl.DataFrame(df_train))
            df_test_list.append(pl.DataFrame(df_test))

        logger.info("Finished generating k-folds for cross-validation.")

        return df_train_list, df_test_list

    @staticmethod
    def _create_stratification_column(
        df: pd.DataFrame, stratify_groups: list[str]
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
        if len(stratify_groups) > 1:
            df["Stratify_group"] = df[stratify_groups].astype(str).agg("_".join, axis=1)
        else:
            df["Stratify_group"] = df[stratify_groups[0]]
        logger.info("Finished creating stratification column.")
        return df

    @staticmethod
    def _perform_clustering(
        df: pd.DataFrame,
        method: str,
        n_clusters: int,
        features: list[str],
        seed: int,
        group: str,
    ) -> pd.DataFrame:
        """
        Perform clustering on the specified features and assign cluster labels.

        Args:
            - df: DataFrame containing data to be clustered.
            - method: Clustering method ("kmeans" or "gmm").
            - n_clusters: Number of clusters.
            - features: List of column names to use as features for clustering.
            - seed: Random seed for reproducibility.
            - group: Name of the unique stratification group.

        Returns:
            - df: DataFrame with an added "Cluster" column for cluster labels.
        """
        logger.info(f"Performing clustering for group {group}.")

        # Extract feature data for clustering
        feature_data = df[features].to_numpy()

        # Initialize the clustering model
        if method == "kmeans":
            cluster_model = KMeans(n_clusters=n_clusters, random_state=seed)
        elif method == "gmm":
            cluster_model = GaussianMixture(n_components=n_clusters, random_state=seed)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # Fit the model and assign cluster labels
        df.loc[:, "Cluster"] = cluster_model.fit_predict(feature_data)

        logger.info("Clustering completed.")

        return df
