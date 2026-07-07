import logging
import sys
import time
from datetime import timedelta
from typing import Any

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

from core.model.model_utils import (
    standardize_continuous_covariates,
    validate_design_matrix_columns,
)
from core.model.pymc_models import GeneralHierarchicalModel, rolled_up_prediction_model
from core.utils.bayesian_utils import plot_prior_distribution


class BayesianHierarchicalModel:
    """
    Class for training and making predictions with Bayesian (hierarchical)
    models implemented in PyMC.
    """

    def __init__(
        self,
        mode: str,
        random_seed: int,
        epsilon: float,
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        logger: logging.Logger,
        site_name_to_idx: dict[str, int],
        taxon_name_to_idx: dict[str, int],
        hierarchy_mapping: dict[str, list[str]],
        rolled_up_mapping: dict[str, Any] | None = None,  # Only if rolled up groups
        save_predictive_distributions: bool = False,
    ) -> None:
        """
        Attributes:
            mode: Either 'training' or 'crossval'.
            random_seed: Random seed for model reproducibility.
            epsilon: Small value to prevent numerical issues in models, e.g.
                when using beta likelihood.
            model_settings: Bayesian hierarchical model settings from
                model_configs.yaml.
            model_vars: Response variable and covariates for the model.
            logger: Logger for run output.
            run_folder_path: Base path for run outputs and temp files.
        """
        # Model settings
        self.mode = mode
        self.random_seed = random_seed
        self.sampling_seed = random_seed
        self.epsilon = epsilon
        self.model_settings = model_settings
        self.model_vars = model_vars
        self.logger = logger
        self.prior_predictive: az.InferenceData | None = None
        self.prior_parameter_summary: pl.DataFrame | None = None
        self.progressbar: bool = sys.stderr.isatty()
        self.training_model_data: dict[str, Any] | None = None

        # Model covariates
        self.response_var = model_vars["response_var"]
        self.categorical_vars = model_vars["categorical_vars"]
        self.continuous_vars = model_vars["continuous_vars"]
        self.interaction_terms = model_vars["interaction_terms"]

        # Additional settings for Bayesian model
        self.site_name_to_idx: dict[str, int] = site_name_to_idx
        self.taxon_name_to_idx: dict[str, int] = taxon_name_to_idx
        self.hierarchy_mapping: dict[str, Any] = hierarchy_mapping
        self.sampler_settings: dict[str, Any] = self.model_settings["sampler"]
        self.save_predictive_distributions: bool = self.model_settings[
            "save_predictive_distributions"
        ]
        self.save_predictive_distributions = save_predictive_distributions
        if rolled_up_mapping:
            self.rolled_up_mapping: dict[str, Any] = rolled_up_mapping

    def get_training_components(self) -> dict[str, bool]:
        """Return training component switches with legacy defaults."""
        components = self.model_settings.get("training_components", {})
        study_effects = self.model_settings.get("study_effects", {})
        return {
            "ecological": components.get("ecological", True),
            "study_intercept": components.get("study_intercept", True),
            "study_slopes": components.get(
                "study_slopes", bool(study_effects.get("slope_terms", []))
            ),
            "block_intercept": components.get("block_intercept", True),
        }

    def get_prediction_components(
        self,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, bool]:
        """Return prediction component switches with explicit zeroed defaults."""
        settings = settings or self.model_settings
        components = settings.get("prediction_components", {})
        return {
            "ecological": components.get("ecological", True),
            "study_intercept": components.get("study_intercept", False),
            "study_slopes": components.get("study_slopes", False),
            "block_intercept": components.get("block_intercept", False),
        }

    @staticmethod
    def uses_random_components(components: dict[str, bool]) -> bool:
        """Return True if prediction components include study or block effects."""
        return any(
            components[component]
            for component in ["study_intercept", "study_slopes", "block_intercept"]
        )

    @staticmethod
    def without_random_components(components: dict[str, bool]) -> dict[str, bool]:
        """Return prediction components with study and block effects disabled."""
        components_no_random = dict(components)
        for component in ["study_intercept", "study_slopes", "block_intercept"]:
            components_no_random[component] = False
        return components_no_random

    def model_settings_with_prediction_components(
        self,
        prediction_components: dict[str, bool],
    ) -> dict[str, Any]:
        """Return model settings with a prediction-component override."""
        settings = dict(self.model_settings)
        settings["prediction_components"] = dict(prediction_components)
        return settings

    @staticmethod
    def format_component_settings(components: dict[str, bool]) -> str:
        """Format component switches for logging."""
        return ", ".join(f"{name}={enabled}" for name, enabled in components.items())

    def log_component_settings(self) -> None:
        """Log which model components are used for fitting and prediction."""
        training_components = self.get_training_components()
        prediction_components = self.get_prediction_components()
        self.logger.info(
            "BHM fitting components: %s.",
            self.format_component_settings(training_components),
        )
        self.logger.info(
            "BHM prediction components: %s.",
            self.format_component_settings(prediction_components),
        )

    def get_study_slope_terms(self) -> list[str]:
        """Return active study slope terms for the training model."""
        if not self.get_training_components()["study_slopes"]:
            return []
        return list(self.model_settings.get("study_effects", {}).get("slope_terms", []))

    @staticmethod
    def taxon_column_for_dataframe(df: pl.DataFrame) -> str:
        """Return the custom taxonomic grouping column present in model data."""
        if "Custom_taxonomic_group_alt" in df.columns:
            return "Custom_taxonomic_group_alt"
        return "Custom_taxonomic_group"

    def get_fold_level_study_counts(
        self,
        level_key: str,
        reference_df: pl.DataFrame,
    ) -> dict[str, int]:
        """Count studies per global hierarchy group in the reference data."""
        if level_key not in self.hierarchy_mapping["column_names"]:
            return {}

        label_col = self.hierarchy_mapping["column_names"][level_key]
        counts = (
            reference_df.select([label_col, "SS"])
            .unique()
            .group_by(label_col)
            .agg(pl.col("SS").n_unique().alias("n_studies"))
        )
        return dict(zip(counts.get_column(label_col), counts.get_column("n_studies")))

    def get_fold_level_group_counts(
        self,
        level_key: str,
        reference_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Count studies and sites per hierarchy group in the reference fold."""
        label_col = self.hierarchy_mapping["column_names"][level_key]
        count_cols = [label_col, "SS", "SSBS"]
        count_exprs = [
            pl.col("SS").n_unique().alias("n_studies"),
            pl.col("SSBS").n_unique().alias("n_sites"),
        ]
        if "Primary_minimal_site" in reference_df.columns:
            count_cols.append("Primary_minimal_site")
            count_exprs.append(
                pl.col("Primary_minimal_site").n_unique().alias("n_ref_sites")
            )

        return (
            reference_df.select(count_cols)
            .unique()
            .group_by(label_col)
            .agg(count_exprs)
        )

    def apply_fold_rollup(
        self,
        df: pl.DataFrame,
        reference_df: pl.DataFrame,
    ) -> pl.DataFrame:
        """Assign prediction roll-up levels using reference-fold group counts."""
        if not (hasattr(self, "rolled_up_mapping") and self.rolled_up_mapping):
            return df

        levels = [
            level
            for level in ["level_1", "level_2", "level_3"]
            if level in self.hierarchy_mapping.get("column_names", {})
        ]
        if not levels:
            return df.with_columns(
                [
                    pl.lit("Population").alias("Final_hierarchical_group"),
                    pl.lit("Population").alias("Final_hierarchical_level"),
                    pl.lit(1).cast(pl.Int8).alias("Rolled_up"),
                ]
            )

        min_studies = int(self.model_settings["min_studies_per_group"])
        min_sites = int(self.model_settings.get("min_sites_per_group", 0))
        min_ref_sites = int(self.model_settings.get("min_ref_sites_per_group", 0))
        df = df.drop(
            [
                col
                for col in [
                    "Final_hierarchical_group",
                    "Final_hierarchical_level",
                    "Rolled_up",
                ]
                if col in df.columns
            ]
        ).with_columns(
            [
                pl.lit(None, dtype=pl.Utf8).alias("Final_hierarchical_group"),
                pl.lit(None, dtype=pl.Utf8).alias("Final_hierarchical_level"),
            ]
        )

        for level in reversed(levels):
            label_col = self.hierarchy_mapping["column_names"][level]
            counts = self.get_fold_level_group_counts(level, reference_df)
            df = df.join(counts, on=label_col, how="left")
            mask = pl.col("Final_hierarchical_group").is_null() & (
                pl.col("n_studies") >= min_studies
            )
            if min_sites > 0 and "n_sites" in counts.columns:
                mask = mask & (pl.col("n_sites") >= min_sites)
            if min_ref_sites > 0 and "n_ref_sites" in counts.columns:
                mask = mask & (pl.col("n_ref_sites") >= min_ref_sites)
            df = df.with_columns(
                [
                    pl.when(mask)
                    .then(pl.col(label_col))
                    .otherwise(pl.col("Final_hierarchical_group"))
                    .alias("Final_hierarchical_group"),
                    pl.when(mask)
                    .then(pl.lit(level))
                    .otherwise(pl.col("Final_hierarchical_level"))
                    .alias("Final_hierarchical_level"),
                ]
            ).drop(
                [
                    col
                    for col in ["n_studies", "n_sites", "n_ref_sites"]
                    if col in df.columns
                ]
            )

        deepest_level = levels[-1]
        return df.with_columns(
            [
                pl.when(pl.col("Final_hierarchical_group").is_null())
                .then(pl.lit("Population"))
                .otherwise(pl.col("Final_hierarchical_group"))
                .alias("Final_hierarchical_group"),
                pl.when(pl.col("Final_hierarchical_level").is_null())
                .then(pl.lit("Population"))
                .otherwise(pl.col("Final_hierarchical_level"))
                .alias("Final_hierarchical_level"),
            ]
        ).with_columns(
            (pl.col("Final_hierarchical_level") != deepest_level)
            .cast(pl.Int8)
            .alias("Rolled_up")
        )

    def prepare_data(
        self, df_train: pl.DataFrame, df_test: pl.DataFrame
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Standardize covariates and format data for PyMC model."""
        self.logger.info("Preparing data for PyMC model.")
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        validate_design_matrix_columns(df_train, df_test, x_vars)
        # Continuous vars are standardized to have mean zero and unit variance
        df_train_std, df_test_std = standardize_continuous_covariates(
            df_train,
            df_test,
            vars_to_standardize=self.continuous_vars + self.interaction_terms,
        )
        df_train_std = self.apply_fold_rollup(df_train_std, df_train_std)
        df_test_std = self.apply_fold_rollup(df_test_std, df_train_std)

        # Format data for PyMC model
        train_data = self.format_data_for_pymc_model(
            df_train_std,
            reference_df=df_train_std,
        )
        test_data = self.format_data_for_pymc_model(
            df_test_std,
            reference_df=df_train_std,
        )

        return train_data, test_data

    def fit(self, train_data: dict[str, Any], run_sampler: bool = True) -> None:
        """
        Instantiate the PyMC model object, do prior predictive sampling, and
        run the NUTS sampler to fit the model. Also calculate sampling
        statistics to evaluate convergence.

        Args:
            - train_data: Dataframe with the scaled covariates and response
                variable. This is the training data for the model.
        """
        # Initialize the PyMC model and return the training model object
        self.training_model_data = train_data
        self.log_component_settings()
        self.model = GeneralHierarchicalModel(
            settings=self.model_settings, epsilon=self.epsilon
        )
        self.model_instance = self.model.build_training_model(model_data=train_data)
        self.prior_parameter_summary = self.sample_prior_parameter_summary()

        if self.model_settings["prior_predictive_checks"]:
            # Do prior predictive sampling before running the model
            self.logger.info("Running prior predictive sampling.")
            prior_plot_pairs = [
                tuple(pair)
                for pair in self.model_settings["prior_predictive_plot_pairs"]
            ]
            prior_var_names = sorted(
                {
                    variable
                    for category, variable in prior_plot_pairs
                    if category in {"prior", "prior_predictive"}
                    and variable in self.model_instance.named_vars
                }
            )
            self.prior_predictive = pm.sample_prior_predictive(
                draws=1000,
                model=self.model_instance,
                var_names=prior_var_names,
                random_seed=self.sampling_seed,
            )
            plot_prior_distribution(
                self.prior_predictive,
                category_variable_pairs=prior_plot_pairs,
                likelihood=self.model_settings["likelihood"],
            )
            user_input = input("Continue sampling process? (y/n): ")
            if user_input.lower() == "n":
                run_sampler = False
                self.logger.info("Training aborted based on prior predictive checks.")
        else:
            self.logger.info(
                "Skipping prior predictive sampling (prior_predictive_checks=False)."
            )

        # Run NUTS sampler on the model and summarize sampling statistics
        if run_sampler:
            self.trace = self.run_sampling()
            if self.model_settings["sampling_statistics"]:
                self.summarize_sampling_statistics()

    def predict(
        self, prediction_data: dict[str, Any], pred_mode: str
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Make predictions on training or test data, return and store updated
        trace object with posterior predictive samples / predictions.

        Args:
            - prediction_data: Either training or test data, depending on the
                mode of the model.
            - pred_mode: Either 'train' or 'crossval'.

        Returns:
            - df_pred: Dataframe with site names, observed values, and
                predictions.
        """
        prediction_components = self.get_prediction_components()

        if pred_mode == "test":
            prediction_components = self.without_random_components(
                prediction_components
            )
            self.trace = self.make_predictions(
                prediction_data,
                mode=pred_mode,
                prediction_components=prediction_components,
                sample_likelihood=self.save_predictive_distributions,
            )
            df_pred, df_pred_distr = self.create_prediction_dataframe(
                prediction_data,
                mode=pred_mode,
                include_predictive_distribution=self.save_predictive_distributions,
            )
            if self.uses_random_components(self.get_prediction_components()):
                df_pred = df_pred.with_columns(
                    [
                        pl.col("Predicted").alias("Predicted_FE"),
                        pl.col("Reference_pred").alias("Reference_pred_FE"),
                    ]
                )
            return df_pred, df_pred_distr

        if pred_mode == "train" and self.uses_random_components(prediction_components):
            fixed_components = self.without_random_components(prediction_components)
            posterior_trace = self.trace
            self.trace = posterior_trace.copy()
            self.trace = self.make_predictions(
                prediction_data,
                mode=pred_mode,
                prediction_components=fixed_components,
                prediction_label="fixed-effect",
                sample_likelihood=False,
            )
            df_pred_fixed, _ = self.create_prediction_dataframe(
                prediction_data,
                mode=pred_mode,
                include_predictive_distribution=False,
            )

            self.trace = posterior_trace.copy()
            self.trace = self.make_predictions(
                prediction_data,
                mode=pred_mode,
                prediction_components=prediction_components,
                prediction_label="fixed + random-effect",
                sample_likelihood=self.save_predictive_distributions,
            )
            df_pred, df_pred_distr = self.create_prediction_dataframe(
                prediction_data,
                mode=pred_mode,
                include_predictive_distribution=self.save_predictive_distributions,
            )
            df_pred = df_pred.with_columns(
                [
                    pl.col("Predicted").alias("Predicted_RE"),
                    df_pred_fixed.get_column("Predicted").alias("Predicted_FE"),
                    df_pred_fixed.get_column("Reference_pred").alias(
                        "Reference_pred_FE"
                    ),
                ]
            )
            return df_pred, df_pred_distr

        self.trace = self.make_predictions(
            prediction_data,
            mode=pred_mode,
            prediction_components=prediction_components,
            sample_likelihood=self.save_predictive_distributions,
        )

        df_pred, df_pred_distr = self.create_prediction_dataframe(
            prediction_data,
            mode=pred_mode,
            include_predictive_distribution=self.save_predictive_distributions,
        )

        return df_pred, df_pred_distr

    def format_data_for_pymc_model(
        self,
        df: pl.DataFrame,
        reference_df: pl.DataFrame | None = None,
    ) -> dict[str, Any]:
        """
        Format the dataframe for use in PyMC models.

        Args:
            - df: Dataframe with the scaled covariates and response variable.

        Returns:
            - output_dict: Dictionary containing the formatted data for the
                PyMC model.
        """
        self.logger.info("Formatting data for PyMC model.")
        if reference_df is None:
            reference_df = df

        # ----- Hierarchical levels and indices -----
        # Use self.hierarchy_mapping to simplify processing
        hierarchy = self.hierarchy_mapping

        # Initialize outputs
        level_indices = {}
        level_values = {}
        level_n_studies = {}

        # Process hierarchical levels dynamically to create index variables
        for level_key in ["level_1", "level_2", "level_3"]:
            if level_key not in hierarchy["column_names"]:
                level_indices[f"{level_key}_idx"] = np.array([], dtype=np.int32)

            else:
                col_name = hierarchy["column_names"][level_key]
                level_dict = hierarchy.get(level_key, {})

                # Build row-level index array for the local subset
                # .replace() uses that global dict to map labels -> integer indexes
                idx_array = (
                    df.get_column(col_name)
                    .replace(level_dict, return_dtype=pl.Int32)
                    .to_numpy()
                )
                level_indices[f"{level_key}_idx"] = idx_array

                # Create list of group names to use for dims in the model
                group_names = list(level_dict.keys())
                level_values[f"{level_key}_values"] = group_names

                # Use reference-data counts for group-size prior scaling.
                study_count_dict = self.get_fold_level_study_counts(
                    level_key,
                    reference_df,
                )
                level_n_studies[f"{level_key}_n_studies"] = np.array(
                    [study_count_dict.get(label, 1) for label in group_names],
                    dtype=np.int32,
                )

        # Create mapping indices between levels, initialized to None
        level_2_to_level_1_idx = None
        level_3_to_level_2_idx = None

        # Level 2 to level 1
        if "level_1" in hierarchy and "level_2" in hierarchy:
            level_2_parents = hierarchy.get("level_2_parents", {})
            dict_level_2 = hierarchy["level_2"]  # e.g. { "RealmX":0, "RealmY":1, ... }
            dict_level_1 = hierarchy["level_1"]  # e.g. { "BiomeA":0, "BiomeB":1, ... }

            n_l2 = len(dict_level_2)
            level_2_to_level_1_idx = np.zeros(n_l2, dtype=np.int32)

            # For each global level_2 label, find the parent's level_1 label,
            # and then find that parent's index.
            for l2_label, l2_idx in dict_level_2.items():
                parent_label = level_2_parents[l2_label]  # e.g. "BiomeA"
                parent_idx = dict_level_1[parent_label]  # e.g. 0
                level_2_to_level_1_idx[l2_idx] = parent_idx

        if "level_2" in hierarchy and "level_3" in hierarchy:
            level_3_parents = hierarchy.get("level_3_parents", {})
            dict_level_3 = hierarchy["level_3"]
            dict_level_2 = hierarchy["level_2"]

            n_l3 = len(dict_level_3)
            level_3_to_level_2_idx = np.zeros(n_l3, dtype=np.int32)

            for l3_label, l3_idx in dict_level_3.items():
                parent_label = level_3_parents[l3_label]  # e.g. "RealmX"
                parent_idx = dict_level_2[parent_label]
                level_3_to_level_2_idx[l3_idx] = parent_idx

        # Level assignment logic for rolled up predictions
        if hasattr(self, "rolled_up_mapping") and self.rolled_up_mapping:
            level_assignment = (
                df.with_columns(
                    pl.when(pl.col("Final_hierarchical_level") == "level_3")
                    .then(3)
                    .when(pl.col("Final_hierarchical_level") == "level_2")
                    .then(2)
                    .when(pl.col("Final_hierarchical_level") == "level_1")
                    .then(1)
                    .otherwise(0)
                    .alias("level_assignment")
                )
                .get_column("level_assignment")
                .to_numpy()
                .astype(np.int32)
            )
        else:
            level_assignment = None

        # ----- Control variables during sampling -----
        # Study and block random effects
        study_names = sorted(df.get_column("SS").unique().to_list())
        study_name_to_idx = {study: idx for idx, study in enumerate(study_names)}
        study_idx = np.array(
            [study_name_to_idx[study] for study in df.get_column("SS").to_list()],
            dtype=np.int32,
        )
        level_study_memberships = {}
        for level_key, col_name in hierarchy.get("column_names", {}).items():
            level_dict = hierarchy.get(level_key, {})
            group_names = list(level_dict.keys())
            membership = np.zeros(
                (len(group_names), len(study_names)),
                dtype=np.float64,
            )
            group_study_pairs = reference_df.select([col_name, "SS"]).unique()
            for group, study in group_study_pairs.iter_rows():
                if group in level_dict and study in study_name_to_idx:
                    membership[level_dict[group], study_name_to_idx[study]] = 1.0
            level_study_memberships[f"{level_key}_study_membership"] = membership

        block_names = sorted(df.get_column("SSB").unique().to_list())
        block_name_to_idx = {block: idx for idx, block in enumerate(block_names)}
        block_idx = np.array(
            [block_name_to_idx[block] for block in df.get_column("SSB").to_list()],
            dtype=np.int32,
        )
        block_pairs = df.select(["SSB", "SS"]).unique().sort("SSB")
        block_to_study_idx = np.array(
            [
                study_name_to_idx[study]
                for study in block_pairs.get_column("SS").to_list()
            ],
            dtype=np.int32,
        )

        # Create response variable vector
        y_obs = df.get_column(self.response_var).to_numpy()

        # Create design matrix
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        x_obs = df.select(x_vars).to_numpy()
        study_slope_terms = self.get_study_slope_terms()
        missing_slope_terms = [term for term in study_slope_terms if term not in x_vars]
        if missing_slope_terms:
            raise ValueError(
                f"Study slope terms are not model covariates: {missing_slope_terms}"
            )
        x_study_slope_obs = (
            df.select(study_slope_terms).to_numpy()
            if study_slope_terms
            else np.zeros((df.height, 0), dtype=float)
        )

        # Add site indices for reference
        site_idx = np.array(
            [self.site_name_to_idx[site] for site in df.get_column("SSBS").to_list()]
        )
        # Add taxon indices for reference if applicable
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            taxon_column = self.taxon_column_for_dataframe(df)
            taxon_idx = np.array(
                [
                    self.taxon_name_to_idx[taxon]
                    for taxon in df.get_column(taxon_column).to_list()
                ]
            )

        # Build output dictionary
        coords = {"idx": np.arange(df.shape[0])}
        coords.update(level_values)
        coords["study_names"] = study_names
        coords["block_names"] = block_names
        coords["x_vars"] = x_vars
        coords["study_slope_vars"] = study_slope_terms

        # Specify coordinates for calibration terms
        coords["x_cal_vars"] = ["y_hat_sqrt", "y_hat", "y_hat_squared"]

        output_dict = {
            "coords": coords,
            "y_obs": y_obs,
            "x_obs": x_obs,
            "x_study_slope_obs": x_study_slope_obs,
            "site_idx": site_idx,
            "study_idx": study_idx,
            "block_idx": block_idx,
            "block_to_study_idx": block_to_study_idx,
            "level_2_to_level_1_idx": level_2_to_level_1_idx,
            "level_3_to_level_2_idx": level_3_to_level_2_idx,
            "level_assignment": level_assignment,
        }
        output_dict.update(level_indices)
        output_dict.update(level_n_studies)
        output_dict.update(level_study_memberships)
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            output_dict["taxon_idx"] = taxon_idx
            output_dict["taxon_column"] = taxon_column

        self.logger.info("Data formatted for PyMC model.")

        return output_dict

    def extract_effects(
        self,
        re_lower_perc: float = 5,
        re_upper_perc: float = 95,
    ) -> dict[str, dict[str, Any]]:
        """
        Extract fixed-effect summaries and optional study/ecological ranges.

        Effects are returned on the response scale as delta(mu) from the
        population intercept, matching the GLMM effect summary structure.
        """
        posterior = self.trace.posterior
        if "mu_beta" not in posterior:
            return {}

        mu_alpha = self._stack_trace_values("mu_alpha").reshape(-1)
        mu_beta = self._stack_trace_values("mu_beta", sample_first_dims=["x_vars"])
        x_vars = [str(value) for value in posterior["mu_beta"].coords["x_vars"].values]
        effect_dict = {}

        for term_idx, term in enumerate(x_vars):
            fixed_eta = mu_beta[:, term_idx]
            fixed_response = self._response_delta(mu_alpha, fixed_eta)
            effect_info = self._summary_dict(fixed_response)

            study_range = self._study_slope_response_range(
                term=term,
                fixed_eta=fixed_eta,
                mu_alpha=mu_alpha,
                lower_perc=re_lower_perc,
                upper_perc=re_upper_perc,
            )
            if study_range:
                effect_info.update(study_range)

            ecological_range = self._ecological_slope_response_range(
                term=term,
                mu_alpha=mu_alpha,
                lower_perc=re_lower_perc,
                upper_perc=re_upper_perc,
            )
            if ecological_range:
                effect_info.update(ecological_range)

            effect_dict[term] = effect_info

        return effect_dict

    def extract_parameter_summary(self) -> pl.DataFrame:
        """
        Return posterior summaries for population, ecological, and random terms.

        Latent-scale summaries are stored in `mean`/`q*` columns. Response-scale
        summaries are stored in `response_*` columns. Slope response effects are
        evaluated relative to the matching intercept: population slopes use
        `mu_alpha`, ecological group slopes use that group's `alpha_k`, and
        study slopes use the study-level intercept when available.
        """
        return self._parameter_summary_from_dataset(self.trace.posterior)

    def sample_prior_parameter_summary(self, draws: int = 1000) -> pl.DataFrame:
        """
        Sample model parameter priors and return compact summaries.

        The returned table uses the same schema as `extract_parameter_summary`,
        but rows summarize prior draws instead of posterior draws. Only
        parameter variables are sampled; prior predictive observations and the
        full prior trace are not retained.
        """
        prior_var_names = self._prior_parameter_var_names()
        if not prior_var_names:
            return pl.DataFrame()

        self.logger.info("Sampling parameter priors for compact summaries.")
        prior = pm.sample_prior_predictive(
            draws=draws,
            model=self.model_instance,
            var_names=prior_var_names,
            random_seed=self.sampling_seed,
        )
        if not hasattr(prior, "prior"):
            return pl.DataFrame()

        return self._parameter_summary_from_dataset(prior.prior)

    def _prior_parameter_var_names(self) -> list[str]:
        """Return prior variables needed for the parameter-summary table."""
        candidate_names = [
            "mu_alpha",
            "mu_beta",
            "sigma_raw",
            "sigma_raw_group_sd",
            "gamma_study",
            "gamma_block",
            "delta_study_slope",
        ]
        for level in range(1, self.model_settings["hierarchical_levels"] + 1):
            candidate_names.extend(
                [f"alpha_{level}", f"beta_{level}", f"sigma_raw_{level}"]
            )

        return [
            variable
            for variable in candidate_names
            if variable in self.model_instance.named_vars
        ]

    def _parameter_summary_from_dataset(self, posterior: Any) -> pl.DataFrame:
        """Return parameter summaries from a posterior-like xarray dataset."""
        rows = []
        mu_alpha = (
            self._stack_trace_values("mu_alpha", posterior=posterior).reshape(-1)
            if "mu_alpha" in posterior
            else None
        )
        mu_beta = None
        x_vars = []

        if mu_alpha is not None:
            rows.append(
                self._parameter_summary_row(
                    parameter="mu_alpha",
                    component="population",
                    effect="intercept",
                    level="population",
                    group=None,
                    covariate=None,
                    values=mu_alpha,
                    response_values=self._response_value(mu_alpha),
                )
            )

        if "mu_beta" in posterior:
            mu_beta = self._stack_trace_values(
                "mu_beta",
                sample_first_dims=["x_vars"],
                posterior=posterior,
            )
            x_vars = [
                str(value) for value in posterior["mu_beta"].coords["x_vars"].values
            ]
            for term_idx, term in enumerate(x_vars):
                rows.append(
                    self._parameter_summary_row(
                        parameter="mu_beta",
                        component="population",
                        effect="slope",
                        level="population",
                        group=None,
                        covariate=term,
                        values=mu_beta[:, term_idx],
                        response_values=(
                            self._response_delta(mu_alpha, mu_beta[:, term_idx])
                            if mu_alpha is not None
                            else None
                        ),
                    )
                )

        for level in range(1, self.model_settings["hierarchical_levels"] + 1):
            alpha_name = f"alpha_{level}"
            beta_name = f"beta_{level}"
            group_names = self._level_group_names(level, posterior=posterior)
            alpha_values = None

            if alpha_name in posterior:
                alpha_dims = self._parameter_dims(alpha_name, posterior=posterior)
                if not alpha_dims:
                    continue
                group_dim = alpha_dims[0]
                alpha_values = self._stack_trace_values(
                    alpha_name,
                    sample_first_dims=[group_dim],
                    posterior=posterior,
                )
                for group_idx in range(alpha_values.shape[1]):
                    rows.append(
                        self._parameter_summary_row(
                            parameter=alpha_name,
                            component="ecological",
                            effect="intercept",
                            level=f"level_{level}",
                            group=group_names[group_idx],
                            covariate=None,
                            values=alpha_values[:, group_idx],
                            response_values=self._response_value(
                                alpha_values[:, group_idx]
                            ),
                        )
                    )

            if beta_name in posterior:
                beta_dims = self._parameter_dims(beta_name, posterior=posterior)
                if len(beta_dims) < 2:
                    continue
                group_dim, covariate_dim = beta_dims[:2]
                beta_values = self._stack_trace_values(
                    beta_name,
                    sample_first_dims=[group_dim, covariate_dim],
                    posterior=posterior,
                )
                x_vars = self._x_vars(posterior=posterior)
                for group_idx in range(beta_values.shape[1]):
                    for term_idx, term in enumerate(x_vars):
                        intercept_values = (
                            alpha_values[:, group_idx]
                            if alpha_values is not None
                            else mu_alpha
                        )
                        rows.append(
                            self._parameter_summary_row(
                                parameter=beta_name,
                                component="ecological",
                                effect="slope",
                                level=f"level_{level}",
                                group=group_names[group_idx],
                                covariate=term,
                                values=beta_values[:, group_idx, term_idx],
                                response_values=(
                                    self._response_delta(
                                        intercept_values,
                                        beta_values[:, group_idx, term_idx],
                                    )
                                    if intercept_values is not None
                                    else None
                                ),
                            )
                        )

        rows.extend(
            self._random_parameter_summary_rows(
                mu_alpha,
                mu_beta,
                x_vars,
                posterior=posterior,
            )
        )
        rows.extend(self._dispersion_parameter_summary_rows(posterior=posterior))

        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)

    def _dispersion_parameter_summary_rows(
        self,
        posterior: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return beta-dispersion summaries when present in the trace."""
        if posterior is None:
            posterior = self.trace.posterior
        rows = []

        if "sigma_raw" in posterior:
            rows.append(
                self._parameter_summary_row(
                    parameter="sigma_raw",
                    component="dispersion",
                    effect="scale",
                    level="population",
                    group=None,
                    covariate=None,
                    values=self._stack_trace_values(
                        "sigma_raw",
                        posterior=posterior,
                    ).reshape(-1),
                )
            )

        if "sigma_raw_group_sd" in posterior:
            rows.append(
                self._parameter_summary_row(
                    parameter="sigma_raw_group_sd",
                    component="dispersion",
                    effect="sd",
                    level="ecological",
                    group=None,
                    covariate=None,
                    values=self._stack_trace_values(
                        "sigma_raw_group_sd",
                        posterior=posterior,
                    ).reshape(-1),
                )
            )

        for level in range(1, self.model_settings["hierarchical_levels"] + 1):
            parameter = f"sigma_raw_{level}"
            if parameter not in posterior:
                continue
            parameter_dims = self._parameter_dims(parameter, posterior=posterior)
            if not parameter_dims:
                continue
            group_dim = parameter_dims[0]
            values = self._stack_trace_values(
                parameter,
                sample_first_dims=[group_dim],
                posterior=posterior,
            )
            group_names = self._level_group_names(level, posterior=posterior)
            for group_idx in range(values.shape[1]):
                rows.append(
                    self._parameter_summary_row(
                        parameter=parameter,
                        component="dispersion",
                        effect="scale",
                        level=f"level_{level}",
                        group=group_names[group_idx],
                        covariate=None,
                        values=values[:, group_idx],
                    )
                )

        return rows

    def _random_parameter_summary_rows(
        self,
        mu_alpha: np.ndarray | None,
        mu_beta: np.ndarray | None,
        x_vars: list[str],
        posterior: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Return total study/block intercept and study-slope summaries."""
        if posterior is None:
            posterior = self.trace.posterior
        rows = []
        study_alpha = None

        if mu_alpha is not None and "gamma_study" in posterior:
            study_dim = self._parameter_dims("gamma_study", posterior=posterior)[0]
            gamma_study = self._stack_trace_values(
                "gamma_study",
                sample_first_dims=[study_dim],
                posterior=posterior,
            )
            study_names = [
                str(value)
                for value in posterior["gamma_study"].coords[study_dim].values
            ]
            study_alpha = mu_alpha[:, None] + gamma_study
            for study_idx, study in enumerate(study_names):
                rows.append(
                    self._parameter_summary_row(
                        parameter="alpha_study",
                        component="random",
                        effect="intercept",
                        level="study",
                        group=study,
                        covariate=None,
                        values=study_alpha[:, study_idx],
                        response_values=self._response_value(study_alpha[:, study_idx]),
                    )
                )

        if mu_alpha is not None and "gamma_block" in posterior:
            block_dim = self._parameter_dims("gamma_block", posterior=posterior)[0]
            gamma_block = self._stack_trace_values(
                "gamma_block",
                sample_first_dims=[block_dim],
                posterior=posterior,
            )
            block_names = [
                str(value)
                for value in posterior["gamma_block"].coords[block_dim].values
            ]
            block_alpha = mu_alpha[:, None] + gamma_block
            for block_idx, block in enumerate(block_names):
                rows.append(
                    self._parameter_summary_row(
                        parameter="alpha_block",
                        component="random",
                        effect="intercept",
                        level="block",
                        group=block,
                        covariate=None,
                        values=block_alpha[:, block_idx],
                        response_values=self._response_value(block_alpha[:, block_idx]),
                    )
                )

        if mu_beta is None or "delta_study_slope" not in posterior:
            return rows

        parameter_dims = self._parameter_dims(
            "delta_study_slope",
            posterior=posterior,
        )
        if len(parameter_dims) < 2:
            return rows
        slope_dim = (
            "study_slope_vars"
            if "study_slope_vars" in parameter_dims
            else parameter_dims[1]
        )
        study_dim = next(dim for dim in parameter_dims if dim != slope_dim)
        study_slope_terms = [
            str(value)
            for value in posterior["delta_study_slope"].coords[slope_dim].values
        ]
        delta_study_slope = (
            posterior["delta_study_slope"]
            .stack(sample=("chain", "draw"))
            .transpose("sample", study_dim, slope_dim)
            .values
        )
        study_names = [
            str(value)
            for value in posterior["delta_study_slope"].coords[study_dim].values
        ]

        for slope_idx, term in enumerate(study_slope_terms):
            if term not in x_vars:
                continue
            fixed_idx = x_vars.index(term)
            for study_idx, study in enumerate(study_names):
                study_beta = (
                    mu_beta[:, fixed_idx] + delta_study_slope[:, study_idx, slope_idx]
                )
                intercept_values = (
                    study_alpha[:, study_idx] if study_alpha is not None else mu_alpha
                )
                rows.append(
                    self._parameter_summary_row(
                        parameter="beta_study",
                        component="random",
                        effect="slope",
                        level="study",
                        group=study,
                        covariate=term,
                        values=study_beta,
                        response_values=(
                            self._response_delta(intercept_values, study_beta)
                            if intercept_values is not None
                            else None
                        ),
                    )
                )

        return rows

    def _stack_trace_values(
        self,
        variable: str,
        sample_first_dims: list[str] | None = None,
        posterior: Any | None = None,
    ) -> np.ndarray:
        """Stack posterior chain/draw dimensions and return sample-first values."""
        if posterior is None:
            posterior = self.trace.posterior
        data_array = posterior[variable].stack(sample=("chain", "draw"))
        if sample_first_dims:
            data_array = data_array.transpose("sample", *sample_first_dims)
        else:
            data_array = data_array.transpose("sample")
        return data_array.values

    def _parameter_dims(
        self,
        variable: str,
        posterior: Any | None = None,
    ) -> list[str]:
        """Return posterior variable dimensions excluding chain and draw."""
        if posterior is None:
            posterior = self.trace.posterior
        return [dim for dim in posterior[variable].dims if dim not in {"chain", "draw"}]

    def _x_vars(self, posterior: Any | None = None) -> list[str]:
        """Return model covariate names in posterior order."""
        if posterior is None:
            posterior = self.trace.posterior
        if "mu_beta" in posterior and "x_vars" in posterior["mu_beta"].coords:
            return [
                str(value) for value in posterior["mu_beta"].coords["x_vars"].values
            ]
        return self.categorical_vars + self.continuous_vars + self.interaction_terms

    def _level_group_names(
        self,
        level: int,
        posterior: Any | None = None,
    ) -> list[str]:
        """Return hierarchy group labels in model index order."""
        level_key = f"level_{level}"
        mapping = self.hierarchy_mapping.get(level_key, {})
        if mapping:
            return [
                str(group)
                for group, _ in sorted(mapping.items(), key=lambda item: item[1])
            ]

        variable = f"alpha_{level}"
        if posterior is None:
            posterior = self.trace.posterior
        if variable in posterior:
            dims = self._parameter_dims(variable, posterior=posterior)
            if dims:
                dim = dims[0]
                size = posterior[variable].sizes[dim]
                return [str(idx) for idx in range(size)]
        return []

    def _response_delta(
        self,
        intercept_eta: np.ndarray,
        delta_eta: np.ndarray,
    ) -> np.ndarray:
        """Transform latent-scale effect deltas to response-scale deltas."""
        if self.model_settings["likelihood"] == "gaussian":
            return delta_eta
        baseline = 1 / (1 + np.exp(-intercept_eta))
        shifted = 1 / (1 + np.exp(-(intercept_eta + delta_eta)))
        return shifted - baseline

    def _response_value(self, eta: np.ndarray) -> np.ndarray:
        """Transform latent-scale values to the response scale."""
        if self.model_settings["likelihood"] == "gaussian":
            return eta
        return 1 / (1 + np.exp(-eta))

    @staticmethod
    def _summary_dict(values: np.ndarray) -> dict[str, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {
                "mean": np.nan,
                "ci_lower_2_5": np.nan,
                "ci_upper_97_5": np.nan,
            }
        return {
            "mean": float(np.mean(finite)),
            "ci_lower_2_5": float(np.quantile(finite, 0.025)),
            "ci_upper_97_5": float(np.quantile(finite, 0.975)),
        }

    @staticmethod
    def _parameter_summary_row(
        parameter: str,
        component: str,
        effect: str,
        level: str,
        group: str | None,
        covariate: str | None,
        values: np.ndarray,
        response_values: np.ndarray | None = None,
    ) -> dict[str, Any]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            mean = q2_5 = q50 = q97_5 = np.nan
        else:
            mean = float(np.mean(finite))
            q2_5 = float(np.quantile(finite, 0.025))
            q50 = float(np.quantile(finite, 0.5))
            q97_5 = float(np.quantile(finite, 0.975))
        if response_values is None:
            response_mean = response_q2_5 = response_q50 = response_q97_5 = np.nan
        else:
            finite_response = response_values[np.isfinite(response_values)]
            if finite_response.size == 0:
                response_mean = response_q2_5 = response_q50 = response_q97_5 = np.nan
            else:
                response_mean = float(np.mean(finite_response))
                response_q2_5 = float(np.quantile(finite_response, 0.025))
                response_q50 = float(np.quantile(finite_response, 0.5))
                response_q97_5 = float(np.quantile(finite_response, 0.975))
        return {
            "parameter": parameter,
            "component": component,
            "effect": effect,
            "level": level,
            "group": group,
            "covariate": covariate,
            "mean": mean,
            "q2_5": q2_5,
            "q50": q50,
            "q97_5": q97_5,
            "response_mean": response_mean,
            "response_q2_5": response_q2_5,
            "response_q50": response_q50,
            "response_q97_5": response_q97_5,
        }

    def _study_slope_response_range(
        self,
        term: str,
        fixed_eta: np.ndarray,
        mu_alpha: np.ndarray,
        lower_perc: float,
        upper_perc: float,
    ) -> dict[str, Any]:
        posterior = self.trace.posterior
        if "delta_study_slope" not in posterior:
            return {}
        parameter_dims = self._parameter_dims("delta_study_slope")
        if len(parameter_dims) < 2:
            return {}
        slope_dim = (
            "study_slope_vars"
            if "study_slope_vars" in parameter_dims
            else parameter_dims[1]
        )
        study_dim = next(dim for dim in parameter_dims if dim != slope_dim)
        study_slope_terms = [
            str(value)
            for value in posterior["delta_study_slope"].coords[slope_dim].values
        ]
        if term not in study_slope_terms:
            return {}
        term_idx = study_slope_terms.index(term)

        deviations = (
            posterior["delta_study_slope"]
            .isel({slope_dim: term_idx})
            .stack(sample=("chain", "draw"))
            .transpose("sample", study_dim)
            .values
        )
        intercept_values = mu_alpha[:, None]
        if "gamma_study" in posterior:
            gamma_study = (
                posterior["gamma_study"]
                .stack(sample=("chain", "draw"))
                .transpose("sample", study_dim)
                .values
            )
            intercept_values = mu_alpha[:, None] + gamma_study
        response_values = self._response_delta(
            intercept_values, fixed_eta[:, None] + deviations
        )
        study_names = [
            str(value)
            for value in posterior["delta_study_slope"].coords[study_dim].values
        ]
        study_means = np.nanmean(response_values, axis=0)
        study_effect_values = {
            study_names[idx]: float(value) for idx, value in enumerate(study_means)
        }
        active_studies = self._active_study_names_for_slope_term(term, study_names)
        if active_studies:
            active_indices = [
                idx for idx, study in enumerate(study_names) if study in active_studies
            ]
            study_names = [study_names[idx] for idx in active_indices]
            response_values = response_values[:, active_indices]
            study_means = np.nanmean(response_values, axis=0)
            study_effect_values = {
                study_names[idx]: float(value) for idx, value in enumerate(study_means)
            }

        random_slope_lower = float(np.nanquantile(study_means, lower_perc / 100))
        random_slope_upper = float(np.nanquantile(study_means, upper_perc / 100))
        random_slope_mean = float(np.nanmean(study_means))

        return {
            "random_slope_lower": random_slope_lower,
            "random_slope_upper": random_slope_upper,
            "random_slope_mean": random_slope_mean,
            "study_effect_values": study_effect_values,
        }

    def _active_study_names_for_slope_term(
        self,
        term: str,
        study_names: list[str],
    ) -> set[str]:
        """
        Return studies where a study-slope term is present in the training data.

        Study random slopes are sampled for every study-term combination. For
        terms absent from a study, the posterior remains close to the common
        prior and should not contribute to displayed study heterogeneity.
        """
        if self.training_model_data is None:
            return set()
        model_data = self.training_model_data
        slope_terms = list(model_data["coords"].get("study_slope_vars", []))
        if term not in slope_terms:
            return set()

        term_idx = slope_terms.index(term)
        x_term = model_data["x_study_slope_obs"][:, term_idx]
        study_idx = model_data["study_idx"]
        active = set()
        for idx, study in enumerate(study_names):
            values = x_term[study_idx == idx]
            if np.any(np.isfinite(values) & (np.abs(values) > 1e-12)):
                active.add(study)
        return active

    def _ecological_slope_response_range(
        self,
        term: str,
        mu_alpha: np.ndarray,
        lower_perc: float,
        upper_perc: float,
    ) -> dict[str, Any]:
        posterior = self.trace.posterior
        level = self.model_settings["hierarchical_levels"]
        beta_name = f"beta_{level}"
        if beta_name not in posterior:
            return {}
        parameter_dims = self._parameter_dims(beta_name)
        if len(parameter_dims) < 2:
            return {}
        x_vars = self._x_vars()
        if term not in x_vars:
            return {}
        group_dim, covariate_dim = parameter_dims[:2]
        term_idx = x_vars.index(term)
        alpha_name = f"alpha_{level}"
        if alpha_name in posterior:
            alpha_group_dim = self._parameter_dims(alpha_name)[0]
            alpha_values = (
                posterior[alpha_name]
                .stack(sample=("chain", "draw"))
                .transpose("sample", alpha_group_dim)
                .values
            )
        else:
            alpha_values = mu_alpha[:, None]

        beta_values = (
            posterior[beta_name]
            .isel({covariate_dim: term_idx})
            .stack(sample=("chain", "draw"))
            .transpose("sample", group_dim)
            .values
        )
        response_values = self._response_delta(alpha_values, beta_values)
        group_names = self._level_group_names(level)
        if len(group_names) != beta_values.shape[1]:
            group_names = [
                str(value) for value in posterior[beta_name].coords[group_dim].values
            ]
        ecological_means = np.nanmean(response_values, axis=0)
        ecological_effect_values = {
            group_names[idx]: float(value) for idx, value in enumerate(ecological_means)
        }
        ecological_slope_lower = float(
            np.nanquantile(ecological_means, lower_perc / 100)
        )
        ecological_slope_upper = float(
            np.nanquantile(ecological_means, upper_perc / 100)
        )
        ecological_slope_mean = float(np.nanmean(ecological_means))

        return {
            "ecological_slope_lower": ecological_slope_lower,
            "ecological_slope_upper": ecological_slope_upper,
            "ecological_slope_mean": ecological_slope_mean,
            "ecological_effect_values": ecological_effect_values,
        }

    def run_sampling(self) -> az.InferenceData:
        """
        Run sampling for the current model. The function uses the No U-turn
        (NUTS) sampler implemented in PyMC, and the 'sampler_settings'
        dictionary is specific to this sampler.

        Returns:
            trace: PyMC trace with posterior distribution info appended.
        """
        self.logger.info("Running NUTS sampler.")
        start = time.time()

        with self.model_instance:
            trace = pm.sample(
                draws=self.sampler_settings["draws"],
                tune=self.sampler_settings["tune"],
                cores=self.sampler_settings["cores"],
                chains=self.sampler_settings["chains"],
                target_accept=self.sampler_settings["target_accept"],
                nuts_sampler=self.sampler_settings["nuts_sampler"],
                progressbar=self.progressbar,
                random_seed=self.sampling_seed,
            )

        runtime = str(timedelta(seconds=int(time.time() - start)))
        self.logger.info(f"Finished sampling in {runtime}.")

        return trace

    def summarize_sampling_statistics(self) -> None:
        """
        Calculate sampling statistics for the model to evaluate the convergence
        of the sampling chains. This includes divergences, acceptance rate,
        R-hat statistics and effective sample size (ESS) statistics.
        """
        var_names = list(self.trace.posterior.data_vars)
        idata = az.convert_to_dataset(self.trace)  # Avoid doing conversion twice

        # Divergences
        divergences = np.sum(self.trace.sample_stats["diverging"].values)
        self.logger.warning(
            f"There are {divergences} divergences in the sampling chains."
        )

        # Acceptance rate
        accept_rate = np.mean(self.trace.sample_stats["acceptance_rate"].values)
        self.logger.warning(f"The mean acceptance rate was {accept_rate:.3f}")

        # R-hat statistics
        for var in var_names:
            try:
                r_hat = az.summary(idata, var_names=var, round_to=2)["r_hat"]
                mean_r_hat = np.mean(r_hat)
                min_r_hat = np.min(r_hat)
                max_r_hat = np.max(r_hat)
                self.logger.info(
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
                self.logger.info(
                    f"ESS for {var} are: {int(mean_ess)} (mean) | {int(min_ess)} "
                    f"(min) | {int(max_ess)} (max)"
                )
            except KeyError:
                continue

    def make_predictions(
        self,
        prediction_data: dict[str, Any],
        mode: str,
        prediction_components: dict[str, bool] | None = None,
        prediction_label: str | None = None,
        sample_likelihood: bool = True,
    ) -> az.InferenceData:
        """
        Sample from the posterior predictive distribution to make predictions.
        If using rolled-up mapping, build a fallback-aware prediction model.

        Args:
            - prediction_data: PyMC model dictionary with new data if
                predictions are made out of sample. Ignored if in training mode.
            - mode: Either 'train' or 'test'. Note that this is different from
                the 'mode' attribute, which is related to the calling task.

        Returns:
            trace: The updated trace object from the model, incl. predictions.
        """
        if prediction_components is None:
            prediction_components = self.get_prediction_components()
        prediction_settings = self.model_settings_with_prediction_components(
            prediction_components
        )
        self.logger.info(
            "BHM components used for %s%s predictions: %s.",
            mode,
            f" {prediction_label}" if prediction_label else "",
            self.format_component_settings(prediction_components),
        )
        train_var_names = ["y_cond", "y_intercept"]
        if sample_likelihood:
            train_var_names.insert(0, "y_like")
        test_var_names = ["y_cond", "y_intercept"]
        if sample_likelihood:
            test_var_names.insert(0, "y_pred")

        use_rolled_up_predictions = (
            hasattr(self, "rolled_up_mapping")
            and self.rolled_up_mapping
            and prediction_components["ecological"]
        )
        if use_rolled_up_predictions:
            self.logger.info("Using rolled-up mapping for predictions")
            # Use fallback-aware prediction model in both train and test
            prediction_model = rolled_up_prediction_model(
                model_data=prediction_data,
                trace=self.trace,
                settings=prediction_settings,
                mode=mode,
                epsilon=self.epsilon,
            )
            if mode == "train":
                with prediction_model:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=train_var_names,
                        predictions=False,
                        extend_inferencedata=True,
                        progressbar=self.progressbar,
                        random_seed=self.sampling_seed + 1,
                    )
            elif mode == "test":
                with prediction_model:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=test_var_names,
                        predictions=True,
                        extend_inferencedata=True,
                        progressbar=self.progressbar,
                        random_seed=self.sampling_seed + 1,
                    )
        else:
            if (
                hasattr(self, "rolled_up_mapping")
                and self.rolled_up_mapping
                and not prediction_components["ecological"]
            ):
                self.logger.info(
                    "Ignoring rolled-up prediction mapping because ecological "
                    "prediction is disabled."
                )
            prediction_model_builder = GeneralHierarchicalModel(
                settings=prediction_settings,
                epsilon=self.epsilon,
            )
            self.pred_model = prediction_model_builder.build_prediction_model(
                model_data=prediction_data,
                mode=mode,
            )
            if mode == "train":
                with self.pred_model:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=train_var_names,
                        predictions=False,
                        extend_inferencedata=True,
                        progressbar=self.progressbar,
                        random_seed=self.sampling_seed + 1,
                    )
            elif mode == "test":
                with self.pred_model:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=test_var_names,
                        predictions=True,
                        extend_inferencedata=True,
                        progressbar=self.progressbar,
                        random_seed=self.sampling_seed + 1,
                    )

        return updated_trace

    def create_prediction_dataframe(
        self,
        prediction_data: pl.DataFrame,
        mode: str,
        include_predictive_distribution: bool = True,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Extract site names, observed values, and predicted values (posterior
        means) and outputs two dataframes:
        1. A summary dataframe for training or cross-validation tasks.
        2. A dataframe containing the full predictive distribution for each
        site.

        Args:
            - prediction_data: Dataframe with new data if predictions are made
                out of sample, otherwise it's the training data.
            - mode: Either 'train' or 'test'. Note that this is different from
                the 'mode' attribute, which is related to the calling task.

        Returns:
            - df_pred: Dataframe with site names, observed values, and
                predictions (both capped and uncapped).
            - df_pred_distr: Dataframe containing the full predictive
                distribution for each site.
        """
        # Get site information and observed values
        site_idx = prediction_data["site_idx"]
        idx_to_site = {idx: name for name, idx in self.site_name_to_idx.items()}
        site_names = [idx_to_site[idx] for idx in site_idx]
        y_obs = prediction_data["y_obs"]

        # Get taxon information if applicable
        include_taxon = False
        taxon_names = None
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            include_taxon = True
            taxon_idx = prediction_data["taxon_idx"]
            idx_to_taxon = {idx: name for name, idx in self.taxon_name_to_idx.items()}
            taxon_names = [idx_to_taxon[idx] for idx in taxon_idx]

        # Determine where to extract predictions from
        if mode == "train":
            if "y_like" in self.trace.posterior_predictive:
                y_pred_samples = self.trace.posterior_predictive["y_like"]
            else:
                y_pred_samples = self.trace.posterior_predictive["y_cond"]
            y_cond_samples = self.trace.posterior_predictive["y_cond"]
            ref_pred_samples = self.trace.posterior_predictive["y_intercept"]
        elif mode == "test":
            if "y_pred" in self.trace.predictions:
                y_pred_samples = self.trace.predictions["y_pred"]
            else:
                y_pred_samples = self.trace.predictions["y_cond"]
            y_cond_samples = self.trace.predictions["y_cond"]
            ref_pred_samples = self.trace.predictions["y_intercept"]

        # Compute the posterior means for summary dataframe
        y_pred = y_pred_samples.mean(dim=("chain", "draw")).values
        y_cond = y_cond_samples.mean(dim=("chain", "draw")).values
        reference_pred = ref_pred_samples.mean(dim=("chain", "draw")).values

        # Create summary dataframe, adding taxon labels only for taxonomic runs.
        prediction_rows = {
            "SSBS": site_names,
            "Observed": y_obs,
            "Predicted": y_cond,
            "y_pred": y_pred,
            "Reference_pred": reference_pred,
        }
        if include_taxon:
            taxon_column = prediction_data.get("taxon_column", "Custom_taxonomic_group")
            prediction_rows[taxon_column] = taxon_names
        df_pred = pl.DataFrame(prediction_rows)

        if include_predictive_distribution:
            # Flatten the full predictive distribution into a long-form dataframe
            chain_dim, draw_dim = y_pred_samples.shape[
                0:2
            ]  # Extract chain and draw dimensions
            nb_sites = y_pred_samples.shape[2]  # Extract the number of sites

            df_pred_distr = pl.DataFrame(
                {
                    # Repeat site names for every combination of chain and draw
                    "SSBS": site_names * (chain_dim * draw_dim),
                    # Chain index, repeated for all draws and sites
                    "Chain": [
                        i for i in range(chain_dim) for _ in range(draw_dim * nb_sites)
                    ],
                    # Draw index, repeated for all sites within each chain
                    "Draw": [
                        j
                        for i in range(chain_dim)
                        for j in range(draw_dim)
                        for _ in range(nb_sites)
                    ],
                    # Flattened predictions, corresponding to (chain, draw, site)
                    "Prediction": y_pred_samples.values.flatten(),
                    "Reference_pred": ref_pred_samples.values.flatten(),
                }
            )
        else:
            df_pred_distr = pl.DataFrame()

        return df_pred, df_pred_distr
