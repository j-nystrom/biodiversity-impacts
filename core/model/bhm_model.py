import logging
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

        # Format data for PyMC model
        train_data = self.format_data_for_pymc_model(df_train_std)
        test_data = self.format_data_for_pymc_model(df_test_std)

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
        self.model = GeneralHierarchicalModel(
            settings=self.model_settings, epsilon=self.epsilon
        )
        self.model_instance = self.model.build_training_model(model_data=train_data)

        if self.model_settings["prior_predictive_checks"]:
            # Do prior predictive sampling before running the model
            self.logger.info("Running prior predictive sampling.")
            self.prior_predictive = pm.sample_prior_predictive(
                draws=1000,
                model=self.model_instance,
                random_seed=self.sampling_seed,
            )
            plot_prior_distribution(
                self.prior_predictive,
                category_variable_pairs=[
                    tuple(pair)
                    for pair in self.model_settings["prior_predictive_plot_pairs"]
                ],
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
        self.trace = self.make_predictions(prediction_data, mode=pred_mode)

        df_pred, df_pred_distr = self.create_prediction_dataframe(
            prediction_data,
            mode=pred_mode,
            include_predictive_distribution=self.save_predictive_distributions,
        )

        return df_pred, df_pred_distr

    def format_data_for_pymc_model(self, df: pl.DataFrame) -> dict[str, Any]:
        """
        Format the dataframe for use in PyMC models.

        Args:
            - df: Dataframe with the scaled covariates and response variable.

        Returns:
            - output_dict: Dictionary containing the formatted data for the
                PyMC model.
        """
        self.logger.info("Formatting data for PyMC model.")

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

                # Get the number of studies for each present group at this level
                study_count_dict = hierarchy.get(f"{level_key}_n_studies", {})
                level_n_studies[f"{level_key}_n_studies"] = np.array(
                    [study_count_dict.get(label, 0) for label in group_names],
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
        study_names = df.get_column("SS").unique().to_list()
        study_idx = df.get_column("SS").cast(pl.Categorical).to_physical().to_numpy()
        block_names = df.get_column("SSB").unique().to_list()
        block_idx = df.get_column("SSB").cast(pl.Categorical).to_physical().to_numpy()
        block_to_study_idx = (
            df.select(["SS", "SSB"])
            .unique()
            .get_column("SS")
            .cast(pl.Categorical)
            .to_physical()
            .to_numpy()
        )

        # Create response variable vector
        y_obs = df.get_column(self.response_var).to_numpy()

        # Create design matrix
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        x_obs = df.select(x_vars).to_numpy()

        # Add site indices for reference
        site_idx = np.array(
            [self.site_name_to_idx[site] for site in df.get_column("SSBS").to_list()]
        )
        # Add taxon indices for reference if applicable
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            taxon_idx = np.array(
                [
                    self.taxon_name_to_idx[taxon]
                    for taxon in df.get_column("Custom_taxonomic_group").to_list()
                ]
            )

        # Build output dictionary
        coords = {"idx": np.arange(df.shape[0])}
        coords.update(level_values)
        coords["study_names"] = study_names
        coords["block_names"] = block_names
        coords["x_vars"] = x_vars

        # Specify coordinates for calibration terms
        coords["x_cal_vars"] = ["y_hat_sqrt", "y_hat", "y_hat_squared"]

        output_dict = {
            "coords": coords,
            "y_obs": y_obs,
            "x_obs": x_obs,
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
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            output_dict["taxon_idx"] = taxon_idx

        self.logger.info("Data formatted for PyMC model.")

        return output_dict

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
        self, prediction_data: dict[str, Any], mode: str
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
        if hasattr(self, "rolled_up_mapping") and self.rolled_up_mapping:
            self.logger.info("Using rolled-up mapping for predictions")
            # Use fallback-aware prediction model in both train and test
            prediction_model = rolled_up_prediction_model(
                model_data=prediction_data,
                trace=self.trace,
                settings=self.model_settings,
                mode=mode,
                epsilon=self.epsilon,
            )
            if mode == "train":
                with prediction_model:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["y_like", "y_cond", "y_intercept"],
                        predictions=False,
                        extend_inferencedata=True,
                        random_seed=self.sampling_seed + 1,
                    )
            elif mode == "test":
                with prediction_model:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["y_pred", "y_cond", "y_intercept"],
                        predictions=True,
                        extend_inferencedata=True,
                        random_seed=self.sampling_seed + 1,
                    )
        else:
            # Use existing approach
            if mode == "train":
                with self.model_instance:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["y_like", "y_cond", "y_intercept"],
                        predictions=False,
                        extend_inferencedata=True,
                        random_seed=self.sampling_seed + 1,
                    )
            elif mode == "test":
                self.pred_model = self.model.build_prediction_model(
                    model_data=prediction_data,
                )
                with self.pred_model:
                    updated_trace = pm.sample_posterior_predictive(
                        self.trace,
                        var_names=["y_pred", "y_cond", "y_intercept"],
                        predictions=True,
                        extend_inferencedata=True,
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
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            include_taxon = True
            taxon_idx = prediction_data["taxon_idx"]
            idx_to_taxon = {idx: name for name, idx in self.taxon_name_to_idx.items()}
            taxon_names = [idx_to_taxon[idx] for idx in taxon_idx]

        # Determine where to extract predictions from
        if mode == "train":
            y_pred_samples = self.trace.posterior_predictive["y_like"]
            y_cond_samples = self.trace.posterior_predictive["y_cond"]
            ref_pred_samples = self.trace.posterior_predictive["y_intercept"]
        elif mode == "test":
            y_pred_samples = self.trace.predictions["y_pred"]
            y_cond_samples = self.trace.predictions["y_cond"]
            ref_pred_samples = self.trace.predictions["y_intercept"]

        # Compute the posterior means for summary dataframe
        y_pred = y_pred_samples.mean(dim=("chain", "draw")).values
        y_cond = y_cond_samples.mean(dim=("chain", "draw")).values
        reference_pred = ref_pred_samples.mean(dim=("chain", "draw")).values

        # Create summary dataframe, now including Reference_pred
        df_pred = pl.DataFrame(
            {
                "SSBS": site_names,
                "Custom_taxonomic_group": taxon_names if include_taxon else None,
                "Observed": y_obs,
                "Predicted": y_cond,
                "y_pred": y_pred,
                "Reference_pred": reference_pred,
            }
        )

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
