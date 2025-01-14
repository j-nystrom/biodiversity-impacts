import logging
import time
from datetime import timedelta
from typing import Any

import arviz as az
import numpy as np
import polars as pl
import pymc as pm

from core.model.base_model_class import BaseModel
from core.model.model_utils import standardize_continuous_covariates
from core.model.pymc_models import general_hierarchical_model


class BayesianHierarchicalModel(BaseModel):
    def __init__(
        self,
        mode: str,
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        logger: logging.Logger,
        site_name_to_idx: dict[str, int],
        hierarchy_mapping: dict[str, list[str]],
    ) -> None:
        super().__init__(mode, model_settings, model_vars, logger)

        # Additional settings for Bayesian model
        self.site_name_to_idx: dict[str, int] = site_name_to_idx
        self.hierarchy_mapping: dict[str, Any] = hierarchy_mapping
        self.sampler_settings: dict[str, Any] = self.model_settings["sampler"]

    def prepare_data(self, df: pl.DataFrame) -> dict[str, Any]:
        # Continuous vars are standardized to have mean zero and unit variance
        df_std = standardize_continuous_covariates(
            df, vars_to_scale=self.continuous_vars + self.interaction_terms
        )

        # Format data for PyMC model
        model_data_dict = self.format_data_for_pymc_model(df_std)

        return model_data_dict

    def fit(self, train_data: dict[str, Any]) -> None:
        # Initialize the PyMC model and return the model object
        self.model_instance = general_hierarchical_model(
            model_data=train_data, settings=self.model_settings
        )

        # Do prior predictive sampling for output analysis
        self.prior_predictive = pm.sample_prior_predictive(
            draws=1000, model=self.model_instance
        )

        # Run NUTS sampler on the model and summarize sampling statistics
        self.trace = self.run_sampling()
        self.summarize_sampling_statistics()

    def predict(self, prediction_data: dict[str, Any], pred_mode: str) -> pl.DataFrame:
        # Make predictions on training or test data, return and store updated
        # trace object with posterior predictive samples / predictions
        # If training mode, the test_data argument is ignored
        self.trace = self.make_predictions(prediction_data, mode=pred_mode)

        df_pred = self.create_prediction_dataframe(prediction_data, mode=pred_mode)

        return df_pred

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

        # Use self.hierarchy_mapping to simplify processing
        hierarchy = self.hierarchy_mapping

        # Initialize outputs
        level_indices = {}
        level_values = {}

        # Process hierarchical levels dynamically to create index variables
        for level, mapping in hierarchy.items():
            if level == "column_names":  # Process mappings only
                continue

            col_name = hierarchy["column_names"][level]
            level_indices[f"{level}_idx"] = (
                df.get_column(col_name)
                .replace(mapping, return_dtype=pl.Int32)
                .to_numpy()
            )
            level_values[f"{level}_values"] = list(mapping.keys())

        # Create mapping indices between levels
        # Level 2 to level 1
        if hierarchy["column_names"]["level_2"]:
            level_1_col = hierarchy["column_names"]["level_1"]
            level_2_col = hierarchy["column_names"]["level_2"]
            level_2_to_level_1_idx = (
                df.select([level_1_col, level_2_col])
                .unique()
                .sort([level_1_col, level_2_col])
                .get_column(level_1_col)
                .cast(pl.Categorical)
                .to_physical()
                .to_numpy()
            )

        # Level 3 to level 2
        if hierarchy["column_names"]["level_3"]:
            level_2_col = hierarchy["column_names"]["level_2"]
            level_3_col = hierarchy["column_names"]["level_3"]
            level_3_to_level_2_idx = (
                df.select([level_2_col, level_3_col])
                .unique()
                .sort([level_2_col, level_3_col])
                .get_column(level_2_col)
                .cast(pl.Categorical)
                .to_physical()
                .to_numpy()
            )

        # Create response variable vector
        response_col_name = (
            f"{self.response_var}_{self.response_var_transform}"
            if self.response_var_transform
            else self.response_var
        )
        y_obs = df.get_column(response_col_name).to_numpy()

        # Create design matrix
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        x_obs = df.select(x_vars).to_numpy()

        # Add site indices for reference
        site_idx = np.array(
            [self.site_name_to_idx[site] for site in df.get_column("SSBS").to_list()]
        )

        # Build output dictionary
        coords = {"idx": np.arange(df.shape[0])}
        coords.update(level_values)
        coords["x_vars"] = x_vars

        output_dict = {
            "coords": coords,
            "y_obs": y_obs,
            "x_obs": x_obs,
            "site_idx": site_idx,
            "level_2_to_level_1_idx": level_2_to_level_1_idx,
            "level_3_to_level_2_idx": level_3_to_level_2_idx,
        }
        output_dict.update(level_indices)

        self.logger.info("Data formatted for PyMC model.")

        return output_dict

    def run_sampling(self) -> az.InferenceData:
        """
        Run sampling for the current model. The function uses the No U-turn
        (NUTS) sampler implemented in PyMC, and the 'sampler_settings'
        dictionary is specific to this sampler.

        Returns:
            trace: PyMC trace with posterior distributions info appended.
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
                # idata_kwargs={"log_likelihood": True},  # Compute log likelihood
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
        How this is done depends on if predictions are made on the training or
        test data (for cross-validation), specified in the 'mode' attribute.
        The method updates the trace object with posterior predictive samples
        or out of sample predictions in-place.

        Args:
            - prediction_data: PyMC model dictionary with new data if
                predictions are made out of sample. Ignored if in training mode.
            - mode: Either 'train' or 'test'. Note that this is different from
                the 'mode' attribute, which is related to the calling task.

        Returns:
            trace: The updated trace object from the model, incl. predictions.
        """
        # If training mode, there is no need to manipulate data
        if mode == "train":
            with self.model_instance:
                updated_trace = pm.sample_posterior_predictive(
                    self.trace, predictions=False, extend_inferencedata=True
                )

        # For cross-validation, we need to add the test data to the model and
        # make sure all indices are correct
        elif mode == "test":
            train_len = len(self.trace.observed_data["idx"])
            test_len = len(prediction_data["x_obs"])
            prediction_data["coords"]["idx"] = np.arange(  # Update idx
                train_len, train_len + test_len  # Start from end of train data
            )
            with self.model_instance:
                pm.set_data(  # Update model with test data
                    {
                        "x_obs": prediction_data["x_obs"],
                        "y_obs": prediction_data["y_obs"],
                        "level_1_idx": prediction_data["level_1_idx"],
                        "level_2_idx": prediction_data["level_2_idx"],
                        "level_3_idx": prediction_data["level_3_idx"],
                        "site_idx": prediction_data["site_idx"],
                    },
                    coords=prediction_data["coords"],
                )
                updated_trace = pm.sample_posterior_predictive(
                    self.trace,
                    predictions=True,
                    extend_inferencedata=True,
                )

        return updated_trace

    def create_prediction_dataframe(
        self, prediction_data: pl.DataFrame, mode: str
    ) -> pl.DataFrame:
        """
        Extract site names, observed values, and predicted values (posterior
        means) and outputs a dataframe in a specified format that can be used
        in the training or cross-validation tasks.

        Args:
            - prediction_data: Dataframe with new data if predictions are made
                out of sample, otherwise it's the training data.
            - mode: Either 'train' or 'test'. Note that this is different from
                the 'mode' attribute, which is related to the calling task.

        Returns:
            - df_pred: Dataframe with site names, observed values, and
                predictions (both capped and uncapped).
        """
        # Get site information and observed values
        site_idx = prediction_data["site_idx"]
        idx_to_site = {idx: name for name, idx in self.site_name_to_idx.items()}
        site_names = [idx_to_site[idx] for idx in site_idx]
        y_obs = prediction_data["y_obs"]

        if mode == "train":  # Posterior predictive samples on training data
            y_pred = (
                self.trace.posterior_predictive["y_like"]
                .mean(dim=("chain", "draw"))
                .values
            )

        elif mode == "test":  # Predictions on out-of-sample data
            y_pred = self.trace.predictions["y_like"].mean(dim=("chain", "draw")).values

        df_pred = pl.DataFrame(
            {
                "SSBS": site_names,
                "Observed": y_obs,
                "Predicted": y_pred,
            }
        )

        return df_pred
