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
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        site_name_to_idx: dict[str, int],
        logger: logging.Logger,
        mode: str,
    ) -> None:
        super().__init__(model_settings, model_vars, site_name_to_idx, logger, mode)

        # Model-specific settings
        self.hierarchy = self.model_settings["hierarchy"]
        self.sampler_settings = self.model_settings["sampler"]

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
        Format the dataframe for use in PyMC models. The function dynamically handles
        one, two, or three hierarchical levels.

        Args:
            - df: Dataframe with the scaled covariates and response variable.

        Returns:
            - output_dict: Dictionary containing the formatted data for the PyMC model.
            - idx_to_site_name: Mapping of indices to site names for reference.
        """
        self.logger.info("Formatting data for PyMC model.")

        # Extract columns for the hierarchical levels
        level_1 = self.hierarchy["level_1"]
        level_2 = self.hierarchy["level_2"]
        level_3 = self.hierarchy["level_3"]

        # Sort dataframe for consistent operations
        df = df.sort(["SS", "SSB", "SSBS"])

        # Process Level 1
        if len(level_1) >= 1:
            level_1_col = "_".join(level_1)
            df = df.with_columns(
                pl.concat_str([pl.col(col) for col in level_1], separator="_").alias(
                    level_1_col
                )
            )
        else:
            level_1_col = level_1[0]

        level_1_values = df.get_column(level_1_col).unique().to_list()
        level_1_idx = (
            df.get_column(level_1_col).cast(pl.Categorical).to_physical().to_numpy()
        )

        # Process Level 2
        if level_2 and all(col in df.columns for col in level_2):
            if len(level_2) >= 1:
                level_2_col = "_".join(level_2)
                pl.concat_str([pl.col(col) for col in level_2], separator="_").alias(
                    level_2_col
                )
            else:
                level_2_col = level_2[0]

            df = df.with_columns(
                pl.concat_str(
                    [pl.col(level_1_col), pl.col(level_2_col)], separator="_"
                ).alias("Combined_Level_2")
            )
            level_2_values = df.get_column("Combined_Level_2").unique().to_list()
            level_2_idx = (
                df.get_column("Combined_Level_2")
                .cast(pl.Categorical)
                .to_physical()
                .to_numpy()
            )
            level_2_to_level_1_idx = (
                df.select([level_1_col, "Combined_Level_2"])
                .unique()
                .sort([level_1_col, "Combined_Level_2"])
                .get_column(level_1_col)
                .cast(pl.Categorical)
                .to_physical()
                .to_numpy()
            )
        else:
            level_2_values = None
            level_2_idx = None
            level_2_to_level_1_idx = None

        # Process Level 3
        if level_3 and all(col in df.columns for col in level_3):
            if len(level_3) >= 1:
                level_3_col = "_".join(level_3)
                pl.concat_str([pl.col(col) for col in level_3], separator="_").alias(
                    level_3_col
                )
            else:
                level_3_col = level_3[0]

            df = df.with_columns(
                pl.concat_str(
                    [pl.col("Combined_Level_2"), pl.col(level_3_col)], separator="_"
                ).alias("Combined_Level_3")
            )
            level_3_values = df.get_column("Combined_Level_3").unique().to_list()
            level_3_idx = (
                df.get_column("Combined_Level_3")
                .cast(pl.Categorical)
                .to_physical()
                .to_numpy()
            )
            level_3_to_level_2_idx = (
                df.select(["Combined_Level_2", "Combined_Level_3"])
                .unique()
                .sort(["Combined_Level_2", "Combined_Level_3"])
                .get_column("Combined_Level_2")
                .cast(pl.Categorical)
                .to_physical()
                .to_numpy()
            )
        else:
            level_3_values = None
            level_3_idx = None
            level_3_to_level_2_idx = None

        # Create the response variable vector
        if self.response_var_transform:
            response_col_name = self.response_var + "_" + self.response_var_transform
        else:
            response_col_name = self.response_var
        y_obs = df.select(response_col_name).to_numpy().flatten()

        # Create the design matrix
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        x_obs = df.select(x_vars).to_numpy()

        # Create coordinates dictionary
        idx = np.arange(len(y_obs))
        site_idx = np.array(
            [self.site_name_to_idx[site] for site in df.get_column("SSBS").to_list()]
        )
        coords = {
            "idx": idx,
            "x_vars": x_vars,
            "level_1_values": level_1_values,
        }
        if level_2_values is not None:
            coords["level_2_values"] = level_2_values
        if level_3_values is not None:
            coords["level_3_values"] = level_3_values

        # Build output dictionary
        output_dict = {
            "coords": coords,
            "y_obs": y_obs,
            "x_obs": x_obs,
            "level_1_idx": level_1_idx,
            "site_idx": site_idx,
        }
        if level_2_idx is not None:
            output_dict["level_2_idx"] = level_2_idx
            output_dict["level_2_to_level_1_idx"] = level_2_to_level_1_idx
        if level_3_idx is not None:
            output_dict["level_3_idx"] = level_3_idx
            output_dict["level_3_to_level_2_idx"] = level_3_to_level_2_idx

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
                idata_kwargs={"log_likelihood": True},  # Compute log likelihood
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
        site_names = [self.idx_to_site[idx] for idx in site_idx]
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
