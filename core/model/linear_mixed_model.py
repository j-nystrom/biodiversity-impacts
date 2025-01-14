import logging
import time
from datetime import timedelta
from typing import Any, Union

import numpy as np
import pandas as pd
import polars as pl
from pymer4.models import Lmer

from core.model.base_model_class import BaseModel
from core.model.model_utils import standardize_continuous_covariates


class LinearMixedModel(BaseModel):
    def __init__(
        self,
        mode: str,
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        super().__init__(mode, model_settings, model_vars, logger)

        # Fixed effects
        self.response_col_name = (
            f"{self.response_var}_{self.response_var_transform}"
            if self.response_var_transform
            else self.response_var
        )
        self.fixed_effects = (
            self.categorical_vars + self.continuous_vars + self.interaction_terms
        )

        # Random effects settings
        self.random_effects_type = self.model_settings["random_effects_type"]
        self.study_effects = self.model_settings["study_effects"]
        self.block_effects = self.model_settings["block_effects"]
        self.solver = self.model_settings["solver"]

    def prepare_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # Continuous vars are standardized to have mean zero and unit variance
        df_std = standardize_continuous_covariates(
            df, vars_to_scale=self.continuous_vars + self.interaction_terms
        )

        # Format data for Lmer model, including removing spaces from col names
        df_model_data, self.model_formula = self.format_data_for_pymer_model(df_std)

        return df_model_data

    def fit(self, df_train: pl.DataFrame) -> None:
        self.model_instance = self.train_pymer_model(
            df=df_train, formula=self.model_formula
        )

    def predict(self, prediction_data: pd.DataFrame, pred_mode: str) -> pl.DataFrame:
        predictions = self.make_predictions(prediction_data, mode=pred_mode)
        df_pred = self.create_prediction_dataframe(
            prediction_data, predictions=predictions
        )

        return df_pred

    def format_data_for_pymer_model(self, df: pl.DataFrame) -> pd.DataFrame:

        # Convert to pandas for compatibility with pymer4
        df_pd = df.to_pandas()

        # Replace spaces in column headers and variable names with underscores
        df_pd.columns = df_pd.columns.str.replace(" ", "_")

        response_col_name = LinearMixedModel._replace_spaces(self.response_col_name)
        fixed_effects = LinearMixedModel._replace_spaces(self.fixed_effects)
        study_vars = LinearMixedModel._replace_spaces(self.study_effects["slopes"])
        block_vars = LinearMixedModel._replace_spaces(self.block_effects["slopes"])

        # Create model formula string, depending on fixed and random effects
        fe_formula = " + ".join(fixed_effects)
        formula = f"{response_col_name} ~ {fe_formula}"

        # Study-level random effects
        if self.study_effects["type"] == "slope":
            if self.random_effects_type == "independent":
                # Independent random effects for each variable in study_vars
                re_study_formula = "(1|SS) + "
                re_study_formula += " + ".join(f"({var}-1|SS)" for var in study_vars)
            else:
                # Correlated random effects, combining study_vars
                re_study_vars = " + ".join(study_vars)
                re_study_formula = f"({re_study_vars}|SS)"
            formula += f" + {re_study_formula}"

        else:
            # Random intercept for study only
            formula += " + (1|SS)"

        # Block-level random effects
        if self.block_effects["type"] == "slope":
            if self.random_effects_type == "independent":
                re_block_formula = "(1|SS:SSB) + "
                # Independent random effects for each variable in block_vars
                re_block_formula += " + ".join(
                    f"({var}-1|SS:SSB)" for var in block_vars
                )
            else:
                # Correlated random effects, combining block_vars
                re_block_vars = " + ".join(block_vars)
                re_block_formula = f"({re_block_vars}|SS:SSB)"
            formula += f" + {re_block_formula}"

        else:
            # Random intercept for block only
            formula += " + (1|SS:SSB)"

        return df_pd, formula

    @staticmethod
    def _replace_spaces(input_data: Union[str, list[str]]) -> Union[str, list[str]]:
        """Replace spaces with underscores in strings or lists of strings."""
        if isinstance(input_data, str):
            return input_data.replace(" ", "_")
        elif isinstance(input_data, list):
            return [item.replace(" ", "_") for item in input_data]
        else:
            raise TypeError("Input data must be a string or a list of strings.")

    def train_pymer_model(self, df: pd.DataFrame, formula: str) -> Lmer:
        start = time.time()
        self.logger.info("Fitting linear mixed model.")

        model = Lmer(formula=self.model_formula, data=df, family="gaussian")
        if self.mode == "crossval":
            conf_int = "Wald"  # Avoid costly bootstrapping for CV
            summary = False
        else:
            conf_int = self.solver["conf_int"]
            summary = self.solver["summary"]

        model.fit(
            conf_int=conf_int,
            nsim=self.solver["nsim"],
            summary=summary,
            REML=self.solver["REML"],
        )

        runtime = str(timedelta(seconds=int(time.time() - start)))
        self.logger.info(f"Model fitting completed in {runtime}.")

        return model

    def make_predictions(
        self, prediction_data: pd.DataFrame, mode: str
    ) -> dict[str, np.array]:
        pred_with_re = self.model_instance.predict(
            prediction_data,
            use_rfx=True,
            skip_data_checks=True,
            verify_predictions=False,
        )
        pred_only_fe = self.model_instance.predict(
            prediction_data,
            use_rfx=False,
            skip_data_checks=True,
            verify_predictions=False,
        )
        predictions = {
            "pred_with_re": np.array(pred_with_re),
            "pred_only_fe": np.array(pred_only_fe),
        }

        return predictions

    def create_prediction_dataframe(
        self, prediction_data: pd.DataFrame, predictions: dict[str, np.array]
    ) -> pl.DataFrame:
        # Extract data from dataframe
        sites = prediction_data["SSBS"]
        response_col_name = LinearMixedModel._replace_spaces(self.response_col_name)
        y_obs = prediction_data[response_col_name]

        # Get predictions
        y_pred_re = predictions["pred_with_re"]
        y_pred_fe = predictions["pred_only_fe"]

        df_pred = pl.DataFrame(
            {
                "SSBS": sites,
                "Observed": y_obs,
                "Predicted_RE": y_pred_re,
                "Predicted_FE": y_pred_fe,
            }
        )

        return df_pred
