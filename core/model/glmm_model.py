import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Union, cast

import polars as pl

from core.model.model_utils import (
    standardize_continuous_covariates,
    validate_design_matrix_columns,
)


class GeneralizedLinearMixedModel:
    """
    Generalized linear mixed model backed by glmmTMB via an R runner.

    This class mirrors the LinearMixedModel interface but delegates fitting
    and prediction to R to support non-Gaussian likelihoods (e.g. beta).
    """

    def __init__(
        self,
        mode: str,
        random_seed: int,
        epsilon: float,
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        logger: logging.Logger,
        run_folder_path: str,
    ) -> None:
        """
        Attributes:
            mode: Either 'training' or 'crossval'.
            random_seed: Random seed for model reproducibility.
            epsilon: Small value to prevent numerical issues in models, e.g.
                when using beta likelihood.
            model_settings: GLMM settings from model_configs.yaml.
            model_vars: Response variable and covariates for the model.
            logger: Logger for run output.
            run_folder_path: Base path for run outputs and temp files.
        """
        self.mode = mode
        self.random_seed = random_seed
        self.epsilon = epsilon
        self.model_settings = model_settings
        self.model_vars = model_vars
        self.logger = logger
        self.run_folder_path = run_folder_path

        # Model covariates
        self.response_var = model_vars["response_var"]
        self.categorical_vars = model_vars["categorical_vars"]
        self.continuous_vars = model_vars["continuous_vars"]
        self.interaction_terms = model_vars["interaction_terms"]
        self.slope_terms = model_vars.get("slope_terms", [])

        # Fixed effects
        self.response_col_name = self.response_var
        self.fixed_effects = (
            self.categorical_vars + self.continuous_vars + self.interaction_terms
        )

        # Random effects settings
        self.random_effects_type = self.model_settings["random_effects_type"]
        self.study_effects = self.model_settings["study_effects"]
        self.block_effects = self.model_settings["block_effects"]

        # GLMM settings
        self.family = self.model_settings.get("family", "beta")
        self.link = self.model_settings.get("link", "logit")
        self.effect_ci_settings = self.model_settings.get("effect_ci", {})
        self.effect_ci_method = self.effect_ci_settings.get("method", "Wald")
        self.effect_ci_n_boot = int(self.effect_ci_settings.get("n_boot", 200))
        self.effect_ci_seed = self.random_seed
        self.effect_ci_re_lower = 5.0
        self.effect_ci_re_upper = 95.0
        self.r_runner_path = os.path.join(
            os.path.dirname(__file__), "r_backend", "glmm_tmb_runner.R"
        )
        self.model_rds_path = ""

    def prepare_data(
        self, df_train: pl.DataFrame, df_test: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Standardize continuous covariates and build an R-compatible formula.

        Returns dataframes ready to be written to parquet and consumed by
        the glmmTMB runner.
        """
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        validate_design_matrix_columns(df_train, df_test, x_vars)
        df_train_std, df_test_std = standardize_continuous_covariates(
            df_train,
            df_test,
            vars_to_standardize=self.continuous_vars + self.interaction_terms,
        )
        df_train_model, self.model_formula = self.format_data_for_r_model(df_train_std)
        df_test_model, _ = self.format_data_for_r_model(df_test_std)

        if self.family == "beta":
            df_train_model = self._clip_beta_response(df_train_model)
            df_test_model = self._clip_beta_response(df_test_model)

        return df_train_model, df_test_model

    def fit(self, df_train: pl.DataFrame) -> None:
        """
        Fit a glmmTMB model in R and persist it as an RDS file.
        """
        self._validate_family_link()
        self._validate_slope_terms()

        self.model_rds_path = os.path.join(self.run_folder_path, "glmm_model.rds")
        training_data_path = self._create_temp_path(suffix=".parquet")
        df_train.write_parquet(training_data_path)

        try:
            self._run_rscript(
                [
                    "--mode=fit",
                    f"--training-data-path={training_data_path}",
                    f"--model-path={self.model_rds_path}",
                    f"--formula={self.model_formula}",
                    f"--family={self.family}",
                    f"--link={self.link}",
                    f"--categorical-vars={','.join(self._categorical_factor_vars())}",
                    f"--seed={self.random_seed}",
                ]
            )
        finally:
            if os.path.exists(training_data_path):
                os.remove(training_data_path)

        self.model_instance = self.model_rds_path

    def extract_effects(self) -> dict[str, Any]:
        """
        Extract fixed-effect summaries and study-level random-slope ranges.

        Confidence interval method and percentile ranges are controlled by
        run_settings.glmm.effect_ci in model config.
        """
        if not self.model_rds_path:
            raise ValueError("Model has not been fit; no RDS path is available.")

        effects_output_path = self._create_temp_path(suffix=".json")
        self.logger.info(
            "Extracting GLMM effects (ci_method=%s, n_boot=%s).",
            self.effect_ci_method,
            self.effect_ci_n_boot,
        )
        self._run_rscript(
            [
                "--mode=extract-effects",
                f"--model-path={self.model_rds_path}",
                f"--effects-output-path={effects_output_path}",
                f"--ci-method={self.effect_ci_method}",
                f"--n-boot={self.effect_ci_n_boot}",
                f"--seed={self.effect_ci_seed}",
                f"--re-lower-perc={self.effect_ci_re_lower}",
                f"--re-upper-perc={self.effect_ci_re_upper}",
            ]
        )
        with open(effects_output_path) as f:
            effects = json.load(f)
        os.remove(effects_output_path)
        return effects

    def extract_beta_phi(self) -> float:
        """
        Extract beta precision (phi) from the fitted GLMM model.
        """
        if not self.model_rds_path:
            raise ValueError("Model has not been fit; no RDS path is available.")
        if self.family != "beta":
            raise ValueError("Beta phi is only defined for beta-family GLMM.")

        phi_output_path = self._create_temp_path(suffix=".json")
        self._run_rscript(
            [
                "--mode=extract-phi",
                f"--model-path={self.model_rds_path}",
                f"--phi-output-path={phi_output_path}",
            ]
        )
        with open(phi_output_path) as f:
            out = json.load(f)
        os.remove(phi_output_path)
        return float(out["phi"])

    def predict(self, prediction_data: pl.DataFrame, pred_mode: str) -> pl.DataFrame:
        """
        Generate predictions using the fitted R model.

        For cross-validation, fixed effects are used when pred_mode is "test",
        while random effects are used when pred_mode is "train".
        """
        prediction_data_path = self._create_temp_path(suffix=".parquet")
        prediction_output_path = self._create_temp_path(suffix=".parquet")
        prediction_data.write_parquet(prediction_data_path)

        try:
            self._run_rscript(
                [
                    "--mode=predict",
                    f"--prediction-data-path={prediction_data_path}",
                    f"--model-path={self.model_rds_path}",
                    f"--prediction-output-path={prediction_output_path}",
                    f"--family={self.family}",
                    f"--link={self.link}",
                    f"--categorical-vars={','.join(self._categorical_factor_vars())}",
                    f"--seed={self.random_seed + 1}",
                ]
            )

            df_pred = pl.read_parquet(prediction_output_path)
            # Keep row-wise alignment between predictions and observed values.
            # Joining on SSBS can create a many-to-many expansion for beta data
            # where one SSBS may appear with multiple reference sites/taxa.
            if df_pred.height != prediction_data.height:
                raise ValueError(
                    "Prediction output row count does not match input data. "
                    f"pred={df_pred.height}, input={prediction_data.height}"
                )

            df_pred = df_pred.with_columns(
                prediction_data.get_column(self.response_col_name).alias("Observed")
            )

            # Preserve identifiers needed to keep beta outputs uniquely keyed.
            id_cols = ["Primary_minimal_site", "Custom_taxonomic_group"]
            for col in id_cols:
                if col in prediction_data.columns and col not in df_pred.columns:
                    df_pred = df_pred.with_columns(
                        prediction_data.get_column(col).alias(col)
                    )

            return df_pred
        finally:
            for tmp_path in (prediction_data_path, prediction_output_path):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def format_data_for_r_model(self, df: pl.DataFrame) -> tuple[pl.DataFrame, str]:
        """
        Prepare the dataframe and construct the glmmTMB formula string.

        Replaces spaces in column names to ensure formula compatibility.
        """
        df_res = self._rename_columns(df)

        self.response_col_name = cast(str, self._replace_spaces(self.response_col_name))
        self.categorical_vars = cast(
            list[str], self._replace_spaces(self.categorical_vars)
        )
        self.continuous_vars = cast(
            list[str], self._replace_spaces(self.continuous_vars)
        )
        self.interaction_terms = cast(
            list[str], self._replace_spaces(self.interaction_terms)
        )
        self.slope_terms = cast(list[str], self._replace_spaces(self.slope_terms))
        self.fixed_effects = (
            self.categorical_vars + self.continuous_vars + self.interaction_terms
        )

        response_col_name = self.response_col_name
        fixed_effects = self.fixed_effects
        fe_formula = " + ".join(fixed_effects)
        formula = f"{response_col_name} ~ {fe_formula}"

        # Study-level random effects
        if self.study_effects["type"] == "slope":
            study_vars = self.slope_terms
            if self.random_effects_type == "independent":
                re_study_formula = "(1|SS) + "
                re_study_formula += " + ".join(f"({var}-1|SS)" for var in study_vars)
            else:
                re_study_vars = " + ".join(study_vars)
                re_study_formula = f"({re_study_vars}|SS)"
            formula += f" + {re_study_formula}"
        else:
            formula += " + (1|SS)"

        # Block-level random effects
        if self.block_effects["type"] == "slope":
            block_vars = self.slope_terms
            if self.random_effects_type == "independent":
                re_block_formula = "(1|SS:SSB) + "
                re_block_formula += " + ".join(
                    f"({var}-1|SS:SSB)" for var in block_vars
                )
            else:
                re_block_vars = " + ".join(block_vars)
                re_block_formula = f"({re_block_vars}|SS:SSB)"
            formula += f" + {re_block_formula}"
        elif self.block_effects["type"] == "intercept":
            formula += " + (1|SS:SSB)"

        return df_res, formula

    def _validate_family_link(self) -> None:
        """
        Validate that the configured family/link combination is supported.
        """
        allowed_links = {
            "beta": {"logit"},
            "gaussian": {"identity"},
        }
        fam = self.family
        link = self.link
        if fam not in allowed_links:
            raise ValueError(
                f"Unsupported family: {fam}. Allowed: {sorted(allowed_links)}"
            )
        if link not in allowed_links[fam]:
            raise ValueError(
                f"Unsupported link '{link}' for family '{fam}'. "
                f"Allowed links: {sorted(allowed_links[fam])}"
            )

    def _validate_slope_terms(self) -> None:
        """
        Ensure slope terms are provided when random slopes are requested.
        """
        wants_study_slopes = self.study_effects.get("type") == "slope"
        wants_block_slopes = self.block_effects.get("type") == "slope"
        if (wants_study_slopes or wants_block_slopes) and not self.slope_terms:
            raise ValueError(
                "Random slope requested (study_effects or block_effects set "
                "to 'slope') but slope_terms is empty."
            )

    @staticmethod
    def _replace_spaces(input_data: Union[str, list[str]]) -> Union[str, list[str]]:
        """
        Replace spaces with underscores in strings or lists of strings.
        """
        if isinstance(input_data, str):
            return input_data.replace(" ", "_")
        if isinstance(input_data, list):
            return [item.replace(" ", "_") for item in input_data]
        raise TypeError("input_data must be a string or list of strings")

    @staticmethod
    def _rename_columns(df: pl.DataFrame) -> pl.DataFrame:
        """
        Normalize dataframe column names for R formula compatibility.
        """
        rename_map = {col: col.replace(" ", "_") for col in df.columns}
        return df.rename(rename_map)

    def _categorical_factor_vars(self) -> list[str]:
        """
        Return categorical columns that must be treated as factors in R.
        """
        # Ensure grouping variables are treated as factors in R.
        return sorted(set(self.categorical_vars + ["SS", "SSB"]))

    def _create_temp_path(self, suffix: str) -> str:
        """
        Create a temporary file path in the run directory.
        """
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.run_folder_path)
        # Close the descriptor immediately; only the path is used downstream.
        os.close(fd)
        return path

    def _run_rscript(self, args: list[str]) -> None:
        """
        Execute the glmmTMB R backend with the provided CLI arguments.
        """
        cmd = ["Rscript", self.r_runner_path] + args
        self.logger.info("Running R glmmTMB runner.")
        subprocess.run(cmd, check=True)

    def _clip_beta_response(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Clip beta response values to the open interval (0, 1) using epsilon.
        """
        lower = self.epsilon
        upper = 1 - self.epsilon
        response_col = self.response_col_name

        n_below = df.select((pl.col(response_col) <= lower).sum()).to_series(0).item()
        n_above = df.select((pl.col(response_col) >= upper).sum()).to_series(0).item()

        n_clipped = n_below + n_above
        if n_clipped > 0:
            self.logger.info(
                "Clipping %s beta response values in '%s' to [%s, %s]. "
                "(below=%s, above=%s)",
                n_clipped,
                response_col,
                lower,
                upper,
                n_below,
                n_above,
            )

        return df.with_columns(
            pl.col(response_col)
            .clip(lower_bound=lower, upper_bound=upper)
            .alias(response_col)
        )
