import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.model.model_functions import (
    add_intercept,
    create_interaction_terms,
    filter_data_scope,
    standardize_continuous_covariates,
    transform_response_variable,
)
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "model_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class ModelDataTask:
    """
    Task to prepare data in a format suitable for model training and cross-
    validation. This step precedes both ModelTrainTask and CrossValidationTask.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            run_folder_path: Path to the folder where model data is saved.
            geographic_scope: Dictionary to filter the geographic scope, e.g.
                certain biomes and regions.
            taxonomic_resolution: The level of taxonomic resolution to be used.
                This determines the feature data file to be used.
            feature_data: Path to file containing response and covariate data,
                for this level of taxonomic resolution.
            species_scope: Species groups to be included for the chosen level
                of taxonomic resolution.
            group_vars: Variables used to group the data. Should include the
                taxonomic resolution as the final key.
            response_var: The response variable to be used in the model.
            categorical_vars: All categorical covariates to be included.
            continuous_vars: All continuous covariates to be included, incl.
                any transformations (part of the column names).
            response_var_transform: The transformation to be applied to the
                response variable.
        """
        # General configs applicable to different models
        self.run_folder_path: str = run_folder_path
        self.geographic_scope: dict[str, list[str]] = configs.scope.geographic

        # Taxonomic resolution specific configs
        self.taxonomic_resolution: str = configs.taxonomic_resolution
        self.feature_data: str = configs.abundance_data[self.taxonomic_resolution]
        self.species_scope: list[str] = configs.scope.species[self.taxonomic_resolution]
        self.group_vars: list[str] = configs.group_vars[self.taxonomic_resolution]

        # Model specific configs
        model_config = configs[configs.selected_model]
        self.response_var: str = model_config.response_var
        self.categorical_vars: list[str] = model_config.categorical_vars
        self.continuous_vars: list[str] = model_config.continuous_vars
        self.response_var_transform: str = model_config.response_var_transform

    def run_task(self) -> None:
        """
        Runs a sequence of steps to create a dataframe that can be used as
        input the model:
        1. Read the combined data from the previous pipeline step.
        2. Filter the data based on the geographic and species scope.
        3. Select the response variable, covariates and grouping variables.
        4. Transform the response variable if required.
        5. Standardize continuous covariates.
        6. Create interaction terms between categorical and continuous vars.
        7. Add an intercept term to the design matrix.
        8. Write the resulting dataframe to a parquet file.
        """
        logger.info("Initiating model data preparation.")
        start = time.time()

        df = pl.read_parquet(self.feature_data)

        # Filter data based on geographic and species scope
        df = filter_data_scope(
            df,
            taxonomic_resolution=self.taxonomic_resolution,
            geographic_scope=self.geographic_scope,
            species_scope=self.species_scope,
        )

        # Select the response variable, covariates and grouping variables
        df = df.select(
            self.group_vars
            + [self.response_var]
            + self.categorical_vars
            + self.continuous_vars
        )

        # If specified in the config, the response variable is transformed
        df = transform_response_variable(df, method=self.response_var_transform)

        # Continuous vars are standardized to have mean zero and unit variance
        df = standardize_continuous_covariates(df, self.continuous_vars)

        # Create interaction terms between categorical and continuous vars
        df = create_interaction_terms(df, self.categorical_vars, self.continuous_vars)

        # Add an intercept term to the design matrix
        df = add_intercept(df, response_var=self.response_var)

        # Write the output to the run folder
        df.write_parquet(os.path.join(self.run_folder_path, "model_data.parquet"))

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Model data preparation finished in {runtime}.")
