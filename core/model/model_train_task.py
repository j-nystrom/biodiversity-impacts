import os
import time
from datetime import timedelta

import dill
import polars as pl
import yaml
from box import Box

from core.model.model_eval import calculate_performance_metrics
from core.model.model_functions import (
    format_data_for_model,
    inverse_transform_response,
    make_in_sample_predictions,
    run_sampling,
    summarize_sampling_statistics,
)
from core.model.models import bii_abund_baseline
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "model_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class ModelTrainingTask:
    """
    Task for training and evaluating a hiearchical Bayesian model. The PyMC
    models are specified in the 'models.py' module and fetched based on the
    specified model in the config.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            run_folder_path: Path to folder where to read and save model data.
            sampler_settings: Required settings for the PyMC MCMC sampler.
            model_dict: Mapping between model names and model objects, used for
                fetching the correct model object to use.
            taxonomic_resolution: The level of taxonomic resolution to be used.
            group_vars: Variables used to group the data. Should include the
                taxonomic resolution as the final key.
            model_spec: The name of the model settings dictionary to be used.
                This includes the model object as well as all variables.
            response_var: The response variable to be used in the model.
            categorical_vars: All categorical covariates to be included.
            continuous_vars: All continuous covariates to be included, incl.
                any transformations (part of the column names).
            response_var_transform: The transformation to be applied to the
                response variable.
            pymc_key_var_names: The names of the key parameters in the PyMC
                model, that should be included in the output.
        """
        # General configs applicable to different models
        self.run_folder_path = run_folder_path
        self.sampler_settings = configs.sampler_settings

        # Dictionary mapping model objects to names recognized in the config
        self.model_dict = {
            "bii_abund_baseline": bii_abund_baseline,
        }

        # Taxonomic resolution specific configs
        self.taxonomic_resolution = configs.taxonomic_resolution
        self.group_vars: list[str] = configs.group_vars[self.taxonomic_resolution]

        # Model specific configs
        selected_model: str = configs.selected_model
        model_config = configs[selected_model]
        self.model_spec: str = model_config.model_spec
        self.response_var: str = model_config.response_var
        self.categorical_vars: list[str] = model_config.categorical_vars
        self.continuous_vars: list[str] = model_config.continuous_vars
        self.response_var_transform: str = model_config.response_var_transform
        self.pymc_key_var_names: list[str] = model_config.pymc_key_var_names

    def run_task(self) -> None:
        """
        Runs a set of model training and evaluation steps that are specifically
        tailored to hierarchical Bayesian models implemented in PyMC:
        1. The model data outputted from the previous pipeline step is read and
        formatted in a way expected by PyMC models.
        2. The relevant PyMC model from 'models.py' is fetched based on the
        model name specified in the config, and the model is instantiated.
        3. The MCMC sampler is run on the model to generate a trace. Sampling
        statistics are summarized and printed to the console to evaluate
        convergence of the sampling chains.
        4. In-sample predictions are made using the trace and performance
        metrics are calculated and reported.
        5. The model, trace and accompanying data are saved to the run folder,
        together with the model configs for reproducibility.
        """
        logger.info(
            f"Training {self.response_var[0]} model with spec {self.model_spec}"
        )
        start = time.time()

        df = pl.read_parquet(os.path.join(self.run_folder_path, "model_data.parquet"))

        # Regenerate interaction terms created in last step for compatibility
        # TODO: Should find better solution for this
        interaction_terms = []
        for col_1 in self.continuous_vars:
            for col_2 in self.categorical_vars:
                interaction_terms.append(f"{col_2} x {col_1}")

        # Format the data in a way that can be used by the PyMC model
        x, y, coords, group_idx, group_code_map, study_idx, block_idx = (
            format_data_for_model(
                df,
                group_vars=self.group_vars,
                response_var=self.response_var,
                response_var_transform=self.response_var_transform,
                categorical_vars=self.categorical_vars,
                continuous_vars=self.continuous_vars,
                interaction_terms=interaction_terms,
            )
        )

        # Fetch the model object and instantiate it
        model_object = self.model_dict[self.model_spec]
        model = model_object(x, y, coords, group_idx, study_idx, block_idx)

        # Run the MCMC sampler on the model and summarize sampling statistics
        trace = run_sampling(model, self.sampler_settings)
        summarize_sampling_statistics(trace, var_names=self.pymc_key_var_names)

        # Make in-sample predictions and calculate performance metrics
        trace, p_pred = make_in_sample_predictions(model, trace, x, y, group_idx)
        if self.response_var_transform:  # TBD if this is necessary for train
            y = inverse_transform_response(y, method=self.response_var_transform)
            p_pred = inverse_transform_response(
                p_pred, method=self.response_var_transform
            )
        calculate_performance_metrics(y, p_pred, group_idx, group_code_map, plot=True)

        # Save the model and trace so they can be analyzed later
        # TODO: At some point this should be done in a more robust way:
        # https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
        output_dict = {
            "model": model,
            "trace": trace,
            "pymc_var_names": self.pymc_key_var_names,
            "covariates": self.categorical_vars + self.continuous_vars,
            "group_mapping": group_code_map,
            "group_idx": group_idx,
            "y_true": y,
            "y_pred": p_pred,
        }

        # Write outputs to the run folder
        with open(
            os.path.join(self.run_folder_path, "model_output.pkl"), "wb"
        ) as out_stream:
            dill.dump(output_dict, out_stream)

        with open(
            os.path.join(self.run_folder_path, "model_configs.yaml"), "w"
        ) as outfile:
            yaml.dump(configs, outfile, default_flow_style=False)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Model training and evaluation completed in {runtime}.")
