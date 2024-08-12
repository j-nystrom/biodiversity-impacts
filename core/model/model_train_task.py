import os
import time
from datetime import timedelta

import dill
import polars as pl
import yaml
from box import Box

from core.model.hierarchical_models import BiomeRealmEcoSlopeModel
from core.model.model_eval import (
    evaluate_model_performance,
    log_model_performance,
    make_predictions,
    plot_prior_distribution,
)
from core.model.model_functions import (
    create_interaction_terms,
    filter_data_scope,
    filter_out_small_groups,
    format_data_for_model,
    run_sampling,
    standardize_continuous_covariates,
    summarize_sampling_statistics,
    transform_response_variable,
)
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
    models are specified in the 'hierarchical_models.py' module and fetched
    based on the specified model in the config.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            run_folder_path: Path to folder where to read and save model data
                and outputs.
            geographic_scope: The geographic scope to be used in the model
                (biomes and UN regions).
            sampler_settings: Required settings for the PyMC MCMC sampler.
            group_vars: Variables used to group the data / as identifiers.
            model_spec: Model object in hierarchical_models.py that is used.
            group_size_threshold: Minimum group size to be used in hierarchy
                (if a biome-realm-ecoregion is smaller, it's filtered out).

            taxonomic_resolution: The level of taxonomic granularity to be used,
                from all species down to Family level.
            species_scope: The species scope to be used in the model, based on
                taxonomic resolution chosen.
            taxonomic_vars: Taxonomic grouping categories to be added, based on
                the selected species scope.
            feature_data: The path to the feature data to be used in the model,
                for this specific taxonomic resolution.
            model_dict: Mapping between model names and model objects, used for
                fetching the correct model object to use.

            selected_model: Model configuration (variables) used for training.
            response_var: The response variable to be used in the model.
            response_var_transform: The transformation to be applied to the
                response variable.
            categorical_vars: All categorical covariates to be included.
            continuous_vars: All continuous covariates to be included, incl.
                any transformations (part of the column names).
            interaction_cols: Continuous variables to be used for creating
                interaction terms with categorical variables.
        """
        # General configs applicable to different models
        self.run_folder_path = run_folder_path
        self.geographic_scope: dict[str, list[str]] = configs.scope.geographic
        self.sampler_settings: dict[str, int | str | float] = configs.sampler_settings
        self.group_vars: list[str] = configs.group_vars
        self.model_spec: str = configs.model_settings.model_spec
        self.group_size_threshold: int = configs.model_settings.group_size_threshold

        # Taxonomic resolution specific configs
        self.taxonomic_resolution = configs.taxonomic_resolution
        self.species_scope: list[str] = configs.scope.species[self.taxonomic_resolution]
        self.taxonomic_vars: list[str] = configs.taxonomic_vars[
            self.taxonomic_resolution
        ]
        self.feature_data: str = configs.abundance_data[self.taxonomic_resolution]

        # Dictionary that maps model objects to names recognized in the config
        self.model_dict = {
            "biome_realm_eco_slope": BiomeRealmEcoSlopeModel,
        }

        # Model specific configs
        selected_model: str = configs.selected_model
        model_config = configs[selected_model]
        self.response_var: str = model_config.response_var
        self.response_var_transform: str = model_config.response_var_transform
        self.categorical_vars: list[str] = model_config.categorical_vars
        self.continuous_vars: list[str] = model_config.continuous_vars
        self.interaction_cols: list[str] = model_config.interaction_cols

    def run_task(self) -> None:
        """
        Runs a set of model training and evaluation steps that are specifically
        tailored to hierarchical Bayesian models implemented in PyMC:

        1. The model data outputted from the previous pipeline step is read and
            formatted in a way expected by PyMC models.
        2. The relevant PyMC model is fetched based on the model name specified
            in the config, and the model is instantiated.
        3. Prior predictive checks are performed to ensure the model is set up
            correctly and the priors are reasonable.
        4. The MCMC sampler is run on the model to generate a trace. Sampling
            statistics are summarized and printed to the console to evaluate
            convergence of the sampling chains.
        5. In-sample predictions are made using the trace and performance
            metrics are calculated and reported.
        6. The model, trace and accompanying data are saved to the run folder,
            together with the model configs for reproducibility.
        """
        logger.info(
            f"Training {self.response_var[0]} model with spec {self.model_spec}"
        )
        start = time.time()

        # Get feature dataframe for this taxonomic granularity
        df = pl.read_parquet(self.feature_data)

        # Filter data based on geographic / species scope, remove small groups
        df = filter_data_scope(
            df,
            taxonomic_resolution=self.taxonomic_resolution,
            geographic_scope=self.geographic_scope,
            species_scope=self.species_scope,
        )
        if self.group_size_threshold > 0:
            df = filter_out_small_groups(df, threshold=self.group_size_threshold)

        # Select the grouping variables, response variable and covariates
        df = df.select(
            self.group_vars
            + self.taxonomic_vars
            + [self.response_var]
            + self.categorical_vars
            + self.continuous_vars
        )

        # If specified in the config, the response variable is transformed
        df = transform_response_variable(
            df, response_var=self.response_var, method=self.response_var_transform
        )

        # Continuous vars are standardized to have mean zero and unit variance
        df = standardize_continuous_covariates(df, vars_to_scale=self.continuous_vars)

        # Create interaction terms between categorical and continuous vars
        df, self.interaction_terms = create_interaction_terms(
            df,
            categorical_vars=self.categorical_vars,
            continuous_vars=self.interaction_cols,
        )

        # Save interim model data to run folder for external consumption
        df.write_parquet(os.path.join(self.run_folder_path, "model_data.parquet"))

        # Create a fixed mapping between site names and indices
        # TODO: Check if this is really needed. If yes, explain why
        site_names = df["SSBS"].unique().to_list()
        site_name_to_idx = {site_name: idx for idx, site_name in enumerate(site_names)}

        # Format the data in a way that can be consumed by the PyMC model
        model_data = format_data_for_model(
            df,
            response_var=self.response_var,
            categorical_vars=self.categorical_vars,
            continuous_vars=self.continuous_vars,
            interaction_terms=self.interaction_terms,
            site_name_to_idx=site_name_to_idx,
        )

        # Instantiate the right model class for fitting
        model_class = self.model_dict[self.model_spec]
        model = model_class().model(model_data)

        # Do prior predictive checks before starting sampling
        plot_prior_distribution(model)

        # Ask user if task should be continued based on prior predictive checks
        user_input = input("Continue training process? (y/n): ")
        if user_input.lower() == "n":
            print("Training aborted based on prior predictive checks.")

        else:
            # Run MCMC sampler on the model and summarize sampling statistics
            trace = run_sampling(model, sampler_settings=self.sampler_settings)
            summarize_sampling_statistics(trace)

            # Make in-sample predictions on the training data
            y_pred, trace = make_predictions(
                model=model,
                trace=trace,
                mode="train",
                data=model_data,
            )

            # Compute performance metrics based on in-sample predictions
            performance_metrics = evaluate_model_performance(
                y_pred=y_pred,
                y_true=model_data["y"],
                trace=trace,
                response_transform=self.response_var_transform,
            )
            log_model_performance(performance_metrics)

            # Save the model and trace so they can be analyzed later
            # TODO: This can potentially be done in a more robust way:
            # https://www.pymc.io/projects/examples/en/latest/howto/model_builder.html
            output_dict = {
                "model_data": model_data,
                "model": model,
                "trace": trace,
                "predictions": y_pred,
                "performance_metrics": performance_metrics,
            }

            # Write outputs to the run folder
            with open(
                os.path.join(self.run_folder_path, "model_output.pkl"), "wb"
            ) as out_stream:
                dill.dump(output_dict, out_stream)

            # Also save the model configuration for reproducibility
            with open(
                os.path.join(self.run_folder_path, "model_configs.yaml"), "w"
            ) as outfile:
                yaml.dump(configs, outfile, default_flow_style=False)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Model training and evaluation completed in {runtime}.")
