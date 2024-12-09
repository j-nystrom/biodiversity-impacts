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
)
from core.model.model_functions import (
    create_interaction_terms,
    create_stratification_column,
    filter_data_scope,
    filter_out_small_groups,
    format_data_for_model,
    generate_kfolds,
    run_sampling,
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


class CrossValidationTask:
    """
    Task for running k-fold cross-validation using a hierachical Bayesian
    model. The PyMC models are specified in the 'hierarchical_models.py' module
    and fetched based on the specified model in the config.
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
            k: Number of folds to be used in cross-validation.
            stratify: Whether to stratify folds based on a grouping variable.
            stratify_groups: The grouping variable to stratify the folds on.

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
        self.k = configs.cross_validation.k
        self.stratify = configs.cross_validation.stratify
        self.stratify_groups = configs.cross_validation.stratify_groups

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
        selected_model: str = configs.model_settings.selected_model
        model_config = configs[selected_model]
        self.response_var: str = model_config.response_var
        self.response_var_transform: str = model_config.response_var_transform
        self.categorical_vars: list[str] = model_config.categorical_vars
        self.continuous_vars: list[str] = model_config.continuous_vars
        self.interaction_cols: list[str] = model_config.interaction_cols

    def run_task(self) -> None:
        logger.info(
            f"Cross-validating {self.response_var[0]} model with spec {self.model_spec}"
        )
        start = time.time()

        # TODO: First 4 steps violate DRY (don't repeat yourself), since they
        # are identical to the steps in the model training task.

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

        # Select the response variable, covariates and grouping variables
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

        # Generate a column that should be used for stratification in K-folds,
        # and generate the folds
        if self.stratify:
            df = create_stratification_column(df, stratify_groups=self.stratify_groups)
        df_train_list, df_test_list = generate_kfolds(
            df,
            k=self.k,
            stratify=self.stratify,
        )

        # Create a fixed mapping between site names and indices
        # TODO: Check if this sorting is really needed for end results
        site_names = df["SSBS"].unique().to_list()
        site_name_to_idx = {site_name: idx for idx, site_name in enumerate(site_names)}

        # Prepare training and test data corresponding to each split
        training_data_list = []
        test_data_list = []
        for df_train, df_test in zip(df_train_list, df_test_list):

            # Start with training data preparation
            # Standardization done separately to avoid data leakage
            df_train = standardize_continuous_covariates(
                df_train, vars_to_scale=self.continuous_vars
            )
            df_train, self.interaction_terms = create_interaction_terms(
                df_train,
                categorical_vars=self.categorical_vars,
                continuous_vars=self.interaction_cols,
            )
            training_data = format_data_for_model(
                df_train,
                response_var=self.response_var,
                categorical_vars=self.categorical_vars,
                continuous_vars=self.continuous_vars,
                interaction_terms=self.interaction_terms,
                site_name_to_idx=site_name_to_idx,
            )
            training_data_list.append(training_data)

            # Continue with test data preparation
            df_test = standardize_continuous_covariates(
                df_test, vars_to_scale=self.continuous_vars
            )
            df_test, self.interaction_terms = create_interaction_terms(
                df_test,
                categorical_vars=self.categorical_vars,
                continuous_vars=self.interaction_cols,
            )
            test_data = format_data_for_model(
                df_test,
                response_var=self.response_var,
                categorical_vars=self.categorical_vars,
                continuous_vars=self.continuous_vars,
                interaction_terms=self.interaction_terms,
                site_name_to_idx=site_name_to_idx,
            )
            test_data_list.append(test_data)

        train_predictions = []
        train_observed = []
        test_predictions = []
        test_observed = []
        test_traces = []
        for i, (training_data, test_data) in enumerate(
            zip(training_data_list, test_data_list)
        ):

            logger.info(f"Running CV for fold {i + 1} of {self.k}")
            start = time.time()

            model_class = self.model_dict[self.model_spec]
            model = model_class().model(training_data)

            # Run the MCMC sampler on the model
            train_trace = run_sampling(model, sampler_settings=self.sampler_settings)

            # Make predictions on the test and training sets and evaluate model
            # performance on each fold
            train_scores = {}
            test_scores = {}

            # Training data
            logger.info("Predicting and evaluating on training set")
            pred_train, _ = make_predictions(
                model=model,
                trace=train_trace,
                mode="train",
                data=training_data,
            )
            train_predictions.append(pred_train)
            train_observed.append(training_data["y"])

            train_performance = evaluate_model_performance(
                y_pred=pred_train,
                y_true=training_data["y"],
                trace=train_trace,
                response_transform=self.response_var_transform,
            )
            log_model_performance(train_performance)
            train_scores[i] = train_performance

            # Test data
            logger.info("Predicting and evaluating on test set")
            pred_test, test_trace = make_predictions(
                model=model,
                trace=train_trace,
                mode="test",
                data=test_data,
            )
            test_predictions.append(pred_test)
            test_observed.append(test_data["y"])
            test_traces.append(test_trace)

            test_performance = evaluate_model_performance(
                y_pred=pred_test,
                y_true=test_data["y"],
                trace=test_trace,
                response_transform=self.response_var_transform,
            )
            log_model_performance(test_performance)
            test_scores[i] = test_performance

            logger.info(
                f"Training for fold {i + 1} completed in "
                f"{timedelta(seconds=time.time() - start)}"
            )

        output_dict = {
            "train_predictions": train_predictions,
            "train_observed": train_observed,
            "test_predictions": test_predictions,
            "test_observed": test_observed,
            "train_scores": train_scores,
            "test_scores": test_scores,
            "test_traces": test_traces,
            "site_name_to_idx": site_name_to_idx,  # TODO: Check if still needed
        }

        # Write outputs to the run folder
        with open(
            os.path.join(self.run_folder_path, "cv_output.pkl"), "wb"
        ) as out_stream:
            dill.dump(output_dict, out_stream)

        # Also save the model configuration for reproducibility
        with open(
            os.path.join(self.run_folder_path, "cv_configs.yaml"), "w"
        ) as outfile:
            yaml.dump(configs, outfile, default_flow_style=False)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Cross-validation completed in {runtime}.")
