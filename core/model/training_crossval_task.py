import gc
import json
import os
import shutil
import time
from datetime import timedelta
from typing import Any, Union

import dill
import polars as pl
import yaml
from box import Box

from core.model.bhm_model import BayesianHierarchicalModel
from core.model.glmm_model import GeneralizedLinearMixedModel
from core.model.model_utils import calculate_performance_metrics
from core.tests.shared.validate_shared import (
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
model_config_path = os.environ.get(
    "MODEL_CONFIG_PATH", os.path.join(script_dir, "model_configs.yaml")
)
configs = Box.from_yaml(filename=model_config_path)

logger = create_logger(__name__)


class BaseModelTask:
    """
    Base class for model training and cross-validation tasks. This class
    handles the initialization of model settings, loading of input data,
    evaluation of model performance and saving of outputs.
    """

    def __init__(self, run_folder_path: str, mode: str) -> None:
        """
        Attributes:

        General:
            run_folder_path: Path to folder where all run outputs are stored.
            mode: Either 'training' or 'crossval'.
            random_seed: Random seed for model reproducibility.
            epsilon: Small value to prevent numerical issues in models, e.g.
                when using beta likelihood.

        Config settings used by all models and modes
            - diversity_type: Diversity metric to be used ('alpha' or 'beta').
            - model_type: Type of model to be used ('bayesian' or 'glmm').
            - model_settings: Model settings for the specified model type.
            - model_vars: Response variable and covariates for the model.
            - continuous_vars: List of covariates that are continuous.

        Shared data paths:
            - interaction_terms_path: Path to JSON file containing all
                interaction terms.
            - site_info_path: Path to file containing info about each site.

        Model-specific data paths
            - hierarchy_mapping_path: Path to JSON file containing a fixed
                mapping of parent-child relationships for hierarchical levels.
            - site_mapping_path: Mapping of site names to indices.
            - taxonomic_resolution: Used to check if taxon mapping is needed.
            - taxon_mapping_path: Path to JSON file containing a fixed mapping
                of taxon names to indices.
        """
        # General
        self.run_folder_path: str = run_folder_path
        self.mode: str = mode
        self.random_seed: int = configs.random_seed
        self.epsilon: float = configs.epsilon

        # Config settings used by all models and modes (but model-specific)
        self.diversity_type: str = configs.data_scope.diversity_type
        self.taxonomic_resolution: str = configs.data_scope.taxonomic.resolution
        self.model_type: str = configs.run_settings.model_type
        self.model_settings: dict[str, Any] = configs.run_settings[self.model_type]
        self.model_vars: dict[str, Any] = configs.model_variables[
            configs.run_settings.model_variables
        ]
        self.continuous_vars: list[str] = self.model_vars["continuous_vars"]

        # Shared data paths
        self.interaction_terms_path: str = os.path.join(
            run_folder_path, "interaction_terms.json"
        )
        self.site_info_path: str = os.path.join(run_folder_path, "site_info.parquet")

        # Model-specific data paths for the Bayesian hierarchical model
        if self.model_type == "bayesian":
            self.hierarchy_mapping_path: str = os.path.join(
                run_folder_path, "complete_hierarchy_mapping.json"
            )
            self.site_mapping_path: str = os.path.join(
                run_folder_path, "site_mapping.json"
            )
            self.save_models_and_traces: bool = self.model_settings[
                "save_models_and_traces"
            ]
            self.save_predictive_distributions: bool = self.model_settings[
                "save_predictive_distributions"
            ]
            if self.model_settings["rolled_up_predictions"]:
                self.rolled_up_mapping_path = os.path.join(
                    run_folder_path, "rolled_up_hierarchy_mapping.json"
                )
            if self.taxonomic_resolution != "All_species":
                self.taxon_mapping_path: str = os.path.join(
                    run_folder_path, "taxon_mapping.json"
                )

    def load_input_data(self) -> None:
        """Validate required input files and load shared resources."""
        validate_input_files(file_paths=[self.interaction_terms_path])
        # Load key information about each site for analysis
        self.df_site_info = pl.read_parquet(self.site_info_path)

        # Load generated interaction terms
        with open(self.interaction_terms_path) as f:
            self.model_vars["interaction_terms"] = json.load(f)

        # Load hierarchical mapping and site mapping if applicable
        if self.model_type == "bayesian":
            validate_input_files(file_paths=[self.hierarchy_mapping_path])
            with open(self.hierarchy_mapping_path) as f:
                self.hierarchy_mapping = json.load(f)
            if self.model_settings["rolled_up_predictions"]:
                validate_input_files(file_paths=[self.rolled_up_mapping_path])
                with open(self.rolled_up_mapping_path) as f:
                    self.rolled_up_mapping = json.load(f)

        if self.model_type == "bayesian":
            validate_input_files(file_paths=[self.site_mapping_path])
            with open(self.site_mapping_path) as f:
                self.site_name_to_idx = json.load(f)

            if self.taxonomic_resolution != "All_species":
                validate_input_files(file_paths=[self.taxon_mapping_path])
                with open(self.taxon_mapping_path) as f:
                    self.taxon_name_to_idx = json.load(f)

    def initialize_model(
        self,
    ) -> Union[
        BayesianHierarchicalModel,
        GeneralizedLinearMixedModel,
    ]:
        """Instantiate the appropriate model class based on the model type."""
        model_classes = {
            "bayesian": BayesianHierarchicalModel,
            "glmm": GeneralizedLinearMixedModel,
        }
        # Shared model class attributes
        model_init_kwargs = {
            "random_seed": self.random_seed,
            "epsilon": self.epsilon,
            "model_settings": self.model_settings,
            "model_vars": self.model_vars,
            "logger": logger,
            "mode": self.mode,
        }
        # Add specific attributes for Bayesian models
        if self.model_type == "bayesian":
            model_init_kwargs["hierarchy_mapping"] = self.hierarchy_mapping
            model_init_kwargs["site_name_to_idx"] = self.site_name_to_idx
            model_init_kwargs["save_predictive_distributions"] = (
                self.save_predictive_distributions
            )
            if self.model_settings["rolled_up_predictions"]:
                model_init_kwargs["rolled_up_mapping"] = self.rolled_up_mapping
            if self.taxonomic_resolution != "All_species":
                model_init_kwargs["taxon_name_to_idx"] = self.taxon_name_to_idx

        if self.model_type == "glmm":
            model_init_kwargs["run_folder_path"] = self.run_folder_path

        return model_classes[self.model_type](**model_init_kwargs)

    def save_outputs(
        self,
        outputs: list[dict[str, Any]],
        output_paths: list[str],
        save_config: bool = True,
    ) -> None:
        """Save outputs and configuration."""
        # Validate and save output dictionary
        for file, path in zip(outputs, output_paths):
            validate_output_files([path], [file for file in file.values()])
            with open(path, "wb") as out_stream:
                dill.dump(file, out_stream)

        if save_config:
            # Save model configurations for reproducibility
            config_output_path = os.path.join(
                self.run_folder_path, "model_configs.yaml"
            )
            validate_output_files([config_output_path], [configs])
            with open(config_output_path, "w") as outfile:
                yaml.dump(configs, outfile, default_flow_style=False)

    def save_glmm_model(self, model_path: str, output_name: str) -> None:
        """Save the fitted GLMM RDS file to the run folder."""
        if not model_path:
            raise ValueError("GLMM model path is empty")
        validate_input_files(file_paths=[model_path])
        output_path = os.path.join(self.run_folder_path, output_name)
        shutil.copy2(model_path, output_path)


class ModelTrainingTask(BaseModelTask):
    """
    Class for training a given model and evaluating its performance on the
    training data.
    """

    def __init__(self, run_folder_path: str, mode: str) -> None:
        """
        Attributes:
            training_data_path: Path to Parquet file containing training data,
                located inside run folder.
        """
        super().__init__(run_folder_path, mode)
        self.training_data_path: str = os.path.join(
            self.run_folder_path, "training_data.parquet"
        )

    def run_task(self) -> None:
        """
        Perform the following processing steps:
            - Load training dataframe and other input data from previous step.
            - Initialize model and prepare data for the specified model type.
            - Fit the model on training data and make in-sample predictions.
            - Evaluate model performance and save predictions, evaluation
                metrics and other outputs.
        """
        logger.info(
            f"Initiating training of '{self.model_type}' model "
            f"and diversity type '{self.diversity_type}'."
        )
        start = time.time()

        # Load dataframe, interaction terms and site mapping from run folder
        validate_input_files(file_paths=[self.training_data_path])
        df_train = pl.read_parquet(self.training_data_path)
        self.load_input_data()

        # Initialize model, prepare data, and train it
        logger.info("Preparing model data and training the model.")
        model = self.initialize_model()
        train_data, _ = model.prepare_data(df_train, df_train)
        model.fit(train_data)

        # Make predictions on in-sample data and evaluate performance
        logger.info("Making predictions and evaluating model performance.")
        if self.model_type == "bayesian":
            df_pred, df_pred_distr = model.predict(train_data, pred_mode="train")
        else:
            df_pred = model.predict(train_data, pred_mode="train")

        pred_metrics = calculate_performance_metrics(
            df_pred,
            model_type=self.model_type,
            mode=self.mode,
        )

        # Write predictions to disk for consistent loading
        key_output_dir = os.path.join(self.run_folder_path, "key_output")
        os.makedirs(key_output_dir, exist_ok=True)
        df_pred.write_parquet(os.path.join(key_output_dir, "train_predictions.parquet"))

        # Create output directories
        logger.info("Saving model outputs.")
        additional_output_dir = os.path.join(self.run_folder_path, "additional_output")
        os.makedirs(additional_output_dir, exist_ok=True)

        # Save key outputs
        metrics_path = os.path.join(key_output_dir, "train_metrics.pkl")
        self.save_outputs(
            outputs=[{"metrics": pred_metrics}],
            output_paths=[metrics_path],
        )

        # Save additional outputs
        df_train.write_parquet(
            os.path.join(additional_output_dir, "train_dataframe.parquet")
        )
        self.save_outputs(
            outputs=[{"train_data": train_data}],
            output_paths=[os.path.join(additional_output_dir, "train_model_data.pkl")],
            save_config=False,
        )

        # Bayesian model-specific outputs
        if isinstance(model, BayesianHierarchicalModel):
            if model.prior_predictive is not None:
                self.save_outputs(
                    outputs=[{"prior_predictive": model.prior_predictive}],
                    output_paths=[
                        os.path.join(additional_output_dir, "prior_predictive.pkl")
                    ],
                    save_config=False,
                )
            if self.save_predictive_distributions:
                df_pred_distr.write_parquet(
                    os.path.join(
                        additional_output_dir,
                        "train_predictive_distribution.parquet",
                    )
                )
            if self.save_models_and_traces:
                fold_output = {
                    "trace": model.trace,
                    "model": model.model_instance,
                }
                self.save_outputs(
                    outputs=[fold_output],
                    output_paths=[
                        os.path.join(additional_output_dir, "training_model_trace.pkl")
                    ],
                    save_config=False,
                )

        if isinstance(model, GeneralizedLinearMixedModel):
            effect_summary = model.extract_effects()
            effects_output_path = os.path.join(key_output_dir, "train_effects.json")
            validate_output_files(
                file_paths=[effects_output_path],
                files=[effect_summary],
            )
            with open(effects_output_path, "w") as out_stream:
                json.dump(effect_summary, out_stream, indent=2)
            if model.family == "beta":
                beta_phi = {"phi": model.extract_beta_phi()}
                phi_output_path = os.path.join(key_output_dir, "train_phi.json")
                validate_output_files(
                    file_paths=[phi_output_path],
                    files=[beta_phi],
                )
                with open(phi_output_path, "w") as out_stream:
                    json.dump(beta_phi, out_stream, indent=2)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Model training completed in {runtime}.")


class CrossValidationTask(BaseModelTask):
    """Class for performing cross-validation on a given model."""

    def __init__(self, run_folder_path: str, mode: str) -> None:
        """
        Attributes:
            cv_folds: Number of cross-validation folds.
            train_data_paths: List of paths to training data for each fold.
            test_data_paths: List of paths to test data for each fold.
        """
        super().__init__(run_folder_path, mode)
        self.cv_folds: int = configs.cv_settings.k
        self.train_data_paths: list[str] = [
            os.path.join(self.run_folder_path, f"train_fold_{i + 1}.parquet")
            for i in range(self.cv_folds)
        ]
        self.test_data_paths: list[str] = [
            os.path.join(self.run_folder_path, f"test_fold_{i + 1}.parquet")
            for i in range(self.cv_folds)
        ]

    def run_task(self) -> None:
        """
        Perform the following processing steps:
            - For each cross-validation fold:
                - Load training and test data for the fold.
                - Initialize model and prepare data for specified model type.
                - Fit the model on training data and make predictions on both
                  training and test data.
                - Evaluate model performance on both training and test data.
            - After all folds are processed:
                - Concatenate all test predictions and evaluate overall
                  performance.
                - Save predictions, evaluation metrics and other outputs.
        """
        logger.info(
            f"Initiating cross-validation of '{self.model_type}' model "
            f"and diversity type '{self.diversity_type}'."
        )
        start = time.time()

        # Validate data paths and load shared data
        validate_input_files(file_paths=self.train_data_paths + self.test_data_paths)
        self.load_input_data()

        # Store per-fold metrics, but do not keep large objects (e.g. traces or
        # prediction dataframes) in memory across folds.
        per_fold_metrics = []
        key_output_dir = os.path.join(self.run_folder_path, "key_output")
        additional_output_dir = os.path.join(self.run_folder_path, "additional_output")
        os.makedirs(key_output_dir, exist_ok=True)
        os.makedirs(additional_output_dir, exist_ok=True)

        for fold_idx, (train_path, test_path) in enumerate(
            zip(self.train_data_paths, self.test_data_paths)
        ):
            logger.info(f"Processing fold {fold_idx + 1} out of {self.cv_folds}.")
            df_train = pl.read_parquet(train_path)
            df_test = pl.read_parquet(test_path)

            # Initialize model, prepare data, and train
            logger.info("Preparing model data and training the model.")
            model = self.initialize_model()
            fold_seed = self.random_seed + fold_idx
            if hasattr(model, "sampling_seed"):
                model.sampling_seed = fold_seed
            if hasattr(model, "random_seed"):
                model.random_seed = fold_seed
            train_data, test_data = model.prepare_data(df_train, df_test)
            model.fit(train_data)

            # Evaluate on train and test
            logger.info("Making predictions and evaluating model performance.")
            if self.model_type == "bayesian":
                df_pred_train, df_pred_train_distr = model.predict(
                    train_data, pred_mode="train"
                )
                df_pred_test, df_pred_test_distr = model.predict(
                    test_data, pred_mode="test"
                )
            else:
                df_pred_train = model.predict(train_data, pred_mode="train")
                df_pred_test = model.predict(test_data, pred_mode="test")

            logger.info("Evaluation results on training set:")
            pred_metrics_train = calculate_performance_metrics(
                df_pred_train,
                model_type=self.model_type,
                mode=self.mode,
            )
            logger.info("Evaluation results on test set")
            pred_metrics_test = calculate_performance_metrics(
                df_pred_test,
                model_type=self.model_type,
                mode=self.mode,
            )
            per_fold_metrics.append(
                {"train": pred_metrics_train, "test": pred_metrics_test}
            )

            fold = fold_idx + 1  # Increment fold index for file naming

            # Save per-fold prediction dataframes to parquet and free memory.
            train_pred_path = os.path.join(
                key_output_dir, f"train_predictions_fold_{fold}.parquet"
            )
            test_pred_path = os.path.join(
                key_output_dir, f"test_predictions_fold_{fold}.parquet"
            )
            df_pred_train.write_parquet(train_pred_path)
            df_pred_test.write_parquet(test_pred_path)

            # Save predictive distribution dataframes (optional; can be very large)
            if (
                isinstance(model, BayesianHierarchicalModel)
                and self.save_predictive_distributions
            ):
                train_distr_path = os.path.join(
                    additional_output_dir,
                    f"train_predictive_distribution_fold_{fold}.parquet",
                )
                test_distr_path = os.path.join(
                    additional_output_dir,
                    f"test_predictive_distribution_fold_{fold}.parquet",
                )
                df_pred_train_distr.write_parquet(train_distr_path)
                df_pred_test_distr.write_parquet(test_distr_path)

            # Save trace and model per fold (optional; can be very large)
            if (
                isinstance(model, BayesianHierarchicalModel)
                and self.save_models_and_traces
            ):
                fold_output = {
                    "trace": model.trace,
                    "model": model.model_instance,
                }
                fold_path = os.path.join(
                    additional_output_dir, f"model_trace_fold_{fold}.pkl"
                )
                self.save_outputs(
                    outputs=[fold_output],
                    output_paths=[fold_path],
                    save_config=False,
                )

            # Free memory between folds
            del df_train, df_test, train_data, test_data, df_pred_train, df_pred_test
            if isinstance(model, BayesianHierarchicalModel):
                del df_pred_train_distr, df_pred_test_distr
                if hasattr(model, "trace"):
                    del model.trace
                if hasattr(model, "prior_predictive"):
                    del model.prior_predictive
                if hasattr(model, "model_instance"):
                    del model.model_instance
            del model
            gc.collect()

        # Aggregate all test fold predictions for overall metrics.
        logger.info("Evaluation results for all test folds:\n")
        df_pred_test_all_folds = pl.concat(
            [
                pl.read_parquet(
                    os.path.join(
                        key_output_dir, f"test_predictions_fold_{i + 1}.parquet"
                    )
                )
                for i in range(self.cv_folds)
            ]
        )
        pred_metrics_test_all = calculate_performance_metrics(
            df_pred_test_all_folds,
            model_type=self.model_type,
            mode=self.mode,
        )
        all_test_pred_path = os.path.join(
            key_output_dir, "test_predictions_all_folds.parquet"
        )
        df_pred_test_all_folds.write_parquet(all_test_pred_path)

        # Save all relevant outputs
        logger.info("Saving cross-validation outputs.")

        # Save metrics in key output directory
        metrics_per_fold_path = os.path.join(
            key_output_dir, "crossval_metrics_per_fold.pkl"
        )
        metrics_overall_path = os.path.join(
            key_output_dir, "crossval_metrics_overall.pkl"
        )
        self.save_outputs(
            outputs=[{"metrics": per_fold_metrics}, {"metrics": pred_metrics_test_all}],
            output_paths=[metrics_per_fold_path, metrics_overall_path],
        )
        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Cross-validation completed in {runtime}.")
