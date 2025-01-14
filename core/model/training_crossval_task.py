import json
import os
import time
from datetime import timedelta
from typing import Any, Union

import dill
import polars as pl
import pymer4 as pymer
import yaml
from box import Box

from core.model.hierarchical_model import BayesianHierarchicalModel
from core.model.linear_mixed_model import LinearMixedModel
from core.model.model_utils import (
    approximate_change_predictions,
    augment_prediction_dataframe,
    calculate_performance_metrics,
)
from core.tests.shared.validate_shared import (
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "model_configs.yaml"))

logger = create_logger(__name__)


class BaseModelTask:
    def __init__(self, run_folder_path: str, mode: str) -> None:
        # Run arguments
        self.run_folder_path: str = run_folder_path
        self.mode: str = mode

        # Config settings used by all models and modes (but model-specific)
        self.diversity_type: str = configs.diversity_type
        self.model_type: str = configs.run_settings.model_type
        self.model_settings: dict[str, Any] = configs.run_settings[self.model_type]
        self.model_vars: dict[str, Any] = configs.model_variables[
            configs.run_settings.model_variables
        ]
        self.continuous_vars: list[str] = self.model_vars["continuous_vars"]
        self.response_var_transform: str = self.model_vars["response_var_transform"]

        # Shared data paths
        self.interaction_terms_path: str = os.path.join(
            run_folder_path, "interaction_terms.json"
        )
        self.site_mapping_path: str = os.path.join(run_folder_path, "site_mapping.json")
        self.site_info_path: str = os.path.join(run_folder_path, "site_info.parquet")
        if self.model_type == "bayesian":
            self.hierarchy_mapping_path: str = os.path.join(
                run_folder_path, "hierarchy_mapping.json"
            )

    def load_shared_data(self) -> None:
        """Validate required input files and load shared resources."""
        validate_input_files(file_paths=[self.interaction_terms_path])
        # Load key information about each site for analysis
        self.df_site_info = pl.read_parquet(self.site_info_path)

        # Load generated interaction terms and mapping of site names to indices
        with open(self.interaction_terms_path) as f:
            self.model_vars["interaction_terms"] = json.load(f)

        if self.model_type == "bayesian":
            validate_input_files(
                file_paths=[self.site_mapping_path, self.hierarchy_mapping_path]
            )
            with open(self.site_mapping_path) as f:
                self.site_name_to_idx = json.load(f)
            with open(self.hierarchy_mapping_path) as f:
                self.hierarchy_mapping = json.load(f)

    def initialize_model(self) -> Union[BayesianHierarchicalModel, LinearMixedModel]:
        """Instantiate the appropriate model class based on the model type."""
        model_classes = {
            "bayesian": BayesianHierarchicalModel,
            "lmm": LinearMixedModel,
        }
        # Shared model class attributes
        model_init_kwargs = {
            "model_settings": self.model_settings,
            "model_vars": self.model_vars,
            "logger": logger,
            "mode": self.mode,
        }
        # Add specific attributes for Bayesian model
        if self.model_type == "bayesian":
            model_init_kwargs["site_name_to_idx"] = self.site_name_to_idx
            model_init_kwargs["hierarchy_mapping"] = self.hierarchy_mapping

        return model_classes[self.model_type](**model_init_kwargs)

    def evaluate_model_performance(
        self, df_pred: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float], dict[str, float]]:
        # Clip and back-transform state predictions, get performance metrics
        df_pred = augment_prediction_dataframe(
            df_pred,
            model_type=self.model_type,
            response_transform=self.response_var_transform,
        )
        logger.info("Performance metrics for state predictions:")
        state_metrics = calculate_performance_metrics(
            df_pred,
            model_type=self.model_type,
            mode=self.mode,
            pred_type="state",
        )

        # Calculate approximate change predictions and performance metrics
        df_pred_change = approximate_change_predictions(
            df_pred,
            self.df_site_info,
            model_type=self.model_type,
            mode=self.mode,
        )
        logger.info("Performance metrics for change predictions:")
        change_metrics = calculate_performance_metrics(
            df_pred_change,
            model_type=self.model_type,
            mode=self.mode,
            pred_type="change",
        )

        return df_pred, df_pred_change, state_metrics, change_metrics

    def save_outputs(self, output_dict: dict[str, Any], output_path: str) -> None:
        """Save outputs and configuration."""
        # Validate and save output dictionary
        validate_output_files([output_path], [file for file in output_dict.values()])
        with open(output_path, "wb") as out_stream:
            dill.dump(output_dict, out_stream)

        # Save model configurations for reproducibility
        config_output_path = os.path.join(self.run_folder_path, "model_configs.yaml")
        validate_output_files([config_output_path], [configs])
        with open(config_output_path, "w") as outfile:
            yaml.dump(configs, outfile, default_flow_style=False)

    def save_pymer_model(self, model_instance: pymer.Lmer, path_suffix: str) -> None:
        """Save Pymer model separately since it cannot be pickled."""
        pymer_model_path = os.path.join(self.run_folder_path, path_suffix)
        validate_output_files([pymer_model_path], [model_instance])
        pymer.save_model(model=model_instance, filepath=pymer_model_path)


class ModelTrainingTask(BaseModelTask):
    def __init__(self, run_folder_path: str, mode: str) -> None:
        super().__init__(run_folder_path, mode)
        self.training_data_path: str = os.path.join(
            self.run_folder_path, "training_data.parquet"
        )

    def run_task(self) -> None:
        logger.info(
            f"Initiating training of '{self.model_type}' model "
            f"and diversity type '{self.diversity_type}'."
        )
        start = time.time()

        # Load dataframe, interaction terms and site mapping from run folder
        validate_input_files(file_paths=[self.training_data_path])
        df_train = pl.read_parquet(self.training_data_path)
        self.load_shared_data()

        # Initialize model, prepare data, and train
        logger.info("Preparing model data and training the model.")
        model = self.initialize_model()
        train_data = model.prepare_data(df_train)
        model.fit(train_data)

        # Make predictions on in-sample data evaluate performance
        logger.info("Making predictions and evaluating model performance.")
        df_pred = model.predict(train_data, pred_mode="train")
        df_pred, df_pred_change, state_metrics, change_metrics = (
            self.evaluate_model_performance(df_pred)
        )

        # Create a dictionary of outputs to be saved
        logger.info("Saving model outputs.")
        output_dict = {
            "data": train_data,
            "state_predictions": df_pred,
            "state_metrics": state_metrics,
            "change_predictions": df_pred_change,
            "change_metrics": change_metrics,
        }
        if self.model_type == "bayesian":
            output_dict["model"] = model.model_instance
            output_dict["prior_predictive"] = model.prior_predictive  # type: ignore
            output_dict["trace"] = model.trace  # type: ignore

        # Save all outputs
        output_path = os.path.join(self.run_folder_path, "training_output.pkl")
        validate_output_files(file_paths=[output_path], files=[output_dict])
        self.save_outputs(output_dict, output_path)

        if self.model_type == "lmm":
            self.save_pymer_model(model.model_instance, "pymer_model.joblib")

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Model training completed in {runtime}.")


class CrossValidationTask(BaseModelTask):
    def __init__(self, run_folder_path: str, mode: str) -> None:
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
        logger.info(
            f"Initiating cross-validation of '{self.model_type}' model "
            f"and diversity type '{self.diversity_type}'."
        )
        start = time.time()

        # Validate data paths and load shared data
        validate_input_files(file_paths=self.train_data_paths + self.test_data_paths)
        self.load_shared_data()

        # Lists for storing outputs of train-test on each fold
        all_model_data = []
        all_state_predictions = []
        all_change_predictions = []
        all_state_metrics = []
        all_change_metrics = []
        all_model_instances = []

        for fold_idx, (train_path, test_path) in enumerate(
            zip(self.train_data_paths, self.test_data_paths)
        ):
            logger.info(f"Processing fold {fold_idx + 1} out of {self.cv_folds}.")
            df_train = pl.read_parquet(train_path)
            df_test = pl.read_parquet(test_path)

            # Initialize model, prepare data, and train
            logger.info("Preparing model data and training the model.")
            model = self.initialize_model()
            train_data = model.prepare_data(df_train)
            test_data = model.prepare_data(df_test)
            model.fit(train_data)

            # Evaluate on train and test
            logger.info("Making predictions and evaluating model performance.")
            df_pred_train = model.predict(train_data, pred_mode="train")
            df_pred_test = model.predict(test_data, pred_mode="test")

            logger.info("Evaluation results on training set:")
            (
                df_pred_train,
                df_pred_change_train,
                state_metrics_train,
                change_metrics_train,
            ) = self.evaluate_model_performance(df_pred_train)

            logger.info("Evaluation results on test set")
            (
                df_pred_test,
                df_pred_change_test,
                state_metrics_test,
                change_metrics_test,
            ) = self.evaluate_model_performance(df_pred_test)

            # Collect outputs
            all_model_data.append({"train": train_data, "test": test_data})
            all_state_predictions.append({"train": df_pred_train, "test": df_pred_test})
            all_change_predictions.append(
                {"train": df_pred_change_train, "test": df_pred_change_test}
            )
            all_state_metrics.append(
                {"train": state_metrics_train, "test": state_metrics_test}
            )
            all_change_metrics.append(
                {"train": change_metrics_train, "test": change_metrics_test}
            )
            all_model_instances.append(model)

            # Save Pymer model for the fold if LMM
            if self.model_type == "lmm":
                self.save_pymer_model(
                    model.model_instance, f"pymer_model_fold_{fold_idx + 1}.joblib"
                )

        # Concatenate all test data and calculate metrics
        logger.info("Evaluation results for all test folds:\n")
        df_pred_test_all = pl.concat([fold["test"] for fold in all_state_predictions])
        (
            df_pred_test_all,
            df_pred_change_test_all,
            state_metrics_test_all,
            change_metrics_test_all,
        ) = self.evaluate_model_performance(df_pred_test_all)

        # Save all relevant outputs
        logger.info("Saving cross-validation outputs.")
        output_dict = {
            "data": all_model_data,
            "state_predictions": all_state_predictions,
            "state_metrics": all_state_metrics,
            "change_predictions": all_change_predictions,
            "change_metrics": all_change_metrics,
            "all_test_results": {
                "state_predictions": df_pred_test_all,
                "state_metrics": state_metrics_test_all,
                "change_predictions": df_pred_change_test_all,
                "change_metrics": change_metrics_test_all,
            },
        }
        if self.model_type == "bayesian":
            output_dict["models"] = all_model_instances  # type: ignore
            output_dict["traces"] = [
                model.trace for model in all_model_instances  # type: ignore
            ]

        output_path = os.path.join(self.run_folder_path, "cv_output.pkl")
        validate_output_files(file_paths=[output_path], files=[output_dict])
        self.save_outputs(output_dict, output_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Cross-validation completed in {runtime}.")
