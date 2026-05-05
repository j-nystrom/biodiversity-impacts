import logging
import sys
import time
from datetime import timedelta
from typing import Any

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
from numpy.typing import NDArray

from core.model.model_utils import (
    standardize_continuous_covariates,
    validate_design_matrix_columns,
)
from core.model.pymc_models import (
    GeneralHierarchicalModel,
    get_ecological_effects_settings,
)
from core.utils.bayesian_utils import plot_prior_distribution


class BayesianHierarchicalModel:
    """
    Class for training and making predictions with Bayesian (hierarchical)
    models implemented in PyMC.
    """

    def __init__(
        self,
        mode: str,
        random_seed: int,
        epsilon: float,
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        logger: logging.Logger,
        site_name_to_idx: dict[str, int],
        taxon_name_to_idx: dict[str, int],
        hierarchy_mapping: dict[str, list[str]],
        rolled_up_mapping: dict[str, Any] | None = None,  # Only if rolled up groups
        save_predictive_distributions: bool = False,
    ) -> None:
        """Initialize Bayesian model settings, mappings, and covariate lists."""
        # Model settings
        self.mode = mode
        self.random_seed = random_seed
        self.sampling_seed = random_seed
        self.epsilon = epsilon
        self.model_settings = model_settings
        self.model_vars = model_vars
        self.logger = logger
        self.prior_predictive: az.InferenceData | None = None
        self.progressbar: bool = sys.stderr.isatty()

        # Model covariates
        self.response_var = model_vars["response_var"]
        self.categorical_vars = model_vars["categorical_vars"]
        self.continuous_vars = model_vars["continuous_vars"]
        self.interaction_terms = model_vars["interaction_terms"]

        # Additional settings for Bayesian model
        self.site_name_to_idx: dict[str, int] = site_name_to_idx
        self.taxon_name_to_idx: dict[str, int] = taxon_name_to_idx
        self.hierarchy_mapping: dict[str, Any] = hierarchy_mapping
        self.sampler_settings: dict[str, Any] = self.model_settings["sampler"]
        self.save_predictive_distributions: bool = self.model_settings[
            "save_predictive_distributions"
        ]
        self.save_predictive_distributions = save_predictive_distributions
        if rolled_up_mapping:
            self.rolled_up_mapping: dict[str, Any] = rolled_up_mapping

    def prepare_data(
        self, df_train: pl.DataFrame, df_test: pl.DataFrame | None = None
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Standardize covariates and format data for PyMC model."""
        self.logger.info("Preparing data for PyMC model.")
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        df_test_for_validation = df_train if df_test is None else df_test
        validate_design_matrix_columns(df_train, df_test_for_validation, x_vars)
        # Continuous vars are standardized to have mean zero and unit variance
        df_train_std, df_test_std = standardize_continuous_covariates(
            df_train,
            df_test_for_validation,
            vars_to_standardize=self.continuous_vars + self.interaction_terms,
        )
        df_train_std, _, levels = self.ensure_hierarchy_label_columns(df_train_std)
        self.log_model_structure(levels=levels)
        self.active_hierarchy_mapping = self.build_fold_hierarchy_mapping(df_train_std)

        # Format data for PyMC model
        train_data = self.format_data_for_pymc_model(df_train_std, role="train")
        test_data = None
        if df_test is not None:
            df_test_std, _, _ = self.ensure_hierarchy_label_columns(df_test_std)
            test_data = self.format_data_for_pymc_model(df_test_std, role="test")

        return train_data, test_data

    def log_model_structure(self, levels: list[str]) -> None:
        """Log the active ecological hierarchy and crossed control structure."""
        ecological_effects = self.get_ecological_effects()
        display_names = self.get_hierarchy_display_names(levels)
        if levels:
            level_text = "; ".join(
                f"{level}={display_names[level]}" for level in levels
            )
        else:
            level_text = "population-level ecological intercept/slopes only"

        self.logger.info(
            "Ecological effects: "
            + level_text
            + f"; fitted_levels={ecological_effects['hierarchical_levels']}; "
            + f"varying_slope_level={ecological_effects['varying_slope_level']}; "
            + "train_on_rolled_up_groups="
            + f"{ecological_effects['train_on_rolled_up_groups']}; "
            + f"min_studies_per_group={ecological_effects['min_studies_per_group']}."
        )

        study_effects = self.model_settings.get("study_effects", {})
        training_components = self.get_training_components()
        configured_study_slope_terms = study_effects.get("slope_terms", [])
        active_study_slope_terms = (
            configured_study_slope_terms if training_components["study_slopes"] else []
        )
        self.logger.info(
            "Training components fitted: "
            + f"ecological={training_components['ecological']}; "
            + f"study_intercept={training_components['study_intercept']}; "
            + f"study_slopes={training_components['study_slopes']}; "
            + f"active_study_slope_terms={len(active_study_slope_terms)}"
            + (
                f" [{', '.join(active_study_slope_terms)}]"
                if active_study_slope_terms
                else ""
            )
            + f"; block_intercept={training_components['block_intercept']}."
        )

        prediction_components = self.get_prediction_components()
        self.logger.info(
            "Prediction components applied to prediction outputs: "
            + f"ecological={prediction_components.get('ecological', True)}; "
            + f"study_intercept={prediction_components.get('study_intercept', False)}; "
            + f"study_slopes={prediction_components.get('study_slopes', False)}; "
            + f"block_intercept={prediction_components.get('block_intercept', False)}."
        )

    def get_training_components(self) -> dict[str, bool]:
        """Return training-time component switches with backward-compatible defaults."""
        study_effects = self.model_settings.get("study_effects", {})
        block_effects = self.model_settings.get("block_effects", {})
        components = self.model_settings.get("training_components", {})
        return {
            "ecological": components.get("ecological", True),
            "study_intercept": components.get(
                "study_intercept", study_effects.get("intercept", True)
            ),
            "study_slopes": components.get(
                "study_slopes", bool(study_effects.get("slope_terms", []))
            ),
            "block_intercept": components.get(
                "block_intercept", block_effects.get("intercept", True)
            ),
        }

    def get_prediction_components(self) -> dict[str, bool]:
        """Return prediction-time component switches with legacy config support."""
        components = self.model_settings.get("prediction_components", {})
        if "test" in components:
            components = components["test"]
        return {
            "ecological": components.get("ecological", True),
            "study_intercept": components.get("study_intercept", False),
            "study_slopes": components.get("study_slopes", False),
            "block_intercept": components.get("block_intercept", False),
        }

    def get_ecological_effects(self) -> dict[str, Any]:
        """Return ecological hierarchy settings with legacy config support."""
        return get_ecological_effects_settings(self.model_settings)

    def get_hierarchy_display_names(self, levels: list[str]) -> dict[str, str]:
        """Return human-readable cumulative hierarchy names for logging."""
        hierarchy = self.get_ecological_effects()["hierarchy"]
        cumulative_cols: list[str] = []
        display_names: dict[str, str] = {}
        for level in levels:
            cumulative_cols.extend(hierarchy[level])
            display_names[level] = " + ".join(cumulative_cols)
        return display_names

    def get_hierarchy_levels(self) -> list[str]:
        """Return configured hierarchy levels used by the model."""
        ecological_effects = self.get_ecological_effects()
        hierarchy = ecological_effects["hierarchy"]
        n_levels = ecological_effects["hierarchical_levels"]
        return [
            f"level_{idx}"
            for idx in range(1, n_levels + 1)
            if hierarchy.get(f"level_{idx}")
        ]

    def ensure_hierarchy_label_columns(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, dict[str, str], list[str]]:
        """
        Ensure cumulative hierarchy label columns exist.

        The configured hierarchy defines incremental columns. For example,
        level_1=[Biome, Custom_taxonomic_group], level_2=[Realm] becomes
        level_1 label Biome_Custom_taxonomic_group and level_2 label
        Biome_Custom_taxonomic_group_Realm.
        """
        hierarchy = self.get_ecological_effects()["hierarchy"]
        levels = self.get_hierarchy_levels()
        label_cols: dict[str, str] = {}
        cumulative_cols: list[str] = []

        for level in levels:
            cumulative_cols.extend(hierarchy[level])
            missing = [col for col in cumulative_cols if col not in df.columns]
            if missing:
                raise ValueError(
                    f"Missing hierarchy source columns for {level}: {missing}"
                )
            label_col = "_".join(cumulative_cols)
            label_cols[level] = label_col
            if label_col not in df.columns:
                df = df.with_columns(
                    pl.concat_str(
                        [pl.col(col).cast(pl.Utf8) for col in cumulative_cols],
                        separator="_",
                    ).alias(label_col)
                )

        return df, label_cols, levels

    def build_fold_hierarchy_mapping(self, df_train: pl.DataFrame) -> dict[str, Any]:
        """
        Build a retained hierarchy from training data only.

        Groups below the study threshold are excluded as final prediction nodes.
        Their observations fall back to the deepest retained ancestor, or to the
        population-level parameters if no ancestor is retained.
        """
        _, label_cols, levels = self.ensure_hierarchy_label_columns(df_train)
        ecological_effects = self.get_ecological_effects()
        min_studies = int(ecological_effects["min_studies_per_group"])
        train_on_rolled = bool(ecological_effects["train_on_rolled_up_groups"])
        if not train_on_rolled:
            min_studies = 1

        if not levels:
            population_mapping = {
                "column_names": {},
                "levels": [],
                "level_display_names": {},
                "min_studies_per_group": min_studies,
                "train_on_rolled_up_groups": train_on_rolled,
                "_all_observed": {},
            }
            self.log_hierarchy_mapping_summary(population_mapping)
            return population_mapping

        counts_by_level: dict[str, dict[str, int]] = {}
        retained: dict[str, set[str]] = {}
        all_observed: dict[str, set[str]] = {}
        parent_pairs: dict[str, dict[str, str]] = {}

        for level in levels:
            label_col = label_cols[level]
            counts = (
                df_train.select([label_col, "SS"])
                .unique()
                .group_by(label_col)
                .agg(pl.col("SS").n_unique().alias("n_studies"))
            )
            count_dict = dict(
                zip(counts.get_column(label_col), counts.get_column("n_studies"))
            )
            counts_by_level[level] = count_dict
            all_observed[level] = set(count_dict)
            retained[level] = {
                label
                for label, n_studies in count_dict.items()
                if n_studies >= min_studies
            }

        for child_idx in range(1, len(levels)):
            child_level = levels[child_idx]
            parent_level = levels[child_idx - 1]
            pairs = df_train.select(
                [label_cols[child_level], label_cols[parent_level]]
            ).unique()
            parent_pairs[child_level] = dict(
                zip(
                    pairs.get_column(label_cols[child_level]),
                    pairs.get_column(label_cols[parent_level]),
                )
            )

        # Retain ancestors needed by retained child groups.
        for child_level in reversed(levels[1:]):
            parent_level = levels[levels.index(child_level) - 1]
            for child_label in list(retained[child_level]):
                parent_label = parent_pairs[child_level][child_label]
                retained[parent_level].add(parent_label)

        mapping: dict[str, Any] = {
            "column_names": label_cols,
            "levels": levels,
            "level_display_names": self.get_hierarchy_display_names(levels),
            "min_studies_per_group": min_studies,
            "train_on_rolled_up_groups": train_on_rolled,
            "_all_observed": {
                level: sorted(labels) for level, labels in all_observed.items()
            },
        }
        for level in levels:
            labels = sorted(retained[level])
            mapping[level] = {label: idx for idx, label in enumerate(labels)}
            mapping[f"{level}_n_studies"] = {
                label: counts_by_level[level][label] for label in labels
            }

        for child_level, parents in parent_pairs.items():
            parent_level = levels[levels.index(child_level) - 1]
            retained_children: dict[str, int] = mapping[child_level]
            retained_parents: dict[str, int] = mapping[parent_level]
            mapping[f"{child_level}_all_parents"] = parents
            mapping[f"{child_level}_parents"] = {
                child: parent
                for child, parent in parents.items()
                if child in retained_children and parent in retained_parents
            }

        self.log_hierarchy_mapping_summary(mapping)
        return mapping

    def log_hierarchy_mapping_summary(self, mapping: dict[str, Any]) -> None:
        """Log retained hierarchy sizes for quick runtime checks."""
        if not mapping["levels"]:
            self.logger.info(
                "Fold ecological mapping built from training data "
                + f"(train_on_rolled_up_groups={mapping['train_on_rolled_up_groups']}, "
                + f"min_studies_per_group={mapping['min_studies_per_group']}): "
                + "population-level ecological parameters only; no group-level "
                + "ecological hierarchy fitted."
            )
            return

        parts = []
        for level in mapping["levels"]:
            level_name = mapping.get("level_display_names", {}).get(level, level)
            retained = len(mapping.get(level, {}))
            observed = len(mapping.get("_all_observed", {}).get(level, []))
            below_threshold = observed - retained
            parts.append(
                f"{level_name}: {retained}/{observed} unique groups retained; "
                f"{below_threshold} without own parameters"
            )
        self.logger.info(
            "Fold hierarchy mapping built from training data "
            + f"(train_on_rolled_up_groups={mapping['train_on_rolled_up_groups']}, "
            + f"min_studies_per_group={mapping['min_studies_per_group']}): "
            + "; ".join(parts)
        )
        self.log_rollup_target_summary(mapping)

    def find_rollup_target_level(
        self,
        mapping: dict[str, Any],
        source_level: str,
        source_label: str,
    ) -> str:
        """Find the retained ancestor level used by a non-retained group."""
        levels = mapping["levels"]
        current_level = source_level
        current_label = source_label
        while current_level != levels[0]:
            parent_level = levels[levels.index(current_level) - 1]
            parent_label = mapping[f"{current_level}_all_parents"][current_label]
            if parent_label in mapping.get(parent_level, {}):
                return parent_level
            current_level = parent_level
            current_label = parent_label
        return "Population"

    def log_rollup_target_summary(self, mapping: dict[str, Any]) -> None:
        """Log where unique groups without own parameters roll up."""
        display_names = mapping.get("level_display_names", {})
        parts = []
        for level in mapping["levels"]:
            observed = set(mapping.get("_all_observed", {}).get(level, []))
            retained = set(mapping.get(level, {}))
            non_retained = observed - retained
            if not non_retained:
                continue

            target_counts: dict[str, int] = {}
            for label in non_retained:
                if level == mapping["levels"][0]:
                    target_level = "Population"
                else:
                    target_level = self.find_rollup_target_level(mapping, level, label)
                target_name = display_names.get(target_level, target_level)
                target_counts[target_name] = target_counts.get(target_name, 0) + 1

            target_summary = ", ".join(
                f"{target}: {count}" for target, count in sorted(target_counts.items())
            )
            parts.append(
                f"{display_names.get(level, level)} groups without own parameters -> "
                + target_summary
            )

        if parts:
            self.logger.info(
                "Roll-up targets for unique training groups: " + "; ".join(parts)
            )

    def log_pymc_data_structure(
        self,
        role: str,
        n_obs: int,
        n_studies: int,
        n_blocks: int,
        study_slope_terms: list[str],
        hierarchy: dict[str, Any],
    ) -> None:
        """Log data cardinalities used by the PyMC model."""
        display_names = hierarchy.get("level_display_names", {})
        if hierarchy["levels"]:
            retained_groups = [
                f"{display_names.get(level, level)}={len(hierarchy.get(level, {}))}"
                for level in hierarchy["levels"]
            ]
        else:
            retained_groups = ["Population=1"]
        training_components = self.get_training_components()
        self.logger.info(
            f"PyMC {role} data structure: observations={n_obs}; "
            f"studies={n_studies}; SSB_blocks_indexed={n_blocks}; "
            f"retained_ecological_group_parameters={{{', '.join(retained_groups)}}}; "
            f"training_controls={{study_intercept="
            f"{training_components['study_intercept']}, "
            f"study_slope_terms={len(study_slope_terms)}, "
            f"block_intercept={training_components['block_intercept']}}}."
        )

    def resolve_hierarchy_for_dataframe(
        self,
        df: pl.DataFrame,
        mapping: dict[str, Any],
        role: str,
    ) -> dict[str, Any]:
        """Resolve every row to retained hierarchy parameters or population."""
        levels = mapping["levels"]
        label_cols = mapping["column_names"]
        if not levels:
            diagnostics = {
                "level_assignment": np.zeros(df.height, dtype=np.int32),
                "resolved_level_name": np.array(
                    ["Population"] * df.height, dtype=object
                ),
                "resolved_group": np.array(["Population"] * df.height, dtype=object),
                "fallback_reason": np.array(
                    ["population_only"] * df.height, dtype=object
                ),
            }
            self.validate_hierarchy_resolution(diagnostics, mapping, role)
            return diagnostics

        deepest_level = levels[-1]
        level_numbers = {level: int(level.split("_")[1]) for level in levels}
        all_observed = {
            level: set(labels)
            for level, labels in mapping.get("_all_observed", {}).items()
        }

        level_idx_arrays: dict[str, list[int]] = {level: [] for level in levels}
        assignments: list[int] = []
        resolved_levels: list[str] = []
        resolved_groups: list[str] = []
        fallback_reasons: list[str] = []

        for row in df.iter_rows(named=True):
            resolved_level: str | None = None
            resolved_group: str | None = None

            for level in reversed(levels):
                label = row[label_cols[level]]
                if label in mapping.get(level, {}):
                    resolved_level = level
                    resolved_group = label
                    break

            for level in levels:
                label = row[label_cols[level]]
                level_idx_arrays[level].append(mapping.get(level, {}).get(label, 0))

            if resolved_level is None or resolved_group is None:
                assignments.append(0)
                resolved_levels.append("Population")
                resolved_groups.append("Population")
                if row[label_cols["level_1"]] in all_observed.get("level_1", set()):
                    fallback_reasons.append("below_threshold_to_population")
                else:
                    fallback_reasons.append("unseen_to_population")
            else:
                assignments.append(level_numbers[resolved_level])
                resolved_levels.append(resolved_level)
                resolved_groups.append(resolved_group)
                if resolved_level == deepest_level:
                    fallback_reasons.append("retained_deepest")
                elif row[label_cols[deepest_level]] in all_observed.get(
                    deepest_level, set()
                ):
                    fallback_reasons.append("below_threshold_rollup")
                else:
                    fallback_reasons.append("unseen_rollup")

        diagnostics = {
            "level_assignment": np.array(assignments, dtype=np.int32),
            "resolved_level_name": np.array(resolved_levels, dtype=object),
            "resolved_group": np.array(resolved_groups, dtype=object),
            "fallback_reason": np.array(fallback_reasons, dtype=object),
        }
        for level, idx_values in level_idx_arrays.items():
            diagnostics[f"{level}_idx"] = np.array(idx_values, dtype=np.int32)

        self.validate_hierarchy_resolution(diagnostics, mapping, role)
        return diagnostics

    def summarize_hierarchy_resolution(
        self,
        resolution: dict[str, Any],
        mapping: dict[str, Any],
    ) -> tuple[dict[str, int], dict[str, int]]:
        """Summarize row-level hierarchy resolution for logging."""
        display_names = mapping.get("level_display_names", {})
        deepest_level = mapping["levels"][-1] if mapping["levels"] else "Population"

        unique_levels, counts = np.unique(
            resolution["resolved_level_name"], return_counts=True
        )
        level_summary = {
            display_names.get(level, level): count
            for level, count in zip(unique_levels.tolist(), counts.tolist())
        }

        fallback_labels = {
            "retained_deepest": (
                f"used retained {display_names.get(deepest_level, deepest_level)}"
            ),
            "below_threshold_rollup": (
                "below-threshold group rolled up to retained parent"
            ),
            "below_threshold_to_population": (
                "below-threshold group rolled up to Population"
            ),
            "unseen_rollup": "unseen group rolled up to retained parent",
            "unseen_to_population": "unseen group rolled up to Population",
            "population_only": "population-level ecological parameters only",
        }
        unique_reasons, reason_counts = np.unique(
            resolution["fallback_reason"], return_counts=True
        )
        reason_summary = {
            fallback_labels.get(reason, reason): count
            for reason, count in zip(unique_reasons.tolist(), reason_counts.tolist())
        }

        return level_summary, reason_summary

    def validate_hierarchy_resolution(
        self,
        resolution: dict[str, Any],
        mapping: dict[str, Any],
        role: str,
    ) -> None:
        """Validate and log hierarchy resolution for train/test data."""
        assignments = resolution["level_assignment"]
        hierarchical_levels = self.get_ecological_effects()["hierarchical_levels"]
        if ((assignments < 0) | (assignments > hierarchical_levels)).any():
            raise ValueError("Invalid hierarchy level assignment encountered.")

        level_summary, reason_summary = self.summarize_hierarchy_resolution(
            resolution, mapping
        )
        self.logger.info(
            f"Hierarchy resolution for {role} observations: "
            f"resolved_to={level_summary}; fallback_reason={reason_summary}."
        )

        for level in mapping["levels"]:
            assigned = assignments == int(level.split("_")[1])
            if assigned.any() and len(mapping.get(level, {})) == 0:
                raise ValueError(f"Rows assigned to {level}, but no parameters exist.")

    def fit(self, train_data: dict[str, Any], run_sampler: bool = True) -> None:
        """
        Instantiate the PyMC model object, do prior predictive sampling, and
        run the NUTS sampler to fit the model. Also calculate sampling
        statistics to evaluate convergence.

        Args:
            - train_data: Dataframe with the scaled covariates and response
                variable. This is the training data for the model.
        """
        # Initialize the PyMC model and return the training model object
        self.model = GeneralHierarchicalModel(
            settings=self.model_settings, epsilon=self.epsilon
        )
        self.model_instance = self.model.build_training_model(model_data=train_data)

        if self.model_settings["prior_predictive_checks"]:
            # Do prior predictive sampling before running the model
            self.logger.info("Running prior predictive sampling.")
            self.prior_predictive = pm.sample_prior_predictive(
                draws=1000,
                model=self.model_instance,
                progressbar=self.progressbar,
                random_seed=self.sampling_seed,
            )
            plot_prior_distribution(
                self.prior_predictive,
                category_variable_pairs=[
                    tuple(pair)
                    for pair in self.model_settings["prior_predictive_plot_pairs"]
                ],
            )
            user_input = input("Continue sampling process? (y/n): ")
            if user_input.lower() == "n":
                run_sampler = False
                self.logger.info("Training aborted based on prior predictive checks.")
        else:
            self.logger.info(
                "Skipping prior predictive sampling (prior_predictive_checks=False)."
            )

        # Run NUTS sampler on the model and summarize sampling statistics
        if run_sampler:
            self.trace = self.run_sampling()
            if self.model_settings["sampling_statistics"]:
                self.summarize_sampling_statistics()

    def predict(
        self, prediction_data: dict[str, Any], pred_mode: str
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Make predictions on training or test data using the configured
        prediction components.

        Args:
            - prediction_data: Either training or test data, depending on the
                mode of the model.
            - pred_mode: Either 'train' or 'test'.

        Returns:
            - df_pred: Dataframe with site names, observed values, and
                predictions.
        """
        self.prediction_trace = self.make_predictions(prediction_data, mode=pred_mode)

        df_pred, df_pred_distr = self.create_prediction_dataframe(
            prediction_data,
            mode=pred_mode,
            include_predictive_distribution=self.save_predictive_distributions,
        )

        return df_pred, df_pred_distr

    def format_data_for_pymc_model(self, df: pl.DataFrame, role: str) -> dict[str, Any]:
        """
        Format the dataframe for use in PyMC models.

        Args:
            - df: Dataframe with the scaled covariates and response variable.

        Returns:
            - output_dict: Dictionary containing the formatted data for the
                PyMC model.
        """
        self.logger.info("Formatting data for PyMC model.")

        # ----- Hierarchical levels and indices -----
        hierarchy = self.active_hierarchy_mapping
        hierarchy_resolution = self.resolve_hierarchy_for_dataframe(
            df,
            hierarchy,
            role=role,
        )
        level_indices = {
            f"{level}_idx": hierarchy_resolution[f"{level}_idx"]
            for level in hierarchy["levels"]
        }
        for level_key in ["level_1", "level_2", "level_3"]:
            level_indices.setdefault(
                f"{level_key}_idx", np.zeros(df.height, dtype=np.int32)
            )

        level_values = {}
        level_n_studies = {}
        for level_key in ["level_1", "level_2", "level_3"]:
            level_dict = hierarchy.get(level_key, {})
            group_names = [
                label
                for label, _ in sorted(level_dict.items(), key=lambda item: item[1])
            ]
            level_values[f"{level_key}_values"] = group_names
            study_count_dict = hierarchy.get(f"{level_key}_n_studies", {})
            level_n_studies[f"{level_key}_n_studies"] = np.array(
                [study_count_dict.get(label, 1) for label in group_names],
                dtype=np.int32,
            )

        level_2_to_level_1_idx: NDArray[np.int32] = np.zeros(
            len(hierarchy.get("level_2", {})), dtype=np.int32
        )
        if "level_2" in hierarchy and hierarchy.get("level_2"):
            for label, idx in hierarchy["level_2"].items():
                parent = hierarchy["level_2_parents"][label]
                level_2_to_level_1_idx[idx] = hierarchy["level_1"][parent]

        level_3_to_level_2_idx: NDArray[np.int32] = np.zeros(
            len(hierarchy.get("level_3", {})), dtype=np.int32
        )
        if "level_3" in hierarchy and hierarchy.get("level_3"):
            for label, idx in hierarchy["level_3"].items():
                parent = hierarchy["level_3_parents"][label]
                level_3_to_level_2_idx[idx] = hierarchy["level_2"][parent]

        # ----- Control variables during sampling -----
        # Study and block random effects
        study_names = sorted(df.get_column("SS").unique().to_list())
        study_name_to_idx = {study: idx for idx, study in enumerate(study_names)}
        study_idx = np.array(
            [study_name_to_idx[study] for study in df.get_column("SS").to_list()],
            dtype=np.int32,
        )
        block_names = sorted(df.get_column("SSB").unique().to_list())
        block_name_to_idx = {block: idx for idx, block in enumerate(block_names)}
        block_idx = np.array(
            [block_name_to_idx[block] for block in df.get_column("SSB").to_list()],
            dtype=np.int32,
        )
        block_pairs = df.select(["SSB", "SS"]).unique().sort("SSB")
        block_to_study_idx = np.array(
            [
                study_name_to_idx[study]
                for study in block_pairs.get_column("SS").to_list()
            ],
            dtype=np.int32,
        )

        # Create response variable vector
        y_obs = df.get_column(self.response_var).to_numpy()

        # Create design matrix
        x_vars = self.categorical_vars + self.continuous_vars + self.interaction_terms
        x_obs = df.select(x_vars).to_numpy()
        configured_study_slope_terms = self.model_settings.get("study_effects", {}).get(
            "slope_terms", []
        )
        training_components = self.get_training_components()
        study_slope_terms = (
            configured_study_slope_terms if training_components["study_slopes"] else []
        )
        missing_slope_terms = [term for term in study_slope_terms if term not in x_vars]
        if missing_slope_terms:
            raise ValueError(
                f"Study slope terms are not model covariates: {missing_slope_terms}"
            )
        x_study_slope_obs = (
            df.select(study_slope_terms).to_numpy()
            if study_slope_terms
            else np.zeros((df.height, 0), dtype=float)
        )
        self.log_pymc_data_structure(
            role=role,
            n_obs=df.height,
            n_studies=len(study_names),
            n_blocks=len(block_names),
            study_slope_terms=study_slope_terms,
            hierarchy=hierarchy,
        )

        # Add site indices for reference
        site_idx = np.array(
            [self.site_name_to_idx[site] for site in df.get_column("SSBS").to_list()]
        )
        # Add taxon indices for reference if applicable
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            taxon_idx = np.array(
                [
                    self.taxon_name_to_idx[taxon]
                    for taxon in df.get_column("Custom_taxonomic_group").to_list()
                ]
            )
        else:
            taxon_idx = np.zeros(df.height, dtype=np.int32)

        # Build output dictionary
        coords: dict[str, Any] = {"idx": np.arange(df.shape[0])}
        coords.update(level_values)
        coords["study_names"] = study_names
        coords["block_names"] = block_names
        coords["x_vars"] = x_vars
        coords["study_slope_vars"] = study_slope_terms

        # Specify coordinates for calibration terms
        coords["x_cal_vars"] = ["y_hat_sqrt", "y_hat", "y_hat_squared"]

        output_dict: dict[str, Any] = {
            "coords": coords,
            "y_obs": y_obs,
            "x_obs": x_obs,
            "x_study_slope_obs": x_study_slope_obs,
            "site_idx": site_idx,
            "taxon_idx": taxon_idx,
            "study_idx": study_idx,
            "block_idx": block_idx,
            "block_to_study_idx": block_to_study_idx,
            "level_2_to_level_1_idx": level_2_to_level_1_idx,
            "level_3_to_level_2_idx": level_3_to_level_2_idx,
            "level_assignment": hierarchy_resolution["level_assignment"],
            "resolved_level_name": hierarchy_resolution["resolved_level_name"],
            "resolved_group": hierarchy_resolution["resolved_group"],
            "fallback_reason": hierarchy_resolution["fallback_reason"],
        }
        output_dict.update(level_indices)
        output_dict.update(level_n_studies)

        self.logger.info("Data formatted for PyMC model.")

        return output_dict

    def run_sampling(self) -> az.InferenceData:
        """
        Run sampling for the current model. The function uses the No U-turn
        (NUTS) sampler implemented in PyMC, and the 'sampler_settings'
        dictionary is specific to this sampler.

        Returns:
            trace: PyMC trace with posterior distribution info appended.
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
                progressbar=self.progressbar,
                random_seed=self.sampling_seed,
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
        divergences: int = int(np.sum(self.trace.sample_stats["diverging"].values))
        self.logger.warning(
            f"There are {divergences} divergences in the sampling chains."
        )

        # Acceptance rate
        accept_rate: float = float(
            np.mean(self.trace.sample_stats["acceptance_rate"].values)
        )
        self.logger.warning(f"The mean acceptance rate was {accept_rate:.3f}")

        # R-hat statistics
        for var in var_names:
            try:
                r_hat = az.summary(idata, var_names=var, round_to=2)["r_hat"]
                mean_r_hat: float = float(np.mean(r_hat))
                min_r_hat: float = float(np.min(r_hat))
                max_r_hat: float = float(np.max(r_hat))
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
                mean_ess: float = float(np.mean(ess))
                min_ess: float = float(np.min(ess))
                max_ess: float = float(np.max(ess))
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

        Args:
            - prediction_data: PyMC model dictionary for the prediction data.
            - mode: Either 'train' or 'test'. Note that this is different from
                the 'mode' attribute, which is related to the calling task.

        Returns:
            trace: InferenceData with predictions for the requested data.
        """
        if mode not in {"train", "test"}:
            raise ValueError(f"Unsupported prediction mode: {mode}")

        self.validate_prediction_setup(prediction_data, mode)
        self.validate_prediction_trace_parameters()
        self.pred_model = self.model.build_prediction_model(
            model_data=prediction_data,
        )
        with self.pred_model:
            prediction_trace = pm.sample_posterior_predictive(
                self.trace,
                var_names=["y_pred", "y_cond", "y_intercept"],
                predictions=True,
                extend_inferencedata=False,
                progressbar=self.progressbar,
                random_seed=self.sampling_seed + 1,
            )

        return prediction_trace

    def validate_prediction_trace_parameters(self) -> None:
        """Validate posterior parameters that are required by prediction models."""
        posterior_vars = set(self.trace.posterior.data_vars)
        likelihood = self.model_settings["likelihood"]

        if likelihood == "beta":
            if "sigma_raw" not in posterior_vars:
                raise ValueError(
                    "Posterior trace is missing sigma_raw, which is required for "
                    "Beta posterior predictive sampling."
                )

            sigma_raw = np.asarray(self.trace.posterior["sigma_raw"].values)
            if not np.isfinite(sigma_raw).all():
                raise ValueError("Posterior sigma_raw contains non-finite values.")

            sigma_min = float(np.min(sigma_raw))
            sigma_max = float(np.max(sigma_raw))
            if sigma_min < -1e-8 or sigma_max > 1 + 1e-8:
                raise ValueError(
                    "Posterior sigma_raw is outside the valid Beta scale range "
                    f"[0, 1]: min={sigma_min:.6g}, max={sigma_max:.6g}."
                )
            if sigma_min <= 0 or sigma_max >= 1:
                self.logger.warning(
                    "Posterior sigma_raw reached the Beta scale boundary "
                    f"(min={sigma_min:.6g}, max={sigma_max:.6g}); prediction "
                    "will clip the scale fraction inside the open interval."
                )

        elif likelihood == "gaussian":
            if "sigma_y" not in posterior_vars:
                raise ValueError(
                    "Posterior trace is missing sigma_y, which is required for "
                    "Gaussian posterior predictive sampling."
                )

    def validate_prediction_setup(
        self,
        prediction_data: dict[str, Any],
        mode: str,
    ) -> None:
        """Runtime checks for hierarchy resolution and prediction controls."""
        level_assignment = prediction_data["level_assignment"]
        if np.isnan(level_assignment).any():
            raise ValueError("Prediction hierarchy assignment contains NaN values.")

        level_summary, reason_summary = self.summarize_hierarchy_resolution(
            prediction_data, self.active_hierarchy_mapping
        )
        self.logger.info(
            f"Prediction setup for {mode}: resolved_observations={level_summary}; "
            f"fallback_reason={reason_summary}."
        )

        components = self.get_prediction_components()
        self.logger.info(
            f"Prediction components applied for {mode}: "
            + f"ecological={components['ecological']}, "
            + f"study_intercept={components['study_intercept']}, "
            + f"study_slopes={components['study_slopes']}, "
            + f"block_intercept={components['block_intercept']}."
        )
        if components["study_intercept"] and components["block_intercept"]:
            self.logger.info(
                "Prediction uses SSB block intercepts as nested study+block "
                "intercepts; study intercepts are not added a second time."
            )

        if mode == "test":
            uses_controls = any(
                components[key]
                for key in ["study_intercept", "study_slopes", "block_intercept"]
            )
            if uses_controls:
                raise ValueError(
                    "Test predictions with study/block controls are not supported "
                    "for fold-specific mappings. Set prediction_components controls "
                    "to false for deployable predictions."
                )

    def create_prediction_dataframe(
        self,
        prediction_data: dict[str, Any],
        mode: str,
        include_predictive_distribution: bool = True,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Extract site names, observed values, and predicted values (posterior
        means) and outputs two dataframes:
        1. A summary dataframe for training or cross-validation tasks.
        2. A dataframe containing the full predictive distribution for each
        site.

        Args:
            - prediction_data: Dataframe with new data if predictions are made
                out of sample, otherwise it's the training data.
            - mode: Either 'train' or 'test'. Note that this is different from
                the 'mode' attribute, which is related to the calling task.

        Returns:
            - df_pred: Dataframe with site names, observed values, and
                predictions (both capped and uncapped).
            - df_pred_distr: Dataframe containing the full predictive
                distribution for each site.
        """
        # Get site information and observed values
        site_idx = prediction_data["site_idx"]
        idx_to_site = {idx: name for name, idx in self.site_name_to_idx.items()}
        site_names = [idx_to_site[idx] for idx in site_idx]
        y_obs = prediction_data["y_obs"]

        # Get taxon information if applicable
        taxon_names: list[str | None] = [None] * len(site_names)
        if hasattr(self, "taxon_name_to_idx") and self.taxon_name_to_idx:
            taxon_idx = prediction_data["taxon_idx"]
            idx_to_taxon = {idx: name for name, idx in self.taxon_name_to_idx.items()}
            taxon_names = [idx_to_taxon[idx] for idx in taxon_idx]

        # Predictions for both train and test use prediction_components.
        y_pred_samples = self.prediction_trace.predictions["y_pred"]
        y_cond_samples = self.prediction_trace.predictions["y_cond"]
        ref_pred_samples = self.prediction_trace.predictions["y_intercept"]

        # Compute the posterior means for summary dataframe
        y_pred = y_pred_samples.mean(dim=("chain", "draw")).values
        y_cond = y_cond_samples.mean(dim=("chain", "draw")).values
        reference_pred = ref_pred_samples.mean(dim=("chain", "draw")).values

        # Create summary dataframe, now including Reference_pred
        df_pred = pl.DataFrame(
            {
                "SSBS": site_names,
                "Custom_taxonomic_group": taxon_names,
                "Observed": y_obs,
                "Predicted": y_cond,
                "y_pred": y_pred,
                "Reference_pred": reference_pred,
                "Hierarchy_level": prediction_data["resolved_level_name"],
                "Hierarchy_group": prediction_data["resolved_group"],
                "Hierarchy_fallback": prediction_data["fallback_reason"],
            }
        )

        if include_predictive_distribution:
            # Flatten the full predictive distribution into a long-form dataframe
            chain_dim, draw_dim = y_pred_samples.shape[
                0:2
            ]  # Extract chain and draw dimensions
            nb_sites = y_pred_samples.shape[2]  # Extract the number of sites

            df_pred_distr = pl.DataFrame(
                {
                    # Repeat site names for every combination of chain and draw
                    "SSBS": site_names * (chain_dim * draw_dim),
                    # Chain index, repeated for all draws and sites
                    "Chain": [
                        i for i in range(chain_dim) for _ in range(draw_dim * nb_sites)
                    ],
                    # Draw index, repeated for all sites within each chain
                    "Draw": [
                        j
                        for i in range(chain_dim)
                        for j in range(draw_dim)
                        for _ in range(nb_sites)
                    ],
                    # Flattened predictions, corresponding to (chain, draw, site)
                    "Prediction": y_pred_samples.values.flatten(),
                    "Reference_pred": ref_pred_samples.values.flatten(),
                }
            )
        else:
            df_pred_distr = pl.DataFrame()

        return df_pred, df_pred_distr
