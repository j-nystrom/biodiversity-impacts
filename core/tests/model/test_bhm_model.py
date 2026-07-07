import logging
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pymc as pm
import xarray as xr

from core.model.bhm_model import BayesianHierarchicalModel
from core.model.pymc_models import GeneralHierarchicalModel


def test_prior_predictive_defaults_to_none() -> None:
    """Optional runtime outputs have safe defaults."""
    model = BayesianHierarchicalModel(
        mode="training",
        random_seed=42,
        epsilon=1e-6,
        model_settings={
            "save_predictive_distributions": False,
            "sampler": {},
        },
        model_vars={
            "response_var": "response",
            "categorical_vars": [],
            "continuous_vars": [],
            "interaction_terms": [],
        },
        logger=logging.getLogger(__name__),
        site_name_to_idx={},
        taxon_name_to_idx={},
        hierarchy_mapping={},
    )

    assert model.prior_predictive is None
    assert model.progressbar is False


def test_parameter_summary_uses_group_intercepts_for_slope_response() -> None:
    """Response-scale slope summaries use matching ecological/study intercepts."""
    hierarchy_mapping: dict[str, Any] = {
        "column_names": {"level_1": "group"},
        "level_1": {"eco_group": 0},
    }
    model = BayesianHierarchicalModel(
        mode="training",
        random_seed=42,
        epsilon=1e-6,
        model_settings={
            "save_predictive_distributions": False,
            "sampler": {},
            "likelihood": "beta",
            "hierarchical_levels": 1,
            "training_components": {
                "ecological": True,
                "study_intercept": True,
                "study_slopes": True,
                "block_intercept": False,
            },
            "study_effects": {"slope_terms": ["x"]},
        },
        model_vars={
            "response_var": "response",
            "categorical_vars": [],
            "continuous_vars": ["x"],
            "interaction_terms": [],
        },
        logger=logging.getLogger(__name__),
        site_name_to_idx={},
        taxon_name_to_idx={},
        hierarchy_mapping=hierarchy_mapping,
    )
    alpha_group = np.log(0.8 / 0.2)
    gamma_study = np.log(0.7 / 0.3)
    posterior = xr.Dataset(
        {
            "mu_alpha": (("chain", "draw"), [[0.0]]),
            "mu_beta": (("chain", "draw", "x_vars"), [[[0.2]]]),
            "alpha_1": (("chain", "draw", "level_1_values"), [[[alpha_group]]]),
            "beta_1": (("chain", "draw", "level_1_values", "x_vars"), [[[[0.2]]]]),
            "gamma_study": (("chain", "draw", "study_names"), [[[gamma_study]]]),
            "delta_study_slope": (
                ("chain", "draw", "study_names", "study_slope_vars"),
                [[[[0.1]]]],
            ),
        },
        coords={
            "chain": [0],
            "draw": [0],
            "x_vars": ["x"],
            "level_1_values": ["eco_group"],
            "study_names": ["study_a"],
            "study_slope_vars": ["x"],
        },
    )
    model.trace = SimpleNamespace(posterior=posterior)

    summary = model.extract_parameter_summary()
    ecological_beta = summary.filter(
        (summary["parameter"] == "beta_1") & (summary["group"] == "eco_group")
    ).row(0, named=True)
    study_beta = summary.filter(
        (summary["parameter"] == "beta_study") & (summary["group"] == "study_a")
    ).row(0, named=True)

    expected_ecological = model._response_delta(
        np.asarray([alpha_group]),
        np.asarray([0.2]),
    )[0]
    expected_study = model._response_delta(
        np.asarray([gamma_study]),
        np.asarray([0.3]),
    )[0]

    assert ecological_beta["response_mean"] == expected_ecological
    assert study_beta["response_mean"] == expected_study


def test_effect_extraction_allows_distinct_alpha_beta_dimension_names() -> None:
    """Effect extraction handles PyMC-generated alpha/beta group dim names."""
    hierarchy_mapping: dict[str, Any] = {
        "column_names": {"level_2": "group"},
        "level_2": {"group_a": 0, "group_b": 1},
    }
    model = BayesianHierarchicalModel(
        mode="training",
        random_seed=42,
        epsilon=1e-6,
        model_settings={
            "save_predictive_distributions": False,
            "sampler": {},
            "likelihood": "beta",
            "hierarchical_levels": 2,
        },
        model_vars={
            "response_var": "response",
            "categorical_vars": [],
            "continuous_vars": ["x"],
            "interaction_terms": [],
        },
        logger=logging.getLogger(__name__),
        site_name_to_idx={},
        taxon_name_to_idx={},
        hierarchy_mapping=hierarchy_mapping,
    )
    posterior = xr.Dataset(
        {
            "mu_alpha": (("chain", "draw"), [[0.0]]),
            "mu_beta": (("chain", "draw", "x_vars"), [[[0.2]]]),
            "alpha_2": (("chain", "draw", "alpha_2_dim_0"), [[[0.0, 1.0]]]),
            "beta_2": (
                ("chain", "draw", "beta_2_dim_0", "beta_2_dim_1"),
                [[[[0.2], [0.3]]]],
            ),
        },
        coords={
            "chain": [0],
            "draw": [0],
            "x_vars": ["x"],
            "alpha_2_dim_0": [0, 1],
            "beta_2_dim_0": [0, 1],
            "beta_2_dim_1": [0],
        },
    )
    model.trace = SimpleNamespace(posterior=posterior)

    effects = model.extract_effects()
    group_values = cast(dict[str, float], effects["x"]["ecological_effect_values"])

    assert set(group_values) == {"group_a", "group_b"}


def test_population_only_training_ignores_unused_hierarchy_data() -> None:
    """SBM-style models should not register unused ecological hierarchy arrays."""
    model = GeneralHierarchicalModel(
        settings={
            "hierarchical_levels": 3,
            "varying_slope_level": 3,
            "likelihood": "beta",
            "hierarchy": {
                "level_1": ["Biome"],
                "level_2": ["Realm"],
                "level_3": [],
            },
            "priors": {
                "group_size_shrinkage": True,
                "group_size_shrinkage_scaling": "log",
                "group_size_shrinkage_max_scale": 1.0,
                "beta": {
                    "alpha_hyper_mean_mu": 0.35,
                    "hyperprior_sd_alpha": 0.75,
                    "hyperprior_sd_beta": 0.5,
                    "random_intercept_sd": 0.75,
                    "random_slope_sd": 0.5,
                    "beta_likelihood": {"alpha": 2, "beta": 12},
                },
            },
            "training_components": {
                "ecological": False,
                "study_intercept": True,
                "study_slopes": False,
                "block_intercept": True,
            },
            "prediction_components": {
                "ecological": False,
                "study_intercept": False,
                "study_slopes": False,
                "block_intercept": False,
            },
            "study_effects": {"slope_terms": []},
        },
        epsilon=1e-6,
    )
    model_data = {
        "coords": {
            "idx": [0],
            "x_vars": ["x"],
            "study_slope_vars": [],
            "study_names": ["study_a"],
            "block_names": ["block_a"],
        },
        "x_obs": np.asarray([[0.1]]),
        "x_study_slope_obs": np.zeros((1, 0)),
        "site_idx": np.asarray([0]),
        "y_obs": np.asarray([0.5]),
        "study_idx": np.asarray([0]),
        "block_idx": np.asarray([0]),
        "block_to_study_idx": np.asarray([0]),
        "level_1_idx": np.asarray([0]),
        "level_2_idx": np.asarray([0]),
        "level_3_idx": np.asarray([], dtype=np.int32),
        "level_2_to_level_1_idx": None,
        "level_3_to_level_2_idx": None,
    }

    pymc_model = model.build_training_model(model_data)

    assert "level_3_to_level_2_idx" not in pymc_model.named_vars


def test_study_effects_are_globally_sum_to_zero_centered() -> None:
    """Study intercepts and slopes are deviations around the global mean."""
    model = GeneralHierarchicalModel(
        settings={
            "hierarchical_levels": 1,
            "varying_slope_level": 1,
            "likelihood": "beta",
            "hierarchy": {
                "level_1": ["Biome"],
                "level_2": [],
                "level_3": [],
            },
            "priors": {
                "group_size_shrinkage": False,
                "group_size_shrinkage_scaling": "log",
                "beta": {
                    "alpha_hyper_mean_mu": 0.35,
                    "hyperprior_sd_alpha": 0.25,
                    "hyperprior_sd_beta": 0.18,
                    "random_intercept_sd": 0.1,
                    "random_slope_sd": 0.05,
                    "beta_likelihood": {"alpha": 2, "beta": 12},
                },
            },
            "training_components": {
                "ecological": False,
                "study_intercept": True,
                "study_slopes": True,
                "block_intercept": False,
            },
            "prediction_components": {
                "ecological": False,
                "study_intercept": False,
                "study_slopes": False,
                "block_intercept": False,
            },
            "study_effects": {"slope_terms": ["x"]},
        },
        epsilon=1e-6,
    )
    model_data = {
        "coords": {
            "idx": [0, 1],
            "x_vars": ["x"],
            "study_slope_vars": ["x"],
            "study_names": ["study_a", "study_b"],
            "block_names": ["block_a", "block_b"],
        },
        "x_obs": np.asarray([[0.0], [1.0]]),
        "x_study_slope_obs": np.asarray([[0.0], [1.0]]),
        "site_idx": np.asarray([0, 1]),
        "y_obs": np.asarray([0.4, 0.6]),
        "study_idx": np.asarray([0, 1]),
        "block_idx": np.asarray([0, 1]),
        "block_to_study_idx": np.asarray([0, 1]),
    }

    pymc_model = model.build_training_model(model_data)

    with pymc_model:
        gamma_study = np.asarray(
            pm.draw(pymc_model["gamma_study"], draws=5, random_seed=1)
        )
        delta_study_slope = np.asarray(
            pm.draw(pymc_model["delta_study_slope"], draws=5, random_seed=2)
        )

    assert np.allclose(gamma_study.sum(axis=-1), 0.0)
    assert np.allclose(delta_study_slope.sum(axis=-2), 0.0)


def test_study_effects_can_be_centered_within_ecological_groups() -> None:
    """Study effects can be centered within each ecological group."""
    model = GeneralHierarchicalModel(
        settings={
            "hierarchical_levels": 1,
            "varying_slope_level": 1,
            "likelihood": "beta",
            "hierarchy": {
                "level_1": ["Biome"],
                "level_2": [],
                "level_3": [],
            },
            "study_effect_centering": "ecological_group",
            "priors": {
                "group_size_shrinkage": False,
                "group_size_shrinkage_scaling": "log",
                "beta": {
                    "alpha_hyper_mean_mu": 0.35,
                    "hyperprior_sd_alpha": 0.25,
                    "group_prior_sd_alpha": 0.15,
                    "hyperprior_sd_beta": 0.18,
                    "group_prior_sd_beta_level_1": 0.07,
                    "random_intercept_sd": 0.1,
                    "random_slope_sd": 0.05,
                    "beta_likelihood": {"alpha": 2, "beta": 12},
                },
            },
            "training_components": {
                "ecological": True,
                "study_intercept": True,
                "study_slopes": True,
                "block_intercept": False,
            },
            "prediction_components": {
                "ecological": True,
                "study_intercept": False,
                "study_slopes": False,
                "block_intercept": False,
            },
            "study_effects": {"slope_terms": ["x"]},
        },
        epsilon=1e-6,
    )
    membership = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
    )
    model_data = {
        "coords": {
            "idx": [0, 1, 2, 3],
            "x_vars": ["x"],
            "study_slope_vars": ["x"],
            "study_names": ["study_a", "study_b", "study_c"],
            "block_names": ["block_a", "block_b", "block_c"],
            "level_1_values": ["group_a", "group_b"],
        },
        "x_obs": np.asarray([[0.0], [1.0], [0.0], [1.0]]),
        "x_study_slope_obs": np.asarray([[0.0], [1.0], [0.0], [1.0]]),
        "site_idx": np.asarray([0, 1, 2, 3]),
        "y_obs": np.asarray([0.4, 0.6, 0.5, 0.7]),
        "study_idx": np.asarray([0, 1, 1, 2]),
        "block_idx": np.asarray([0, 1, 1, 2]),
        "block_to_study_idx": np.asarray([0, 1, 2]),
        "level_1_idx": np.asarray([0, 0, 1, 1]),
        "level_1_n_studies": np.asarray([2, 2]),
        "level_1_study_membership": membership,
        "level_2_to_level_1_idx": None,
        "level_3_to_level_2_idx": None,
    }

    pymc_model = model.build_training_model(model_data)

    with pymc_model:
        gamma_study_group = np.asarray(
            pm.draw(pymc_model["gamma_study_group"], draws=5, random_seed=1)
        )
        delta_study_slope_group = np.asarray(
            pm.draw(pymc_model["delta_study_slope_group"], draws=5, random_seed=2)
        )

    assert np.allclose(
        (gamma_study_group * membership[None, :, :]).sum(axis=-1),
        0.0,
    )
    assert np.allclose(
        (delta_study_slope_group * membership[None, :, :, None]).sum(axis=-2),
        0.0,
    )


def test_ecological_group_dispersion_adds_group_level_sigma() -> None:
    """Opt-in beta dispersion varies by the deepest ecological group."""
    model = GeneralHierarchicalModel(
        settings={
            "hierarchical_levels": 1,
            "varying_slope_level": 1,
            "likelihood": "beta",
            "hierarchy": {
                "level_1": ["Biome"],
                "level_2": [],
                "level_3": [],
            },
            "ecological_group_dispersion": True,
            "priors": {
                "group_size_shrinkage": False,
                "group_size_shrinkage_scaling": "log",
                "beta": {
                    "alpha_hyper_mean_mu": 0.35,
                    "hyperprior_sd_alpha": 0.25,
                    "group_prior_sd_alpha": 0.15,
                    "hyperprior_sd_beta": 0.18,
                    "group_prior_sd_beta_level_1": 0.07,
                    "random_intercept_sd": 0.1,
                    "random_slope_sd": 0.05,
                    "ecological_group_dispersion_sd": 0.5,
                    "beta_likelihood": {"alpha": 2, "beta": 12},
                },
            },
            "training_components": {
                "ecological": True,
                "study_intercept": False,
                "study_slopes": False,
                "block_intercept": False,
            },
            "prediction_components": {
                "ecological": True,
                "study_intercept": False,
                "study_slopes": False,
                "block_intercept": False,
            },
            "study_effects": {"slope_terms": []},
        },
        epsilon=1e-6,
    )
    model_data = {
        "coords": {
            "idx": [0, 1],
            "x_vars": ["x"],
            "study_slope_vars": [],
            "study_names": ["study_a"],
            "block_names": ["block_a"],
            "level_1_values": ["group_a", "group_b"],
        },
        "x_obs": np.asarray([[0.0], [1.0]]),
        "x_study_slope_obs": np.zeros((2, 0)),
        "site_idx": np.asarray([0, 1]),
        "y_obs": np.asarray([0.4, 0.6]),
        "study_idx": np.asarray([0, 0]),
        "block_idx": np.asarray([0, 0]),
        "block_to_study_idx": np.asarray([0]),
        "level_1_idx": np.asarray([0, 1]),
        "level_1_n_studies": np.asarray([2, 2]),
        "level_2_to_level_1_idx": None,
        "level_3_to_level_2_idx": None,
    }

    pymc_model = model.build_training_model(model_data)

    assert "sigma_raw" in pymc_model.named_vars
    assert "sigma_raw_1" in pymc_model.named_vars
    assert "sigma_raw_group_sd" in pymc_model.named_vars


def test_ecological_group_dispersion_uses_configured_level() -> None:
    """Group-level beta dispersion can target a non-deepest hierarchy level."""
    model = GeneralHierarchicalModel(
        settings={
            "hierarchical_levels": 2,
            "varying_slope_level": 2,
            "likelihood": "beta",
            "hierarchy": {
                "level_1": ["Biome"],
                "level_2": ["Custom_taxonomic_group"],
                "level_3": [],
            },
            "ecological_group_dispersion": True,
            "ecological_group_dispersion_level": "level_1",
            "priors": {
                "group_size_shrinkage": False,
                "group_size_shrinkage_scaling": "log",
                "beta": {
                    "alpha_hyper_mean_mu": 0.35,
                    "hyperprior_sd_alpha": 0.25,
                    "group_prior_sd_alpha": 0.15,
                    "hyperprior_sd_beta": 0.18,
                    "group_prior_sd_beta_level_1": 0.07,
                    "group_prior_sd_beta_level_2": 0.05,
                    "random_intercept_sd": 0.1,
                    "random_slope_sd": 0.05,
                    "ecological_group_dispersion_sd": 0.5,
                    "beta_likelihood": {"alpha": 2, "beta": 12},
                },
            },
            "training_components": {
                "ecological": True,
                "study_intercept": False,
                "study_slopes": False,
                "block_intercept": False,
            },
            "prediction_components": {
                "ecological": True,
                "study_intercept": False,
                "study_slopes": False,
                "block_intercept": False,
            },
            "study_effects": {"slope_terms": []},
        },
        epsilon=1e-6,
    )
    model_data = {
        "coords": {
            "idx": [0, 1],
            "x_vars": ["x"],
            "study_slope_vars": [],
            "study_names": ["study_a"],
            "block_names": ["block_a"],
            "level_1_values": ["biome_a"],
            "level_2_values": ["taxon_a", "taxon_b"],
        },
        "x_obs": np.asarray([[0.0], [1.0]]),
        "x_study_slope_obs": np.zeros((2, 0)),
        "site_idx": np.asarray([0, 1]),
        "y_obs": np.asarray([0.4, 0.6]),
        "study_idx": np.asarray([0, 0]),
        "block_idx": np.asarray([0, 0]),
        "block_to_study_idx": np.asarray([0]),
        "level_1_idx": np.asarray([0, 0]),
        "level_2_idx": np.asarray([0, 1]),
        "level_1_n_studies": np.asarray([2]),
        "level_2_n_studies": np.asarray([2, 2]),
        "level_2_to_level_1_idx": np.asarray([0, 0]),
        "level_3_to_level_2_idx": None,
    }

    pymc_model = model.build_training_model(model_data)

    assert "sigma_raw" in pymc_model.named_vars
    assert "sigma_raw_1" in pymc_model.named_vars
    assert "sigma_raw_2" not in pymc_model.named_vars
