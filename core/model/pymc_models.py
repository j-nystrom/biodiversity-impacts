from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def general_hierarchical_model(
    model_data: dict[str, Any],
    settings: dict[str, Any],
) -> pm.Model:

    # Unpack the required model data from the dictionary
    coords = model_data["coords"]
    y_obs = model_data["y_obs"]
    x_obs = model_data["x_obs"]
    level_1_idx = model_data["level_1_idx"]
    level_2_idx = model_data["level_2_idx"]
    level_3_idx = model_data["level_3_idx"]
    level_2_to_level_1_idx = model_data["level_2_to_level_1_idx"]
    level_3_to_level_2_idx = model_data["level_3_to_level_2_idx"]
    site_idx = model_data["site_idx"]

    # Unpack model settings
    likelihood = settings["likelihood"]
    hierarchical_levels = settings["hierarchical_levels"]
    most_granular_slope_level = settings["most_granular_slope_level"]
    prior_values = settings["prior_values"]
    intercept_hyperprior_mean = prior_values["intercept_hyperprior_mean"]
    coef_prior_var = prior_values["coef_prior_var"]
    gaussian_noise = prior_values["gaussian_noise"]
    alpha_var_param = prior_values["alpha_var_param"]
    beta_var_param = prior_values["beta_var_param"]

    with pm.Model(coords=coords) as model:
        # Observed data variables; updated when predicting on test data
        y_obs = pm.Data("y_obs", y_obs, dims="idx")
        x_obs = pm.Data("x_obs", x_obs, dims=("idx", "x_vars"))
        level_1_idx = pm.Data("level_1_idx", level_1_idx, dims="idx")
        level_2_idx = pm.Data("level_2_idx", level_2_idx, dims="idx")
        level_3_idx = pm.Data("level_3_idx", level_3_idx, dims="idx")
        site_idx = pm.Data("site_idx", site_idx, dims="idx")  # noqa: F841

        # Hyperpriors for intercept and slope terms
        mu_a = pm.Normal("mu_a", mu=intercept_hyperprior_mean, sigma=coef_prior_var)
        sigma_a = pm.HalfNormal("sigma_a", sigma=coef_prior_var)
        mu_b = pm.Normal("mu_b", mu=0, sigma=coef_prior_var, dims="x_vars")
        sigma_b = pm.HalfNormal("sigma_b", sigma=coef_prior_var, dims="x_vars")

        # Top-level intercept and slope priors (non-centered parameterization)
        level_1_offset_1 = pm.Normal(
            "level_1_offset_1", mu=0, sigma=1, dims="level_1_values"
        )
        alpha_level_1 = pm.Deterministic(
            "alpha_level_1", mu_a + level_1_offset_1 * sigma_a
        )

        if most_granular_slope_level >= 1:
            level_1_offset_2 = pm.Normal(
                "level_1_offset_2", mu=0, sigma=1, dims=("level_1_values", "x_vars")
            )
            beta_level_1 = pm.Deterministic(
                "beta_level_1", mu_b + level_1_offset_2 * sigma_b
            )

        # Expected values (linear) at level 1
        if hierarchical_levels == 1:
            if most_granular_slope_level == 1:  # L1 varying slopes
                y_hat_linear = alpha_level_1[level_1_idx] + pt.sum(
                    x_obs * beta_level_1[level_1_idx], axis=1
                )
            elif most_granular_slope_level == 0:  # L1 varying intercepts
                y_hat_linear = alpha_level_1[level_1_idx] + pt.sum(x_obs * mu_b, axis=1)

        # Second-level intercept and slope priors
        # The mean values are sampled from the first level priors
        if hierarchical_levels >= 2:
            mu_a_level_2 = pm.Deterministic(
                "mu_a_level_2", alpha_level_1[level_2_to_level_1_idx]
            )
            sigma_a_level_2 = pm.HalfNormal("sigma_a_level_2", sigma=coef_prior_var)
            level_2_offset_1 = pm.Normal(
                "level_2_offset_1", mu=0, sigma=1, dims="level_2_values"
            )
            alpha_level_2 = pm.Deterministic(
                "alpha_level_2", mu_a_level_2 + level_2_offset_1 * sigma_a_level_2
            )

            if most_granular_slope_level >= 2:
                mu_b_level_2 = pm.Deterministic(
                    "mu_b_level_2", beta_level_1[level_2_to_level_1_idx]
                )
                sigma_b_level_2 = pm.HalfNormal("sigma_b_level_2", sigma=coef_prior_var)
                level_2_offset_2 = pm.Normal(
                    "level_2_offset_2", mu=0, sigma=1, dims=("level_2_values", "x_vars")
                )
                beta_level_2 = pm.Deterministic(
                    "beta_level_2", mu_b_level_2 + level_2_offset_2 * sigma_b_level_2
                )

        # Expected values (linear) at level 2
        if hierarchical_levels == 2:
            if most_granular_slope_level == 2:  # L2 varying slopes
                y_hat_linear = alpha_level_2[level_2_idx] + pt.sum(
                    x_obs * beta_level_2[level_2_idx], axis=1
                )
            elif most_granular_slope_level == 1:  # L2 intercepts, L1 slopes
                y_hat_linear = alpha_level_2[level_2_idx] + pt.sum(
                    x_obs * beta_level_1[level_1_idx], axis=1
                )
            elif most_granular_slope_level == 0:  # L2 intercepts, no varying slopes
                y_hat_linear = alpha_level_2[level_2_idx] + pt.sum(x_obs * mu_b, axis=1)

        # Third-level intercept and slope priors
        if hierarchical_levels == 3:
            mu_a_level_3 = pm.Deterministic(
                "mu_a_level_3", alpha_level_2[level_3_to_level_2_idx]
            )
            sigma_a_level_3 = pm.HalfNormal("sigma_a_level_3", sigma=coef_prior_var)
            level_3_offset_1 = pm.Normal(
                "level_3_offset_1", mu=0, sigma=1, dims="level_3_values"
            )
            alpha_level_3 = pm.Deterministic(
                "alpha_level_3", mu_a_level_3 + level_3_offset_1 * sigma_a_level_3
            )

            if most_granular_slope_level == 3:
                mu_b_level_3 = pm.Deterministic(
                    "mu_b_level_3", beta_level_2[level_3_to_level_2_idx]
                )
                sigma_b_level_3 = pm.HalfNormal("sigma_b_level_3", sigma=coef_prior_var)
                level_3_offset_2 = pm.Normal(
                    "level_3_offset_2", mu=0, sigma=1, dims=("level_3_values", "x_vars")
                )
                beta_level_3 = pm.Deterministic(
                    "beta_level_3", mu_b_level_3 + level_3_offset_2 * sigma_b_level_3
                )

        # Expected values (linear) at level 3
        if hierarchical_levels == 3:
            if most_granular_slope_level == 3:  # L3 varying slopes
                y_hat_linear = alpha_level_3[level_3_idx] + pt.sum(
                    x_obs * beta_level_3[level_3_idx], axis=1
                )
            elif most_granular_slope_level == 2:  # L3 intercepts, L2 slopes
                y_hat_linear = alpha_level_3[level_3_idx] + pt.sum(
                    x_obs * beta_level_2[level_2_idx], axis=1
                )
            elif most_granular_slope_level == 1:  # L3 intercepts, L1 slopes
                y_hat_linear = alpha_level_3[level_3_idx] + pt.sum(
                    x_obs * beta_level_1[level_1_idx], axis=1
                )
            elif most_granular_slope_level == 0:  # L3 intercepts, no varying slopes
                y_hat_linear = alpha_level_3[level_3_idx] + pt.sum(x_obs * mu_b, axis=1)

        # For Gaussian likelihood, the identity link function is used
        # The variance is assumed to be independent within and between studies
        if likelihood == "gaussian":
            y_hat = pm.Deterministic("y_hat", y_hat_linear)
            sigma_y = pm.HalfNormal("sigma_y", sigma=gaussian_noise)
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

        # For Beta likelihood, the expected values are transformed to 0-1 range
        # using the inverse logit link function The Beta distribution is
        # parameterized by the mean and variance
        elif likelihood == "beta":
            y_hat = pm.Deterministic("y_hat", pm.math.invlogit(y_hat_linear))
            sigma_raw = pm.Beta("sigma_raw", alpha=alpha_var_param, beta=beta_var_param)
            sigma_y = pm.Deterministic(
                "sigma", sigma_raw * pt.sqrt(y_hat * (1 - y_hat))
            )  # Upper bound is a function of the mean parameter
            y_like = pm.Beta(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

        return model


def two_level_varying_slope_gaussian_likelihood(model_data: dict[str, Any]) -> pm.Model:
    """
    Define a generic hierarchical model with two levels of grouping, normal
    intercept and slope priors (half-normal for variance terms), and a Gaussian
    likelihood function. Both intercept and slope terms are assumed to vary at
    the second level of grouping.

    Args:
        model_data: A dictionary containing all the data used by the model.

    Returns:
        model: A PyMC model object with the defined hierarchical model.
    """

    # Unpack the required model data from the dictionary
    coords = model_data["coords"]
    y_obs = model_data["y_obs"]
    x_obs = model_data["x_obs"]
    level_2_idx = model_data["level_2_idx"]
    level_2_to_level_1_idx = model_data["level_2_to_level_1_idx"]
    site_idx = model_data["site_idx"]

    with pm.Model(coords=coords) as model:
        # Observed data variables. Updated when predicting on test data
        y_obs = pm.Data("y_obs", y_obs, dims="idx")
        x_obs = pm.Data("x_obs", x_obs, dims=("idx", "x_vars"))
        level_2_idx = pm.Data("level_2_idx", level_2_idx, dims="idx")
        site_idx = pm.Data("site_idx", site_idx, dims="idx")  # noqa: F841

        # Hyperpriors for intercept and slope terms
        mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.05)
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.05)
        mu_b = pm.Normal("mu_b", mu=0, sigma=0.05, dims="x_vars")
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.05, dims="x_vars")

        # Top-level priors (non-centered parameterization)
        level_1_offset_1 = pm.Normal(
            "level_1_offset_1", mu=0, sigma=1, dims="level_1_values"
        )
        level_1_offset_2 = pm.Normal(
            "level_1_offset_2", mu=0, sigma=1, dims=("level_1_values", "x_vars")
        )
        alpha_level_1 = pm.Deterministic(
            "alpha_level_1", mu_a + level_1_offset_1 * sigma_a
        )
        beta_level_1 = pm.Deterministic(
            "beta_level_1", mu_b + level_1_offset_2 * sigma_b
        )

        # Second-level intercept and slope priors
        # The mean values are sampled from the first level priors
        mu_a_level_2 = pm.Deterministic(
            "mu_a_level_2", alpha_level_1[level_2_to_level_1_idx]
        )
        mu_b_level_2 = pm.Deterministic(
            "mu_b_level_2", beta_level_1[level_2_to_level_1_idx]
        )
        sigma_a_level_2 = pm.HalfNormal("sigma_a_level_2", sigma=0.05)
        sigma_b_level_2 = pm.HalfNormal("sigma_b_level_2", sigma=0.05)

        level_2_offset_1 = pm.Normal(
            "level_2_offset_1", mu=0, sigma=1, dims="level_2_values"
        )
        level_2_offset_2 = pm.Normal(
            "level_2_offset_2", mu=0, sigma=1, dims=("level_2_values", "x_vars")
        )
        alpha_level_2 = pm.Deterministic(
            "alpha_level_2", mu_a_level_2 + level_2_offset_1 * sigma_a_level_2
        )
        beta_level_2 = pm.Deterministic(
            "beta_level_2", mu_b_level_2 + level_2_offset_2 * sigma_b_level_2
        )

        # Data variance assumed to be independent within and between studies
        # NOTE: Should check this assumption for the scaled data
        sigma_y = pm.HalfNormal("sigma_y", sigma=0.05)

        # Formula for the expected values, with varying intercepts and slopes
        # at the second level of grouping
        y_hat = pm.Deterministic(
            "y_hat",
            alpha_level_2[level_2_idx]
            + pt.sum(x_obs * beta_level_2[level_2_idx], axis=1),
        )

        # Likelihood function
        y_like = pm.Normal(  # noqa: F841
            "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
        )

        return model


def two_level_varying_slope_beta_likelihood(model_data: dict[str, Any]) -> pm.Model:
    """
    Define a generic hierarchical model with two levels of grouping, normal
    intercept and slope priors (half-normal for variance terms), and a Beta
    likelihood function. Both intercept and slope terms are assumed to vary at
    the second level of grouping.

    Args:
        model_data: A dictionary containing all the data used by the model.

    Returns:
        model: A PyMC model object with the defined hierarchical model.
    """

    # Unpack the required model data from the dictionary
    coords = model_data["coords"]
    y_obs = model_data["y_obs"]
    x_obs = model_data["x_obs"]
    level_2_idx = model_data["level_2_idx"]
    level_2_to_level_1_idx = model_data["level_2_to_level_1_idx"]
    site_idx = model_data["site_idx"]

    with pm.Model(coords=coords) as model:
        # Observed data variables. Updated when predicting on test data
        y_obs = pm.Data("y_obs", y_obs, dims="idx")
        x_obs = pm.Data("x_obs", x_obs, dims=("idx", "x_vars"))
        level_2_idx = pm.Data("level_2_idx", level_2_idx, dims="idx")
        site_idx = pm.Data("site_idx", site_idx, dims="idx")  # noqa: F841

        # Hyperpriors for intercept and slope terms
        mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.05)
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.05)
        mu_b = pm.Normal("mu_b", mu=0, sigma=0.05, dims="x_vars")
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.05, dims="x_vars")

        # Top-level priors (non-centered parameterization)
        level_1_offset_1 = pm.Normal(
            "level_1_offset_1", mu=0, sigma=1, dims="level_1_values"
        )
        level_1_offset_2 = pm.Normal(
            "level_1_offset_2", mu=0, sigma=1, dims=("level_1_values", "x_vars")
        )
        alpha_level_1 = pm.Deterministic(
            "alpha_level_1", mu_a + level_1_offset_1 * sigma_a
        )
        beta_level_1 = pm.Deterministic(
            "beta_level_1", mu_b + level_1_offset_2 * sigma_b
        )

        # Second-level intercept and slope priors
        # The mean values are sampled from the first level priors
        mu_a_level_2 = pm.Deterministic(
            "mu_a_level_2", alpha_level_1[level_2_to_level_1_idx]
        )
        mu_b_level_2 = pm.Deterministic(
            "mu_b_level_2", beta_level_1[level_2_to_level_1_idx]
        )
        sigma_a_level_2 = pm.HalfNormal("sigma_a_level_2", sigma=0.05)
        sigma_b_level_2 = pm.HalfNormal("sigma_b_level_2", sigma=0.05)

        level_2_offset_1 = pm.Normal(
            "level_2_offset_1", mu=0, sigma=1, dims="level_2_values"
        )
        level_2_offset_2 = pm.Normal(
            "level_2_offset_2", mu=0, sigma=1, dims=("level_2_values", "x_vars")
        )
        alpha_level_2 = pm.Deterministic(
            "alpha_level_2", mu_a_level_2 + level_2_offset_1 * sigma_a_level_2
        )
        beta_level_2 = pm.Deterministic(
            "beta_level_2", mu_b_level_2 + level_2_offset_2 * sigma_b_level_2
        )

        # Formula for the expected values, with varying intercepts and slopes
        # at the second level of grouping
        y_hat = pm.Deterministic(
            "y_hat",
            pm.math.invlogit(
                alpha_level_2[level_2_idx]
                + pt.sum(x_obs * beta_level_2[level_2_idx], axis=1),
            ),
        )

        # Variance that satisfies mean-variance relationship for the Beta distr
        # 0 < sigma < sqrt(mu * (1 - mu))
        sigma_raw = pm.Beta("sigma_raw", alpha=2, beta=5)  # Raw sigma in (0, 1)
        sigma_y = pm.Deterministic("sigma", sigma_raw * pt.sqrt(y_hat * (1 - y_hat)))

        # Likelihood function
        y_like = pm.Beta(  # noqa: F841
            "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
        )

        return model


def three_level_varying_slope_gaussian_likelihood(
    model_data: dict[str, Any]
) -> pm.Model:
    """
    Defines a generic hierarchical model with three levels of grouping, normal
    intercept and slope priors (half-normal for variance terms), and a Gaussian
    likelihood function. Both intercept and slope terms are assumed to vary at
    each level of grouping.

    Args:
        model_data: A dictionary containing all the data used by the model.

    Returns:
        model: A PyMC model object with the defined hierarchical model.
    """

    # Unpack the required model data from the dictionary
    coords = model_data["coords"]
    y_obs = model_data["y_obs"]
    x_obs = model_data["x_obs"]
    level_3_idx = model_data["level_3_idx"]
    level_2_to_level_1_idx = model_data["level_2_to_level_1_idx"]
    level_3_to_level_2_idx = model_data["level_3_to_level_2_idx"]
    site_idx = model_data["site_idx"]

    with pm.Model(coords=coords) as model:
        # Observed data variables. Updated when predicting on test data
        y_obs = pm.Data("y_obs", y_obs, dims="idx")
        x_obs = pm.Data("x_obs", x_obs, dims=("idx", "x_vars"))
        level_3_idx = pm.Data("level_3_idx", level_3_idx, dims="idx")
        site_idx = pm.Data("site_idx", site_idx, dims="idx")  # noqa: F841

        # Hyperpriors for intercept and slope terms
        mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.1)
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.1)
        mu_b = pm.Normal("mu_b", mu=0, sigma=0.1, dims="x_vars")
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.1, dims="x_vars")

        # Top-level priors (Level 1)
        level_1_offset_1 = pm.Normal(
            "level_1_offset_1", mu=0, sigma=1, dims="level_1_values"
        )
        level_1_offset_2 = pm.Normal(
            "level_1_offset_2", mu=0, sigma=1, dims=("level_1_values", "x_vars")
        )
        alpha_level_1 = pm.Deterministic(
            "alpha_level_1", mu_a + level_1_offset_1 * sigma_a
        )
        beta_level_1 = pm.Deterministic(
            "beta_level_1", mu_b + level_1_offset_2 * sigma_b
        )

        # Second-level priors
        mu_a_level_2 = pm.Deterministic(
            "mu_a_level_2", alpha_level_1[level_2_to_level_1_idx]
        )
        mu_b_level_2 = pm.Deterministic(
            "mu_b_level_2", beta_level_1[level_2_to_level_1_idx]
        )
        sigma_a_level_2 = pm.HalfNormal("sigma_a_level_2", sigma=0.1)
        sigma_b_level_2 = pm.HalfNormal("sigma_b_level_2", sigma=0.1)

        level_2_offset_1 = pm.Normal(
            "level_2_offset_1", mu=0, sigma=1, dims="level_2_values"
        )
        level_2_offset_2 = pm.Normal(
            "level_2_offset_2", mu=0, sigma=1, dims=("level_2_values", "x_vars")
        )
        alpha_level_2 = pm.Deterministic(
            "alpha_level_2", mu_a_level_2 + level_2_offset_1 * sigma_a_level_2
        )
        beta_level_2 = pm.Deterministic(
            "beta_level_2", mu_b_level_2 + level_2_offset_2 * sigma_b_level_2
        )

        # Third-level priors
        mu_a_level_3 = pm.Deterministic(
            "mu_a_level_3", alpha_level_2[level_3_to_level_2_idx]
        )
        mu_b_level_3 = pm.Deterministic(
            "mu_b_level_3", beta_level_2[level_3_to_level_2_idx]
        )
        sigma_a_level_3 = pm.HalfNormal("sigma_a_level_3", sigma=0.1)
        sigma_b_level_3 = pm.HalfNormal("sigma_b_level_3", sigma=0.1)

        level_3_offset_1 = pm.Normal(
            "level_3_offset_1", mu=0, sigma=1, dims="level_3_values"
        )
        level_3_offset_2 = pm.Normal(
            "level_3_offset_2", mu=0, sigma=1, dims=("level_3_values", "x_vars")
        )
        alpha_level_3 = pm.Deterministic(
            "alpha_level_3", mu_a_level_3 + level_3_offset_1 * sigma_a_level_3
        )
        beta_level_3 = pm.Deterministic(
            "beta_level_3", mu_b_level_3 + level_3_offset_2 * sigma_b_level_3
        )

        # Data variance assumed to be independent within and between studies
        sigma_y = pm.HalfNormal("sigma_y", sigma=0.1)

        # Expected values with varying intercepts and slopes at Level 3
        y_hat = pm.Deterministic(
            "y_hat",
            alpha_level_3[level_3_idx]
            + pt.sum(x_obs * beta_level_3[level_3_idx], axis=1),
        )

        # Likelihood function
        y_like = pm.Normal(  # noqa: F841
            "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
        )

        return model


def biome_realm_ecoregion_slope_model(model_data: dict[str, Any]) -> pm.Model:
    """
    Defines the hierarchical model, starting with hyperparameters for the
    biome-level intercept and slope terms. The model then samples the
    intercept and slope terms for the realm and ecoregion levels from the
    corresponding biome and realm level parameters. The numeric priors are
    defined in this model object.

    Args:
        model_data: A dictionary containing all the data used by the model.

    Returns:
        model: A PyMC model object with the defined hierarchical model.
    """

    # Unpack the required model data from dictionary
    coords = model_data["coords"]
    y = model_data["y"]
    x = model_data["x"]
    realm_to_biome_idx = model_data["realm_to_biome_idx"]
    eco_to_realm_idx = model_data["eco_to_realm_idx"]
    biome_realm_eco_idx = model_data["biome_realm_eco_idx"]
    biome_realm_idx = model_data["biome_realm_idx"]

    with pm.Model(coords=coords) as model:
        # Observed data variables
        y_obs = pm.Data("y_obs", y, dims="idx")
        x_obs = pm.Data("x_obs", x, dims=("idx", "x_vars"))
        biome_realm_eco_idx = pm.Data(
            "biome_realm_eco_idx", biome_realm_eco_idx, dims="idx"
        )
        biome_realm_idx = pm.Data("biome_realm_idx", biome_realm_idx, dims="idx")
        site_idx = pm.Data(  # noqa: F841
            "site_idx", model_data["site_idx"], dims="idx"
        )  # This index is updated when predicting on test data

        # Hyperpriors for biome-level intercept and slope terms
        mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.1)
        sigma_a = pm.HalfNormal("sigma_a", sigma=0.1)
        mu_b = pm.Normal("mu_b", mu=0, sigma=0.1, dims="x_vars")
        sigma_b = pm.HalfNormal("sigma_b", sigma=0.1, dims="x_vars")

        # Biome-level priors (non-centered parameterization)
        biome_offset_1 = pm.Normal("biome_offset_1", mu=0, sigma=1, dims="biomes")
        biome_offset_2 = pm.Normal(
            "biome_offset_2", mu=0, sigma=1, dims=("biomes", "x_vars")
        )
        alpha_biome = pm.Deterministic("alpha_biome", mu_a + biome_offset_1 * sigma_a)
        beta_biome = pm.Deterministic("beta_biome", mu_b + biome_offset_2 * sigma_b)

        # Realm-level intercepts and slopes, sampled from the corresponding biomes
        mu_a_realm = pm.Deterministic("mu_a_realm", alpha_biome[realm_to_biome_idx])
        mu_b_realm = pm.Deterministic("mu_b_realm", beta_biome[realm_to_biome_idx])
        sigma_a_realm = pm.HalfNormal("sigma_a_realm", sigma=0.1)
        sigma_b_realm = pm.HalfNormal("sigma_b_realm", sigma=0.1)

        realm_offset_1 = pm.Normal("realm_offset_1", mu=0, sigma=1, dims="biome_realm")
        realm_offset_2 = pm.Normal(
            "realm_offset_2", mu=0, sigma=1, dims=("biome_realm", "x_vars")
        )

        alpha_realm = pm.Deterministic(
            "alpha_realm", mu_a_realm + realm_offset_1 * sigma_a_realm
        )
        beta_realm = pm.Deterministic(
            "beta_realm", mu_b_realm + realm_offset_2 * sigma_b_realm
        )

        # Ecoregion-level intercepts and slopes, sampled from realms
        mu_a_eco = pm.Deterministic("mu_a_eco", alpha_realm[eco_to_realm_idx])
        mu_b_eco = pm.Deterministic("mu_b_eco", beta_realm[eco_to_realm_idx])
        sigma_a_eco = pm.HalfNormal("sigma_a_eco", sigma=0.1)
        sigma_b_eco = pm.HalfNormal("sigma_b_eco", sigma=0.1)

        eco_offset_1 = pm.Normal("eco_offset_1", mu=0, sigma=1, dims="biome_realm_eco")
        eco_offset_2 = pm.Normal(
            "eco_offset_2", mu=0, sigma=1, dims=("biome_realm_eco", "x_vars")
        )

        alpha_eco = pm.Deterministic("alpha_eco", mu_a_eco + eco_offset_1 * sigma_a_eco)
        beta_eco = pm.Deterministic("beta_eco", mu_b_eco + eco_offset_2 * sigma_b_eco)

        # Variance assumed independent within and between studies
        sigma_y = pm.HalfNormal("sigma_y", sigma=0.1)

        # Expected values
        y_hat = pm.Deterministic(
            "y_hat",
            alpha_eco[biome_realm_eco_idx]
            + pt.sum(x_obs * beta_eco[biome_realm_eco_idx], axis=1),
        )

        # Ecoregion intercepts at the site level
        # Used for final scaling of outputs (dividing prediction by intercept)
        alpha_eco_site = pm.Deterministic(  # noqa: F841
            "alpha_eco_site", alpha_eco[biome_realm_eco_idx]
        )

        # Likelihood function
        y_like = pm.Normal(  # noqa: F841
            "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
        )

        return model


class CompletePoolingModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        x = model_data["x"]

        with pm.Model(coords=coords) as model:
            # Observed data variables
            y_obs = pm.Data("y_obs", y, dims="idx")
            x_obs = pm.Data("x_obs", x, dims=("idx", "x_vars"))

            # Priors on intercept and slopes
            alpha = pm.Normal("alpha", mu=0.5, sigma=0.25)
            beta = pm.Normal("beta", mu=0, sigma=0.1, dims="x_vars")

            # Independent noise
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.25)

            # Expected values
            y_hat = pm.Deterministic("y_hat", alpha + pt.dot(x_obs, beta))

            # Likelihood function
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

        return model


class StudyInterceptModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        x = model_data["x"]
        study_idx = model_data["study_idx"]

        with pm.Model(coords=coords) as model:
            # Observed data variables
            y_obs = pm.Data("y_obs", y, dims="idx")
            x_obs = pm.Data("x_obs", x, dims=("idx", "x_vars"))

            # Hyperpriors for study-level intercept terms
            mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.25)
            sigma_a = pm.HalfNormal("sigma_a", sigma=0.25)

            # Study-level intercept priors (non-centered parameterization)
            study_offset = pm.Normal("study_offset", mu=0, sigma=1, dims="studies")
            alpha_study = pm.Deterministic("alpha_study", mu_a + study_offset * sigma_a)

            # Population level priors for slope parameters
            beta = pm.Normal("beta", mu=0, sigma=0.1, dims="x_vars")

            # Variance assumed independent within and between studies
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.25)

            # Expected values
            y_hat = pm.Deterministic(
                "y_hat", alpha_study[study_idx] + pt.dot(x_obs, beta)
            )

            # Likelihood function
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

        return model


class StudyBlockInterceptModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        x = model_data["x"]
        block_idx = model_data["block_idx"]
        block_to_study_idx = model_data["block_to_study_idx"]

        with pm.Model(coords=coords) as model:
            # Observed data variables
            y_obs = pm.Data("y_obs", y, dims="idx")
            x_obs = pm.Data("x_obs", x, dims=("idx", "x_vars"))

            # Hyperpriors for study-level intercept terms
            mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.25)
            sigma_a = pm.HalfNormal("sigma_a", sigma=0.25)

            # Study-level intercept priors (non-centered parameterization)
            study_offset = pm.Normal("study_offset", mu=0, sigma=1, dims="studies")
            alpha_study = pm.Deterministic("alpha_study", mu_a + study_offset * sigma_a)

            # Block-level intercepts, sampled from the corresponding studies
            mu_block = pm.Deterministic("mu_block", alpha_study[block_to_study_idx])
            sigma_block = pm.HalfNormal("sigma_block", sigma=0.1)
            block_offset = pm.Normal("block_offset", mu=0, sigma=1, dims="blocks")
            alpha_block = pm.Deterministic(
                "alpha_block", mu_block + block_offset * sigma_block
            )

            # Population level fixed effects priors for slope parameters
            beta = pm.Normal("beta", mu=0, sigma=0.1, dims="x_vars")

            # Variance assumed independent within and between studies and blocks
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.25)

            # Expected values
            y_hat = pm.Deterministic(
                "y_hat", alpha_block[block_idx] + pt.dot(x_obs, beta)
            )

            # Likelihood function
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

            return model


class StudySlopeModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        z = model_data["z"]
        x_z_diff = model_data["x_z_diff"]
        study_idx = model_data["study_idx"]
        block_to_study_idx = model_data["block_to_study_idx"]
        block_idx = model_data["block_idx"]

        with pm.Model(coords=coords) as model:
            # Observed data variables
            y_obs = pm.Data("y_obs", y, dims="idx")
            z_obs = pm.Data("z_study", z, dims=("idx", "z_vars"))
            x_res = pm.Data("x_res", x_z_diff, dims=("idx", "x_z_diff_vars"))

            # Hyperpriors for study-level intercept and slopes
            mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.25)
            mu_b = pm.Normal("mu_b", mu=0, sigma=0.1, dims="z_vars")

            # Biome-level priors with covariance structure, on some slopes
            # (non-centered parameterization)
            z_dim = len(coords["z_vars_int"])
            sd_study = pm.Exponential.dist(lam=1, shape=z_dim)
            chol_study, _, _ = pm.LKJCholeskyCov(
                "chol_study", n=z_dim, eta=2, sd_dist=sd_study
            )
            study_offset = pm.Normal(
                "study_offset", mu=0, sigma=1, dims=("studies", "z_vars_int")
            )
            alpha_beta_study = pm.Deterministic(
                "alpha_beta_study",
                pt.concatenate([pt.reshape(mu_a, (1,)), mu_b], axis=0)
                + pt.dot(chol_study, study_offset.T).T,
                dims=("studies", "z_vars_int"),
            )

            # Block-level intercepts, sampled from the corresponding studies
            mu_block = pm.Deterministic(
                "mu_block", alpha_beta_study[block_to_study_idx, 0]
            )
            sigma_block = pm.HalfNormal("sigma_block", sigma=0.1)
            block_offset = pm.Normal("block_offset", mu=0, sigma=1, dims="blocks")
            alpha_block = pm.Deterministic(
                "alpha_block", mu_block + block_offset * sigma_block
            )

            # Population-level priors for the other slope parameters
            beta = pm.Normal("beta", mu=0, sigma=0.1, dims="x_z_diff_vars")

            # Variance assumed independent within and between studies
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.25)

            # Expected values
            y_hat = pm.Deterministic(
                "y_hat",
                alpha_block[block_idx]
                + pt.sum(z_obs * alpha_beta_study[study_idx, 1:], axis=1)
                + pt.dot(x_res, beta),
            )

            # Likelihood function
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

            return model


class BiomeRealmIndependentModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        x = model_data["x"]
        biome_realm_idx = model_data["biome_realm_idx"]
        realm_to_biome_idx = model_data["realm_to_biome_idx"]

        with pm.Model(coords=coords) as model:
            # Observed data variables
            y_obs = pm.Data("y_obs", value=y, dims="idx")
            x_obs = pm.Data("x_obs", value=x, dims=("idx", "x_vars"))

            # Hyperpriors for biome-level intercept and slope terms
            mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.25)
            sigma_a = pm.HalfNormal("sigma_a", sigma=0.25)
            mu_b = pm.Normal("mu_b", mu=0, sigma=0.1, dims="x_vars")
            sigma_b = pm.HalfNormal("sigma_b", sigma=0.25, dims="x_vars")

            # Biome-level priors (non-centered parameterization)
            biome_offset_1 = pm.Normal("biome_offset_1", mu=0, sigma=1, dims="biomes")
            biome_offset_2 = pm.Normal(
                "biome_offset_2", mu=0, sigma=1, dims=("biomes", "x_vars")
            )
            alpha_biome = pm.Deterministic(
                "alpha_biome", mu_a + biome_offset_1 * sigma_a
            )
            beta_biome = pm.Deterministic("beta_biome", mu_b + biome_offset_2 * sigma_b)

            # Realm-level intercepts and slopes, sampled from the corresponding biomes
            mu_a_realm = pm.Deterministic("mu_a_realm", alpha_biome[realm_to_biome_idx])
            mu_b_realm = pm.Deterministic("mu_b_realm", beta_biome[realm_to_biome_idx])
            sigma_a_realm = pm.HalfNormal("sigma_a_realm", sigma=0.25)
            sigma_b_realm = pm.HalfNormal("sigma_b_realm", sigma=0.25)

            realm_offset_1 = pm.Normal(
                "realm_offset_1", mu=0, sigma=1, dims="biome_realm"
            )
            realm_offset_2 = pm.Normal(
                "realm_offset_2", mu=0, sigma=1, dims=("biome_realm", "x_vars")
            )

            alpha_realm = pm.Deterministic(
                "alpha_realm", mu_a_realm + realm_offset_1 * sigma_a_realm
            )
            beta_realm = pm.Deterministic(
                "beta_realm", mu_b_realm + realm_offset_2 * sigma_b_realm
            )

            # Variance assumed independent within and between studies
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.25)

            # Expected values
            y_hat = pm.Deterministic(
                "y_hat",
                alpha_realm[biome_realm_idx]
                + pt.sum(x_obs * beta_realm[biome_realm_idx], axis=1),
            )

            # Likelihood function
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

            return model


class BiomeRealmEcoInterceptModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        x = model_data["x"]
        realm_to_biome_idx = model_data["realm_to_biome_idx"]
        biome_realm_idx = model_data["biome_realm_idx"]
        biome_realm_eco_idx = model_data["biome_realm_eco_idx"]
        eco_to_realm_idx = model_data["eco_to_realm_idx"]

        with pm.Model(coords=coords) as model:
            # Observed data variables
            y_obs = pm.Data("y_obs", value=y, dims="idx")
            x_obs = pm.Data("x_obs", value=x, dims=("idx", "x_vars"))

            # Hyperpriors for biome-level intercept and slope terms
            mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.25)
            sigma_a = pm.HalfNormal("sigma_a", sigma=0.25)
            mu_b = pm.Normal("mu_b", mu=0, sigma=0.1, dims="x_vars")
            sigma_b = pm.HalfNormal("sigma_b", sigma=0.25, dims="x_vars")

            # Biome-level priors (non-centered parameterization)
            biome_offset_1 = pm.Normal("biome_offset_1", mu=0, sigma=1, dims="biomes")
            biome_offset_2 = pm.Normal(
                "biome_offset_2", mu=0, sigma=1, dims=("biomes", "x_vars")
            )
            alpha_biome = pm.Deterministic(
                "alpha_biome", mu_a + biome_offset_1 * sigma_a
            )
            beta_biome = pm.Deterministic("beta_biome", mu_b + biome_offset_2 * sigma_b)

            # Realm-level intercepts and slopes, sampled from the corresponding biomes
            mu_a_realm = pm.Deterministic("mu_a_realm", alpha_biome[realm_to_biome_idx])
            mu_b_realm = pm.Deterministic("mu_b_realm", beta_biome[realm_to_biome_idx])
            sigma_a_realm = pm.HalfNormal("sigma_a_realm", sigma=0.1)
            sigma_b_realm = pm.HalfNormal("sigma_b_realm", sigma=0.1)

            realm_offset_1 = pm.Normal(
                "realm_offset_1", mu=0, sigma=1, dims="biome_realm"
            )
            realm_offset_2 = pm.Normal(
                "realm_offset_2", mu=0, sigma=1, dims=("biome_realm", "x_vars")
            )

            alpha_realm = pm.Deterministic(
                "alpha_realm", mu_a_realm + realm_offset_1 * sigma_a_realm
            )
            beta_realm = pm.Deterministic(
                "beta_realm", mu_b_realm + realm_offset_2 * sigma_b_realm
            )

            # Ecoregion-level intercepts and slopes, sampled from realms
            mu_a_eco = pm.Deterministic("mu_a_eco", alpha_realm[eco_to_realm_idx])
            # mu_b_eco = pm.Deterministic("mu_b_eco", beta_realm[eco_to_realm_idx])
            sigma_a_eco = pm.HalfNormal("sigma_a_eco", sigma=0.1)
            # sigma_b_eco = pm.HalfNormal("sigma_b_eco", sigma=0.1)

            eco_offset_1 = pm.Normal(
                "eco_offset_1", mu=0, sigma=1, dims="biome_realm_eco"
            )
            # eco_offset_2 = pm.Normal(
            # "eco_offset_2", mu=0, sigma=1, dims=("biome_realm_eco", "x_vars")
            # )

            alpha_eco = pm.Deterministic(
                "alpha_eco", mu_a_eco + eco_offset_1 * sigma_a_eco
            )
            # beta_eco = pm.Deterministic(
            # "beta_eco", mu_b_eco + eco_offset_2 * sigma_b_eco
            # )

            # Variance assumed independent within and between studies
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.25)

            # Expected values
            y_hat = pm.Deterministic(
                "y_hat",
                alpha_eco[biome_realm_eco_idx]
                + pt.sum(x_obs * beta_realm[biome_realm_idx], axis=1),
            )

            # Likelihood function
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

            return model


class BiomeRealmCovarianceModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        x = model_data["x"]
        biome_realm_idx = model_data["biome_realm_idx"]
        realm_to_biome_idx = model_data["realm_to_biome_idx"]

        with pm.Model(coords=coords) as model:

            # Observed data variables
            y_obs = pm.Data("y_obs", value=y, dims="idx")
            x_obs = pm.Data("x_obs", value=x, dims=("idx", "x_vars"))

            # Hyperpriors for biome-level intercept and slopes
            mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.1)
            mu_b = pm.Normal("mu_b", mu=0, sigma=0.1, dims="x_vars")

            # Biome-level priors with covariance structure
            # (non-centered parameterization)
            x_dim = len(coords["x_vars_int"])
            sd_biome = pm.Exponential.dist(lam=1, shape=x_dim)
            chol_biome, _, _ = pm.LKJCholeskyCov(
                "chol_biome", n=x_dim, eta=2, sd_dist=sd_biome
            )
            biome_offset = pm.Normal(
                "biome_offset", mu=0, sigma=1, dims=("biomes", "x_vars_int")
            )
            alpha_beta_biome = pm.Deterministic(
                "alpha_beta_biome",
                pt.concatenate([pt.reshape(mu_a, (1,)), mu_b], axis=0)
                + pt.dot(chol_biome, biome_offset.T).T,
                dims=("biomes", "x_vars_int"),
            )

            # Realm-level intercepts and slopes, sampled from the corresponding biomes
            sigma_realm = pm.HalfNormal("sigma_realm", sigma=0.1, dims="x_vars_int")
            realm_offset = pm.Normal(
                "realm_offset", mu=0, sigma=1, dims=("biome_realm", "x_vars_int")
            )
            alpha_beta_realm = pm.Deterministic(
                "alpha_beta_realm",
                alpha_beta_biome[realm_to_biome_idx] + realm_offset * sigma_realm,
                dims=("biome_realm", "x_vars_int"),
            )

            # Variance assumed independent within and between studies
            sigma_y = pm.HalfNormal("sigma_y", sigma=0.25)

            # Expected values
            y_hat = pm.Deterministic(
                "y_hat",
                alpha_beta_realm[biome_realm_idx, 0]
                + pt.sum(x_obs * alpha_beta_realm[biome_realm_idx, 1:], axis=1),
            )

            # Likelihood function
            y_like = pm.Normal(  # noqa: F841
                "y_like", mu=y_hat, sigma=sigma_y, observed=y_obs, dims="idx"
            )

            return model


class BiomeRealmBetaModel:
    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
        """Docstring."""

        # Unpack the required model data from dictionary
        coords = model_data["coords"]
        y = model_data["y"]
        x = model_data["x"]
        biome_realm_idx = model_data["biome_realm_idx"]
        realm_to_biome_idx = model_data["realm_to_biome_idx"]

        with pm.Model(coords=coords) as model:
            # Observed data variables
            y_obs = pm.Data("y_obs", value=y, dims="idx")
            x_obs = pm.Data("x_obs", value=x, dims=("idx", "x_vars"))

            # Hyperpriors for biome-level intercept and slope terms
            mu_a = pm.Normal("mu_a", mu=0.5, sigma=0.25)
            sigma_a = pm.HalfNormal("sigma_a", sigma=0.25)
            mu_b = pm.Normal("mu_b", mu=0, sigma=0.1, dims="x_vars")
            sigma_b = pm.HalfNormal("sigma_b", sigma=0.25, dims="x_vars")

            # Biome-level priors (non-centered parameterization)
            biome_offset_1 = pm.Normal("biome_offset_1", mu=0, sigma=1, dims="biomes")
            biome_offset_2 = pm.Normal(
                "biome_offset_2", mu=0, sigma=1, dims=("biomes", "x_vars")
            )
            alpha_biome = pm.Deterministic(
                "alpha_biome", mu_a + biome_offset_1 * sigma_a
            )
            beta_biome = pm.Deterministic("beta_biome", mu_b + biome_offset_2 * sigma_b)

            # Realm-level intercepts and slopes, sampled from the corresponding biomes
            mu_a_realm = pm.Deterministic("mu_a_realm", alpha_biome[realm_to_biome_idx])
            mu_b_realm = pm.Deterministic("mu_b_realm", beta_biome[realm_to_biome_idx])
            sigma_a_realm = pm.HalfNormal("sigma_a_realm", sigma=0.25)
            sigma_b_realm = pm.HalfNormal("sigma_b_realm", sigma=0.25)

            realm_offset_1 = pm.Normal(
                "realm_offset_1", mu=0, sigma=1, dims="biome_realm"
            )
            realm_offset_2 = pm.Normal(
                "realm_offset_2", mu=0, sigma=1, dims=("biome_realm", "x_vars")
            )

            alpha_realm = pm.Deterministic(
                "alpha_realm", mu_a_realm + realm_offset_1 * sigma_a_realm
            )
            beta_realm = pm.Deterministic(
                "beta_realm", mu_b_realm + realm_offset_2 * sigma_b_realm
            )

            y_hat = pm.Deterministic(
                "y_hat",
                pm.math.invlogit(
                    alpha_realm[biome_realm_idx]
                    + pt.sum(x_obs * beta_realm[biome_realm_idx], axis=1)
                ),
            )

            # Variance is defined in relation to the expected value
            # a = pm.Beta("a", alpha=2, beta=5)
            # sigma_y = pm.Deterministic("sigma_y", a * pt.sqrt(y_hat * (1 - y_hat)))
            sigma_y = pm.HalfNormal("sigma_y", sigma=1)

            # Likelihood function
            y_like = pm.Beta(  # noqa: F841
                "y_like", alpha=y_hat, beta=sigma_y, observed=y_obs, dims="idx"
            )

            return model


def bii_abund_indep_re(
    y: np.array,
    x: np.array,
    coords: dict[str, np.array],
    study_idx: np.array,
    block_idx: np.array,
    block_to_study_idx: np.array,
    z_s: np.array = None,
    z_b: np.array = None,
    s_intercept: bool = False,
    b_intercept: bool = False,
) -> pm.Model:
    """Docstring."""
    with pm.Model(coords=coords) as model:

        # Observed data that be changed later on for train-test runs
        y = pm.MutableData("y", y, dims="idx")
        x = pm.MutableData("x", x, dims=("idx", "x_var"))
        if z_s:
            z_s = pm.MutableData("z_s", z_s, dims=("idx", "z_s_var"))
        if z_b:
            z_b = pm.MutableData("z_b", z_b, dims=("idx", "z_b_var"))

        # Population level fixed effects
        # Priors independently sampled from univariate normal
        beta = pm.Normal("beta", mu=0, sigma=1, dims="x_var")

        # Random slope priors
        if z_s:
            # Study level random effects
            gamma_s = pm.Normal("gamma_s", mu=0, sigma=1, dims=("studies", "z_s_var"))
        if z_b:
            # Block level random effects, sampled hierarchically from study effects
            gamma_b_mu = gamma_s[block_to_study_idx]
            gamma_b = pm.Normal(
                "gamma_b", mu=gamma_b_mu, sigma=1, dims=("blocks", "z_b_var")
            )

        # Random intercept priors
        if not z_s and s_intercept:
            # Study level random intercepts
            gamma_s = pm.Normal("gamma_s", mu=0, sigma=1, dims="studies")

        if not z_b and b_intercept:
            # Block level random intercepts
            gamma_b_mu = gamma_s[block_to_study_idx]
            gamma_b = pm.Normal("gamma_b", mu=gamma_b_mu, sigma=1, dims="blocks")

        # Expected abundance value based on fixed effects
        fe_sum = pm.math.sum(x * beta, axis=1)

        re_study_sum, re_block_sum = 0, 0  # Default values

        # Random effects contribution depending on structure
        if z_s:
            re_study_sum = pm.math.sum(z_s * gamma_s[study_idx], axis=1)
        if not z_s and s_intercept:
            re_study_sum = gamma_s[study_idx]

        if z_b:
            re_block_sum = pm.math.sum(z_s * gamma_b[block_idx], axis=1)
        if not z_b and b_intercept:
            re_block_sum = gamma_b[block_idx]

        # Overall expected value
        mu_obs = pm.Deterministic("mu_obs", fe_sum + re_study_sum + re_block_sum)

        # Variance assumed independent within and between studies and blocks
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Likelihood function
        y_like = pm.Normal(  # noqa: F841
            "y_like", mu=mu_obs, sigma=sigma, observed=y, dims="idx"
        )

    return model


def bii_abund_corr_re(
    y: np.array,
    x: np.array,
    z_s: np.array,
    z_b: np.array,
    coords: dict[str, np.array],
    study_idx: np.array,
    block_idx: np.array,
    block_to_study_idx: np.array,
) -> pm.Model:
    """
    This model is the translation of the BII abundance model to a hiearchical
    Bayesian setting. The original model is implemented in the R lme4 package.

    Args:
        y: The response variable vector for the model.
        x: The design matrix for the model.
        coords: The coordinates include lists for variables, groups, studies
            and blocks.
        group_idx: Observation indexed by the taxonomic group it belongs to.
        study_idx: Observations indexed by the study they are part of.
        block_idx: Observations indexed by the spatial block within a study.

    Returns:
        model: The PyMC model object that can be sampled from.
    """
    with pm.Model(coords=coords) as model:

        # Observed data that be changed later on for train-test runs
        y = pm.MutableData("y", y, dims="idx")
        x = pm.MutableData("x", x, dims=("idx", "x_var"))
        z_s = pm.MutableData("z_s", z_s, dims=("idx", "z_s_var"))
        z_b = pm.MutableData("z_b", z_b, dims=("idx", "z_b_var"))

        # Population level fixed effects
        # Priors independently sampled from univariate normal
        beta = pm.Normal("beta", mu=0, sigma=1, dims="x_var")

        # Study level random effects
        # Priors sampled from multivariate normal
        # LKJ hyperprior on covariance matrix
        z_s_dim = len(coords["z_s_var"])
        sd_s = pm.Exponential.dist(lam=1, shape=z_s_dim)
        chol_s, corr_s, stds_s = pm.LKJCholeskyCov(  # noqa: F841
            "chol_s", n=z_s_dim, eta=2, sd_dist=sd_s
        )
        print(chol_s.eval().shape)
        cov_s = pm.Deterministic(  # noqa: F841
            "cov_s", pm.math.dot(chol_s, chol_s.T), dims=("z_s_var", "z_s_var")
        )
        print(cov_s.eval().shape)
        # Independent, weakly regularizing prior on the mean vector
        mu_s = pm.Normal("mu_s", mu=0, sigma=1, dims="z_s_var")
        gamma_s = pm.MvNormal(
            "gamma_s",
            mu=mu_s,
            chol=chol_s,
            dims=("studies", "z_s_var"),
        )

        # Block level random effects
        z_b_dim = len(coords["z_b_var"])
        sd_b = pm.Exponential.dist(lam=1, shape=z_b_dim)
        chol_b, corr_b, stds_b = pm.LKJCholeskyCov(
            "chol_b", n=z_b_dim, eta=2, sd_dist=sd_b
        )
        gamma_b = pm.MvNormal(
            "gamma_b",
            mu=mu_s[study_idx],
            chol=chol_b,
            dims=("blocks", "z_b_var"),
        )

        # Expected abundance value based on fixed and random effects
        fe_sum = pm.math.sum(x * beta, axis=1)
        re_study_sum = pm.math.sum(z_s * gamma_s[study_idx], axis=1)
        re_block_sum = pm.math.sum(z_b * gamma_b[block_idx], axis=1)
        mu_obs = pm.Deterministic("mu_obs", fe_sum + re_study_sum + re_block_sum)

        # Variance assumed independent within and between studies and blocks
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Likelihood function
        y_like = pm.Normal(  # noqa: F841
            "y_like", mu=mu_obs, sigma=sigma, observed=y, dims="idx"
        )

    return model
