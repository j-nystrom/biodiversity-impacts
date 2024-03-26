import numpy as np
import pymc as pm


def bii_abund_baseline(
    x: np.array,
    y: np.array,
    coords: dict[str, np.array],
    group_idx: np.array,
    study_idx: np.array,
    block_idx: np.array,
) -> pm.Model:
    """
    This model is the translation of the BII abundance model to a hiearchical
    Bayesian setting. The original model is implemented in the R lme4 package.

    Args:
        x: The design matrix for the model.
        y: The response variable vector for the model.
        coords: The coordinates include lists for variables, groups, studies
            and blocks.
        group_idx: Observation indexed by the taxonomic group it belongs to.
        study_idx: Observations indexed by the study they are part of.
        block_idx: Observations indexed by the spatial block within a study.

    Returns:
        model: The PyMC model object that can be sampled from.
    """
    with pm.Model(coords=coords) as model:

        # Observed data that be changed later on for train-test runs (hence MutableData)
        # w_study are study-level mean population and road density
        x_obs = pm.MutableData("x_obs", x, dims=("idx", "covariates"))
        y_obs = pm.MutableData("y_obs", y, dims="idx")

        # Hyperpriors for the mean and variance of the covariate parameters
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, dims="covariates")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, dims="covariates")

        # Covariate parameter priors with non-centered parameterization
        beta_offset = pm.Normal(
            "beta_offset", mu=0, sigma=1, dims=("groups", "covariates")
        )
        beta = pm.Deterministic("beta", mu_beta + beta_offset * sigma_beta)

        # Block and study level random effects (intercept)
        nu_block = pm.Normal("nu_block", mu=0, sigma=1, dims="blocks")
        nu_study = pm.Normal("nu_study", mu=0, sigma=1, dims="studies")

        # Expected value based on parameters and covariates
        x_sum = pm.math.sum(x_obs * beta[group_idx], axis=1)
        random_intercepts = nu_block[block_idx] + nu_study[study_idx]
        mu_obs = pm.Deterministic("mu_obs", x_sum + random_intercepts)
        # Variance, assumed to be independent between groups, studies, blocks
        phi = pm.HalfNormal("phi_obs", sigma=1)

        # Likelihood function
        y_like = pm.Normal(  # noqa: F841
            "y_like", mu=mu_obs, sigma=phi, observed=y_obs, dims="idx"
        )

    return model


def abund_study_controls(
    x: np.array,
    y: np.array,
    w_study: np.array,
    coords: dict[str, np.array],
    group_idx: np.array,
    study_idx: np.array,
    block_idx: np.array,
) -> pm.Model:
    """Add docstring."""
    with pm.Model(coords=coords) as model:

        # Observed data that be changed later on for train-test runs (hence MutableData)
        # w_study are study-level mean population and road density
        x_obs = pm.MutableData("x_obs", x, dims=("idx", "covariates"))
        y_obs = pm.MutableData("y_obs", y, dims="idx")
        w_study = pm.MutableData("w_study", w_study, dims=("idx", "controls"))

        # Hyperpriors for the mean and variance of the covariate parameters
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=1, dims="covariates")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1, dims="covariates")

        # Covariate parameter priors with non-centered parameterization
        beta_offset = pm.Normal(
            "beta_offset", mu=0, sigma=1, dims=("groups", "covariates")
        )
        beta = pm.Deterministic("beta", mu_beta + beta_offset * sigma_beta)

        # Block and study level random effects (intercept)
        nu_block = pm.Normal("nu_block", mu=0, sigma=1, dims="blocks")
        nu_study = pm.Normal("nu_study", mu=0, sigma=1, dims="studies")
        # Study-level control variables
        gamma = pm.Normal("gamma", mu=0, sigma=1, dims=("studies", "controls"))

        # Expected value based on parameters and covariates
        x_sum = pm.math.sum(x_obs * beta[group_idx], axis=1)
        if w_study.ndim == 1:
            w_sum = w_study * gamma[study_idx]
        else:
            w_sum = pm.math.sum(w_study * gamma[study_idx], axis=1)
        random_intercepts = nu_block[block_idx] + nu_study[study_idx]

        mu_obs = pm.Deterministic("mu_obs", x_sum + w_sum + random_intercepts)
        # Variance term, assumed to be independent between groups, studies and blocks
        # This should probably be refined
        phi = pm.HalfNormal("phi_obs", sigma=1)

        # Likelihood function
        y_like = pm.Normal(  # noqa: F841
            "y_like", mu=mu_obs, sigma=phi, observed=y_obs, dims="idx"
        )

    return model
