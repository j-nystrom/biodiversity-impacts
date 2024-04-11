import numpy as np
import pymc as pm


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
