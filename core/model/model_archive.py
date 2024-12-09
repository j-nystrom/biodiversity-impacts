from typing import Any

import pymc as pm
import pytensor.tensor as pt

# --- Archive of various model structures used during experimentation ---


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
