from typing import Any

import pymc as pm
import pytensor.tensor as pt


class BiomeRealmEcoSlopeModel:
    """
    Contains the final version of the hierarchical model used in the thesis.
    The model is a three-level hierarchical model with biome, realm, and
    ecoregion levels. The full set of parameters is estimated for each level.
    """

    def __init__(self) -> None:
        pass

    def model(self, model_data: dict[str, Any]) -> pm.Model:
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
            mu_b_eco = pm.Deterministic("mu_b_eco", beta_realm[eco_to_realm_idx])
            sigma_a_eco = pm.HalfNormal("sigma_a_eco", sigma=0.1)
            sigma_b_eco = pm.HalfNormal("sigma_b_eco", sigma=0.1)

            eco_offset_1 = pm.Normal(
                "eco_offset_1", mu=0, sigma=1, dims="biome_realm_eco"
            )
            eco_offset_2 = pm.Normal(
                "eco_offset_2", mu=0, sigma=1, dims=("biome_realm_eco", "x_vars")
            )

            alpha_eco = pm.Deterministic(
                "alpha_eco", mu_a_eco + eco_offset_1 * sigma_a_eco
            )
            beta_eco = pm.Deterministic(
                "beta_eco", mu_b_eco + eco_offset_2 * sigma_b_eco
            )

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
