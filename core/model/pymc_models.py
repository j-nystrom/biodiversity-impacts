from typing import Any

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.math import clip, invlogit, logit


class GeneralHierarchicalModel:
    """
    Build PyMC models with a hierarchical structure with up to three levels.
    Supports Gaussian and Beta likelihoods with varying intercepts and slopes.
    Control variables for study and block random effects are included during
    training to account for study design variation.
    """

    def __init__(self, settings: dict[str, Any], epsilon: float) -> None:
        """
        Store settings after validating required configuration fields.

        Attributes:
            - settings: Dictionary of model configuration settings.
            - eps: Small value for clipping values, e.g. in Beta likelihoods.
            - hierarchical_levels: Number of hierarchical levels (1, 2, or 3).
            - likelihood: Likelihood distribution to use ('gaussian' or 'beta').
            - priors: Dictionary of prior settings for the selected likelihood.
        """
        validate_model_settings(settings)
        self.settings = settings
        self.eps = epsilon
        self.hierarchical_levels = self.settings["hierarchical_levels"]
        self.likelihood = self.settings["likelihood"]
        self.priors = self.settings["priors"][self.likelihood]

    def build_training_model(self, model_data: dict[str, Any]) -> pm.Model:
        """
        Build the training model with hierarchical priors and likelihood.

        Args:
            - model_data: Dictionary of arrays and coords from data preparation.

        Returns:
            - model: PyMC model instance for training.
        """
        # Get key settings from configuration

        with pm.Model(coords=model_data["coords"]) as model:
            # Set up data nodes and hierarchical indexing
            self.add_data_nodes(model_data, self.settings)

            # Define hierarchical priors for the intercepts and slopes
            # Returns the most granular level, which is used in the likelihood
            alpha, beta = self.define_hierarchical_normal_priors(
                model_data,
                priors=self.priors,
                hierarchical_levels=self.hierarchical_levels,
                varying_slope_level=self.settings["varying_slope_level"],
            )

            # Get priors for study and block control variables, used in training
            # but not for out-of-sample prediction
            gamma_block = self.define_control_variable_priors(
                model_data,
                priors=self.priors,
            )

            # Compute linear conditional mean and intercept
            level_idx = model[f"level_{self.hierarchical_levels}_idx"]
            y_cond_linear = (
                alpha[level_idx]
                + pt.sum(model["x_obs"] * beta[level_idx], axis=1)
                + gamma_block[model["block_idx"]]
            )
            y_intercept_linear = alpha[level_idx] + gamma_block[model["block_idx"]]

            # Define data variance prior
            sigma_y = self.define_data_variance_prior(
                likelihood=self.likelihood,
                y_cond_linear=y_cond_linear,
                priors=self.priors,
            )

            # Add likelihood function and other model outputs
            add_likelihood_outputs(
                likelihood=self.likelihood,
                y_cond_linear=y_cond_linear,
                y_intercept_linear=y_intercept_linear,
                mode="train",
                eps=self.eps,
                observed=model["y_obs"],
                sigma_y=sigma_y,
            )

            return model

    def build_prediction_model(self, model_data: dict[str, Any]) -> pm.Model:
        """
        Build a prediction model that reuses the trained hierarchical
        parameters, but injects new data.

        Args:
            - model_data: Dictionary of arrays and coords for prediction data.

        Returns:
            - pred_model: PyMC model for posterior predictive sampling.
        """
        with pm.Model(coords=model_data["coords"]) as pred_model:
            # Set up data nodes and hierarchical indexing
            self.add_data_nodes(model_data, self.settings)

            # Get the most granular hierarchical level
            level = self.settings["hierarchical_levels"]

            # Get posterior samples from the training model for this level
            alpha = pm.Flat(f"alpha_{level}", dims=f"level_{level}_values")
            beta = pm.Flat(f"beta_{level}", dims=(f"level_{level}_values", "x_vars"))
            level_idx = model_data[f"level_{level}_idx"]

            # Linear conditional mean and intercept
            y_cond_linear = alpha[level_idx] + pt.sum(
                pred_model["x_obs"] * beta[level_idx], axis=1
            )
            y_intercept_linear = alpha[level_idx]

            # Data variance placeholders, depending on likelihood
            sigma_y = pm.Flat("sigma_y") if self.likelihood == "gaussian" else None
            sigma_raw = pm.Flat("sigma_raw") if self.likelihood == "beta" else None

            add_likelihood_outputs(
                likelihood=self.likelihood,
                y_cond_linear=y_cond_linear,
                y_intercept_linear=y_intercept_linear,
                mode="test",
                eps=self.eps,
                sigma_y=sigma_y,
                sigma_raw=sigma_raw,
            )

            return pred_model

    def add_data_nodes(self, model_data: dict, settings: dict[str, Any]) -> None:
        """Add input data nodes to the PyMC model."""
        pm.Data("x_obs", model_data["x_obs"], dims=("idx", "x_vars"))
        pm.Data("site_idx", model_data["site_idx"], dims="idx")
        pm.Data("taxon_idx", model_data["taxon_idx"], dims="idx")
        if self.likelihood == "beta":
            y_obs = np.clip(model_data["y_obs"], self.eps, 1 - self.eps)
        else:
            y_obs = model_data["y_obs"]
        pm.Data("y_obs", y_obs, dims="idx")

        # Random effects for studies and blocks
        pm.Data("study_idx", model_data["study_idx"], dims="idx")
        pm.Data("block_idx", model_data["block_idx"], dims="idx")
        pm.Data(
            "block_to_study_idx", model_data["block_to_study_idx"], dims="block_names"
        )

        # Process each hierarchical level and add relevant indices and mappings
        hierarchical_levels = settings["hierarchical_levels"]
        if hierarchical_levels >= 1:
            pm.Data("level_1_idx", model_data["level_1_idx"], dims="idx")
        if hierarchical_levels >= 2:
            pm.Data("level_2_idx", model_data["level_2_idx"], dims="idx")
            pm.Data(
                "level_2_to_level_1_idx",
                model_data["level_2_to_level_1_idx"],
                dims="level_2_values",
            )
        if hierarchical_levels == 3:
            pm.Data("level_3_idx", model_data["level_3_idx"], dims="idx")
            pm.Data(
                "level_3_to_level_2_idx",
                model_data["level_3_to_level_2_idx"],
                dims="level_3_values",
            )

    def define_hierarchical_normal_priors(
        self,
        model_data: dict,
        priors: dict,
        hierarchical_levels: int,
        varying_slope_level: int,
    ) -> tuple[pm.Deterministic, pm.Deterministic]:
        """
        Define hierarchical normal priors for varying intercepts and slopes, up
        to three levels. Slopes are only modeled down to varying_slope_level.
        """
        priors_config = self.settings["priors"]
        use_group_size_shrinkage = bool(priors_config["group_size_shrinkage"])
        shrinkage_scaling = priors_config["group_size_shrinkage_scaling"]

        def _group_prior_sd_beta(level: str) -> float:
            level_key = f"group_prior_sd_beta_level_{level}"
            if level_key not in priors:
                raise ValueError(
                    "Missing slope prior for hierarchical level: " f"{level_key}."
                )
            return priors[level_key]

        def _make_level(
            level: str,
            parent_alpha: Any,
            parent_beta: Any,
            dims: str,
            make_slopes: bool,
            n_studies: np.ndarray,
        ) -> tuple[pm.Deterministic, pm.Deterministic]:
            # Global scale parameters for the level
            tau_alpha = pm.HalfNormal(
                f"tau_alpha_{level}", sigma=priors["group_prior_sd_alpha"]
            )

            # Offsets for non-centered parameterization
            offset_alpha = pm.Normal(f"offset_alpha_{level}", mu=0, sigma=1, dims=dims)

            # Final hierarchical priors for intercepts
            if use_group_size_shrinkage:
                # Optional dynamic shrinkage based on group size.
                # One scale is computed and reused for alpha and beta.
                n_studies_values = n_studies.astype(float)
                if (not np.isfinite(n_studies_values).all()) or (
                    n_studies_values <= 0
                ).any():
                    raise ValueError(
                        f"Invalid n_studies for level {level}. "
                        "All values must be finite and > 0 when "
                        "group_size_shrinkage is enabled."
                    )
                if shrinkage_scaling == "sqrt":
                    raw_scale = np.sqrt(n_studies_values) - 1.0
                else:  # "log"
                    raw_scale = np.log(n_studies_values) - 1.0
                scale = np.maximum(raw_scale, self.eps)

                sigma_alpha = pm.Deterministic(
                    f"sigma_alpha_{level}",
                    tau_alpha * scale,
                    dims=dims,
                )
                alpha = pm.Deterministic(
                    f"alpha_{level}", parent_alpha + sigma_alpha * offset_alpha
                )
            else:
                alpha = pm.Deterministic(
                    f"alpha_{level}", parent_alpha + tau_alpha * offset_alpha
                )

            # Final hierarchical priors for slopes.
            # For levels deeper than varying_slope_level, slopes are inherited
            # from the parent level (no new slope priors at this level).
            if make_slopes:
                tau_beta = pm.HalfNormal(
                    f"tau_beta_{level}", sigma=_group_prior_sd_beta(level)
                )
                offset_beta = pm.Normal(
                    f"offset_beta_{level}", mu=0, sigma=1, dims=(dims, "x_vars")
                )
                if use_group_size_shrinkage:
                    sigma_beta = pm.Deterministic(
                        f"sigma_beta_{level}",
                        tau_beta * scale,
                        dims=dims,
                    )
                    beta = pm.Deterministic(
                        f"beta_{level}", parent_beta + sigma_beta[:, None] * offset_beta
                    )
                else:
                    beta = pm.Deterministic(
                        f"beta_{level}", parent_beta + tau_beta * offset_beta
                    )
            else:
                if parent_beta.ndim == 1:
                    n_groups = offset_alpha.shape[0]
                    n_x = parent_beta.shape[-1]
                    beta = pm.Deterministic(
                        f"beta_{level}",
                        pt.broadcast_to(parent_beta, (n_groups, n_x)),
                    )
                else:
                    beta = pm.Deterministic(f"beta_{level}", parent_beta)

            return alpha, beta

        # Global hyperpriors for intercept and slopes, depending on likelihood
        if self.likelihood == "gaussian":
            mu_alpha_mean = priors["alpha_hyper_mean"]
        elif self.likelihood == "beta":
            mu_alpha_mean = logit(priors["alpha_hyper_mean_mu"])

        mu_alpha = pm.Normal(
            "mu_alpha", mu=mu_alpha_mean, sigma=priors["hyperprior_sd_alpha"]
        )
        mu_beta = pm.Normal(
            "mu_beta", mu=0, sigma=priors["hyperprior_sd_beta"], dims="x_vars"
        )

        # Level 1 priors
        alpha, beta = _make_level(
            level="1",
            parent_alpha=mu_alpha,
            parent_beta=mu_beta,
            dims="level_1_values",
            make_slopes=1 <= varying_slope_level,
            n_studies=model_data["level_1_n_studies"],
        )

        # Level 2 priors, if applicable
        if hierarchical_levels >= 2:
            idx_2_to_1 = model_data["level_2_to_level_1_idx"]
            alpha, beta = _make_level(
                level="2",
                parent_alpha=alpha[idx_2_to_1],
                parent_beta=beta[idx_2_to_1],
                dims="level_2_values",
                make_slopes=2 <= varying_slope_level,
                n_studies=model_data["level_2_n_studies"],
            )

        # Level 3 priors, if applicable
        if hierarchical_levels == 3:
            idx_3_to_2 = model_data["level_3_to_level_2_idx"]
            alpha, beta = _make_level(
                level="3",
                parent_alpha=alpha[idx_3_to_2],
                parent_beta=beta[idx_3_to_2],
                dims="level_3_values",
                make_slopes=3 <= varying_slope_level,
                n_studies=model_data["level_3_n_studies"],
            )

        return alpha, beta

    def define_control_variable_priors(
        self,
        model_data: dict[str, Any],
        priors: dict[str, Any],
    ) -> pt.TensorVariable:
        """
        Define random effects for study and block intercepts. These are used as
        control variables during training.

        Args:
            - model_data: Dictionary of arrays and coords from the data task.
            - priors: Prior settings for random intercepts.
        """
        # Priors on study and block IDs
        mu_gamma = pm.Normal(
            "mu_gamma",
            mu=0,
            sigma=priors["random_intercept_sd"],
        )

        # Study level priors
        sigma_gamma_study = pm.HalfNormal(
            "sigma_gamma_study", sigma=priors["random_intercept_sd"]
        )
        offset_gamma_study = pm.Normal(
            "offset_gamma_study", mu=0, sigma=1, dims="study_names"
        )
        gamma_study = pm.Deterministic(
            "gamma_study", mu_gamma + sigma_gamma_study * offset_gamma_study
        )

        # Block level priors
        block_to_study_idx = model_data["block_to_study_idx"]
        sigma_gamma_block = pm.HalfNormal(
            "sigma_gamma_block", sigma=priors["random_intercept_sd"]
        )
        offset_gamma_block = pm.Normal(
            "offset_gamma_block", mu=0, sigma=1, dims="block_names"
        )
        gamma_block = pm.Deterministic(
            "gamma_block",
            gamma_study[block_to_study_idx] + sigma_gamma_block * offset_gamma_block,
        )

        return gamma_block

    def define_data_variance_prior(
        self,
        likelihood: str,
        y_cond_linear: pt.TensorVariable,
        priors: dict[str, Any],
    ) -> pt.TensorVariable:
        """
        Define the observation noise prior for the given likelihood.

        Args:
            - likelihood: Either "gaussian" or "beta".
            - y_cond_linear: Linear conditional mean from the model.
            - priors: Likelihood-specific prior settings.
        """
        if likelihood == "gaussian":
            sigma_y = pm.HalfNormal("sigma_y", sigma=priors["sigma_y_sd"])

        elif likelihood == "beta":
            sigma_raw = pm.Beta(
                "sigma_raw",
                alpha=priors["beta_likelihood"]["alpha"],
                beta=priors["beta_likelihood"]["beta"],
            )
            y_cond = clip(invlogit(y_cond_linear), self.eps, 1 - self.eps)
            sigma_y = pm.Deterministic(
                "sigma_y", sigma_raw * pt.sqrt(y_cond * (1 - y_cond))
            )

        return sigma_y


def add_likelihood_outputs(
    likelihood: str,
    y_cond_linear: pt.TensorVariable,
    y_intercept_linear: pt.TensorVariable,
    mode: str,
    eps: float,
    observed: pt.TensorVariable | None = None,
    sigma_y: pt.TensorVariable | None = None,
    sigma_raw: pt.TensorVariable | None = None,
) -> None:
    """
    Add shared model outputs: conditional mean (y_cond), likelihood predictions,
    and intercepts for training or test models.
    """
    pred_name = "y_like" if mode == "train" else "y_pred"

    # Gaussian likelihood case
    if likelihood == "gaussian":
        pm.Deterministic("y_cond", y_cond_linear, dims="idx")
        pm.Deterministic("y_intercept", y_intercept_linear, dims="idx")

        # For posterior predictive / predictions, observed is not provided
        if observed is None:
            pm.Normal(pred_name, mu=y_cond_linear, sigma=sigma_y, dims="idx")
        # For training model with observed data
        else:
            pm.Normal(
                pred_name,
                mu=y_cond_linear,
                sigma=sigma_y,
                observed=observed,
                dims="idx",
            )

    # Beta likelihood case
    elif likelihood == "beta":
        y_cond = clip(invlogit(y_cond_linear), eps, 1 - eps)
        y_intercept = clip(invlogit(y_intercept_linear), eps, 1 - eps)
        pm.Deterministic("y_cond", y_cond, dims="idx")
        pm.Deterministic("y_intercept", y_intercept, dims="idx")

        # For posterior predictive / predictions, compute sigma_y from sigma_raw
        if sigma_y is None:
            sigma_y = pm.Deterministic(
                "sigma_y", sigma_raw * pt.sqrt(y_cond * (1 - y_cond))
            )

        # For posterior predictive / predictions, observed is not provided
        if observed is None:
            pm.Beta(pred_name, mu=y_cond, sigma=sigma_y, dims="idx")

        # For training model with observed data
        else:
            pm.Beta(pred_name, mu=y_cond, sigma=sigma_y, observed=observed, dims="idx")


def validate_model_settings(settings: dict[str, Any]) -> None:
    """Validate and extract configuration settings for the model."""
    # Required top-level fields
    required_fields = {
        "likelihood",
        "hierarchical_levels",
        "varying_slope_level",
        "priors",
        "hierarchy",
    }

    for field in required_fields:
        if field not in settings:
            raise ValueError(f"Missing required setting: '{field}'")

    #  Likelihood structure
    allowed_distributions = {"gaussian", "beta"}
    distribution = settings["likelihood"]

    if distribution not in allowed_distributions:
        raise ValueError(f"Unsupported likelihood. Allowed: {allowed_distributions}")

    # Hierarchy structure
    hierarchy = settings["hierarchy"]
    for level in ["level_1", "level_2", "level_3"]:
        if level not in hierarchy:
            raise ValueError(f"Missing hierarchy key: {level}")
        if not isinstance(hierarchy[level], list):
            raise ValueError(f"Hierarchy '{level}' must be a list.")
        if not all(isinstance(col, str) for col in hierarchy[level]):
            raise ValueError(f"All entries in hierarchy '{level}' must be strings.")

    # Value range checks
    if not (1 <= settings["hierarchical_levels"] <= 3):
        raise ValueError("hierarchical_levels must be 1, 2, or 3")

    if not (1 <= settings["varying_slope_level"] <= settings["hierarchical_levels"]):
        raise ValueError("varying_slope_level must be <= hierarchical_levels")

    priors_cfg = settings["priors"]
    if "group_size_shrinkage" not in priors_cfg:
        raise ValueError("Missing required setting: priors.group_size_shrinkage")
    if not isinstance(priors_cfg["group_size_shrinkage"], bool):
        raise ValueError("priors.group_size_shrinkage must be a boolean.")

    if "group_size_shrinkage_scaling" not in priors_cfg:
        raise ValueError(
            "Missing required setting: priors.group_size_shrinkage_scaling"
        )
    shrinkage_scaling = priors_cfg["group_size_shrinkage_scaling"]
    if shrinkage_scaling not in {"sqrt", "log"}:
        raise ValueError("priors.group_size_shrinkage_scaling must be 'sqrt' or 'log'.")


def rolled_up_prediction_model(
    model_data: dict[str, Any],
    trace: az.InferenceData,
    settings: dict[str, Any],
    mode: str,
    epsilon: float,
) -> pm.Model:
    """
    Build a prediction model that applies rolled-up fallback logic.

    For each observation, the model uses parameters from the most specific
    hierarchical level available (level 3/2/1). If the observation belongs to
    a group that does not meet the study threshold, it falls back to the parent
    level, and ultimately to population-level parameters.

    This model is used with pm.sample_posterior_predictive(trace, ...) to draw
    conditional means and posterior predictive samples for new data.
    """
    hierarchical_levels = settings["hierarchical_levels"]
    likelihood = settings["likelihood"]
    eps = epsilon

    def _update_coords_from_trace(
        existing_coords: dict[str, Any],
        trace: az.InferenceData,
    ) -> dict[str, Any]:
        coords = dict(existing_coords)  # start from model_data's coords
        for v in trace.posterior.variables:
            dims = trace.posterior[v].dims
            shape = trace.posterior[v].shape
            for dim, size in zip(dims, shape):
                if dim not in ("chain", "draw") and dim not in coords:
                    coords[dim] = np.arange(size)
        return coords

    # Update coords from trace, so all dims needed for Flat variables are present
    coords = _update_coords_from_trace(model_data["coords"], trace)

    with pm.Model(coords=coords) as prediction_model:
        # Data nodes
        x_obs = pm.Data("x_obs", model_data["x_obs"], dims=("idx", "x_vars"))
        pm.Data("site_idx", model_data["site_idx"], dims="idx")
        pm.Data("taxon_idx", model_data["taxon_idx"], dims="idx")
        if likelihood == "beta":
            y_obs = np.clip(model_data["y_obs"], eps, 1 - eps)
        else:
            y_obs = model_data["y_obs"]
        pm.Data("y_obs", y_obs, dims="idx")

        level_assignment = pm.Data(
            "level_assignment", model_data["level_assignment"], dims="idx"
        )
        pm.Data("level_1_idx", model_data["level_1_idx"], dims="idx")
        if hierarchical_levels >= 2:
            pm.Data("level_2_idx", model_data["level_2_idx"], dims="idx")
        if hierarchical_levels == 3:
            pm.Data("level_3_idx", model_data["level_3_idx"], dims="idx")

        # Load sampled posterior variables with matching dims
        mu_alpha = pm.Flat("mu_alpha")
        mu_beta = pm.Flat("mu_beta", dims="x_vars")
        alpha_1 = pm.Flat("alpha_1", dims="alpha_1_dim_0")
        beta_1 = pm.Flat("beta_1", dims=("beta_1_dim_0", "beta_1_dim_1"))
        if hierarchical_levels >= 2:
            alpha_2 = pm.Flat("alpha_2", dims="alpha_2_dim_0")
            beta_2 = pm.Flat("beta_2", dims=("beta_2_dim_0", "beta_2_dim_1"))
        if hierarchical_levels == 3:
            alpha_3 = pm.Flat("alpha_3", dims="alpha_3_dim_0")
            beta_3 = pm.Flat("beta_3", dims=("beta_3_dim_0", "beta_3_dim_1"))

        # Initialize linear predictors
        y_cond_linear = pt.zeros_like(model_data["y_obs"])
        y_intercept_linear = pt.zeros_like(model_data["y_obs"])

        def _set_level_prediction(
            y_cond_linear: pt.TensorVariable,
            y_intercept_linear: pt.TensorVariable,
            mask: pt.TensorVariable,
            x_obs: pt.TensorVariable,
            group_idx: Any,
            alpha: Any,
            beta: Any,
        ) -> tuple[pt.TensorVariable, pt.TensorVariable]:
            obs_idx = pt.nonzero(mask)[0]
            groups = pt.take(group_idx, obs_idx)
            alpha_sel = pt.take(alpha, groups)
            beta_sel = pt.take(beta, groups, axis=0)  # Symbolic batch gather
            x_sel = pt.take(x_obs, obs_idx, axis=0)
            y_pred = alpha_sel + pt.sum(x_sel * beta_sel, axis=1)
            y_cond_linear = pt.set_subtensor(y_cond_linear[obs_idx], y_pred)
            y_intercept_linear = pt.set_subtensor(
                y_intercept_linear[obs_idx], alpha_sel
            )

            return y_cond_linear, y_intercept_linear

        # Population-level fallback (level 0)
        mask_0 = pt.eq(level_assignment, 0)
        obs_idx_0 = pt.nonzero(mask_0)[0]
        x_sel_0 = pt.take(x_obs, obs_idx_0, axis=0)
        y_pred_0 = mu_alpha + pt.sum(x_sel_0 * mu_beta, axis=1)
        y_cond_linear = pt.set_subtensor(y_cond_linear[obs_idx_0], y_pred_0)
        y_intercept_linear = pt.set_subtensor(y_intercept_linear[obs_idx_0], mu_alpha)

        # Level 1
        mask_1 = pt.eq(level_assignment, 1)
        y_cond_linear, y_intercept_linear = _set_level_prediction(
            y_cond_linear,
            y_intercept_linear,
            mask_1,
            x_obs,
            model_data["level_1_idx"],
            alpha_1,
            beta_1,
        )

        # Level 2
        if hierarchical_levels >= 2:
            mask_2 = pt.eq(level_assignment, 2)
            y_cond_linear, y_intercept_linear = _set_level_prediction(
                y_cond_linear,
                y_intercept_linear,
                mask_2,
                x_obs,
                model_data["level_2_idx"],
                alpha_2,
                beta_2,
            )

        # Level 3
        if hierarchical_levels == 3:
            mask_3 = pt.eq(level_assignment, 3)
            y_cond_linear, y_intercept_linear = _set_level_prediction(
                y_cond_linear,
                y_intercept_linear,
                mask_3,
                x_obs,
                model_data["level_3_idx"],
                alpha_3,
                beta_3,
            )

        # Data variance placeholders, depending on likelihood
        sigma_y = pm.Flat("sigma_y") if likelihood == "gaussian" else None
        sigma_raw = pm.Flat("sigma_raw") if likelihood == "beta" else None

        add_likelihood_outputs(
            likelihood=likelihood,
            y_cond_linear=y_cond_linear,
            y_intercept_linear=y_intercept_linear,
            mode=mode,
            eps=eps,
            sigma_y=sigma_y,
            sigma_raw=sigma_raw,
        )

    return prediction_model
