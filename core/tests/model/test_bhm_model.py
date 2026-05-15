import logging

from core.model.bhm_model import BayesianHierarchicalModel


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
