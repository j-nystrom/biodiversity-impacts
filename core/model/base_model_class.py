import logging
from typing import Any


class BaseModel:
    def __init__(
        self,
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        site_name_to_idx: dict[str, int],
        logger: logging.Logger,
        mode: str,
    ) -> None:
        # Model settings
        self.model_settings = model_settings
        self.model_vars = model_vars
        self.site_name_to_idx = site_name_to_idx
        self.idx_to_site = {idx: name for name, idx in self.site_name_to_idx.items()}
        self.logger = logger
        self.mode = mode

        # Model covariates
        self.response_var = model_vars["response_var"]
        self.response_var_transform = model_vars["response_var_transform"]
        self.categorical_vars = model_vars["categorical_vars"]
        self.continuous_vars = model_vars["continuous_vars"]
        self.interaction_terms = model_vars["interaction_terms"]
