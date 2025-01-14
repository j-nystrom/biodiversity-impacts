import logging
from typing import Any


class BaseModel:
    def __init__(
        self,
        mode: str,
        model_settings: dict[str, Any],
        model_vars: dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        # Model settings
        self.mode = mode
        self.model_settings = model_settings
        self.model_vars = model_vars
        self.logger = logger

        # Model covariates
        self.response_var = model_vars["response_var"]
        self.response_var_transform = model_vars["response_var_transform"]
        self.categorical_vars = model_vars["categorical_vars"]
        self.continuous_vars = model_vars["continuous_vars"]
        self.interaction_terms = model_vars["interaction_terms"]
