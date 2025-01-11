import polars as pl

from core.tests.shared.validate_shared import (
    check_no_nans,
    check_non_negative,
    check_within_bounds,
)

# AlphaDiversityTask --------------------------------------


def validate_alpha_diversity_calculations(df: pl.DataFrame, metric_name: str) -> None:
    """
    Validate calculated alpha diversity metrics:
        - No NaN values in the calculated columns.
        - Non-negative values in calculated columns.
        - Scaled columns ("Max_scaled_...") are between 0 and 1.

    Args:
        - df: Input DataFrame containing the alpha diversity metrics.
        - metric_name: Name of the metric to validate (e.g., "Shannon").

    Raises:
        - ValueError: If any validation fails.
    """
    # Columns to validate
    base_column = metric_name
    scaled_column = f"Scaled_{metric_name}"
    study_max_column = f"Study_max_{metric_name}"

    # Perform validations
    check_no_nans(df, [base_column, scaled_column, study_max_column])
    check_non_negative(df, [base_column, study_max_column])
    check_within_bounds(df, [scaled_column], lower=0.0, upper=1.0)
