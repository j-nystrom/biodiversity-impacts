import polars as pl
from polars.testing import assert_series_equal

from core.features.generate_features_task import GenerateFeaturesTask


def test_rescale_continuous_covariates_does_not_rescale_temperature() -> None:
    """Temperature stays in input units while precipitation is rescaled."""
    task = GenerateFeaturesTask.__new__(GenerateFeaturesTask)
    df = pl.DataFrame(
        {
            "Annual_mean_temp_1km": [15.0, None],
            "Annual_precip_1km": [1000.0, None],
        }
    )

    result = task.rescale_continuous_covariates(
        df,
        variables=["Annual_mean_temp_1km", "Annual_precip_1km"],
    )

    assert_series_equal(
        result.get_column("Annual_mean_temp_1km"),
        pl.Series("Annual_mean_temp_1km", [15.0, None]),
    )
    assert_series_equal(
        result.get_column("Annual_precip_1km"),
        pl.Series("Annual_precip_1km", [100.0, None]),
    )
