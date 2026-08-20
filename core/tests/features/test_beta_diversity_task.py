import numpy as np
import polars as pl

from core.features.beta_diversity_task import BetaDiversityTask


def environmental_distance_task() -> BetaDiversityTask:
    """Create a minimal task configured for environmental-distance tests."""
    task = BetaDiversityTask.__new__(BetaDiversityTask)
    task.environmental_dist_vars = {
        "alpha_equivalents": ["temperature"],
        "all": ["temperature", "precipitation"],
    }
    return task


def test_environmental_distance_uses_study_wide_ranges() -> None:
    """Numeric differences are scaled by all site values in the study."""
    df = pl.DataFrame(
        {
            "temperature_reference": [0.0, 10.0],
            "temperature": [10.0, 20.0],
            "precipitation_reference": [5.0, 5.0],
            "precipitation": [5.0, 5.0],
        }
    )

    result = environmental_distance_task().calculate_environmental_distance(df)

    np.testing.assert_allclose(
        result.get_column("Gower_distance_alpha_feat"),
        [0.5, 0.5],
    )
    np.testing.assert_allclose(
        result.get_column("Gower_distance_all_feat"),
        [0.25, 0.25],
    )


def test_environmental_distance_is_clipped_to_unit_interval() -> None:
    """Final Gower distances stay within the unit interval."""
    df = pl.DataFrame(
        {
            "temperature_reference": [0.0, 0.0],
            "temperature": [0.0, 1.0],
            "precipitation_reference": [2.0, 2.0],
            "precipitation": [2.0, 2.0],
        }
    )

    result = environmental_distance_task().calculate_environmental_distance(df)

    for column in ["Gower_distance_alpha_feat", "Gower_distance_all_feat"]:
        assert result.get_column(column).is_between(0.0, 1.0).all()
