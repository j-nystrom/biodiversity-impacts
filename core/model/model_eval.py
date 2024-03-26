import numpy as np
from sklearn.metrics import r2_score

from core.utils.general_utils import create_logger

logger = create_logger(__name__)


def calculate_smape(y_true: np.array, y_pred: np.array, divisor: int = 1) -> float:
    """
    Function to calculate prediction accuracy, defined as 1 - symmetric mean
    absolute percentage error. (sMAPE). Can either use the "classic" version
    that produces outputs on a scale 0-200% (by using 2 as the divisor) or a
    version on the scale 0-100% (by using 1 instead). More details:
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error

    Args:
        y_true (np.array): The observed values.
        y_pred (np.array): The corresponding model predictions.
        divisor (int): Number to divide the sum of the prediction and actual
            value with. Defaults to 1.

    Returns:
        The sMAPE value for these two arrays.
    """
    # Check that the divisor argument is valid
    assert divisor in [1, 2], "Divisor argument must be 1 or 2"

    # Perform the accuracy calculation of 1 - sMAPE
    smape = 100 * (
        1
        - np.mean(
            np.abs(y_pred - y_true) / ((np.abs(y_pred) + np.abs(y_true)) / divisor)
        )
    )
    return smape


def calculate_performance_metrics(
    y_true: np.array,
    y_pred: np.array,
    group_idx: np.array,
    group_code_map: dict[str, int],
) -> None:
    """Add description in docstring."""
    # Do back-transformation

    r2 = r2_score(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)

    logger.info(f"Overall R^2 score: {round(r2, 3)}")
    logger.info(f"Overall accuracy (1 - sMAPE): {round(smape, 2)}%")

    for name, code in group_code_map.items():
        idx = group_idx == code
        r2 = r2_score(y_true[idx], y_pred[idx])
        smape = calculate_smape(y_true[idx], y_pred[idx])

        logger.info(f"R^2 score for {name}: {round(r2, 3)}")
        logger.info(f"Accuracy (1 - sMAPE) for {name}: {round(smape, 2)}%")
