import numpy as np

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
    """
    Calculates and logs the conditional R^2 and accuracy (1 - sMAPE) scores on
    the training set.

    Args:
        y_true: The true observed values for the response variable.
        y_pred: The corresponding model predictions.
        group_idx: The hierarchical group identity for each observation.
        group_code_map: A mapping of group names to group codes.
    """

    cond_r2 = np.var(y_pred) / np.var(y_true)
    smape = calculate_smape(y_true, y_pred)

    logger.info(f"Overall conditional R^2 score: {round(cond_r2, 3)}")
    logger.info(f"Overall accuracy (1 - sMAPE): {round(smape, 2)}%")

    if group_idx:
        for name, code in group_code_map.items():
            idx = group_idx == code
            cond_r2 = np.var(y_pred[idx]) / np.var(y_true[idx])
            smape = calculate_smape(y_true[idx], y_pred[idx])

            logger.info(f"Conditional R^2 score for {name}: {round(cond_r2, 3)}")
            logger.info(f"Accuracy (1 - sMAPE) for {name}: {round(smape, 2)}%")
