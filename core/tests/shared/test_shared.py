import polars as pl

from core.tests.shared.validate_shared import check_duplicates, get_unique_value_count


def test_check_duplicates() -> None:
    """
    Test the check_duplicates function to ensure it correctly identifies
    duplicate rows in a DataFrame.
    """
    # Test case with duplicates
    df = pl.DataFrame({"col1": [1, 2, 2, 3], "col2": [4, 5, 5, 6]})
    assert check_duplicates(df) == 1

    # Test case without duplicates
    df_no_duplicates = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    assert check_duplicates(df_no_duplicates) == 0


def test_get_unique_value_count() -> None:
    """
    Test the get_unique_value_count function to ensure it correctly counts
    the unique values in a specified column.
    """
    # Test case with duplicates in the column
    df = pl.DataFrame({"col1": [1, 1, 2, 3]})
    assert get_unique_value_count(df, "col1") == 3
