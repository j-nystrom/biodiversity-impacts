import os
from typing import Any, Union

import arviz as az
import geopandas as gpd
import pandas as pd
import polars as pl
import pymc as pm
from pymer4.models import Lmer

# Input and Output Validation -------------------------------------------------


def validate_input_files(file_paths: list[str]) -> None:
    """Validate that input files in the given list exist and are non-empty."""
    for file_path in file_paths:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Input file is empty: {file_path}")


def validate_output_files(
    file_paths: list[str],
    files: list[Union[pl.DataFrame, pd.DataFrame, gpd.GeoDataFrame]],
    allow_overwrite: bool = False,
) -> None:
    """
    Validate that output file paths are writable, and provided files are non-empty.

    Args:
        - file_paths: List of output file paths.
        - files: List of dataframes or other objects to validate.
        - allow_overwrite: If False, raises an error if output file already exists.
    """
    for file_path in file_paths:
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
        if os.path.exists(file_path) and not allow_overwrite:
            raise FileExistsError(f"Output file already exists: {file_path}")

    def _is_empty(obj: Any) -> bool:
        """Check if an object is empty."""
        if obj is None:
            return True
        if isinstance(obj, (list, dict, set, tuple, str)):
            return len(obj) == 0
        if isinstance(obj, pl.DataFrame):
            return obj.is_empty()
        if isinstance(obj, (pd.DataFrame, gpd.GeoDataFrame)):
            return obj.empty
        if isinstance(obj, az.InferenceData):
            return not obj.groups()
        if isinstance(obj, pm.Model):
            return not bool(obj.coords)
        if isinstance(obj, Lmer):
            return not getattr(obj, "fitted", False)
        return False  # Default for unsupported types

    for file_object in files:
        if _is_empty(file_object):
            raise ValueError(
                f"The provided {type(file_object).__name__} object is empty."
            )


# Dataframe Validation --------------------------------------------------------


def check_duplicates(df: Union[pl.DataFrame, pd.DataFrame, gpd.GeoDataFrame]) -> int:
    """Check for duplicate rows in a dataframe."""
    if isinstance(df, pl.DataFrame):
        return df.is_duplicated().sum()
    if isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
        return df.duplicated().sum()
    raise TypeError("Unsupported dataframe type. Use Polars, Pandas, or GeoPandas.")


def get_unique_value_count(df: pl.DataFrame, column: str) -> int:
    """Get the number of unique values in a specified column."""
    return df.select(column).n_unique()


def check_no_nans(df: pl.DataFrame, columns: list[str]) -> None:
    """Ensure specified columns contain no NaN values."""
    for col in columns:
        if df.select(pl.col(col).is_nan()).to_series().sum() > 0:
            raise ValueError(f"Column '{col}' contains NaN values.")


def check_within_bounds(
    df: pl.DataFrame, columns: list[str], lower: float = 0, upper: float = 1
) -> None:
    """Ensure specified columns' values are within a specific range."""
    for col in columns:
        out_of_bounds = df.filter((pl.col(col) < lower) | (pl.col(col) > upper))
        if not out_of_bounds.is_empty():
            raise ValueError(
                f"Column '{col}' contains values outside the range [{lower}, {upper}]."
            )


def check_non_negative(df: pl.DataFrame, columns: list[str]) -> None:
    """Ensure specified columns contain only non-negative values."""
    for column in columns:
        negative_values = df.filter(pl.col(column) < 0)
        if not negative_values.is_empty():
            raise ValueError(f"Column '{column}' contains negative values.")
