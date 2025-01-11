import geopandas as gpd
import polars as pl

from core.tests.shared.validate_shared import check_duplicates, get_unique_value_count

# PredictsConcatenationTask --------------------------------------


def validate_concatenate_predicts_dataframes(
    df_2016: pl.DataFrame,
    df_2022: pl.DataFrame,
    df_concat: pl.DataFrame,
    col_order: list[str],
) -> None:
    """Validate the output of the 'concatenate_predicts_dataframes' function in
    'PredictsConcatenationTask'."""
    # Check for missing or extra columns in the inputted col_order list
    missing_cols = set(df_concat.columns) - set(col_order)
    if missing_cols:
        raise ValueError(
            f"Missing columns in 'col_order': {missing_cols}, specify all columns."
        )
    extra_cols = set(col_order) - set(df_concat.columns)
    if extra_cols:
        raise ValueError(
            f"Extra columns in 'col_order': {extra_cols}, remove from input list."
        )

    # Check for duplicate rows
    duplicate_count = check_duplicates(df_concat)
    if duplicate_count > 0:
        raise ValueError(f"Duplicate rows detected: {duplicate_count} rows.")

    # Check that the total number of rows adds up
    expected_rows = df_2016.shape[0] + df_2022.shape[0]
    if df_concat.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, but got {df_concat.shape[0]} rows."
        )

    # Check that the total number of unique sites add up
    unique_sites_2016 = get_unique_value_count(df_2016, "SSBS")
    unique_sites_2022 = get_unique_value_count(df_2022, "SSBS")
    expected_unique_sites = unique_sites_2016 + unique_sites_2022
    actual_unique_sites = get_unique_value_count(df_concat, "SSBS")
    if actual_unique_sites != expected_unique_sites:
        raise ValueError(
            f"Unique site count mismatch: Expected {expected_unique_sites} sites, "
            f"but got {actual_unique_sites} sites."
        )


# SiteBufferingTask --------------------------------------


def validate_site_coordinate_data(
    df: pl.DataFrame, required_columns: list[str]
) -> None:
    """
    Validate that the input DataFrame contains the required columns and has
    no missing coordinate values.
    """
    # Check for missing required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in input data: {missing_cols}")

    # Check for rows with missing coordinates
    invalid_coords = df.filter(
        pl.col("Longitude").is_null() | pl.col("Latitude").is_null()
    )
    if not invalid_coords.is_empty():
        raise ValueError(
            f"Input DataFrame contains rows with missing coordinates."
            f"Invalid rows: {invalid_coords}"
        )


def validate_site_geometry_output(
    gdf: gpd.GeoDataFrame,
    expected_crs: str,
) -> None:
    """Validate the output GeoDataFrame from create_site_coord_geometries."""
    # Check that the required columns exist
    required_columns = {"SSBS", "geometry"}
    missing_columns = required_columns - set(gdf.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in GeoDataFrame: {missing_columns}")

    # Check that the 'geometry' column contains valid Point geometries
    if not all(gdf.geometry.notna() & (gdf.geometry.geom_type == "Point")):
        raise ValueError("'geometry' column contains invalid or missing geometries.")

    # Check that all 'SSBS' values are unique
    if check_duplicates(gdf) > 0:
        raise ValueError("The dataframe contains duplicate values.")

    # Check that the CRS is correctly set
    if gdf.crs is None or gdf.crs.to_string() != expected_crs:
        raise ValueError(
            f"Invalid CRS. Expected '{expected_crs}', but got '{gdf.crs}'."
        )
