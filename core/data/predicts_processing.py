# Standard package imports
import polars as pl
import geopandas as gpd
from shapely.geometry import Point
from box import Box
from typing import List

# Load config file into a box object
configs = Box.from_yaml(filename="data_configs.yaml")


def run():

    # Load the two PREDICTS datasets (2016 and 2022)
    df_2016 = pl.read_csv(source=configs.predicts.source_file_2016)
    df_2022 = pl.read_csv(source=configs.predicts.source_file_2016)

    # Concatenate them together
    df_concat = concatenate_datasets(df_2016, df_2022)

    # Generate a geodataframe with coordinates for every sampling site
    gdf_site_coords = create_site_coord_geometries(df_concat)

    # Save both dataframes to disk
    df_concat.write_parquet(file=configs.predicts.concat_data_output_path)
    gdf_site_coords.to_file(filename=configs.predicts.site_coord_output_path)


def concatenate_datasets(
    df_2016: pl.DataFrame, df_2022: pl.DataFrame, col_order: List[str]
) -> pl.DataFrame:

    # Find out if there are any columns that are not overlapping
    unique_cols_2016 = list(set(df_2016.columns) - set(df_2022.columns))
    unique_cols_2022 = list(set(df_2022.columns) - set(df_2016.columns))
    all_unique_cols = unique_cols_2016 + unique_cols_2022

    # Drop non-overlapping columns from both dataframes
    df_2016 = df_2016.drop(all_unique_cols)
    df_2022 = df_2022.drop(all_unique_cols)

    # Make sure we have the same column order in both
    df_2022 = df_2022.select(df_2016.columns)

    # Append new data to old and then sort the columns in the right order
    df_concat = pl.concat([df_2016, df_2022], how="vertical")
    df_concat = df_concat.select(col_order)

    return df_concat


def create_site_coord_geometries(df_concat: pl.DataFrame) -> gpd.GeoDataFrame:

    # Get the coordinates for each unique site in the dataset
    df_long_lat = df_concat.groupby("SSBS").agg(
        [pl.col("Longitude").first(), pl.col("Latitude").first()]
    )

    # Generate coordinate tuples from the long-lat columns
    coordinates = zip(
        df_long_lat.select("Longitude").to_list(),
        df_long_lat.select("Latitude").to_list(),
    )

    # Create shapely Point geometries for all coordinates
    geometry = [Point(x, y) for x, y in coordinates]

    # Create a geodataframe containing site id and coordinates
    gdf_site_coords = gpd.GeoDataFrame(
        {"SSBS": df_long_lat.select("SSBS"), "geometry": geometry}
    )
    gdf_site_coords.crs = "EPSG:4326"

    # Add the UN region to enable filtering when working with the geodata
    df_region = (
        df_concat.groupby("SSBS").agg(pl.col("UN_region").first()).to_pandas()
    )  # Convert to pandas to be compatible with geopandas
    gdf_site_coords = gdf_site_coords.join(
        df_region, on="SSBS", how="left", validate="1:1"
    )

    return gdf_site_coords
