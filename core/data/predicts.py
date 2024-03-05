import os
import time
from datetime import timedelta
from typing import List

import geopandas as gpd
import polars as pl
from box import Box
from shapely.geometry import Point

from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "data_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class PredictsProcessingTask:
    """
    Task for loading and concatenating PREDICTS datasets, and extracting
    sampling site coordinates as Point geometries.
    """

    def __init__(self) -> None:
        """Initialize the class instance."""
        pass

    def run_task(
        self,
        source_file_2016: str = configs.predicts.source_file_2016,
        source_file_2022: str = configs.predicts.source_file_2022,
        col_order: List[str] = configs.predicts.col_order,
        concat_data_output_path: str = configs.predicts.concat_data_output_path,
        site_coord_output_path: str = configs.predicts.site_coord_output_path,
    ) -> None:
        """
        Orchestrate the processing of PREDICTS data, incl. loading,
        concatenating, generating site coordinates and saving dataframes.

        This function performs the following steps:
        1. Loads the PREDICTS datasets for 2016 and 2022.
        2. Concatenates the datasets together.
        3. Generates a geodataframe with coordinates for each sampling site.
        4. Saves both dataframes to disk.

        Args:
            source_file_2016 (str):
            source_file_2022 (str):
            col_order (List[str]):
            concat_data_output_path (str):
            site_coord_output_path (str):

        Returns:
            None
        """
        logger.info("Starting processing of PREDICTS data.")
        start = time.time()

        # Load the two PREDICTS datasets (2016 and 2022)
        read_kwargs = {"infer_schema_length": 100000, "null_values": ["NA"]}
        df_2016 = pl.read_csv(source_file_2016, **read_kwargs)
        df_2022 = pl.read_csv(source_file_2022, **read_kwargs)

        # Concatenate them together
        df_concat = self.concatenate_datasets(df_2016, df_2022, col_order=col_order)

        # Generate a geodataframe with coordinates for every sampling site
        gdf_site_coords = self.create_site_coord_geometries(df_concat)

        # Save both dataframes to disk
        df_concat.write_parquet(concat_data_output_path)
        gdf_site_coords.to_file(site_coord_output_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"PREDICTS data processing finished in {runtime}.")

    def concatenate_datasets(
        self, df_2016: pl.DataFrame, df_2022: pl.DataFrame, col_order: List[str]
    ) -> pl.DataFrame:
        """
        Concatenate the two PREDICTS datasets, ensuring identical column
        structures and specified order.

        Args:
            df_2016 (pl.DataFrame): The PREDICTS dataset from 2016.
            df_2022 (pl.DataFrame): The PREDICTS dataset from 2022.
            col_order (List[str]): Column order in concatenated dataframe.

        Returns:
            pl.DataFrame: Concatenated dataframe with ordered columns.

        Raises:
            ValueError: If 'col_order' has extra or missing columns from the
                concatenated dataframe.
        """
        logger.info("Concatenating the PREDICTS datasets.")

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

        # Ensure 'col_order' contains all columns present in the dataframe
        # and that 'col_order' doesn't contain any extra columns
        missing_cols = set(df_concat.columns) - set(col_order)
        if missing_cols:
            raise ValueError(f"Missing columns in 'col_order': {missing_cols}")

        extra_cols = set(col_order) - set(df_concat.columns)
        if extra_cols:
            raise ValueError(f"Extra columns in 'col_order': {extra_cols}")

        # Reorder the columns according to 'col_order'
        df_concat = df_concat.select(col_order)

        logger.info("Finished concatenating datasets.")

        return df_concat

    def create_site_coord_geometries(self, df_concat: pl.DataFrame) -> gpd.GeoDataFrame:
        """
        Generate a geodataframe with Point geometries for each unique site
        based on longitude and latitude, and adds UN region information for
        filtering.

        Args:
            df_concat (pl.DataFrame): The concatenated PREDICTS data.

        Returns:
            gdf_site_coords (gpd.GeoDataFrame): Geodataframe with Point
                coordinates and region information for each sampling site.
        """
        logger.info("Creating Point geometries for site coordinates.")

        # Get the coordinates for each unique site in the dataset
        df_long_lat = df_concat.groupby("SSBS").agg(
            [
                pl.col("Longitude").first().alias("Longitude"),
                pl.col("Latitude").first().alias("Latitude"),
            ]
        )

        # Generate coordinate tuples from the long-lat columns
        coordinates = zip(
            df_long_lat.get_column("Longitude").to_list(),
            df_long_lat.get_column("Latitude").to_list(),
        )

        # Create shapely Point geometries for all coordinates
        geometry = [Point(x, y) for x, y in coordinates]

        # Create a geodataframe containing site id and coordinates
        gdf_site_coords = gpd.GeoDataFrame(
            {"SSBS": df_long_lat.get_column("SSBS"), "geometry": geometry}
        )
        gdf_site_coords.crs = "EPSG:4326"

        # Add the UN region to enable filtering when working with the geodata
        df_region = (
            df_concat.groupby("SSBS").agg(pl.col("UN_region").first()).to_pandas()
        )  # Need to convert to pandas to be compatible with geopandas

        # Make sure column 'SSBS' is a string, then merge the dataframes
        gdf_site_coords = gdf_site_coords.merge(
            df_region, on="SSBS", how="left", validate="1:1"
        )

        logger.info("Finished creating Point geometries.")

        return gdf_site_coords
