import os
import re
import time
from datetime import timedelta

import geopandas as gpd
import pandas as pd
from box import Box

from core.data.data_processing import calculate_raster_stats
from core.utils.general_utils import create_logger

# Load config file and set up logger
script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "data_configs.yaml"))

logger = create_logger(__name__)


class CalculateRasterStatsTask:
    """
    Base class for calculating various statistics from a set of sampling site
    polygons, and a raster dataset containing some information. Separate
    classes for specific raster datasets, that inherit from this class, are
    implemented further down.

    Future improvement: Separate non-overlapping and overlapping polygons.
    Process the first group using pygeoprocessing for increased speed. See:
    https://stackoverflow.com/questions/47471872/find-non-overlapping-polygons-in-geodataframe

    TODO: Refactor to use polars instead of pandas
    """

    def __init__(self) -> None:
        """
        Attributes:
            all_site_coords: Geodataframe with coords of all sampling sites.
            global_site_polygons: List of shapefiles with buffered site
                polygons at previously buffered scales.
        """
        self.all_site_coords: str = configs.predicts.all_site_coords
        self.global_site_polygons: str = configs.raster_data.global_site_polygons

    def run_mode(self, mode: str) -> None:
        """
        Run the calculation / extraction of statistics from one or several
        pairs of raster datasets and polygon shapefiles that overlap spatially.
        It's assumed that every combination of raster paths and polygon paths
        should be processed.

        The 'mode' argument determines the raster data sources to use, e.g. for
        population density or bioclimatic variables. These correspond to unique
        input data paths in the 'data_configs.yaml' file.
        """
        assert mode in [
            "pop_density",
            "bioclimatic",
            "topographic",
        ], "'mode' needs to be in ['pop_density', 'bioclimatic', 'topographic']"
        logger.info(f"Starting raster data extraction for mode {mode}.")
        start = time.time()

        # Load the geodataframe that will hold the results, keeping the site id
        df_sites = pd.DataFrame(gpd.read_file(self.all_site_coords)["SSBS"])

        # Get the configs for this particular mode
        mode_configs = configs.raster_data[mode]
        self.polygon_sizes = mode_configs.polygon_sizes
        self.raster_paths = mode_configs.raster_paths
        self.result_col_names = mode_configs.result_col_names
        self.agg_metrics = mode_configs.rasterstats_settings.metrics
        self.include_all_pixels = mode_configs.rasterstats_settings.include_all_pixels
        self.output_paths = mode_configs.output_paths

        # Keep only the polygon paths that match the desired scales
        polygon_paths = []
        for path in self.global_site_polygons:
            for size in self.polygon_sizes:
                if re.search(f"{size}km", path):
                    polygon_paths.append(path)

        # Iterate through every combination of polygon datasets (shapefiles)
        # and raster datasets to extract the desired statistics
        i = 0
        for polygon_path, output_path in zip(polygon_paths, self.output_paths):
            df_result = df_sites.copy()
            for raster_path in self.raster_paths:
                logger.info(
                    f"Processing polygon {polygon_path} and raster {raster_path}."
                )
                start_step = time.time()

                # Calculate the statistics for the current polygon and raster
                stats = calculate_raster_stats(
                    polygon_path,
                    raster_path,
                    metric=self.agg_metrics,
                    include_all_pixels=self.include_all_pixels,
                )

                # Add the results to the dataframe as a new column
                df_result.loc[:, self.result_col_names[i]] = stats
                i += 1

                runtime_step = str(timedelta(seconds=int(time.time() - start_step)))
                logger.info(f"Processing finished in {runtime_step}.")

            # Save final dataframe for this polygon path (i.e. buffer size)
            df_result.to_parquet(output_path)
            logger.info(f"Saved results for polygon {polygon_path}.")

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Raster data extraction finished in {runtime}.")


class PopulationDensityTask(CalculateRasterStatsTask):
    """Population density data."""

    def __init__(self, run_folder_path: str) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode="pop_density")


class BioclimaticFactorsTask(CalculateRasterStatsTask):
    """Bioclimatic factors data."""

    def __init__(self, run_folder_path: str) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode="bioclimatic")


class TopographicFactorsTask(CalculateRasterStatsTask):
    """Topographic factors data."""

    def __init__(self, run_folder_path: str) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode="topographic")
