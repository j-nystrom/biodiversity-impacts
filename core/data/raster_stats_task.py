import os
import time
from datetime import timedelta

import geopandas as gpd
import pandas as pd
from box import Box

from core.data.data_processing import calculate_raster_stats
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "data_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class CalculateRasterStatsTask:
    """
    Base class for calculating various statistics from a set of sampling site
    polygons, and a raster dataset containing some information. Separate
    classes for specific raster datasets, that inherit from this class, are
    implemented further down.

    TODO: Separate non-overlapping and overlapping polygons. Process the first
    group using pygeoprocessing for increased speed. See:
    https://stackoverflow.com/questions/47471872/find-non-overlapping-polygons-in-geodataframe
    """

    def __init__(self) -> None:
        """
        Attributes:
            all_site_coords:
            buffer_distances:
        """
        self.all_site_coords: str = configs.predicts.all_site_coords
        self.buffer_distances: list[int] = configs.geodata.buffer_distances

    def run_mode(self, mode: str) -> None:
        """
        Runs the calculation / extraction of statistics from one or several
        pair of overlapping raster datasets and polygon shapefiles. It's
        assumed that every combination of raster paths and polygon paths should
        be processed.

        The 'mode' argument determines the raster data sources to use, e.g. for
        population density, elevation or bioclimatic variables. These
        correspond to unique data paths in the 'data_configs.yaml' file.
        """
        assert mode in [
            "pop_density",
            "elevation",
            "bioclimatic",
        ], "'mode' needs to be one of ['pop_density', 'elevation', 'bioclimatic']"
        logger.info("Starting raster data extraction.")
        start = time.time()

        # Load the geodataframe that will hold the results
        df_sites = pd.DataFrame(gpd.read_file(self.all_site_coords)["SSBS"])

        # Get the specific configs for this particular mode
        mode_configs = configs.geodata[mode]
        polygon_paths = mode_configs.site_polygon_paths
        raster_paths = mode_configs.raster_paths
        result_col_names = mode_configs.result_col_names
        metric = mode_configs.rasterstats_settings.metrics
        include_all_pixels = mode_configs.rasterstats_settings.include_all_pixels
        output_paths = mode_configs.output_paths

        # Iterate through every combination of polygon and raster paths
        i = 0
        for polygon_path, output_path in zip(polygon_paths, output_paths):
            df_result = df_sites.copy()
            for raster_path in raster_paths:
                logger.info(
                    f"Processing polygon {polygon_path} and raster {raster_path}."
                )
                start_step = time.time()

                stats = calculate_raster_stats(
                    polygon_path,
                    raster_path,
                    metric=metric,
                    include_all_pixels=include_all_pixels,
                )
                df_result.loc[:, result_col_names[i]] = stats
                i += 1

                runtime_step = str(timedelta(seconds=int(time.time() - start_step)))
                logger.info(f"Processing finished in {runtime_step}.")

            # Save final dataframe for this polygon path (i.e. buffer radius)
            df_result.to_parquet(output_path)
            logger.info(f"Saved results for polygon {polygon_path}.")

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Raster data extraction finished in {runtime}.")


class PopulationDensityTask(CalculateRasterStatsTask):
    def __init__(self) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode="pop_density")


class ElevationTask(CalculateRasterStatsTask):
    """NOTE: Not implemented in config yet."""

    def __init__(self) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode="elevation")


class BioclimaticTask(CalculateRasterStatsTask):
    """NOTE: Not implemented in config yet."""

    def __init__(self) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode="bioclimatic")
