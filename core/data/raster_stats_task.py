import os
import re
import time
from datetime import timedelta

import geopandas as gpd
import pandas as pd
import rasterstats
from box import Box

from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "data_configs.yaml"))

logger = create_logger(__name__)


class CalculateRasterStatsTask:
    """
    Base class for calculating various statistics from a set of sampling site
    polygons, and a raster dataset containing some information. Separate
    classes for specific raster datasets, that inherit from this class, are
    implemented further down.
    TODO: Low prio: Refactor to use polars instead of pandas.
    """

    def __init__(self) -> None:
        """
        Attributes:
            all_site_coords: Geodataframe with coords of all sampling sites.
            global_site_polygons: List of shapefiles with buffered site
                polygons at previously buffered scales.
        """
        self.all_site_coords: str = configs.site_geodata.site_coords_path
        self.global_site_polygons: list[str] = list(
            configs.site_geodata.global_polygon_paths
        )

    def configure_mode(self, mode: str) -> None:
        """Load and validate the raster settings for one processing mode."""
        if mode not in ["pop_density", "bioclimatic", "topographic"]:
            raise ValueError(
                "'mode' needs to be in ['pop_density', 'bioclimatic', 'topographic']"
            )

        mode_configs = configs.raster_data[mode]
        self.polygon_sizes = list(mode_configs.polygon_sizes_km)
        self.raster_paths = list(mode_configs.input_raster_paths)
        self.result_col_names = list(mode_configs.result_col_names)
        self.agg_metrics = mode_configs.agg_settings.metrics
        self.include_all_pixels = mode_configs.agg_settings.include_all_pixels
        self.output_paths = list(mode_configs.output_paths)

        self.polygon_paths = []
        for size in self.polygon_sizes:
            matching_paths = [
                path
                for path in self.global_site_polygons
                if re.search(rf"_{size}km\.shp$", path)
            ]
            if len(matching_paths) != 1:
                raise ValueError(
                    f"Expected one global polygon path for {size} km, "
                    f"found {len(matching_paths)}."
                )
            self.polygon_paths.append(matching_paths[0])

        if len(self.output_paths) != len(self.polygon_sizes):
            raise ValueError("Each polygon size must have one raster output path.")

        expected_columns = len(self.polygon_sizes) * len(self.raster_paths)
        if len(self.result_col_names) != expected_columns:
            raise ValueError(
                "Raster result-column count does not match polygon sizes "
                "and input rasters."
            )

    def run_mode(self, mode: str) -> None:
        """
        Run the calculation / extraction of statistics from one or several
        pairs of raster datasets and polygon shapefiles that overlap spatially.
        It's assumed that every combination of raster paths and polygon paths
        should be processed.

        The 'mode' argument determines the raster data sources to use, e.g. for
        population density or bioclimatic variables. These correspond to unique
        input data paths in the 'data_configs.yaml' file.

        Attributes:
            mode: One of 'pop_density', 'bioclimatic', 'topographic'.
            polygon_sizes: List of polygon buffer sizes (in km) to process.
            raster_paths: List of raster dataset paths to process for this mode.
            result_col_names: List of column names for the results dataframe,
                including the type of data and the polygon size.
            agg_metrics: List of metrics to compute from the raster data. Right
                now only using the mean.
            include_all_pixels: Whether to include all pixels that touch the
                polygon boundaries or just pixels with center points within it.
            output_paths: List of output paths for saving the result dataframes.
        """
        logger.info(f"Starting raster data extraction for mode {mode}.")
        start = time.time()

        self.configure_mode(mode)

        # Load the dataframe that will hold the results, keeping the site id
        df_sites = pd.DataFrame(gpd.read_file(self.all_site_coords)["SSBS"])

        # Iterate through every combination of polygon datasets (shapefiles)
        # and raster datasets to extract the desired statistics
        i = 0
        for polygon_path, output_path in zip(self.polygon_paths, self.output_paths):
            df_result = df_sites.copy()
            for raster_path in self.raster_paths:
                logger.info(
                    f"Processing polygon {polygon_path} and raster {raster_path}."
                )
                start_step = time.time()

                # Calculate the statistics for the current polygon and raster
                stats = self.calculate_raster_stats(
                    polygon_path,
                    raster_path,
                    metrics=self.agg_metrics,
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

    @staticmethod
    def calculate_raster_stats(
        polygon_path: str,
        raster_path: str,
        metrics: str = "mean",
        include_all_pixels: bool = True,
    ) -> list[float]:
        """
        Compute statistical metrics for raster pixels that overlap with the
        polygons (representing sampling sites) that should be analyzed.

        Args:
            polygon_path: Path to polygon shapefile with sampling sites.
            raster_path: Path to raster file containing data for extraction.
            metrics: Statistical metrics to compute. Defaults to 'mean'.
            include_all_pixels: Whether to include all pixels that touch the
                polygon boundaries, or just pixels with center points within it.

        Returns:
            result: List of computed values, one for each polygon.
        """

        # Calculate zonal statistics
        stats = rasterstats.zonal_stats(
            vectors=polygon_path,
            raster=raster_path,
            stats=metrics,
            all_touched=include_all_pixels,
        )

        # Extract stats from each dictionary in the output list
        result = [x[metrics] for x in stats]

        return result


class PopulationDensityTask(CalculateRasterStatsTask):
    """Population density data."""

    mode = "pop_density"

    def __init__(self, run_folder_path: str) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode=self.mode)


class BioclimaticFactorsTask(CalculateRasterStatsTask):
    """Bioclimatic factors data."""

    mode = "bioclimatic"

    def __init__(self, run_folder_path: str) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode=self.mode)


class TopographicFactorsTask(CalculateRasterStatsTask):
    """Topographic factors data."""

    mode = "topographic"

    def __init__(self, run_folder_path: str) -> None:
        super().__init__()

    def run_task(self) -> None:
        self.run_mode(mode=self.mode)
