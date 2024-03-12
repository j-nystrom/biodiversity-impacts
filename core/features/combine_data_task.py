import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.features.feature_engineering import interpolate_population_density
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "feature_configs.yaml")
configs = Box.from_yaml(filename=config_path)

# For this task we also need to access the 'data_configs.yaml' file
project_dir = os.path.dirname(script_dir)  # Navigate up to the project root
data_config_path = os.path.join(project_dir, "data", "data_configs.yaml")
data_configs = Box.from_yaml(filename=data_config_path)

logger = create_logger(__name__)


class CombineDataSourcesTask:
    """
    Task for creating a unified dataset from PREDICTS, population density and
    road density.
    """

    def __init__(self) -> None:
        """
        Attributes:
            all_predicts_data: Path to file with concatenated PREDICTS data.
            pop_density_data: List of paths to population data files, one for
                each resolution (1, 10, 50 km).
            road_density_data: List of paths to road density data files, one
                per UN region.
            year_intervals: The year intervals that population data needs to be
                interpolated between. The first year in PREDICTS is 1984 and
                the last is 2018. Population data is available 2000, 2005, 2010,
                2015 and 2020.
            pixel_resolution: The different resolutions of the population data.
            combined_data_file: Output path for the final combined file.
        """
        # TODO: Create two config files for this particular task as workaround
        self.all_predicts_data: str = data_configs.predicts.all_predicts_data
        self.pop_density_data: list[str] = data_configs.geodata.pop_density.output_paths
        self.road_density_data: list[str] = data_configs.geodata.roads.road_density_data
        self.year_intervals: list[tuple[int, int]] = (
            configs.combined_data.year_intervals
        )
        self.pixel_resolutions: list[str] = configs.combined_data.pixel_resolutions
        self.combined_data_file: str = configs.combined_data.combined_data_file

    def run_task(self) -> None:
        """
        Runs a sequence of steps to create the unified dataset:
        1. PREDICTS data is loaded.
        2. Road density data for each region is loaded and concatenated, then
        joined with the PREDICTS data using site id ('SSBS') as key.
        3. Population data for each resolution is loaded, and in-between years
        are interpolated.
        4. This is joined with the first dataframe using 'SSBS' and the
        sampling year as join keys. This implies each site will have population
        data matching the sampling year for that site / study.
        """
        logger.info("Starting process of merging different data sources.")
        start = time.time()

        # Load PREDICTS and road density data
        # NOTE: '__index_level_0__' is a pandas to polars artefact, should be
        # fixed in the future
        df_predicts = pl.read_parquet(self.all_predicts_data)

        df_road_density = pl.DataFrame()
        for path in self.road_density_data:
            df = pl.read_parquet(path).drop("__index_level_0__")
            df_road_density = pl.concat([df_road_density, df], how="vertical")

        # Join these two datasets together
        df_predicts_roads = df_predicts.join(
            df_road_density, on="SSBS", how="left", validate="m:1"
        )

        # The population data requires more processing, specifically doing
        # interpolation between available years. We do that individually for
        # each buffer resolution, then concatenate
        pop_density_dfs = []
        for path, resolution in zip(self.pop_density_data, self.pixel_resolutions):
            # Load data and remove column prefixes, so only the year remains
            df = pl.read_parquet(path).rename(
                lambda col: col if col == "SSBS" else col[-4:]
            )
            # Do interpolation and append data for this resolution to list
            df_interpol = interpolate_population_density(
                df, year_intervals=self.year_intervals, pixel_resolution=resolution
            )
            pop_density_dfs.append(df_interpol)

        # Convert first df to datetime format, for joining with population data
        df_predicts_roads = df_predicts_roads.with_columns(
            pl.col("Sample_midpoint")
            .str.to_datetime("%Y-%m-%d")
            .dt.year()
            .alias("Sample_year")
        )

        # Join the population densities of the year matching the sample year
        df_all = df_predicts_roads.clone()
        for df in pop_density_dfs:
            df_all = df_all.join(
                df,
                how="left",
                left_on=["SSBS", "Sample_year"],
                right_on=["SSBS", "Year"],
            )

        # Write the final dataframe to file
        df_all.write_parquet(self.combined_data_file)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Data merging finished in {runtime}.")
