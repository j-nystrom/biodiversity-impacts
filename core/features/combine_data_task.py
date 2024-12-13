import os
import time
from datetime import timedelta

import numpy as np
import polars as pl
from box import Box

from core.utils.general_utils import create_logger

# Load config file
script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "feature_configs.yaml"))

# For this task we also need to access the 'data_configs.yaml' file
project_dir = os.path.dirname(script_dir)  # Navigate up to the project root
data_config_path = os.path.join(project_dir, "data", "data_configs.yaml")
data_configs = Box.from_yaml(filename=data_config_path)

logger = create_logger(__name__)


class CombineDataTask:
    """
    Task for creating a unified dataset from PREDICTS, population density,
    road density, bioclimatic variables and topographic variables that were
    generated in the data pipeline.

    NOTE: If more data sources are added in the future in the data pipeline,
    they need to be incorporated here as well.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            run_folder_path: Folder where logs and certain outputs are stored.
            predicts_data: Path to file with concatenated PREDICTS data.
            pop_density_data: List of paths to population data files, one for
                each buffer size / resolution, containing the years 2000-2020
                in 5-year increments.
            road_density_data: Paths to road density data files, one
                per UN region that includes all resolutions. This data is
                static in time.
            bioclimatic_data: Paths to bioclimatic data files (temperature and
                precipitation), one for each resolution. Also static in time.
            topegraphic_data: Same but for topographic factors (elevation and
                terrain features).
            year_intervals: The year intervals that population data needs to be
                interpolated between. The first year in PREDICTS is 1984 and
                the last is 2018. Population data is available 2000, 2005,
                2010, 2015 and 2020.
            pop_density_polygon_sizes: The different resolutions used when
                calculating population density in the data pipeline.
            combined_data_path: Output path for the final combined file.
        """
        self.run_folder_path = run_folder_path
        self.predicts_data: str = data_configs.predicts.merged_data_path
        self.pop_density_data: list[str] = (
            data_configs.raster_data.pop_density.output_paths
        )
        self.road_density_data: list[str] = data_configs.road_density.output_paths
        self.bioclimatic_data: list[str] = (
            data_configs.raster_data.bioclimatic.output_paths
        )
        self.topographic_data: list[str] = (
            data_configs.raster_data.topographic.output_paths
        )
        self.year_intervals: list[tuple[int, int]] = configs.combine_data.year_intervals
        self.pop_density_polygon_sizes: list[str] = (
            data_configs.raster_data.pop_density.polygon_sizes_km
        )
        self.combined_data_path: str = configs.combine_data.combined_data_path

    def run_task(self) -> None:
        """
        The following key processing steps are performed:
        - PREDICTS data is loaded.
        - Road density data for each region is loaded and concatenated, then
            joined with the PREDICTS data using site id ('SSBS') as key.
        - Bioclimatic and topographic data is loaded and joined to the
            dataframe based on the site id.
        - Population data for each resolution is loaded, and in-between years
            are interpolated.
        - This is joined with the first dataframe using 'SSBS' and the
            sampling year as join keys. This implies each site will have
            population data matching the sampling year for that site / study.
        """
        logger.info("Starting process of merging different data sources.")
        start = time.time()

        # Load the PREDICTS data
        df_predicts = pl.read_parquet(self.predicts_data)

        # Load road density dataset and concatenate them together
        # NOTE: '__index_level_0__' is a pandas-polars artefact, to be removed
        # once the road density task has been refactored
        df_road_density = pl.DataFrame()
        for path in self.road_density_data:
            df = pl.read_parquet(path).drop("__index_level_0__")
            df_road_density = pl.concat([df_road_density, df], how="vertical")

        # Join PREDICTS and road density datasets together
        df_combined = df_predicts.join(
            df_road_density, on="SSBS", how="left", validate="m:1"
        ).sort("SSBS")

        # Load bioclimatic variables and join with the rest
        df_bioclimatic = pl.DataFrame()
        for i, path in enumerate(self.bioclimatic_data):
            df = pl.read_parquet(path).sort("SSBS")
            # If not the first file, drop SSBS column to avoid duplicates
            if i > 0:
                df = df.drop("SSBS")
            df_bioclimatic = pl.concat([df_bioclimatic, df], how="horizontal")

        df_combined = df_combined.join(
            df_bioclimatic, on="SSBS", how="left", validate="m:1"
        ).sort("SSBS")

        # Do the same for topographic features
        df_topography = pl.DataFrame()
        for i, path in enumerate(self.topographic_data):
            df = pl.read_parquet(path).sort("SSBS")
            if i > 0:
                df = df.drop("SSBS")
            df_topography = pl.concat([df_topography, df], how="horizontal")

        df_combined = df_combined.join(
            df_topography, on="SSBS", how="left", validate="m:1"
        )

        # The population data requires more processing, specifically doing
        # interpolation between available years. We do that individually for
        # each buffer resolution, then concatenate
        pop_density_dfs = []
        for path, buffer_size in zip(
            self.pop_density_data, self.pop_density_polygon_sizes
        ):
            # Load data and remove column prefixes for the year-related cols,
            # so only the year remains
            df = pl.read_parquet(path).rename(
                lambda col: col if col == "SSBS" else col[-4:]
            )
            # Do interpolation and append data for this resolution to list
            df_interpol = self.interpolate_population_density(
                df, year_intervals=self.year_intervals, buffer_size=buffer_size
            )
            pop_density_dfs.append(df_interpol)

        # Create a sampling year column in the dataframe with the other data
        df_combined = df_combined.with_columns(
            pl.col("Sample_midpoint")
            .str.to_datetime("%Y-%m-%d")
            .dt.year()
            .alias("Sample_year")
        )

        # Join the population densities of the year matching the sampling year
        for df in pop_density_dfs:
            df_combined = df_combined.join(
                df,
                how="left",
                left_on=["SSBS", "Sample_year"],
                right_on=["SSBS", "Year"],
            )

        # Write the final dataframe to file
        df_combined.write_parquet(self.combined_data_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Data merging finished in {runtime}.")

    def interpolate_population_density(
        self,
        df: pl.DataFrame,
        buffer_size: str,
        year_intervals: list[tuple[int, int]],
    ) -> pl.DataFrame:
        """
        Interpolate population density values in between available years, as
        well as going back to the first year in the PREDICTS database. The
        growth rate for each interval is based on the start and end year of
        that interval, assuming an exponential growth curve.

        Args:
            df: Dataframe containing one site id column ('SSBS') and a number
                of year columns. These should match the start and end dates of
                the year intervals.
            buffer_size: The resolution of the population data in the df.
            year_interval: The year intervals that population data needs to be
                interpolated between.

        Returns:
            df: Original df with additional columns for all interpolated years.
        """
        logger.info(f"Population density interpolation for resolution {buffer_size}.")

        def _calculate_growth_rate(
            df: pl.DataFrame, start_year: int, end_year: int
        ) -> pl.Series:
            """Calculate the growth rate between two years."""

            # Average growth rate between start and end year
            rates = np.log(
                df.get_column(str(end_year)) / df.get_column(str(start_year))
            ) / (end_year - start_year)

            # If there are NaN or inf values, fill with zeros and log this
            if rates.is_nan().any() or rates.is_infinite().any():
                rates = (
                    pl.when(rates.is_nan() | rates.is_infinite())
                    .then(0)
                    .otherwise(rates)
                )
                logger.warning(
                    f"NaN or inf values in growth rate for {start_year}-{end_year}. \
                    Setting growth rates to 0."
                )

            return rates

        # Extrapolate back to 1984 using the growth rate from 2000 to 2005
        r_backwards = _calculate_growth_rate(
            df, year_intervals[1][0], year_intervals[1][1]  # 2000-2005
        )
        df = df.with_columns(
            (
                df[str(year_intervals[1][0])]
                * np.exp(r_backwards * (year_intervals[0][0] - year_intervals[0][1]))
            ).alias("1984")
        )

        # Loop through each interval to calculate growth rates and interpolate
        for start_year, end_year in year_intervals:
            r = _calculate_growth_rate(df, start_year, end_year)
            for year in range(start_year, end_year + 1):
                if year not in df.columns:
                    df = df.with_columns(
                        (df[str(start_year)] * np.exp(r * (year - start_year))).alias(
                            str(year)
                        )
                    )

        # Reorder the columns to have them in chronological order
        df = df[["SSBS"] + sorted(df.columns[1:], key=int)]

        # Melt dataframe to go from wide to long format
        df = df.melt(
            id_vars=["SSBS"],
            value_vars=df.columns[1:],
            variable_name="Year",
            value_name=f"Pop_density_{buffer_size}km",
        ).sort(["SSBS", "Year"])

        # Convert to datetime format
        df = df.with_columns(pl.col("Year").str.strptime(pl.Datetime, "%Y").dt.year())

        logger.info("Finished population density interpolation.")

        return df
