import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.features.feature_engineering import (
    calculate_scaled_abundance,
    calculate_study_mean_densities,
    combine_land_use_and_intensity,
    filter_out_insufficient_data_studies,
)
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "feature_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class AbundanceFeatureEngineeringTask:
    """
    Task to create dataframes for modelling based on total species abundance as
    the response variable. One file is generated for each level of taxonomic
    granularity considered for the modelling.
    """

    def __init__(self) -> None:
        """
        Attributes:
            combined_data:
            cols_to_keep:
            taxonomic_levels:
            lui_col_order:
            abundance_data:
            study_mean_cols:
        """
        self.combined_data: str = configs.combined_data.combined_data_file
        self.cols_to_keep: list[str] = configs.feature_engineering.cols_to_keep
        self.study_mean_cols: list[str] = configs.feature_engineering.study_mean_cols
        self.lui_col_order: list[str] = configs.feature_engineering.lui_col_order
        self.taxonomic_levels: list[str] = configs.abundance.taxonomic_levels
        self.abundance_data: list[str] = configs.abundance.abundance_data

    def run_task(self) -> None:
        """Add docstring."""
        logger.info("Initiating feature pipeline for abundance.")
        start = time.time()

        # Read PREDICTS data and keep only columns that we need
        df = pl.read_parquet(self.combined_data)
        df = df.select(self.cols_to_keep)

        # Filter out cases where land-use or intensity is missing
        df = filter_out_insufficient_data_studies(df)

        # Calculate mean values for population and road density, per resolution
        df = calculate_study_mean_densities(df, cols_to_incl=self.study_mean_cols)

        # Combine land use and intensity and create dummy variables from this
        df = combine_land_use_and_intensity(df, lui_col_order=self.lui_col_order)

        # List of columns to group by. For each deeper level in the taxonomic
        # hierarchy, another column is added to the base list
        groupby_cols = ["SS", "SSB", "SSBS"]

        for i, path in enumerate(self.abundance_data):
            df_abund = calculate_scaled_abundance(df, groupby_cols=groupby_cols)

            # Get the first instance of each SSBS for the specified covariates
            # Drop columns that relate to individual taxon measurements
            # Drop columns with more granular species info than the grouping
            df_first = df.group_by("SSBS").agg(pl.first(df.columns))
            df_first = df_first.drop(
                [
                    "Taxon_name_entered",
                    "Measurement",
                    "Effort_corrected_measurement",
                ]
            )
            df_first = df_first.drop(self.taxonomic_levels[i:])
            df_res = df_abund.join(
                df_first,
                on=groupby_cols,
                how="left",
            )

            # Save output file for this level of aggregation
            df_res.write_parquet(path)

            # Update the level used for grouping
            if i < len(self.taxonomic_levels):
                groupby_cols.append(self.taxonomic_levels[i])
            else:
                break

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Abundance calculations finished in {runtime}.")
