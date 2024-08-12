import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.features.feature_engineering import (
    calculate_scaled_mean_abundance,
    calculate_scaled_species_richness,
    calculate_scaled_total_abundance,
    calculate_study_mean_densities,
    combine_biogeographical_variables,
    combine_land_use_intensity_columns,
    create_land_use_dummies,
    filter_out_insufficient_data_studies,
    group_land_use_types_and_intensities,
    transform_continuous_covariates,
)
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "feature_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class AbundanceFeaturesTask:
    """
    Task to create dataframes for modelling based on total species abundance as
    the response variable. One file is generated for each level of taxonomic
    granularity considered for the modelling.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            combined_data: Path to file with combined PREDICTS and other data.
            cols_to_keep: Columns to keep from the original PREDICTS data.
            continuous_vars: Continuous variables that should be transformed
                for use in the model.
            study_mean_cols: Population and road density columns for which
                within-study mean values should be calculated.
            land_use_col_order: The order of land-use dummy columns in the
                final dataframe
            lui_col_order: The order of combined land-use and intensity dummy
                columns in the final dataframe.
            secondary_veg_col_order: The order of dummy variables created by
                grouping different types of secondary vegetation.
            taxonomic_levels: The levels in the taxonomic hierarchy that should
                be used as groupby columns when calculating abundance.
            abundance_data: Output path for the final dataframe.
        """
        self.combined_data: str = configs.combined_data.combined_data_file
        self.cols_to_keep: list[str] = configs.feature_engineering.cols_to_keep
        self.density_vars: list[str] = configs.feature_engineering.density_vars
        self.bioclim_vars: list[str] = configs.feature_engineering.bioclim_vars
        self.land_use_col_order: list[str] = (
            configs.feature_engineering.land_use_col_order
        )
        self.lui_col_order: list[str] = configs.feature_engineering.lui_col_order
        self.secondary_veg_col_order: list[str] = (
            configs.feature_engineering.secondary_veg_col_order
        )
        self.taxonomic_levels: list[str] = configs.abundance.taxonomic_levels
        self.abundance_data: list[str] = configs.abundance.abundance_data

    def run_task(self) -> None:
        """
        Runs a sequence of steps to create a dataframe that can be used as
        input to the model pipeline, with total species abundance as the
        response variable:
        1. Read the combined data from the previous pipeline step.
        2. Filter out incomplete-data site observations.
        3. Generate dummy variables for land-use, combined land-use and
        intensity, as well as some special case dummies from BII.
        4. Transform continuous variables (road and population density)
        5. Calculate mean values for population and road density.
        6. For each groupby key (different levels of granularity), calculate
        and scale abundance per site.
        """
        logger.info("Initiating feature pipeline for abundance.")
        start = time.time()

        # Read PREDICTS data and keep only columns that we need
        df = pl.read_parquet(self.combined_data, columns=self.cols_to_keep)

        # Filter out cases where land-use or intensity is missing
        df = filter_out_insufficient_data_studies(df)

        # Create dummy variables for land-use related columns
        df = create_land_use_dummies(df, col_order=self.land_use_col_order)
        df = combine_land_use_intensity_columns(df, col_order=self.lui_col_order)
        df = group_land_use_types_and_intensities(
            df, col_order=self.secondary_veg_col_order
        )

        # Combine biogeographical variables (biome, realm and ecoregion)
        df = combine_biogeographical_variables(df)

        # Generate various transformations for all continuous variables
        df, new_cols = transform_continuous_covariates(
            df, continuous_vars=self.density_vars
        )

        # Calculate mean values for population and road density, per resolution
        df = calculate_study_mean_densities(
            df, cols_to_incl=self.density_vars + new_cols
        )

        # List of columns to group by. For each deeper level in the taxonomic
        # hierarchy, another column is added to the base list
        groupby_cols = ["SS", "SSB", "SSBS"]

        for i, path in enumerate(self.abundance_data):
            df_total_abund = calculate_scaled_total_abundance(
                df, groupby_cols=groupby_cols
            )
            df_mean_abund = calculate_scaled_mean_abundance(
                df, groupby_cols=groupby_cols
            )
            df_richness = calculate_scaled_species_richness(
                df, groupby_cols=groupby_cols
            )

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
            df_res = df_total_abund.join(
                df_mean_abund,
                on=groupby_cols,
                how="left",
            )
            df_res = df_res.join(
                df_richness,
                on=groupby_cols,
                how="left",
            )
            df_res = df_res.join(
                df_first,
                on=groupby_cols,
                how="left",
            )

            # Save output file for this level of aggregation
            df_res.write_parquet(path)

            # Update the level used for grouping. 'taxonomic_levels' is a list
            # that contains 5 levels of the taxonomic hierarchy. We continue
            # the loop until all levels in that list have been added to the
            # groupby_cols list.
            if i < len(self.taxonomic_levels):
                groupby_cols.append(self.taxonomic_levels[i])
            else:
                break

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Abundance calculations finished in {runtime}.")
