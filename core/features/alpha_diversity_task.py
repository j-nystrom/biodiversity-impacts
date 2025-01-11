import os
import time
from datetime import timedelta
from typing import Callable

import polars as pl
from box import Box

from core.tests.features.validate_features import validate_alpha_diversity_calculations
from core.tests.shared.validate_shared import (
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

# Load config file
script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "feature_configs.yaml"))

logger = create_logger(__name__)


class AlphaDiversityTask:
    """
    Task to create dataframes for modeling of alpha diversity as the response
    variable. Each output dataframe aggregates abundance data at different
    levels of the taxonomic hierarchy (Kingdom, Phylum, Class, Order). This
    pre-computation allows for flexibly experimenting with different levels of
    hierarchical granularity in the model later on.

    The alpha metrics currently included are total species abundance,
    mean species abundance, species richness and two versions of the Shannon
    index. (Mean species abundance is not to be confused with the MSA metric
    from the GLOBIO model, which is a beta diversity metric).

    NOTE: There are more data for species richness than for abundance-based
    metrics. If we really want to develop a model for richness it needs to be a
    separate mode of this task, which doesn't filter out non-abundance studies.
    Right now they are filtered out to match the sites included for the
    abundance-based metrics.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            - run_folder_path: Folder for storing logs and certain outputs.
            - feature_data_path: Path to the shared feature data generated in
                previous task (shared between alpha and beta diversity tasks).
            - groupby_cols: 'SS', 'SSB' and 'SSBS' always used for grouping.
            - taxonomic_levels: The levels in the taxonomic hierarchy that
                should be used as groupby columns when calculating diversity.
            - output_data_paths: Output path for the final dataframe.
        """
        self.run_folder_path = run_folder_path
        self.feature_data_path: str = configs.feature_generation.feature_data_path
        self.groupby_cols: list[str] = configs.diversity_metrics.groupby_cols
        self.taxonomic_levels: list[str] = configs.diversity_metrics.taxonomic_levels
        self.output_data_paths: dict[str, str] = (
            configs.diversity_metrics.output_data_paths.alpha
        )

    def run_task(self) -> None:
        """
        xxx
        """
        logger.info("Initiating calculation of alpha diversity metrics.")
        start = time.time()

        # Read feature data from previous step
        validate_input_files(file_paths=[self.feature_data_path])
        df = pl.read_parquet(self.feature_data_path)

        # Iterate through all taxonomic levels, starting with all species
        # In the first iteration, 'SS', 'SSB' and 'SSBS' is used for grouping
        for i, path in enumerate(self.output_data_paths.values()):
            logger.info(
                f"Calculating at the following aggregation level: {self.groupby_cols}"
            )

            df_tot_abund = self.calculate_total_abundance(df)
            df_mean_abund = self.calculate_mean_abundance(df)
            df_richness = self.calculate_species_richness(df)
            df_shannon = self.calculate_shannon_index(df)

            # Get the first instance of each SSBS from the original dataframe
            # Drop columns that relate to individual taxon measurements
            # Drop columns with more granular species info than the grouping
            # (as that would not be meaningful in that dataframe)
            df_first = df.group_by("SSBS").first()
            df_first = df_first.drop(
                [
                    "Taxon_name_entered",
                    "Measurement",
                    "Effort_corrected_measurement",
                ]
            )
            df_first = df_first.drop(self.taxonomic_levels[i:])

            # Sequentially join the alpha diversity metrics to create one df
            df_res = df_tot_abund.join(  # Total abundance + mean abundance
                df_mean_abund,
                on=self.groupby_cols,
                how="left",
            )
            df_res = df_res.join(  # + Species richness
                df_richness,
                on=self.groupby_cols,
                how="left",
            )
            df_res = df_res.join(  # + Shannon index
                df_shannon,
                on=self.groupby_cols,
                how="left",
            )

            # Finally, include all other features in the original dataframe
            # at the right level of aggregation
            df_res = df_res.join(
                df_first,
                on=self.groupby_cols,
                how="left",
            )

            # Save output file for this level of taxonomic aggregation
            validate_output_files(
                file_paths=[path], files=[df_res], allow_overwrite=True
            )
            df_res.write_parquet(path)

            logger.info(f"Finished calculations at level: {self.groupby_cols}")

            # Update the list of grouping columns for the next iteration
            # 'taxonomic_levels' contains four taxonomic levels, we continue
            # the loop until all levels have been added to groupby_cols
            if i < len(self.taxonomic_levels):
                self.groupby_cols.append(self.taxonomic_levels[i])
            else:
                break

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Alpha diversity calculations finished in {runtime}.")

    def calculate_total_abundance(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate the sum of the abundance of all species in each group."""
        logger.info("Calculating total species abundance.")

        metric = "Total_abundance"
        df_res = self.calculate_abundance_or_richness_metric(
            df=df,
            metric_name=metric,
            agg_function=pl.sum,
        )
        validate_alpha_diversity_calculations(df_res, metric_name=metric)

        logger.info("Total abundance calculations finished.")

        return df_res

    def calculate_mean_abundance(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate the mean abundance of all species in each group."""
        logger.info("Calculating mean species abundance.")

        metric = "Mean_abundance"
        df_res = self.calculate_abundance_or_richness_metric(
            df=df,
            metric_name=metric,
            agg_function=pl.mean,
        )
        validate_alpha_diversity_calculations(df_res, metric_name=metric)

        logger.info("Mean abundance calculations finished.")

        return df_res

    def calculate_species_richness(self, df: pl.DataFrame) -> pl.DataFrame:
        """Count the bumber of unique species in each group."""
        logger.info("Calculating species richness.")

        # Filter out non-occurences so we can just use count as agg function
        df = df.filter(pl.col("Effort_corrected_measurement") != 0)

        metric = "Species_richness"
        df_res = self.calculate_abundance_or_richness_metric(
            df=df,
            metric_name=metric,
            agg_function=pl.count,
        )
        validate_alpha_diversity_calculations(df_res, metric_name=metric)

        logger.info("Species richness calculations finished.")

        return df_res

    def calculate_shannon_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate the standard Shannon diversity index, as well as a modified
        version that take total abundance at a site into account.

        The formula for the Shannon index is: H = -sum(p_i * ln(p_i)), where
        p_i is the proportion of species i at the site. The modified Shannon
        index isgiven by: H' = ln(a + 1) * H, where a is the total abundance at
        the site.

        Args:
            - df: Dataframe with input data.

        Returns:
            - df_shannon: Dataframe with Shannon index and modified Shannon index,
                including their scaled versions.
        """
        logger.info("Calculating site-level Shannon index numbers.")

        # Reuse existing functions for total abundance and species richness
        df_tot_abund = self.calculate_total_abundance(df)
        df_richness = self.calculate_species_richness(df)

        # Join the total abundance and species richness data
        df = df.join(
            df_tot_abund.select(self.groupby_cols + ["Total_abundance"]),
            on=self.groupby_cols,
            how="inner",  # Inner join to filter down to only abundance sites
        ).join(
            df_richness.select(self.groupby_cols + ["Species_richness"]),
            on=self.groupby_cols,
            how="inner",
        )

        # Calculate relative abundance (p_i in the formula above)
        df = df.with_columns(
            pl.when(pl.col("Total_abundance") == 0)
            .then(0)
            .otherwise(
                pl.col("Effort_corrected_measurement") / pl.col("Total_abundance")
            )
            .alias("Relative_abundance")
        )

        # Calculate Shannon component, setting it to zero when there were either
        # one or zero species at the site
        df = df.with_columns(
            pl.when(pl.col("Species_richness") == 1)  # Handle single-taxon sites
            .then(0)
            .when(pl.col("Relative_abundance") == 0)  # From previous step
            .then(0)
            .otherwise(
                -pl.col("Relative_abundance") * pl.col("Relative_abundance").log()
            )
            .alias("Shannon_component")
        )

        # Sum Shannon components to get the Shannon index
        df_shannon = df.group_by(self.groupby_cols).agg(
            pl.sum("Shannon_component").alias("Shannon_index"),
            pl.first("Total_abundance"),
        )

        # Calculate a modified Shannon index
        df_shannon = df_shannon.with_columns(
            ((pl.col("Total_abundance") + 1).log() * pl.col("Shannon_index")).alias(
                "Modified_Shannon_index"
            )
        )

        # Scale Shannon indices
        df_shannon = self.scale_by_study_max(
            df_shannon, diversity_metric="Shannon_index"
        )
        df_shannon = self.scale_by_study_max(
            df_shannon, diversity_metric="Modified_Shannon_index"
        )

        validate_alpha_diversity_calculations(df_shannon, metric_name="Shannon_index")
        validate_alpha_diversity_calculations(
            df_shannon, metric_name="Modified_Shannon_index"
        )

        logger.info("Shannon index calculations finished.")

        return df_shannon

    def calculate_abundance_or_richness_metric(
        self,
        df: pl.DataFrame,
        metric_name: str,
        agg_function: Callable,
    ) -> pl.DataFrame:
        """
        Generalized function to calculate diversity metrics based on species
        abundance richness. This is also scaled by the max value within each
        study. It's called by the functions for each diversity metric below.

        Args:
            - df: Dataframe with input data.
            - metric_name: Name of the diversity metric to calculate.
            - agg_function: Aggregation function to compute the metric
                (e.g., pl.sum, pl.mean).

        Returns:
            - df_scaled: Dataframe with the metric and its scaled version.
        """
        logger.info(f"Calculating {metric_name}.")

        # Filter dataframe to only include abundance studies
        df = df.filter(pl.col("Diversity_metric_type") == "Abundance")

        # Calculate the metric for the given grouping
        df_metric = df.group_by(self.groupby_cols).agg(
            agg_function("Effort_corrected_measurement").alias(metric_name)
        )

        # Scale the metric within each study
        df_scaled = AlphaDiversityTask.scale_by_study_max(
            df_metric, diversity_metric=metric_name
        )

        logger.info(f"{metric_name} calculations finished.")

        return df_scaled

    @staticmethod
    def scale_by_study_max(df: pl.DataFrame, diversity_metric: str) -> pl.DataFrame:
        """
        Scale diversity metrics by dividing them by the maximum value within
        the study to which they belong. This is done to make the metrics
        comparable across studies, so that data can be pooled for training.

        Args:
            - df: Dataframe containing the diversity metric to be scaled.
            - diversity_metric: Column name of the metric to scale.

        Returns:
            - df_scaled: Dataframe with the scaled metric added as new column.
        """
        logger.info(f"Scaling {diversity_metric} by max values within studies.")

        # Calculate the max value within each study
        df_max = df.group_by("SS").agg(
            pl.max(diversity_metric).alias(f"Study_max_{diversity_metric}")
        )

        # Join the max values back to the original dataframe
        df_scaled = df.join(
            df_max.select(["SS", f"Study_max_{diversity_metric}"]),
            on="SS",
            how="left",
        )

        # Perform the scaling, including handling of zero-abundance studies
        df_scaled = df_scaled.with_columns(
            pl.when(pl.col(f"Study_max_{diversity_metric}") == 0)
            .then(0)
            .otherwise(
                pl.col(diversity_metric) / pl.col(f"Study_max_{diversity_metric}")
            )
            .alias(f"Scaled_{diversity_metric}")
        )

        logger.info(f"Finished scaling {diversity_metric}.")
        return df_scaled
