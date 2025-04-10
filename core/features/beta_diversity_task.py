import os
import time
from datetime import timedelta

import gower
import numpy as np
import polars as pl
from box import Box
from sklearn.metrics.pairwise import haversine_distances

from core.tests.shared.validate_shared import (
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

# Load config file
script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "feature_configs.yaml"))

logger = create_logger(__name__)


class BetaDiversityTask:
    """
    Task to compute pairwise compositional similarity between study sites,
    based on the (minimally-used primary vegetation) reference sites vs. all
    other sites within each study. The metric implemented is the Bray–Curtis
    similarity, which is the equivalent of the abundance-version of the
    Sorensen–Dice index.

    The similarity index is defined as:
        S = 2 * Σ min(a_i, b_i) / (Σ a_i + Σ b_i)
    where a_i and b_i are the abundances of taxon i at sites A and B.

    NOTE: To implement other metrics, this class can be reused with minimal
    refactoring. The only change needed is to implement the new metric in a
    separate function and call it in the pairwise_similarity_scores method.
    Additionally, the current implementation only uses minimally used reference
    sites, but this can be changed to use all primary vegetation sites.

    The task also computes pairwise feature differences for all continuous
    pressure variables (e.g. population density, road density) and calculates
    spatial and environmental distances between sites.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            - run_folder_path: Folder for storing logs and certain outputs.
            - feature_data_path: Path to the shared feature data generated in
                previous task (shared between alpha and beta diversity tasks)
            - groupby_cols: 'SS', 'SSB' and 'SSBS' always used for grouping.
            - taxonomic_levels: The levels in the taxonomic hierarchy that
                should be used as groupby columns when calculating diversity at
                that particular level.
            - output_data_path: output_data_paths: Output path for the final
                dataframes for each grouping level.
            - density_vars:
        """
        self.run_folder_path = run_folder_path
        self.feature_data_path: str = configs.feature_generation.feature_data_path
        self.groupby_cols: list[str] = configs.diversity_metrics.groupby_cols
        self.taxonomic_levels: list[str] = configs.diversity_metrics.taxonomic_levels
        self.output_data_paths: dict[str, str] = (
            configs.diversity_metrics.beta.output_data_paths
        )
        self.density_vars = configs.feature_generation.density_vars
        self.bioclimatic_vars = configs.feature_generation.bioclimatic_vars
        self.topographic_vars = configs.feature_generation.topographic_vars
        self.land_use_col_order = configs.feature_generation.land_use_col_order
        self.lui_col_order = configs.feature_generation.lui_col_order
        self.secondary_veg_col_order = (
            configs.feature_generation.secondary_veg_col_order
        )
        self.environmental_dist_vars: list[str] = (
            configs.diversity_metrics.beta.environmental_dist_vars
        )

    def run_task(self) -> None:
        """
        Perform the following processing steps:
            - Compute pairwise compositional similarity (Bray–Curtis) between
                baseline sites (minimally-used primary vegetation) and all
                other sites, for every study that meets the filtering criteria
            - Compute pairwise feature differences for all continuous pressure
                variables (e.g. population density, road density)
            - Calculate spatial and environmental distances between sites
        """
        logger.info("Initiating calculation of beta diversity (Bray–Curtis).")
        start_time = time.time()

        # Read feature data from previous step
        validate_input_files(file_paths=[self.feature_data_path])
        df = pl.read_parquet(self.feature_data_path)

        # Get the first row of each site to get site-level attributes
        df_site_attr = df.group_by("SSBS").first()

        # Compute the median of Max_linear_extent_metres among all sites
        # WHY?
        median_extent_all_data = df_site_attr.get_column(
            "Max_linear_extent_metres"
        ).median()

        # Create a list of all features and drop this from the similarity frame
        # They are not needed for this step, but will be used later on
        all_predictors = (
            self.land_use_col_order
            + self.lui_col_order
            + self.secondary_veg_col_order
            + self.density_vars
            + self.bioclimatic_vars
            + self.topographic_vars
        )
        df_for_similarity = df.drop(all_predictors)

        # Iterate through all taxonomic grouping levels (there is one output
        # path for each)
        for i, path in enumerate(self.output_data_paths.values()):
            logger.info(f"Calculating at aggregation level: {self.groupby_cols}")

            # Filter down to studies that contain reference sites and meet
            # other criteria
            df_filtered = self.filter_studies(df_for_similarity)
            logger.info(
                f"After filtering, {df_filtered['SS'].n_unique()} studies remain."
            )

            # Get the list of included studies
            studies = df_filtered["SS"].unique().to_list()
            all_results = []

            # Iterate over the filtered studies, processing each one incrementally
            for study_id in studies:
                df_study = df_filtered.filter(pl.col("SS") == study_id)
                df_features_group = df_site_attr.filter(pl.col("SS") == study_id)

                # Compute pairwise similarity scores
                results = self.pairwise_similarity_scores(study_id, df_study)

                if results:  # Only process if valid site pairs exist in study
                    df_comp_similarity_group = pl.DataFrame(results)

                    # Compute feature differences and distances for this study
                    df_comp_similarity_group = self.pairwise_feature_differences(
                        df_comp_similarity_group, df_features_group
                    )
                    df_comp_similarity_group = self.calculate_spatial_distance(
                        df_comp_similarity_group,
                        median_extent_all_data,
                    )
                    df_comp_similarity_group = self.calculate_environmental_distance(
                        df_comp_similarity_group
                    )
                    # Drop the reference site columns, since they are not used
                    # for the modeling
                    df_comp_similarity_group = df_comp_similarity_group.drop(
                        [
                            col
                            for col in df_comp_similarity_group.columns
                            if col.endswith("_reference")
                        ]
                    )

                    # Store results from this study
                    all_results.append(df_comp_similarity_group)

            # Concatenate all processed results into a single DataFrame
            df_final: pl.DataFrame = pl.concat(all_results)
            df_final = df_final.rename({"Other_site": "SSBS"})

            validate_output_files(
                file_paths=[path], files=[df_final], allow_overwrite=True
            )
            df_final.write_parquet(path)

            logger.info(f"Finished beta calculations at level: {self.groupby_cols}")

            # Update the list of grouping columns for the next iteration
            if i < len(self.taxonomic_levels):
                self.groupby_cols.append(self.taxonomic_levels[i])
            else:
                break

        run_time = str(timedelta(seconds=int(time.time() - start_time)))
        logger.info(f"Beta diversity calculations finished in {run_time}.")

    def filter_studies(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filters studies to include:
          1) Single sampling effort across all sites in a study
          2) At least one minimally-used primary vegetation site
          3) More than one taxon surveyed
        Returns the filtered DataFrame.
        """
        logger.info(
            "Filtering studies based on sampling effort, land use, and taxon count."
        )

        # 1) Identify studies with consistent sampling effort
        # NOTE: This is based on the BII, but might not be strictly necessary
        # if we would use the effort corrected measurements
        single_effort_studies = (
            df.group_by("SS")
            .agg(pl.col("Sampling_effort").n_unique().alias("unique_effort"))
            .filter(pl.col("unique_effort") == 1)
            .get_column("SS")
            .to_list()
        )

        # 2) Find studies with at least one minimally-used primary vegetation site
        min_primary_studies = (
            df.filter(
                (pl.col("Predominant_land_use") == "Primary vegetation")
                & (pl.col("Use_intensity") == "Minimal use")
            )
            .get_column("SS")
            .unique()
            .to_list()
        )

        # 3) Identify studies with more than one taxon
        multi_taxa_studies = (
            df.group_by("SS")
            .agg(pl.col("Taxon_name_entered").n_unique().alias("unique_taxon"))
            .filter(pl.col("unique_taxon") > 1)
            .get_column("SS")
            .unique()
            .to_list()
        )

        # Intersection of the three sets
        studies_to_include = (
            set(single_effort_studies)
            & set(min_primary_studies)
            & set(multi_taxa_studies)
        )

        # Filter dataframe to only those studies
        df_filtered = df.filter(pl.col("SS").is_in(list(studies_to_include)))

        return df_filtered

    def pairwise_similarity_scores(
        self, study_id: str, df_study: pl.DataFrame
    ) -> list[dict]:
        """
        Within a single study (already filtered for consistent effort), compute
        the similarity score for each pair of a minimally-used primary
        vegetation site and another site.

        NOTE: Currently, only the Bray-Curtis similarity metric is implemented.
        If implementing other metrics, this function can be reused with minimal
        refactoring.

        Args:
            study_id: ID of the study being processed.
            df_study: Polars dataframe containing data for the study.

        Returns:
            List of results with a dictionary for each site-site pair, that is
                later used to create the similarity dataframe.
        """
        logger.info(f"Calculating pairwise similarity scores for study: {study_id}")

        # Identify baseline (minimally-used primary vegetation) sites
        min_primary_sites = (
            df_study.filter(
                (pl.col("Predominant_land_use") == "Primary vegetation")
                & (pl.col("Use_intensity") == "Minimal use")
            )
            .get_column("SSBS")
            .unique()
            .to_list()
        )
        # Get all the sites in the study, including reference sites
        all_sites = df_study.get_column("SSBS").unique().to_list()

        results = []
        for site_1 in min_primary_sites:
            for site_2 in all_sites:
                if site_1 != site_2:
                    score = self.calculate_bray_curtis(
                        df_study, study_id, site_1, site_2
                    )
                    results.append(score)

        logger.info(f"Finished calculating for study: {study_id}")

        return results

    def calculate_bray_curtis(
        self, df_study: pl.DataFrame, study_id: str, site_1: str, site_2: str
    ) -> dict:
        """
        Calculate the Bray-Curtis similarity metric between a pair of sites,
        where site_1 is a minimal primary vegetation site and site_2 is any
        other site from the same study. Sampling effort is consistent which
        implies that we can use the raw 'Measurement' instead of the effort
        corrected one.

        Args:
            df_study: DataFrame containing the study data.
            study_id: ID of the study being processed.
            site_1: ID of the first site (minimal primary vegetation).
            site_2: ID of the second site (any other site in the study).

        Returns:
            A dictionary with the Bray–Curtis score of the site pair.
        """
        # Select only the relevant data for each site
        df_site_1 = (
            df_study.filter(pl.col("SSBS") == site_1)
            .select(["SSBS", "Taxon_name_entered", "Measurement"])
            .sort("Taxon_name_entered")
        )
        df_site_2 = (
            df_study.filter(pl.col("SSBS") == site_2)
            .select(["SSBS", "Taxon_name_entered", "Measurement"])
            .sort("Taxon_name_entered")
        )

        # Calculate total abundance at each site, which is the denominator of
        # the Bray-Curtis formula
        s1_tot_abund = df_site_1.select(pl.col("Measurement").sum()).item()
        s2_tot_abund = df_site_2.select(pl.col("Measurement").sum()).item()

        # Handle cases with zero abundance
        if s1_tot_abund == 0 and s2_tot_abund == 0:
            bray_curtis_sim = np.nan
        elif s1_tot_abund == 0 or s2_tot_abund == 0:
            bray_curtis_sim = 0.0
        else:
            # Merge in case taxon sets differ, so we can line up the measurements
            # for each taxon
            df_merged = df_site_1.join(
                df_site_2, on="Taxon_name_entered", how="outer", suffix="_site2"
            ).fill_null(0)

            # Convert to NumPy for simple processing
            # Calculate the sum of the smaller taxon abundances, which is the
            # numerator of the Bray-Curtis formula
            arr_merged = np.column_stack(
                [
                    df_merged.get_column("Measurement"),
                    df_merged.get_column("Measurement_site2"),
                ]
            )
            min_abundance_sum = np.sum(np.min(arr_merged, axis=1))

            # Bray–Curtis similarity (1 == identical, 0 == no overlap)
            bray_curtis_sim = (2.0 * min_abundance_sum) / (s1_tot_abund + s2_tot_abund)

        return {
            "SS": study_id,
            "Primary_minimal_site": site_1,
            "Other_site": site_2,
            "Bray_Curtis_score": bray_curtis_sim,
        }

    def pairwise_feature_differences(
        self, df_comp_similarity: pl.DataFrame, df_features: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate pairwise feature differences of all continuous features
        between each site-reference site pair. These differences are used in
        addition to the features of the other site itself in the beta diversity
        modeling.

        Args:
            df_comp_similarity: DataFrame containing the pairwise similarity
                scores and identifiers for each site pair.
            df_features: DataFrame containing the features for each site, that
                will be used to calculate the differences.

        Returns:
            df_comp_similarity: DataFrame with additional columns for feature
                differences, linked to the other site.
        """
        logger.info("Calculating pairwise feature differences.")

        # Get all features of the "other" site
        df_comp_similarity = df_comp_similarity.join(
            df_features,
            left_on="Other_site",
            right_on="SSBS",
            how="left",
        )

        # Identify all continuous features, including transformations
        continuous_base_vars = (
            self.density_vars + self.bioclimatic_vars + self.topographic_vars
        )
        df_cols = df_features.columns
        all_continuous_vars = set()

        # For each base_col, find anything exactly named base_col or a
        # a transformation of it, which has the same base name
        for base_col in continuous_base_vars:
            for col in df_cols:
                if col == base_col or col.startswith(f"{base_col}_"):
                    all_continuous_vars.add(col)

        all_continuous_vars_list = list(all_continuous_vars)

        # Get continuous features for the reference site
        # Also get longitude and latitude for the sites to calculate spatial
        # distance in the next function
        df_comp_similarity = df_comp_similarity.join(
            df_features.select(
                ["SSBS"] + all_continuous_vars_list + ["Latitude", "Longitude"]
            ),
            left_on="Primary_minimal_site",
            right_on="SSBS",
            suffix="_reference",
        )

        # Calculate pairwise differences for all continuous features
        # The reference site values are subtracted from the other site values
        for col in all_continuous_vars_list:
            df_comp_similarity = df_comp_similarity.with_columns(
                (pl.col(f"{col}") - pl.col(f"{col}_reference")).alias(f"{col}_diff")
            )

        logger.info("Finished calculating pairwise feature differences.")

        return df_comp_similarity

    def calculate_spatial_distance(
        self,
        df_comp_similarity: pl.DataFrame,
        median_extent_all_data: float,
    ) -> pl.DataFrame:
        """
        Calculate the spatial distance between each site pair using the
        Haversine distance formula. Dividing the site-site distance by the
        median of the maximum linear extent of all sites is based on De Palma
        et al (2021). There is explanation in the paper, but is probably done
        to adjust distances to be between the edges of the sites, rather than
        the center. The unit of the distance is not expressed, but is most
        likely in meters since the linear extent is given in meters. This
        implmentation outputs a distance in meters.

        Args:
            df_comp_similarity: DataFrame containing the pairwise similarity
                scores and identifiers for each site pair.
            median_extent_all_data: Median of Max_linear_extent_metres among
                all sites.

        Returns:
            df_comp_similarity: DataFrame with an additional column for the
                Haversine distance and the log of the distance.
        """
        logger.info("Calculating spatial distances.")

        # Convert coordinates in degrees to radians
        lat_lon = np.radians(  # Other site
            np.column_stack(
                [
                    df_comp_similarity.get_column("Latitude").to_numpy(),
                    df_comp_similarity.get_column("Longitude").to_numpy(),
                ]
            )
        )
        lat_lon_ref = np.radians(  # Reference site
            np.column_stack(
                [
                    df_comp_similarity.get_column("Latitude_reference").to_numpy(),
                    df_comp_similarity.get_column("Longitude_reference").to_numpy(),
                ]
            )
        )

        # Vectorized haversine distance (returns distance in radians)
        dist_rad = haversine_distances(lat_lon, lat_lon_ref)
        dist_meters = np.diagonal(dist_rad) * 6371000  # Convert radians to meters

        # Create a new distance column in the dataframe
        df_comp_similarity = df_comp_similarity.with_columns(
            pl.when(pl.lit(dist_meters) > 0)
            .then(pl.lit(dist_meters) / median_extent_all_data)
            .otherwise(pl.lit(np.nan))
            .alias("Haversine_distance")
        )

        # Do log and cube root transformations to align with other features
        # from the GenerateFeaturesTask. Log is used in De Palma et al (2021)
        df_comp_similarity = df_comp_similarity.with_columns(
            ((pl.col("Haversine_distance") + 1).log()).alias("Haversine_distance_log")
        )
        df_comp_similarity = df_comp_similarity.with_columns(
            (pl.col("Haversine_distance") ** (1 / 3)).alias("Haversine_distance_cbrt")
        )

        # Drop reference coordinate columns
        df_comp_similarity = df_comp_similarity.drop(
            ["Latitude_reference", "Longitude_reference"]
        )

        logger.info("Finished calculating spatial distances.")

        return df_comp_similarity

    def calculate_environmental_distance(
        self, df_comp_similarity: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Calculate the Gower distance between two sites based on environmental
        variables. The Gower distance is a measure of dissimilarity between two
        entities and is suitable for mixed data types.

        Args:
            df_comp_similarity: DataFrame containing the attributes of each
                site pair needed for the calculation.

        Returns:
            - Updated DataFrame with Gower distance added, including
                transformations.
        """
        logger.info("Calculating environmental distances.")

        def _gower_distance(row: dict) -> float:
            # Construct 2-row array for Gower: [references site, other site]
            data = np.array(
                [
                    [row[f"{var}_reference"] for var in self.environmental_dist_vars],
                    [row[var] for var in self.environmental_dist_vars],
                ],
                dtype=float,
            )

            dist = gower.gower_matrix(data)[0, 1]
            return dist

        # Build struct of all required columns (reference + non-reference)
        env_columns = [pl.col(var) for var in self.environmental_dist_vars]
        ref_columns = [
            pl.col(f"{var}_reference") for var in self.environmental_dist_vars
        ]
        all_env_cols = env_columns + ref_columns

        # Apply the Gower function row-wise
        df_comp_similarity = df_comp_similarity.with_columns(
            pl.struct(all_env_cols)
            .map_elements(_gower_distance, return_dtype=pl.Float64)
            .alias("Gower_distance")
        )

        # Do log and cube root transformations to align with other features
        # from the GenerateFeaturesTask. Cbrt is used in De Palma et al (2021)
        df_comp_similarity = df_comp_similarity.with_columns(
            ((pl.col("Gower_distance") + 1).log()).alias("Gower_distance_log")
        )
        df_comp_similarity = df_comp_similarity.with_columns(
            (pl.col("Gower_distance") ** (1 / 3)).alias("Gower_distance_cbrt")
        )

        logger.info("Finished calculating environmental distances.")

        return df_comp_similarity
