import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.tests.shared.validate_shared import (
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "feature_configs.yaml"))

logger = create_logger(__name__)


class GenerateFeaturesTask:
    """
    Task to generate the covariate / feature set for the model. The output of
    this task is used to construct the dataframes for both alpha diversity and
    beta diversity in the next steps of the pipeline. The features are based on
    all the data sources derived in the data pipeline and combined in the
    previous step. These features include land use and other human pressure
    variables, and environmental information such as bioclimatic and
    topographic variables.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            - run_folder_path: Folder for storing logs and certain outputs.
            - combined_data_path: Path to file with the combined dataframe
                containing all source data, generated in the previous step.
            - cols_to_keep: Columns to keep from the original PREDICTS data.
            - density_vars: Columns related to population and road density.
            - bioclimatic_vars: Columns related to bioclimatic information.
            - topographic_vars: Columns related to topographic information.
            - land_use_col_order: Desired order of land use types in final
                dataframe, sorted from least to most disturbed.
            - lui_col_order: Same, but also including use intensity.
            - secondary_veg_col_order: Same, but for the case where all
                secondary vegetation types are grouped together.
            - feature_data_path: Path to the final dataframe, later used to
                construct the alpha and beta diversity dataframes.
        """
        self.run_folder_path = run_folder_path
        self.combined_data_path: str = configs.combine_data.combined_data_path
        self.cols_to_keep: list[str] = configs.feature_generation.cols_to_keep
        self.density_vars: list[str] = configs.feature_generation.density_vars
        self.bioclimatic_vars: list[str] = configs.feature_generation.bioclimatic_vars
        self.topographic_vars: list[str] = configs.feature_generation.topographic_vars
        self.land_use_col_order: list[str] = (
            configs.feature_generation.land_use_col_order
        )
        self.lui_col_order: list[str] = configs.feature_generation.lui_col_order
        self.secondary_veg_col_order: list[str] = (
            configs.feature_generation.secondary_veg_col_order
        )
        self.feature_data_path: str = configs.feature_generation.feature_data_path

    def run_task(self) -> None:
        """
        Perform the following processing steps:
            - Generate dummy variables (one-hot encoding) for the categorical
                land-use data.
            - Combine land-use type and intensity into one column and generate
                dummy variables for this.
            - Handle special cases from the BII model where certain land-use
                types and intensities are grouped together.
            - Create new variables that combine different biogeographical
                variables (biomes, realms and ecoregions).
            - Do log, square root and cube root transformations for all
                population and road density variables.
            - Calculate the mean population and road density for each study, to
                be used as a control variable in the model (based on BII).
        """
        logger.info("Initiating pipeline to generate explanatory features.")
        start = time.time()

        # Read PREDICTS data and drop columns that are not needed
        validate_input_files(file_paths=[self.combined_data_path])
        df = pl.read_parquet(self.combined_data_path, columns=self.cols_to_keep)

        # Create dummy variables for categorical land-use related columns
        df = self.create_land_use_dummies(df)
        df = self.combine_land_use_intensity_columns(df)
        df = self.group_land_use_types_and_intensities(df)

        # Rescale WorldClim data (temperature and precipitation)
        df = self.rescale_continuous_covariates(df, variables=self.bioclimatic_vars)

        # Generate non-linear transformations for continuous variables
        # All variables are given sublinear transformations (log, sqrt), while
        # only bioclimatic and topographic variables are given exponential
        sublinear_vars = (
            self.density_vars + self.bioclimatic_vars + self.topographic_vars
        )
        exponential_vars = self.bioclimatic_vars + self.topographic_vars
        df, new_cols = self.transform_continuous_covariates(
            df, sublinear=sublinear_vars, exponential=exponential_vars
        )

        # TODO: Generate interaction terms (instead of model data task)

        # Calculate mean values for population and road density
        # NOTE: Should be expanded if we build an explanatory model, in which
        # case site values should be expressed relative to the study mean
        # values (also for bioclimatic and topographic variables)
        df = self.calculate_study_mean_densities(
            df, variables=self.density_vars + new_cols  # Include transformed cols
        )

        # Handle any invalid values in continuous columns
        all_continuous_vars = (
            self.density_vars + self.bioclimatic_vars + self.topographic_vars + new_cols
        )
        df = self.handle_invalid_values(df, columns=all_continuous_vars)

        # Create a custom species grouping logic that is used in the alpha and
        # beta diversity tasks
        df = self.create_custom_taxonomic_grouping(df)

        # Save the final dataframe to disk
        validate_output_files(
            file_paths=[self.feature_data_path], files=[df], allow_overwrite=True
        )
        df.write_parquet(self.feature_data_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Feature generation finished in {runtime}.")

    def create_land_use_dummies(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate dummy columns for each value in the 'Predominant_land_use'
        column.

        Args:
            - df: Dataframe containing a 'Predominant_land_use' column.
            - col_order: Order of the land-use dummy columns in the output df.

        Returns:
            - df_res: Updated df with land use dummy columns added.
        """
        logger.info("Creating land-use dummy columns.")

        # Generate dummy columns for the land-use column
        df_dummies_lu = self._create_dummy_columns(
            df=df,
            column_name="Predominant_land_use",
            prefix_to_strip="Predominant_land_use",
            col_order=self.land_use_col_order,
        )

        # Combine the original df with the dummy columns
        df_res = pl.concat([df, df_dummies_lu], how="horizontal")

        logger.info("Finished creating land-use dummy columns.")
        return df_res

    def combine_land_use_intensity_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create a combined land-use and land-use intensity column and generate
        dummy columns based on this.

        Args:
            - df: Containing 'Predominant_land_use' and 'Use_intensity'.

        Returns:
            - df_res: Updated df with combined and dummy columns added.
        """
        logger.info("Creating combined land use-intensity (LUI) dummy columns.")

        # Create a combined column for land use type and intensity
        df = df.with_columns(
            pl.concat_str(
                [pl.col("Predominant_land_use"), pl.col("Use_intensity")], separator="_"
            ).alias("LU_type_intensity")
        )

        # Generate dummy columns for the combined land-use and intensity column
        df_dummies_comb = self._create_dummy_columns(
            df=df,
            column_name="LU_type_intensity",
            prefix_to_strip="LU_type_intensity",
            col_order=self.lui_col_order,
        )

        # Combine the original df with the dummy columns
        df_res = pl.concat([df, df_dummies_comb], how="horizontal")

        logger.info("Finished creating land-use + intensity variables.")
        return df_res

    def group_land_use_types_and_intensities(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create certain special groupings of land-use types and intensities.
        This is based on the latest published version of the BII model, see
        De Palma et al (2019). This is mainly motivated by limited data in
        these land use types. It includes the following groupings:
            - All secondary vegetation types are combined into one column, but
                still with different levels of intensity.
            - Plantation forest is grouped with secondary vegetation.
            - Urban land use of all intensities are combined into one column.
            - Light and intense use of cropland and pasture are combined.

        Args:
            - df: Containing 'Predominant_land_use' and 'Use_intensity'.

        Returns:
            - df: Updated df with the special groupings added.
        """

        # Helper function to group secondary vegetation types
        def _secondary_veg_intensity(row: pl.String) -> str:
            if "secondary" in str(row).lower():
                new_row = (
                    "Secondary vegetation_" + str(row).split("_")[1]
                )  # keep intensity
                return new_row
            else:
                return row

        # Run the grouping of secondary vegetation, preserving intensity info
        df = df.with_columns(
            pl.col("LU_type_intensity")
            .map_elements(lambda row: _secondary_veg_intensity(row))
            .alias("Secondary_veg_intensity")
        )

        # Group plantation forest with secondary vegetation
        # The allocation depends on the intensity of the plantation forest
        df = df.with_columns(
            pl.when(pl.col("Secondary_veg_intensity").str.contains("(?i)plantation"))
            .then(
                pl.when(pl.col("Secondary_veg_intensity").str.contains("(?i)minimal"))
                .then(pl.lit("Secondary vegetation_Light use"))  # Minimal --> Light
                .otherwise(
                    pl.lit("Secondary vegetation_Intense use")
                )  # Other --> Intense
            )
            .otherwise(pl.col("Secondary_veg_intensity"))  # Non-plantation not changed
            .alias("Secondary_veg_intensity")
        )

        # Finally, create a secondary vegetation column without intensities
        df_res = df.with_columns(
            pl.when(
                pl.col("Predominant_land_use").str.contains("(?i)secondary|plantation")
            )
            .then(1)
            .otherwise(0)
            .alias("Secondary vegetation_All uses")
        )

        # Create dummy columns from the combined secondary vegetation column
        df_dummies_secondary_veg = self._create_dummy_columns(
            df=df,
            column_name="Secondary_veg_intensity",
            prefix_to_strip="Secondary_veg_intensity",
            col_order=self.secondary_veg_col_order,
        )

        # Combine the original dataframe with the dummy columns
        df_res = pl.concat([df, df_dummies_secondary_veg], how="horizontal")

        # Combine all intensities for urban, cropland and pasture
        # The urban variable is used in both BII models (alpha and beta),
        # while cropland and pasture are only used in the beta model
        df_res = df_res.with_columns(
            pl.when(pl.col("Predominant_land_use") == "Urban")
            .then(1)
            .otherwise(0)
            .alias("Urban_All uses")
        )
        df_res = df_res.with_columns(
            pl.when(pl.col("Predominant_land_use") == "Cropland")
            .then(1)
            .otherwise(0)
            .alias("Cropland_All uses")
        )
        df_res = df_res.with_columns(
            pl.when(pl.col("Predominant_land_use") == "Pasture")
            .then(1)
            .otherwise(0)
            .alias("Pasture_All uses")
        )

        # Combine light and intense use for cropland and pasture (used in the
        # BII alpha model)
        df_res = df_res.with_columns(
            pl.when(
                (pl.col("Predominant_land_use") == "Cropland")
                & (pl.col("Use_intensity").is_in(["Light use", "Intense use"]))
            )
            .then(1)
            .otherwise(0)
            .alias("Cropland_Light_Intense")
        )

        df_res = df_res.with_columns(
            pl.when(
                (pl.col("Predominant_land_use") == "Pasture")
                & (pl.col("Use_intensity").is_in(["Light use", "Intense use"]))
            )
            .then(1)
            .otherwise(0)
            .alias("Pasture_Light_Intense")
        )

        df_res = df_res.with_columns(
            pl.when(
                (pl.col("Predominant_land_use") == "Primary vegetation")
                & (pl.col("Use_intensity").is_in(["Light use", "Intense use"]))
            )
            .then(1)
            .otherwise(0)
            .alias("Primary vegetation_Light_Intense")
        )

        df_res = df_res.with_columns(
            pl.when(
                (pl.col("Predominant_land_use").str.contains("(?i)secondary"))
                & (pl.col("Use_intensity").is_in(["Light use", "Intense use"]))
            )
            .then(1)
            .otherwise(0)
            .alias("Secondary vegetation_Light_Intense")
        )

        return df_res

    def rescale_continuous_covariates(
        self,
        df: pl.DataFrame,
        variables: list[str],
    ) -> pl.DataFrame:
        """
        Rescale bioclimatic data from WorldClim. The original temperature data
        is in degrees C multiplied by 10 (to reduce filesize). Precipitation
        data in in mm. Both are rescaled by dividing them by 10.

        Args:
            - df: Dataframe containing the variables to be rescaled.
            - variables: List of column names to evaluate and rescale.

        Returns:
            - Updated Polars DataFrame with rescaled columns.
        """
        logger.info("Rescaling temperature and precipitation variables.")

        for var in variables:
            if "temp" in var.lower() or "precip" in var.lower():
                df = df.with_columns(
                    pl.when(pl.col(var).is_not_null())
                    .then(pl.col(var) / 10)
                    .otherwise(pl.col(var))
                    .alias(var)
                )
                logger.info(f"Rescaled variable: {var}")

        logger.info("Finished rescaling variables.")

        return df

    def transform_continuous_covariates(
        self,
        df: pl.DataFrame,
        sublinear: list[str],
        exponential: list[str],
    ) -> pl.DataFrame:
        """
        Applies a set of transformations to all continuous variables in the
        dataset and adds the transformed columns to the dataframe. The
        transformations include the natural log, square root and cube root, as
        well as square and exponential transformations.

        Args:
            - df: Dataframe containing all covariates to be transformed.
            - sublinear: List of variables to be transformed with sublinear
                transformations (log, sqrt, cbrt).
            - exponential: List of variables to be transformed with
                exponential transformations (square).

        Returns:
            - df: Updated df with transformed columns added.
        """
        logger.info("Creating transformations for density variables.")

        new_cols = []
        for col in sublinear:
            # Natural logarithm
            df = df.with_columns((pl.col(col) + 1).log().alias(f"{col}_log"))
            new_cols.append(f"{col}_log")

            # Square root
            df = df.with_columns(pl.col(col).sqrt().alias(f"{col}_sqrt"))
            new_cols.append(f"{col}_sqrt")

            # Cube root (for BII compatibility only)
            df = df.with_columns(pl.col(col).pow(1 / 3).alias(f"{col}_cbrt"))
            new_cols.append(f"{col}_cbrt")

        for col in exponential:
            # Square
            df = df.with_columns(pl.col(col).pow(2).alias(f"{col}_square"))
            new_cols.append(f"{col}_square")

        logger.info("Finished creating transformations.")

        return df, new_cols

    def create_custom_taxonomic_grouping(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create a custom species grouping logic that is used as a grouping
        variable in the alpha and beta diversity tasks, complementing standard
        taxonomic levels like Kingdom, Phylum, etc.

        Args:
            - df: Dataframe containing taxonomic data data.

        Returns:
            - df: Dataframe with a column representing the new species grouping.
        """
        df = df.with_columns(
            [
                pl.when(pl.col("Phylum") == "Arthropoda")
                .then(  # Split between insects and other arthropods
                    pl.when(pl.col("Class") == "Insecta")
                    .then(pl.lit("Insecta"))
                    .otherwise(pl.lit("Other Arthropoda"))
                )
                .when(pl.col("Phylum") == "Chordata")
                .then(  # Split birds and mammals
                    pl.when(pl.col("Class").is_in(["Aves", "Mammalia"]))
                    .then(pl.col("Class"))  # Combine amphibians and reptiles (?)
                    .when(pl.col("Class").is_in(["Amphibia", "Reptilia"]))
                    .then(pl.lit("Amphibia_Reptilia"))
                    .otherwise(pl.lit("Other Chordata"))
                )
                .when(pl.col("Phylum") == "Tracheophyta")
                .then(
                    pl.lit("Tracheophyta")
                )  # Keep all vascular plants as one group (?)
                .when(pl.col("Kingdom") == "Fungi")  # No splits at all here
                .then(pl.lit("Fungi"))
                .otherwise(
                    pl.lit("Other ") + pl.col("Kingdom")
                )  # Other animals and plants
                .alias("Custom_taxonomic_group")
            ]
        )

        return df

    @staticmethod
    def calculate_study_mean_densities(
        df: pl.DataFrame, variables: list[str]
    ) -> pl.DataFrame:
        """
        Calculate the within-study mean values for the selected variables, for
        each level of resolution that those covariates exist. Currently,
        population and road density columns are included. These can act as
        control variables on sampling bias due to higher accessibility.

        NOTE: Doing the same for bioclimatic and topographic variables should
        be considered, as they can act as control variables for environmental
        factors that might otherwise bias the inference. This is important if
        we attempt to build an explanatory model.

        Args:
            - df: Dataframe containing all columns in the 'variables' list.
            - variables: Columns for which within-study means should be
                calculated.

        Returns:
            - df_res: Updated df with the within-study mean columns added.
        """
        logger.info(f"Calculating study-mean values for {variables}")

        # Only do calculations for population and road density variables
        target_vars = [
            col
            for col in variables
            if "pop_density" in col.lower() or "road_density" in col.lower()
        ]

        # Calculate the mean values at different resolutions
        mean_expressions = [
            pl.col(col).mean().alias(f"Mean_{col.lower()}") for col in target_vars
        ]
        df_study_mean = df.group_by("SS").agg(mean_expressions)

        # Join the mean columns with the original dataframe
        df_res = df.join(df_study_mean, on="SS", how="left", validate="m:1")

        logger.info("Finished study-mean calculations.")

        return df_res

    @staticmethod
    def _create_dummy_columns(
        df: pl.DataFrame, column_name: str, prefix_to_strip: str, col_order: list[str]
    ) -> pl.DataFrame:
        """
        Create dummy variables for a given column, rename them and sort columns.

        Args:
            - df: Dataframe containing the target column.
            - column_name: Name of the column to create dummy variables for.
            - prefix_to_strip: Prefix to remove from dummy column names.
            - col_order: Desired order of the dummy columns in the output.

        Returns:
            - df_dummies: Dataframe with sorted and renamed dummy columns.
        """
        logger.info(f"Creating dummy columns for {column_name}.")

        # Generate dummy columns
        df_dummies = df.select(column_name).to_dummies(column_name)

        # Strip the specified prefix from column names
        old_cols = df_dummies.columns
        new_cols = [col.replace(f"{prefix_to_strip}_", "") for col in old_cols]
        df_dummies = df_dummies.rename(
            {old: new for old, new in zip(old_cols, new_cols)}
        )

        # Sort columns as per the specified order
        df_dummies = df_dummies[col_order]

        logger.info(f"Finished creating dummy columns for {column_name}.")
        return df_dummies

    @staticmethod
    def handle_invalid_values(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        """
        Imputes invalid (NaN, null, infinite) values in a specified list of
        continuous variables using a three-level fallback:
        1. Block-level mean ('SSB')
        2. Study-level mean ('SS')
        3. Global mean

        Parameters:
        - df: Input dataframe containing all continuous variables.
        - columns: List of column names to check and impute.

        Returns:
        - df: DataFrame with imputed values.
        """
        logger.info("Handling invalid values in specified (continuous) columns.")
        total_rows = df.height

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in the DataFrame.")
                continue

            invalid_mask = (
                pl.col(col).is_nan() | pl.col(col).is_null() | pl.col(col).is_infinite()
            )
            total_invalid = df.filter(invalid_mask).height

            if total_invalid == 0:
                logger.info(f"No invalid values found in column '{col}'.")
                continue

            logger.warning(
                f"Column '{col}' contains {total_invalid} invalid values. "
                "Imputing missing values based on block or study mean values."
            )

            # Step 1: Block-level imputation (SSB)
            valid_df = df.filter(~invalid_mask)
            block_means = valid_df.group_by("SSB").agg(
                pl.col(col).mean().alias("mean_block")
            )
            df = df.join(block_means, on="SSB", how="left")

            df = df.with_columns(
                pl.when(invalid_mask)
                .then(pl.col("mean_block"))
                .otherwise(pl.col(col))
                .alias(col)
            ).drop("mean_block")

            # Step 2: Study-level imputation (SS)
            invalid_mask = (
                pl.col(col).is_nan() | pl.col(col).is_null() | pl.col(col).is_infinite()
            )
            remaining_invalid = df.filter(invalid_mask).height

            if remaining_invalid > 0:
                study_means = (
                    df.filter(~invalid_mask)
                    .group_by("SS")
                    .agg(pl.col(col).mean().alias("mean_study"))
                )
                df = df.join(study_means, on="SS", how="left")

                df = df.with_columns(
                    pl.when(invalid_mask)
                    .then(pl.col("mean_study"))
                    .otherwise(pl.col(col))
                    .alias(col)
                ).drop("mean_study")

            # Check before global mean fallback
            nan_count = df.filter(pl.col(col).is_nan()).height
            null_count = df.filter(pl.col(col).is_null()).height
            inf_count = df.filter(pl.col(col).is_infinite()).height
            total_left = nan_count + null_count + inf_count

            if total_left > 0:
                logger.warning(
                    f"Imputation still incomplete for '{col}': "
                    f"{nan_count} NaNs ({nan_count / total_rows:.1%}), "
                    f"{null_count} NULLs ({null_count / total_rows:.1%}), "
                    f"{inf_count} Inf ({inf_count / total_rows:.1%}) remaining. "
                    "Falling back to global mean values."
                )

            # Step 3: Global mean fallback
            invalid_mask = (
                pl.col(col).is_nan() | pl.col(col).is_null() | pl.col(col).is_infinite()
            )
            still_invalid = df.filter(invalid_mask).height

            if still_invalid > 0:
                global_mean = df.filter(~invalid_mask).select(pl.col(col).mean()).item()
                df = df.with_columns(
                    pl.when(invalid_mask)
                    .then(pl.lit(global_mean))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

            # Final check
            nan_count = df.filter(pl.col(col).is_nan()).height
            null_count = df.filter(pl.col(col).is_null()).height
            inf_count = df.filter(pl.col(col).is_infinite()).height
            total_left = nan_count + null_count + inf_count

            if total_left > 0:
                logger.warning(
                    f"Imputation incomplete for '{col}': "
                    f"{nan_count} NaNs ({nan_count / total_rows:.1%}), "
                    f"{null_count} NULLs ({null_count / total_rows:.1%}), "
                    f"{inf_count} Inf ({inf_count / total_rows:.1%}) remaining."
                )
            else:
                logger.info(f"Successfully imputed all values in column '{col}'.")

        return df
