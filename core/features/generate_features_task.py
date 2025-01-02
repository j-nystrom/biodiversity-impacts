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
        The following key processing steps are performed:
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

        # Create dummy variables for land-use related columns
        df = self.create_land_use_dummies(df)
        df = self.combine_land_use_intensity_columns(df)
        df = self.group_land_use_types_and_intensities(df)

        # Generate non-linear transformations for population and road density
        # NOTE: Consider doing this for bioclimatic and topographic variables
        df, new_cols = self.transform_continuous_covariates(df)

        # Calculate mean values for population and road density
        # NOTE: Consider bioclimate and topographic variables as controls
        df = GenerateFeaturesTask.calculate_study_mean_densities(
            df, variables=self.density_vars + new_cols  # Include transformed cols
        )

        # Save the final dataframe to disk
        validate_output_files(
            file_paths=[self.feature_data_path], files=[df], allow_overwrite=True
        )
        df.write_parquet(self.feature_data_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Feature generation finished in {runtime}.")

    def create_land_use_dummies(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate dummy columns for each value in the 'Predominant_land_use' column.

        Args:
            - df: Dataframe containing a 'Predominant_land_use' column.
            - col_order: Order of the land-use dummy columns in the output df.

        Returns:
            - df_res: Updated df with land use dummy columns added.
        """
        logger.info("Creating land-use dummy columns.")

        # Generate dummy columns for the land-use column
        df_dummies_lu = GenerateFeaturesTask._create_dummy_columns(
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
        df_dummies_comb = GenerateFeaturesTask._create_dummy_columns(
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
                with different levels of intensity.
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
        df_dummies_secondary_veg = GenerateFeaturesTask._create_dummy_columns(
            df=df,
            column_name="Secondary_veg_intensity",
            prefix_to_strip="Secondary_veg_intensity",
            col_order=self.secondary_veg_col_order,
        )

        # Combine the original dataframe with the dummy columns
        df_res = pl.concat([df, df_dummies_secondary_veg], how="horizontal")

        # Combine urban land use of all intensities
        df_res = df_res.with_columns(
            pl.when(pl.col("Predominant_land_use") == "Urban")
            .then(1)
            .otherwise(0)
            .alias("Urban_All uses")
        )

        # Combine light and intense use for cropland and pasture
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

        return df_res

    def transform_continuous_covariates(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Applies a set of transformations to all continuous variables in the
        dataset and adds the transformed columns to the dataframe. The
        transformations include log, square root and cube root.

        Args:
            - df: Dataframe containing all covariates to be transformed.
            - variables: The columns that should be transformed.

        Returns:
            - df: Updated df with transformed columns added.
        """
        logger.info("Creating transformations for density variables.")

        new_cols = []
        for col in self.density_vars:
            df = df.with_columns((pl.col(col) + 1).log().alias(f"{col}_log"))  # Log
            new_cols.append(f"{col}_log")
            df = df.with_columns(pl.col(col).sqrt().alias(f"{col}_sqrt"))  # Square root
            new_cols.append(f"{col}_sqrt")
            df = df.with_columns(
                pl.col(col).pow(1 / 3).alias(f"{col}_cbrt")
            )  # Cube root
            new_cols.append(f"{col}_cbrt")

        logger.info("Finished creating transformations.")

        return df, new_cols

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
        factors that might otherwise bias the inference.

        Args:
            - df: Dataframe containing all columns in the 'variables' list.
            - variables: Columns for which within-study means should be
                calculated.

        Returns:
            - df_res: Updated df with the within-study mean columns added.
        """
        logger.info(f"Calculating study-mean values for {variables}")

        # Calculate the mean values at different resolutions
        mean_expressions = [
            pl.col(col).mean().alias(f"Mean_{col.lower()}") for col in variables
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
