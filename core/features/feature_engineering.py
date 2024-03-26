import numpy as np
import polars as pl

from core.utils.general_utils import create_logger

logger = create_logger(__name__)


def interpolate_population_density(
    df: pl.DataFrame, pixel_resolution: str, year_intervals: list[tuple[int, int]]
) -> pl.DataFrame:
    """
    Interpolates population density values in between available years, as well
    as going back to the first year in the PREDICTS database. The growth rate
    for each interval is based on the start and end year of that intervale.
    The calculation assumes an exponential growth curve.

    Args:
        df: Dataframe containing one site id column ('SSBS') and a number of
            year columns. These should match with the year intervals.
        pixel_resolution: The resolutions of this population data.
        year_interval: The year intervals that population data needs to be
            interpolated between. The first year in PREDICTS is 1984 and the
            last is 2018. Population data is available 2000, 2005, 2010, 2015
            and 2020.

    Returns:
        df: Original df with additional columns for all interpolated years.
    """
    logger.info(f"Population density interpolation for resolution {pixel_resolution}.")

    # Helper function to calculate growth rate between two years
    def _calculate_growth_rate(
        df: pl.DataFrame, start_year: int, end_year: int
    ) -> pl.Series:
        rates = np.log(
            df.get_column(str(end_year)) / df.get_column(str(start_year))
        ) / (end_year - start_year)

        # If there are NaN or inf values, fill with zeros
        logger.warning(
            f"NaN or inf values in growth rate for {start_year}-{end_year}. \
                Setting growth rates to 0."
        )
        if rates.is_nan().any() or rates.is_infinite().any():
            rates = (
                pl.when(rates.is_nan() | rates.is_infinite()).then(0).otherwise(rates)
            )

        return rates

    # Extrapolate back to 1984 using the growth rate from 2000 to 2005
    r_backwards = _calculate_growth_rate(df, year_intervals[1][0], year_intervals[1][1])
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
        value_name=f"Pop_density_{pixel_resolution}",
    ).sort(["SSBS", "Year"])

    # Convert to datetime format
    df = df.with_columns(pl.col("Year").str.strptime(pl.Datetime, "%Y").dt.year())

    logger.info("Finished population density interpolation.")

    return df


def filter_out_insufficient_data_studies(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filters out observations where land-use or land-use intensity is not known.

    Args:
        df: Dataframe with all combined data from PREDICTS and other sources.

    Returns:
        df: Filtered version of the same dataframe.
    """
    logger.info("Filtering out n/a land-use and intensity.")

    # Remove cases where land use or intensity is not known
    df = df.filter(
        ~(
            (pl.col("Predominant_land_use") == "Cannot decide")
            | (pl.col("Use_intensity") == "Cannot decide")
        )
    )

    logger.info("Filtering completed.")

    return df


def calculate_study_mean_densities(
    df: pl.DataFrame, cols_to_incl: list[str]
) -> pl.DataFrame:
    """
    Calculates the within-study mean population and road density, for each
    level of resolution that those covariates exist. This can act as a control
    variable on sampling bias due to higher accessibility.

    Args:
        df: Dataframe with all combined data from PREDICTS and other sources.

    Returns:
        cols_to_incl: That density columns for which the within-study means
            should be calculated.
    """
    logger.info(f"Calculating study-mean values for {cols_to_incl}")

    # Calculate mean population density at different resolutions
    mean_expressions = [
        pl.col(col).mean().alias(f"Mean_{col.lower()}") for col in cols_to_incl
    ]
    df_study_mean = df.group_by("SS").agg(mean_expressions)

    # Join the mean columns with the original dataframe
    df_res = df.join(df_study_mean, on="SS", how="left", validate="m:1")

    logger.info("Finished study-mean calculations.")

    return df_res


def create_land_use_dummies(
    df: pl.DataFrame,
    col_order: list[str],
) -> pl.DataFrame:
    """
    Generates dummy columns for each value in the land-use column, to be used
    as features in the model.

    Args:
        df: Dataframe containing 'Predominant_land_use' column.
        col_order: The order of the land-use dummy columns in the output df.

    Returns:
        df_res: Updated df with dummy columns added.

    """
    logger.info("Creating land-use dummy columns.")

    # Generate dummy columns for the original land-use column
    df_dummies_lu = df.select("Predominant_land_use").to_dummies("Predominant_land_use")

    # Strip 'Predominant_land_use' from column names
    old_cols = df_dummies_lu.columns
    new_cols = [col.replace("Predominant_land_use_", "") for col in old_cols]

    # Create a column mapping dictionary and rename columns
    mapping_dict = {old_col: new_col for old_col, new_col in zip(old_cols, new_cols)}
    df_dummies_lu = df_dummies_lu.rename(mapping_dict)

    # Sort the columns in the logical order
    df_dummies_lu = df_dummies_lu[col_order]

    # Join the dummy columns with the original dataframe
    df_res = pl.concat([df, df_dummies_lu], how="horizontal")

    logger.info("Finished creating LUI columns.")

    return df_res


def combine_land_use_intensity_columns(
    df: pl.DataFrame, col_order: list[str]
) -> pl.DataFrame:
    """
    Creates a combined land-use and land-use intensity column to capture the
    characteristics of the sampling site. From this column, a set of dummy
    variables columns are created, that can be used in the model.

    Args:
        df: Dataframe containing 'Predominant_land_use' and 'Use_intensity'
            columns.
        col_order: The order of the combined dummies in the output df.

    Returns:
        df_res: Updated df with combined and dummy columns added.

    """
    logger.info("Creating combined land use-intensity (LUI) dummy columns.")

    # Create a column that combines land use type and intensity
    df = df.with_columns(
        pl.concat_str(
            [pl.col("Predominant_land_use"), pl.col("Use_intensity")], separator="_"
        ).alias("LU_type_intensity")
    )

    # Create dummy columns from the combined column
    df_dummies_comb = df.select("LU_type_intensity").to_dummies("LU_type_intensity")

    # Strip 'LU_type_intensity' from column names
    old_cols = df_dummies_comb.columns
    new_cols = [col.replace("LU_type_intensity_", "") for col in old_cols]

    # Create a column mapping dictionary and rename columns
    mapping_dict = {old_col: new_col for old_col, new_col in zip(old_cols, new_cols)}
    df_dummies_comb = df_dummies_comb.rename(mapping_dict)

    # Sort the columns in the logical order
    df_dummies_comb = df_dummies_comb[col_order]

    # Join the dummy columns with the original dataframe
    df_res = pl.concat([df, df_dummies_comb], how="horizontal")

    logger.info("Finished creating LUI columns.")

    return df_res


def group_land_use_types_and_intensities(
    df: pl.DataFrame, col_order: list[str]
) -> pl.DataFrame:
    """
    Creates certain special groupings of land-use types and intensities:
    1. Urban land use of all intensities are combined into one column.
    2. All secondary vegetation types are combined into one column, but with
    different levels of intensity.
    3. Plantation forest is grouped with secondary vegetation.

    Args:
        df: Dataframe containing 'Predominant_land_use' and 'Use_intensity'
            columns.
        col_order: The order of the combined dummies in the output df.

        Returns:
            df: Updated df with the special groupings added.
    """
    # Combine urban land use of all intensities
    df = df.with_columns(
        pl.when(pl.col("Predominant_land_use") == "Urban")
        .then(1)
        .otherwise(0)
        .alias("Urban_All uses")
    )

    # Combine all secondary vegetation types
    def _secondary_veg_intensity(row: pl.String) -> str:
        if "secondary" in str(row).lower():
            new_row = "Secondary vegetation_" + str(row).split("_")[1]
            return new_row
        else:
            return row

    df = df.with_columns(
        pl.col("LU_type_intensity")
        .map_elements(lambda row: _secondary_veg_intensity(row))
        .alias("Secondary_veg_intensity")
    )

    # Group plantation forest with secondary vegetation
    df = df.with_columns(
        pl.when(pl.col("Secondary_veg_intensity").str.contains("(?i)plantation"))
        .then(
            pl.when(pl.col("Secondary_veg_intensity").str.contains("(?i)minimal"))
            .then(pl.lit("Secondary vegetation_Light use"))
            .otherwise(pl.lit("Secondary vegetation_Intense use"))
        )
        .otherwise(pl.col("Secondary_veg_intensity"))
        .alias("Secondary_veg_intensity")
    )

    # Create dummy columns from the combined secondary vegetation column
    df_dummies = df.select("Secondary_veg_intensity").to_dummies(
        "Secondary_veg_intensity"
    )

    old_cols = df_dummies.columns
    new_cols = [col.replace("Secondary_veg_intensity_", "") for col in old_cols]
    mapping_dict = {old_col: new_col for old_col, new_col in zip(old_cols, new_cols)}
    df_dummies = df_dummies.rename(mapping_dict)

    df_dummies = df_dummies[col_order]
    df = pl.concat([df, df_dummies], how="horizontal")

    # Combine light and intense use for cropland and pasture
    df = df.with_columns(
        pl.when(
            (pl.col("Predominant_land_use") == "Cropland")
            & (pl.col("Use_intensity").is_in(["Light use", "Intense use"]))
        )
        .then(1)
        .otherwise(0)
        .alias("Cropland_Light_Intense")
    )

    df = df.with_columns(
        pl.when(
            (pl.col("Predominant_land_use") == "Pasture")
            & (pl.col("Use_intensity").is_in(["Light use", "Intense use"]))
        )
        .then(1)
        .otherwise(0)
        .alias("Pasture_Light_Intense")
    )

    return df


def transform_continuous_covariates(
    df: pl.DataFrame, continuous_vars: list[str]
) -> pl.DataFrame:
    """
    Applies a set of transformations to all continuous variables in the dataset
    and adds the transformed columns to the dataframe.

    Args:
        df: Dataframe containing all covariates that should be transformed.
        continuous_vars: The columns that should be transformed.

    Returns:
        df_res: Updated df with transformed columns added.
    """
    logger.info("Creating transformations for continuous variables.")

    new_cols = []
    for col in continuous_vars:
        df = df.with_columns((pl.col(col) + 1).log().alias(f"{col}_log"))
        new_cols.append(f"{col}_log")
        df = df.with_columns(pl.col(col).sqrt().alias(f"{col}_sqrt"))
        new_cols.append(f"{col}_sqrt")
        df = df.with_columns(pl.col(col).pow(1 / 3).alias(f"{col}_cbrt"))
        new_cols.append(f"{col}_cbrt")

    logger.info("Finished creating transformations.")

    return df, new_cols


def calculate_scaled_abundance(
    df: pl.DataFrame, groupby_cols: list[str]
) -> pl.DataFrame:
    """
    Calculates the total species abundance for each sampling site ('SSBS') at
    a level of granularity defined by the grouping column input. This can range
    from all species through Kingdom -> Phylum -> Class -> Order -> Family.
    This abundance number is then scaled, by dividing with the maximum value
    within that study.

    Args:
        df: Dataframe with all combined data from PREDICTS and other sources.
        groupby_cols: The columns that together make up the groupby key.

    Returns:
        df_scaled: Updated df with abundance and scaled abundance columns.
    """
    logger.info("Calculating site-level scaled abundance numbers.")

    # Filter dataframe to only contain abundance numbers
    df = df.filter(pl.col("Diversity_metric_type") == "Abundance")

    # Calculate abundance for this level of grouping
    df_abundance = df.group_by(groupby_cols).agg(
        pl.sum("Effort_corrected_measurement").alias("Abundance")
    )

    # Calculate the max abundance within each study at this grouping level
    df_study_max = df_abundance.group_by("SS").agg(
        pl.max("Abundance").alias("Study_max_abundance")
    )

    # Join the dataframes together
    df_abundance = df_abundance.join(
        df_study_max.select(["SS", "Study_max_abundance"]), on="SS", how="left"
    )

    # Perform max scaling
    df_scaled = df_abundance.with_columns(
        (pl.col("Abundance") / pl.col("Study_max_abundance")).alias(
            "Max_scaled_abundance"
        )
    )

    logger.info("Abundance calculations finished.")

    return df_scaled
