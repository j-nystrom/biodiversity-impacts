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
    ) -> float:
        return np.log(df[str(end_year)] / df[str(start_year)]) / (end_year - start_year)

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
        ~(pl.col("Predominant_land_use") == "Cannot decide")
        | (pl.col("Use_intensity") == "Cannot decide")
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

    # Make a copy of the dataframe
    df = df.clone()

    # Calculate mean population density at different resolutions
    mean_expressions = [
        pl.col(col).mean().alias(f"Mean_{col.lower()}") for col in cols_to_incl
    ]
    df_study_mean = df.group_by("SS").agg(mean_expressions)

    # Join the mean columns with the original dataframe
    df_res = df.join(df_study_mean, on="SS", how="left", validate="m:1")

    logger.info("Finished study-mean calculations.")

    return df_res


def combine_land_use_and_intensity(
    df: pl.DataFrame, lui_col_order: list[str]
) -> pl.DataFrame:
    """
    Creates a combined land-use and land-use intensity column to capture the
    characteristics of the sampling site. From this column, a set of dummy
    variables columns are created, that can be used in the model.

    Args:
        df: Dataframe containing 'Predominant_land_use' and 'Use_intensity'
            columns.
        lui_col_order: The order of the generated columns in the output df.

    Returns:
        df_res: Updated df with combined and dummy columns added.

    """
    logger.info("Creating combined land use-intensity (LUI) dummy columns.")

    # Make a copy of the dataframe
    df = df.clone()

    # Create a column that combines land use type and intensity
    # Except for urban, where intensities are grouped together
    df = df.with_columns(
        pl.when(pl.col("Predominant_land_use").str.contains("Urban"))
        .then(pl.lit("Urban_All use"))  # pl.lit to set value, not access col
        .otherwise(
            pl.concat_str(
                [pl.col("Predominant_land_use"), pl.col("Use_intensity")], separator="_"
            )
        )
        .alias("LU_type_intensity")
    )

    # Create dummy columns from the combined column
    df_dummies = df.select("LU_type_intensity").to_dummies("LU_type_intensity")

    # Strip 'LU_type_intensity' from column names
    old_cols = df_dummies.columns
    new_cols = [col.replace("LU_type_intensity_", "") for col in old_cols]

    # Create a column mapping dictionary and rename columns
    mapping_dict = {old_col: new_col for old_col, new_col in zip(old_cols, new_cols)}
    df_dummies = df_dummies.rename(mapping_dict)

    # Sort the columns in the logical order
    df_dummies = df_dummies[lui_col_order]

    # Join the dummy columns with the original dataframe
    df_res = pl.concat([df, df_dummies], how="horizontal")

    logger.info("Finished creating LUI columns.")

    return df_res


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