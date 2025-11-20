import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.tests.shared.validate_shared import (
    check_duplicates,
    get_unique_value_count,
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "data_configs.yaml"))

logger = create_logger(__name__)


class PredictsConcatenationTask:
    """
    Task for loading and concatenating the two PREDICTS datasets from 2016 and
    2022 and ensuring consistency between them. Each dataset contains a unique
    set of studies and sites.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            - run_folder_path: Folder for storing logs and key outputs.
            - predicts_2016_path: Path to the PREDICTS 2016 data.
            - predicts_2022_path: Path to the PREDICTS 2022 data.
            - infer_schema_length: Nb of rows for schema inference when loading CSVs.
            - output_col_order: Desired column order in concatenated dataframe.
            - merged_data_path: Output path for the concatenated dataframe.
        """
        self.run_folder_path = run_folder_path
        self.predicts_2016_path: str = configs.predicts.predicts_2016_path
        self.predicts_2022_path: str = configs.predicts.predicts_2022_path
        self.infer_schema_length: int = configs.predicts.infer_schema_length
        self.output_col_order: list[str] = configs.predicts.output_col_order
        self.merged_data_path: str = configs.predicts.merged_data_path

    def run_task(self) -> None:
        """
        Perform the following processing steps:
            - Load the PREDICTS datasets for 2016 and 2022.
            - Concatenate the datasets together, ensuring that all columns are
                consistent and in the correct order.
            - Validate the concatenated dataframe.
            - Save dataframe as a parquet file.
        """
        logger.info("Starting processing of PREDICTS data.")
        start = time.time()

        # Number of rows used to infer the schema, to speed up reading
        # This was implmented as there were some issues reading the files
        read_kwargs = {
            "infer_schema_length": self.infer_schema_length,
            "null_values": ["NA"],
        }

        # Read and concatenate the two PREDICTS datasets
        validate_input_files(
            file_paths=[self.predicts_2016_path, self.predicts_2022_path]
        )
        df_2016 = pl.read_csv(self.predicts_2016_path, **read_kwargs)
        df_2022 = pl.read_csv(self.predicts_2022_path, **read_kwargs)
        df_concat = self.concatenate_predicts_dataframes(df_2016, df_2022)

        logger.info(
            f"Shape of concatenated dataframe: {df_concat.shape[0]:,} rows,"
            f" {df_concat.shape[1]:,} columns | "
            f"Number of unique sites: {get_unique_value_count(df_concat, 'SSBS'):,}"
        )

        # Save dataframe to disk (overwriting the previous file if it exists)
        validate_output_files(
            file_paths=[self.merged_data_path], files=[df_concat], allow_overwrite=True
        )
        df_concat.write_parquet(self.merged_data_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"PREDICTS data processing finished in {runtime}.")

    def concatenate_predicts_dataframes(
        self,
        df_2016: pl.DataFrame,
        df_2022: pl.DataFrame,
    ) -> pl.DataFrame:
        """
        Concatenate the two PREDICTS datasets, ensuring identical column
        structures and specified order.

        Args:
            - df_2016: Dataframe with the PREDICTS dataset from 2016.
            - df_2022: Dataframe with the PREDICTS dataset from 2022.

        Returns:
            - df_concat: Concatenated dataframe with ordered columns.
        """
        logger.info("Concatenating the PREDICTS datasets.")

        # Find out if there are any columns that are not overlapping
        unique_cols_2016 = list(set(df_2016.columns) - set(df_2022.columns))
        unique_cols_2022 = list(set(df_2022.columns) - set(df_2016.columns))
        all_unique_cols = unique_cols_2016 + unique_cols_2022

        # Drop non-overlapping columns from both dataframes, and make sure we
        # have the same column order in both dataframes. None of the identified
        # columns are important for modeling or analysis, so can be dropped
        if all_unique_cols:
            logger.warning(
                f"Dropping non-overlapping columns: {all_unique_cols}\n"
                f"Columns only in 2016: {unique_cols_2016}\n"
                f"Columns only in 2022: {unique_cols_2022}"
            )
            df_2016 = df_2016.drop(all_unique_cols, strict=False)  # Ignore missing cols
            df_2022 = df_2022.drop(all_unique_cols, strict=False)

        # Use 2016 columns as reference for which columns to pick
        df_2022 = df_2022.select(df_2016.columns)

        # Append new data to the old df, then sort columns in the right order
        df_concat = pl.concat([df_2016, df_2022], how="vertical")
        df_concat = df_concat.select(self.output_col_order).sort(
            "SSBS", descending=False
        )

        # Check the results before returning
        self.validate_concatenate_predicts_dataframes(
            df_2016,
            df_2022,
            df_concat,
            col_order=self.output_col_order,
        )

        logger.info("Finished concatenating PREDICTS datasets.")

        return df_concat

    @staticmethod
    def validate_concatenate_predicts_dataframes(
        df_2016: pl.DataFrame,
        df_2022: pl.DataFrame,
        df_concat: pl.DataFrame,
        col_order: list[str],
    ) -> None:
        """Validate the outputs of the concatenation operations."""
        # Check for missing or extra columns in the inputted col_order list
        missing_cols = set(df_concat.columns) - set(col_order)
        if missing_cols:
            raise ValueError(
                f"Missing columns in 'col_order': {missing_cols}, specify all columns."
            )
        extra_cols = set(col_order) - set(df_concat.columns)
        if extra_cols:
            raise ValueError(
                f"Extra columns in 'col_order': {extra_cols}, remove from input list."
            )

        # Check for duplicate rows
        duplicate_count = check_duplicates(df_concat)
        if duplicate_count > 0:
            raise ValueError(f"Duplicate rows detected: {duplicate_count} rows.")

        # Check that the total number of rows adds up
        expected_rows = df_2016.shape[0] + df_2022.shape[0]
        if df_concat.shape[0] != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} rows, but got {df_concat.shape[0]} rows."
            )

        # Check that the total number of unique sites add up
        unique_sites_2016 = get_unique_value_count(df_2016, "SSBS")
        unique_sites_2022 = get_unique_value_count(df_2022, "SSBS")
        expected_unique_sites = unique_sites_2016 + unique_sites_2022
        actual_unique_sites = get_unique_value_count(df_concat, "SSBS")
        if actual_unique_sites != expected_unique_sites:
            raise ValueError(
                f"Unique site count mismatch: Expected {expected_unique_sites} sites, "
                f"but got {actual_unique_sites} sites."
            )
