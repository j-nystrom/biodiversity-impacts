import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.utils.general_utils import create_logger

# Load config file
script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "data_configs.yaml"))

logger = create_logger(__name__)


class PredictsConcatenationTask:
    """
    Task for loading and concatenating the two PREDICTS datasets from 2016 and
    2022 and ensuring consistency between them.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            run_folder_path: Folder for storing logs and certain outputs.
            predicts_2016_path: Path to the PREDICTS 2016 data.
            predicts_2022_path: Path to the PREDICTS 2022 data.
            output_col_order: Desired column order in concatenated dataframe.
            merged_data_path: Output path for the concatenated dataframe.
        """
        self.run_folder_path = run_folder_path
        self.predicts_2016_path: str = configs.predicts.predicts_2016_path
        self.predicts_2022_path: str = configs.predicts.predicts_2022_path
        self.output_col_order: list[str] = configs.predicts.output_col_order
        self.merged_data_path: str = configs.predicts.merged_data_path

    def run_task(self) -> None:
        """
        The following key processing steps are performed:
        - Load the PREDICTS datasets for 2016 and 2022
        - Concatenate the datasets together, ensuring that all columns are
            consistent and in the correct order
        - Save dataframe as a parquet file
        """
        logger.info("Starting processing of PREDICTS data.")
        start = time.time()

        # Number of rows to read to infer schema, to speed up reading
        read_kwargs = {"infer_schema_length": 100000, "null_values": ["NA"]}

        # Read and concatenate the two PREDICTS datasets
        df_2016 = pl.read_csv(self.predicts_2016_path, **read_kwargs)
        df_2022 = pl.read_csv(self.predicts_2022_path, **read_kwargs)
        df_concat = self.concatenate_predicts_dataframes(
            df_2016, df_2022, col_order=self.output_col_order
        )

        # Save dataframe to disk
        df_concat.write_parquet(self.merged_data_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"PREDICTS data processing finished in {runtime}.")

    def concatenate_predicts_dataframes(
        self, df_2016: pl.DataFrame, df_2022: pl.DataFrame, col_order: list[str]
    ) -> pl.DataFrame:
        """
        Concatenate the two PREDICTS datasets, ensuring identical column
        structures and specified order.

        Args:
            df_2016: Dataframe with the PREDICTS dataset from 2016.
            df_2022: Dataframe with the PREDICTS dataset from 2022.
            col_order: Desired column order in concatenated dataframe.

        Returns:
            df_concat: Concatenated dataframe with ordered columns.

        Raises:
            ValueError: If 'col_order' has extra or missing columns compated to
                the concatenated dataframe.
        """
        logger.info("Concatenating the PREDICTS datasets.")

        # Find out if there are any columns that are not overlapping
        # Note: These are not important for modeling or analysis
        unique_cols_2016 = list(set(df_2016.columns) - set(df_2022.columns))
        unique_cols_2022 = list(set(df_2022.columns) - set(df_2016.columns))
        all_unique_cols = unique_cols_2016 + unique_cols_2022

        # Drop non-overlapping columns from both dataframes, and make sure we
        # have the same column order in both dataframes
        if all_unique_cols:
            logger.warning(f"Dropping non-overlapping columns: {all_unique_cols}")
            df_2016 = df_2016.drop(all_unique_cols, strict=False)  # Ignore missing cols
            df_2022 = df_2022.drop(all_unique_cols, strict=False)

        df_2022 = df_2022.select(df_2016.columns)

        # Append new data to the old df, then sort columns in the right order
        # Ensure 'col_order' contains all columns present in the dataframe
        # and that 'col_order' doesn't contain any extra columns not present
        df_concat = pl.concat([df_2016, df_2022], how="vertical")
        missing_cols = set(df_concat.columns) - set(col_order)
        if missing_cols:
            raise ValueError(
                f"Missing columns in 'col_order': {missing_cols}, specify all columns."
            )

        extra_cols = set(col_order) - set(df_concat.columns)
        if extra_cols:
            raise ValueError(
                f"Extra columns in 'col_order': {extra_cols}, remove from list."
            )

        # Reorder the columns according to 'col_order', and sort data by 'SSBS'
        df_concat = df_concat.select(col_order)
        df_concat = df_concat.sort("SSBS", descending=False)

        logger.info(f"Shape of concatenated DataFrame: {df_concat.shape}")
        logger.info("Finished concatenating PREDICTS datasets.")

        return df_concat
