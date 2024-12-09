import os
import time
from datetime import timedelta

import polars as pl
from box import Box

from core.data.data_processing import (
    concatenate_predicts_datasets,
    create_site_coord_geometries,
)
from core.utils.general_utils import create_logger

# Load config file
script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "data_configs.yaml"))

logger = create_logger(__name__)


class PredictsProcessingTask:
    """
    Task for loading and concatenating PREDICTS datasets, and extracting
    sampling site coordinates as Point geometries.
    """

    def __init__(self) -> None:
        """
        Attributes:
            predicts_2016: Path to the PREDICTS 2016 data.
            predicts_2016: Path to the PREDICTS 2016 data.
            target_col_order: Target column order in concatenated dataframe.
            all_predicts_data: Output path for the concatenated dataframe.
            all_site_coords: Output path for geodataframe with site coords.
        """
        self.predicts_2016: str = configs.predicts.predicts_2016
        self.predicts_2022: str = configs.predicts.predicts_2022
        self.target_col_order: list[str] = configs.predicts.target_col_order
        self.all_predicts_data: str = configs.predicts.all_predicts_data
        self.all_site_coords: str = configs.predicts.all_site_coords

    def run_task(self) -> None:
        """
        Orchestrate the processing of PREDICTS data, incl. loading,
        concatenating, generating site coordinates and saving dataframes.

        This function performs the following steps
        - Loads the PREDICTS datasets for 2016 and 2022
        - Concatenates the datasets together
        - Generates a geodataframe with coordinates for each sampling site
        - Saves both dataframes to disk.

        TODO: Add asserts that check that the length of the generated frames
        are correct.
        """
        logger.info("Starting processing of PREDICTS data.")
        start = time.time()

        # Load the two PREDICTS datasets (2016 and 2022)
        read_kwargs = {"infer_schema_length": 100000, "null_values": ["NA"]}
        df_2016 = pl.read_csv(self.predicts_2016, **read_kwargs)
        df_2022 = pl.read_csv(self.predicts_2022, **read_kwargs)

        # Concatenate them together
        df_concat = concatenate_predicts_datasets(
            df_2016, df_2022, col_order=self.target_col_order
        )

        # Generate a geodataframe with coordinates for every sampling site
        gdf_site_coords = create_site_coord_geometries(df_concat)

        # Save both dataframes to disk
        df_concat.write_parquet(self.all_predicts_data)
        gdf_site_coords.to_file(self.all_site_coords)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"PREDICTS data processing finished in {runtime}.")
