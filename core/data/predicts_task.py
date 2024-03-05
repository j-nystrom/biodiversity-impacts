import os
import time
from datetime import timedelta
from typing import List

import polars as pl
from box import Box

from core.data.data_processing import (
    concatenate_predicts_datasets,
    create_site_coord_geometries,
)
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "data_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class PredictsProcessingTask:
    """
    Task for loading and concatenating PREDICTS datasets, and extracting
    sampling site coordinates as Point geometries.
    """

    def __init__(self) -> None:
        pass

    def run_task(
        self,
        source_file_2016: str = configs.predicts.source_file_2016,
        source_file_2022: str = configs.predicts.source_file_2022,
        col_order: List[str] = configs.predicts.col_order,
        concat_data_output_path: str = configs.predicts.concat_data_output_path,
        site_coord_output_path: str = configs.predicts.site_coord_output_path,
    ) -> None:
        """
        Orchestrate the processing of PREDICTS data, incl. loading,
        concatenating, generating site coordinates and saving dataframes.

        This function performs the following steps:
        1. Loads the PREDICTS datasets for 2016 and 2022.
        2. Concatenates the datasets together.
        3. Generates a geodataframe with coordinates for each sampling site.
        4. Saves both dataframes to disk.

        Args:
            source_file_2016 (str):
            source_file_2022 (str):
            col_order (List[str]):
            concat_data_output_path (str):
            site_coord_output_path (str):

        Returns:
            None
        """
        logger.info("Starting processing of PREDICTS data.")
        start = time.time()

        # Load the two PREDICTS datasets (2016 and 2022)
        read_kwargs = {"infer_schema_length": 100000, "null_values": ["NA"]}
        df_2016 = pl.read_csv(source_file_2016, **read_kwargs)
        df_2022 = pl.read_csv(source_file_2022, **read_kwargs)

        # Concatenate them together
        df_concat = concatenate_predicts_datasets(df_2016, df_2022, col_order=col_order)

        # Generate a geodataframe with coordinates for every sampling site
        gdf_site_coords = create_site_coord_geometries(df_concat)

        # Save both dataframes to disk
        df_concat.write_parquet(concat_data_output_path)
        gdf_site_coords.to_file(site_coord_output_path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"PREDICTS data processing finished in {runtime}.")
