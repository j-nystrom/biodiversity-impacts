import os
import re
import time
from datetime import timedelta

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterstats
from box import Box

from core.utils.general_utils import create_logger

# Load config file
script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "data_configs.yaml"))

logger = create_logger(__name__)


class LandUseFractionTask:
    """
    Class for calculating land-use fractions for sampling sites and years
    using pre-buffered polygons.
    """

    def __init__(self, run_folder: str) -> None:
        """
        Initialize paths and configurations for land-use fraction calculations.
        """
        self.all_site_coords: str = configs.site_geodata.site_coords_path
        self.global_polygon_paths = configs.site_geodata.global_polygon_paths
        self.input_raster_base_path = configs.lu_fractions.input_raster_base_path
        self.output_base_path = configs.lu_fractions.output_base_path
        self.lu_mapping = configs.lu_fractions.simplified_lu_mapping
        self.include_all_pixels = configs.lu_fractions.agg_settings.include_all_pixels

    def run_task(self) -> None:
        """
        Main entry point for calculating land-use fractions per site and year.
        """
        logger.info(
            "Starting land-use fraction calculations using pre-buffered polygons."
        )
        start = time.time()

        # Iterate through each buffered polygon shapefile (1km, 5km, etc.)
        for polygon_path in self.global_polygon_paths:
            match = re.search(r"(\d+)km", polygon_path)
            if match:
                buffer_size = match.group(1)
            else:
                raise ValueError(f"No buffer size found in path: {polygon_path}")
            logger.info(f"Processing polygons for buffer size: {buffer_size} km")

            # Load the polygons and extract site names
            gdf_polygons = gpd.read_file(polygon_path)
            sites = gdf_polygons["SSBS"]

            # Iterate over unique years in the polygons
            sampling_years = gdf_polygons["Year"].sort_values().unique()
            for year in sampling_years:
                logger.info(
                    f"Processing year: {year} for buffer size: {buffer_size} km"
                )
                year_start = time.time()

                # Load the raster for the current year
                raster_year = max(1992, year)  # Fallback to 1992 if year < 1992
                raster_path = f"{self.input_raster_base_path}{raster_year}.tif"

                # Compute raster stats
                polygon_counts = self.calculate_raster_fractions(
                    polygon_path=polygon_path, raster_path=raster_path
                )

                # Process stats to calculate land-use fractions
                results = []
                for idx, count in enumerate(polygon_counts):
                    simplified_counts = self.simplify_classes(count)

                    # Calculate fractions for each land-use class
                    total_pixels = sum(simplified_counts.values())
                    # Handle the case where there are no valid pixels
                    if total_pixels == 0:
                        fractions = {
                            f"{class_name}_{buffer_size}km": 0.0
                            for class_name in simplified_counts.keys()
                        }
                    else:
                        fractions = {
                            f"{class_name}_{buffer_size}km": count / total_pixels
                            for class_name, count in simplified_counts.items()
                        }

                    # Append results for the current site
                    results.append(
                        {
                            "SSBS": sites[idx],
                            "Year": year,
                            **fractions,
                            "Sum_fractions": sum(fractions.values()),  # Debugging
                            "Highest_fraction": max(fractions.values()),  # Debugging
                            "Total_pixels": total_pixels,  # Debugging
                        }
                    )

                # Convert results to DataFrame and save
                df_results = pd.DataFrame(results)
                output_path = f"{self.output_base_path}{year}_{buffer_size}km.parquet"
                df_results.to_parquet(output_path)

                runtime_year = str(timedelta(seconds=int(time.time() - year_start)))
                logger.info(f"Finished processing for year {year} in {runtime_year}.")

        runtime_total = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Land-use fraction calculations completed in {runtime_total}.")

    def calculate_raster_fractions(
        self,
        polygon_path: str,
        raster_path: str,
    ) -> list[dict]:
        """
        Compute class counts for raster pixels overlapping polygons.

        Args:
            polygon_path: Path to polygon shapefile with sampling sites.
            raster_path: Path to raster file containing data for extraction.

        Returns:
            result: List of dictionaries containing class counts, one per polygon.
        """
        nb_of_polygons = len(gpd.read_file(polygon_path))

        # Calculate zonal statistics
        polygon_counts = rasterstats.zonal_stats(
            vectors=polygon_path,
            raster=raster_path,
            stats=["count"],
            all_touched=False,
            raster_out=True,  # Include raw raster values for each polygon
        )

        # Process each polygon's raster values
        results = []
        for idx, polygon in enumerate(polygon_counts):
            try:
                if "mini_raster_array" not in polygon:
                    logger.warning(
                        f"Polygon {idx + 1}/{nb_of_polygons}: Missing "
                        f"'mini_raster_array' in zonal_stats output."
                    )
                    class_counts = {0: 0}  # No data
                else:
                    # Extract raw raster values for polygon, count class occurrences
                    raster_values = polygon[
                        "mini_raster_array"
                    ].compressed()  # Ignore nodata
                    if raster_values.size == 0:
                        logger.warning(
                            f"Polygon {idx + 1}/{nb_of_polygons}: "
                            f"No valid raster values found."
                        )
                        class_counts = {0: 0}  # No data
                    else:
                        class_counts = {
                            value: (raster_values == value).sum()
                            for value in np.unique(raster_values)
                        }
            except Exception as e:
                logger.error(
                    f"Error processing polygon {idx + 1}/{nb_of_polygons}: {e}"
                )
                class_counts = {0: 0}  # Default to no data for failed polygons

            # Append the class counts for this polygon to the results
            results.append(class_counts)

        # Check that we have the expected number of results
        if len(results) != nb_of_polygons:
            logger.warning(
                f"Expected {nb_of_polygons} results, but got {len(results)}."
            )

        return results

    def simplify_classes(self, class_counts: dict) -> dict:
        """
        Simplify raster land-use classes based on the mapping.

        Args:
            class_counts (dict): Dictionary of raw raster class counts,
                where keys are class values and values are counts.

        Returns:
            dict: Simplified land-use class counts.
        """
        # Initialize counts for each simplified class to zero
        simplified_counts = {class_name: 0 for class_name in self.lu_mapping.keys()}

        # Map raw classes to simplified classes and aggregate counts
        for class_name, lu_codes in self.lu_mapping.items():
            for raw_value in lu_codes:
                if raw_value in class_counts.keys():
                    simplified_counts[class_name] += class_counts[raw_value]

        return simplified_counts
