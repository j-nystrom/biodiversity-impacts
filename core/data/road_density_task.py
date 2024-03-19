import os
import time
from datetime import timedelta

import geopandas as gpd
import pandas as pd
from box import Box

from core.data.data_processing import (
    Projections,
    intersect_sites_and_roads,
    split_multi_line_strings,
)
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "data_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class CalculateRoadDensityTask:
    """
    Task for calculating the road density inside sampling site polygons with
    varying radius.

    NOTE: Should change pandas to polars at some point.
    """

    def __init__(self) -> None:
        """
        Attributes:
            all_site_coords: Path to dataframe containing coordinates of all
                sampling sites.
            road_network_data: Path to shapefiles with road Linestrings.
            utm_site_polygons: Paths to site polygons in UTM format.
            un_regions: List of UN regions in the same order as the road data
                files, to restrict the sampling sites used in calculations.
            buffer_distances: Radii that were used in previous buffering.
            road_densities: Output paths for calculated densities per region.
        """
        self.all_site_coords: str = configs.predicts.all_site_coords
        self.road_network_data: list[str] = configs.geodata.roads.road_network_data
        self.utm_site_polygons: list[str] = configs.geodata.utm_site_polygons
        self.un_regions: list[str] = configs.geodata.roads.un_regions
        self.buffer_distances: list[int] = configs.geodata.buffer_distances
        self.road_density_data: list[str] = configs.geodata.roads.road_density_data

    def run_task(self) -> None:
        """
        Calculates the length of the intersection between each site polygon and
        any adjacent road segments. Each shapefile with regional data on roads
        is loaded; any MultiLineString is slit into individual LineStrings;
        these are projected into local UTM format; the intersection between
        sites in that region and all roads are calculated; and finally, one
        file with road densities (lengths) per region is saved.
        """
        logger.info("Starting road density calculation process.")
        start = time.time()

        # Instantiate a Projections class object
        proj = Projections()

        df_all_sites = gpd.read_file(self.all_site_coords)

        for region, input, output in zip(
            self.un_regions, self.road_network_data, self.road_density_data
        ):
            logger.info(f"Processing road data for {region}.")
            start_region = time.time()

            df_sites_reg = df_all_sites.loc[df_all_sites["UN_region"] == region]

            # Load the road network data for this region
            gdf_roads = gpd.GeoDataFrame(gpd.read_file(input)["geometry"])

            # Check if rows contain MultiLineStrings and split where needed
            gdf_roads = split_multi_line_strings(gdf_roads["geometry"])
            gdf_roads = gdf_roads.rename(columns={"geometry": "global_coord"})

            # Project each linestring to local UTM coordinates
            gdf_roads[["utm_coord", "epsg_code"]] = gdf_roads.apply(
                lambda row: proj.project_to_local_utm(row["global_coord"]),
                axis=1,
                result_type="expand",
            )

            # Iterate through each buffer radius, load the corresponding site
            # polygons and intersect with the roads
            df_region_res = pd.DataFrame(df_sites_reg["SSBS"])
            for dist, path in zip(self.buffer_distances, self.utm_site_polygons):
                gdf_all_sites = gpd.read_file(path)
                site_polygons = gdf_all_sites.loc[gdf_all_sites["UN_region"] == region][
                    "geometry"
                ]
                result = intersect_sites_and_roads(site_polygons, gdf_roads)
                df_region_res[f"Road_density_{dist}km"] = result

            # Save the final dataframe with all sites and densities to disk
            df_region_res.to_parquet(output)

            runtime_region = str(timedelta(seconds=int(time.time() - start_region)))
            logger.info(f"Processing for {region} finished in {runtime_region}")

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Road density calculation finished in {runtime}.")
