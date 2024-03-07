import os
import time
from datetime import timedelta

import geopandas as gpd
from box import Box

from core.data.data_processing import Projections, buffer_points_in_utm
from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "data_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class ProjectAndBufferTask:
    """
    Task for local projection of site coordinates, buffering into polygons and
    reprojection to global format.
    """

    def __init__(self) -> None:
        """
        Attributes:
            all_site_coords: Path to dataframe containing coordinates of all
                sampling sites.
            buffer_distances: List of radii that should be used in buffering.
            glob_site_polygons: Output paths of the polygons in global format.
            utm_site_polygons: Output paths of the polygons in UTM format.
        """
        self.all_site_coords: str = configs.predicts.all_site_coords
        self.buffer_distances: list[int] = configs.geodata.buffer_distances
        self.glob_site_polygons: list[str] = configs.geodata.glob_site_polygons
        self.utm_site_polygons: list[str] = configs.geodata.utm_site_polygons

    def run_task(self) -> None:
        """
        Runs a sequence of functions to create polygons of different sizes from
        point coordinates representing different sampling sites. Coordinates
        are first projected from global EPSG:4326 to local UTM format. They are
        then buffered into polygons. Finally, the polygon coordinates are
        reprojected into the global format.
        """
        logger.info("Starting projection-buffering-reprojection of site coordinates.")
        start = time.time()

        # Instantiate a Projections class object
        proj = Projections()

        # Load the geodataframe with site coordinates from PREDICTS
        # Rename geometry column for clarity, since there will be multiple ones
        gdf = gpd.read_file(self.all_site_coords)
        gdf = gdf.rename(columns={"geometry": "global_coord"})

        # Project each Point to local UTM and return UTM coords + EPSG codes
        logger.info("Performing projections to local UTM zones.")
        gdf[["utm_coord", "epsg_code"]] = gdf.apply(
            lambda row: proj.project_to_local_utm(row["global_coord"]),
            axis=1,
            result_type="expand",
        )
        logger.info("Finished local projections.")

        # Buffer polygons for each specified radius and append as new columns
        # The list of distances are in km, hence the 1000 multiplication
        for dist in self.buffer_distances:
            gdf[f"utm_{dist}km"] = buffer_points_in_utm(
                gdf["utm_coord"],
                dist,
                polygon_type="square",
            )

        # Reproject the polygons to global coordinate format
        logger.info("Performing reprojections to global coordinates.")
        for dist in self.buffer_distances:
            gdf[f"glob_{dist}km"] = gdf.apply(
                lambda row: proj.reproject_to_global(
                    row[f"utm_{dist}km"], row["epsg_code"]
                ),
                axis=1,
            )
        logger.info("Finished global reprojections.")

        # Save one shapefile for each buffer distance in UTM and global formats
        for dist, path in zip(self.buffer_distances, self.glob_site_polygons):
            gdf_res = gpd.GeoDataFrame(
                gdf[["SSBS", "UN_region", f"glob_{dist}km"]], geometry=f"glob_{dist}km"
            )
            gdf_res.to_file(path)

        for dist, path in zip(self.buffer_distances, self.utm_site_polygons):
            gdf_res = gpd.GeoDataFrame(
                gdf[["SSBS", "UN_region", f"utm_{dist}km"]], geometry=f"utm_{dist}km"
            )
            gdf_res.to_file(path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Projection-buffering-reprojection finished in {runtime}.")
