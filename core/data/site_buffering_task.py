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
        pass

    def run_task(
        self,
        site_coord_source_file: str = configs.geodata.site_coord_source_file,
        site_polygon_paths: list[str] = configs.predicts.site_polygon_paths,
        buffer_distances: list[int] = configs.geodata.buffer_distances,
    ) -> None:
        """
        Runs a sequence of functions to create polygons of different sizes from
        point coordinates representing different sampling sites. Coordinates
        are first projected from global EPSG:4326 to local UTM format. They are
        then buffered into polygons. Finally, the polygon coordinates are
        reprojected into the global format.

        Args:
            site_coord_source_file (str): Path to dataframe containing
                coordinates of sites.
            site_polygon_paths (List[str]):

        Returns:
            None
        """
        logger.info(
            "Starting projection-buffering-reprojection of sampling site coordinates."
        )
        start = time.time()

        # Instantiate a Projections class object
        proj = Projections()

        # Load the geodataframe with site coordinates from PREDICTS
        # Rename geometry column for clarity, since there will be multiple ones
        gdf = gpd.read_file(site_coord_source_file)
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
        for dist in buffer_distances:
            gdf[f"utm_{dist}km"] = buffer_points_in_utm(
                gdf["utm_coord"],
                dist * 1000,
                polygon_type="square",
            )

        # Reproject the polygons to global coordinate format
        for dist in buffer_distances:
            gdf[f"glob_{dist}km"] = gdf.apply(
                lambda row: proj.reproject_to_global(
                    row[f"utm_{dist}km"], row["epsg_code"]
                ),
                axis=1,
            )

        # Save one shapefile for each buffer distance
        for dist, path in zip(buffer_distances, site_polygon_paths):
            gdf_res = gdf[["SSBS", f"glob_{dist}km"]].rename(
                columns={f"glob_{dist}km": "geometry"}
            )
            gdf_res.to_file(path)

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Projection-buffering-reprojection finished in {runtime}.")
