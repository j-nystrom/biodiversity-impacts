import os
import time
from datetime import timedelta
from typing import Union

import geopandas as gpd
import polars as pl
import shapely
from box import Box
from pyproj import CRS, Transformer
from shapely import LineString, Point, Polygon

from core.tests.data.validate_data import (
    validate_site_coordinate_data,
    validate_site_geometry_output,
)
from core.tests.shared.validate_shared import (
    validate_input_files,
    validate_output_files,
)
from core.utils.general_utils import create_logger

script_dir = os.path.dirname(os.path.abspath(__file__))
configs = Box.from_yaml(filename=os.path.join(script_dir, "data_configs.yaml"))

logger = create_logger(__name__)


class SiteBufferingTask:
    """
    Task for creating polygons around sampling sites. This involves performing
    local projection of site coordinates, buffering these sites into polygons
    and reprojection the polygons to global format. The reason for this is to
    achieve equal-sized polygons around each site, later used for extracting
    various features from raster data and shapefiles.

    NOTE: This can be applied to other input data than PREDICTS, as long as
    it contains site coordinates in a specified reference format.
    """

    def __init__(self, run_folder_path: str) -> None:
        """
        Attributes:
            - run_folder_path: Folder for storing logs and certain outputs.
            - predicts_data_path: Path to the concatenated PREDICTS dataset.
            - polygon_sizes_km: List of polygon sizes (radius, in km) that
                should be used in buffering.
            - site_coords_crs: Reference system for coordinates in df above.
            - site_coords_path: Output path for site coordinates (non-buffered)
                which is an interim output in this step.
            - global_polygon_paths: Output paths of polygons in global format.
            - utm_polygon_paths: Output paths of polygons in UTM format.

        Raises:
             - ValueError: If the site_coords_crs is not EPSG:4326.
        """
        self.run_folder_path = run_folder_path
        self.predicts_data_path: str = configs.predicts.merged_data_path
        self.polygon_sizes_km: list[int] = configs.site_geodata.polygon_sizes_km
        self.polygon_type: str = configs.site_geodata.polygon_type
        self.site_coords_crs: str = configs.site_geodata.site_coords_crs
        self.site_coords_path: str = configs.site_geodata.site_coords_path
        self.global_polygon_paths: list[str] = configs.site_geodata.global_polygon_paths
        self.utm_polygon_paths: list[str] = configs.site_geodata.utm_polygon_paths

        if self.site_coords_crs != "EPSG:4326":
            raise ValueError("Input coordinates must be in EPSG:4326 format.")

    def run_task(self) -> None:
        """
        Perform the following processing steps:
            - Sampling site coordinates are extracted from the concatenated
                PREDICTS dataframe, and saved as an interim output.
            - Coords are projected from global EPSG:4326 to local UTM format.
            - They are then buffered into polygons based on the specified
                polygon sizes in 'polygon_sizes_km'.
            - The polygon coordinates are reprojected into global format.
        """
        logger.info("Starting projection-buffering-reprojection of site coordinates.")
        start = time.time()

        # Read df with sites and extract site coordinates
        validate_input_files(file_paths=[self.predicts_data_path])
        df_with_sites = pl.read_parquet(self.predicts_data_path)
        gdf_coords = self.create_site_coord_geometries(df_with_sites)

        # Save the site coordinates to file
        validate_output_files(
            file_paths=[self.site_coords_path], files=[gdf_coords], allow_overwrite=True
        )
        gdf_coords.to_file(self.site_coords_path)

        # Instantiate a Projections object to be used row-wise on coords
        proj = Projections(input_crs=self.site_coords_crs)

        # Project each Point to local UTM and return UTM coords + EPSG codes
        # Rename geometry column for clarity, since there will be multiple ones
        # NOTE: Could potentially be optimized using zip and map
        logger.info("Performing projections to local UTM zones.")
        gdf_coords = gdf_coords.rename(columns={"geometry": "global_coords"})
        gdf_coords[["utm_coords", "epsg_code"]] = gdf_coords.apply(
            lambda row: proj.project_to_local_utm(row["global_coords"]),
            axis=1,
            result_type="expand",
        )
        logger.info("Finished local projections.")

        # Buffer polygons for each specified radius and append as new columns
        logger.info("Buffering site coordinates into polygons.")
        for dist in self.polygon_sizes_km:
            gdf_coords[f"utm_{dist}km"] = self.buffer_points_in_utm(
                gdf_coords["utm_coords"],
                polygon_size=dist,
                polygon_type=self.polygon_type,
            )
        logger.info("Finished buffering.")

        # Reproject the polygons to global coordinate format
        # NOTE: Could potentially be optimized using zip and map
        logger.info("Performing reprojections to global coordinates.")
        for dist in self.polygon_sizes_km:
            gdf_coords[f"glob_{dist}km"] = gdf_coords.apply(
                lambda row: proj.reproject_to_global(
                    row[f"utm_{dist}km"], row["epsg_code"]
                ),
                axis=1,
            )
        logger.info("Finished global reprojections. Writing output files.")

        # Save one shapefile for each buffer distance in UTM and global formats
        for dist, path in zip(self.polygon_sizes_km, self.global_polygon_paths):
            gdf_res = gpd.GeoDataFrame(
                gdf_coords[["SSBS", f"glob_{dist}km"]],
                geometry=f"glob_{dist}km",
            )
            # Save to file, using Fiona engine to avoid issues with missing CRS
            # This is not an issue as files are only used internally, and there
            # is no easy way of setting this for the UTM files
            validate_output_files(
                file_paths=[path], files=[gdf_res], allow_overwrite=True
            )
            gdf_res.to_file(path, engine="fiona")

        for dist, path in zip(self.polygon_sizes_km, self.utm_polygon_paths):
            gdf_res = gpd.GeoDataFrame(
                gdf_coords[["SSBS", f"utm_{dist}km"]],
                geometry=f"utm_{dist}km",
            )
            validate_output_files(
                file_paths=[path], files=[gdf_res], allow_overwrite=True
            )
            gdf_res.to_file(path, engine="fiona")

        runtime = str(timedelta(seconds=int(time.time() - start)))
        logger.info(f"Projection-buffering-reprojection finished in {runtime}.")

    def create_site_coord_geometries(self, df: pl.DataFrame) -> gpd.GeoDataFrame:
        """
        Generate a geodataframe with Point geometries for each unique site
        based on longitude and latitude.

        NOTE: This should be made more generic if expanding data to GBIF.

        Args:
            - df: Dataframe with sampling data containing longitude and
                latitude of sampling sites.

        Returns:
            - gdf_site_coords: Geodataframe with point coordinates for each
                sampling site.

        Raises:
            - ValueError: If the input dataframe is missing required columns or
                has rows with missing coordinates.
        """
        logger.info("Creating Point geometries for sampling site coordinates.")

        # Check that the input data is valid
        required_columns = ["SSBS", "Longitude", "Latitude"]
        validate_site_coordinate_data(df, required_columns)

        # Get the coordinates for each unique site and generate coord tuples
        df_long_lat = df.group_by("SSBS").agg(
            [
                pl.first("Longitude"),
                pl.first("Latitude"),
            ]
        )
        coordinates = zip(
            df_long_lat.get_column("Longitude").to_list(),
            df_long_lat.get_column("Latitude").to_list(),
        )

        # Create Point geometries for coordinates and put into dataframe
        geometry = [Point(x, y) for x, y in coordinates]
        gdf_coords = (
            gpd.GeoDataFrame(
                {"SSBS": df_long_lat.get_column("SSBS"), "geometry": geometry}
            )
            .set_crs(self.site_coords_crs)
            .sort_values("SSBS", ascending=True)
            .reset_index(drop=True)
        )

        # Validate the output GeoDataFrame
        validate_site_geometry_output(gdf_coords, self.site_coords_crs)

        logger.info("Finished creating Point geometries.")

        return gdf_coords

    def buffer_points_in_utm(
        self, points: gpd.GeoSeries, polygon_size: int, polygon_type: str = "square"
    ) -> gpd.GeoSeries:
        """
        Create a Polygon from Point coordinates, by creating a buffer around it
        according to the specified radius.

        Args:
            - points: Geoseries with points to be buffered into polygons.
            - polygon_size: The radius of the current buffer in km.
            - polygon_type: The shape of the buffered Polygon. Can be any of
                ['square', 'round', 'flat']. Defaults to 'square'.

        Returns:
            - utm_coords_buff: Polygons consisting of the buffered points.

        Raises:
            - ValueError: If polygon_type is not one of the specified options.
        """
        logger.info(
            f"Buffering Points into {polygon_type} polygons "
            f"with radius {polygon_size} km."
        )
        if polygon_type not in ["square", "round", "flat"]:
            raise ValueError(
                "'polygon_type' must be one of ['square', 'round', 'flat']"
            )

        # Buffer array of Points into the chosen size and type
        utm_coords_buffered = shapely.buffer(
            points, polygon_size * 1000, cap_style=polygon_type
        )
        logger.info("Finished buffering points.")

        return utm_coords_buffered


class Projections:
    """
    This class implements methods to project global coordinates into local UTM
    format, and to reproject them to a global format.

    The reason they are defined within a separate class, unlike the rest of the
    data processing functions, is that already created transformer objects are
    stored as instance attributes to speed up the processing.
    """

    def __init__(self, input_crs: str) -> None:
        """
        Attributes:
            - input_crs: Reference system for site coordinates being processed.
            utm_transformer_dict: For every new UTM zone code, the transformer
                object for local projection is stored, for future re-use when a
                new site in that zone is encountered.
            global_transformer_dict: Same, but for the reprojections.
        """
        self.input_crs = input_crs
        self.utm_transformer_dict: dict[str, Transformer] = {}
        self.global_transformer_dict: dict[str, Transformer] = {}

    def project_to_local_utm(
        self, geometry: Union[Point, LineString]
    ) -> tuple[Union[Point, LineString], str]:
        """
        Calculates the local UTM zone for a Point or LineString that is given
        in the global EPSG:4326 format, and then transforms the geometry
        coordinates from global to local UTM format. For now, input coordinates
        must be in EPSG:4326 format.

        Args:
            geometry: Coordinates of e.g. a sampling site or road segment.

        Returns:
            local_coords: The coordinates of the input geometry transformed to
                local UTM coordinates.
            epsg_code: The local EPSG code for this tranformation, used for
                reprojection in a later stage.

        Raises:
            TypeError: If the input geometry is not a Point or LineString.
            ValueError: If the input coordinates are not in EPSG:4326 format.
        """
        if not isinstance(geometry, (Point, LineString)):
            raise TypeError("geometry should be a Point or LineString")
        if not self.input_crs == "EPSG:4326":
            raise ValueError("Input coordinates must be in EPSG:4326 format.")

        # Get the coordinate values (based on first point for Linestrings)
        first_point = geometry.coords[0]
        long, lat = first_point[0], first_point[1]

        # Determine the UTM zone and hemisphere of these coordinates
        zone_number = int((long + 180) // 6) + 1  # Calculate UTM zone
        if lat < 0:  # Southern hemisphere
            epsg_code = f"EPSG:{32700 + zone_number}"
        else:  # Northern hemisphere
            epsg_code = f"EPSG:{32600 + zone_number}"

        # Check if the generate EPSG code is valid
        try:
            CRS.from_epsg(int(epsg_code.split(":")[1]))
        except Exception:
            raise ValueError(f"Invalid EPSG code generated: {epsg_code}")

        # Fetch existing or initialize and save a pyproj Transformer object
        # always_xy implies that the method expects coordinates as long-lat
        if epsg_code in self.utm_transformer_dict.keys():
            utm_transformer = self.utm_transformer_dict[epsg_code]
        else:
            utm_transformer = Transformer.from_crs(
                crs_from=self.input_crs, crs_to=epsg_code, always_xy=True
            )
            self.utm_transformer_dict[epsg_code] = utm_transformer

        # Perform transformation, with the approach depending on type of geometry
        if isinstance(geometry, Point):
            local_coords = Point(utm_transformer.transform(long, lat))

        elif isinstance(geometry, LineString):
            xx, yy = geometry.coords.xy
            xx_utm, yy_utm = utm_transformer.transform(xx, yy)
            local_coords = LineString([xy for xy in zip(xx_utm, yy_utm)])

        return local_coords, epsg_code

    def reproject_to_global(self, polygon: Polygon, epsg_code: str) -> Polygon:
        """
        Take a Polygon defined by local UTM coordinates and reproject it to
        global EPSG:4326 coordinates.

        Args:
            polygon: The buffered site Polygon that should be reprojected.
            epsg_code: The stored local EPSG code for this tranformation.

        Returns:
            global_polygon: The buffered site Polygon in global coordinates.
        """

        # Fetch existing or initialize new Transformer object for reprojection
        # always_xy implies that the method expects coordinates as long-lat
        if epsg_code in self.global_transformer_dict.keys():
            global_transformer = self.global_transformer_dict[epsg_code]
        else:
            global_transformer = Transformer.from_crs(
                crs_from=epsg_code, crs_to=self.input_crs, always_xy=True
            )
            self.global_transformer_dict[epsg_code] = global_transformer

        # Get long-lat coordinates of the Polygon and perform transformation
        xx, yy = polygon.exterior.coords.xy
        xx_global, yy_global = global_transformer.transform(xx, yy)

        # Create a new Polygon from the reprojected coordinates
        global_coords = [xy for xy in zip(xx_global, yy_global)]
        global_polygon = Polygon(global_coords)

        return global_polygon
