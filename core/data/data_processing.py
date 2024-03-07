import os
from typing import List, Tuple, Union

import geopandas as gpd
import polars as pl
import rasterstats
import shapely
from box import Box
from pyproj import Transformer
from shapely import LineString, MultiLineString, Point, Polygon

from core.utils.general_utils import create_logger

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "data_configs.yaml")
configs = Box.from_yaml(filename=config_path)

logger = create_logger(__name__)


class Projections:
    """
    This class implements methods to project global coordinates into local UTM
    format, and to reproject them to a global format.

    The reason they are defined within a class unlike the rest of the data
    processing functions, is that already created transformer objects are
    stored in instance attributes to speed up the processing.
    """

    def __init__(self) -> None:
        """
        Attributes:
            utm_transformer_dict (dict): For every new UTM zone code, the
                transformer object for local projection is stored for future
                re-use when a new site in that zone is encountered.
            global_transformer_dict (dict): Same, but for the reprojections.
        """
        self.utm_transformer_dict: dict[str, Transformer] = {}
        self.global_transformer_dict: dict[str, Transformer] = {}

    def project_to_local_utm(
        self, geometry: Union[Point, LineString]
    ) -> Tuple[Union[Point, LineString], str]:
        """
        Calculates the local UTM zone for a Point or LineString that is given
        in the global EPSG:4326 format, and then transforms the geometry
        coordinates into from global to local UTM format.

        Args:
            geometry (Point or LineString): Coordinates of e.g. a sampling site
                or road segment.

        Returns:
            local_coords (Point or LineString): The coordinates of the input
                geometry transformed to local UTM coordinates.
            epsg_code (str): The local EPSG code for this tranformation, used
                for reprojection in a later stage.
        """
        assert isinstance(
            geometry, (Point, LineString)
        ), "geometry should be a Point or LineString"

        # Get the coordinate values (based on first point for Linestrings)
        first_point = geometry.coords[0]
        long, lat = first_point[0], first_point[1]

        # Determine the UTM zone and hemisphere of these coordinates
        zone_number = int((long + 180) // 6) + 1
        epsg_code = f"EPSG:{32700 + zone_number if lat < 0 else 32600 + zone_number}"

        # Fetch existing or initialize and save a pyproj Transformer object
        # always_xy implies that the method expects coordinates as long-lat
        if epsg_code in self.utm_transformer_dict.keys():
            utm_transformer = self.utm_transformer_dict[epsg_code]
        else:
            utm_transformer = Transformer.from_crs(
                crs_from="EPSG:4326", crs_to=epsg_code, always_xy=True
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
        Takes a Polygon defined by local UTM coordinates and reprojects it to
        global EPSG:4326 coordinates.

        Args:
            polygon (Polygon): The buffered site that should be reprojected.
            epsg_code (str): The stored local EPSG code for this tranformation.

        Returns:
            global_polygon (Polygon): The buffered site in global coordinates.
        """

        # Fetch existing or initialize new Transformer object for reprojection
        # always_xy implies that the method expects coordinates as long-lat
        if epsg_code in self.global_transformer_dict.keys():
            global_transformer = self.global_transformer_dict[epsg_code]
        else:
            global_transformer = Transformer.from_crs(
                crs_from=epsg_code, crs_to="EPSG:4326", always_xy=True
            )
            self.global_transformer_dict[epsg_code] = global_transformer

        # Get long-lat coordinates of the Polygon and perform transformation
        xx, yy = polygon.exterior.coords.xy
        xx_global, yy_global = global_transformer.transform(xx, yy)

        # Create a new Polygon from the reprojected coordinates
        global_coords = [xy for xy in zip(xx_global, yy_global)]
        global_polygon = Polygon(global_coords)

        return global_polygon


def buffer_points_in_utm(
    points: gpd.GeoSeries, buffer_dist: int, polygon_type: str = "square"
) -> gpd.GeoSeries:
    """
    Creates a Polygon from Point coordinates, by buffering according to the
    specified radius.

    Args:
        points (gpd.GeoSeries): A GeoSeries with all the points that should
            be buffered into Polygons.
        buffer_dist (int): Buffer radius expressed in kilometers.
        polygon_type (str): The shape of the buffered Polygon. Can be any
            of ['square', 'round', 'flat']. Defaults to 'square'.

    Returns:
        utm_coords_buff (gpd.GeoSeries): Polygons with the buffered points.
    """
    logger.info(f"Buffering Points into Polygons with radius {buffer_dist} km.")
    assert polygon_type in [
        "square",
        "round",
        "flat",
    ], "polygon_type must be one of ['square', 'round', 'flat']"

    # Buffer array of Points into the chosen size and type
    utm_coords_buffered = shapely.buffer(
        points, buffer_dist * 1000, cap_style=polygon_type
    )
    logger.info("Finished buffering points.")

    return utm_coords_buffered


def concatenate_predicts_datasets(
    df_2016: pl.DataFrame, df_2022: pl.DataFrame, col_order: List[str]
) -> pl.DataFrame:
    """
    Concatenate the two PREDICTS datasets, ensuring identical column
    structures and specified order.

    Args:
        df_2016 (pl.DataFrame): The PREDICTS dataset from 2016.
        df_2022 (pl.DataFrame): The PREDICTS dataset from 2022.
        col_order (List[str]): Column order in the concatenated dataframe.

    Returns:
        pl.DataFrame: Concatenated dataframe with ordered columns.

    Raises:
        ValueError: If 'col_order' has extra or missing columns from the
            concatenated dataframe.
    """
    logger.info("Concatenating the PREDICTS datasets.")

    # Find out if there are any columns that are not overlapping
    unique_cols_2016 = list(set(df_2016.columns) - set(df_2022.columns))
    unique_cols_2022 = list(set(df_2022.columns) - set(df_2016.columns))
    all_unique_cols = unique_cols_2016 + unique_cols_2022

    # Drop non-overlapping columns from both dataframes
    df_2016 = df_2016.drop(all_unique_cols)
    df_2022 = df_2022.drop(all_unique_cols)

    # Make sure we have the same column order in both
    df_2022 = df_2022.select(df_2016.columns)

    # Append new data to old and then sort the columns in the right order
    df_concat = pl.concat([df_2016, df_2022], how="vertical")

    # Ensure 'col_order' contains all columns present in the dataframe
    # and that 'col_order' doesn't contain any extra columns
    missing_cols = set(df_concat.columns) - set(col_order)
    if missing_cols:
        raise ValueError(f"Missing columns in 'col_order': {missing_cols}")

    extra_cols = set(col_order) - set(df_concat.columns)
    if extra_cols:
        raise ValueError(f"Extra columns in 'col_order': {extra_cols}")

    # Reorder the columns according to 'col_order', and sort data by 'SSBS'
    df_concat = df_concat.select(col_order)
    df_concat = df_concat.sort("SSBS", descending=False)

    logger.info("Finished concatenating datasets.")

    return df_concat


def create_site_coord_geometries(df_concat: pl.DataFrame) -> gpd.GeoDataFrame:
    """
    Generate a geodataframe with Point geometries for each unique site
    based on longitude and latitude, and adds UN region information for
    filtering in other tasks.

    Args:
        df_concat (pl.DataFrame): The concatenated PREDICTS data.

    Returns:
        gdf_site_coords (gpd.GeoDataFrame): Geodataframe with Point
            coordinates and region information for each sampling site.
    """
    logger.info("Creating Point geometries for sampling site coordinates.")

    # Get the coordinates for each unique site in the dataset
    df_long_lat = df_concat.groupby("SSBS").agg(
        [
            pl.col("Longitude").first().alias("Longitude"),
            pl.col("Latitude").first().alias("Latitude"),
        ]
    )

    # Generate coordinate tuples from the long-lat columns
    coordinates = zip(
        df_long_lat.get_column("Longitude").to_list(),
        df_long_lat.get_column("Latitude").to_list(),
    )

    # Create shapely Point geometries for all coordinates
    geometry = [Point(x, y) for x, y in coordinates]

    # Create a geodataframe containing site id and coordinates
    gdf_site_coords = gpd.GeoDataFrame(
        {"SSBS": df_long_lat.get_column("SSBS"), "geometry": geometry}
    )
    gdf_site_coords.crs = "EPSG:4326"

    # Add the UN region to enable filtering when working with the geodata
    df_region = (
        df_concat.groupby("SSBS").agg(pl.col("UN_region").first()).to_pandas()
    )  # Need to convert to pandas to be compatible with geopandas

    # Make sure column 'SSBS' is a string, then merge the dataframes, and
    # finally sort it
    gdf_site_coords = gdf_site_coords.merge(
        df_region, on="SSBS", how="left", validate="1:1"
    )
    gdf_site_coords = gdf_site_coords.sort_values("SSBS", ascending=True)

    logger.info("Finished creating Point geometries.")

    return gdf_site_coords


def calculate_raster_stats(
    polygon_path: str,
    raster_path: str,
    metric: List[str] = ["mean"],
    include_all_pixels: bool = True,
) -> List[float]:
    """
    Computes statistical metrics for raster pixels that overlap with polygons
    (representing sampling sites) that should be analyzed.

    Args:
        polygon_path (str): Path to polygon shapefile with sampling sites.
        raster_path (str): Path to raster file containing data for extraction.
        metric (List[str]): Statistical metrics to compute.
        include_all_pixels (bool): Whether to include all pixels that touch the
            polygon boundaries, or just pixels with center points within it.

    Returns:
        result (list): Computed values, one for each polygon in the shapefile.
    """

    # Calculate zonal statistics
    stats = rasterstats.zonal_stats(
        vectors=polygon_path,
        raster=raster_path,
        stats=metric,
        all_touched=include_all_pixels,
    )

    # Extract stats from each dictionary in the output list
    result = [x[metric] for x in stats]

    return result


def split_multi_line_strings(road_linestrings: gpd.GeoSeries) -> gpd.GeoDataFrame:
    """
    Checks a set of geometries that are expected to be Linestrings, and if
    necessary splits any geometries that turn out to be MultiLineStrings.

    Args:
        road_linestrings (gpd.GeoSeries): A set of geometries that are a mix
            of Linestrings and MultiLineStrings.

    Returns:
        split_linestrings (gpd.GeoDataFrame): Containing geometries that are
            only single Linestrings.
    """

    result = []
    for geometry in road_linestrings:
        if isinstance(geometry, MultiLineString):
            split_string = [LineString(string) for string in geometry.geoms]
            result += split_string
        elif isinstance(geometry, LineString):
            result.append(geometry)
        else:
            continue

    split_linestrings = gpd.GeoDataFrame(geometry=result)

    return split_linestrings


def intersect_sites_and_roads(
    site_polygons: gpd.GeoSeries, gdf_roads: gpd.GeoDataFrame
) -> list[float]:
    """
    Calculate the intersection between every sampling site polygon (with a
    certain radius) and the road segments that are present in the same region.
    The length of that intersection gives a measure of density.

    Args:
        site_polygons (gpd.GeoSeries): Site polygons for which the road density
            should be calculated.
        gdf_roads (gpd.GeoDataFrame): Road segments that form the basis of the
            density calculations.

    Returns:
        site_road_len (list[float]): List of density values for each site.
    """
    # Extract geoseries containing road linestrings
    road_linestrings = MultiLineString(gdf_roads["utm_coord"].tolist())

    # List for storing results
    site_road_len = []

    # Iterate through every site polygon of this size
    for polygon in site_polygons:
        # Calculate intersection between site polygon and all road linestrings
        intersect_len = shapely.intersection(polygon, road_linestrings).length
        site_road_len.append(intersect_len)

    return site_road_len
