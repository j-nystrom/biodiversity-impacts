import polars as pl

from core.data.raster_stats_task import (
    BioclimaticFactorsTask,
    PopulationDensityTask,
    TopographicFactorsTask,
)
from core.data.road_density_task import RoadDensityTask
from core.data.site_buffering_task import SiteBufferingTask


def test_raster_tasks_use_current_configuration_keys() -> None:
    """Raster tasks resolve inputs and outputs from the current YAML schema."""
    expected_settings = [
        (PopulationDensityTask, "pop_density", [1, 10, 50]),
        (BioclimaticFactorsTask, "bioclimatic", [1, 10]),
        (TopographicFactorsTask, "topographic", [1, 10]),
    ]

    for task_class, mode, polygon_sizes in expected_settings:
        task = task_class("run-folder")
        task.configure_mode(mode)

        assert task.polygon_sizes == polygon_sizes
        assert len(task.output_paths) == len(polygon_sizes)
        assert len(task.result_col_names) == len(task.raster_paths) * len(polygon_sizes)


def test_road_density_task_uses_matching_utm_buffers() -> None:
    """Road-density radii and polygon files are aligned in the same order."""
    task = RoadDensityTask("run-folder")

    assert task.polygon_sizes == [1, 10, 50]
    assert [path.rsplit("_", 1)[-1] for path in task.utm_site_polygons] == [
        "1km.shp",
        "10km.shp",
        "50km.shp",
    ]
    assert len(task.un_regions) == len(task.road_network_data)
    assert len(task.un_regions) == len(task.road_density_data)


def test_site_geometries_retain_un_region_for_road_density() -> None:
    """Buffered-site inputs retain the region required by the road task."""
    task = SiteBufferingTask("run-folder")
    df = pl.DataFrame(
        {
            "SSBS": ["site-1", "site-2"],
            "Longitude": [10.0, 20.0],
            "Latitude": [50.0, 60.0],
            "Sample_midpoint": ["2000-01-01", "2001-01-01"],
            "UN_region": ["Europe", "Europe"],
        }
    )

    result = task.create_site_coord_geometries(df)

    assert result["UN_region"].tolist() == ["Europe", "Europe"]
