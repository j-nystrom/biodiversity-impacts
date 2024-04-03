from core.dags.dag_setup import BaseDAG, parse_args, run_dags
from core.data.predicts_task import PredictsProcessingTask
from core.data.raster_stats_task import PopulationDensityTask
from core.data.road_density_task import RoadDensityTask
from core.data.site_buffering_task import ProjectAndBufferTask
from core.features.abundance_task import AbundanceFeaturesTask
from core.features.combine_data_task import CombineDataTask
from core.model.model_data_task import ModelDataTask
from core.model.model_train_task import ModelTrainingTask

# from core.data.raster_stats_task import BioclimaticTask
# from core.data.raster_stats_task import ElevationTask


# from core.model.cross_validation_task import CrossValidationTask


class PredictsProcessingDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [PredictsProcessingTask, ProjectAndBufferTask]


class GeoProcessingDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [RoadDensityTask, PopulationDensityTask]


class AbundanceFeaturesDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [CombineDataTask, AbundanceFeaturesTask]


class ModelTrainingDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [ModelDataTask, ModelTrainingTask]


# Mapping of DAG command line names to their corresponding classes
dag_mapping = {
    "predicts": PredictsProcessingDAG,
    "geodata": GeoProcessingDAG,
    "abundance": AbundanceFeaturesDAG,
    "training": ModelTrainingDAG,
    # "crossval": CrossValidationDAG,
}


if __name__ == "__main__":
    args = parse_args()
    run_dags(args.dags, dag_mapping)
