from core.dags.dag_setup import BaseDAG, parse_args, run_dags
from core.data.predicts_task import PredictsConcatenationTask
from core.data.raster_stats_task import (
    BioclimaticFactorsTask,
    PopulationDensityTask,
    TopographicFactorsTask,
)
from core.data.road_density_task import RoadDensityTask
from core.data.site_buffering_task import SiteBufferingTask
from core.features.alpha_diversity_task import AlphaDiversityFeatureTask
from core.features.combine_data_task import CombineDataTask
from core.model.crossval_task import CrossValidationTask
from core.model.model_train_task import ModelTrainingTask


class PredictsProcessingDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [PredictsConcatenationTask, SiteBufferingTask]


class PopulationDensityDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [PopulationDensityTask]


class RoadDensityDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [RoadDensityTask]


class BioclimaticFactorsDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [BioclimaticFactorsTask]


class TopographicFactorsDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [TopographicFactorsTask]


class AlphaDiversityFeatureDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [CombineDataTask, AlphaDiversityFeatureTask]


class ModelTrainingDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [ModelTrainingTask]


class CrossValidationDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [CrossValidationTask]


# Mapping of DAG command line names to their corresponding classes
# E.g. running python dags.py predicts will trigger the PredictsProcessingDAG
# TODO: This should be moved to a config file to separate code from configs
dag_mapping = {
    "predicts": PredictsProcessingDAG,
    "population": PopulationDensityDAG,
    "roads": RoadDensityDAG,
    "bioclimatic": BioclimaticFactorsDAG,
    "topographic": TopographicFactorsDAG,
    "alpha": AlphaDiversityFeatureDAG,
    "training": ModelTrainingDAG,
    "crossval": CrossValidationDAG,
}


if __name__ == "__main__":
    args = parse_args()
    run_dags(args.dags, dag_mapping)
