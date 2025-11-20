from core.dags.dag_setup import BaseDAG, parse_args, run_dags
from core.data.land_use_fraction_task import LandUseFractionTask
from core.data.predicts_task import PredictsConcatenationTask
from core.data.raster_stats_task import (
    BioclimaticFactorsTask,
    PopulationDensityTask,
    TopographicFactorsTask,
)
from core.data.road_density_task import RoadDensityTask
from core.data.site_buffering_task import SiteBufferingTask
from core.features.alpha_diversity_task import AlphaDiversityTask
from core.features.beta_diversity_task import BetaDiversityTask
from core.features.combine_data_task import CombineDataTask
from core.features.generate_features_task import GenerateFeaturesTask
from core.model.model_data_task import ModelDataTask
from core.model.training_crossval_task import CrossValidationTask, ModelTrainingTask


# Below are all the DAG classes available in the codebase
class PredictsProcessingDAG(BaseDAG):
    """Concatenate PREDICTS data and create buffers around sampling sites."""

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [PredictsConcatenationTask, SiteBufferingTask]


class PopulationDensityDAG(BaseDAG):
    """Extract population density around sampling sites."""

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [PopulationDensityTask]


class RoadDensityDAG(BaseDAG):
    """Extract road density around sampling sites."""

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [RoadDensityTask]


class BioclimaticFactorsDAG(BaseDAG):
    """Extract bioclimatic factors around sampling sites."""

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [BioclimaticFactorsTask]


class TopographicFactorsDAG(BaseDAG):
    """Extract topographic factors around sampling sites."""

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [TopographicFactorsTask]


class LandUseFractionDAG(BaseDAG):
    """
    Extract land use fractions around sampling sites.
    NOTE: Not used in the publication, should be removed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [LandUseFractionTask]


class CombineDataDAG(BaseDAG):
    """Combine data from the DAGs above into a single dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [CombineDataTask]


class AlphaDiversityDAG(BaseDAG):
    """
    Calculate alpha diversity metrics from the PREDICTS data and generate
    features for modeling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [CombineDataTask, GenerateFeaturesTask, AlphaDiversityTask]


class BetaDiversityDAG(BaseDAG):
    """
    Calculate beta diversity metrics from the PREDICTS data and generate
    features for modeling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tasks = [CombineDataTask, GenerateFeaturesTask, BetaDiversityTask]


class ModelTrainingDAG(BaseDAG):
    """Prepare data, train model and evaluate in-sample performance."""

    def __init__(self) -> None:
        super().__init__(mode="training")
        self.tasks = [ModelDataTask, ModelTrainingTask]


class CrossValidationDAG(BaseDAG):
    """Perform cross-validation to evaluate model performance."""

    def __init__(self) -> None:
        super().__init__(mode="crossval")
        self.tasks = [ModelDataTask, CrossValidationTask]


# Mapping of DAG command line names to their corresponding classes
# E.g. running python dags.py predicts will trigger the PredictsProcessingDAG
dag_mapping = {
    "predicts": PredictsProcessingDAG,
    "population": PopulationDensityDAG,
    "roads": RoadDensityDAG,
    "bioclimatic": BioclimaticFactorsDAG,
    "topographic": TopographicFactorsDAG,
    "landuse": LandUseFractionDAG,
    "combine": CombineDataDAG,
    "alpha": AlphaDiversityDAG,
    "beta": BetaDiversityDAG,
    "training": ModelTrainingDAG,
    "crossval": CrossValidationDAG,
}


if __name__ == "__main__":
    args = parse_args()
    run_dags(args.dags, dag_mapping)
