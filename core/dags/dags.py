from core.dags.dag_setup import BaseDAG, parse_args, run_dags
from core.data.predicts_task import PredictsProcessingTask
from core.data.site_buffering_task import ProjectAndBufferTask


class PredictsProcessingDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [PredictsProcessingTask]


class GeoProcessingDAG(BaseDAG):
    def __init__(self) -> None:
        super().__init__()
        self.tasks = [ProjectAndBufferTask]


# Mapping of DAG names to their corresponding classes
dag_mapping = {
    "predicts": PredictsProcessingDAG,
    "geodata": GeoProcessingDAG,
}


if __name__ == "__main__":
    args = parse_args()
    run_dags(args.dags, dag_mapping)
