import argparse
from typing import Optional, Type

from core.utils.general_utils import create_logger, create_run_folder_path

logger = create_logger(__name__)


class BaseDAG:
    """
    Base DAG from which other DAGs inherit. It implements a 'run_dag' method
    that runs all the Tasks of a DAG in sequence.
    """

    def __init__(self, mode: Optional[str] = None) -> None:
        """
        Args:
            mode: Optional mode to pass to tasks (e.g., "training", "crossval").
        """
        self.tasks: list = []
        self.run_folder_path = create_run_folder_path()
        self.mode = mode

    def run_dag(self) -> None:
        for task in self.tasks:
            # Pass `mode` to tasks if they expect it
            if self.mode:
                task_instance = task(self.run_folder_path, mode=self.mode)
            else:
                task_instance = task(self.run_folder_path)
            task_instance.run_task()


def parse_args() -> argparse.Namespace:
    """Parse the DAG names passed as command line arguments."""
    parser = argparse.ArgumentParser(description="Run specified DAGs.")
    parser.add_argument(
        "dags", nargs="+", help="Names of DAGs to run (space-separated if multiple)."
    )
    return parser.parse_args()


def run_dags(dag_names: list[str], dag_mapping: dict[str, Type[BaseDAG]]) -> None:
    """Run all DAGs specified on command line in sequence."""
    for name in dag_names:
        if name in dag_mapping.keys():
            dag_class = dag_mapping[name]
            dag_instance = dag_class()
            logger.info(f"Running {dag_class}.")
            dag_instance.run_dag()
            logger.info(f"Successfully finished {dag_class}.")
        else:
            logger.warning(f"DAG named {name} not found.")
