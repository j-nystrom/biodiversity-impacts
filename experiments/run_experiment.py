import argparse
import logging
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.utils.general_utils import create_run_folder_path  # noqa: E402

"""
Run model experiments defined in a single YAML file.

Experiment file format (top-level mapping):
  <experiment_name>:
    dag: <dag_name>
    overrides: <partial model_configs.yaml overrides>  # optional

Behavior:
  - Each experiment runs as a separate DAG execution.
  - A dedicated run folder is created per experiment with:
      - config.yaml (resolved base + overrides)
      - experiment.log (stdout/stderr from the DAG run)

Usage examples:
  # Run all experiments sequentially
  python experiments/run_experiment.py --experiments-file experiments/bhm_test.yaml

  # Run selected experiments sequentially
  python experiments/run_experiment.py --experiments-file experiments/bhm_test.yaml \
    --experiment bhm_alpha_training

  # Run 2 experiments in parallel locally
  python experiments/run_experiment.py --experiments-file experiments/bhm_test.yaml \
    --parallel 2
"""


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge nested dictionaries."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve a relative path against CWD first, then the repo root."""
    path = Path(path_str)
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate
    return base_dir / path


def _resolve_runs_base(repo_root: Path) -> Path:
    """Resolve the runs base directory from util_configs.yaml."""
    util_cfg_path = repo_root / "core/utils/util_configs.yaml"
    with open(util_cfg_path) as f:
        util_cfg = yaml.safe_load(f)
    base_str = util_cfg["run_folder_path"]
    base_path = Path(base_str)
    if base_path.is_absolute():
        return base_path
    return (util_cfg_path.parent / base_path).resolve()


def _create_run_folder(runs_base: Path, experiment_name: str) -> Path:
    """Create a unique run folder based on the shared utility helper."""
    safe_name = experiment_name.replace(" ", "_")
    run_folder = create_run_folder_path(
        base_path=str(runs_base),
        suffix=safe_name,
    )
    return Path(run_folder)


def _load_experiments(path: Path) -> dict:
    """Load experiments from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Experiments file must contain a mapping: {path}")
    return data


def _run_experiment(
    name: str,
    experiment_cfg: dict,
    base_cfg: dict,
    runs_base: Path,
    work_dir: Path,
    logger: logging.Logger,
) -> None:
    """Execute a single experiment and write its config/logs to run folder."""
    dag = experiment_cfg["dag"]
    overrides = experiment_cfg.get("overrides", {})

    # Merge per-experiment overrides on top of the base config.
    merged_cfg = deep_update(deepcopy(base_cfg), overrides)
    run_folder = _create_run_folder(runs_base, name)

    # Save the resolved config so the run is fully reproducible.
    config_path = run_folder / "config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(merged_cfg, f, sort_keys=False)

    # Pass config and run folder into the DAG process via env vars.
    env = os.environ.copy()
    env["MODEL_CONFIG_PATH"] = str(config_path)
    env["RUN_FOLDER_PATH"] = str(run_folder)

    log_path = run_folder / "experiment.log"
    cmd = ["python", str(work_dir / "dags.py"), dag]

    logger.info(
        "Starting experiment '%s' (dag=%s). Run folder: %s.",
        name,
        dag,
        run_folder,
    )
    # Capture stdout/stderr in the run folder for post-mortem debugging.
    with open(log_path, "w", encoding="utf-8") as log_stream:
        subprocess.run(
            cmd,
            cwd=work_dir,
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    logger.info("Completed experiment '%s'. Run folder: %s.", name, run_folder)


def main() -> None:
    """Parse CLI args and execute one or more experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Runtime diagnostics for Slurm/local parity checks.
    logger.info("Host: %s | PID: %s", os.uname().nodename, os.getpid())
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
        logger.info(
            "CPU affinity count=%d | sample=%s",
            len(affinity),
            affinity[:16],
        )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments-file",
        default="experiments/experiments.yaml",
        help="Path to the experiments YAML file.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        help="Experiment name to run (repeatable). If omitted, runs all.",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Run experiments locally in parallel (max concurrent runs).",
    )
    args = parser.parse_args()

    # Resolve paths relative to the repo root for local/cluster parity.
    experiments_path = _resolve_path(args.experiments_file, repo_root)
    base_config_path = repo_root / "core/model/model_configs.yaml"
    work_dir = repo_root / "core/dags"
    runs_base = _resolve_runs_base(repo_root)

    # Load base and experiments config once to avoid repeated disk I/O.
    with open(base_config_path) as f:
        base_cfg = yaml.safe_load(f)

    experiments = _load_experiments(experiments_path)
    if args.experiment:
        names = args.experiment
    else:
        names = list(experiments.keys())

    missing = [name for name in names if name not in experiments]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Experiment(s) not found in {experiments_path}: {missing_list}"
        )

    logger.info("Running %d experiment(s) from %s.", len(names), experiments_path)

    if args.parallel <= 1:
        # Execute experiments in sequence to keep resource usage predictable.
        for name in names:
            _run_experiment(
                name,
                experiments[name],
                base_cfg,
                runs_base,
                work_dir,
                logger,
            )
        return

    # Run experiments in parallel locally, up to the requested limit.
    from concurrent.futures import ThreadPoolExecutor, as_completed

    max_parallel = max(1, int(args.parallel))
    logger.info("Running experiments in parallel (max %d).", max_parallel)

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_map = {
            executor.submit(
                _run_experiment,
                name,
                experiments[name],
                base_cfg,
                runs_base,
                work_dir,
                logger,
            ): name
            for name in names
        }
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Experiment '%s' failed: %s", name, exc)
                raise


if __name__ == "__main__":
    main()
