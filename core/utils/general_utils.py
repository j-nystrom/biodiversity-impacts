import datetime
import logging
import os

from box import Box

# Load the config file into box object; ensure that it can be found regardless
# of where the module is loaded / run from
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, "util_configs.yaml")
configs = Box.from_yaml(filename=config_path)


def create_logger(
    module_name: str,
    output_path: str = configs.logger_objects.output_path,
    filename_date_format: str = configs.logger_objects.filename_date_format,
    file_ending: str = configs.logger_objects.file_ending,
    logger_format: str = configs.logger_objects.logger_format,
    logger_date_format: str = configs.logger_objects.logger_date_format,
) -> logging.Logger:
    """
    Create a customized logger object for the current module run.

        Args:
            module_name: name of the calling module
            output_path: where the log file should be saved
            filename_date_format: date and time format in the filename
            file_ending: saved as csv file
            logger_format: information and format of each logger entry
            logger_date_format: date format of the logger entries

        Returns:
            logger: the logger object
    """

    # Create a custom logger object
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.DEBUG)

    # Create and configure stream handler (printing to the console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_format = logging.Formatter(logger_format, datefmt=logger_date_format)
    stream_handler.setFormatter(stream_format)

    # Add handlers to the logger object
    # File handler disabled for now
    # logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def create_run_folder_path(
    base_path: str = configs.run_folder_path,
    suffix: str | None = None,
) -> str:
    """
    Generate a unique run folder based on the current date and time.

    Args:
        base_path: Base directory where run folders are created.
        suffix: Optional suffix appended to the run folder name.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    run_folder_name = f"run_folder_{timestamp}"
    if suffix:
        safe_suffix = suffix.replace(" ", "_")
        run_folder_name = f"{run_folder_name}_{safe_suffix}"

    base_candidate = os.path.join(base_path, run_folder_name)
    counter = 1
    while True:
        candidate = base_candidate if counter == 1 else f"{base_candidate}_{counter}"
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            counter += 1
