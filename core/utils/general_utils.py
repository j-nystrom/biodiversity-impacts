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

    # Create and configure file handler
    # filename = "_".join(
    # [module_name, "_logs_", datetime.datetime.now().strftime(filename_date_format)]
    # )
    # logging_path = "".join([output_path, filename, file_ending])
    # file_handler = logging.FileHandler(logging_path)
    # file_handler.setLevel(logging.DEBUG)
    # file_format = logging.Formatter(logger_format, datefmt=logger_date_format)
    # file_handler.setFormatter(file_format)

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
