"""Project-level logging factory utilities."""

from logging import Logger, Formatter, StreamHandler
from logging import getLogger
from logging import DEBUG


def custom_logger(logger_name: str) -> Logger:
    """Creates or reuses a configured logger for application modules.

    Args:
        logger_name: Name of the logger.

    Returns:
        Logger: Configured logger instance.
    """

    logger = getLogger(f"{logger_name} - ")
    logger.setLevel(DEBUG)
    # No Hugging Face/transformers integration required for logging
    if not logger.hasHandlers():
        console_handler = StreamHandler()
        console_handler.setLevel(DEBUG)
        formatter = Formatter(
            "%(asctime)s.%(msecs)03d - %(name)s%(levelname)s: %(message)s",
            datefmt="%m/%d/%Y %I:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    logger.propagate = True
    return logger
