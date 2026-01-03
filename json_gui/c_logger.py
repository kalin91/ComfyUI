"""Logger setup for child processes."""

import logging
from logging import handlers
import sys
from typing import Callable, Optional
from torch import multiprocessing as mlp


def setup_child_logger(queue: mlp.Queue) -> None:
    """Sets up the logger for child processes."""
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Limpia handlers heredados (MUY importante)
    root.handlers.clear()

    queue_handler = handlers.QueueHandler(queue)
    root.addHandler(queue_handler)


def worker_wrapper(
    func: Callable[[mlp.Queue], list[str]], log_queue: mlp.Queue, flow_queue: mlp.Queue
) -> Optional[list[str]]:
    """Wrapper to setup child logger and execute the function."""
    try:
        setup_child_logger(log_queue)
        logger = logging.getLogger("STDOUT")
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
        logger = logging.getLogger(__name__)
        logger.info("Child process logger initialized.")
        return func(flow_queue)
    except Exception:
        logging.exception("Error in worker wrapper")


class StreamToLogger:
    """Redirects writes to a logger instance."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message) -> None:
        """Write message to logger."""
        message = message.strip()
        if message:
            self.logger.log(self.level, message)

    def flush(self) -> None:
        """Flush method (no-op)."""
