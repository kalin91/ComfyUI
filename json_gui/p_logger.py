"""Logger setup for parent process in JSON GUI module."""

import logging
from queue import Empty
from app.logger import setup_logger
from comfy.cli_args import args
from torch import multiprocessing as mlp


logger = logging.getLogger()
if logger.hasHandlers():
    logger.handlers.clear()

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)

# Use spawn context to avoid CUDA fork issues
MP_CONTEXT = mlp.get_context("spawn")

# Global queue for child process logging (must use spawn context)
LOG_QUEUE: mlp.Queue = MP_CONTEXT.Queue()


def poll_log_queue() -> int:
    """Poll the log queue and process any pending records.
    
    Returns:
        Number of records processed.
    """
    count = 0
    while True:
        try:
            record = LOG_QUEUE.get_nowait()
            if record is None:
                break
            logger.handle(record)
            count += 1
        except Empty:
            break
    return count


def get_log_queue() -> mlp.Queue:
    """Get the global log queue for child processes."""
    return LOG_QUEUE


def get_mp_context():
    """Get the spawn multiprocessing context."""
    return MP_CONTEXT
