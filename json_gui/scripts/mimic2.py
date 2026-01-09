"""Mimic Parent Class and Utilities."""

import logging
import uuid
from typing import Any, Callable, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
import os
import inspect
import numpy as np
import torch
from torch import Tensor, multiprocessing as mlp
from PIL import Image, ImageOps, ImageSequence
import node_helpers
import folder_paths
from json_gui import p_logger, c_logger

T = TypeVar("T")


class DataWrapper(Generic[T]):
    """A generic parameter wrapper class."""

    @property
    def identifier(self) -> str:
        """Get the unique identifier for this wrapper."""
        return self._identifier

    def __init__(self, value: T, identifier: Optional[str] = None):
        self.value: T = value
        self._identifier: str = identifier if identifier else str(uuid.uuid4())

    def get(self) -> T:
        """Get the wrapped value."""
        return self.value


def safe_reference_compare(var1, var2) -> bool:
    """
    if var1 and var2 are of basic types (int, float, str, bool, tuple, bytes, NoneType),
    compare by value (==); else compare by identity (is).
    """
    # 1. For DataWrapper, always compare by identifier
    if isinstance(var1, DataWrapper):
        return var1.identifier == var2.identifier

    # 2. For basic types, compare by value
    optimizable_types = (int, str, float, bool, tuple, bytes, type(None))

    if isinstance(var1, optimizable_types):
        # assuming their __eq__ methods are standard and safe.
        return var1 == var2

    # 3. For EVERYTHING else (lists, dicts, your own classes, etc.)
    # we use 'is' to avoid triggering custom or slow __eq__ methods.
    return var1 is var2


class MimicNode(ABC):
    """A mimic class for various nodes."""

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        """Returns the key for the mimic node."""

    def _unwrap_data_dict(self, *args, **kwargs) -> tuple[list[Any], dict[str, Any]]:
        """Unwraps any DataWrapper instances in the provided dictionary."""

        unwrapped_args = []
        unwrapped_kwargs = {}
        for item in args:
            if isinstance(item, DataWrapper):
                unwrapped_args.append(item.get())
            else:
                unwrapped_args.append(item)
        for key, item in kwargs.items():
            if isinstance(item, DataWrapper):
                unwrapped_kwargs[key] = item.get()
            else:
                unwrapped_kwargs[key] = item
        return unwrapped_args, unwrapped_kwargs

    def _has_kwargs(self, func: Callable) -> bool:
        """Checks if func accepts keyword arguments."""
        sig = inspect.signature(func)
        return any(
            p.kind is inspect.Parameter.VAR_KEYWORD or p.kind is inspect.Parameter.VAR_POSITIONAL
            for p in sig.parameters.values()
        )

    def _feed_function(self, func, /, *args, **kwargs) -> Any:
        """Calls func with only the keyword arguments that it accepts."""
        if self._has_kwargs(func):
            return func(*args, **kwargs)
        logging.debug("Feeding function %s with args: %s and kwargs: %s", func.__name__, args, list(kwargs.keys()))
        sig = inspect.signature(func)
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return func(*args, **filtered_kwargs)

    @property
    def init_args(self) -> dict[str, Any]:
        """Returns the last arguments used."""
        return self._init_args

    @init_args.deleter
    def init_args(self) -> None:
        """Deletes the cached init arguments."""
        self._init_args = {}

    @property
    def exec_args(self) -> dict[str, Any]:
        """Returns the last execution arguments used."""
        return self._exec_args

    @property
    def last_output(self) -> Optional[Any]:
        """Returns the last outputs produced."""
        return self._last_output

    @property
    def save_tensor(self) -> Optional[Callable[[Tensor], None]]:
        """Returns a function to save the tensor."""
        return self._save_tensor

    @save_tensor.setter
    def save_tensor(self, value: Callable[[Tensor], None]) -> None:
        """Sets the function to save the tensor."""
        assert callable(value), "save_tensor must be a callable function."
        self._save_tensor = value

    def __init__(self):
        self._save_tensor: Optional[Callable[[Tensor, str], None]] = None
        self._return_cache = False
        self._init_args: dict[str, Any] = {}
        self._exec_args: dict[str, Any] = {}
        self._last_output: Optional[Any] = None

    def _upload_image(self, image_name: str) -> Tensor:
        """Uploads an image given its name."""
        assert image_name, "Image name must be provided for ControlNetImgPreprocessor."
        logging.info("Loading %s Image...", self.__class__.key())
        input_folder = folder_paths.get_input_directory()
        image_path = os.path.join(input_folder, image_name)
        img = node_helpers.pillow(Image.open, image_path)

        # Process image to tensor (similar to LoadImage node)
        output_images = []
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            output_images.append(image)

        if len(output_images) > 1:
            # If multiple frames, stack them? For now assume single image as per workflow
            img_tensor = torch.cat(output_images, dim=0)
        else:
            img_tensor = output_images[0]
        return img_tensor

    def update(self, *args, **kwargs) -> None:
        """Updates the node."""
        try:
            if not self._init_args or not self.__use_cache(self._init_args, *args, **kwargs):
                logging.info("Updating %s", self.__class__.__name__)
                uw_args, uw_kwargs = self._unwrap_data_dict(*args, **kwargs)
                self._update_impl(*uw_args, **uw_kwargs)
                self._init_args = {"args": args, "kwargs": kwargs}
                self._last_output = None
        except Exception as e:
            logging.exception("Error updating %s: %s", self.__class__.__name__, str(e))
            raise e

    @abstractmethod
    def _update_impl(self, *args, **kwargs) -> None:
        """Abstract method to update the node."""

    @abstractmethod
    def _process_impl(self, *args, **kwargs) -> Any:
        """Abstract method to process data."""

    def process(self, *args, **kwargs) -> Any:
        """Processes data and returns the result, using caching if available."""
        if self._return_cache and self._last_output is not None and self.__use_cache(self._exec_args, *args, **kwargs):
            logging.info("====== Using cached output for %s ======", self.__class__.__name__)
            return self._last_output
        self._return_cache = False
        logging.info("====== Processing %s ======", self.__class__.key())
        uw_args, uw_kwargs = self._unwrap_data_dict(*args, **kwargs)
        res = self._feed_function(self._process_impl, *uw_args, **uw_kwargs)
        self._exec_args = {"args": args, "kwargs": kwargs}
        self._last_output = res
        self._return_cache = True
        return res

    def __use_cache(self, cached_args: dict, *args, **kwargs) -> bool:
        """Evaluates whether to use cached output based on the provided arguments."""
        cache_invalid: bool = False
        if cached_args and "args" in cached_args and "kwargs" in cached_args:
            cache_args = cached_args["args"]
            cache_kwargs = cached_args["kwargs"]
            for i, vl_1 in enumerate(args):
                if len(cache_args) > i:
                    vl_2 = cache_args[i]
                    if safe_reference_compare(vl_1, vl_2):
                        continue
                cache_invalid = True
                break
            if not cache_invalid:
                for key, vl_1 in kwargs.items():
                    if key in cache_kwargs:
                        vl_2 = cache_kwargs[key]
                        if safe_reference_compare(vl_1, vl_2):
                            continue
                    cache_invalid = True
                    break
        return not cache_invalid


def _node_executor_target(
    node_cls: type[MimicNode],
    node_init_args: dict[str, Any],
    node_exec_args: dict[str, Any],
    raw_nodes_serialized: dict[str, tuple[type[MimicNode], dict[str, Any]]],
    result_queue: mlp.Queue,
) -> None:
    """
    Top-level function to execute a MimicNode in a child process.

    This function is pickle-able because it's defined at module level.
    It receives only serializable data (classes + dicts), not instances with callbacks.
    """
    try:
        # Reconstruct the main node from its class and init args
        node = node_cls(**node_init_args)

        # Reconstruct raw_nodes and add them to exec_args
        for key, (cls, init_args) in raw_nodes_serialized.items():
            node_exec_args[key] = cls(**init_args)

        # Execute the node
        output = node.process(**node_exec_args)
        result_queue.put(("success", output))
    except Exception as e:
        logging.exception("Error executing %s in child process", node_cls.__name__)
        result_queue.put(("error", e))


def _worker_target(flow_queue: mlp.Queue) -> Optional[Any]:
    """
    Top-level wrapper that c_logger.worker_wrapper will call.

    Unpacks arguments from the flow_queue and calls _node_executor_target.
    """
    # Get the serialized arguments from the queue
    args = flow_queue.get()
    node_cls, node_init_args, node_exec_args, raw_nodes_serialized, result_queue = args
    _node_executor_target(node_cls, node_init_args, node_exec_args, raw_nodes_serialized, result_queue)
    return None


class NodeExecutor:
    """Class to execute mimic nodes with multiprocessing support."""

    @property
    def raw_nodes(self) -> dict[MimicNode, dict[str, Any]]:
        """Get the list of raw mimic nodes."""
        return self._raw_nodes

    @property
    def node(self) -> MimicNode:
        """Get the mimic node."""
        return self._node

    @property
    def node_process_args(self) -> dict[str, Any]:
        """Get the node arguments."""
        return self._node_process_args

    @property
    def result_queue(self) -> mlp.Queue:
        """Get the multiprocessing result queue."""
        return self._result_queue

    def __init__(
        self,
        node: MimicNode,
        node_process_args: dict[str, Any],
        raw_nodes: Optional[dict[MimicNode, dict[str, Any]]] = None,
    ):
        """
        Initialize the NodeExecutor.

        Args:
            node: The MimicNode instance to execute.
            node_process_args: Arguments to pass to node.process().
            raw_nodes: Optional dict mapping MimicNode instances to their init_args.
                       These will be reconstructed in the child process and added to node_process_args.
        """
        self._node: MimicNode = node
        self._node_process_args: dict[str, Any] = node_process_args
        self._raw_nodes: dict[MimicNode, dict[str, Any]] = raw_nodes or {}

        # Use spawn context from p_logger to avoid CUDA fork issues
        mp_context = p_logger.get_mp_context()
        self._result_queue: mlp.Queue = mp_context.Queue()
        self._log_queue: mlp.Queue = p_logger.get_log_queue()
        self._mp_context = mp_context
        self._process: Optional[mlp.Process] = None

    def _serialize_for_child(self) -> tuple[
        type[MimicNode],
        dict[str, Any],
        dict[str, Any],
        dict[str, tuple[type[MimicNode], dict[str, Any]]],
    ]:
        """
        Serialize all data needed by the child process.

        Returns pickleable data only: classes and dicts, not instances with callbacks.
        """
        # Get the class and init_args from the main node
        node_cls = self._node.__class__
        node_init_args = dict(self._node.init_args.get("kwargs", {}))

        # Filter node_process_args to only include pickleable items
        # (exclude MimicNode instances - they'll be in raw_nodes)
        node_exec_args = {}
        for key, value in self._node_process_args.items():
            if not isinstance(value, MimicNode):
                node_exec_args[key] = value

        # Serialize raw_nodes: key -> (class, init_args)
        raw_nodes_serialized: dict[str, tuple[type[MimicNode], dict[str, Any]]] = {}
        for node_inst, init_args in self._raw_nodes.items():
            raw_nodes_serialized[node_inst.key()] = (node_inst.__class__, init_args)

        return node_cls, node_init_args, node_exec_args, raw_nodes_serialized

    def execute(self, timeout: Optional[float] = None, poll_interval: float = 0.1) -> Any:
        """
        Execute the node in a child process and return the result.

        Args:
            timeout: Maximum time to wait for result (None = wait forever).
            poll_interval: How often to poll the log queue while waiting.

        Returns:
            The output from node.process().

        Raises:
            Exception: If the child process raised an exception.
            TimeoutError: If timeout is reached.
        """
        node_cls, node_init_args, node_exec_args, raw_nodes_serialized = self._serialize_for_child()

        # Create a queue to pass arguments to the child (since we can't use lambdas with spawn)
        args_queue: mlp.Queue = self._mp_context.Queue()
        args_queue.put((node_cls, node_init_args, node_exec_args, raw_nodes_serialized, self._result_queue))

        # Create the child process using spawn context
        self._process = self._mp_context.Process(
            target=c_logger.worker_wrapper,
            args=(
                _worker_target,
                self._log_queue,
                args_queue,
            ),
        )

        self._process.start()

        # Poll log queue while waiting for result
        import time

        start_time = time.time()

        while self._process.is_alive():
            # Poll logs from child
            p_logger.poll_log_queue()

            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                self._process.terminate()
                self._process.join(timeout=5)
                raise TimeoutError(f"Node execution timed out after {timeout}s")

            time.sleep(poll_interval)

        # Process finished - drain remaining logs
        p_logger.poll_log_queue()

        # Get result from queue
        if self._result_queue.empty():
            raise RuntimeError("Child process ended without returning a result")

        status, result = self._result_queue.get()

        if status == "error":
            raise result

        return result

    def execute_async(self) -> "NodeExecutor":
        """
        Start the node execution in a child process without waiting.

        Use poll() to check for completion and get_result() to retrieve the result.

        Returns:
            self, for method chaining.
        """
        node_cls, node_init_args, node_exec_args, raw_nodes_serialized = self._serialize_for_child()

        # Create a queue to pass arguments to the child
        args_queue: mlp.Queue = self._mp_context.Queue()
        args_queue.put((node_cls, node_init_args, node_exec_args, raw_nodes_serialized, self._result_queue))

        self._process = self._mp_context.Process(
            target=c_logger.worker_wrapper,
            args=(
                _worker_target,
                self._log_queue,
                args_queue,
            ),
        )

        self._process.start()
        return self

    def poll(self) -> bool:
        """
        Poll logs from child and check if execution is complete.

        Returns:
            True if execution is complete, False if still running.
        """
        p_logger.poll_log_queue()
        return self._process is not None and not self._process.is_alive()

    def get_result(self) -> Any:
        """
        Get the result after execution is complete.

        Call poll() first to ensure execution is complete.

        Returns:
            The output from node.process().

        Raises:
            Exception: If the child process raised an exception.
            RuntimeError: If no result is available.
        """
        if self._process is None:
            raise RuntimeError("No process has been started")

        if self._process.is_alive():
            raise RuntimeError("Process is still running")

        # Drain remaining logs
        p_logger.poll_log_queue()

        if self._result_queue.empty():
            raise RuntimeError("Child process ended without returning a result")

        status, result = self._result_queue.get()

        if status == "error":
            raise result

        return result

    def terminate(self) -> None:
        """Terminate the child process if running."""
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5)
