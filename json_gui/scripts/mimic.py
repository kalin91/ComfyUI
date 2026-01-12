"""Mimic Parent Class and Utilities."""

import types
import time
import logging
import uuid
from functools import wraps, partial
from typing import Any, Callable, Optional, Type, TypeVar, Generic
from abc import ABC, abstractmethod
import os
import signal
import pickle
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

    @property
    def is_latent_tensor(self) -> bool:
        """Check if the wrapped value is a latent tensor."""
        return self._is_latent_tensor

    @property
    def image_name(self) -> Optional[str]:
        """Get the image name if applicable."""
        return self._image_name

    def __init__(
        self,
        value: T,
        identifier: Optional[str] = None,
        is_latent_tensor: bool = False,
        image_name: Optional[str] = None,
    ):
        self.value: T = value
        self._identifier: str = identifier if identifier else str(uuid.uuid4())
        self._is_latent_tensor: bool = is_latent_tensor
        self._image_name: Optional[str] = image_name

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


class MimicNode(ABC, Generic[T]):
    """A mimic class for various nodes."""

    def process_args_dict(self, *args, **kwargs) -> dict[str, Any]:
        """
        Given args and kwargs, returns a dict mapping parameter names to values for _process_impl.
        Useful for introspection, serialization, or dynamic invocation.
        """
        sig = inspect.signature(self._process_impl)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        """Returns the key for the mimic node."""

    @classmethod
    def use_class_param(
        cls: Type[T], processor: Callable[[T], dict[str, Any]]
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator to extract a class parameter and process it."""
        param = cls.key()

        def decorator(func) -> Callable:
            """Decorator function."""

            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                """Wrapper function."""
                if param not in kwargs:
                    raise TypeError(f"Missing required parameter: {param}")
                node: T = kwargs.pop(param)
                additional_params: dict[str, Any] = processor(node)
                kwargs.update(additional_params)
                return func(*args, **kwargs)

            # Register the parameter on the wrapper so _feed_function knows it's allowed
            wrapper._mimic_extra_params = getattr(func, "_mimic_extra_params", set()) | {  # pylint: disable=W0212
                param
            }

            return wrapper

        return decorator

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
        # Allow parameters explicitly requested by decorators
        extra_params = getattr(func, "_mimic_extra_params", set())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or k in extra_params}
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
    def save_tensor(self) -> Optional[Callable[[Tensor, str], None]]:
        """Returns a function to save the tensor."""
        return self._save_tensor

    @save_tensor.setter
    def save_tensor(self, value: Callable[[Tensor, str], None]) -> None:
        """Sets the function to save the tensor."""
        assert callable(value), "save_tensor must be a callable function."
        self._save_tensor = value

    @save_tensor.deleter
    def save_tensor(self) -> None:
        """Deletes the save tensor function."""
        self._save_tensor = None

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


def _is_unserializable_callable(obj: Any) -> bool:
    """
    Check if obj is a callable that cannot be pickled.

    Lambdas, local functions, and closures typically cannot be pickled
    because they reference local scope that pickle cannot capture.
    """

    if not callable(obj):
        return False

    # Check if it's a lambda (name is '<lambda>')
    if isinstance(obj, types.FunctionType):
        if obj.__name__ == "<lambda>":
            return True
        # Check if it's a local/nested function (has '<locals>' in qualname)
        if obj.__qualname__ and "<locals>" in obj.__qualname__:
            return True

    # Check for bound methods with unserializable functions
    if isinstance(obj, types.MethodType):
        return _is_unserializable_callable(obj.__func__)

    # partial objects wrapping unserializable functions
    if isinstance(obj, partial):
        return _is_unserializable_callable(obj.func)

    return False


def prepare_for_serialization(obj: T, memo: dict[int, Any] | None = None) -> T:
    """
    Recursively prepares objects for pickle serialization.

    - Moves tensors to CPU, detaches from computation graph, and clones
    - Replaces unserializable callables (lambdas, local functions) with None
    """
    if memo is None:
        memo = {}

    obj_id = id(obj)
    if obj_id in memo:
        return memo[obj_id]

    # Handle unserializable callables first (lambdas, local functions, etc.)
    if _is_unserializable_callable(obj):
        memo[obj_id] = None
        return None

    if isinstance(obj, torch.Tensor):
        # detach() removes from computation graph, clone() creates independent memory,
        # contiguous() ensures memory layout is standard for pickle
        res = obj.detach().cpu().clone().contiguous()
        # res.share_memory_()
        memo[obj_id] = (res,)
        return res
    if isinstance(obj, dict):
        res = {}
        memo[obj_id] = res
        for k, v in obj.items():
            res[k] = prepare_for_serialization(v, memo)
        return res
    if isinstance(obj, list):
        res = []
        memo[obj_id] = res
        for x in obj:
            res.append(prepare_for_serialization(x, memo))
        return res
    if isinstance(obj, (tuple, set)):
        cls = type(obj)
        res = cls(prepare_for_serialization(x, memo) for x in obj)
        memo[obj_id] = res
        return res
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        # For custom objects, recursively process their attributes
        memo[obj_id] = obj
        for k, v in list(obj.__dict__.items()):
            setattr(obj, k, prepare_for_serialization(v, memo))
        return obj
    memo[obj_id] = obj
    return obj


def _node_executor_target(
    node_cls: type[MimicNode],
    node_init_args: dict[str, Any],
    node_exec_args: dict[str, Any],
    raw_nodes_serialized: dict[str, tuple[type[MimicNode], dict[str, Any]]],
    save_call: Optional[Callable[[dict[str, bool | list[str]], Tensor, str], None]],
    save_data: dict[str, bool | list[str]],
    result_queue: mlp.Queue,
) -> None:
    """
    Top-level function to execute a MimicNode in a child process.

    This function is pickle-able because it's defined at module level.
    It receives only serializable data (classes + dicts), not instances with callbacks.
    """
    try:
        assert (
            "last_saved_to_temp" in save_data and "created_images" in save_data
        ), "Data dict must contain 'last_saved_to_temp' and 'created_images' keys."
        logging.info("Executing %s in child process", node_cls.__name__)
        # Reconstruct the main node from its class and init args
        node = node_cls(*node_init_args["args"], **node_init_args["kwargs"])
        node.save_tensor = lambda tensor, identifier=node.__class__.key(), data=save_data: save_call(
            data, tensor, identifier
        )

        # Reconstruct raw_nodes and add them to exec_args
        for key, (cls, init_args) in raw_nodes_serialized.items():
            assert "args" in init_args and "kwargs" in init_args, "init_args must contain 'args' and 'kwargs' keys"
            node_exec_args[key] = cls(*init_args["args"], **init_args["kwargs"])

        # Execute the node
        output = node.process(**node_exec_args)

        # Prepare tensors for serialization: move to CPU, detach from graph, clone
        output = prepare_for_serialization(output)

        logging.info("Putting result in queue... Output type: %s", type(output))

        # Validate serialization before putting in queue (queue.put uses background thread
        # that swallows pickle errors silently)
        try:
            dumped_data = pickle.dumps(output)  # Test serialization
            logging.info("Result serialization test successful, size: %d kb", len(dumped_data) // 1024)
        except Exception as pickle_err:
            logging.exception("Failed to serialize result: %s", pickle_err)
            result_queue.put(("error", save_data, RuntimeError(f"Serialization failed: {pickle_err}")))
            return

        result_tuple = ("success", save_data, output)

        result_queue.put(result_tuple)
        logging.info("Result successfully put in queue.")
        signal.pause()
    except Exception as e:
        logging.exception("Error executing %s in child process", node_cls.__name__)
        result_queue.put(("error", save_data, e))


class NodeExecutor:
    """Class to execute mimic nodes with multiprocessing support."""

    @property
    def raw_nodes(self) -> dict[str, tuple[type[MimicNode], dict[str, Any]]]:
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
        """Get the multiprocessing queue."""
        return self._result_queue

    @property
    def save_data(self) -> dict[str, bool | list[str]]:
        """Get the save data dictionary."""
        return self._save_data

    def __init__(
        self,
        node: MimicNode,
        pre_node_process_args: dict[str, Any],
        pre_raw_nodes: dict[type[MimicNode], dict[str, Any]],
        save_data: dict[str, bool | list[str]],
    ):
        self._save_data: dict[str, bool | list[str]] = save_data
        self._node: MimicNode = node
        self._node_process_args: dict[str, Any] = {}
        for key, value in pre_node_process_args.items():
            # move tensors to CPU so they can be pickled
            if isinstance(value, Tensor):
                self._node_process_args[key] = value.cpu()
            if isinstance(value, MimicNode):
                del value.save_tensor
            else:
                self._node_process_args[key] = value
        self._result_queue: mlp.Queue = p_logger.get_mp_context().Queue()
        self._log_queue: mlp.Queue = p_logger.get_log_queue()
        self._raw_nodes: dict[str, tuple[type[MimicNode], dict[str, Any]]] = {
            t.key(): (t, args) for t, args in pre_raw_nodes.items()
        }
        self._process: Optional[mlp.Process] = None

    def execute(
        self,
        save_call: Optional[Callable[[dict[str, bool | list[str]], Tensor, str], None]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> Any:
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
        raw_init_args = self._node.init_args
        init_args = {"args": [], "kwargs": {}}
        for val in raw_init_args["args"]:
            if isinstance(val, Tensor):
                init_args["args"].append(val.cpu())
            elif isinstance(val, MimicNode):
                del val.save_tensor
            else:
                init_args["args"].append(val)
        for key, val in raw_init_args["kwargs"].items():
            if isinstance(val, Tensor):
                init_args["kwargs"][key] = val.cpu()
            elif isinstance(val, MimicNode):
                del val.save_tensor
            else:
                init_args["kwargs"][key] = val

        worker_target = partial(
            _node_executor_target,
            self._node.__class__,
            self._node.init_args,
            self._node_process_args,
            self._raw_nodes,
            save_call,
            self._save_data,
        )
        self._process = p_logger.get_mp_context().Process(
            target=c_logger.worker_wrapper,
            name=f"MimicNodeExecutor-{self._node.__class__.__name__}",
            args=(worker_target, self._log_queue, self._result_queue),
        )
        self._process.start()

        # Poll log queue while waiting for result
        start_time = time.time()

        while self._process.is_alive() and self._result_queue.empty():
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
            logging.error("No result returned from child process for %s", self._node.__class__.__name__)
            raise RuntimeError("Child process ended without returning a result")

        status, s_data, result = self._result_queue.get()

        logging.info("ending child process for %s", self._node.__class__.__name__)
        self._process.terminate()
        self._process.join(timeout=5)

        self._save_data.update(s_data)

        if status == "error":
            logging.error("Child process raised an exception for %s", self._node.__class__.__name__)
            raise result

        logging.info("Node %s executed successfully in child process", self._node.__class__.__name__)

        return result

    @classmethod
    def _executable(
        cls,
        node: MimicNode,
        node_exec_args: dict[str, Any],
        queue: mlp.Queue,
        raw_nodes: dict[MimicNode, dict[str, Any]],
    ) -> None:
        """Executes the node and puts the result in the queue."""
        try:
            for t, args in raw_nodes.items():
                node_exec_args[t.key()] = t(**args)
            output = node.process(**node_exec_args)
            queue.put(output)
        except Exception as e:
            logging.exception("Error executing %s", node.__class__.key())
            queue.put(e)
