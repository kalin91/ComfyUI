"""Node Executor for Mimic Nodes with Multiprocessing Support."""

import logging
import types
import time
import pickle
import signal
from functools import partial
from typing import Any, Callable, TypeVar, Optional
from torch import Tensor, multiprocessing as mlp
from json_gui import p_logger, c_logger
from json_gui.typedicts import CreationDict, SavedImagesDict, get_empty_creation_dict
from json_gui.scripts.mimic_classes import MimicNode
from json_gui.utils import EndOfFlowException

T = TypeVar("T")


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


def prepare_for_serialization(obj: T, memo: dict[int, Any] | None = None) -> Optional[T]:
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

    if isinstance(obj, Tensor):
        # detach() removes from computation graph, clone() creates independent memory,
        # contiguous() ensures memory layout is standard for pickle
        res = obj.detach().cpu().clone().contiguous()
        # res.share_memory_()
        memo[obj_id] = res
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


class NodeExecutor:
    """Class to execute mimic nodes with multiprocessing support."""

    @staticmethod
    def _execute_target_node(
        node_cls: type[MimicNode],
        node_init_args: CreationDict,
        node_exec_args: dict[str, Any],
        raw_nodes_serialized: dict[str, tuple[type[MimicNode], CreationDict]],
        save_call: Callable[[SavedImagesDict, Tensor, str], None],
        save_data: SavedImagesDict,
    ) -> tuple[str, SavedImagesDict, Any]:
        """_summary_

        Args:
            node_cls (type[MimicNode]): Input mimic node class to execute.
            node_init_args (CreationDict): Initialization arguments for the mimic node.
            node_exec_args (dict[str, Any]): Execution arguments for the mimic node.
            raw_nodes_serialized (dict[str, tuple[type[MimicNode], CreationDict]]): Is a dict mapping
                raw node keys to tuples of (mimic node class, init args dict), used to reconstruct raw nodes
                used as execution arguments.
            save_call (Callable[[SavedImagesDict, Tensor, str], None]): Callback function to save tensors.
            save_data (SavedImagesDict): Dictionary to store saved image data.

        Returns:
            tuple[str, SavedImagesDict, Any]: A tuple containing status, save data, and output or exception.
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
            result_tuple = ("success", save_data, output)
        except EndOfFlowException as e:
            logging.info("EndOfFlowException caught in %s: %s", node_cls.__name__, str(e))
            result_tuple = ("error", save_data, e)
        except Exception as e:
            logging.exception("Error executing %s in child process", node_cls.__name__)
            result_tuple = ("error", save_data, e)
        return result_tuple

    @staticmethod
    def _node_executor_target(
        node_cls: type[MimicNode],
        node_init_args: CreationDict,
        node_exec_args: dict[str, Any],
        raw_nodes_serialized: dict[str, tuple[type[MimicNode], CreationDict]],
        save_call: Callable[[SavedImagesDict, Tensor, str], None],
        save_data: SavedImagesDict,
        result_queue: mlp.Queue,
    ) -> None:
        """
        Static function to execute a MimicNode in a child process.

        This function is pickle-able because it's defined as a static method.
        It receives only serializable data (classes + dicts), not instances with callbacks.

        Args:
            node_cls (type[MimicNode]): Input mimic node class to execute.
            node_init_args (CreationDict): Initialization arguments for the mimic node.
            node_exec_args (dict[str, Any]): Execution arguments for the mimic node.
            raw_nodes_serialized (dict[str, tuple[type[MimicNode], CreationDict]]): Is a dict mapping
                raw node keys to tuples of (mimic node class, init args dict), used to reconstruct raw nodes
                used as execution arguments.
            save_call (Callable[[SavedImagesDict, Tensor, str], None]): Callback function to save tensors.
            save_data (SavedImagesDict): Dictionary to store saved image data.
            result_queue (mlp.Queue): Multiprocessing queue to put the result into.

        Returns:
            tuple[str, SavedImagesDict, Any]: A tuple containing status, save data, and output or exception.
        """

        output_tuple = NodeExecutor._execute_target_node(
            node_cls,
            node_init_args,
            node_exec_args,
            raw_nodes_serialized,
            save_call,
            save_data,
        )
        # Validate serialization before putting in queue (queue.put uses background thread
        # that swallows pickle errors silently)
        result_tuple = prepare_for_serialization(output_tuple)
        if result_tuple is None:
            raise RuntimeError("Result tuple could not be prepared for serialization.")
        logging.info("Putting result in queue... Output type: %s", type(result_tuple[2]).__name__)
        try:
            dumped_data = pickle.dumps(result_tuple)  # Test serialization
            logging.info("Result serialization test successful, size: %d kb", len(dumped_data) // 1024)
        except Exception as pickle_err:
            logging.exception("Failed to serialize result: %s", pickle_err)
            result_queue.put(("error", save_data, RuntimeError(f"Serialization failed: {pickle_err}")))
            return
        result_queue.put(result_tuple)
        logging.info("Result successfully put in queue.")
        signal.pause()

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
    def save_data(self) -> SavedImagesDict:
        """Get the save data dictionary."""
        return self._save_data

    def __init__(
        self,
        node: MimicNode,
        pre_node_process_args: dict[str, Any],
        pre_raw_nodes: dict[type[MimicNode], dict[str, Any]],
        save_data: SavedImagesDict,
    ):
        self._save_data: SavedImagesDict = save_data
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
        save_call: Callable[[SavedImagesDict, Tensor, str], None],
        timeout: Optional[float] = None,
        poll_interval: float = 0.05,
    ) -> tuple[Any, SavedImagesDict]:
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
        init_args = get_empty_creation_dict()
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
            NodeExecutor._node_executor_target,
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
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                self._process.terminate()
                self._process.join(timeout=5)
                raise TimeoutError(f"Node execution timed out after {timeout}s")

            time.sleep(poll_interval)

        # Get result from queue
        if self._result_queue.empty():
            logging.error("No result returned from child process for %s", self._node.__class__.__name__)
            raise RuntimeError("Child process ended without returning a result")
        s_data: SavedImagesDict
        status, s_data, result = self._result_queue.get()

        logging.info("ending child process for %s", self._node.__class__.__name__)
        self._process.terminate()
        self._process.join(timeout=5)

        if status == "error":
            if not isinstance(result, EndOfFlowException):
                logging.error("Child process raised an exception for %s", self._node.__class__.__name__)
                raise result
            else:
                logging.warning(
                    "EndOfFlowException propagated from child process for %s", self._node.__class__.__name__
                )

        logging.info("Node %s executed successfully in child process", self._node.__class__.__name__)

        return result, s_data

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
