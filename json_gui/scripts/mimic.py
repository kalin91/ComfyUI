"""Mimic Parent Class and Utilities."""

import logging
import uuid
from typing import Any, Callable, Optional, TypeVar, Generic
from abc import ABC, abstractmethod
import os
import inspect
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
import node_helpers
import folder_paths

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

    @property
    def exec_args(self) -> dict[str, Any]:
        """Returns the last execution arguments used."""
        return self._exec_args

    @property
    def last_output(self) -> Optional[Any]:
        """Returns the last outputs produced."""
        return self._last_output

    @property
    def save_tensor(self) -> Optional[Callable[[torch.Tensor], None]]:
        """Returns a function to save the tensor."""
        return self._save_tensor

    @save_tensor.setter
    def save_tensor(self, value: Callable[[torch.Tensor], None]) -> None:
        """Sets the function to save the tensor."""
        assert callable(value), "save_tensor must be a callable function."
        self._save_tensor = value

    def __init__(self):
        self._save_tensor: Optional[Callable[[torch.Tensor, str], None]] = None
        self._return_cache = False
        self._init_args: dict[str, Any] = {}
        self._exec_args: dict[str, Any] = {}
        self._last_output: Optional[Any] = None

    def _upload_image(self, image_name: str) -> torch.Tensor:
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

    def _process(self, *args, **kwargs) -> Any:
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
