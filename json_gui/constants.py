"""Constants for JSON GUI module."""

from comfy.samplers import SAMPLER_NAMES, SCHEDULER_NAMES, SCHEDULER_HANDLERS
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.core import ADDITIONAL_SCHEDULERS
from custom_nodes.ComfyUI_Impact_Pack.modules.impact.impact_pack import FaceDetailer


COMBO_CONSTANTS = {
    "sampler_names": SAMPLER_NAMES,
    "scheduler_names": SCHEDULER_NAMES,
    "scheduler_handlers": list(SCHEDULER_HANDLERS) + ADDITIONAL_SCHEDULERS,
    "sam_detection_hint": FaceDetailer.INPUT_TYPES()["required"]["sam_detection_hint"][0],
}

JSON_CANVAS_NAME = "json_canvas"
JSON_SCROLL_FRAME_NAME = "json_scrollable_frame"
