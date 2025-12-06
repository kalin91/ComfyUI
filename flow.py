""""""

import os
import json
from mimic_classes import OpenPosePose, ApplyControlNet, EmptyLatent, SimpleKSampler, FaceDetailer
import logging
IMAGES_PATH = os.path.join(os.getcwd(), "images/")


class Flow:

    @property
    def positive(self) -> str:
        """Returns the positive prompt."""
        return self._positive

    @property
    def negative(self) -> str:
        """Returns the negative prompt."""
        return self._negative

    @property
    def openpose_pose(self) -> OpenPosePose:
        """Returns the OpenPosePose instance."""
        return self._openpose_pose

    @property
    def apply_control_net(self) -> ApplyControlNet:
        """Returns the ApplyControlNet instance."""
        return self._apply_control_net

    @property
    def empty_latent(self) -> EmptyLatent:
        """Returns the EmptyLatent instance."""
        return self._empty_latent

    @property
    def simple_k_sampler(self) -> list[SimpleKSampler]:
        """Returns the SimpleKSampler instance."""
        return self._simple_k_sampler

    @property
    def face_detailer(self) -> FaceDetailer:
        """Returns the FaceDetailer instance."""
        return self._face_detailer

    def __init__(
        self,
        image_id: str,
    ):
        filepath = os.path.join(IMAGES_PATH, image_id + ".json")
        
        assert os.path.exists(filepath), f"Flow file {filepath} does not exist."
        logging.info(f"Loading flow from {filepath}...")
        with open(filepath, "r", encoding="utf-8") as file:
            json_props = json.load(file)
        self._positive = json_props["positive"]
        self._negative = json_props["negative"]
        self._openpose_pose = OpenPosePose(**json_props["openpose_pose"])
        self._apply_control_net = ApplyControlNet(**json_props["apply_control_net"])
        self._empty_latent = EmptyLatent(**json_props["empty_latent"])
        self._simple_k_sampler = [SimpleKSampler(**s) for s in json_props["simple_k_sampler"]]
        self._face_detailer = FaceDetailer(**json_props["face_detailer"])
