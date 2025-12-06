""""""

import json
from mimic_classes import OpenPosePose, ApplyControlNet, EmptyLatent, SimpleKSampler, FaceDetailer


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
        positive: str,
        negative: str,
        openpose_pose: OpenPosePose,
        apply_control_net: ApplyControlNet,
        empty_latent: EmptyLatent,
        simple_k_sampler: list[SimpleKSampler],
        face_detailer: FaceDetailer,
    ):
        self._positive = positive
        self._negative = negative
        self._openpose_pose = openpose_pose
        self._apply_control_net = apply_control_net
        self._empty_latent = empty_latent
        self._simple_k_sampler = simple_k_sampler
        self._face_detailer = face_detailer
