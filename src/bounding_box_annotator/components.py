from typing import NamedTuple, List

import numpy as np


class BBox(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int


class Object(NamedTuple):
    class_name: str
    bbox: BBox


class FrameAnnotations(NamedTuple):
    frame: np.ndarray
    objects: List[Object]
    image_width: int
    image_height: int
