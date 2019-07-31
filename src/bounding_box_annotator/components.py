from typing import NamedTuple, List

import numpy as np
import enum


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


RoomType = enum.Enum('RoomType',
                     ['staircase', 'entrance', 'laundry',
                      'bed', 'bath', 'kitchen', 'living',
                      'dining', 'hallway', 'garage',
                      'closet', 'other'])


class RoomAnnotation(NamedTuple):
    class_name: RoomType
    bbox: BBox


class FloorPlanAnnotation(NamedTuple):
    floor_num: int
    floor_plan: np.ndarray
    rooms: List[RoomAnnotation]
    image_width: int
    image_height: int
