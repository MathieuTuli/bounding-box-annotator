from __future__ import division
from typing import Any, Tuple, Optional, List
import pathlib

from torch.autograd import Variable
import torch
import numpy as np
import cv2


from util import load_classes, write_results
from components import FrameAnnotations, BBox, Object
from preprocess import letterbox_image
from darknet import Darknet


class YOLO:
    def __init__(self,
                 cfg_file: pathlib.Path,
                 weights_file: pathlib.Path,
                 class_names_file: pathlib.Path,
                 resolution: int = 416,
                 class_filters: List[str] = None) -> None:
        self.net: Any = None
        self.input_dim: int = None
        self.load_net(cfg_file, weights_file, resolution)
        self.class_names = load_classes(class_names_file)
        self.num_classes = len(self.class_names)
        self.class_filters = class_filters

    def load_net(self,
                 cfg_file: pathlib.Path,
                 weights_file: pathlib.Path,
                 resolution: int) -> None:
        self.net = Darknet(str(cfg_file))
        self.net.load_weights(str(weights_file))
        self.net.net_info['height'] = resolution
        self.net.cuda()
        self.input_dim = self.net.net_info['height']
        if self.input_dim % 32 != 0 or self.input_dim <= 32:
            raise ValueError("Bad input dimension. Resolution is bad")
        # self.net(get_test_input(self.input_dim, True), True)
        self.net.eval()

    def prep_frame(self, frame: np.ndarray) -> Tuple[np.ndarray,
                                                     np.ndarray,
                                                     Tuple[int, int]]:
        original_frame = frame
        dim = original_frame.shape[1], original_frame.shape[0]
        frame = (letterbox_image(original_frame, (self.input_dim,
                                                  self.input_dim)))
        frame_ = frame[:, :, ::-1].transpose((2, 0, 1)).copy()
        frame_ = torch.from_numpy(frame_).float().div(255.0).unsqueeze(0)
        return frame_, original_frame, dim

    def format_output(self,
                      output: Any,
                      threshold: float,
                      frame_dimensions: Tuple[int, int]) -> Optional[Any]:
        output = write_results(output, threshold,
                               self.num_classes, nms=True, nms_conf=threshold)
        if isinstance(output, int):
            # means no output
            return None
        frame_dimensions = frame_dimensions.repeat(output.size(0), 1)
        scaling_factor = torch.min(self.input_dim /
                                   frame_dimensions, 1)[0].view(-1, 1)
        output[:, [1, 3]] -= (self.input_dim -
                              scaling_factor *
                              frame_dimensions[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (self.input_dim -
                              scaling_factor *
                              frame_dimensions[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor
        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0,
                                            frame_dimensions[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0,
                                            frame_dimensions[i, 1])
        return output

    def get_detections(self,
                       frame: np.ndarray,
                       threshold: float = 0.7) -> FrameAnnotations:
        if frame is None:
            return FrameAnnotations(frame=frame,
                                    objects=list(),
                                    image_width=None,
                                    image_height=None)
        new_frame, frame, dimensions = self.prep_frame(frame)
        new_frame = new_frame.cuda()
        frame_dimensions = torch.FloatTensor(dimensions).repeat(1, 2).cuda()
        with torch.no_grad():
            output = self.net(Variable(new_frame), True)
        output = self.format_output(output, threshold, frame_dimensions)
        objects = list()
        if output is not None:
            for obj in output:
                if self.class_filters is not None:
                    if self.class_names[int(obj[-1])] not in \
                            self.class_filters:
                        continue
                objects.append(Object(
                    class_name=self.class_names[int(obj[-1])],
                    bbox=BBox(left=int(obj[1]),
                              top=int(obj[2]),
                              right=int(obj[3]),
                              bottom=int(obj[4]))))
        return FrameAnnotations(frame=frame,
                                objects=objects,
                                image_width=frame.shape[1],
                                image_height=frame.shape[0])


if __name__ == "__main__":
    weights_file = pathlib.Path('yolov3.weights')
    cfg_file = pathlib.Path('yolov3.cfg')
    class_names_file = pathlib.Path('coco.names')
    yolo = YOLO(cfg_file, weights_file, class_names_file)
    cap = cv2.VideoCapture(0)
    r, frame = cap.read()
    print(yolo.get_detections(frame))
