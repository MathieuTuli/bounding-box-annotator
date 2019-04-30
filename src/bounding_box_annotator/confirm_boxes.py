# TODO: Make Frame class that has frame: np.ndarray and annotions: List[str] as
# @ members that can handle all the drawing on its own
from typing import Any, Dict, Tuple, List
from argparse import ArgumentParser
import pathlib
import shutil
import sys
import os

import numpy as np
import cv2

from pytorch_yolo_api import YOLO
from components import FrameAnnotations, Object, BBox

REF_PTS = list()


class BoundingBoxAnnotator:
    def __init__(self,
                 input_bank: pathlib.Path,
                 image_bank: pathlib.Path,
                 annotation_file_bank: pathlib.Path,
                 object_detector_settings: Dict[str, Any],
                 threshold: int = 0.7,
                 frame_jump: int = 10,
                 start_at: str = '0.txt') -> None:
        self.input_bank = input_bank
        self.image_bank = image_bank
        self.annotation_file_bank = annotation_file_bank
        self.confirm_directories()
        self.threshold = threshold
        self.frame_counter = 0
        self.object_detector: Any = None
        self.object_detector_settings: Dict[str, Any] = \
            object_detector_settings
        self.frame_jump = frame_jump
        self.start_at = start_at

    @staticmethod
    def record_click(event, x, y, flags, param):
        global REF_PTS
        if event == cv2.EVENT_LBUTTONDOWN:
            REF_PTS = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            REF_PTS.append((x, y))

    def confirm_directories(self,):
        if not self.input_bank.is_dir():
            self.input_bank.mkdir(parents=True)
        if not self.image_bank.is_dir():
            self.image_bank.mkdir(parents=True)
        if not self.annotation_file_bank.is_dir():
            self.annotation_file_bank.mkdir(parents=True)

    def load_object_detector(self,) -> None:
        weights_file = self.object_detector_settings['weights_file']
        cfg_file = self.object_detector_settings['cfg_file']
        class_names_file = self.object_detector_settings['class_names_file']
        resolution = self.object_detector_settings['resolution']
        class_filters = self.object_detector_settings['class_filters']
        self.object_detector = YOLO(cfg_file,
                                    weights_file,
                                    class_names_file,
                                    resolution,
                                    class_filters)

    def write_results(self, results: FrameAnnotations) -> str:
        image_path = self.image_bank / f"{self.frame_counter}.jpg"
        annotation_path = self.annotation_file_bank / \
            f"{self.frame_counter}.txt"
        cv2.imwrite(str(image_path), results.frame)
        with open(annotation_path, "w") as f:
            image_width = results.image_width
            image_height = results.image_height
            f.write("{class_name},{left},{top},{right}," +
                    "{bottom},{image_width},{image_height}\n")
            for obj in results.objects:
                f.write(f"{obj.class_name},{obj.bbox.left},{obj.bbox.top}" +
                        f",{obj.bbox.right},{obj.bbox.bottom},{image_width}," +
                        f"{image_height}\n")
        self.frame_counter += 1
        return image_path

    def get_detections(self,
                       frame: np.ndarray,
                       threshold: float = 0.7) -> FrameAnnotations:
        return self.object_detector.get_detections(frame, threshold)

    def draw_boxes(self,
                   frame: np.ndarray,
                   results: FrameAnnotations) -> bool:
        for result in results.objects:
            frame_height, frame_width = frame.shape[:2]
            # Drawing parameters
            thickness = int((2 * frame_width) / 1664)
            thickness = 1 if thickness == 0 else thickness

            bounds = result.bbox
            x1 = bounds.left
            y1 = bounds.top
            x2 = bounds.right
            y2 = bounds.bottom
            bbx_x1 = x1 if x1 > 0 else 1
            bbx_y1 = y1 if y1 > 0 else 1
            bbx_x2 = x2 if x2 < frame_width else frame_width - 1
            bbx_y2 = y2 if y2 < frame_height else frame_height - 1

            cv2.rectangle(img=frame,
                          pt1=(bbx_x1, bbx_y1),
                          pt2=(bbx_x2, bbx_y2),
                          color=(255, 0, 0),
                          thickness=thickness)
        return frame

    def process_inputs(self) -> None:
        self.load_object_detector()
        for input_file in self.input_bank.iterdir():
            cap = cv2.VideoCapture(str(input_file))
            frame_count = -1
            while True:
                r, frame = cap.read()
                if not r:
                    cap.release()
                    break
                results = self.get_detections(frame, self.threshold)
                if len(results.objects) == 0:
                    continue
                frame_count += 1
                if frame_count % self.frame_jump:
                    continue
                if not isinstance(results, FrameAnnotations):
                    raise ValueError(
                            "results were not of type 'FrameAnnotations'")
                self.write_results(results)
                frame = self.draw_boxes(frame, results)
                cv2.imshow("image", frame)
                cv2.moveWindow("image", 20,20);
                q = cv2.waitKey(1)
                if q & 0xFF == ord('q'):
                    print("Stopping annotation generator")
                    cv2.destroyAllWindows()
                    break

    def draw_new_box(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        cv2.destroyAllWindows()
        cv2.namedWindow("clickable_image")
        cv2.moveWindow("clickable_image", 20,20);
        cv2.setMouseCallback("clickable_image",
                             BoundingBoxAnnotator.record_click)
        output = "{class_name},{left},{top},{right},{bottom}," +\
            "{image_width},{image_height}"
        old_frame = frame.copy()
        print("Drawing new box")
        print("Press [q] at any time to quit this new box drawing session")
        while True:
            global REF_PTS
            frame = old_frame.copy()
            if len(REF_PTS) == 2:
                cv2.rectangle(frame, REF_PTS[0], REF_PTS[1], (0, 255, 0), 1)
                cv2.imshow('clickable_image', frame)
                print("Press [y] to confirm box, else press any other key.")
                key = cv2.waitKey(0)
                if key & 0xFF == ord('y'):
                    print("You pressed [y], adding the new bounding box")
                    break
                if key & 0xFF == ord('q'):
                    print("Quitting new box session")
                    return
                REF_PTS = list()
            else:
                cv2.imshow('clickable_image', frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    print("Quitting new box session")
                    return

        # This would happen if you click right and drag left to draw box
        # drawn rt to lb
        if REF_PTS[0][0] > REF_PTS[1][0] and REF_PTS[0][1] < REF_PTS[1][1]:
            rt = REF_PTS[0]
            lb = REF_PTS[1]
            REF_PTS[0] = (lb[0], rt[1])
            REF_PTS[1] = (rt[0], lb[1])
        # drawn rb to lt
        elif REF_PTS[0][0] > REF_PTS[1][0] and REF_PTS[0][1] > REF_PTS[1][1]:
            temp = REF_PTS[0]
            REF_PTS[0] = REF_PTS[1]
            REF_PTS[1] = temp
        # drawn lb to rt
        elif REF_PTS[0][0] < REF_PTS[1][0] and REF_PTS[0][1] > REF_PTS[1][1]:
            lb = REF_PTS[0]
            rt = REF_PTS[1]
            REF_PTS[0] = (lb[0], rt[1])
            REF_PTS[1] = (rt[0], lb[1])
        result = output.format(class_name='person',
                               left=REF_PTS[0][0],
                               top=REF_PTS[0][1],
                               right=REF_PTS[1][0],
                               bottom=REF_PTS[1][1],
                               image_width=frame.shape[1],
                               image_height=frame.shape[0])
        REF_PTS = list()
        cv2.destroyAllWindows()
        return frame, result

    def get_image_and_annotations(self, image) -> Tuple[np.ndarray,
                                                        List[str], List[str]]:
        new_annotations = list()
        image_file_name = str(image)
        frame = cv2.imread(image_file_name)
        annotation_file_name = self.annotation_file_bank / \
            (image_file_name.split('/')[-1][:-3] + "txt")
        annotations = list()
        with open(annotation_file_name, "r") as f:
            annotations = f.readlines()
        if not annotations:
            return frame, []
        new_annotations.append(annotations[0].strip())
        annotations = annotations[1:]
        return frame, annotations, new_annotations, annotation_file_name

    def dynamic_confirmation(self,) -> None:
        self.load_object_detector()
        print("loaded object detector")
        for input_file in self.input_bank.iterdir():
            print(f"processing {input_file}")
            cap = cv2.VideoCapture(str(input_file))
            frame_count = -1
            while True:
                r, frame = cap.read()
                if not r:
                    cap.release()
                    break
                results = self.get_detections(frame, self.threshold)
                if len(results.objects) == 0:
                    continue
                frame_count += 1
                if frame_count % self.frame_jump:
                    continue
                if not isinstance(results, FrameAnnotations):
                    raise ValueError(
                            "results were not of type 'FrameAnnotations'")
                print("Got Detections\n\n")
                image_path = self.write_results(results)
                frame, annotations, new_annotations, annotation_file_name = \
                        self.get_image_and_annotations(image_path)
                self.confirm_box(frame, annotations, new_annotations, annotation_file_name)

    def confirm_boxes(self,) -> None:
        start = False
        for image in self.image_bank.iterdir():
            frame, annotations, new_annotations, annotation_file_name = \
                    self.get_image_and_annotations(image)
            if annotation_file_name == pathlib.Path(self.annotation_file_bank,
                                                    self.start_at):
                start = True
            if not start:
                continue
            self.confirm_box(frame, annotations, new_annotations, annotation_file_name)

    def confirm_box(self, frame, annotations, new_annotations, annotation_file_name) -> None:
        for annotation in annotations:
            annotation = annotation.strip()
            annotation_str = annotation
            frame_copy = frame.copy()
            annotation = annotation.split(',')
            if len(annotation) != 7:
                print(f"Annotation {annotation} is of bad format")
                continue
            class_name, left, top, right, \
                bottom, width, height = annotation
            single_box = FrameAnnotations(
                    frame=frame,
                    objects=[Object(class_name=class_name,
                                    bbox=BBox(left=int(left),
                                              top=int(top),
                                              right=int(right),
                                              bottom=int(bottom),))],
                    image_width=int(width),
                    image_height=int(height),)
            frame_copy = self.draw_boxes(frame_copy, single_box)
            while True:
                cv2.imshow("image", frame_copy)
                cv2.moveWindow("image", 20,20);
                print("Press [y] to confirm this box or [n] to delete it")
                print("Pres [q] to fully quit the program.")
                key = cv2.waitKey(0)
                if key & 0xFF == ord('y'):
                    print("Confirmed")
                    new_annotations.append(annotation_str)
                    frame = self.draw_boxes(frame, single_box)
                    break
                elif key & 0xFF == ord('n'):
                    print("Deleted")
                    break
                elif key & 0xFF == ord('q'):
                    print("Are you sure you want to quit [y]/[n]")
                    while True:
                        q = cv2.waitKey(0)
                        if q & 0xFF == ord('y'):
                            sys.exit(0)
                        elif q & 0xFF == ord('n'):
                            break
        print("Done with generated annotations, now for custom if things were missed.")
        while True:
            cv2.imshow("image", frame)
            cv2.moveWindow("image", 20,20);
            print("Press [a] to add a new annotation or [n] to move to " +
                  "next frame")
            key = cv2.waitKey(0)
            if key & 0xFF == ord('n'):
                print("Moving to next frame")
                break
            if key & 0xFF == ord('a'):
                results = self.draw_new_box(frame)
                if results is not None:
                    frame, new_box = results
                    new_annotations.append(new_box)

        self.write_new_results(annotation_file_name, new_annotations)
        cv2.destroyAllWindows()

    def write_new_results(self, annotation_file_name: str,
                          new_annotations: List[str],) -> None:
        annotation_file_name = str(annotation_file_name) + '.good'
        with open(annotation_file_name, "w") as f:
            for line in new_annotations:
                print(f"Saving New Annotations: {line}")
                f.write(line)
                f.write('\n')
        print(f"done with {str(annotation_file_name)}")
        print("\n\n")


parser = ArgumentParser(description=__doc__)
parser.add_argument('--cfg-file', default='yolov3.cfg',
                    type=str)
parser.add_argument('--weights-file', default='yolov3.weights',
                    type=str)
parser.add_argument('--coco-names-file', default='coco.names',
                    type=str)
parser.add_argument('--reso', default=640,
                    type=int)
parser.add_argument('--class-filters', default=['person'],
                    nargs="+", type=str)
parser.add_argument('--input-bank', default='input',
                    type=str)
parser.add_argument('--image-bank', default='data/images',
                    type=str)
parser.add_argument('--annotation-file-bank', default='data/annotations',
                    type=str)
parser.add_argument('--threshold', default=0.2,
                    type=int)
parser.add_argument('--frame-jump', default=50,
                    type=int)
parser.add_argument('--process-inputs', dest='process_inputs',
                    action='store_true')
parser.set_defaults(process_inputs=False)
parser.add_argument('--confirm-boxes', dest='confirm_boxes',
                    action='store_true')
parser.set_defaults(confirm_boxes=False)
parser.add_argument('--dynamic-confirmation', dest='dynamic_confirmation',
                    action='store_true')
parser.set_defaults(confirm_boxes=False)
parser.add_argument('--start-at', default='0.txt',
                    type=str)
args = parser.parse_args()
if __name__ == "__main__":
    object_detector_settings = {
            'cfg_file': args.cfg_file,
            'weights_file': args.weights_file,
            'class_names_file': args.coco_names_file,
            'resolution': args.reso,
            'class_filters': args.class_filters}
    input_bank = pathlib.Path(args.input_bank)
    image_bank = pathlib.Path(args.image_bank)
    annotation_file_bank = pathlib.Path(args.annotation_file_bank)
    annotator = BoundingBoxAnnotator(
            input_bank=input_bank,
            image_bank=image_bank,
            annotation_file_bank=annotation_file_bank,
            object_detector_settings=object_detector_settings,
            threshold=args.threshold,
            frame_jump=args.frame_jump,
            start_at=args.start_at)
    if args.process_inputs:
        for bank in [annotation_file_bank, image_bank]:
            if os.path.isdir(bank):
                shutil.rmtree(bank)
            os.mkdir(bank)
        annotator.process_inputs()
    if args.confirm_boxes:
        annotator.confirm_boxes()
    if args.dynamic_confirmation:
        annotator.dynamic_confirmation()
