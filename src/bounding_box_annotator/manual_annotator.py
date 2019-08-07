# TODO: Make Frame class that has frame: np.ndarray and annotions: List[str] as
# @ members that can handle all the drawing on its own
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple, List
import importlib.resources
import shutil
import sys
import os

import scipy.io
import numpy as np
import cv2

from mapsdg.google_maps_api import GoogleMapsAPI
from mapsdg.components import LatLon
import mapsdg

from .components import RoomType, RoomAnnotation, FloorPlanAnnotation, BBox

REF_PTS = list()
API_KEY = importlib.resources.read_text('bounding_box_annotator', '.api_key')


class ManualBoxAnnotator:
    def __init__(self,
                 input_bank: List[Path],
                 save_to: Path,
                 dataset: str) -> None:
        self.input_bank = input_bank
        self.save_to = save_to
        self.frame_counter = 0
        choices = {'sydney-house': self.write_results_sydney_house}
        self.write_results = choices[dataset]
        self.build_directories()
        self.counter = 0

    @staticmethod
    def record_click(event, x, y, flags, param):
        global REF_PTS
        if event == cv2.EVENT_LBUTTONDOWN:
            REF_PTS = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            REF_PTS.append((x, y))

    def build_directories(self):
        if not self.save_to.is_dir():
            self.save_to.mkdir(parents=True)
        children = []
        for child in children:
            new_folder = self.save_to / child
            if not new_folder.is_dir():
                new_folder.mkdir(parents=True)

    def write_results_sydney_house(self, house: str,
                                   results: FloorPlanAnnotation) -> None:
        save_dir = self.save_to / house
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"floor_{results.floor_num}.jpg"
        cv2.imwrite(str(image_path), results.floor_plan)
        annotation_path = save_dir / f"floor_{results.floor_num}.txt"
        with open(annotation_path, "w") as f:
            image_width = results.image_width
            image_height = results.image_height
            f.write("{class_name},{left},{top},{right}," +
                    "{bottom},{image_width},{image_height}\n")
            for room in results.rooms:
                f.write(f"{room.class_name},{room.bbox.left}," +
                        f"{room.bbox.top},{room.bbox.right}," +
                        f"{room.bbox.bottom},{image_width},{image_height}\n")

    def draw_boxes(self,
                   frame: np.ndarray,
                   results: FloorPlanAnnotation) -> bool:
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

    def conform_corners(
            self,
            ref_pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        '''
        Conform to TL BR coordinates
        '''
        # This would happen if you click right and drag left to draw box
        # drawn rt to lb
        if ref_pts[0][0] > ref_pts[1][0] and ref_pts[0][1] < ref_pts[1][1]:
            rt = ref_pts[0]
            lb = ref_pts[1]
            ref_pts[0] = (lb[0], rt[1])
            ref_pts[1] = (rt[0], lb[1])
        # drawn rb to lt
        elif ref_pts[0][0] > ref_pts[1][0] and ref_pts[0][1] > ref_pts[1][1]:
            temp = ref_pts[0]
            ref_pts[0] = ref_pts[1]
            ref_pts[1] = temp
        # drawn lb to rt
        elif ref_pts[0][0] < ref_pts[1][0] and ref_pts[0][1] > ref_pts[1][1]:
            lb = ref_pts[0]
            rt = ref_pts[1]
            ref_pts[0] = (lb[0], rt[1])
            ref_pts[1] = (rt[0], lb[1])

        return ref_pts

    def crop(self, frame: np.ndarray) -> np.ndarray:
        print('\n')
        print("CROPING")
        cv2.destroyAllWindows()
        cv2.namedWindow("clickable_image")
        cv2.moveWindow("clickable_image", 0, 0)
        cv2.setMouseCallback("clickable_image",
                             ManualBoxAnnotator.record_click)
        old_frame = frame.copy()
        while True:
            global REF_PTS
            if len(REF_PTS) == 2 and REF_PTS[0] != REF_PTS[1]:
                REF_PTS = self.conform_corners(REF_PTS)
                cv2.imshow('clickable_image',
                           frame[REF_PTS[0][1]:REF_PTS[1][1],
                                 REF_PTS[0][0]:REF_PTS[1][0]])
                cv2.moveWindow("clickable_image", 0, 0)
                print('\n')
                print("[y] to confirm")
                print("[*] to decline")
                print("[q] to quit")
                key = cv2.waitKey(0)
                if key & 0xFF == ord('y'):
                    print("confirmed")
                    frame = frame[REF_PTS[0][1]:REF_PTS[1][1],
                                  REF_PTS[0][0]:REF_PTS[1][0]]
                    while True:
                        # current_floor = input(
                        #     "Current floor num [-2, -1, 0, 1, 2, ...]: ")
                        print("Current floor num [-2, -1, 0, 1, 2, ...]?")
                        key = cv2.waitKey(0)
                        if chr(key & 0xFF) == 'l':
                            current_floor = -1
                        else:
                            current_floor = chr(key & 0xFF)
                        print(f"Floor = {current_floor}")
                        try:
                            current_floor = int(current_floor)
                            break
                        except Exception:
                            continue
                    REF_PTS = list()
                    return frame, current_floor  # result
                else:
                    frame = old_frame.copy()
                REF_PTS = list()
            else:
                cv2.imshow('clickable_image', frame)
                key = cv2.waitKey(1)
        REF_PTS = list()
        cv2.destroyAllWindows()

    def draw_boxes_sydney_house(
            self,
            frame: np.ndarray) -> Tuple[np.ndarray, str]:
        rooms = list()
        frame, floor_num = self.crop(frame)
        original_frame = frame.copy()
        image_height, image_width, _ = frame.shape
        while True:
            cv2.destroyAllWindows()
            cv2.namedWindow("clickable_image")
            cv2.setMouseCallback("clickable_image",
                                 ManualBoxAnnotator.record_click)
            old_frame = frame.copy()
            print("Drawing new box")
            print("Press [q] at any time to quit this new box drawing session")
            while True:
                global REF_PTS
                frame = old_frame.copy()
                if len(REF_PTS) == 2:
                    cv2.rectangle(
                        frame, REF_PTS[0], REF_PTS[1], (0, 255, 0), 1)
                    cv2.imshow('clickable_image', frame)
                    cv2.moveWindow("clickable_image", 0, 0)
                    print(
                        "Press [y] to confirm box, else press any other key.")
                    key = cv2.waitKey(0)
                    if key & 0xFF == ord('y'):
                        print("You press [y], creating new box annotation")
                        for i, r in enumerate(RoomType):
                            if i == 9:
                                print(f"g: {r}")
                            elif i == 10:
                                print(f"c: {r}")
                            elif i == 11:
                                print(f"o: {r}")
                            else:
                                print(f"{i + 1}: {r}")
                        while True:
                            try:
                                print("Class name?")
                                key = cv2.waitKey(0)
                                if chr(key & 0xFF) == 'c':
                                    key = 11
                                elif chr(key & 0xFF) == 'g':
                                    key = 10
                                elif chr(key & 0xFF) == 'o':
                                    key = 12
                                else:
                                    key = int(chr(key & 0xFF))
                                class_name = int(key)
                                class_name = RoomType(class_name).name
                                print(f"{key} = {class_name}")
                                break
                            except Exception:
                                print('Need a valid int')
                        old_frame = frame
                        REF_PTS = self.conform_corners(REF_PTS)
                        result = RoomAnnotation(class_name=class_name,
                                                bbox=BBox(
                                                    left=REF_PTS[0][0],
                                                    top=REF_PTS[0][1],
                                                    right=REF_PTS[1][0],
                                                    bottom=REF_PTS[1][1]))
                        rooms.append(result)
                        REF_PTS = list()
                        break
                    elif key & 0xFF == ord('q'):
                        print("Quitting")
                        return original_frame, FloorPlanAnnotation(
                            floor_num=floor_num,
                            floor_plan=original_frame.copy(),
                            rooms=rooms, image_width=image_width,
                            image_height=image_height)
                    REF_PTS = list()
                else:
                    cv2.imshow('clickable_image', frame)
                    cv2.moveWindow("clickable_image", 0, 0)
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        print("Quitting new box session")
                        return original_frame, FloorPlanAnnotation(
                            floor_num=floor_num,
                            floor_plan=original_frame.copy(),
                            rooms=rooms, image_width=image_width,
                            image_height=image_height)
        REF_PTS = list()
        return original_frame, FloorPlanAnnotation(
            floor_num=floor_num,
            floor_plan=original_frame.copy(),
            rooms=rooms, image_width=image_width,
            image_height=image_height)

    def annotate_sydney_house(self,) -> None:
        for image in self.input_bank:
            self.counter += 1
            print(f"{self.counter}/{len(self.input_bank)}: {image}")
            frame = cv2.imread(str(image))
            height, width, channels = frame.shape
            if height > 1080:
                ratio = 1080 / height
                frame = cv2.resize(frame, (int(width * ratio), 1080))
            frame_copy = frame.copy()
            skip = False
            while True:
                cv2.namedWindow('temp')
                cv2.moveWindow("temp", 0, 0)
                cv2.imshow('temp', frame_copy)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('s'):
                    skip = True
                    break
                if key & 0xFF == ord('n'):
                    break
            if skip:
                continue
            frame, results = self.draw_boxes_sydney_house(frame_copy)
            image_split = str(image).split('/')
            self.write_results(house=image_split[3], results=results)
            while True:
                x = input("More floors?: ")
                if x == 'yes':
                    frame, results = self.draw_boxes_sydney_house(frame_copy)
                    image_split = str(image).split('/')
                    self.write_results(house=image_split[3], results=results)
                elif x == 'no':
                    break

    def align_boxes_with_sat_image(self, sydney_data_loc: Path) -> None:
        gapi = GoogleMapsAPI(key=API_KEY)
        counter = 0
        for house_folder in self.input_bank:
            house_num = str(house_folder).split('/')[-1]
            house = sydney_data_loc / house_num
            mat = scipy.io.loadmat(str(house / 'location_data.mat'))
            floor_plan = cv2.imread(str(house / 'floorplan.png'))
            height, width, channels = floor_plan.shape
            if height > 1080:
                ratio = 1080 / height
                floor_plan = cv2.resize(floor_plan, (int(width * ratio), 1080))
            url = gapi.get_static_image_url(addr=LatLon(
                lat=mat['location_data'][0][0][0][0][0],
                lon=mat['location_data'][0][0][1][0][0]),
                image_zoom=20)
            image = mapsdg.get_image_from_url(url)
            if image is not None:
                print(f"{counter} / {len(self.input_bank)}")
                counter += 1
                skip = False
                if counter < 63:
                    continue
                # files = [x for x in house_folder.iterdir() if x.is_file()]
                # if len(files) <= 2:
                while True:
                    cv2.imshow('', image)
                    key = cv2.waitKey(0)
                    if chr(key & 0xFF) == 's':
                        skip = True
                        break
                    if chr(key & 0xFF) == 'y':
                        break
                if skip:
                    continue

                image = self.rotate_zoom_sat_image(image, floor_plan)
                boxes = self.fit_boxes(image, house_folder, floor_plan)
                self.save_aligned_boxes(image, boxes, house_num)

    def save_aligned_boxes(
            self,
            image: np.ndarray,
            boxes: List[Tuple[str, int, int, int, int, int, int]],
            house_num: str,) -> None:
        yolo_classes = [floor + "_" + room
                        for floor in ['basement', 'main',
                                      'first', 'second']
                        for room in ['staircase', 'entrance', 'laundry',
                                     'bed', 'bath', 'kitchen', 'living',
                                     'dining', 'hallway', 'garage',
                                     'closet', 'other']]
        save_dir = self.save_to / house_num
        save_dir.mkdir(parents=True, exist_ok=True)
        image_path = save_dir / f"{house_num}.jpg"
        cv2.imwrite(str(image_path), image)
        annotation_path = save_dir / f"{house_num}.txt"
        with open(annotation_path, "w") as f:
            image_height, image_width, _ = image.shape
            f.write("{class_name},{left},{top},{right}," +
                    "{bottom},{image_width},{image_height}\n")
            for box in boxes:
                class_name, left, top, right, bottom, w, h = box
                f.write(f"{class_name},{left}," +
                        f"{top},{right}," +
                        f"{bottom},{image_width},{image_height}\n")
        annotation_path = save_dir / f"{house_num}_yolo.txt"
        with open(annotation_path, "w") as f:
            image_height, image_width, _ = image.shape
            for box in boxes:
                class_name, left, top, right, bottom, w, h = box
                left = float(left)
                top = float(top)
                right = float(right)
                bottom = float(bottom)
                image_width = float(image_width)
                image_height = float(image_height)
                center_x = ((left + right) / 2) / image_width
                center_y = ((top + bottom) / 2) / image_height
                width = (right - left) / image_width
                height = (bottom - top) / image_height
                f.write(f"{yolo_classes.index(class_name)} {center_x} " +
                        f"{center_y} {width} " +
                        f"{height}\n")

    def fit_boxes(self, image: np.ndarray,
                  house_folder: Path, floor_plan: np.ndarray) -> None:
        all_boxes = list()
        image_h, image_w, _ = image.shape
        for _file in house_folder.iterdir():
            print(_file)
            if _file.suffix == '.txt':
                cv2.destroyAllWindows()
                cv2.namedWindow("clickable_image")
                cv2.moveWindow("clickable_image", 0, 0)
                cv2.imshow('clickable_image', floor_plan)
                if 'floor_0' in str(_file):
                    floor = 'main'
                elif 'floor_1' in str(_file):
                    floor = 'first'
                elif 'floor_2' in str(_file):
                    floor = 'second'
                elif 'floor_-1' in str(_file):
                    floor = 'basement'
                with open(_file, 'r') as f:
                    lines = f.readlines()[1:]
                boxes = list()
                for line in lines:
                    class_name, left, top, right, bottom, w, h = \
                        line.split(',')
                    boxes.append((f"{floor}_{class_name}",
                                  int(left), int(top), int(right), int(bottom),
                                  int(w), int(h)))
                image_clean = image.copy()
                scale_w = 1.05
                scale_h = 1.05
                while True:
                    image = image_clean.copy()
                    height, width, _ = image.shape
                    for box in boxes:
                        class_name, left, top, right, bottom, w, h = box
                        color = (0, 0, 255)
                        if 'staircase' in class_name:
                            color = (255, 0, 0)
                        cv2.rectangle(image, (left, top), (right, bottom),
                                      color=color, thickness=1)
                    cv2.imshow('', image)
                    key = cv2.waitKey(0)
                    if chr(key & 0xFF) == 'y':
                        cv2.destroyAllWindows()
                        break
                    new_boxes = list()
                    for box in boxes:
                        class_name, left, top, right, bottom, w, h = box
                        if chr(key & 0xFF) == 'k':
                            top -= 3
                            bottom -= 3
                        elif chr(key & 0xFF) == 'j':
                            top += 3
                            bottom += 3
                        elif chr(key & 0xFF) == 'h':
                            left -= 3
                            right -= 3
                        elif chr(key & 0xFF) == 'l':
                            left += 3
                            right += 3
                        elif chr(key & 0xFF) == 'w':
                            top = int(top * scale_h)
                            bottom = int(bottom * scale_h)
                        elif chr(key & 0xFF) == 's':
                            top = int(top / scale_h)
                            bottom = int(bottom / scale_h)
                        elif chr(key & 0xFF) == 'a':
                            left = int(left / scale_w)
                            right = int(right / scale_w)
                        elif chr(key & 0xFF) == 'd':
                            left = int(left * scale_w)
                            right = int(right * scale_w)
                        new_boxes.append((class_name, left, top, right, bottom,
                                          image_w, image_h))
                    boxes = new_boxes
                all_boxes += boxes
        return all_boxes

    def rotate_zoom_sat_image(self, image: np.ndarray,
                              floor_plan: np.ndarray) -> np.ndarray:
        # cv2.destroyAllWindows()
        # cv2.namedWindow("clickable_image")
        # cv2.moveWindow("clickable_image", 0, 0)
        # cv2.setMouseCallback("clickable_image",
        #                      ManualBoxAnnotator.record_click)
        original_image = image.copy()
        image_clean = image.copy()
        rotation = 0
        scale = 1
        translation_x = 0
        translation_y = 0
        cv2.namedWindow("clickable_image")
        cv2.moveWindow("clickable_image", 0, 0)
        cv2.imshow('clickable_image', floor_plan)
        while True:
            rows, cols, _ = image.shape
            for i in range(0, cols, 20):
                cv2.line(image, (i, 0), (i, rows), (0, 0, 255), thickness=1)
            for j in range(0, rows, 20):
                cv2.line(image, (0, j), (cols, j), (0, 0, 255), thickness=1)
            cv2.imshow("", image)
            key = cv2.waitKey(0)
            if chr(key & 0xFF) == 'k':
                rotation += 0.2
            elif chr(key & 0xFF) == 'j':
                rotation -= 0.2
            elif chr(key & 0xFF) == '[':
                rotation += 1
            elif chr(key & 0xFF) == ']':
                rotation -= 1
            elif chr(key & 0xFF) == 'z':
                scale += 0.1
            elif chr(key & 0xFF) == 'x':
                scale -= 0.1
            elif chr(key & 0xFF) == 'w':
                translation_y += 4
            elif chr(key & 0xFF) == 's':
                translation_y -= 4
            elif chr(key & 0xFF) == 'a':
                translation_x += 4
            elif chr(key & 0xFF) == 'd':
                translation_x -= 4
            elif chr(key & 0xFF) == 'y':
                cv2.destroyAllWindows()
                return image_clean
            elif chr(key & 0xFF) == 'q':
                cv2.destroyAllWindows()
                return original_image
            matrix = cv2.getRotationMatrix2D((cols/2, rows/2), rotation, 1)
            image = cv2.warpAffine(original_image, matrix, (cols, rows))
            image_clean = cv2.warpAffine(original_image, matrix, (cols, rows))

            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
            image_clean = cv2.resize(image_clean, None, fx=scale, fy=scale,
                                     interpolation=cv2.INTER_CUBIC)

            matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            image = cv2.warpAffine(image, matrix, (cols, rows))
            image_clean = cv2.warpAffine(image_clean, matrix, (cols, rows))


if __name__ == "__main__":
    # houses_folder = Path('sydney-house/rent_crawler/goodhouses')
    # houses = list()
    # for house_folder in houses_folder.iterdir():
    #     for house in house_folder.iterdir():
    #         if 'floorplan.png' in str(house) and house.suffix == '.png':
    #             houses.append(house)

    # annotator = ManualBoxAnnotator(
    #     houses,
    #     Path('processed_houses/floor-plans'),
    #     dataset='sydney-house')
    # annotator.annotate_sydney_house()

    houses_folder = Path('processed_houses/floor-plans/')
    houses = list()
    for house_folder in houses_folder.iterdir():
        houses.append(house_folder)

    annotator = ManualBoxAnnotator(
        houses,
        Path('final_dataset/'),
        dataset='sydney-house')
    annotator.align_boxes_with_sat_image(
        Path('sydney-house/rent_crawler/goodhouses'))
