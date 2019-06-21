import math
import time
from collections import defaultdict, deque
import numpy as np
import cv2
import os

import config

OBJECTS = config.LABELS
STATES = ["start", "wheel-stage", "wheel-compare"]
resources = os.path.abspath("resources/images")
video_url = "0.0.0.0:9095/"
stable_threshold = 20
wheel_compare_threshold = 20


class FrameRecorder:
    def __init__(self, size):
        self.deque = deque()
        self.size = size
        self.clear_count = 0

    def add(self, obj):
        self.deque.append(obj)

        if len(self.deque) > self.size:
            self.deque.popleft()

        self.clear_count = 0

    def is_center_stable(self):
        if len(self.deque) != self.size:
            return False

        prev_frame = self.deque[0]
        for i in range(1, len(self.deque)):
            frame = self.deque[i]
            diff = bbox_diff(frame["dimensions"], prev_frame["dimensions"])

            if diff > stable_threshold:
                return False

            prev_frame = frame

        return True

    def staged_clear(self):
        self.clear_count += 1
        if self.clear_count > self.size:
            self.clear()

    def clear(self):
        self.deque = deque()

    def averaged_bbox(self):
        out = [0, 0, 0, 0]

        for i in range(len(self.deque)):
            dim = self.deque[i]["dimensions"]
            for u in range(len(dim)):
                out[u] += dim[u]

        return [v / len(self.deque) for v in out]


class Task:
    def __init__(self, init_state=None):
        if init_state is None:
            self.current_state = "start"
        else:
            if init_state not in STATES:
                raise ValueError('Unknown init state: {}'.format(init_state))
            self.current_state = init_state

        self.left_hole_state = FrameRecorder(5)
        self.right_hole_state = FrameRecorder(5)

        self.left_wheel_state = FrameRecorder(5)
        self.right_wheel_state = FrameRecorder(5)

        self.last_id = None

        self.wait_count = 0


    def get_instruction(self, objects, header=None):
        if header is not None and "task_id" in header:
            if self.last_id is None:
                self.last_id = header["task_id"]
            elif self.last_id != header["task_id"]:
                self.last_id = header["task_id"]
                self.current_state = "start"

        result = defaultdict(lambda: None)
        result['status'] = "success"
        vis_objects = np.asarray([])

        # the start, branch into desired instruction
        if self.current_state == "start":
            self.current_state = "frame-branch"
            empty_holes = get_objects_by_categories(objects, {"hole_empty"})
            if len(empty_holes) > 0:
                self.current_state = "frame-branch"
            else:
                self.wait_count += 1
                time.sleep(1)
                if self.wait_count > 5:
                    self.current_state = "wheel-branch"

        elif self.current_state == "frame-branch":
            result['speech'] = "Please show me the black frame, holding it like this."
            image_path = os.path.join(resources, "frame-empty-hole.jpg")
            result['image'] = cv2.imread(image_path)
            self.current_state = "frame-empty-hole"

        elif self.current_state == "frame-empty-hole":
            empty_holes = get_objects_by_categories(objects, {"hole_empty"})
            if len(empty_holes) == 2:
                left, right = separate_left_right(objects)
                self.left_hole_state.add(left)
                self.right_hole_state.add(right)

                if self.left_hole_state.is_center_stable() and self.right_hole_state.is_center_stable():
                    result['speech'] = "Great! Please put in the green washer into the slot on the right, like this."
                    result['video'] = video_url + "frame-green-washer-insert.mp4"
                    self.current_state = "frame-green-washer-holes"

                    self.left_hole_state.clear()
                    self.right_hole_state.clear()
                else:
                    self.left_hole_state.staged_clear()
                    self.right_hole_state.staged_clear()

        elif self.current_state == "frame-green-washer-holes":
            holes = get_objects_by_categories(objects, {"hole_empty", "hole_green", "hole_gold"})
            if len(holes) == 2:
                _, right = separate_left_right(objects)
                if right["class_name"] == "hole_green":
                    self.right_hole_state.add(right)

                    if self.right_hole_state.is_center_stable():
                        result['speech'] = "Nice! Finally, fit the gold washer into the green washer, like this. Make sure the smaller end goes in first."
                        result['video'] = video_url + "frame-gold-washer-insert.mp4"
                        self.current_state = "frame-gold-washer-holes"
                    else:
                        self.right_hole_state.staged_clear()

        elif self.current_state == "frame-gold-washer-holes":
            holes = get_objects_by_categories(objects, {"hole_empty", "hole_green", "hole_gold"})
            if len(holes) == 2:
                _, right = separate_left_right(objects)
                if right["class_name"] == "hole_gold":
                    self.right_hole_state.add(right)

                    if self.right_hole_state.is_center_stable():
                        result[
                            'speech'] = "Excellent job!"
                        result['video'] = video_url + "frame-gold-washer-insert.mp4"
                        self.current_state = "nothing"
                    else:
                        self.right_hole_state.staged_clear()



        elif self.current_state == "wheel-branch":
            result['speech'] = "Please grab one each of the big and small wheels."
            image_path = os.path.join(resources, "wheel-stage-1.jpg")
            result['image'] = cv2.imread(image_path)
            self.current_state = "wheel-stage-1"

        elif self.current_state == "wheel-stage-1":
            wheels = get_objects_by_categories(objects, {"thick_wheel_top", "thin_wheel_top"})
            if len(wheels) == 2:
                result["speech"] = "Excellent! Please line them up with the legend."
                image_path = os.path.join(resources, "tire-legend.png")
                result['legend'] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                self.current_state = "wheel-stage-2"

        elif self.current_state == "wheel-stage-2":
            wheels = get_objects_by_categories(objects, {"thick_wheel_side", "thin_wheel_side", "thick_tire", "thin_tire"})
            if len(wheels) == 2:
                left, right = separate_left_right(objects)
                print("left: %s    right: %s    diff: %s" % (bbox_height(left["dimensions"]), bbox_height(right["dimensions"]), abs(bbox_height(left["dimensions"]) - bbox_height(right["dimensions"]))))

                self.left_wheel_state.add(left)
                self.right_wheel_state.add(right)

                if self.left_wheel_state.is_center_stable() and self.right_wheel_state.is_center_stable():
                    compare = wheel_compare(self.left_wheel_state.averaged_bbox(), self.right_wheel_state.averaged_bbox(), wheel_compare_threshold)

                    if compare == "same":
                        result["speech"] = "Those wheels are the same size. Please get two different-sized wheels."
                        self.left_wheel_state.clear()
                        self.right_wheel_state.clear()

                    else:
                        left_speech, right_speech = ("bigger", "smaller") if compare == "first" else ("smaller", "bigger")
                        result["speech"] = "Great job! The one on the left is the %s wheel and the one on the right is the %s wheel" % (left_speech, right_speech)
                        self.current_state = "nothing"
            else:
                self.left_wheel_state.staged_clear()
                self.right_wheel_state.staged_clear()

        elif self.current_state == "nothing":
            time.sleep(3)
            self.current_state = "start"


        return vis_objects, result



def get_objects_by_categories(objects, categories):
    out = []
    for obj in objects:
        if obj["class_name"] in categories:
            out.append(obj)
    return out


def separate_left_right(objects):
    obj1 = objects[0]
    obj2 = objects[1]

    if obj1["dimensions"][0] < obj2["dimensions"][0]:
        left = obj1
        right = obj2
    else:
        left = obj2
        right = obj1

    return left, right

def bbox_center(dims):
    return dims[2] - dims[0], dims[3] - dims[1]

def bbox_height(dims):
    return dims[3] - dims[1]

def bbox_diff(box1, box2):
    center1 = bbox_center(box1)
    center2 = bbox_center(box2)

    x_diff = abs(center1[0] - center2[0])
    y_diff = abs(center1[1] - center2[1])

    return math.sqrt(x_diff**2 + y_diff**2)

def wheel_compare(box1, box2, threshold):
    height1 = bbox_height(box1)
    height2 = bbox_height(box2)

    diff = abs(height1 - height2)
    if diff < threshold:
        return "same"

    return "first" if height1 > height2 else "second"
