import math
import time
from collections import defaultdict, deque
import numpy as np
import cv2
import os
from requests import get

import config
from car_model import CarModel, COMPONENTS_SPEECH, STATES_SPEECH

ip = get('https://api.ipify.org').text

OBJECTS = config.LABELS
STATES = ["start", "wheel-stage", "wheel-compare"]
resources = os.path.abspath("resources/images")
video_url = "http://" + ip + ":9095/"
stable_threshold = 20
wheel_compare_threshold = 20


class FrameRecorder:
    def __init__(self, size):
        self.deque = deque()
        self.size = size
        self.clear_count = 0

        self.obj_class = None

    def add(self, obj):
        if self.obj_class is None:
            self.obj_class = obj["class_name"]

        if self.obj_class != obj["class_name"]:
            self.clear()
            self.obj_class = obj["class_name"]

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
    def __init__(self):
        self.model = CarModel()

        self.frame_recs = defaultdict(lambda: FrameRecorder(5))

        self.last_id = None


    def get_instruction(self, objects, header=None):
        if header is not None and "task_id" in header:
            if self.last_id is None:
                self.last_id = header["task_id"]
            elif self.last_id != header["task_id"]:
                self.last_id = header["task_id"]
                self.model = CarModel()

        result = defaultdict(lambda: None)
        result['status'] = "success"
        vis_objects = np.asarray([])

        holes = get_objects_by_categories(objects, {"hole_empty", "hole_green", "hole_gold"})
        side_marker = get_objects_by_categories(objects, {"frame_marker_left", "frame_marker_right"})
        horn = get_objects_by_categories(objects, {"frame_horn"})

        # stop if bad frame
        if len(holes) != 2 or len(side_marker) != 1 or len(horn) != 1:
            return vis_objects, result

        left_obj, right_obj = separate_left_right(holes)
        self.frame_recs[0].add(left_obj)
        self.frame_recs[1].add(right_obj)
        self.frame_recs[2].add(side_marker[0])

        if not self.frame_recs[0].is_center_stable() or not self.frame_recs[1].is_center_stable() \
                or not self.frame_recs[2].is_center_stable():
            return vis_objects, result

        left_comp, right_comp = get_orientation(side_marker[0], horn[0])
        left_speech = speech_from_update(*self.model.check_update(left_comp, left_obj["class_name"]))
        right_speech = speech_from_update(*self.model.check_update(right_comp, right_obj["class_name"]))

        if left_speech != "":
            result["speech"] = left_speech
        elif right_speech != "":
            result["speech"] = right_speech

        self.clear_frame_recs()

        return vis_objects, result

    def clear_frame_recs(self):
        for frame_rec in self.frame_recs.values():
            frame_rec.clear()


def get_orientation(side_marker, horn):
    side = "left" if side_marker["class_name"] == "frame_marker_left" else "right"

    left_obj, right_obj = separate_left_right([side_marker, horn])
    if side == "left":
        flipped = left_obj["class_name"] == "frame_horn"
    else:
        flipped = right_obj["class_name"] == "frame_horn"

    if side == "left":
        left_hole_fb = "back"
        right_hole_fb = "front"
    else:
        left_hole_fb = "front"
        right_hole_fb = "back"

    if flipped:
        placeholder = left_hole_fb
        left_hole_fb = right_hole_fb
        right_hole_fb = placeholder

    return "hole_%s_%s" % (side, left_hole_fb), "hole_%s_%s" % (side, right_hole_fb)

def speech_from_update(update, data):
    if update == "next_step":
        return "You put the %s into the %s." % \
               (STATES_SPEECH[data["actual_state"]], COMPONENTS_SPEECH[data["actual_comp"]])
    if update == "back_step":
        return "You took the %s out of the %s." % \
               (STATES_SPEECH[data["before_state"]], COMPONENTS_SPEECH[data["actual_comp"]])

    return ""


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
    return dims[0] + (dims[2] - dims[0])/2, dims[1] + (dims[3] - dims[1])/2

def bbox_diff(box1, box2):
    center1 = bbox_center(box1)
    center2 = bbox_center(box2)

    x_diff = abs(center1[0] - center2[0])
    y_diff = abs(center1[1] - center2[1])

    return math.sqrt(x_diff**2 + y_diff**2)

