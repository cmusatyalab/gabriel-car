import math
import time
from collections import defaultdict, deque
import numpy as np
import cv2
import os

import config

OBJECTS = config.LABELS
STATES = ["start", "wheel-stage", "wheel-compare", "tire-rim-pairing"]
images_store = os.path.abspath("images_feedback")
stable_threshold = 50
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

    def is_center_stable(self, threshold):
        if len(self.deque) != self.size:
            return False

        prev_frame = self.deque[0]
        for i in range(1, len(self.deque)):
            frame = self.deque[i]
            diff = bbox_diff(frame["dimensions"], prev_frame["dimensions"])

            if diff > threshold:
                return False

            prev_frame = frame

        return True

    def staged_clear(self):
        self.clear_count += 1
        if self.clear_count > self.size / 2:
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

        self.left_frames = FrameRecorder(10)
        self.right_frames = FrameRecorder(10)

        self.last_id = None


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

        # the start
        if self.current_state == "start":
            result['speech'] = "Please show me all four rims and four tires."
            image_path = os.path.join(images_store, "tire-rim-stage-1.jpg")
            result['image'] = cv2.imread(image_path)
            self.current_state = "tire-rim-pairing-stage-1"

        elif self.current_state == "tire-rim-pairing-stage-1":
            tires = get_objects_by_categories(objects, {"thick_tire", "thin_tire"})
            rims = get_objects_by_categories(objects, {"thick_rim", "thin_rim"})
            if (len(tires) >= 4) and (len(rims) >= 4):
                self.current_state = "tire-rim-pairing-stage-2"
                result['speech'] = "Good job! Please find the two biggest tires, two biggest rims, and show me this configuration."
                image_path = os.path.join(images_store,"tire-rim-legend.jpg")
                result['image'] = cv2.imread(image_path)

        elif self.current_state == "tire-rim-pairing-stage-2":
            tires = get_objects_by_categories(objects, {"thick_tire", "thin_tire"})
            rims = get_objects_by_categories(objects, {"thick_rim", "thin_rim"})
            if len(tires) == 2 and len(rims) == 2:
                # image_path = os.path.join(images_store,"tire-rim-legend.png")
                # result['legend'] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                #check for the size of the bounding box of both rim and tire to see if the height matches within a certain range
                self.current_state = "wheel-stage"

        elif self.current_state == "wheel-stage":
            result['speech'] = "Please grab one each of the big and small wheels."
            image_path = os.path.join(images_store, "wheel-stage-1.jpg")
            result['image'] = cv2.imread(image_path)
            self.current_state = "wheel-stage-1"

        elif self.current_state == "wheel-stage-1":
            wheels = get_objects_by_categories(objects, {"thick_wheel_top", "thin_wheel_top"})
            if len(wheels) == 2:
                result["speech"] = "Excellent! Please line them up with the legend."
                image_path = os.path.join(images_store, "tire-legend.png")
                result['legend'] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                self.current_state = "wheel-stage-2"

        elif self.current_state == "wheel-stage-2":
            wheels = get_objects_by_categories(objects, {"thick_wheel_side", "thin_wheel_side", "thick_tire", "thin_tire"})
            if len(wheels) == 2:
                left, right = separate_left_right(objects)
                print("left: %s    right: %s    diff: %s" % (bbox_height(left["dimensions"]), bbox_height(right["dimensions"]), abs(bbox_height(left["dimensions"]) - bbox_height(right["dimensions"]))))

                self.left_frames.add(left)
                self.right_frames.add(right)

                if self.left_frames.is_center_stable(stable_threshold) and self.right_frames.is_center_stable(stable_threshold):
                    compare = wheel_compare(self.left_frames.averaged_bbox(), self.right_frames.averaged_bbox(), wheel_compare_threshold)

                    if compare == "same":
                        result["speech"] = "Those wheels are the same size. Please get two different-sized wheels."
                        self.left_frames.clear()
                        self.right_frames.clear()

                    else:
                        left_speech, right_speech = ("bigger", "smaller") if compare == "first" else ("smaller", "bigger")
                        result["speech"] = "Great job! The one on the left is the %s wheel and the one on the right is the %s wheel" % (left_speech, right_speech)
                        self.current_state = "nothing"
            else:
                self.left_frames.staged_clear()
                self.right_frames.staged_clear()

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
