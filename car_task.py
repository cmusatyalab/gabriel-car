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
wheel_compare_threshold = 15
tire_rim_thresold = 18

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
        self.left_frames_2 = FrameRecorder(10)
        self.right_frames_2 = FrameRecorder(10)

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
            self.current_state = "tire-rim-pairing-stage-2" #forced stage
            tires = get_objects_by_categories(objects, {"thick_tire", "thin_tire"})
            rims = get_objects_by_categories(objects, {"thick_rim", "thin_rim"})
            if (len(tires) >= 4) and (len(rims) >= 4):
                self.current_state = "tire-rim-pairing-stage-2"
                result['speech'] = "Good job! Please find the two biggest tires, two biggest rims, and show me this configuration."
                image_path = os.path.join(images_store,"tire-rim-legend.jpg")
                result['image'] = cv2.imread(image_path)

        elif self.current_state == "tire-rim-pairing-stage-2":
            tires = get_objects_by_categories(objects, {"thick_wheel_side", "thin_wheel_side"})
            rims = get_objects_by_categories(objects, {"thick_rim_side", "thin_rim_side"})
 
            if len(tires) >= 2 and len(rims) >= 2:
                tires = tires[0:2]
                rims = rims[0:2]
                left_tire, right_tire = separate_left_right(tires)
                left_rim, right_rim = separate_left_right(rims)

                self.left_frames.add(left_tire)
                self.right_frames.add(right_tire)
                self.left_frames_2.add(left_rim)
                self.right_frames_2.add(right_rim)

                if self.left_frames.is_center_stable(stable_threshold) and self.right_frames.is_center_stable(stable_threshold) and self.left_frames_2.is_center_stable(stable_threshold) and self.right_frames_2.is_center_stable(stable_threshold):
                    compare_tire = wheel_compare(self.left_frames.averaged_bbox(), self.right_frames.averaged_bbox(), wheel_compare_threshold)
                    compare_rim = wheel_compare(self.left_frames_2.averaged_bbox(), self.right_frames_2.averaged_bbox(), wheel_compare_threshold)  

                    if compare_tire == 'same' and compare_rim == 'same':
                        compare_tire_rim = wheel_compare(self.left_frames.averaged_bbox(), self.left_frames_2.averaged_bbox(), tire_rim_thresold)
                        if compare_tire_rim == 'same':
                            self.left_frames.clear()
                            self.right_frames.clear()
                            self.left_frames_2.clear()
                            self.right_frames_2.clear()
                            result["speech"] = "Great Job! Now, assemble this set of tires and rims and then assemble the remaining tires and rims"
                            time.sleep(3)
                            # self.current_state = "tire-rim-pairing-stage-3"
                        else:
                            part = "rims" if compare_tire_rim == "first" else "tires"  
                            result["speech"] = "The pair of %s is the wrong size, please get the other pair." % part
                            time.sleep(3)
                    elif compare_tire != 'same' and compare_rim != 'same':
                        side1 = "right" if compare_tire == "first" else "left"
                        side2 = "right" if compare_rim == "first" else "left"
                        result["speech"] = "The one tire on the %s and rim on the %s are smaller parts. Please find the bigger tire and rim and replace those with the smaller parts" % (side1,side2)
                        time.sleep(3)
                    elif compare_tire == 'same':
                        side = "right" if compare_rim == "first" else "left"
                        result["speech"] = "The one rim on the %s is a small rim. Please find a bigger rim and replace it with that." % side
                        time.sleep(3)
                    else:
                        side = "right" if compare_tire == "first" else "left"
                        result["speech"] = "The one on the %s is a small tire. Please find a bigger tire and replace it with that." % side
                        time.sleep(3)
            else:
                self.left_frames.staged_clear()
                self.right_frames.staged_clear()
                self.left_frames_2.staged_clear()
                self.right_frames_2.staged_clear()
        elif self.current_state == "tire-rim-pairing-stage-3":
            # result['video'] = video_url + "tire-rim-combine.mp4"
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

def separate_topleft_topright_bottomleft_bottomright(objects):
    obj1 = objects[0]
    obj2 = objects[1]
    obj3 = objects[2]
    obj4 = objects[3]

    if obj1["dimensions"][0] < obj2["dimensions"][0] and abs(obj1["dimensions"][0] - obj2["dimensions"][0]) > 0.1:
        left = [obj1]
        right = [obj2]
    else:
        left = [obj2]
        right = [obj1]

    return left, right

def bbox_center(dims):
    return (dims[0] + (dims[2] - dims[0])) / 2 , (dims[1] + (dims[3] - dims[1])) / 2

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
    print(diff)
    if diff < threshold:
        return "same"

    return "first" if height1 > height2 else "second"


