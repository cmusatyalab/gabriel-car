import math
import time
from collections import defaultdict, deque
import numpy as np
import cv2
import os
from requests import get

import config

OBJECTS = config.LABELS
STATES = ["start", "wheel-stage", "wheel-compare", "tire-rim-pairing"]
images_store = os.path.abspath("images_feedback")

stable_threshold = 50
wheel_compare_threshold = 15

video_url = "http://" + ip + ":9095/"

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
    
    def averaged_class(self):
        all_class = []
        for i in range(len(self.deque)):
            if self.deque[i]["class_name"] not in all_class:
                all_class.append(self.deque[i]["class_name"])
        return max(set(all_class), key = all_class.count) 

# class Tire_rim_config:
#     def __init__(self):
#         #state: [left_rim(thick),right_rim(thin),left_tire(thick),right_tire(thin)]
#         self.state = [0,0,0,0]
    
#     def change_state(self,part):
#         if part == "left_rim":
#             self.state[0] = 1
#         if part == "right_rim":
#             self.state[1] = 1
#         if part == "left_tire":
#             self.state[2] = 1
#         if part == "right_tire":
#             self.state[3] = 1

#     def check_state(self):
#         indices = [i for i, x in enumerate(self.state) if x == 0]

        
        


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
        # if self.current_state == "start":
        #     result['speech'] = "Please show me all four rims and four tires."
        #     image_path = os.path.join(images_store, "tire-rim-stage-1.jpg")
        #     result['image'] = cv2.imread(image_path)
        #     self.current_state = "tire-rim-pairing-stage-1"

        # elif self.current_state == "tire-rim-pairing-stage-1":
        #     tires = get_objects_by_categories(objects, {"thick_tire", "thin_tire"})
        #     rims = get_objects_by_categories(objects, {"thick_rim", "thin_rim"})
        #     if (len(tires) >= 4) and (len(rims) >= 4):
        #         self.current_state = "tire-rim-pairing-stage-2"
        #         result['speech'] = "Good job! Please find the two biggest tires, two biggest rims, and show me this configuration."
        #         image_path = os.path.join(images_store,"tire-rim-legend.jpg")
        #         result['image'] = cv2.imread(image_path)

        if self.current_state == "start":
            result['speech'] = "Good job! Please find the two biggest tires, two biggest rims, and show me this configuration."
            image_path = os.path.join(images_store,"tire-rim-legend.jpg")
            result['image'] = cv2.imread(image_path)
            self.current_state = "tire-rim-pairing-stage-2"
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
                    #check for rim(on the top) and tire(on the bottom) orientation
                    if self.left_frames.averaged_bbox()[1] > self.left_frames_2.averaged_bbox()[1] and self.right_frames.averaged_bbox()[1] > self.right_frames_2.averaged_bbox()[1]:
                        #third implementation with config 2 and more confident tpod container
                        if self.left_frames.averaged_class() == "thick_wheel_side" and self.right_frames.averaged_class() == "thin_wheel_side" and self.left_frames_2.averaged_class() == "thick_rim_side" and self.right_frames_2.averaged_class() == "thin_rim_side":
                            result["speech"] = "Great Job! Now, assemble this set of tires and rims and then assemble the remaining tires and rims."
                            self.current_state = "tire-rim-pairing-stage-3"
                        elif self.left_frames.averaged_class() != "thick_wheel_side":
                            result["speech"] = "Please switch out the left tire with a bigger tire."
                        elif self.right_frames.averaged_class() != "thin_wheel_side":
                            result["speech"] = "Please switch out the right tire with a smaller tire."
                        elif self.left_frames_2.averaged_class() != "thick_rim_side":
                            result["speech"] = "Please switch out the left rim with a bigger rim."
                        elif self.right_frames_2.averaged_class() != "thin_rim_side":
                            result["speech"] = "Please switch out the right rim with a smaller rim."
                    elif self.left_frames.averaged_bbox()[1] > self.left_frames_2.averaged_bbox()[1] and self.right_frames.averaged_bbox()[1] < self.right_frames_2.averaged_bbox()[1]:
                        result["speech"] = "The orientation of tire and rim on the right is wrong. Please switch their positions"
                    elif self.left_frames.averaged_bbox()[1] < self.left_frames_2.averaged_bbox()[1] and self.right_frames.averaged_bbox()[1] > self.right_frames_2.averaged_bbox()[1]:
                        result["speech"] = "The orientation of tire and rim on the left is wrong. Please switch their positions"
                    else:
                        result["speech"] = "The orientation of tire and rim on the left and the right is wrong. Please switch the positions of the tire and rim on the left and then switch the positions of the tire and rim on the right."
                    time.sleep(3)
            else:
                self.left_frames.staged_clear()
                self.right_frames.staged_clear()
                self.left_frames_2.staged_clear()
                self.right_frames_2.staged_clear()
        elif self.current_state == "tire-rim-pairing-stage-3":
            result['video'] = video_url + "tire-rim-combine.mp4"
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


