import math
import time
from collections import defaultdict, deque
import numpy as np
import cv2
import os
from requests import get

import config
import object_detection

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

    def add_and_check_stable(self, obj):
        self.add(obj)
        return self.is_center_stable()

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

    def averaged_class(self):
        all_class = []
        for i in range(len(self.deque)):
            if self.deque[i]["class_name"] not in all_class:
                all_class.append(self.deque[i]["class_name"])
        return max(set(all_class), key = all_class.count) 


class Task:
    def __init__(self, init_state=None):
        if init_state is None:
            self.current_state = "start"
        else:
            if init_state not in STATES:
                raise ValueError('Unknown init state: {}'.format(init_state))
            self.current_state = init_state

        self.frame_recs = defaultdict(lambda: FrameRecorder(5))
        self.last_id = None
        self.wait_count = 0
        self.history = defaultdict(lambda: False)
        self.delay_flag = False

        self.detector = object_detection.Detector()
        self.frame_count = 0

    def get_objects_by_categories(self, img, categories):
        return self.detector.detect_object(img, categories, self.frame_count)

    def get_instruction(self, img, header=None):
        if header is not None and "task_id" in header:
            if self.last_id is None:
                self.last_id = header["task_id"]
            elif self.last_id != header["task_id"]:
                self.last_id = header["task_id"]
                self.current_state = "start"
                self.history.clear()

        if self.delay_flag is True:
            time.sleep(3)
            self.delay_flag = False

        result = defaultdict(lambda: None)
        result['status'] = "success"
        self.frame_count += 1

        inter = defaultdict(lambda: None)

        # the start, branch into desired instruction
        if self.current_state == "start":
            self.current_state = "insert_axle_1"
        elif self.current_state == "configurate_wheels_rims":
            inter = self.configurate_wheels_rims(img)
            if inter["next"] is True:
                self.current_state = "combine_wheel_rim"
        elif self.current_state == "combine_wheel_rim":
            inter = self.combine_wheel_rim()
            if inter["next"] is True and self.history["configurate_wheels_rims_2"] is True:
                self.current_state = "nothing"
                # self.current_state = "acquire_axle"
            elif inter["next"] is True and self.history["configurate_wheels_rims_1"] is True:
                self.current_state = "configurate_wheels_rims"
        elif self.current_state == "acquire_axle":
            inter = self.acquire_axle_1(img)
            if inter["next"] is True:
                self.current_state = "axle_into_wheel_1"
        elif self.current_state == "axle_into_wheel_1":
            inter = self.axle_into_wheel_1(img)
            if inter["next"] is True:
                self.current_state = "acquire_black_frame_1"
        elif self.current_state == "acquire_black_frame_1":
            inter = self.acquire_black_frame_1(img)
            if inter["next"] is True:
                self.current_state = "insert_front_green_washer_1"
        elif self.current_state == "insert_front_green_washer_1":
            inter = self.insert_front_green_washer_1(img)
            if inter["next"] is True:
                self.current_state = "insert_front_gold_washer_1"
        elif self.current_state == "insert_front_gold_washer_1":
            inter = self.insert_front_gold_washer_1(img)
            if inter["next"] is True:
                self.current_state = "insert_pink_gear_1"
        elif self.current_state == "insert_pink_gear_1":
            inter = self.insert_pink_gear_1(img)
            if inter["next"] is True:
                self.current_state = "insert_axle_1"
        elif self.current_state == "insert_axle_1":
            inter = self.insert_axle_1(img)
            if inter["next"] is True:
                self.current_state = "insert_back_green_washer_1"
        elif self.current_state == "insert_back_green_washer_1":
            inter = self.insert_back_green_washer_1(img)
            if inter["next"] is True:
                self.current_state = "insert_back_gold_washer_1"
        elif self.current_state == "insert_back_gold_washer_1":
            inter = self.insert_back_gold_washer_1(img)
            if inter["next"] is True:
                self.current_state = "press_wheel_1"
        elif self.current_state == "press_wheel_1":
            inter = self.press_wheel_1(img)
            if inter["next"] is True:
                self.current_state = "check_front_wheel_assembly"
        elif self.current_state == "check_front_wheel_assembly":
            inter = self.check_front_wheel_assembly(img)
            if inter["next"] is True:
                self.current_state = "front_wheel_assembly_complete"
        elif self.current_state == "front_wheel_assembly_complete":
            inter = self.front_wheel_assembly_complete()
            self.current_state = "nothing"
        elif self.current_state == "nothing":
            time.sleep(3)
            self.current_state = "start"

        for field in inter.keys():
            if field != "next":
                result[field] = inter[field]

        return [], result

    def configurate_wheels_rims(self, img):
        out = defaultdict(lambda: None)
        if self.history["configurate_wheels_rims_1"] is False:
            self.clear_states()
            self.history["configurate_wheels_rims_1"] = True
            out['speech'] = 'Please find two different sized rims,two different sized tires, and show me this configuration.'
            out['image'] = read_image('tire-rim-legend')
            return out
        elif self.history["configurate_wheels_rims_2"] is False:
            self.history["configurate_wheels_rims_2"] = True
            out['speech'] = 'Please find the other set of two different sized rims,two different sized tires, and show me this configuration.'
            out['image'] = read_image('tire-rim-legend')
            return out
        
        tires = self.get_objects_by_categories(img, {"thick_wheel_side", "thin_wheel_side"})
        rims = self.get_objects_by_categories(img, {"thick_rim_side", "thin_rim_side"})

        left_tire, right_tire = separate_two(tires)
        left_rim, right_rim = separate_two(rims)

        if len(tires) >= 2 and len(rims) >= 2:
            if self.frame_recs[0].add_and_check_stable(left_tire) and self.frame_recs[1].add_and_check_stable(right_tire) and self.frame_recs[2].add_and_check_stable(left_rim) and self.frame_recs[3].add_and_check_stable(right_rim): 
                if self.frame_recs[0].averaged_bbox()[1] > self.frame_recs[2].averaged_bbox()[1] and self.frame_recs[1].averaged_bbox()[1] > self.frame_recs[3].averaged_bbox()[1]:
                    if self.frame_recs[0].averaged_class() == "thick_wheel_side" and self.frame_recs[1].averaged_class() == "thin_wheel_side" and self.frame_recs[2].averaged_class() == "thick_rim_side" and self.frame_recs[3].averaged_class() == "thin_rim_side":
                        out['next'] = True
                    elif self.frame_recs[0].averaged_class() != "thick_wheel_side":
                        out["speech"] = "Please switch out the left tire with a bigger tire."
                    elif self.frame_recs[1].averaged_class() != "thin_wheel_side":
                        out["speech"] = "Please switch out the right tire with a smaller tire."
                    elif self.frame_recs[2].averaged_class() != "thick_rim_side":
                        out["speech"] = "Please switch out the left rim with a bigger rim."
                    elif self.frame_recs[3].averaged_class() != "thin_rim_side":
                        out["speech"] = "Please switch out the right rim with a smaller rim."
                elif self.frame_recs[0].averaged_bbox()[1] > self.frame_recs[2].averaged_bbox()[1] and self.frame_recs[1].averaged_bbox()[1] < self.frame_recs[3].averaged_bbox()[1]:
                    out['speech'] = "The orientation of tire and rim on the right is wrong. Please switch their positions"
                elif self.frame_recs[0].averaged_bbox()[1] < self.frame_recs[2].averaged_bbox()[1] and self.frame_recs[1].averaged_bbox()[1] > self.frame_recs[3].averaged_bbox()[1]:
                    out["speech"] = "The orientation of tire and rim on the left is wrong. Please switch their positions"
                else:
                    out["speech"] = "The orientation of tire and rim on the left and the right is wrong. Please switch the positions of the tire and rim on the left and then switch the positions of the tire and rim on the right."
                self.clear_states()
        else:
            self.frame_recs[0].staged_clear()
            self.frame_recs[1].staged_clear()
            self.frame_recs[2].staged_clear()
            self.frame_recs[3].staged_clear()

        return out

    def combine_wheel_rim(self):
        out = defaultdict(lambda: None)
        if self.history["combine_wheel_rim"] is False:
            self.clear_states()
            self.history["combine_wheel_rim"] = True
            out["speech"] = "Well done. Now assemble the tires and rims as shown in the video"
        else:
            out["video"] = video_url + "tire-rim-combine.mp4"
            out["next"] = True
            self.delay_flag = True
        return out

    def acquire_axle_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["acquire_axle_1"] is False:
            self.clear_states()
            self.history["acquire_axle_1"] = True
            out['speech'] = "Please grab the wheel axle. Note that it has no gears at the ends."
            out['image'] = read_image("wheel_axle.jpg")
            return out

        # gears = self.get_objects_by_categories(img, {"axle_on_gear"})
        # if len(gears) > 0:
        #     if self.frame_recs[1].add_and_check_stable(gears[0]) is True:
        #         out["speech"] = "You grabbed the wrong part. Please look for a smaller axle without any gears on the ends."
        #         self.clear_states()
        #         self.delay_flag = True
        #     return out
        # else:
        #     self.frame_recs[1].staged_clear()

        axles = self.get_objects_by_categories(img, {"wheel_axle"})
        if len(axles) == 1:
            if self.frame_recs[0].add_and_check_stable(axles[0]) is True:
                out["next"] = True
        else:
            self.frame_recs[0].staged_clear()

        return out

    def axle_into_wheel_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["axle_into_wheel_1"] is False:
            self.clear_states()
            self.history["axle_into_wheel_1"] = True
            out["speech"] = "Great! Please insert the axle into one of the thinner wheels. Then hold it up like this."
            out["image"] = read_image("wheel_in_axle.jpg")
            return out


        thick = self.get_objects_by_categories(img, {"wheel_in_axle_thick"})
        thin = self.get_objects_by_categories(img, {"wheel_in_axle_thin"})

        if len(thick) != 1 and len(thin) != 1:
            self.all_staged_clear()
            return out

        if len(thick) == 1:
            thick_check = self.frame_recs[0].add_and_check_stable(thick[0])
            if thick_check is True:
                out["speech"] = "You have the thicker wheel. Please use the thinner wheel instead"
                self.delay_flag = True
                self.clear_states()
        else:
            self.frame_recs[0].staged_clear()

        if len(thin) == 1:
            thin_check = self.frame_recs[1].add_and_check_stable(thin[0])
            if thin_check is True:
                out["next"] = True
        else:
            self.frame_recs[1].staged_clear()

        return out

    def acquire_black_frame_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["acquire_black_frame_1"] is False:
            self.clear_states()
            self.history["acquire_black_frame_1"] = True
            out["speech"] = "Put the axle down and grab the black frame. Show it to me like this."
            out['video'] = video_url + "get_frame.mp4"
            return out

        frame_marker = self.get_objects_by_categories(img, {"frame_marker_right", "frame_marker_left"})
        horn = self.get_objects_by_categories(img, {"frame_horn"})

        if len(frame_marker) != 1 and len(horn) != 1:
            self.all_staged_clear()
            return out

        marker_check = False
        if len(frame_marker) == 1:
            if self.frame_recs[0].add_and_check_stable(frame_marker[0]):
                marker_check = True

        horn_check = False
        if len(horn) == 1:
            if self.frame_recs[1].add_and_check_stable(horn[0]):
                horn_check = True

        if marker_check is True and horn_check is True:
            out["next"] = True

        return out

    def insert_front_green_washer_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_front_green_washer_1"] is False:
            self.clear_states()
            self.history["insert_front_green_washer_1"] = True
            out["speech"] = "Insert the green washer into the left hole."
            out["video"] = video_url + "green_washer_1.mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green"})

        if len(holes) == 2:
            left, _ = separate_two(holes)
            if left["class_name"] == "hole_green":
                if self.frame_recs[0].add_and_check_stable(left):
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_front_gold_washer_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_front_gold_washer_1"] is False:
            self.clear_states()
            self.history["insert_front_gold_washer_1"] = True
            out["speech"] = "Great, now insert the gold washer into the green washer."
            out["video"] = video_url + "gold_washer_1.mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green", "hole_gold"})

        if len(holes) == 2:
            left, _ = separate_two(holes)
            if left["class_name"] == "hole_gold":
                if self.frame_recs[0].add_and_check_stable(left):
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_pink_gear_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_pink_gear_1"] is False:
            self.clear_states()
            self.history["insert_pink_gear_1"] = True
            out['speech'] = "Lay the black frame down. Now place a pink gear as shown."
            out['video'] = video_url + "pink_gear_front.mp4"
            return out

        bad_pink = self.get_objects_by_categories(img, {"front_gear_bad"}) # TODO: pink gear orientation, not sure how to solve
        if len(bad_pink) >= 1:
            out["speech"] = "Please flip the pink gear around."
            self.frame_recs[0].clear()
            self.delay_flag = True
            return out

        good_pink = self.get_objects_by_categories(img, {"front_gear_good"})
        if len(good_pink) == 1:
            if self.frame_recs[0].add_and_check_stable(good_pink[0]) is True:
                out["next"] = True
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_axle_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_axle_1"] is False:
            self.clear_states()
            self.history["insert_axle_1"] = True
            out["speech"] = "Great, now insert the axle through the washers and the pink gear. "
            out["video"] = video_url + "axle_into_frame_front.mp4"
            return out

        axles = self.get_objects_by_categories(img, {"wheel_axle"})

        thick = self.get_objects_by_categories(img, {"wheel_in_axle_thick"})
        thin = self.get_objects_by_categories(img, {"wheel_in_axle_thin"})

        if len(thick) == 1:
            thick_check = self.frame_recs[0].add_and_check_stable(thick[0])
            if thick_check is True:
                out["speech"] = "You have the thicker wheel. Please use the thinner wheel instead"
                self.delay_flag = True
                self.clear_states()
        else:
            self.all_staged_clear()

        if len(thin) == 1 and len(axles) == 1:
            thin_check = self.frame_recs[1].add_and_check_stable(thin[0])
            axle_check = self.frame_recs[2].add_and_check_stable(axles[0])
            if thin_check and axle_check:
                out["next"] = True
        else:
            self.all_staged_clear()

        return out

    def insert_back_green_washer_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_back_green_washer_1_flip"] is False:
            self.clear_states()
            self.history["insert_back_green_washer_1_flip"] = True
            out["speech"] = "Great, now flip over to the other side."
            out["video"] = video_url + "flip_frame_1.mp4"
            return out
        elif self.history["insert_back_green_washer_1_prompt"] is False:
            time.sleep(5)
            self.history["insert_back_green_washer_1_prompt"] = True
            out["speech"] = "Insert the green washer through the axle into the back left hole."
            out["video"] = video_url + "green_washer_2.mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green"})

        if len(holes) == 2:
            left, _ = separate_two(holes)
            if left["class_name"] == "hole_green":
                if self.frame_recs[0].add_and_check_stable(left):
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_back_gold_washer_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_back_gold_washer_1"] is False:
            self.clear_states()
            self.history["insert_back_gold_washer_1"] = True
            out["speech"] = "Great, now insert the gold washer into the green washer."
            out["video"] = video_url + "gold_washer_2.mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green", "hole_gold"})

        if len(holes) == 2:
            left, _ = separate_two(holes)
            if left["class_name"] == "hole_gold":
                if self.frame_recs[0].add_and_check_stable(left):
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.frame_recs[0].staged_clear()

        return out

    def press_wheel_1(self, img):
        out = defaultdict(lambda: None)
        if self.history["press_wheel_1"] is False:
            self.clear_states()
            self.history["press_wheel_1"] = True
            out["speech"] = "Finally, press the other thin wheel into the axle. It should be the same size as the first wheel."
            out["video"] = video_url + "press_wheel_1.mp4"
            return out

        wheels = self.get_objects_by_categories(img, {"thin_wheel_side"})

        if len(wheels) == 2:
            if self.frame_recs[0].add_and_check_stable(wheels[0]) and self.frame_recs[1].add_and_check_stable(wheels[1]):
                out["next"] = True
        else:
            self.frame_recs[0].staged_clear()

        return out

    def check_front_wheel_assembly(self, img):
        out = defaultdict(lambda: None)
        if self.history["check_front_wheel_assembly"] is False:
            self.clear_states()
            self.history["check_front_wheel_assembly"] = True
            out["speech"] = "Please show me what you have, like this."
            out["video"] = video_url + "check_front_wheel_assembly.mp4"
            return out

        wheels = self.get_objects_by_categories(img, {"thin_wheel", "thick_wheel"})

        if len(wheels) == 2:
            top, bottom = separate_two(wheels, False)
            top_check = self.frame_recs[0].add_and_check_stable(top)
            bottom_check = self.frame_recs[1].add_and_check_stable(bottom)

            if top_check is True and bottom_check is True:
                if compare(self.frame_recs[0].averaged_bbox(), self.frame_recs[1].averaged_bbox(), wheel_compare_threshold) == "same":
                    out["next"] = True
        else:
            self.frame_recs[0].staged_clear()

        return out

    def front_wheel_assembly_complete(self):
        out = defaultdict(lambda: None)
        if self.history["check_front_wheel_assembly"] is False:
            self.history["check_front_wheel_assembly"] = True
            out["speech"] = "Great job! We're finished with assembling the front wheels."

        return out

    def clear_states(self):
        for rec in self.frame_recs.values():
            rec.clear()

    def all_staged_clear(self):
        for rec in self.frame_recs.values():
            rec.staged_clear()


def check_insert_axle_1(pink_gear, axle):
    return pink_gear["dimensions"][1] < axle["dimensions"][1]

def separate_two(objects, left_right=True):
    obj1 = objects[0]
    obj2 = objects[1]

    dim = 0 if left_right is True else 1  # left-right vs top-bottom

    if obj1["dimensions"][dim] < obj2["dimensions"][dim]:
        one = obj1
        two = obj2
    else:
        one = obj2
        two = obj1

    return one, two

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

def compare(box1, box2, threshold):
    height1 = bbox_height(box1)
    height2 = bbox_height(box2)

    diff = abs(height1 - height2)
    if diff < threshold:
        return "same"

    return "first" if height1 > height2 else "second"

def read_image(name):
    image_path = os.path.join(resources, name)
    return cv2.imread(image_path)

def get_orientation(side_marker, horn):
    side = "left" if side_marker["class_name"] == "frame_marker_left" else "right"

    left_obj, right_obj = separate_left_right([side_marker, horn])
    if side == "left":
        flipped = left_obj["class_name"] == "frame_horn"
    else:
        flipped = right_obj["class_name"] == "frame_horn"

    return side, flipped

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

