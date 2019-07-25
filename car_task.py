import math
import time
from collections import defaultdict, deque
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
wheel_compare_threshold = 15
dark_pixel_threshold = 0.3
pink_gear_side_threshold = 0.5

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

        self.frame_recs = defaultdict(lambda: FrameRecorder(10))
        self.last_id = None
        self.wait_count = 0
        self.history = defaultdict(lambda: False)
        self.delay_flag = False

        self.detector = object_detection.Detector()
        self.frame_count = 0
        self.image = None 

    def get_image(self, image_frame):
        self.image = image_frame

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
                self.detector.cleanup()

        if self.delay_flag is True:
            time.sleep(7)
            self.delay_flag = False

        result = defaultdict(lambda: None)
        result['status'] = "success"
        self.frame_count += 1

        inter = defaultdict(lambda: None)

        # the start, branch into desired instruction
        if self.current_state == "start":
            self.current_state = "layout_wheels_rims_1"
        elif self.current_state == "layout_wheels_rims_1":
            inter = self.layout_wheels_rims(img, 1)
            if inter["next"] is True:
                self.current_state = "combine_wheel_rim_1"
        elif self.current_state == "combine_wheel_rim_1":
            inter = self.combine_wheel_rim(1)
            if inter["next"] is True:
                self.current_state = "layout_wheels_rims_2"
        elif self.current_state == "layout_wheels_rims_2":
            inter = self.layout_wheels_rims(img, 2)
            if inter["next"] is True:
                self.current_state = "combine_wheel_rim_2"
        elif self.current_state == "combine_wheel_rim_2":
            inter = self.combine_wheel_rim(2)
            if inter["next"] is True:
                self.current_state = "acquire_axle_1"
        elif self.current_state == "acquire_axle_1":
            inter = self.acquire_axle(1)
            if inter["next"] is True:
                self.current_state = "axle_into_wheel_1"
        elif self.current_state == "axle_into_wheel_1":
            inter = self.axle_into_wheel(img, 1)
            if inter["next"] is True:
                self.current_state = "acquire_frame_1"
        elif self.current_state == "acquire_frame_1":
            inter = self.acquire_frame(img, 1)
            if inter["next"] is True:
                self.current_state = "insert_green_washer_1"
        elif self.current_state == "insert_green_washer_1":
            inter = self.insert_green_washer(img, 1)
            if inter["next"] is True:
                self.current_state = "insert_gold_washer"
        elif self.current_state == "insert_gold_washer":
            inter = self.insert_gold_washer(img, 1)
            if inter["next"] is True:
                self.current_state = "insert_pink_gear_front"
        elif self.current_state == "insert_pink_gear_front":
            inter = self.insert_pink_gear_front(img)
            if inter["next"] is True:
                self.current_state = "insert_axle_1"
        elif self.current_state == "insert_axle_1":
            inter = self.insert_axle(img, 1)
            if inter["next"] is True:
                self.current_state = "insert_green_washer_2"
        elif self.current_state == "insert_green_washer_2":
            inter = self.insert_green_washer(img, 2)
            if inter["next"] is True:
                self.current_state = "insert_gold_washer_2"
        elif self.current_state == "insert_gold_washer_2":
            inter = self.insert_gold_washer(img, 2)
            if inter["next"] is True:
                self.current_state = "press_wheel_1"
        elif self.current_state == "press_wheel_1":
            inter = self.press_wheel(img, 1)
            if inter["next"] is True:
                self.current_state = "acquire_axle_2"
        elif self.current_state == "acquire_axle_2":
            inter = self.acquire_axle(2)
            if inter["next"] is True:
                self.current_state = "axle_into_wheel_2"
        elif self.current_state == "axle_into_wheel_2":
            inter = self.axle_into_wheel(img, 2)
            if inter["next"] is True:
                self.current_state = "acquire_frame_2"
        elif self.current_state == "acquire_frame_2":
            inter = self.acquire_frame(img, 2)
            if inter["next"] is True:
                self.current_state = "insert_green_washer_3"
        elif self.current_state == "insert_green_washer_3":
            inter = self.insert_green_washer(img, 3)
            if inter["next"] is True:
                self.current_state = "insert_gold_washer_3"
        elif self.current_state == "insert_gold_washer_3":
            inter = self.insert_gold_washer(img, 3)
            if inter["next"] is True:
                self.current_state = "insert_pink_gear_back"
        elif self.current_state == "insert_pink_gear_back":
            inter = self.insert_pink_gear_back(img)
            if inter["next"] is True:
                self.current_state = "insert_brown_gear"
        elif self.current_state == "insert_brown_gear":
            inter = self.insert_brown_gear_back(img)
            if inter["next"] is True:
                self.current_state = "insert_axle_2"
        elif self.current_state == "insert_axle_2":
            inter = self.insert_axle(img, 2)
            if inter["next"] is True:
                self.current_state = "insert_green_washer_4"
        elif self.current_state == "insert_green_washer_4":
            inter = self.insert_green_washer(img, 4)
            if inter["next"] is True:
                self.current_state = "insert_gold_washer_4"
        elif self.current_state == "insert_gold_washer_4":
            inter = self.insert_gold_washer(img, 4)
            if inter["next"] is True:
                self.current_state = "press_wheel_2"
        elif self.current_state == "press_wheel_2":
            inter = self.press_wheel(img, 2)
            if inter["next"] is True:
                self.current_state = "add_gear_axle"
        elif self.current_state == "add_gear_axle":
            inter = self.add_gear_axle(img)
            if inter["next"] is True:
                self.current_state = "final_check"
        elif self.current_state == "final_check":
            inter = self.final_check(img)
            if inter["next"] is True:
                self.current_state = "complete"
        elif self.current_state == "complete":
            inter = self.complete()
            if inter["next"] is True:
                self.current_state = "nothing"
        elif self.current_state == "nothing":
            self.history = defaultdict(lambda: False)
            time.sleep(10)
            self.current_state = "start"

        for field in inter.keys():
            if field != "next":
                result[field] = inter[field]

        exclude = {"frame_marker_left", "frame_marker_right", "frame_horn"}
        viz_objects = [obj for obj in self.detector.all_detected_objects() if obj["class_name"] not in exclude]

        return viz_objects, result

    def layout_wheels_rims(self, img, count):
        name = "layout_wheels_rims_%s" % count
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out['image'] = read_image('tire-rim-legend.jpg')
            speech = {
                1: 'Please find two different sized rims,two different sized tires, and arrange them like this.',
                2: 'Find the other set of two different sized rims, two different sized tires, and show me this configuration.'
            }
            out['speech'] = speech[count]
            return out
        
        tires = self.get_objects_by_categories(img, {"thick_wheel_side", "thin_wheel_side"})
        rims = self.get_objects_by_categories(img, {"thick_rim_side", "thin_rim_side"})

        if len(tires) == 2 and len(rims) == 2:
            left_tire, right_tire = separate_two(tires)
            left_rim, right_rim = separate_two(rims)

            if self.frame_recs[0].add_and_check_stable(left_tire) and self.frame_recs[1].add_and_check_stable(right_tire) and self.frame_recs[2].add_and_check_stable(left_rim) and self.frame_recs[3].add_and_check_stable(right_rim):
                left_tire = self.frame_recs[0]
                right_tire = self.frame_recs[1]
                left_rim = self.frame_recs[2]
                right_rim = self.frame_recs[3]
                if left_tire.averaged_bbox()[1] > left_rim.averaged_bbox()[1] and right_tire.averaged_bbox()[1] > right_rim.averaged_bbox()[1]:
                    rim_diff = compare(left_rim.averaged_bbox(),right_rim.averaged_bbox(),wheel_compare_threshold)

                    if left_tire.averaged_class() == "thick_wheel_side" and right_tire.averaged_class() == "thin_wheel_side" and\
                    left_rim.averaged_class() == "thick_rim_side" and right_rim.averaged_class() == "thin_rim_side":
                        out['next'] = True
                    elif (left_tire.averaged_class() == "thin_wheel_side" and right_tire.averaged_class() == "thick_wheel_side"):
                        out["speech"] = "Please switch the positions of the tires."
                    elif left_tire.averaged_class() != "thick_wheel_side":
                        out["speech"] = "Please switch out the left tire with a bigger tire."
                    elif right_tire.averaged_class() != "thin_wheel_side":
                        out["speech"] = "Please switch out the right tire with a smaller tire."
                    elif rim_diff == "second":
                        out["speech"] = "Please switch the positions of the rims."
                    elif rim_diff == "same":
                        if left_rim.averaged_class() != "thick_rim_side":
                            out["speech"] = "Please switch out the left rim with a bigger rim."
                        elif right_rim.averaged_class() != "thin_rim_side":
                            out["speech"] = "Please switch out the right rim with a smaller rim."

                elif left_tire.averaged_bbox()[1] > left_rim.averaged_bbox()[1] and right_tire.averaged_bbox()[1] < right_rim.averaged_bbox()[1]:
                    out['speech'] = "The orientation of tire and rim on the right is wrong. Please switch their positions"
                elif left_tire.averaged_bbox()[1] < left_rim.averaged_bbox()[1] and right_tire.averaged_bbox()[1] > right_rim.averaged_bbox()[1]:
                    out["speech"] = "The orientation of tire and rim on the left is wrong. Please switch their positions"
                else:
                    out["speech"] = "The orientation of tire and rim on the left and the right is wrong. Please switch the positions of the tire and rim on the left and then switch the positions of the tire and rim on the right."
                self.clear_states()
        else:
            self.all_staged_clear()

        return out

    def combine_wheel_rim(self, count):
        name = "combine_wheel_rim_%s" % count
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.history[name] = True
            out["speech"] = "Well done. Now assemble the tires and rims as shown in the video."
            out["video"] = video_url + "tire_rim_combine.mp4"
        else:
            out["next"] = True
            time.sleep(10)
        return out
    
    def acquire_axle(self, count):
        name = "acquire_axle_%s" % count
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.history[name] = True
            out["speech"] = "Grab the wheel axle. Note that it has no yellow gears at the end."
            out["image"] = read_image("wheel_axle.jpg")
        else:
            out["next"] = True
            time.sleep(5)
        return out

    def axle_into_wheel(self, img, count):
        name = "axle_into_wheel_%s" % count
        out = defaultdict(lambda: None)
        good_str = "thin" if count == 1 else "thick"
        bad_str = "thick" if count == 1 else "thin"
        
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["image"] = read_image("wheel_in_axle_%s.jpg" % good_str)
            out["speech"] = "Then insert the axle into one of the %s wheels. Then hold it up like this." % good_str
            return out

        good = self.get_objects_by_categories(img, {"wheel_in_axle_%s" % good_str})
        bad = self.get_objects_by_categories(img, {"wheel_in_axle_%s" % bad_str})

        if len(good) != 1 and len(bad) != 1:
            self.all_staged_clear()
            return out

        if len(bad) == 1:
            bad_check = self.frame_recs[0].add_and_check_stable(bad[0])
            if bad_check is True:
                out["speech"] = "You have the %s wheel. Please use the %s wheel instead" % (bad_str, good_str)
                self.delay_flag = True
                self.clear_states()
        else:
            self.frame_recs[0].staged_clear()

        if len(good) == 1:
            good_check = self.frame_recs[1].add_and_check_stable(good[0])
            if good_check is True:
                out["next"] = True
        else:
            self.frame_recs[1].staged_clear()

        return out

    def acquire_frame(self, img, count):
        name = "acquire_frame_%s" % count
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Put the axle down and grab the black frame. Show it to me like this."
            out['video'] = video_url + name + ".mp4"
            return out

        frame_marker = self.get_objects_by_categories(img, {"frame_marker_right", "frame_marker_left"})
        
        marker_check = False
        if len(frame_marker) == 1:
            if self.frame_recs[0].add_and_check_stable(frame_marker[0]):
                marker_check = True

        if marker_check is True:
            out["next"] = True

        return out

    def insert_green_washer(self, img, count):
        name = "green_washer_%s" % count
        side_str = "left" if count == 1 or count == 2 else "right"
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Insert the green washer into the %s hole." % side_str
            out["video"] = video_url + name + ".mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green"})

        if 0 < len(holes) < 3:
            if len(holes) == 2:
                left, right = separate_two(holes)
                hol = left if side_str == "left" else right
            else:
                hol = holes[0]

            if hol["class_name"] == "hole_green":
                if self.frame_recs[0].add_and_check_stable(hol):
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_gold_washer(self, img, count):
        name = "gold_washer_%s" % count
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Great, now insert the gold washer into the green washer."
            out["video"] = video_url + name + ".mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green", "hole_gold"})

        if 0 < len(holes) < 3:
            if len(holes) == 2:
                left, right = separate_two(holes)
                hol = left if count <= 2 else right
            else:
                hol = holes[0]

            if hol["class_name"] == "hole_gold":
                if self.frame_recs[0].add_and_check_stable(hol):
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_pink_gear_front(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_pink_gear_front"] is False:
            self.clear_states()
            self.history["insert_pink_gear_front"] = True
            out['speech'] = "Lay the black frame down. Now place a pink gear as shown."
            out['video'] = video_url + "pink_gear_1.mp4"
            return out

        bad_pink = self.get_objects_by_categories(img, {"front_gear_bad"})
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

    def insert_axle(self, img, count):
        name = "axle_into_frame_%s" % count
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Great, now insert the axle through the washers and the pink gear. Then lay the black frame down."
            out["video"] = video_url + name + ".mp4"
            return out

        axles = self.get_objects_by_categories(img, {"axle_in_frame_good"})

        if 0 < len(axles) < 3:

            if count == 1 and len(axles) == 1:
                ax = axles[0]
            elif count == 2 and len(axles) == 2:
                _, ax = separate_two(axles, True)
            else:
                self.all_staged_clear()
                return out
            axle_check = self.frame_recs[2].add_and_check_stable(ax)
            if axle_check:
                out["next"] = True
        else:
            self.all_staged_clear()

        return out

    def press_wheel(self, img, count):
        name = "press_wheel_%s" % count
        out = defaultdict(lambda: None)
        good_str = "thin" if count == 1 else "thick"

        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Finally, press the other %s wheel into the axle. It should be the same size as the first wheel." % good_str
            out["video"] = video_url + name + ".mp4"
            return out

        wheels = self.get_objects_by_categories(img, {"%s_wheel_side" % good_str})

        if len(wheels) == 2 or len(wheels) == 4:
            if len(wheels) == 2:
                check_wheels = wheels
            else:
                four_wheels = separate_four_rect(wheels)
                indices = (0, 2) if count == 1 else (1, 3)
                check_wheels = [four_wheels[indices[0]], four_wheels[indices[1]]]
            if self.frame_recs[0].add_and_check_stable(check_wheels[0]) and \
                self.frame_recs[1].add_and_check_stable(check_wheels[1]):
                if self.frame_recs[0].averaged_class() != self.frame_recs[1].averaged_class():
                    correct_str = "thin" if good_str == "thick" else "thick"
                    out["speech"] = "You pressed the %s wheel. Please redo and press the %s wheel." % (correct_str,good_str)
                else:
                    out["next"] = True
        else:
            self.frame_recs[0].staged_clear()
            self.frame_recs[1].staged_clear()

        return out

    def insert_brown_gear(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_brown_gear"] is False:
            self.clear_states()
            self.history["insert_brown_gear"] = True
            out["speech"] = "Place the brown gear as shown. Orient it such that the part that sticks out is facing in."
            out["video"] = video_url + "brown_gear.mp4"
            return out

        bad_brown = self.get_objects_by_categories(img, {"brown_bad"})
        if len(bad_brown) >= 1:
            out["speech"] = "Make sure the gear is oriented correctly. The part that sticks out should be facing the inside of the frame."
            self.frame_recs[0].clear()
            self.delay_flag = True
            return out

        good_brown = self.get_objects_by_categories(img, {"brown_good"})
        if len(good_brown) == 1:
            if self.frame_recs[0].add_and_check_stable(good_brown[0]) is True:
                out["next"] = True
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_pink_gear_back(self, img):
        out = defaultdict(lambda: None)
        if self.history["back_pink_gear_1"] is False:
            self.clear_states()
            self.history["back_pink_gear_1"] = True
            out["speech"] = "Please find the pink gear and place in the slot shown in the picture. Make sure the teeths are points away from the center of the black frame."
            out["image"] = read_image("back_pink_gear.jpg")
            return out

        gear = self.get_objects_by_categories(img,{"back_pink","pink_back"})
        
        if len(gear) == 1:
            if self.frame_recs[0].add_and_check_stable(gear[0]):
                img = img[int(gear[0]['dimensions'][1]):int(gear[0]['dimensions'][3]),int(gear[0]['dimensions'][0]):int(gear[0]['dimensions'][2])]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                #resize
                scale_percent = 400
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


                #cut black parts from the up
                throw_out_cols_cap = 0 
                for y in range(img.shape[0]):
                    white_pixels = 0
                    for x in range(img.shape[1]):
                        if not check_dark_pixel(img[y][x],dark_pixel_threshold):
                            white_pixels += 1
                    if float(white_pixels) / float(img.shape[1]) > pink_gear_side_threshold:
                        break
                    else:
                        throw_out_cols_cap = y
                img = img[throw_out_cols_cap:,0:img.shape[1]]

                #cut black parts from the down
                for y in reversed(range(img.shape[0])):
                    white_pixels = 0
                    for x in reversed(range(img.shape[1])):
                        if not check_dark_pixel(img[y][x],dark_pixel_threshold):
                            white_pixels += 1
                    if float(white_pixels) / float(img.shape[1]) > pink_gear_side_threshold:
                        break
                    else:
                        throw_out_cols_cap = y
                img = img[0:throw_out_cols_cap,0:img.shape[1]]

                #count dark pixels for left and right side of the screen
                height = img.shape[0]
                midpoint = height / 2

                up_dark_pixels = 0
                down_dark_pixels = 0
                for x in range(img.shape[1]):
                    for y in range(img.shape[0]):
                        if y <= midpoint:
                            if check_dark_pixel(img[y][x],dark_pixel_threshold):
                                up_dark_pixels += 1
                        else:
                            if check_dark_pixel(img[y][x],dark_pixel_threshold):
                                down_dark_pixels += 1
                if up_dark_pixels > down_dark_pixels:
                    out["next"] = True
                    out["speech"] = "Great! you're done"
                    out["next"] = True
                else:
                    out["speech"] = "Please turn the pink gear around so that the teeth are pointed away from the center of the black frame."
                    self.delay_flag = True
                self.frame_recs.clear()

        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_brown_gear_back(self, img):
        out = defaultdict(lambda: None)
        if self.history["insert_brown_gear_back"] is False:
            self.clear_states()
            self.history["insert_brown_gear_back"] = True
            out["speech"] = "Find the brown gear and place it in the slot shown in the picture. Make sure the nudge on the brown gear is points to the center of the black frame."
            out["image"] = read_image("brown_gear.jpg")
            return out
        
        brown_gear = self.get_objects_by_categories(img, {"brown_good","brown_bad"})

        if len(brown_gear) == 1:
            if self.frame_recs[0].add_and_check_stable(brown_gear[0]):
                if self.frame_recs[0].averaged_class() != "brown_good":
                    out["speech"] = "Please make sure the nudge on the brown gear is points to the center of the black frame."
                else:
                    out["next"] = True
        else:
            self.frame_recs[0].staged_clear()
        return out

    def add_gear_axle(self, img):
        out = defaultdict(lambda: None)
        if self.history["add_gear_axle"] is False:
            self.clear_states()
            self.history["add_gear_axle"] = True
            out["speech"] = "Finally, find the gear axle. Use it to connect the two gear systems together."
            out["video"] = video_url + "gear_axle.mp4"
            return out

        gear_on_axle = self.get_objects_by_categories(img, {"gear_on_axle"})
        front_pink_gear = self.get_objects_by_categories(img, {"front_gear_good"})
        back_pink_gear = self.get_objects_by_categories(img, {"back_pink"})
        back_brown_gear = self.get_objects_by_categories(img, {"brown_good"})

        front_check = False
        back_check = False

        if 0 < len(gear_on_axle) < 3:
            if len(gear_on_axle) == 2:
                left, right = separate_two(gear_on_axle, True)
            else:
                left = gear_on_axle[0]
            left_check = self.frame_recs[0].add_and_check_stable(left)
            # right_check = self.frame_recs[1].add_and_check_stable(right)

            if len(front_pink_gear) == 1:
                front_gear_check = self.frame_recs[2].add_and_check_stable(left)
                if left_check is True and front_gear_check is True and \
                    check_gear_axle_front(self.frame_recs[0].averaged_bbox(), self.frame_recs[2].averaged_bbox()):
                    front_check = True
            else:
                self.frame_recs[2].staged_clear()

            if front_check:
                out["next"] = True

            # if len(back_pink_gear) == 1 and len(back_brown_gear) == 1:
            #     back_pink_check = self.frame_recs[3].add_and_check_stable(back_pink_gear[0])
            #     back_brown_check = self.frame_recs[4].add_and_check_stable(back_brown_gear[0])
            #     if right_check and back_pink_check is True and back_brown_check is True and \
            #         check_gear_axle_back(self.frame_recs[1].averaged_bbox(), self.frame_recs[3].averaged_bbox(),
            #                              self.frame_recs[4].averaged_bbox()):
            #         back_check = True
            # else:
            #     self.frame_recs[3].staged_clear()
            #     self.frame_recs[4].staged_clear()

            # if front_check is True and back_check is True:
            #     out["next"] = True
        else:
            self.all_staged_clear()

        return out

    def final_check(self, img):
        out = defaultdict(lambda: None)
        if self.history["final_check_1"] is False:
            self.clear_states()
            self.history["final_check_1"] = True
            out["speech"] = "Let me do a final check on everything. Please show me what you have, like this."
            out["image"] = read_image("final_check.jpg")
            return out
        elif self.history["final_check_2"] is False:
            wheels = self.get_objects_by_categories(img, {"thin_wheel_side", "thick_wheel_side"})

            if len(wheels) == 4:
                wheel_1 = self.frame_recs[0].add_and_check_stable(wheels[0])
                wheel_2 = self.frame_recs[1].add_and_check_stable(wheels[1])
                wheel_3 = self.frame_recs[2].add_and_check_stable(wheels[2])
                wheel_4 = self.frame_recs[3].add_and_check_stable(wheels[3])

                if wheel_1 and wheel_2 and wheel_3 and wheel_4:
                    thin_wheels = []
                    thick_wheels = []
                    for i in range(4):
                        if self.frame_recs[i].averaged_class() == "thin_wheel_side":
                            thin_wheels.append(self.frame_recs[i].averaged_bbox())
                        else:
                            thick_wheels.append(self.frame_recs[i].averaged_bbox())
                    
                    verify_1 = 2
                    for i in range(len(thin_wheels)):
                        verify_2 = 2
                        for j in range(len(thick_wheels)):
                            if thin_wheels[i][0] < thick_wheels[i][0]:
                                verify_2 -= 1
                        if verify_2 == 0:
                            verify_1 -= 1
                    
                    if verify_1 == 0:
                        self.history["final_check_2"] = True
                        out["speech"] = "The wheels look good! Please keep the camera still for a little longer. Now I'm checking the gears."
                        self.clear_states()
            else:
                self.all_staged_clear()

        elif self.history["final_check_3"] is False:
            gears = self.get_objects_by_categories(img, {"front_gear_good", "front_gear_bad", "back_pink", "brown_bad", "brown_good", "pink_back"})
            
            if len(gears) == 3:
                if self.frame_recs[0].add_and_check_stable(gears[0]) and self.frame_recs[1].add_and_check_stable(gears[1]) and self.frame_recs[2].add_and_check_stable(gears[2]):
                    brown_gear = []
                    pink_gear = []

                    for i in range(3):
                        if self.frame_recs[i].averaged_class() == "brown_good":
                            brown_gear.append(self.frame_recs[i].averaged_bbox())
                        elif self.frame_recs[i].averaged_class() == "pink_back":
                            pink_gear.append(self.frame_recs[i].averaged_bbox())
                        elif self.frame_recs[i].averaged_class() == "brown_bad":
                            out["speech"] = "Brown gear is in the wrong orientation. Please fix it."
                            return out
                        elif self.frame_recs[i].averaged_class() == "front_gear_bad":
                            out["speech"] = "Left pink gear is in the wrong orientation. Please fix it."
                            return out
                    if brown_gear[0][1] < pink_gear[0][1]:
                        out["next"] = True
                        out["speech"] = "final check done"
            else:
                self.frame_recs[0].staged_clear()
                self.frame_recs[1].staged_clear()
                self.frame_recs[2].staged_clear()


        return out

    def complete(self):
        out = defaultdict(lambda: None)
        if self.history["complete"] is False:
            self.history["complete"] = True
            out["speech"] = "Everything looks good. Great job! We've finished assembling the wheels and gear train!"
            out["next"] = True

        return out

    def clear_states(self):
        for rec in self.frame_recs.values():
            rec.clear()

    def all_staged_clear(self):
        for rec in self.frame_recs.values():
            rec.staged_clear()

def check_gear_axle_front(gear_on_axle_box, pink_box):
    intersecting = object_detection.intersecting_bbox(gear_on_axle_box, pink_box)
    to_right = bbox_center(gear_on_axle_box)[0] > bbox_center(pink_box)[0]

    return intersecting and to_right

def check_gear_axle_back(gear_on_axle_box, pink_box, brown_box):
    intersecting = object_detection.intersecting_bbox(gear_on_axle_box, pink_box) and \
                   object_detection.intersecting_bbox(gear_on_axle_box, brown_box)

    gear_axle_center = bbox_center(gear_on_axle_box)
    between = bbox_center(pink_box)[1] < gear_axle_center[1] < bbox_center(brown_box)[1]

    return intersecting and between

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

def separate_four_rect(objects):
    pairwise_x_dist = {}
    pairwise_y_dist = {}
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            i_center = bbox_center(objects[i]["dimensions"])
            j_center = bbox_center(objects[j]["dimensions"])

            x_dist = abs(i_center[0] - j_center[0])
            y_dist = abs(i_center[1] - j_center[1])

            pairwise_x_dist[x_dist] = (objects[i], objects[j])
            pairwise_y_dist[y_dist] = (objects[i], objects[j])

    sorted_y = sorted(list(pairwise_y_dist.keys()))
    num_rows = 2

    rows = [pairwise_y_dist[sorted_y[i]] for i in range(num_rows)]
    for ro in rows:
        ro.sort(key=lambda obj: bbox_center(obj["dimensions"][0]))  # sort values in rows by x coord
    rows.sort(key=lambda r: bbox_center(r[0]["dimensions"])[1])  # sort rows by first value's y coord

    top_left = rows[0][0]
    top_right = rows[0][1]
    bottom_left = rows[1][0]
    bottom_right = rows[1][1]

    return top_left, top_right, bottom_left, bottom_right


def bbox_center(dims):
    return abs(dims[2] - dims[0])/2 + dims[0], abs(dims[3] - dims[1])/2 + dims[1]

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

    left_obj, right_obj = separate_two([side_marker, horn], True)
    if side == "left":
        flipped = left_obj["class_name"] == "frame_horn"
    else:
        flipped = right_obj["class_name"] == "frame_horn"

    return side, flipped

def check_dark_pixel(pixel,threshold):
    return True if pixel <= threshold * 255 else False
