import math
import time
from collections import defaultdict, deque
import cv2
import os
from requests import get

import config
import object_detection

"""
This file contains the Task object for the model car kit, which handles all processing of a frame, that is:
1. Conditions for triggering the next step and catching any errors
2. Corresponding media (image, video speech)
3. Additional features such as stable frame detection
"""

ip = get('https://api.ipify.org').text

resources = os.path.abspath("resources/images")  # for images, which are sent directly from this library
video_url = "http://" + ip + ":9095/"  # for videos, which are accessed from a separate resource server
tpod_url = "http://0.0.0.0:8000"  # object detection classifier URL

#  max Euclidean distance between consecutive frames in pixels, to be considered stable
stable_threshold = 50
#  difference in bbox height in pixels to be considered different sized-wheels
wheel_compare_threshold = 15
#  for pink gear orientation using traditional image processing, threshold (out of 1) to consider a pixel a "dark" pixel
dark_pixel_threshold = 0.3
#  number of "light" pixels detected to know when to stop cropping gear bbox
pink_gear_side_threshold = 0.5
#  number of frames needed to consider a workspace cluttered
clutter_threshold = 5
clutter_speech = "Your workspace is cluttered. Please remove any stray parts from my view."

class FrameRecorder:
    """
    FrameRecorder is used to check whether or not a detected object in a frame is "stable" that is:
    1. Same object is detected in the last n frames
    2. Object detected in each frame does not move past a threshold from one frame to another

    One FrameRecorder per object
    This object is given detected object bounding boxes, not raw frames. It is up to the user to figure out which object
    in the frame to pass in.
    """
    def __init__(self, size):
        self.deque = deque()
        self.size = size
        self.clear_count = 0

    def add(self, obj):
        """
        Add an object from a single frame to the recorder
        Resets the staged clear counter
        :param obj: to add
        """
        self.deque.append(obj)

        if len(self.deque) > self.size:
            self.deque.popleft()

        self.clear_count = 0

    def is_center_stable(self):
        """
        Returns whether or not the object being recorded is stable
        """
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
        """
        Add a new frame and return if the object is stable
        """
        self.add(obj)
        return self.is_center_stable()

    def staged_clear(self):
        """
        A special clear that needs to be called multiple times (max number of frames) to actually clear
        """
        self.clear_count += 1
        if self.clear_count > self.size:
            self.clear()

    def clear(self):
        """
        Clears the FrameRecorder's frames
        """
        self.deque = deque()

    def averaged_bbox(self):
        """
        Return the averaged bbox from all recorded frames
        """
        out = [0, 0, 0, 0]

        for i in range(len(self.deque)):
            dim = self.deque[i]["dimensions"]
            for u in range(len(dim)):
                out[u] += dim[u]

        return [v / len(self.deque) for v in out]

    def averaged_class(self):
        """
        Return the mode detected object class from all recorded frames
        """
        all_class = []
        for i in range(len(self.deque)):
            if self.deque[i]["class_name"] not in all_class:
                all_class.append(self.deque[i]["class_name"])
        return max(set(all_class), key = all_class.count) 


class Task:
    """
    Object that keeps track of user's state, returns corresponding guidance, and handles all frame processing.
    Bulk of AAA exists here.
    """

    def __init__(self, init_state=None):
        if init_state is None:
            self.current_state = "start"
        else:
            self.current_state = init_state

        self.frame_recs = defaultdict(lambda: FrameRecorder(15))  # dictionary of frame recorders for different objs
        self.session_id = None  # ID from client to know the same session is still going on
        self.history = defaultdict(lambda: False)  # keeps track of which steps were completed
        self.delay_flag = False  # set to True to delay processing (usually after user makes mistake, needs time to fix)

        self.detector = object_detection.Detector(tpod_url)  # Detector object for object detection
        self.frame_id = 0  #  unique ID for each frame, for detector's cache

        self.clutter_count = 0  #  tracks number of times workspace was detected to be cluttered, before triggering message

        #  used for waiting based on time (instead of frames)
        self.time = None
        self.time_trigger = False

    def get_objects_by_categories(self, img, categories, image_id=None):
        """
        Detects objects in a given frame/image. Need to supply objects to be detected

        :param img: to detect objects in
        :param categories: object labels to look for
        :param image_id: if a specific classifier is to be used (instead of looking up based on objects), supply its
                         Docker image ID
        """
        return self.detector.detect_object(img, categories, self.frame_id, image_id)

    def get_instruction(self, img, header=None):
        """
        Get the next instruction, given a new frame

        :param img: frame with objects to detect
        :param header: from client's request, used to track session ID
        :return: tuple of objects to visualize and a response object with image, video, and/or speech references
        """

        # reset if different client
        if header is not None and "task_id" in header:
            if self.session_id is None:
                self.session_id = header["task_id"]
            elif self.session_id != header["task_id"]:
                self.session_id = header["task_id"]
                self.current_state = "start"
                self.history.clear()
                self.detector.reset()

        # sleep if previous frame set the delay flag
        if self.delay_flag is True:
            time.sleep(4)
            self.delay_flag = False

        result = defaultdict(lambda: None)
        result['status'] = "success"
        self.frame_id += 1

        inter = defaultdict(lambda: None)

        # the start, branch into desired instruction
        if self.current_state == "start":
            self.current_state = "intro"
        elif self.current_state == "intro":
            inter = self.intro()
            if inter["next"] is True:
                self.current_state = "layout_wheel_rim_1"
        elif self.current_state == "layout_wheel_rim_1":
            inter = self.layout_wheel_rim(img, 1)
            if inter["next"] is True:
                self.current_state = "combine_wheel_rim_1"
        elif self.current_state == "combine_wheel_rim_1":
            inter = self.combine_tire_rim(img, 1)
            if inter["next"] is True:
                self.current_state = "confirm_combine_wheel_rim_1"
        elif self.current_state == "confirm_combine_wheel_rim_1":
            inter = self.confirm_combine_tire_rim(img, 1)
            if inter["next"] is True:
                self.current_state = "layout_wheel_rim_2"
        elif self.current_state == "layout_wheel_rim_2":
            inter = self.layout_wheel_rim(img, 2)
            if inter["next"] is True:
                self.current_state = "combine_wheel_rim_2"
        elif self.current_state == "combine_wheel_rim_2":
            inter = self.combine_tire_rim(img, 2)
            if inter["next"] is True:
                self.current_state = "confirm_combine_wheel_rim_2"
        elif self.current_state == "confirm_combine_wheel_rim_2":
            inter = self.confirm_combine_tire_rim(img, 2)
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
                self.current_state = "insert_gold_washer_1"
        elif self.current_state == "insert_gold_washer_1":
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

        # copy intermediate response to output
        for field in inter.keys():
            if field != "next":
                result[field] = inter[field]

        # set up objects with instructions on how to visualize
        exclude = {"frame_marker_left", "frame_marker_right", "frame_horn"}  # exclude these unused objects
        viz_objects = [obj for obj in self.detector.all_detected_objects() if obj["class_name"] not in exclude]
        for obj in viz_objects:
            if "color" not in obj.keys():
                obj["color"] = "blue" if inter["good_frame"] else "red"  # color based on if frame was used or not

        return viz_objects, result
    
    """
    Helper functions for get_instruction
    
    Common patterns:
    1. We use self.history to detect whether or not it's the first frame processed for a new step. This means we need to
    give the user guidance via speech/image/video. If it's not the first frame, then we don't give guidance.
    2. Detect a bunch of different objects, and see if all the conditions (number of detected objects, distances between
    detected objects, etc.) are met before moving on.
    3. Simultaneously detecting if any error conditions are met and returning corresponding guidance.
    4. Setting out["next"] to True as a signal to the get_instruction to advance to the next step
    5. Setting self.delay_flag to True to pause processing for a short time
    """
    def intro(self):
        """
        Beginning tutorial on how to read the interface
        """
        out = defaultdict(lambda: None)
        if self.history["intro_1"] is False:
            self.history["intro_1"] = True 
            out["speech"] = "Hi, thanks for using our Auto Assembly Assistant."
        elif self.history["intro_2"] is False:
            self.history["intro_2"] = True
            out["speech"] = "My name is Gabriel and I will be your assistant in building this car model."
        elif self.history["intro_3"] is False:
            self.history["intro_3"] = True
            out["speech"] = "Let me guide you through this interface."
        elif self.history["intro_4"] is False:
            self.history["intro_4"] = True
            out["speech"] = "During use, you'll see red or blue boxes around parts of the camera stream."
        elif self.history["intro_5"] is False:
            self.history["intro_5"] = True
            out["speech"] = "These boxes are certain parts or states I'm recognizing."
        elif self.history["intro_6"] is False:
            self.history["intro_6"] = True
            out["speech"] = "Red boxes are bad. Blue boxes are good."
        elif self.history["intro_7"] is False:
            self.history["intro_7"] = True
            out["speech"] = "Red boxes indicate dedicated objects, but that I can't determine anything off of them."
            out["image"] = read_image("red_box_ex.jpg")
        elif self.history["intro_8"] is False:
            self.history["intro_8"] = True
            out["speech"] = "Blue boxes mean I can determine things, and just need you to hold still."
            out["image"] = read_image("blue_box_ex.jpg")
        elif self.history["intro_9"] is False:
            self.history["intro_9"] = True
            out["speech"] = "That's it. Looking forward to working with you!"
            out['next'] = True
        self.delay_flag = True

        return out


    def layout_wheel_rim(self, img, count):
        """
        Layout two different sets of rims and tires

        Errors:
        1. Not having two different sets of rims and tires e.g. 2 thin rims

        Completion: two different sets of rims and tires are detected
        """
        name = "layout_wheels_rims_%s" % count
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out['image'] = read_image('tire-rim-legend.jpg')
            speech = {
                1: 'Please find two different sized green rims and two different sized black tires.',
                2: 'Find the other two sets of rims and tires.'
            }
            out['speech'] = speech[count]
            return out

        thin_rim = self.get_objects_by_categories(img, {"thin_rim_side"})
        thin_wheel = self.get_objects_by_categories(img, {"thin_wheel_side"})
        thick_rim = self.get_objects_by_categories(img, {"thick_rim_side"})
        thick_wheel = self.get_objects_by_categories(img, {"thick_wheel_side"})

        if len(thin_rim) == 1 and len(thick_rim) == 1 and len(thin_wheel) == 1 and len(thick_wheel) == 1:
            thin_rim_check = self.frame_recs[0].add_and_check_stable(thin_rim[0])
            thick_rim_check = self.frame_recs[1].add_and_check_stable(thick_rim[0])
            thin_wheel_check = self.frame_recs[2].add_and_check_stable(thin_wheel[0])
            thick_wheel_check = self.frame_recs[3].add_and_check_stable(thick_wheel[0])

            if thin_rim_check and thick_rim_check and thin_wheel_check and thick_wheel_check:
                out["good_frame"] = True
                out["next"] = True

        return out

    def combine_tire_rim(self, img, count):
        """
        Combine the layout of tires and rims by coloring the matching pairs together

        Completion: automatically after 10 seconds
        """
        name = "combine_tire_rim_%s" % count
        out = defaultdict(lambda: None)
        if self.history[name] is False:
            self.history[name] = True
            out["speech"] = "Well done. Now put the tires and rims together by color."
            out["video"] = video_url + "tire_rim_combine.mp4"
            self.time = time.time()

        thin_rim = self.get_objects_by_categories(img, {"thin_rim_side"})
        thin_wheel = self.get_objects_by_categories(img, {"thin_wheel_side"})
        thick_rim = self.get_objects_by_categories(img, {"thick_rim_side"})
        thick_wheel = self.get_objects_by_categories(img, {"thick_wheel_side"})

        # hack using time because we need to process each frame, but want to also play for 10 seconds
        if len(thin_rim) == 1 and len(thick_rim) == 1 and len(thin_wheel) == 1 and len(thick_wheel) == 1:
            if not self.time_trigger:
                self.time_trigger = True
                self.time = time.time()

        # color matching pairs
        if len(thin_rim) == 1:
            self.detector.color_detected_object({
                "thin_rim_side": "yellow",
                "thin_wheel_side": "yellow"
            })
        if len(thin_wheel) == 1:
            self.detector.color_detected_object({
                "thin_wheel_side": "yellow"
            })
        if len(thick_rim) == 1:
            self.detector.color_detected_object({
                "thick_rim_side": "orange",
            })
        if len(thick_wheel) == 1:
            self.detector.color_detected_object({
                "thick_wheel_side": "orange"
            })

        if self.time_trigger and time.time() > self.time + 10:
            out["next"] = True

        return out

    def confirm_combine_tire_rim(self, img, count):
        """
        Checks that the previous step of combining tire and rims was completed

        Errors: Combining the wrong pairs, via detecting thick rim in thin tire

        Completion: the combined wheels from a birds-eye position
        """
        name = "confirm_combine_tire_rim_%s" % count
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Then, show me the wheels like this."
            out["image"] = read_image("wheels_assembled.jpg")

        wheels = self.get_objects_by_categories(img, {"wrong_wheel", "thick_wheel_side", "thin_wheel_side"}, "a4b34fd8f0f6")
        if len(wheels) == 2:
            out["good_frame"] = True
            left_wheel, right_wheel = separate_two(wheels)
            if self.frame_recs[0].add_and_check_stable(left_wheel) and self.frame_recs[1].add_and_check_stable(right_wheel):
                if self.frame_recs[0].averaged_class() == "wrong_wheel" or \
                      self.frame_recs[1].averaged_class() == "wrong_wheel":
                    out["speech"] = "You combined the wrong tire rim pairs. Please swap the parts of each pair."
                    self.delay_flag = True
                    self.clear_states()
                else:
                    out["next"] = True
        else:
            self.all_staged_clear()
        return out

    def acquire_axle(self, count):
        """
        Get the wheel axle

        Completion: automatically after delay
        """
        name = "acquire_axle_%s" % count
        out = defaultdict(lambda: None)
        self.delay_flag = True
        if self.history[name] is False:
            self.history[name] = True
            speech = {
                1: "Moving on. Grab the wheel axle. Note that it has no yellow gears at the end.",
                2: "Moving on. Grab the other wheel axle."
            }
            out['speech'] = speech[count]
            out["image"] = read_image("wheel_axle.jpg")
        else:
            out["next"] = True
        return out

    def axle_into_wheel(self, img, count):
        """
        Insert the axle into the correct type of wheel

        Errors:
        1. Inserted axle into the wrong wheel

        Completion: the combined wheels from a birds-eye position
        """
        name = "axle_into_wheel_%s" % count
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        good_str = "thin" if count == 1 else "thick"
        bad_str = "thick" if count == 1 else "thin"
        
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["image"] = read_image("wheel_in_axle_%s.jpg" % good_str)
            out["speech"] = "Now, insert the axle into one of the %s wheels. Then hold it up like this." % good_str
            return out

        # detects a small area where axle meets wheel, small distinguishing feature
        good = self.get_objects_by_categories(img, {"wheel_in_axle_%s" % good_str})
        bad = self.get_objects_by_categories(img, {"wheel_in_axle_%s" % bad_str})

        if len(good) != 1 and len(bad) != 1:
            self.all_staged_clear()
            return out

        if len(bad) == 1:
            out["good_frame"] = True
            bad_check = self.frame_recs[0].add_and_check_stable(bad[0])
            if bad_check is True:
                good_str = "smaller" if good_str == "thin" else "bigger"
                out["speech"] = "You have the %s wheel. Please use the %s wheel instead" % (bad_str, good_str)
                self.delay_flag = True
                self.clear_states()
        else:
            self.frame_recs[0].staged_clear()

        if len(good) == 1:
            out["good_frame"] = True
            good_check = self.frame_recs[1].add_and_check_stable(good[0])
            if good_check is True:
                out["next"] = True
        else:
            self.frame_recs[1].staged_clear()

        return out

    def acquire_frame(self, img, count):
        """
        Get the black frame in a side view

        Completion: frame marker from the side view
        """

        name = "acquire_frame_%s" % count
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Moving on. Grab the black frame. Show me a side view of the axle holes like this."
            out['video'] = video_url + name + ".mp4"
            return out

        # doesn't matter which side, just that a frame marker is found
        # TODO: should detect that correct side is shown, but needs more training data
        frame_marker = self.get_objects_by_categories(img, {"frame_marker_right", "frame_marker_left"})
        
        marker_check = False
        if len(frame_marker) == 1:
            out["good_frame"] = True
            if self.frame_recs[0].add_and_check_stable(frame_marker[0]):
                marker_check = True

        if marker_check is True:
            out["next"] = True

        return out

    def insert_green_washer(self, img, count):
        """
        Insert the green washer into one of the holes

        Errors:
        1. Inserting the green washer into the wrong hole (only on same side)

        Completion: state of frame axle hole with green washer inside
        """
        name = "green_washer_%s" % count
        find = "find_green_washer_%s" % count
        side_str = "left" if count == 1 or count == 2 else "right"
        out = defaultdict(lambda: None)
        out["good_frame"] = False

        if count == 1 and self.history[find] is False:
            self.clear_states()
            self.history[find] = True
            out["speech"] = "Great, now find a green washer."
            out["image"] = read_image("green_washer.png")
            return out
        if self.history[name] is False:
            time.sleep(4)
            self.history[name] = True
            speech = {1: "Insert the green washer into the %s hole. Then, show me a side view of the holes like in the video." % side_str,
                      2: "Now, insert a green washer into the %s hole. Then, show me a side view of the holes." % side_str,
                      3: "Now, insert a green washer into the %s hole. Then, show me a side view of the holes." % side_str,
                      4: "Now, insert a green washer into the %s hole. Then, show me a side view of the holes." % side_str}
            out["speech"] = speech[count]

            out["video"] = video_url + name + ".mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green"})

        if 0 < len(holes) < 3:
            out["good_frame"] = True
            if len(holes) == 2:
                left, right = separate_two(holes)
                hol = left if side_str == "left" else right

                other_hol = right if side_str == "left" else left
                if other_hol["class_name"] == "hole_green":
                    if self.frame_recs[1].add_and_check_stable(other_hol):
                        out["speech"] = "You put the green washer in the wrong hole. Please put it in the %s hole." % side_str
                        self.delay_flag = True
                        self.clear_states()
            else:
                hol = holes[0]

            if hol["class_name"] == "hole_green":
                if self.frame_recs[0].add_and_check_stable(hol):
                    self.clutter_reset()
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.all_staged_clear()

        if len(holes) > 2:
            self.clutter_add()
        if self.clutter_check(clutter_threshold):
            out["speech"] = clutter_speech
            self.delay_flag = True

        return out

    def insert_gold_washer(self, img, count):
        """
        Insert the gold washer into one of the holes

        Completion: state of frame axle hole with gold washer inside
        """
        name = "gold_washer_%s" % count
        find = "find_gold_washer_%s" % count
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if count == 1 and self.history[find] is False:
            self.clear_states()
            self.history[find] = True
            out["speech"] = "Great, now find a gold washer." 
            out["image"] = read_image("gold_washer.png")
            return out
        if self.history[name] is False:
            time.sleep(4)
            self.history[name] = True
            out["speech"] = "Insert the gold washer into the green washer."
            out["video"] = video_url + name + ".mp4"
            return out

        holes = self.get_objects_by_categories(img, {"hole_empty", "hole_green", "hole_gold"})

        if 0 < len(holes) < 3:
            out["good_frame"] = True
            if len(holes) == 2:
                left, right = separate_two(holes)
                hol = left if count <= 2 else right
            else:
                hol = holes[0]

            if hol["class_name"] == "hole_gold":
                if self.frame_recs[0].add_and_check_stable(hol):
                    self.clutter_reset()
                    out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
        else:
            self.frame_recs[0].staged_clear()

        if len(holes) > 2:
            self.clutter_add()

        if self.clutter_check(clutter_threshold):
            out["speech"] = clutter_speech
            self.delay_flag = True
        return out

    def insert_pink_gear_front(self, img):
        """
        Place the front pink gear into the frame

        Errors:
        1. Pink gear put in facing the wrong way

        Completion: Pink gear in the frame facing the correct way, birds-eye view
        """
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history["insert_pink_gear_front"] is False:
            self.clear_states()
            self.history["insert_pink_gear_front"] = True
            out['speech'] = "Great job! Now, Lay the black frame down and give me a birds-eye view. Then, insert the pink gear, making sure its teeth are facing towards you."
            out['image'] = read_image("pink_gear_1.jpg")
            return out

        # good and bad states were trained on and can be detected
        bad_pink = self.get_objects_by_categories(img, {"front_gear_bad"})
        if len(bad_pink) >= 1:
            out["good_frame"] = True
            if self.frame_recs[1].add_and_check_stable(bad_pink[0]):
                out["speech"] = "Please make sure the teeth are facing towards you."
                self.frame_recs[0].clear()
                self.delay_flag = True
                return out
        else:
            self.frame_recs[1].staged_clear()

        good_pink = self.get_objects_by_categories(img, {"front_gear_good"})
        if len(good_pink) == 1:
            out["good_frame"] = True
            if self.frame_recs[0].add_and_check_stable(good_pink[0]) is True:
                out["next"] = True
        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_axle(self, img, count):
        """
        Insert the wheel axle with wheel through the washers and pink gear, through to other side

        Completion: axle detected inside frame (trained on specific lighting conditions of axle in frame)
        """
        name = "axle_into_frame_%s" % count
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Great, now insert the axle through the washers and the pink gear. Then give me a birds eye view."
            out["video"] = video_url + name + ".mp4"
            return out

        axles = self.get_objects_by_categories(img, {"axle_in_frame_good"})

        # need to handle 1 and 2 cases because more than one could be visible based on first and second repetition
        if 0 < len(axles) < 3:
            if count == 1 and len(axles) == 1:
                ax = axles[0]
            elif count == 2 and len(axles) == 2:
                _, ax = separate_two(axles, True)
            else:
                self.all_staged_clear()
                return out
            out["good_frame"] = True
            axle_check = self.frame_recs[2].add_and_check_stable(ax)
            if axle_check:
                out["next"] = True
        else:
            self.all_staged_clear()

        return out

    def press_wheel(self, img, count):
        """
        Press the other wheel into the axle inside the frame

        Errors:
        1. Pressing the wrong type of wheel

        Completion: both wheels detected from birds-eye view
        """
        name = "press_wheel_%s" % count
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        good_str = "thin" if count == 1 else "thick"

        if self.history[name] is False:
            self.clear_states()
            self.history[name] = True
            out["speech"] = "Press the other %s wheel into the axle. Then, show me the bird's eye view." % good_str
            out["video"] = video_url + name + ".mp4"
            return out

        wheels = self.get_objects_by_categories(img, {"thick_wheel_side", "thin_wheel_side"})

        # need to handle two cases for two iterations (second iteration has front wheels in already)
        if len(wheels) == 2 or len(wheels) == 4:
            out["good_frame"] = True
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
                    out["speech"] = "You put in the %s wheel. Please use the %s wheel instead." % (correct_str,good_str)
                    self.clear_states()
                    self.delay_flag = True
                else:
                    out["next"] = True
                    self.clutter_reset()
        elif (len(wheels) > 2 and count == 1) or len(wheels) > 4:
            self.clutter_add()
        else:
            self.frame_recs[0].staged_clear()
            self.frame_recs[1].staged_clear()
        
        if self.clutter_check(clutter_threshold):
            out["speech"] = clutter_speech
            self.delay_flag = True
            
        return out

    def insert_pink_gear_back(self, img):
        """
        Place the back pink gear into the frame

        Errors:
        1. Pink gear put in facing the wrong way

        Completion: Pink gear in the frame facing the correct way, birds-eye view
        """
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history["back_pink_gear_1"] is False:
            self.clear_states()
            self.history["back_pink_gear_1"] = True
            out["speech"] = "Now. Place the other pink gear into the frame. Make sure the teeth point away from you."
            out["image"] = read_image("pink_gear_2.jpg")
            return out

        gear = self.get_objects_by_categories(img,{"back_pink","pink_back"})

        # traditional image processing. side with more dark pixels == probably where the teeth are
        if len(gear) == 1:
            out["good_frame"] = True
            if self.frame_recs[0].add_and_check_stable(gear[0]):
                img = img[int(gear[0]['dimensions'][1]):int(gear[0]['dimensions'][3]),int(gear[0]['dimensions'][0]):int(gear[0]['dimensions'][2])]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # resize (unclear if this matters)
                scale_percent = 400
                width = int(img.shape[1] * scale_percent / 100)
                height = int(img.shape[0] * scale_percent / 100)
                dim = (width, height)
                img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


                # cut black parts from the up
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

                # cut black parts from the down
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

                # count dark pixels for left and right side of the screen
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
                    out["speech"] = "Please turn the pink gear around so that the teeth are pointed away from you."
                    self.delay_flag = True
                self.frame_recs.clear()

        else:
            self.frame_recs[0].staged_clear()

        return out

    def insert_brown_gear_back(self, img):
        """
        Place the back brown gear into the frame

        Errors:
        1. Brown gear put in facing the wrong way

        Completion: Brown gear in the frame facing the correct way, birds-eye view
        """
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history["insert_brown_gear_back"] is False:
            self.clear_states()
            self.history["insert_brown_gear_back"] = True
            out["speech"] = "Find the brown gear and place it next to the pink gear. Make sure the nudge on the brown gear is points towards the pink gear."
            out["image"] = read_image("brown_gear.jpg")
            return out
        
        brown_gear = self.get_objects_by_categories(img, {"brown_good", "brown_bad"})

        # detects the correct state using ML object detection
        if len(brown_gear) == 1:
            out["good_frame"] = True
            if self.frame_recs[0].add_and_check_stable(brown_gear[0]):
                if self.frame_recs[0].averaged_class() != "brown_good":
                    out["speech"] = "Please make sure the nudge on the brown gear points towards the pink gear."
                else:
                    out["next"] = True
        else:
            self.frame_recs[0].staged_clear()
        return out

    def add_gear_axle(self, img):
        """
        Place the gear axle to connect both front and back gear systems

        Errors:
        1. Gears on gear axle aren't close to the front pink gear (back gear on gear axle) had difficulty getting picked up)

        Completion:
        1. Gears on gear axle placed close to the front pink gear
        """
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history["add_gear_axle"] is False:
            self.clear_states()
            self.history["add_gear_axle"] = True
            out["speech"] = "Finally, find the gear axle. Use it to connect the two gear systems together."
            out["video"] = video_url + "gear_axle.mp4"
            return out

        gear_on_axle = self.get_objects_by_categories(img, {"gear_on_axle"})
        front_pink_gear = self.get_objects_by_categories(img, {"front_gear_good"})

        # only checks that the gear on gear axle for the front gear is in, back gear couldn't be detected
        # TODO make it work for both sides
        front_check = False

        if 0 < len(gear_on_axle) < 3:
            out["good_frame"] = True
            if len(gear_on_axle) == 2:
                left, right = separate_two(gear_on_axle, True)
            else:
                left = gear_on_axle[0]
            left_check = self.frame_recs[0].add_and_check_stable(left)

            if len(front_pink_gear) == 1:
                front_gear_check = self.frame_recs[2].add_and_check_stable(front_pink_gear[0])
                if left_check is True and front_gear_check is True:
                    if check_gear_axle_front(self.frame_recs[0].averaged_bbox(), self.frame_recs[2].averaged_bbox()):
                        front_check = True
                    else:
                        out["speech"] = "The gear axle doesn't seem to be placed correctly. Make sure its teeth touch the teeth of the pink gears."
                        self.clear_states()
                        self.delay_flag = True
            else:
                self.frame_recs[2].staged_clear()

            if front_check:
                out["next"] = True
        else:
            self.all_staged_clear()

        return out

    def final_check(self, img):
        """
        Final check of everything

        Completion: wheels aren't mismatched, gears are facing correct way, gear axle placed correctly
        """
        out = defaultdict(lambda: None)
        out["good_frame"] = False
        if self.history["final_check_1"] is False:
            self.clear_states()
            self.history["final_check_1"] = True
            out["speech"] = "Great job! Now, let me do a final check on everything. Please show me a birds-eye view."
            out["image"] = read_image("final_check.jpg")
            return out
        elif self.history["final_check_2"] is False:  # check wheels
            wheels = self.get_objects_by_categories(img, {"thin_wheel_side", "thick_wheel_side"})

            if len(wheels) == 4:
                out["good_frame"] = True
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
                        out["speech"] = "The wheels look good! Please stay still for a little longer. Now I'm checking the gears."
                        self.clear_states()
            else:
                self.all_staged_clear()

        elif self.history["final_check_3"] is False:  # check gears
            gears = self.get_objects_by_categories(img, {"front_gear_good", "front_gear_bad", "back_pink", "brown_bad", "brown_good", "pink_back"})
            
            if len(gears) == 3:
                out["good_frame"] = True
                left, right = separate_two(gears, True)
                if self.frame_recs[0].add_and_check_stable(left) and self.frame_recs[1].add_and_check_stable(right):

                    if self.frame_recs[1].averaged_class() == "brown_bad":
                        out["speech"] = "The brown gear is facing the wrong way. Please flip it."
                        self.delay_flag = True
                        self.clear_states()
                    elif self.frame_recs[0].averaged_class() == "front_gear_bad":
                        out["speech"] = "The left pink gear is facing the wrong way. Please flip it."
                        self.delay_flag = True
                        self.clear_states()
                    else:
                        out["next"] = True
            else:
                self.frame_recs[0].staged_clear()
                self.frame_recs[1].staged_clear()
                self.frame_recs[2].staged_clear()


        return out

    def complete(self):
        """
        Completed assembly message
        """
        out = defaultdict(lambda: None)
        if self.history["complete"] is False:
            self.history["complete"] = True
            out["speech"] = "Everything looks good. Great job! We've finished assembling the wheels and gear train!"
            out["next"] = True

        return out

    # Utility functions
    def clear_states(self):
        """
        Clear all frame recorders
        """
        for rec in self.frame_recs.values():
            rec.clear()

    def all_staged_clear(self):
        """
        Staged clear all frame recorders
        """
        for rec in self.frame_recs.values():
            rec.staged_clear()
    
    def clutter_add(self):
        self.clutter_count += 1

    def clutter_reset(self):
        self.clutter_count = 0

    def clutter_check(self, clutter_limit):
        """
        Check whether or not the workspace is cluttered
        """
        if self.clutter_count == clutter_limit:
            self.clutter_reset()
            return True
        return False

def check_gear_axle_front(gear_on_axle_box, pink_box):
    """
    Check that the front gear on the gear axle intersects with the front pink gear
    """
    intersecting = object_detection.intersecting_bbox(gear_on_axle_box, pink_box)
    # to_right = bbox_center(gear_on_axle_box)[0] > bbox_center(pink_box)[0]
    
    return intersecting

def check_gear_axle_back(gear_on_axle_box, pink_box, brown_box):
    """
    Check that the back gear on the gear axle intersects with the brown and pink back gears
    Unused
    """
    intersecting = object_detection.intersecting_bbox(gear_on_axle_box, pink_box) and \
                   object_detection.intersecting_bbox(gear_on_axle_box, brown_box)

    gear_axle_center = bbox_center(gear_on_axle_box)
    between = bbox_center(pink_box)[1] < gear_axle_center[1] < bbox_center(brown_box)[1]

    return intersecting and between

def separate_two(objects, left_right=True):
    """
    Given two objects, return which is on the left and which is on the right
    """
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
    """
    Given four objects, return which is in the top left, top right, bottom left, bottom right in that order
    """
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
    """
    Euclidean distance between the centers of two bounding boxes
    """
    center1 = bbox_center(box1)
    center2 = bbox_center(box2)

    x_diff = abs(center1[0] - center2[0])
    y_diff = abs(center1[1] - center2[1])

    return math.sqrt(x_diff**2 + y_diff**2)

def compare(box1, box2, threshold):
    """
    Return a string based on which box is the taller box, or "same" if the difference is under a threshold
    """
    height1 = bbox_height(box1)
    height2 = bbox_height(box2)

    diff = abs(height1 - height2)
    if diff < threshold:
        return "same"

    return "first" if height1 > height2 else "second"

def read_image(name):
    """
    Helper for reading image from a resource directory
    """
    image_path = os.path.join(resources, name)
    return cv2.imread(image_path)

def get_orientation(side_marker, horn):
    """
    Get the orientation of the black frame from a side view
    Unused for deliverable. Used for state based.
    """
    # determine which side is shown, where left and right is determined from a birds-eye view of frame
    side = "left" if side_marker["class_name"] == "frame_marker_left" else "right"

    left_obj, right_obj = separate_two([side_marker, horn], True)
    if side == "left":
        flipped = left_obj["class_name"] == "frame_horn"  # determine whether or not frame is flipped over
    else:
        flipped = right_obj["class_name"] == "frame_horn"

    return side, flipped

def check_dark_pixel(pixel,threshold):
    """
    Binary reduction of pixel into whether or not it's a light or dark pixel, based on a threshold
    """
    return True if pixel <= threshold * 255 else False
