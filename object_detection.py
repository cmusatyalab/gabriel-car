import requests
import cv2
import ast
import docker
import time
import atexit

class Detector:
    """
    Object that handles all aspects of object detection including:
    1. Spinning up a specific TPOD classifier container
    2. Sending image to container for object detection
    3. Cleaning up container once finished

    Detector works with an API call containing
    1. the image
    2. what objects you want to detect
    """
    def __init__(self, url):
        self.tpod_url = url

        """
        registry of TPOD classifier docker image IDs and the objects they should be used to recognize 
        (does not need to include every object a container recognizes)
        
        every key and value should be unique. for containers that detect the same objects, just list the object in the
        container that should be used for detection
        """
        self.docker_image_to_objs = {
            "7af40405c31b": {
                "wheel_in_axle_thick",
                "wheel_in_axle_thin",
                "wheel_axle"
            },
            "2bd476517575": {
                "hole_empty",
                "hole_green",
                "hole_gold",
                "frame_marker_left",
                "frame_marker_right",
                "frame_horn"
            },
            "f1440988bafa": {
                "thick_rim_side",
                "thick_wheel_side",
                "thin_rim_side",
                "thin_wheel_side"
            },
            "8a79c18a0006": {
                "back_pink",
                "brown_bad",
                "brown_good",
                "front_gear_bad",
                "front_gear_good",
                "gear_on_axle",
                "pink_back"
            },
            "a8d3d274845f":{
                "axle_in_frame_good"
            },
            "a4b34fd8f0f6":{
                "wrong_wheel",
                "thick_rim_side",
                "thick_wheel_side",
                "thick_wheel_top",
                "thin_rim_side",
                "thin_wheel_side",
                "thin_wheel_top"
            }
        }

        # reverse look up dict
        self.objs_to_docker_image = {}
        for url in self.docker_image_to_objs.keys():
            objs = self.docker_image_to_objs[url]
            for o in objs:
                self.objs_to_docker_image[o] = url

        self.last_id = None  # image ID of last classifier (to determine whether or not to spin up a new one)
        self.cache = []  # cache of detected objects to avoid multiple calls with same image. wiped on new frame

        # Docker API to spin up/destroy containers
        self.client = docker.from_env()
        self.last_image = None
        self.container = None

        atexit.register(self.cleanup)


    def init_docker_classifier(self, objects, image_id=None):
        """
        Spin up the Docker container to detect certain objects
        :param objects: to detect
        :param image_id: overrides registry look up and spins up a specific classifier by image ID
        """
        if image_id is not None:
            image_for_objects = image_id
        else:
            for obj in objects:
                if obj not in self.objs_to_docker_image.keys():
                    raise ValueError("Unknown object %s. Make sure object is registered in object_detection.py")
                image_for_objects = self.objs_to_docker_image[obj]

        if image_for_objects == self.last_image:
            return

        self.last_image = image_for_objects
        self.cleanup()
        self.container = self.client.containers.run(image_for_objects,
                                                    "/bin/bash run_server.sh",
                                                    ports={8000: "8000/tcp"},
                                                    remove=True,
                                                    detach=True,
                                                    runtime="nvidia")
        time.sleep(4)


    def detect_object(self, img, objects, f_id, image_id=None):
        """
        Detects objects in an image

        :param img: to detect
        :param objects: expected in the img to detect
        :param f_id: frame ID to determine whether or not use cache
        :param image_id: overrides registry look up and spins up a specific classifier by image ID
        :return: list of detected objects in the form
            {
            "class_name": label of classified object
            "dimensions": bounding box dimensions in form [top-left corner x, t-l y, bottom-right corner x, br y]
                units are pixels
            "norm": normed dimensions (0 to 1)
            "confidence": confidence of detection (0 to 1)
            }
        """
        # clear cache if new frame
        if f_id != self.last_id:
            self.last_id = f_id
            self.cache = []

        self.init_docker_classifier(objects, image_id)

        out = []

        if len(self.cache) == 0:
            detected_objs = tpod_request(img, self.tpod_url)
            self.cache = detected_objs
        else:
            detected_objs = self.cache

        for d in detected_objs:
            if d["class_name"] in objects:
                out.append(d)

        return out

    def color_detected_object(self, color_dict):
        """
        Adds a color field to detected object in cache
        :param color_dict: mapping objects to colors
        """
        for obj in self.cache:
            if obj["class_name"] in color_dict.keys():
                obj["color"] = color_dict[obj["class_name"]]

    def all_detected_objects(self):
        """
        Returns all object detections from this frame, regardless of the objects requested in detect_object call
        :return: list of detected objects
        """
        return self.cache[:]

    def cleanup(self):
        """
        Stop Docker container if it's running
        """
        if self.container is not None:
            self.container.kill()
            self.container = None

    def reset(self):
        """
        Reset detector for a new client connection
        """
        self.last_image = None
        self.cleanup()


def tpod_request(img, url):
    """
    Send a TPOD HTTP request for object detection
    If bounding boxes of the same class or certain groups of classes intersect, only the highest confidence is returned
    :param img: to detect
    :param url: of TPOD classifier
    :return: objects detected
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    payload = {"confidence": 0.5, "format": "box"}

    _, img_encoded = cv2.imencode('.jpg', img)
    files = {'media': img_encoded}

    session = requests.Session()
    response = session.post(url + "/detect", headers=headers, data=payload, files=files)

    converted = ast.literal_eval(response.text)
    detected_objects = []

    by_class = {}
    for obj_list_form in converted:
        class_name = group_class_names(obj_list_form[0])
        if class_name not in by_class.keys():
            by_class[class_name] = []

        # norm dimensions field
        norm = obj_list_form[1][:]
        norm[0] /= img.shape[1]
        norm[2] /= img.shape[1]
        norm[1] /= img.shape[0]
        norm[3] /= img.shape[0]

        intermediate = {"class_name": obj_list_form[0], "dimensions": obj_list_form[1], "confidence": obj_list_form[2], "norm": norm}

        # wipe intersecting bounding boxes for same class or certain groups of classes
        conflicts = [x for x in by_class[class_name] if intersecting_objs(intermediate, x)]
        non_conflicts = [x for x in by_class[class_name] if x not in conflicts]

        highest_confidence_obj = intermediate
        for other in conflicts:
            if other["confidence"] > highest_confidence_obj["confidence"]:
                highest_confidence_obj = other

        filtered = [x for x in conflicts if not intersecting_objs(highest_confidence_obj, x)]
        resolved = non_conflicts + filtered + [highest_confidence_obj]

        by_class[class_name] = resolved

    for class_name in by_class:
        for obj in by_class[class_name]:
            detected_objects.append({
                "class_name": obj["class_name"],
                "dimensions": obj["dimensions"],
                "confidence": obj["confidence"],
                "norm": obj["norm"]})

    return detected_objects


def group_class_names(name):
    """
    Returns the group name of a class, for intersecting bounding box reduction
    :param name: of class
    :return: group name of class
    """
    if name in {"thin_wheel_top", "thick_wheel_top"}:
        return "wheel"
    elif name in {"thin_wheel_side", "thick_wheel_side"}:
        return "tire"
    elif name in {"thin_rim_side", "thick_rim_side"}:
        return "rim"
    elif name in {"hole_green", "hole_gold"}:
        return "hole_filled"
    elif name in {"front_gear_bad", "front_gear_good"}:
        return "front_gear"
    elif name in {"brown_bad", "brown_good"}:
        return "brown_gear"

    return name

def intersecting_objs(obj1, obj2):
    """
    Checks if two objects are intersecting and returns the more confident one if so.
    :param obj1: object to check intersection
    :param obj2: other object to check intersection
    :return: the object of higher confidence if bounding boxes intersect, otherwise False
    """
    if intersecting_bbox(obj1["dimensions"], obj2["dimensions"]):
        return obj1 if obj1["confidence"] > obj2["confidence"] else obj2

    return False

def intersecting_bbox(box1, box2):
    return intersection_helper(box1, box2) or intersection_helper(box2, box1)

def intersection_helper(box1, box2):
    horiz_inter = box1[2] > box2[0] and box1[0] < box2[2]
    vert_inter = box1[3] > box2[1] and box1[1] < box2[3]

    return horiz_inter and vert_inter