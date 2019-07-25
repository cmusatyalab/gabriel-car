import requests
import cv2
import ast
import docker
import time
import atexit

class Detector:
    def __init__(self):
        self.tpod_url = "http://0.0.0.0:8000"

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
            }
        }
        
        self.objs_to_docker_image = {}
        for url in self.docker_image_to_objs.keys():
            objs = self.docker_image_to_objs[url]
            for o in objs:
                self.objs_to_docker_image[o] = url
        
        self.last_id = None
        self.cache = []

        self.client = docker.from_env()
        self.last_image = None
        self.container = None
        atexit.register(self.cleanup)


    def init_docker_classifier(self, objects):
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

        return True


    def detect_object(self, img, objects, f_id):
        if f_id != self.last_id:
            self.last_id = f_id
            self.cache = []

        self.init_docker_classifier(objects)

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


    def all_detected_objects(self):
        return self.cache[:]

    def cleanup(self):
        if self.container is not None:
            self.container.kill()
            self.container = None


def tpod_request(img, url):
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

        norm = obj_list_form[1][:]
        norm[0] /= img.shape[1]
        norm[2] /= img.shape[1]
        norm[1] /= img.shape[0]
        norm[3] /= img.shape[0]

        intermediate = {"class_name": obj_list_form[0], "dimensions": obj_list_form[1], "confidence": obj_list_form[2], "norm": norm}

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
    if name in {"thin_wheel_top", "thick_wheel_top"}:
        return "wheel"
    elif name in {"thin_wheel_side", "thick_wheel_side"}:
        return "tire"
    elif name in {"thin_rim_side", "thick_rim_side"}:
        return "rim"
    elif name in {"hole_green", "hole_gold"}:
        return "hole_filled"

    return name

def intersecting_objs(obj1, obj2):
    if intersecting_bbox(obj1["dimensions"], obj2["dimensions"]):
        return obj1 if obj1["confidence"] > obj2["confidence"] else obj2

    return False

def intersecting_bbox(box1, box2):
    return intersection_helper(box1, box2) or intersection_helper(box2, box1)

def intersection_helper(box1, box2):
    horiz_inter = box1[2] > box2[0] and box1[0] < box2[2]
    vert_inter = box1[3] > box2[1] and box1[1] < box2[3]

    return horiz_inter and vert_inter