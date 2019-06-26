import requests
import cv2
import ast

import handtracking.utils.detector_utils as detector_utils


def detect_hand(img, detection_graph, sess):
    detected_hands = []
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    width, height = img.shape[1], img.shape[0]

    boxes, scores = detector_utils.detect_objects(rgb, detection_graph, sess)

    for i in range(min(2, len(boxes))):
        if scores[i] > 0.45:
            (left, right, top, bottom) = (boxes[i][1] * width, boxes[i][3] * width, boxes[i][0] * height, boxes[i][2] * height)
            detected_hands.append({
                "class_name": "hand",
                "dimensions": (left, top, right, bottom),
                "confidence": scores[i]})

    if len(detected_hands) > 1:
        intersecting = intersecting_objs(detected_hands[0], detected_hands[1])
        if intersecting is not False:
            detected_hands = [intersecting]

    return detected_hands


def detect_object(img, url):
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

        intermediate = {"class_name": obj_list_form[0], "dimensions": obj_list_form[1], "confidence": obj_list_form[2]}

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
                "confidence": obj["confidence"]})

    return detected_objects


def group_class_names(name):
    if name in {"thin_wheel_top", "thick_wheel_top"}:
        return "wheel"
    elif name in {"thin_wheel_side", "thick_wheel_side"}:
        return "tire"

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
