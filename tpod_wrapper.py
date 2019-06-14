import requests
import cv2
import ast

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
        if obj_list_form[0] not in by_class.keys():
            by_class[obj_list_form[0]] = []

        intermediate = {"class_name": obj_list_form[0], "dimensions": obj_list_form[1], "confidence": obj_list_form[2]}

        conflicts = [x for x in by_class[obj_list_form[0]] if intersecting_bbox(intermediate, x)]
        non_conflicts = [x for x in by_class[obj_list_form[0]] if x not in conflicts]

        highest_confidence_obj = intermediate
        for other in conflicts:
            if other["confidence"] > highest_confidence_obj["confidence"]:
                highest_confidence_obj = other

        filtered = [x for x in conflicts if not intersecting_bbox(highest_confidence_obj, x)]
        resolved = non_conflicts + filtered + [highest_confidence_obj]

        by_class[obj_list_form[0]] = resolved

    for class_name in by_class:
        for obj in by_class[class_name]:
            detected_objects.append({
                "class_name": obj["class_name"],
                "dimensions": obj["dimensions"],
                "confidence": obj["confidence"]})

    return detected_objects


def intersecting_bbox(obj1, obj2):
    if intersection_helper(obj1["dimensions"], obj2["dimensions"]) or \
            intersection_helper(obj2["dimensions"], obj1["dimensions"]):
        return obj1 if obj1["confidence"] > obj2["confidence"] else obj2

    return False

def intersection_helper(box1, box2):
    horiz_inter = box1[2] > box2[0] and box1[0] < box2[2]
    vert_inter = box1[3] > box2[1] and box1[1] < box2[3]

    return horiz_inter and vert_inter
