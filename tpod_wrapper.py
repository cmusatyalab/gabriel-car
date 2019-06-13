import requests
import cv2
import json
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

    for obj in converted:
        formatted = {}
        formatted["class_name"] = obj[0]
        formatted["dimensions"] = obj[1]
        formatted["confidence"] = obj[2]
        detected_objects.append(formatted)

    return detected_objects
