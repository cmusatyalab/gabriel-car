#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>
#   Modified by: Junjue Wang <junjuew@cs.cmu.edu>
#
#   Copyright (C) 2011-2013 Carnegie Mellon University
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#

'''
This is a simple library file for common CV tasks
'''
from __future__ import absolute_import, division, print_function

import cv2
import numpy as np


def raw2cv_image(raw_data, gray_scale=False):
    img_array = np.asarray(bytearray(raw_data), dtype=np.int8)
    if gray_scale:
        cv_image = cv2.imdecode(img_array, 0)
    else:
        cv_image = cv2.imdecode(img_array, -1)
    return cv_image


def cv_image2raw_jpg(img, jpeg_quality=95):
    result, data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    raw_data = data.tostring()
    return raw_data

def cv_image2raw_png(img):
    result, data = cv2.imencode('.png', img)
    raw_data = data.tostring()
    return raw_data


def vis_detections(img, dets, thresh=0.5):
    # dets format: [{"class_name": *object name*, "dimensions": *bounding box dimensions*, "confidence": *confidence of recognition*}]

    img_detections = img.copy()

    for obj in dets:
        color = (77, 255, 9) if obj["class_name"] == "hand" else (0, 0, 255)

        bbox = obj["dimensions"]
        cv2.rectangle(img_detections, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 8)
        text = "%s : %f" % (obj["class_name"], obj["confidence"])
        cv2.putText(img_detections, text, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return img_detections
