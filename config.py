#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#   - Task Assistance
#
#   Author: Zhuo Chen <zhuoc@cs.cmu.edu>, Junjue Wang <junjuew@cs.cmu.edu>
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

# If True, configurations are set to process video stream in real-time (use with proxy.py)
# If False, configurations are set to process one independent image (use with img.py)
IS_STREAMING = True

# Pure state detection or generate feedback as well
RECOGNIZE_ONLY = False

# Port for communication between proxy and task server
TASK_SERVER_IP = "128.2.211.75"
TASK_SERVER_PORT = 2722

# Configs for object detection
USE_GPU = True

# Whether or not to save the displayed image in a temporary directory
SAVE_IMAGE = False

# Max image width and height
IMAGE_MAX_WH = 640

# Display
DISPLAY_MAX_PIXEL = 640
DISPLAY_SCALE = 1
DISPLAY_LIST_ALL = ['input', 'object', 'instruction']
DISPLAY_LIST_TEST = ['input', 'object', 'instruction']
DISPLAY_LIST_STREAM = []
DISPLAY_LIST_TASK = ['input', 'object', 'instruction']

# Used for cvWaitKey
DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500

ROTATE_IMAGE = False
RESIZE_IMAGE = False
VISUALIZE_ALL = False

with open('model/labels.txt', 'r') as f:
    content = f.read().splitlines()
    LABELS = content


def setup(is_streaming):
    global IS_STREAMING, DISPLAY_LIST, DISPLAY_WAIT_TIME, SAVE_IMAGE
    IS_STREAMING = is_streaming
    if not IS_STREAMING:
        DISPLAY_LIST = DISPLAY_LIST_TEST
    else:
        if RECOGNIZE_ONLY:
            DISPLAY_LIST = DISPLAY_LIST_STREAM
        else:
            DISPLAY_LIST = DISPLAY_LIST_TASK
    DISPLAY_WAIT_TIME = 1 if IS_STREAMING else 500
    SAVE_IMAGE = not IS_STREAMING
