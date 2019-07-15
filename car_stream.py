#!/usr/bin/env python
#
# Cloudlet Infrastructure for Mobile Computing
#
#   Author: Kiryong Ha <krha@cmu.edu>
#           Zhuo Chen <zhuoc@cs.cmu.edu>
#           Junjue Wang <junjuew@cs.cmu.edu>
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

from __future__ import print_function

import json
import multiprocessing
import os
import pprint
import Queue
import struct
import sys
import time
from base64 import b64encode
from optparse import OptionParser

import config
import cv2
import gabriel
import gabriel.proxy
import car_task_stream
import util
import object_detection

# from handtracking.utils import detector_utils

LOG = gabriel.logging.getLogger(__name__)
ANDROID_CLIENT = True
config.setup(is_streaming=True)
display_list = config.DISPLAY_LIST_TASK
# detection_graph, sess = detector_utils.load_inference_graph()


def process_command_line(argv):
    VERSION = 'gabriel proxy : %s' % gabriel.Const.VERSION
    DESCRIPTION = "Gabriel cognitive assistance"

    parser = OptionParser(usage='%prog [option]', version=VERSION,
                          description=DESCRIPTION)

    parser.add_option(
        '-s', '--address', action='store', dest='address',
        help="(IP address:port number) of directory server")
    parser.add_option(
        '-i', '--state', action='store', dest='init_state',
        default=None,
        help="initial state")
    settings, args = parser.parse_args(argv)
    if len(args) >= 1:
        parser.error("invalid arguement")

    if hasattr(settings, 'address') and settings.address is not None:
        if settings.address.find(":") == -1:
            parser.error("Need address and port. Ex) 10.0.0.1:8081")
    return settings, args


class CarApp(gabriel.proxy.CognitiveProcessThread):

    def __init__(self, image_queue, output_queue, engine_id, init_state=None):
        super(CarApp, self).__init__(image_queue, output_queue, engine_id)
        self.is_first_image = True
        self.first_n_cnt = 0
        self.last_msg = ""
        self.dup_msg_cnt = 0
        # task initialization
        self.task = car_task_stream.Task(init_state=init_state)

    def add_to_byte_array(self, byte_array, extra_bytes):
        return struct.pack("!{}s{}s".format(len(byte_array), len(extra_bytes)), byte_array, extra_bytes)

    def add_output_item(self, header, data, itm_header_key, itm_data):
        header[itm_header_key] = (len(data), len(itm_data))
        return self.add_to_byte_array(data, itm_data)

    def gen_output(self, header, img, speech):
        rtn_data = ""
        if img is not None:
            _, buf = cv2.imencode(".jpg", img)
            rtn_data = self.add_output_item(header,
                                            rtn_data,
                                            gabriel.Protocol_result.JSON_KEY_IMAGE,
                                            buf.tobytes())
        if speech is not None:
            rtn_data = self.add_output_item(header,
                                            rtn_data,
                                            gabriel.Protocol_result.JSON_KEY_SPEECH,
                                            speech)
        return rtn_data

    @staticmethod
    def rotate_90(img):
        rows, cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -90, 1)
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def handle(self, header, data):
        # PERFORM Cognitive Assistance Processing
        LOG.info("processing: ")
        LOG.info("%s\n" % header)

        rtn_data = {}

        if self.first_n_cnt < 10:
            header['status'] = 'success'
            # rtn_data = self.gen_output(header, None, None)
            self.first_n_cnt += 1
            return json.dumps(rtn_data)

        ## preprocessing of input image
        img = util.raw2cv_image(data)
        if config.ROTATE_IMAGE:
            img = self.rotate_90(img)
        if config.RESIZE_IMAGE:
            img = cv2.resize(img, (720, 480))

        objects = object_detection.tpod_request(img, "http://0.0.0.0:8000")
        # hands = tpod_wrapper.detect_hand(img, detection_graph, sess)
        # objects.extend(hands)

        vis_objects, instruction = self.task.get_instruction(objects, header)
        header['status'] = 'success'

        print("object detection result: %s" % [obj["class_name"] for obj in objects])
        LOG.info("object detection result: %s" % objects)

        if config.VISUALIZE_ALL:
            vis_objects = objects

        img_object = util.vis_detections(img, objects)
        rtn_data['image'] = b64encode(util.cv_image2raw_png(img_object))

        return json.dumps(rtn_data)


if __name__ == "__main__":
    result_queue = multiprocessing.Queue()
    print(result_queue._reader)

    settings, args = process_command_line(sys.argv[1:])
    ip_addr, port = gabriel.network.get_registry_server_address(settings.address)
    service_list = gabriel.network.get_service_list(ip_addr, port)
    LOG.info("Gabriel Server :")
    LOG.info(pprint.pformat(service_list))

    video_ip = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_IP)
    video_port = service_list.get(gabriel.ServiceMeta.VIDEO_TCP_STREAMING_PORT)
    ucomm_ip = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_IP)
    ucomm_port = service_list.get(gabriel.ServiceMeta.UCOMM_SERVER_PORT)

    # image receiving and processing threads
    image_queue = Queue.Queue(gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    print("TOKEN SIZE OF OFFLOADING ENGINE: %d" % gabriel.Const.APP_LEVEL_TOKEN_SIZE)
    video_receive_client = gabriel.proxy.SensorReceiveClient((video_ip, video_port), image_queue)
    video_receive_client.start()
    video_receive_client.isDaemon = True
    car_app = CarApp(image_queue, result_queue, engine_id='ribLoc', init_state=settings.init_state)
    car_app.start()
    car_app.isDaemon = True

    # result publish
    result_pub = gabriel.proxy.ResultPublishClient((ucomm_ip, ucomm_port), result_queue)
    result_pub.start()
    result_pub.isDaemon = True

    try:
        while True:
            time.sleep(1)
    except Exception as e:
        pass
    except KeyboardInterrupt as e:
        sys.stdout.write("user exits\n")
    finally:
        if video_receive_client is not None:
            video_receive_client.terminate()
        if car_app is not None:
            car_app.terminate()
        result_pub.terminate()

