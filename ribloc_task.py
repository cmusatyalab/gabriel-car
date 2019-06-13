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
from __future__ import print_function

from collections import defaultdict

import cv2
import numpy as np

import config

OBJECTS = config.LABELS
STATES = ["start", "nothing", "bone", "gauge", "measure", "read",
          "grp-tg", "assemble", "bone-grp-tg", "gdd", "drill", "screw", "gnb", "finished"]


class Task:
    def __init__(self, init_state=None):
        # build a mapping between faster-rcnn recognized object order to a standard order
        if init_state is None:
            self.current_state = "start"
        else:
            if init_state not in STATES:
                raise ValueError('Unknown init state: {}'.format(init_state))
            self.current_state = init_state
        self.cnt = 0

    def get_objects_by_categories(self, objects, categories):
        cats_objs = []
        for cat in categories:
            cat_objs = self.get_objects_by_category(objects, cat)
            cats_objs.append(cat_objs)
        return np.vstack(cats_objs)

    def get_objects_by_category(self, objects, category):
        cat_objs = objects[np.where(objects[:, -1] == OBJECTS.index(category))]
        # sort by confidence
        cat_objs[np.argsort(cat_objs[:, -2])[::-1]]
        return cat_objs

    @staticmethod
    def intersection(a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if (x2 - x1) < 0 or (y2 - y1) < 0: return (0, 0, 0, 0)  # or (0,0,0,0) ?
        return [x1, y1, x2, y2]

    @staticmethod
    def area(bx):
        return (bx[2] - bx[0]) * (bx[3] - bx[1])

    def _check_drill(self, objects):
        bone_bx = self.get_objects_by_category(objects, 'bone')
        dh_bx = self.get_objects_by_category(objects, 'dh')
        gd_bx = self.get_objects_by_category(objects, 'gd')
        if len(bone_bx) > 0 and len(dh_bx) == 0 and len(gd_bx) == 0:
            self.cnt += 1
        else:
            self.cnt = 0
        if self.cnt == 6:
            self.cnt = 0
            return True
        else:
            return False

    def _check_measure(self, objects):
        bone_bx = self.get_objects_by_category(objects, 'bone')[0]
        gauge_bx = self.get_objects_by_category(objects, 'gauge')[0]
        intersect = self.intersection(bone_bx, gauge_bx)
        intersect_percent = self.area(intersect) / self.area(gauge_bx)
        print('gauge_bx: {} area:{} '.format(gauge_bx, self.area(gauge_bx)))
        print('intersect: {} area: {} percent:{}'.format(intersect,
                                                         self.area(intersect),
                                                         intersect_percent))
        if intersect_percent > 0.3 and intersect_percent < 0.6:
            return "wrong position"

        if intersect_percent > 0.6 \
                and (gauge_bx[3] - bone_bx[1]) > 0 \
                and (gauge_bx[3] - bone_bx[1]) / (bone_bx[3] - bone_bx[1]) < 0.7:  # vertical range
            bone_bx_w = bone_bx[2] - bone_bx[0]
            print('gauge_bx[2]:{} bone_bx[0]:{} bone_w:{}'.format(gauge_bx[2], bone_bx[0], bone_bx_w))
            if gauge_bx[2] > (bone_bx[0] + 0.3 * bone_bx_w) and gauge_bx[2] < (bone_bx[0] + 0.8 * bone_bx_w):
                return "wrong position"
            return True
        return False

    def _check_secure(self, objects):
        bone_bx = self.get_objects_by_category(objects, 'sbv')[0]
        sa_bx = self.get_objects_by_category(objects, 'sa')[0]
        intersect = self.intersection(bone_bx, sa_bx)
        intersect_percent = self.area(intersect) / self.area(sa_bx)
        print('sa_bx: {} sbv:{} area:{} '.format(sa_bx, bone_bx, self.area(sa_bx)))
        print('intersect: {} area: {} percent:{}'.format(intersect,
                                                         self.area(intersect),
                                                         intersect_percent))
        if intersect_percent <= 0.3:
            return False

        if intersect_percent > 0.4 and intersect_percent < 1.0:
            return "wrong position"

        if (sa_bx[1] - bone_bx[1]) > 0.2 * (bone_bx[3] - bone_bx[1]) \
                and (sa_bx[3] - bone_bx[1]) < 0.8 * (bone_bx[3] - bone_bx[1]) \
                and (sa_bx[0] - bone_bx[0]) > 0.3 * (bone_bx[3] - bone_bx[1]) \
                and (sa_bx[0] - bone_bx[0]) < 0.8 * (bone_bx[3] - bone_bx[1]):
            return True
        else:
            return "wrong position"

    def _check_assemble(self, objects):
        grp_bx = self.get_objects_by_category(objects, 'grp')[0]
        tgs_bx = self.get_objects_by_category(objects, 'tg')[:2]
        succ = True
        for tg_bx in tgs_bx:
            y_center = (tg_bx[1] + tg_bx[3]) / 2.0
            tg_h = tg_bx[3] - tg_bx[1]
            # x should be aligned with grp
            if (tg_bx[0] < 0.95 * grp_bx[0]) or (tg_bx[2] > 1.05 * grp_bx[2]):
                succ = False;
            # y should be relatively close
            if abs(grp_bx[3] - y_center) > (0.55 * tg_h):
                succ = False

        return succ

    def get_instruction(self, objects, header=None):
        result = defaultdict(lambda: None)
        result['status'] = "success"
        vis_objects = np.asarray([])

        # the start
        if self.current_state == "start":
            result['speech'] = "Put the bone on the table"
            image_path = "images_feedback/bone.png"
            result['image'] = cv2.imread(image_path) if image_path else None
            self.current_state = "nothing"
            return vis_objects, result

        # objects is an numpy array of (detect_idx, 6)
        if len(objects.shape) < 2:  # nothing detected
            return vis_objects, result

        # get the count of detected objects
        object_counts = defaultdict(lambda: 0)
        for i in xrange(len(OBJECTS)):
            object_counts[OBJECTS[i]] = sum(objects[:, -1] == i)

        print('objects:{}'.format(object_counts))
        if self.current_state == "nothing":
            vis_objects = self.get_objects_by_categories(objects, ['bone'])
            if object_counts['bone'] > 0:
                result['speech'] = "Now find the gauge and put it on the table"
                image_path = "images_feedback/bone-gauge.png"
                result['image'] = cv2.imread(image_path) if image_path else None
                self.current_state = "gauge"
        elif self.current_state == "gauge":
            vis_objects = self.get_objects_by_categories(objects, ['gauge'])
            if object_counts['gauge'] > 0:
                result['speech'] = "Good job. Now measure the bone thickness"
                image_path = "images_feedback/measure.png"
                result['image'] = cv2.imread(image_path) if image_path else None
                self.current_state = "measure"
        elif self.current_state == "measure":
            vis_objects = self.get_objects_by_categories(objects, ['bone', 'gauge'])
            if object_counts['bone'] > 0 and object_counts['gauge'] > 0:
                if self._check_measure(objects) == True:
                    result['speech'] = "Great. Please read the color on the gauge"
                    image_path = "images_feedback/read.png"
                    result['image'] = cv2.imread(image_path) if image_path else None
                    self.current_state = "read"
                elif self._check_measure(objects) == "wrong position":
                    result['speech'] = "Please place the gauge near the fracture"
        elif self.current_state == "read":
            if header != None and ('gaugeColor' in header):
                if header['gaugeColor'] == 'green':
                    result['speech'] = "Great. Please put the green ribplate and two target guides on the table"
                    image_path = "images_feedback/grp-tg.png"
                    result['image'] = cv2.imread(image_path) if image_path else None
                    header['clear_color'] = True
                    self.current_state = "grp-tg"
                else:
                    # should be green
                    result['speech'] = "The color is wrong. Please measure again and tell me the reading"
                    image_path = "images_feedback/measure.png"
                    result['image'] = cv2.imread(image_path) if image_path else None
                    header['clear_color'] = True
                    self.current_state = "read"
        elif self.current_state == "grp-tg":
            vis_objects = self.get_objects_by_categories(objects, ['brp', 'frp', 'grp', 'tg'])
            if object_counts['brp'] > 0 or object_counts['frp'] > 0:
                result['speech'] = "You are using the wrong ribplate. Please find the green one"
            elif object_counts['grp'] > 0 and object_counts['tg'] > 1:
                result['speech'] = "Great. Now assemble the target guides onto the ribplate"
                image_path = "images_feedback/assembled.png"
                result['image'] = cv2.imread(image_path) if image_path else None
                self.current_state = "assemble"
        elif self.current_state == "assemble":
            vis_objects = self.get_objects_by_categories(objects, ['grp', 'tg'])
            if object_counts['grp'] > 0 and object_counts['tg'] > 1:
                if self._check_assemble(objects):
                    result['speech'] = "Good. Put the ribplate onto the fracture. Show me the sideview."
                    image_path = "images_feedback/bone-grp-tg-sbv.png"
                    result['image'] = cv2.imread(image_path) if image_path else None
                    self.current_state = "bone-grp-tg"
        elif self.current_state == "bone-grp-tg":
            vis_objects = self.get_objects_by_categories(objects, ['sa', 'saf', 'sbv'])
            sa_bxes = self.get_objects_by_category(objects, 'sa')
            if len(sa_bxes) < 1:
                self.cnt = 0
            saf_bxes = self.get_objects_by_category(objects, 'saf')
            if object_counts['sbv'] > 0 and len(saf_bxes) > 0 and saf_bxes[0][-2] > 0.6 and (
                    object_counts['sa'] == 0 or (len(sa_bxes) > 0 and sa_bxes[0][-2] < 0.6)):
                self.cnt = 0
                result['speech'] = "You are putting the ribplate upside down. Please put it on the top edge"
            elif object_counts['sbv'] > 0 and object_counts['sa'] > 0 and sa_bxes[0][-2] > 0.7:
                if self._check_secure(objects) == True:
                    if self.cnt < 1:
                        self.cnt += 1
                    else:
                        self.cnt = 0
                        result['speech'] = "Great. Put the green drill driver onto the power drill"
                        image_path = "images_feedback/gdd.png"
                        result['image'] = cv2.imread(image_path) if image_path else None
                        self.current_state = "gdd"
                elif self._check_secure(objects) == "wrong position":
                    self.cnt = 0
                    result['speech'] = "Please place the ribplate onto the fracture"

        elif self.current_state == "gdd":
            vis_objects = self.get_objects_by_categories(objects, ['dh', 'bd', 'fd', 'gd'])
            if object_counts['dh'] > 0:
                if object_counts['bd'] > 0 or object_counts['fd'] > 0:
                    result['speech'] = "You are using the wrong drill driver. Please find the green one."
                elif object_counts['gd'] > 0:
                    result['speech'] = "Good. Drill through the target guides and put away the power drill when done"
                    image_path = "images_feedback/bone-grp-tg.png"
                    result['image'] = cv2.imread(image_path) if image_path else None
                    self.current_state = "drill"
        elif self.current_state == "drill":
            vis_objects = self.get_objects_by_categories(objects, ['bone', 'dh'])
            if object_counts['bone'] > 0 or object_counts['dh'] > 0:
                if self._check_drill(objects):
                    result['speech'] = "Good. Show me a few green screws"
                    image_path = "images_feedback/gn.png"
                    result['image'] = cv2.imread(image_path) if image_path else None
                    self.current_state = "gn"
        elif self.current_state == "gn":
            vis_objects = self.get_objects_by_categories(objects, ['gn', 'bn', 'fn'])
            bn_bxes = self.get_objects_by_category(objects, 'bn')
            fn_bxes = self.get_objects_by_category(objects, 'fn')
            if (object_counts['bn'] > 0 and bn_bxes[0][-2] > 0.7) or (object_counts['fn'] > 0 and fn_bxes[0][-2] > 0.7):
                result['speech'] = "You are using the wrong screws. Please find the green one."
            if (object_counts['gn'] + object_counts['gd']) > 1:
                result['speech'] = "Good. Insert Screws through the targeting guide and " \
                                   "remove the targeting guide when done."
                image_path = "images_feedback/gnb.png"
                result['image'] = cv2.imread(image_path) if image_path else None
                self.current_state = "gnb"
        elif self.current_state == "gnb":
            vis_objects = self.get_objects_by_categories(objects, ['gnb'])
            if object_counts['gnb'] >= 2:
                result['speech'] = "Congradulations. You have finished"
                image_path = "images_feedback/gnb.png"
                result['image'] = cv2.imread(image_path) if image_path else None
                self.current_state = "finished"
        return vis_objects, result

    def get_current_state(self):
        return self.current_state
