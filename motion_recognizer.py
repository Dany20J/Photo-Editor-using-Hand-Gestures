

import numpy as np
from abc import ABC, abstractmethod

from temporal_frame_info_list import TemporalFrameInfoList

class MotionRecognizer(ABC):
    def __init__(self, mn_threshold_to_react, temporal_frame_info_list=TemporalFrameInfoList()) -> None:
        self.temporal_frame_info_list = temporal_frame_info_list
        self.mn_threshold_to_react = mn_threshold_to_react
        # self.mn_threshold_to_react = 15

    @abstractmethod
    def recognize_motion(self):
        pass

    def apply_evaluation_to_points_list(self, evaluation):
        points_num_in_consideration = np.arange(
            0, 50) + self.mn_threshold_to_react

        for pn_in_consi in points_num_in_consideration:
            points = []
            nothing_left = len(self.temporal_frame_info_list) < pn_in_consi
            for ind, (if_consumed_frame, frame_info) in enumerate(self.temporal_frame_info_list.reversed_iterator()):
                if ind == pn_in_consi and evaluation(points):
                    self.temporal_frame_info_list.modify_as_consumed_from_end(
                        ind)
                    return self.get_proper_event(True)
                if ind == pn_in_consi:
                    break
                if if_consumed_frame:
                    nothing_left = True
                if nothing_left:
                    break

                points.append(frame_info.deep_copy_point())
            if nothing_left:
                break
        return self.get_proper_event(False)

    def distance(self, point_1, point_2):
        return np.sqrt(np.sum(np.square(point_1 - point_2)))

    def angle_between_2_intersected_lines(self, point_cen, point_1, point_2):
        point_1 -= point_cen
        point_2 -= point_cen

        ori_point_1_angle = np.arctan2(point_1[1], point_1[0])
        ori_point_2_angle = np.arctan2(point_2[1], point_2[0])

        return ((ori_point_2_angle - ori_point_1_angle) * 180 / np.pi + 360) % 360
    @abstractmethod
    def get_proper_event(self):
        pass

