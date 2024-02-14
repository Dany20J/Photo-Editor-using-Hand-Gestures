

import numpy as np
import cv2 as cv
from motion_recognizer import MotionRecognizer
from event import NoEvent, TopToBottomLineMotionEvent, BottomToTopLineMotionEvent, LeftToRightLineMotionEvent, RightToLeftLineMotionEvent


class LineMotionRecognizer(MotionRecognizer):
    def __init__(self, horizontal=True, vertical=False, left_to_right=True, top_to_bottom=False,
                 temporal_frame_info_list=None, mn_threshold_to_react=20,
                 rectangle_width=1000, rectangle_height=300, distance_percentage=0.75,
                 in_points_percentage=0.7, directions_percentage=0.6) -> None:
        if temporal_frame_info_list == None:
            super(LineMotionRecognizer, self).__init__(mn_threshold_to_react)
        else:
            super(LineMotionRecognizer, self).__init__(mn_threshold_to_react,
                                                       temporal_frame_info_list)

        self.horizontal = horizontal
        self.vertical = vertical
        self.left_to_right = left_to_right
        self.top_to_bottom = top_to_bottom
        self.rectangle_width = rectangle_width
        self.rectangle_height = rectangle_height
        self.distance_percentage = distance_percentage
        self.in_points_percentage = in_points_percentage
        self.directions_percentage = directions_percentage
        assert(horizontal != vertical)
        assert((horizontal and not top_to_bottom)
               or (vertical and not left_to_right))

    def recognize_motion(self):
        return self.apply_evaluation_to_points_list(self.evaluate_line)

    def is_point_in_rectangle(self, point, centre_point):
        if self.horizontal:
            top_left_point = centre_point - \
                np.array([self.rectangle_height / 2, 0], dtype=np.int32)
            top_left_point[1] = 0
            bottom_right_point = centre_point + \
                np.array([self.rectangle_height / 2, 0], dtype=np.int32)
            bottom_right_point[1] = self.rectangle_width
        elif self.vertical:
            top_left_point = centre_point - \
                np.array([0, self.rectangle_width / 2], dtype=np.int32)
            top_left_point[0] = 0
            bottom_right_point = centre_point + \
                np.array([0, self.rectangle_width / 2], dtype=np.int32)
            bottom_right_point[0] = self.rectangle_height
        
        return (np.logical_and((point >= top_left_point), (point <= bottom_right_point))).all()

    def is_valid_distance(self, distances):
        distances = np.array(distances)
        return (np.sum(distances) / self.rectangle_width if self.horizontal else self.rectangle_height) >= self.distance_percentage

    def are_valid_points(self, in_points, num_points):
        return (in_points / num_points) >= self.in_points_percentage

    def project_point(self, point):
        if self.horizontal:
            proj = np.array([1, 0], dtype=np.float32)
            return np.dot(point, proj) * proj

        elif self.vertical:
            proj = np.array([0, 1], dtype=np.float32)
            return np.dot(point, proj) * proj
        return None

    def direction(self, cur_point, prev_point):
        if np.abs(np.sum(cur_point - prev_point)) == 0:
            return 1
        if self.horizontal:
            if self.left_to_right:
                return np.sum(prev_point - cur_point) / np.abs(np.sum(cur_point - prev_point))
            elif not self.left_to_right:
                return np.sum(cur_point - prev_point) / np.abs(np.sum(cur_point - prev_point))

        elif self.vertical:
            if self.top_to_bottom:
                return np.sum(cur_point - prev_point) / np.abs(np.sum(cur_point - prev_point))
            elif not self.top_to_bottom:
                return np.sum(prev_point - cur_point) / np.abs(np.sum(cur_point - prev_point))

    def is_valid_direction(self, directions, num_directions):
        directions = np.array(directions)
        directions[directions < 0] = 0
        return ((np.sum(directions) / num_directions) >= self.directions_percentage)

    def evaluate_line(self, points):
        num_points = len(points)
        distances = np.zeros(num_points - 3)
        directions = np.zeros(num_points - 3)
        centre_point = np.sum(points, axis=0) / num_points

        in_points = 0
        proj = []
        for idx, point in enumerate(points):
            if self.is_point_in_rectangle(point, centre_point):
                in_points += 1
            if idx >= 3:
                cur_point = self.project_point(np.copy(point))
                prev_point = self.project_point(np.copy(points[idx - 3]))
                distances[idx - 3] = self.distance(cur_point, prev_point)
                directions[idx - 3] = self.direction(cur_point, prev_point)
            proj.append(self.project_point(np.copy(point)))

        is_valid_distance = self.is_valid_distance(distances)
        if not is_valid_distance:
            return False
        are_valid_points = self.are_valid_points(in_points, num_points)
        if not are_valid_points:
            return False
        is_valid_direction = self.is_valid_direction(
            directions, num_points - 1)

        return is_valid_distance and are_valid_points and is_valid_direction

    def get_proper_event(self, found=False):
        if not found:
            return NoEvent()
        else:
            if self.horizontal and self.left_to_right:
                return LeftToRightLineMotionEvent()
            elif self.horizontal and not self.left_to_right:
                return RightToLeftLineMotionEvent()
            elif self.vertical and self.top_to_bottom:
                return TopToBottomLineMotionEvent()
            elif self.vertical and not self.top_to_bottom:
                return BottomToTopLineMotionEvent()
