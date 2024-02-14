
import numpy as np

from motion_recognizer import MotionRecognizer
from event import NoEvent, ClockwiseCircularMotionEvent, CounterClockwiseCircularMotionEvent


class CircleMotionRecognizer(MotionRecognizer):

    def __init__(self, temporal_frame_info_list=None, clockwise=True, mn_threshold_to_react=20,
                 mn_circle_radius=200, mx_circle_radius=500, in_percentage_points=0.6,
                 quadrant_percentage=0.1, in_direction_percentage=0.55) -> None:
        if temporal_frame_info_list == None:
            super(CircleMotionRecognizer, self).__init__(mn_threshold_to_react)
        else:
            super(CircleMotionRecognizer, self).__init__(mn_threshold_to_react,
                                                         temporal_frame_info_list)
        assert(mn_circle_radius <= mx_circle_radius)
        self.clockwise = clockwise
        self.mn_circle_radius = mn_circle_radius
        self.mx_circle_radius = mx_circle_radius
        assert(in_percentage_points >= 0.0 and in_percentage_points <= 1.0)
        self.in_percentage_points = in_percentage_points
        self.quadrant_percentage = quadrant_percentage
        self.in_direction_percentage = in_direction_percentage

    def recognize_motion(self):
        return self.apply_evaluation_to_points_list(self.evaluate_circleness)

    def update_sum_num(self, sum_points, num_points, point):
        return sum_points + point, num_points + 1

    def is_point_in_circle_domain(self, point, centre_point):
        dist = self.distance(point, centre_point)
        return dist >= self.mn_circle_radius and dist <= self.mx_circle_radius

    def is_valid_distance(self, distances):
        distance = np.sum(distances)
        return distance >= self.mn_circle_radius * 2 * np.pi and distance <= self.mx_circle_radius * 2 * np.pi

    def are_valid_points(self, in_points, num_points):
        return (in_points / num_points) >= self.in_percentage_points

    def is_valid_angles_percentage(self, angles):

        lt_90 = sum(angles < 90) / len(angles)
        lt_180 = sum((90 <= angles) & (angles < 180)) / len(angles)
        lt_270 = sum((180 <= angles) & (angles < 270)) / len(angles)
        lt_360 = sum((270 <= angles) & (angles < 360)) / len(angles)

        return np.min(np.array([lt_90, lt_180, lt_270, lt_360]) > self.quadrant_percentage)

    def direction(self, prev_angle, cur_angle, next_angle):
        if self.clockwise:
            return np.sum((prev_angle <= cur_angle) & (cur_angle <= next_angle)) == 1
        elif not self.clockwise:
            return np.sum((prev_angle >= cur_angle) & (cur_angle >= next_angle)) == 1

    def is_valid_direction(self, directions):
        return np.sum(directions) / len(directions) > self.in_direction_percentage

    def evaluate_circleness(self, points):

        num_points = len(points)
        distances = np.zeros(num_points - 2)
        directions = np.zeros(num_points)
        angles = np.zeros(num_points, dtype=np.int32)
        centre_point = np.sum(np.array(points), axis=0) // num_points

        in_points = 0
        for idx, point in enumerate(points):
            if self.is_point_in_circle_domain(point, centre_point):
                in_points += 1
            if idx >= 2:
                distances[idx - 2] = self.distance(points[idx - 2], point)
            angles[idx] = self.angle_between_2_intersected_lines(
                centre_point, centre_point + np.array([1, 0], dtype=np.int32), point)
        for idx, point in enumerate(points):
            prev_idx = ((idx - 2) + num_points) % num_points
            next_idx = (idx + 2) % num_points
            directions[idx] = self.direction(
                points[prev_idx], points[idx], points[next_idx])

        is_valid_distance = self.is_valid_distance(distances)
        if not is_valid_distance:
            return False
        are_valid_points = self.are_valid_points(in_points, num_points)
        if not are_valid_points:
            return False
        is_valid_direction = self.is_valid_direction(directions)
        if not is_valid_direction:
            return False
        is_valid_angle_percentage = self.is_valid_angles_percentage(angles)

        return is_valid_distance and is_valid_angle_percentage and are_valid_points and is_valid_direction

    def get_proper_event(self, found=False):
        if not found:
            return NoEvent()
        else:
            if self.clockwise:
                return ClockwiseCircularMotionEvent()
            else:
                return CounterClockwiseCircularMotionEvent()
