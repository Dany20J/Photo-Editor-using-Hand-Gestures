from utils import *
import numpy as np
from event import *


class HandRecognitzerLocalizer:

    def __init__(self, dim_percentage_for_radius=1) -> None:
        self.rows_ind_mesh = None
        self.cols_ind_mesh = None
        self.dim_percentage_for_radius = dim_percentage_for_radius

        self.top_left = np.array([0, 0], dtype=np.int32)
        self.bottom_right = np.array([0, 0], dtype=np.int32)
        self.point = np.array([0, 0], dtype=np.int32)
        self.num_fingers = 0

    def recognize_hand(self, frame):
        frame=  cv.dilate(
            frame, np.ones((5, 5)), iterations=4)
        self.cleaned_blob, self.biggest_blob_size = keep_biggest_blobs(
            frame, blobs_to_keep=1, connectivity=4)
        # self.cleaned_blob = cv.dilate(
        #     self.cleaned_blob, np.ones((5, 5)), iterations=1)
        self.dist_transform = skeleton_distance_transform_based(
            self.cleaned_blob)
        mx_dist = np.max(self.dist_transform)
        if mx_dist > 0:
            self.skeleton = np.array(
                self.dist_transform / mx_dist * 255, np.uint8)
        else:
            self.skeleton = np.array(self.dist_transform, np.uint8)
        return mx_dist > 30

    def localize_hand_bb(self):

        abbr = self.cleaned_blob

        if self.rows_ind_mesh is None or self.cols_ind_mesh is None:
            rows_ind = np.arange(abbr.shape[0]) + 1
            cols_ind = np.arange(abbr.shape[1]) + 1
            mesh_1, mesh_2 = np.meshgrid(rows_ind, cols_ind)

            self.rows_ind_mesh = mesh_1.T
            self.cols_ind_mesh = mesh_2.T
        bigger = abbr > 0

        wanted_rows = np.multiply(bigger, self.rows_ind_mesh)
        wanted_cols = np.multiply(bigger, self.cols_ind_mesh)

        wanted_rows_for_min = wanted_rows.copy()
        wanted_cols_for_min = wanted_cols.copy()

        wanted_rows_for_min[wanted_rows_for_min == 0] = 1e9
        wanted_cols_for_min[wanted_cols_for_min == 0] = 1e9

        self.top_left = np.array(
            [np.min(wanted_rows_for_min) - 1, np.min(wanted_cols_for_min) - 1], dtype=np.int32)
        self.bottom_right = np.array(
            [np.max(wanted_rows) - 1, np.max(wanted_cols) - 1], dtype=np.int32)

        return self.top_left, self.bottom_right

    def localize_hand_center(self):
        # option 1 (get centre of bounding box )
        self.point = np.array(
            (self.top_left + self.bottom_right) // 2, dtype=np.int32)
        # option 2 (get skeleton brightest point)
        # self.point, self.mx = find_center_point_skeleton(self.dist_transform)
        return self.point


    # not used
    def localize_fingers(self):
        self.contours, _ = cv .findContours(
            self.cleaned_blob, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.contours = np.array(self.contours, dtype=object)

        largest_contour = self.get_largest_contour(self.contours)

        if largest_contour is not None:
            convex_hull = cv.convexHull(largest_contour)
            # dont return it, instead localize fingers
            return convex_hull
        return None

    def get_largest_contour(self, contours):
        if len(contours) > 1:
            if len(contours) == 1:
                return contours.squeeze()
            return contours[np.argmax(np.array(list(map(cv.contourArea, contours))))]
        return None


    def localize_fingers_option_2(self):
        

        self.width = self.bottom_right[1] - self.top_left[1]
        self.height = self.bottom_right[0] - self.top_left[0]

        self.radius_of_intersection = int(
            self.dim_percentage_for_radius * min(self.width, self.height) / 2)

        self.hand_bb = self.cleaned_blob[self.top_left[0]: self.bottom_right[0], self.top_left[1]: self.bottom_right[1]]

        self.hand_bb = cv.dilate(self.hand_bb, np.ones((5, 1)), iterations= 4)

        num_bright = np.sum(self.hand_bb[0, :] > 0)
        # if min(self.width, self.height) / max(self.width, self.height) > 0.9:
        if num_bright / self.width > 0.3:
            self.num_fingers = 0
            return 0
        empty = np.zeros(self.hand_bb.shape, dtype=np.uint8)
        circle_mask = empty
        cv.circle(circle_mask, (self.point[1] - self.top_left[1], self.point[0] - self.top_left[0]),
                  self.radius_of_intersection, (255, 0, 0), 10)

        circle_mask[circle_mask != 0] = 1
        self.possible_fingers_blobs = np.multiply(
            circle_mask, self.hand_bb)

        

        hand_blob, biggest_blob_size = keep_biggest_blobs(
            self.possible_fingers_blobs, blobs_to_keep=1)

        self.possible_fingers_blobs = self.possible_fingers_blobs - hand_blob

        self.biggest_fingers_blobs, blobs_sizes = keep_biggest_blobs(
            self.possible_fingers_blobs, blobs_to_keep=5)


        self.num_fingers = np.sum(blobs_sizes > 10)

        self.biggest_fingers_blobs, blobs_sizes = keep_biggest_blobs(
            self.possible_fingers_blobs, blobs_to_keep=self.num_fingers)
        # end debugging
        return self.num_fingers

    def recognize_hand_event(self, frame):
        if self.recognize_hand(frame):
            self.localize_hand_bb()
            self.localize_hand_center()
            self.localize_fingers_option_2()
        else:
            # return ClosedHandEvent(self.point)
           return NoEvent()
        if self.num_fingers == 0:
            return ClosedHandEvent(self.point)
        elif self.num_fingers >= 1 and self.num_fingers <= 5:
            return UpFingersEvent(self.point, self.num_fingers)
        else:
            # return ClosedHandEvent(self.point)
            return NoEvent()

