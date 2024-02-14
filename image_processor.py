from frame_info import FrameInfo
from temporal_frame_info_list import TemporalFrameInfoList
import system_state as ss
from line_motion_recognizer import LineMotionRecognizer
from circular_motion_recognizer import CircleMotionRecognizer
from hand_recognitzer_localizer import HandRecognitzerLocalizer
from event import NoEvent
import cv2 as cv
import numpy as np
from utils import *
from hand_histogram import *
from static_event_approx import StaticEventApprox
from event import *
import time

class ImageProcessor:
    def __init__(self) -> None:
        self.cap = cv.VideoCapture(0)

        self.static_event_approx = StaticEventApprox()
        self.system_state = None

        self.temporal_frame_info_list = TemporalFrameInfoList()

        self.motion_recognizers = []

        self.left_to_right_line_motion_recognizer = LineMotionRecognizer(
            horizontal=True, vertical=False, left_to_right=True, top_to_bottom=False,
            rectangle_width=1000, rectangle_height=150,
            temporal_frame_info_list=self.temporal_frame_info_list
            ,mn_threshold_to_react=13)
        self.motion_recognizers.append(
            self.left_to_right_line_motion_recognizer)

        self.right_to_left_line_motion_recognizer = LineMotionRecognizer(
            horizontal=True, vertical=False, left_to_right=False, top_to_bottom=False,
            rectangle_width=1000, rectangle_height=150,
            temporal_frame_info_list=self.temporal_frame_info_list
            ,mn_threshold_to_react=13)
        self.motion_recognizers.append(
            self.right_to_left_line_motion_recognizer)

        self.top_to_bottom_line_motion_recognizer = LineMotionRecognizer(
            horizontal=False, vertical=True, left_to_right=False, top_to_bottom=True,
            rectangle_width=100, rectangle_height=1000,
            temporal_frame_info_list=self.temporal_frame_info_list
            ,mn_threshold_to_react=13)
        self.motion_recognizers.append(
            self.top_to_bottom_line_motion_recognizer)

        self.bottom_to_top_line_motion_recognizer = LineMotionRecognizer(
            horizontal=False, vertical=True, left_to_right=False, top_to_bottom=False,
            rectangle_width=100, rectangle_height=1000,
            temporal_frame_info_list=self.temporal_frame_info_list
            ,mn_threshold_to_react=13)
        self.motion_recognizers.append(
            self.bottom_to_top_line_motion_recognizer)

        self.clockwise_circular_motion_recognizer = CircleMotionRecognizer(
            temporal_frame_info_list=self.temporal_frame_info_list, clockwise=True
            ,mn_threshold_to_react=10)
        self.motion_recognizers.append(
            self.clockwise_circular_motion_recognizer)

        self.counter_clockwise_circular_motion_recognizer = CircleMotionRecognizer(
            temporal_frame_info_list=self.temporal_frame_info_list, clockwise=False
            ,mn_threshold_to_react=10)
        self.motion_recognizers.append(
            self.counter_clockwise_circular_motion_recognizer)

        self.hand_recognitzer_localizer = HandRecognitzerLocalizer()

        self.frame_num = 1
        self.quit = False
        self.hand_captured = False
        self.calculated_hand_histogram = False
        self.hand_histogram = None

        self.boxWidth = 90
        self.boxHeight = 140

        self.frame_width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.capture_boxes = []
        self.capture_boxes.append(
            (int(0.2 * self.frame_width), int(0.2 * self.frame_height)))

        self.hand_region_width = 720
        self.hand_region_height = 720

        # avg background variables
        self.frames_elapsed = 0
        self.background = None
        self.CALIBRATION_TIME = 30
        self.BG_WEIGHT = 0.5
        self.OBJ_THRESHOLD = 10

        self.region_top = 0
        self.region_bottom = 900
        self.region_left = 0
        self.region_right = 900

        self.recurrent_event = "NoEvent"


        self.interval = 5
        self.current_step = 0

    def setGUI(self, gui):
        self.gui = gui
        self.system_state = ss.SystemState(self.gui)

    def mainLoop(self):

        if self.quit:
            self.cap.release()
            cv.destroyAllWindows()
            return
        self.calculate_histogram_window = "Calculate Histogram"
        self.mutual_section()

    def camera_loop(self):

        self.calculate_histogram_window = "Calculate Histogram"
        while self.cap.isOpened() and not self.quit:
            self.mutual_section()

        self.cap.release()
        cv.destroyAllWindows()

    def mode_1(self):
        _, frame = self.cap.read()
        frame = cv.flip(frame, 1)
        ori = np.copy(frame)
        frame = frame[self.region_top:self.region_bottom,
                    self.region_left:self.region_right]
        # frame = improve(frame)
        # frame= cv.GaussianBlur(frame, (5, 5), 3)
        self.increment_frame_num()
        self.wait_key()

        if not self.hand_captured and not self.calculated_hand_histogram:
            self.draw_boxes_frame(frame)
            cv.imshow(self.calculate_histogram_window, frame)

        if self.hand_captured and not self.calculated_hand_histogram:
            cv.destroyWindow(self.calculate_histogram_window)
            self.hand_histogram = calcHandHistogram(
                frame, self.capture_boxes, self.boxWidth, self.boxHeight)
            self.calculated_hand_histogram = True

        if self.hand_captured and self.calculated_hand_histogram:
            thres = handThreshold(frame, self.hand_histogram)
            self.process_frame(np.copy(thres), ori)

    def mode_2(self):
        self.increment_frame_num()
        _, frame = self.cap.read()
        frame = cv.flip(frame, 1)
        ori = np.copy(frame)
        ''' 
        region_right = h
        region_bottom = w 
        '''
        cv.rectangle(frame, (self.region_left, self.region_top),
                        (self.region_right, self.region_bottom), (0, 0, 255), 0)
        self.wait_key()

        roi = frame[self.region_top:self.region_bottom,
                    self.region_left:self.region_right]
        roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        # roi = cv.GaussianBlur(roi, (5, 5), 0)

        if self.background is None:
            self.background = roi.copy().astype("float")

        if self.frame_num < self.CALIBRATION_TIME:
            cv.accumulateWeighted(roi, self.background, self.BG_WEIGHT)
        else:  # after CALIBRATION_TIME
            region_pair = self.segment(roi, self.background)
            if region_pair is not None:
                (thresholded_region, _) = region_pair

                # kernel = np.ones((3, 3), np.uint8)
                # thresholded_region = cv.morphologyEx(
                #     thresholded_region, cv.MORPH_CLOSE, kernel, iterations=1)
                thresholded_region, _ = keep_biggest_blobs(
                    thresholded_region, 1, 4)
                cv.imshow("thresholded Image", thresholded_region)
                self.process_frame(thresholded_region, ori)
        cv.imshow("ROI", frame)
    def mode_3(self):
        pass
    def mutual_section(self):
        self.mode = 1
        if self.mode == 1:
            self.mode_1()
        elif self.mode == 2:
            self.mode_2()
        elif self.mode == 3:    
            self.mode_3()
    def draw_boxes_frame(self, frame):
        for (pos_x, pos_y) in self.capture_boxes:
            cv.rectangle(frame, (pos_x, pos_y), (pos_x +
                                                 self.boxWidth, pos_y + self.boxHeight), (0, 0, 255), 0)

    def increment_frame_num(self):
        self.frame_num += 1

    def wait_key(self):
        k = cv.waitKey(10)

        if k & 0xFF == ord('q'):
            self.quit = True

        if k & 0xFF == ord('h'):
            self.hand_captured = True

        if k & 0xFF == ord('r'):
            self.hand_captured = False
            self.calculated_hand_histogram = False

        static = ClosedHandEvent(None)
        if k & 0xFF == ord('1'):
            motion = LeftToRightLineMotionEvent()
            self.system_state.apply_event([static, motion])
        if k & 0xFF == ord('2'):
            motion = RightToLeftLineMotionEvent()
            self.system_state.apply_event([static, motion])
        if k & 0xFF == ord('3'):
            motion =  TopToBottomLineMotionEvent()
            self.system_state.apply_event([static, motion])
        if k & 0xFF == ord('4'):
            motion = BottomToTopLineMotionEvent()
            self.system_state.apply_event([static, motion])
        if k & 0xFF == ord('5'):
            motion = CounterClockwiseCircularMotionEvent()
            self.system_state.apply_event([static, motion])
        if k & 0xFF == ord('6'):
            motion = ClockwiseCircularMotionEvent()  
            self.system_state.apply_event([static, motion])
            
    def process_frame(self, frame, ori):

        current_static_event = self.hand_recognitzer_localizer.recognize_hand_event(
            frame)
        if type(current_static_event) is not NoEvent:
            current_frame_info = FrameInfo(self.frame_num)
            self.temporal_frame_info_list.add_frame(
                frame_info=current_frame_info)
            current_frame_info.point = self.hand_recognitzer_localizer.point


        self.static_event_approx.add_new_event(current_static_event)
        static_event = self.static_event_approx.get_event_approx()
        if str(static_event) != str(NoDecisionYetEvent()):
            self.recurrent_event = str(static_event)
        motion_event = NoEvent()


        if type(static_event) is NoEvent:
            self.static_event_approx.clear()
            self.temporal_frame_info_list.consume_frame(-1)
            self.temporal_frame_info_list.release_consumed_frames()
        else:
            pass
            # motion_event = self.get_motion_event()
            # if(str(motion_event) != "NoEvent"):
            #     print(motion_event, self.frame_num)
        

        top_left = self.hand_recognitzer_localizer.top_left
        bottom_right = self.hand_recognitzer_localizer.bottom_right
        if current_static_event is NoEvent:
            centre_point = ClosedHandEvent().center
        else:
            centre_point = self.hand_recognitzer_localizer.point


        prev = -1
        num = 0

        for pp in reversed(self.temporal_frame_info_list.frame_infos):
            if prev != -1:
                ps = (pp.point[1], pp.point[0])
                cv.line(frame, prev, ps, (0, 255, 9), 2)

            prev = (pp.point[1], pp.point[0])
            num += 1
            if num == 15:
                break
        frame = cv.cvtColor(frame, cv.COLOR_GRAY2RGB)



        self.draw_bb(top_left, bottom_right, frame)

        self.draw_text(frame, self.recurrent_event)
        self.draw_circle(frame, centre_point, 20)

        self.show_image(frame)

        # self.hand_recognitzer_localizer.possible_fingers_blobs

        if self.system_state is not None :
            self.system_state.apply_event([static_event, motion_event])
            


    def get_motion_event(self):
        self.current_step += 1
        if self.current_step == self.interval:
            self.current_step = 0
            return NoEvent()

        for motion_recognizer in self.motion_recognizers:
            event = motion_recognizer.recognize_motion()
            if str(event) != str(NoEvent()):
                break
        self.temporal_frame_info_list.release_consumed_frames()

        return event

    def draw_bb(self, top_left, bottom_right, frame):
        cv.rectangle(frame, (top_left[1], top_left[0]),
                     (bottom_right[1], bottom_right[0]), (255, 0, 0), 2)

    def draw_convex_hull(self, frame, convex_hull):
        if convex_hull is not None:
            cv.drawContours(frame, [convex_hull], 0, (0, 0, 255))

    def draw_circle(self, frame, point, radius=30):
        cv.circle(frame, (point[1], point[0]), radius, (0, 255, 0), 2)

    def draw_text(self, frame, text, point=(10, 500)):
        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 2
        fontColor = (0, 150, 255)
        thickness = 4
        lineType = 2

        cv.putText(frame, text,
                   point,
                   font,
                   fontScale,
                   fontColor,
                   thickness,
                   lineType)

    def show_image(self, frame, title="messi"):
        cv.imshow(title, frame)

    def segment(self, roi, background):
        diff = cv.absdiff(background.astype(np.uint8), roi)
        thresholdedROI = cv.threshold(
            diff, self.OBJ_THRESHOLD, 255, cv.THRESH_BINARY)[1]
        (contours, _) = cv.findContours(thresholdedROI.copy(),
                                        cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return
        else:
            segmented_region = max(contours, key=cv.contourArea)
            return (thresholdedROI, segmented_region)
