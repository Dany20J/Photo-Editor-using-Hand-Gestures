
import cv2 as cv
import numpy as np
from line_motion_recognizer import LineMotionRecognizer
from frame_info import FrameInfo

from temporal_frame_info_list import TemporalFrameInfoList
from image_processor import ImageProcessor


ImageProcessor().camera_loop()
