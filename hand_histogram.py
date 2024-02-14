import cv2
import numpy as np
import math


def improve(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    h = clahe.apply(h)
    s = clahe.apply(s)
    lab = cv2.merge([h, s, v])
    return cv2.cvtColor(lab, cv2.COLOR_HSV2BGR)


def calcHandHistogram(frame, capture_boxes, boxWidth, boxHeight):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ROI = cv2.vconcat([hsv[x1:x1+boxWidth, y1:y1+boxHeight]
                      for x1, y1 in capture_boxes])
    hand_hist = cv2.calcHist([ROI], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist


def getHandContour(thresh):
    (contours, _) = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    handContour = None
    if len(contours) > 0:
        handContour = max(contours, key=cv2.contourArea)
    return handContour


def disc(morph):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))


def handThreshold(frame, hand_hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    back_projection = cv2.calcBackProject(
        [hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)
    back_projection = cv2.filter2D(back_projection, -1, disc(7))
    thresh = cv2.threshold(back_projection, 20, 255, cv2.THRESH_BINARY)[1]
    return thresh
