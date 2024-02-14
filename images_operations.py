
import numpy as np
import cv2
from math import radians

def getTranslatedImage(image, canvasWidth, canvasHeight, newX, newY):
    width, height = image.shape[0], image.shape[1]
    canvas = np.zeros(shape = (canvasWidth, canvasHeight, image.shape[2]), dtype = 'uint8')

    widthMargin = int(abs((canvasWidth - width) / 2.0))
    heightMargin = int(abs((canvasHeight - height) / 2.0))

    startX = max(0, widthMargin + newX)
    startY = max(0, heightMargin + newY)
    endX = min(canvas.shape[0], widthMargin + width + newX)
    endY = min(canvas.shape[1], heightMargin + height + newY)

    amountX = min(image.shape[0], endX - startX)
    amountY = min(image.shape[1], endY - startY)

    canvas[startX:endX, startY:endY] = image[0:amountX, 0:amountY]
    return canvas

def getScaledImage(image, scaleX, scaleY):
    width = int(image.shape[1] * scaleX)
    height = int(image.shape[0] * scaleY)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_CUBIC)

def norm(x,y):
    return np.linalg.norm(np.array([x,y]), 2)


def drawOnImage(image, x, y, px, py, thickness, color):
    result = image.copy()
    result = cv2.line(result, (px, py), (x, y), color, thickness, cv2.LINE_AA)
    return result


def getStacked(imageCanvas, drawingCanvas):
    return np.where(drawingCanvas != 0, drawingCanvas, imageCanvas)


def getRotatedImage(image, angle):
    radianAngle = radians(angle)
    width, height, channels = image.shape[0], image.shape[1], image.shape[2]

    point0 = np.array([-width / 2.0, height / 2.0])
    point1 = np.array([width / 2.0, height / 2.0])
    point2 = np.array([width / 2.0, -height / 2.0])
    point3 = np.array([-width / 2.0, -height / 2.0])

    stackedPoints = np.stack([point0, point1, point2, point3]).T

    rotationMatrix = np.array([
        [np.cos(radianAngle), -np.sin(radianAngle)],
        [np.sin(radianAngle), np.cos(radianAngle)]
    ])

    stackedPoints = rotationMatrix @ stackedPoints
    
    newWidth = int(np.max(stackedPoints[0,:]) * 2) + 1
    newHeight = int(np.max(stackedPoints[1,:]) * 2) + 1
    containerDim = max(newWidth, newHeight)

    widthMargin = (containerDim - width) // 2 + 1
    heightMargin = (containerDim - height) // 2 + 1

    container = np.zeros(shape=(containerDim, containerDim, channels), dtype='uint8')
    container[widthMargin:widthMargin+width, heightMargin:heightMargin+height] = image

    axisX, axisY = containerDim / 2.0, containerDim / 2.0
    rotationMatrix = cv2.getRotationMatrix2D((axisX, axisY), angle, 1.0)
    rotatedImage = cv2.warpAffine(container, rotationMatrix, (containerDim, containerDim))

    widthMargin = max((containerDim - newWidth) // 2, 0)
    heightMargin = max((containerDim - newHeight) // 2, 0)

    return rotatedImage[widthMargin:widthMargin+newWidth, heightMargin:heightMargin+newHeight]

