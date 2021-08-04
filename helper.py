import cv2
import numpy as np
from math import sqrt

"""
Script with all the helper functions necessary for gauge reading
"""

def calculate_brightness(image : np.array) -> float:
    """
    Checks whether the gauge has a black or a white background based on image brightness
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    pixels = sum(hist)
    brightness = scale = len(hist)

    for idx in range(scale):
        ratio = hist[idx] / pixels
        brightness += ratio * (-scale + idx)

    return 1.0 if brightness == 255 else float(brightness/scale)


def define_circle(p1 : tuple, p2 : tuple, p3 : tuple) -> tuple:
    """
    Returns the center and radius of the circle passing the given 3 points
    In case the 3 points form a line, returns (center, radius) = (None, infinity)
    """
    temp = p2[0] * p2[0] + p2[1] * p2[1]
    bc = (p1[0] * p1[0] + p1[1] * p1[1] - temp) / 2
    cd = (temp - p3[0] * p3[0] - p3[1] * p3[1]) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1.0e-6:
        return (None, np.inf)

    # Center of circle
    cx = (bc*(p2[1] - p3[1]) - cd*(p1[1] - p2[1])) / det
    cy = ((p1[0] - p2[0]) * cd - (p2[0] - p3[0]) * bc) / det

    radius = np.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)
    return ((int(cx), int(cy)), int(radius))


def point_inside_circle(center, radius, point) -> bool:
    """
    Checks whether a given point lies inside the circle or not
    """
    dx = abs(center[0] - point[0])
    dy = abs(center[1] - point[1])
    if dx**2 + dy**2 <= radius**2: 
        return True
    else:
        return False

def euclidean_dist(point1 : tuple, point2 : tuple) -> int:
    """
    Returns pixel euclidean distance between 2 points
    """
    return int(sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))

def warp_image(canny : np.array) -> np.array:
    """
    Computes an affine warp transformation matrix to transform an ellipse back to a circle
    """
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    sorted_areas = np.sort(areas)
    cnt = contours[areas.index(sorted_areas[-1])] ## Largest contour

    params = cv2.fitEllipse(cnt)
    angle = params[2]; scale = params[1]
    scale = scale[0] / scale[1]

    M = cv2.getRotationMatrix2D((canny.shape[0]/2, canny.shape[1]/2), angle, 1)
    M[:,0:2] = np.array([[1,0],[0,scale]]) @ M[:,0:2]
    M[1,2] = M[1,2] * scale
    return M
