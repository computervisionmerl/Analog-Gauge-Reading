import numpy as np
import cv2
from math import sqrt

"""
This script defines helper functions for the gauge reading to work
"""

def euclidean_dist(point1 : tuple, point2 : tuple) -> int:
    """
    Returns pixel euclidean distance between 2 points
    """
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_brightness(image : np.array) -> float:
    """
    Checks whether the gauge has a black or white background depending on the image brightness
    This is under the assumption that the gauge occupies the majority of area in an image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    pixels = sum(hist)
    brightness = scale = len(hist)

    for idx in range(scale):
        ratio = hist[idx] / pixels
        brightness += ratio * (-scale + idx)

    return 1.0 if brightness == 255 else float(brightness/scale)

def func(x, a, b, c):
    """
    Objective function for least squares minimization
    """
    return a*x**2 + b*x + c    

def residual(p, x, y):
    """
    Residual function for least squares optimization
    """
    return y - func(x, *p)

def find_quadrant(point : tuple, center : tuple) -> int:
    """
    Returns the quadrant of a point based on the the coordinates of the point and center 
    Coordinate system is anticlockwise starting from top right as 1st quadrant
    """
    ## First quadrant --> x(+ve), y(-ve)
    if point[0] >= center[0] and point[1] < center[1]:
        return 1
    
    ## Second quadrant --> x(-ve), y(-ve)
    elif point[0] < center[0] and point[1] <= center[1]:
        return 2
    
    ## Third quadrant --> x(-ve), y(+ve)
    elif point[0] <= center[0] and point[1] > center[1]:
        return 3

    ## Fourth quadrant --> x(+ve), y(+ve)
    elif point[0] > center[0] and point[1] >= center[1]:
        return 4

    else:
        raise ValueError("Quadrant is not identified")

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
