import numpy as np
import cv2
from random import randint

"""
This script defines helper functions for the gauge reading to work
"""

def euclidean_dist(point1 : tuple, point2 : tuple) -> float:
    """
    Returns pixel euclidean distance between 2 points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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

def get_arc_length(x1 : float, x2 : float, bestmodel : np.poly1d, n_steps : int = 1000) -> float:
    """
    Calculates the arclength along a polynomial curve using piecewise linear estimation
    of the curve (Usually done using line integrals but can be approximated to piecewise
    linear estimate)
    """
    length = 0.0
    dx = (x2 - x1) / n_steps
    curr_x = x1

    while curr_x <= x2:
        prev_x = curr_x
        prev_y = np.polyval(bestmodel, prev_x)
        curr_x = curr_x + dx
        curr_y = np.polyval(bestmodel, curr_x)

        x_start = prev_x; y_start = prev_y
        x_finish = curr_x; y_finish = curr_y

        length += euclidean_dist((x_start, y_start), (x_finish, y_finish))
    
    return length 

def fit_curve(x : np.array, y : np.array) -> np.ndarray:    
    """
    Fits a quadratic curve through a set of data points using
    least squares fitting
    """
    iterations = 0
    bestmodel = None
    besterr = 1e20
    while iterations < 50:
        x_fit = []; y_fit = []
        for _ in range(50): # 50
            random_idx = randint(0,len(x)-1)
            x_fit.append(x[random_idx]); y_fit.append(y[random_idx])
        
        poly = np.polyfit(x_fit, y_fit, 2)
        z_fit = np.polyval(poly, x_fit)
        err = np.sum((z_fit - y_fit)**2)
        if err < besterr:
            besterr = err
            bestmodel = poly

        iterations += 1
    
    return bestmodel

def fit_line(x : np.array, y : np.array) -> np.ndarray:
    """
    Line fitting using 2-point formula
    """
    return np.polyfit(x, y, 1)

def warp_image(image : np.array) -> np.array:
    """
    Computes an affine warp transformation matrix to transform an ellipse back to a circle
    """
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    sorted_areas = np.sort(areas)
    cnt = contours[areas.index(sorted_areas[-1])] ## Largest contour

    params = cv2.fitEllipse(cnt)
    angle = params[2]; scale = params[1]
    scale = scale[0] / scale[1]

    M = cv2.getRotationMatrix2D((image.shape[0]/2, image.shape[1]/2), angle, 1)
    M[:,0:2] = np.array([[1,0],[0,scale]]) @ M[:,0:2]
    M[1,2] = M[1,2] * scale
    return M
