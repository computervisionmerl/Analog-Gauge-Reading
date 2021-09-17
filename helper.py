import numpy as np
import cv2
from math import degrees, atan2

def euclidean_dist(point1 : tuple, point2 : tuple) -> float:
    """
    Returns pixel euclidean distance between 2 points
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def parabola (x, a, b, c):
    """
    Quadratic polynomial curve fitting
    """
    return a*x**2 + b*x + c

def linear(x, a, b):
    """
    Line fitting based on 2 point formula
    """
    return a*x + b 

def calculate_brightness(image : np.ndarray) -> float:
    """
    Checks whether the gauge has a black or white background depending on the image brightness
    This is under the assumption that the gauge occupies the majority of area in an image.
    """
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    hist = cv2.calcHist([image], [0], None, [256], [0,256])
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

def find_angle_based_on_quad(quadrant : int, point : tuple, center : tuple) -> float:
    """
    Computes the angle between the line joining the center to the point given the quadrant
    Coordinate system is anticlockwise starting from top right as 1st quadrant.
    """
    slope = degrees(atan2(abs(point[1] - center[1]), abs(point[0] - center[0])))
    if quadrant == 1:
        return 270 - slope
    elif quadrant == 2:
        return 90 + slope
    elif quadrant == 3:
        return 90 - slope
    elif quadrant == 4:
        return 270 + slope
    else:
        raise ValueError("Incorrect quadrant !!")

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
        prev_y = parabola(prev_x, *bestmodel)
        curr_x = curr_x + dx
        curr_y = parabola(curr_x, *bestmodel)

        x_start = prev_x; y_start = prev_y
        x_finish = curr_x; y_finish = curr_y

        length += euclidean_dist((x_start, y_start), (x_finish, y_finish))
    
    return length 