import numpy as np
import cv2
import math

def ellipse(x, y, a, b, c, d, e, f):
    """
    General equation to fit an ellipse through a set of 6 points or more
    More points ==> More precise ellipse
    """
    return a*x**2 + b*y**2 + c*x*y + d*x + e*y + f

def ellipse_solve(x, a, b, c, d, e, f):
    """
    Solves the ellipse equation for a given x value --> Each x value can have 
    2 corresponding y values (quadratic in x and y)
    """
    B = c*x + e
    A = b
    C = a*x**2 + d*x + f
    det  = math.sqrt(B**2 - 4*A*C)
    return [(-1.*B + det) / 2*A, (-1.*B - det) / 2*A]

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

def euclidean_dist(point1 : tuple, point2 : tuple) -> float:
    """
    Returns pixel euclidean distance between 2 points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

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
        raise ValueError("Quadrant is not identified (Should be 1, 2, 3, or 4)")

def find_angle_based_on_quad(quadrant : int, point : tuple, center : tuple) -> float:
    """
    Computes the angle between the line joining the center and point given, and the negative
    y-axis given the quadrant. Coordinate system is anticlockwise starting from top right as 
    the 1st quadrant.
    """
    slope = math.degrees(math.atan2(abs(point[1] - center[1]), abs(point[0] - center[0])))
    if quadrant == 1:
        return 270 - slope
    elif quadrant == 2:
        return 90 + slope
    elif quadrant == 3:
        return 90 - slope
    elif quadrant == 4:
        return 270 + slope
    else:
        raise ValueError("Incorrect quadrant !! (Should be 1, 2, 3, or 4)")