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
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)