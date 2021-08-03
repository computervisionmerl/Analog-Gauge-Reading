import cv2
import numpy as np
from math import atan2, sqrt, sin, cos, degrees

def euclidean_dist(point1 : tuple, point2 : tuple) -> int:
    """
    Returns pixel euclidean distance between 2 points
    """
    return int(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))

class Needle:
    def __init__(self) -> None:
        self.center = None
        self.needle = None

        ## Canny edge parameters
        self.canny_thresh_1 = 85 #150
        self.canny_thresh_2 = 255 #180

        ## Hough line parameters
        self.rho = 1
        self.theta = np.pi/180
        self.hough_thresh = 120
        self.srn = 2
        self.stn = 10

        ## Hough circle parameters
        self.dp = 1
        self.minDist = 20
        self.param1 = 20
        self.param2 = 12
        self.minRadius = 0
        self.maxRadius = 30

    @staticmethod
    def _extend_line_segment(center_line_x : int, needle_line : np.array) -> tuple:
        ## Returns the coordinates of the point of intersection ==> Center of the needle
        x1, y1, x2, y2 = needle_line

        if x1 == x2:
            raise Exception("Needle and center lines are parallel")

        if y1 == y2:
            return (center_line_x, y1)

        ## Based on line equation (y-y1)/(y2-y1) = (x-x1)/(x2-x1)
        y_line_for_x = ((y2 - y1) * (center_line_x - x1) / (x2 - x1)) + y1
        if y_line_for_x < y1 and y_line_for_x < y2:
            ## This means that the extrapolation has been done on the wrong side
            return None
        else:
            return (center_line_x, int(y_line_for_x))

    def _isolate_needle(self, image : np.array) -> None:
        """
        The longest hough line detected is assumed to be the needle 
        Needle is mostly the longest hough line if the gauge takes up the image completely
        """
        self.w, self.h = image.shape
        linesP = cv2.HoughLinesP(image.copy(), self.rho, self.theta, self.hough_thresh, None, self.srn, self.stn)
        max_length = 0
        if linesP is not None:
            for i in range(len(linesP)):
                l = linesP[i][0]
                length = euclidean_dist((l[0],l[1]), (l[2],l[3]))
                if length > max_length :
                    max_length = length
                    self.needle = l
        else:
            raise ValueError("No hough lines detected !!")

    
    def _find_center(self, image : np.array) -> None:
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, self.dp, self.minDist, None, self.param1, self.param2, self.minRadius, self.maxRadius)        
        min_dist = 1e20
        center = (image.shape[0]/2, image.shape[1]/2)
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                dist = sqrt((i[0]-center[0])**2 + (i[1]-center[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    self.center = (i[0], i[1])
        return;

    def _find_quadrant(self, point : tuple) -> int:
        ## The quadrant is found based on pixel coordinate system with center of needle as origin
        ## First quadrant --> x(+ve), y(-ve)
        if point[0] >= self.center[0] and point[1] < self.center[1]:
            return 1
        
        ## Second quadrant --> x(-ve), y(-ve)
        elif point[0] < self.center[0] and point[1] <= self.center[1]:
            return 2
        
        ## Third quadrant --> x(-ve), y(+ve)
        elif point[0] <= self.center[0] and point[1] > self.center[1]:
            return 3

        ## Fourth quadrant --> x(+ve), y(+ve)
        elif point[0] > self.center[0] and point[1] >= self.center[1]:
            return 4

        else:
            raise ValueError("Quadrant is not identified")

    def _find_angle_based_on_quadrant(self, quad : int, point : tuple) -> float:
        ## Gives the angle of the line joining the point to the center of the needle, with respect to the negative y-axis
        slope = degrees(atan2(abs(point[1] - self.center[1]), abs(point[0] - self.center[0])))
        if quad == 1:
            return 270 - slope
        elif quad == 2:
            return 90 + slope
        elif quad == 3:
            return 90 - slope
        elif quad == 4:
            return 270 + slope
        else:
            raise ValueError("Not a valid quadrant parameter")


    def _visualize(self, image : np.array) -> None:
        if self.needle is not None:
            cv2.line(image, (self.needle[0],self.needle[1]), (self.needle[2],self.needle[3]), (255,0,0), 2)

        if self.center is not None:
            cv2.circle(image, (self.center[0], self.center[1]), 5, (255,0,0), -1, cv2.LINE_AA)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        return;

def main() -> None:
    needle = Needle()
    
    image = cv2.resize(cv2.imread("gauge images/qualitrol_temp_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_oil_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_winding_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_transformer_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_negative_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    needle._isolate_needle(image)
    needle._visualize(image)
    return;

if __name__ == "__main__":
    main()