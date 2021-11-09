import cv2
import numpy as np
from helper import *

class Needle:
    def __init__(self) -> None:
        self.needle = None

    def isolate_needle(self, edge_map : np.ndarray) -> None:
        """
        Isolates the needle in the entire image. Takes the edge map and computes the probabilistic 
        hough transform. The longest hough line is taken as the needle of the gauge
        """
        linesP = cv2.HoughLinesP(edge_map, 1, np.pi/180, 120, None, 2, 10)
        if linesP is not None:
            max_length = 1e-20
            for i in range(len(linesP)):
                l = linesP[i][0]
                length = euclidean_dist((l[0],l[1]), (l[2], l[3]))
                if length > max_length:
                    max_length = length
                    self.needle = l

    def visualize(self, image : np.ndarray) -> None:
        if self.needle is not None:
            cv2.line(image, (self.needle[0], self.needle[1]), (self.needle[2], self.needle[3]), (0,255,0), 2)
        
        cv2.imshow("image", image)
        cv2.waitKey(0)

def main():
    image = cv2.resize(cv2.imread("substation_images/ferguson_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/proflo_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/mitsubishi_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/negative_pressure_gauge.jpg"), (800,800), cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/trafag_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)

    needle = Needle()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.std() > 70 or gray.std() < 35:
        gray = cv2.equalizeHist(gray)
    
    blur = cv2.GaussianBlur(gray, (5,5), 5)
    if calculate_brightness(image) > 0.52:
        hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (24,24)))
    else:
        hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (24,24)))
    thresh = cv2.threshold(hat, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
    canny = cv2.Canny(thresh, 85, 255)
    cv2.imshow("asdsa", canny)

    needle.isolate_needle(canny)
    needle.visualize(image)

if __name__ == "__main__":
    main()