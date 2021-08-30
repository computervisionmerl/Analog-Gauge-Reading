import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

from helper import *
from region_props import *
from gauge_tick_marks import Gauge

class Gauge_minmax(Gauge):
    def __init__(self) -> None:
        super().__init__()

    def _read_gauge(self, image: np.array, visualize: bool, needle) -> None:
        start = time.time()
        ## Needle isolation
        center = (image.shape[0]//2, image.shape[1]//2)
        self.needle._isolate_needle(image.copy(), color="white")
        pt1 = (self.needle.line_white[0], self.needle.line_white[1])
        pt2 = (self.needle.line_white[2], self.needle.line_white[3])
        if euclidean_dist(pt1, center) > euclidean_dist(pt2, center):
            tip = pt1
        else :
            tip = pt2
        l = self.needle.line_white
        cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2)

        ## Get the normal region
        ## Normal region is usually defined between the 2 red semi circles
        blur = cv2.GaussianBlur(image, (5,5), 5)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(hsv, (170,50,20), (180,255,255))
        red = cv2.bitwise_or(mask1, mask2)
        labels = measure.label(red, connectivity=2)
        properties = measure.regionprops(labels, cache=False)

        roi_points = []
        dist_dict = {}
        for prop in properties:
            if prop.area > 300:
                roi_points.append((prop.centroid[1],prop.centroid[0])) 
        
        for pt in roi_points:
            dist_dict[euclidean_dist(tip, pt)] = pt

        ## dist[0] --> Centroid of closest region to tip
        ## dist[1] --> Centroid of second closest region
        dist = dict(sorted(dist_dict.items()))
        dist = list(dist.values())[0:2]
        x1, x2 = dist[0][0], dist[1][0]
        x_tip = tip[0]

        if (x_tip - x1 > 0 and x_tip - x2 < 0) or (x_tip - x1 < 0 and x_tip - x2 > 0):
            print("normal")
        else:
            if x_tip - x1 > 0 and x_tip - x2 > 0 : 
                print("Max / High")
            elif x_tip - x1 < 0 and x_tip - x2 < 0:
                print("Min / Low")
            else:
                print("Something went wrong, let's try again !")

        print("Time elapsed = {:4.6f}s".format(time.time() - start))
       
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        plt.plot([pt[0] for pt in roi_points], [pt[1] for pt in roi_points], 'r+')
        plt.show()
        
def main(idx : int):
    gauge = Gauge_minmax()

    try:
        if idx == 0:
            image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_minmax_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
            gauge._read_gauge(image, True, "white")

        elif idx == 1:
            image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_lowhigh_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
            gauge._read_gauge(image, True, "white")

        else:
            print("Check image file path")
    
    except ValueError as e:
        print(e)

    return;

if __name__ == "__main__":
    main(1)