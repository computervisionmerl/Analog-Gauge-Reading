import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt

from needle import Needle
from ocr import Ocr, box_object
from helper import *
from rectangle_detector import Rectangle_detector

class InterpolationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class Ocr_non_numeric(Ocr):
    def __init__(self) -> None:
        super().__init__()
        super().reset()

    def __construct_lookup(self, image : np.ndarray) -> None:
        """
        Constructs the lookup dictionary as it is. There is no filtering done inside
        this method. Dictionary value correction is all done after it is classified 
        into a particular type of non-numeric gauge
        """
        boxes = self._reader.readtext(image.copy())
        if boxes:
            for box, text, conf in boxes:
                (tl, tr, br, bl) = box
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                bl = (int(bl[0]), int(bl[1]))
                br = (int(br[0]), int(br[1]))

                centroid = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
                try:
                    if text in self.lookup and conf > self.lookup[text].conf:
                        self.lookup[text] = box_object([tl,tr,br,bl], centroid, conf)
                    elif text not in self.lookup:
                        self.lookup[text] = box_object([tl,tr,br,bl], centroid, conf)
                    else :
                        continue
                    
                except KeyError:
                    continue
                    
    def run_ocr(self, image: np.ndarray) -> None:
        """
        Preprocessing (if needed) + construct lookup dictionary
        """
        if len(image.shape) > 2:
            image = super()._preprocessing(image.copy())
        
        self.__construct_lookup(image.copy())
        return;

class Gauge_non_numeric:
    def __init__(self) -> None:
        self.needle = Needle()
        self.ocr = Ocr_non_numeric()
        self.props = Rectangle_detector()
        self.pairs = {}
        self.reset()
    
    def reset(self) -> None:
        self.needle.reset()
        self.ocr.reset()
        self.props.reset()
        self.val = None
        self.gauge_type = None
        self.pairs.clear()

    def __extract_red_ticks(self, image : np.ndarray) -> np.ndarray:
        """
        Masks out everything other than red color regions --> HSV + 
        color range definition + masking
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,50,20), (3,255,255)) #cv2.inRange(hsv, (0,50,20), (5,255,255))
        mask2 = cv2.inRange(hsv, (175,50,20), (180,255,255)) #cv2.inRange(hsv, (175,50,20), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
        masked = cv2.cvtColor(cv2.bitwise_and(image, image, mask=mask), cv2.COLOR_RGB2GRAY)
        masked = cv2.erode(masked, np.ones((5, 5), dtype="uint8"))
        return cv2.threshold(masked, 10, 255, cv2.THRESH_BINARY)[1]
    
    def __extract_white_ticks(self, image : np.ndarray) -> np.ndarray:
        """
        Image denoising + Binary thresholding
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if image.std() > 70 or image.std() < 35:
            image = cv2.equalizeHist(image)

        image[image<200] = 0
        return cv2.threshold(image, 10, 255, cv2.THRESH_BINARY)[1]

    def __extract_yellow_ticks(self, image : np.ndarray) -> np.ndarray:
        """
        Masks out everything other than yellow color regions --> HSV + 
        color range definition + masking
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([22,93,0],dtype="uint8"), np.array([45,255,255], dtype="uint8"))
        masked = cv2.cvtColor(cv2.bitwise_and(image, image, mask=mask), cv2.COLOR_RGB2GRAY)
        erode = cv2.erode(masked, np.ones((5,5), dtype="uint8"))
        return cv2.threshold(erode, 10, 255, cv2.THRESH_BINARY)[1]

    def __get_equation_and_pts(self, needle : str) -> tuple[np.poly1d, tuple, tuple]:
        """
        Computes the 2 end points of the needle and the equation of the line joning these 
        2 points using 2-point formula. The end point further away from the center of the 
        image is the tip of the needle (point of interest)
        """
        if self.needle.line_white is not None and needle == "white":
            x_white = np.array([self.needle.line_white[0]/self.norm_x, self.needle.line_white[2]/self.norm_x]) 
            y_white = np.array([self.needle.line_white[1]/self.norm_y, self.needle.line_white[3]/self.norm_y]) 
            line = np.poly1d(np.polyfit(x_white, y_white,1))
            pt1 = (self.needle.line_white[0], self.needle.line_white[1])
            pt2 = (self.needle.line_white[2], self.needle.line_white[3])

        if self.needle.line_red is not None and needle == "red":
            x_red = np.array([self.needle.line_red[0]/self.norm_x, self.needle.line_red[2]/self.norm_x]) 
            y_red = np.array([self.needle.line_red[1]/self.norm_y, self.needle.line_red[3]/self.norm_y]) 
            line = np.poly1d(np.polyfit(x_red, y_red,1))
            pt1 = (self.needle.line_red[0], self.needle.line_red[1])
            pt2 = (self.needle.line_red[2], self.needle.line_red[3])

        return line, pt1, pt2

    def __compute_needle_pivot(self, pt1 : tuple, pt2 : tuple) -> tuple:
        """
        The pivot of the needle can be assumed that point which is a center of one of
        the estimated hough circles, closest to the line and the geometric center of 
        the image. Minimizing these 2 based on the centers of circles, we can compute 
        pivot point and in most cases that is the center of the gauge too
        """
        p1, p2 = np.array(pt1), np.array(pt2)
        min_dist = 1e20
        min_dist_from_center = 1e20
        try:
            for i in self.needle.circles[0,:]:
                ## Center of the circle
                tip = np.array([i[0], i[1]])
                dist = np.linalg.norm(np.cross(p2-p1, p1-tip))/np.linalg.norm(p2-p1)
                if dist < min_dist:
                    dist_from_center = euclidean_dist((i[0], i[1]), (self.norm_x//2,self.norm_y//2))
                    if dist_from_center < min_dist_from_center:
                        center = (i[0], i[1]) 
                        min_dist = dist
                        min_dist_from_center = dist_from_center
        except:
            raise TypeError("No hough circles found for center estimation")

        return center

    def __get_fit_direction(self, p1 : tuple,p2 : tuple, p3 : tuple) -> str:
        delta_x = max(abs(p1[0]-p2[0]), abs(p1[0]-p3[0]), abs(p2[0]-p3[0]))
        delta_y = max(abs(p1[1]-p2[1]), abs(p1[1]-p3[1]), abs(p2[1]-p3[1]))
        if delta_x >= delta_y:
            return "vertical"
        else: 
            return "horizontal"

    def read_gauge(self, image : np.ndarray, color : str = "white", visualize : bool = True) -> None:
        """
        Wrapper method that calls all the methods which give information required 
        to read the gauge. Gives an option of visualizing results if needed
        """
        ## General parameters
        start = time()
        self.norm_x, self.norm_y = image.shape[0], image.shape[1]
        
        ## OCR + Classify + Needle
        self.ocr.run_ocr(image.copy())
        self.needle.isolate_needle(image.copy(), color)
        _, pt1, pt2 = self.__get_equation_and_pts(color)
        center = self.__compute_needle_pivot(pt1, pt2)
        tip = pt1 if euclidean_dist(pt1, center) >= euclidean_dist(pt2, center) else pt2
        self.ocr.lookup, gauge_type = classify_gauge_and_clean_lookup(self.ocr.lookup.copy())

        if gauge_type == "MIN-MAX" or gauge_type == "LOW-HIGH":
            thresh = self.__extract_red_ticks(image.copy())
        elif gauge_type == "HI-LO":
            thresh = self.__extract_white_ticks(image.copy())
        elif gauge_type == "PUMP-ON-OFF":
            thresh = self.__extract_yellow_ticks(image.copy())
        else:
            raise ValueError("Gauge type is invalid !!")

        ## Regionprops, detect ticks and pair with text
        ticks = self.props.detect_rect(thresh, False)
        ticks = [tick.centroid for tick in ticks]
        self.pairs = self.props.pair_ticks_with_text(self.ocr.lookup, ticks, center)
        if len(self.pairs) != 2:
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.plot([tick[0] for tick in ticks], [tick[1] for tick in ticks], 'b+', markersize=10)
            for tc, nc in self.pairs.values():
                plt.plot([tc[0], nc[0]], [tc[1], nc[1]], 'r', linewidth=2)
            plt.show()
            self.ocr.visualize(image.copy())
            print("Need exactly 2 numbers / points of reference to interpolate tip")
            return;

        ## Interpolate to the tip
        centroids = [tc[0] for tc in self.pairs.values()] 
        fit = self.__get_fit_direction(centroids[0], centroids[1], tip)
        if fit == "vertical":
            d1 = tip[0] - centroids[0][0]
            d2 = tip[0] - centroids[1][0]

            if np.sign(d1) != np.sign(d2):
                print("Normal")
            elif d1 < 0 and d2 < 0:
                print("Too low")
            elif d1 > 0 and d2 > 0:
                print("Too high")
            else:
                raise InterpolationError("Tick interpolation error !!")

        elif fit == "horizontal":
            d1 = tip[1] - centroids[0][1]
            d2 = tip[1] - centroids[1][1]

            if np.sign(d1) != np.sign(d2):
                print("Normal")
            elif d1 < 0 and d2 < 0:
                print("Too high")
            elif d1 > 0 and d2 > 0:
                print("Too low")
            else:
                raise InterpolationError("Tick interpolation error !!")
        print("Time taken = {:4.4f}s".format(time() - start))

        if visualize:
            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.plot([tick[0] for tick in ticks], [tick[1] for tick in ticks], 'b+', markersize=10)
            for tc, nc in self.pairs.values():
                plt.plot([tc[0], nc[0]], [tc[1], nc[1]], 'r', linewidth=2)
            plt.plot(center[0], center[1], 'g*', markersize=10)
            plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]],'orange',linewidth=3)
            plt.plot(tip[0], tip[1], 'go', markersize=10)
            plt.show()

def main(idx : int):
    if idx == 0:
        image = cv2.resize(cv2.imread("level_gauges/hilo_normal.png"),(800,800),cv2.INTER_CUBIC)
    if idx == 1:
        image = cv2.resize(cv2.imread("level_gauges/spot_ptz_hilo_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    if idx == 2:
        image = cv2.resize(cv2.imread("level_gauges/IMG-4811.jpg"),(800,800),cv2.INTER_CUBIC)    
    if idx == 3:
        image = cv2.resize(cv2.imread("level_gauges/IMG-4812.jpg"),(800,800),cv2.INTER_CUBIC)   

    if idx == 4:
        image = cv2.resize(cv2.imread("level_gauges/spot_ptz_lowhigh_gauge.jpg"),(800,800),cv2.INTER_CUBIC) 
    if idx == 5:
        image = cv2.resize(cv2.imread("level_gauges/IMG-4815.jpg"),(800,800),cv2.INTER_CUBIC)    
    if idx == 6:
        image = cv2.resize(cv2.imread("level_gauges/IMG-4816.jpg"),(800,800),cv2.INTER_CUBIC)    
    if idx == 7:
        image = cv2.resize(cv2.imread("level_gauges/IMG-4817.jpg"),(800,800),cv2.INTER_CUBIC)
        
    if idx == 8:
        image = cv2.resize(cv2.imread("level_gauges/minmax_low.jpg"),(800,800),cv2.INTER_CUBIC)
    if idx == 9:
        image = cv2.resize(cv2.imread("level_gauges/minmax_high.jpg"),(800,800),cv2.INTER_CUBIC)
    if idx == 10:
        image = cv2.resize(cv2.imread("level_gauges/spot_ptz_minmax_gauge.jpg"),(800,800),cv2.INTER_CUBIC) 
    
    gauge = Gauge_non_numeric()
    gauge.read_gauge(image)

if __name__ == "__main__":
    main(4)