import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import curve_fit

from helper import *
from needle import Needle
from ocr import Ocr
from region_props import Regionprops

class DirectionError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

@dataclass
class region:
    number : int
    tick_centroid : tuple
    bb_centroid : tuple

class Gauge_numeric(object):
    def __init__(self, mode : str) -> None:
        super().__init__()
        self.needle = Needle()
        self.ocr = Ocr()
        self.props = Regionprops()
        self.val = None
        self.params = [-0.1, 0.5, 1.2]
        self.ratio_thresh = 0.85
        self.area_thresh = 200
        self.mode = mode

    def reset(self) -> None:
        self.needle.reset()
        self.ocr.reset()
        self.props.reset()
        self.val = None
        self.mode = "curve"

    def _read_gauge(self, image : np.array, visualize : bool = True, needle = "white", fit : str = "vertical") -> None:
        start = time.time()
        ## Common parameters
        center = (image.shape[0]//2, image.shape[1]//2)
        self.norm_x = image.shape[0]; self.norm_y = image.shape[1]

        ## Run OCR and build a lookup dictionary
        self.ocr._run_ocr(image.copy())
        if len(self.ocr.lookup) < 2:
            print("Not enough values detected by OCR !!")
            print("Need at least 2 OCR values")
            return;

        ## Get the equation of the line estimating the needle and the needle tip
        self.needle._isolate_needle(image.copy(), color="white")
        self.needle._isolate_needle(image.copy(), color="red")
        if self.needle.line_white is None and self.needle.line_red is None:
            print("Needle not found !")
            return;
        line, pt1, pt2 = self._get_needle_line_and_pts(needle)
        tip = pt1 if euclidean_dist(pt1, center) > euclidean_dist(pt2, center) else pt2

        ## Compute regionprops and extract tick mark locations (Rectangular regions)
        dist = {}
        ticks, _ = self.props._get_tick_marks(image.copy(), self.area_thresh, self.ratio_thresh)
        pairs = self.props._pair_numbers_with_ticks(ticks, self.ocr.lookup, center)
        for number, (tick_centroid, bb_centroid) in pairs.items():
            dist[euclidean_dist(tip, bb_centroid)] = region(number, tick_centroid, bb_centroid)

        ## Calculate gauge value using relative positions of the needle tip, 2 nearest tick marks,
        ## and numbers associated with those ticks marks
        dist = dict(sorted(dist.items()))
        dist = list(dist.values())[0:2]
        if not dist:
            print("No tick marks identified near the needle tip")
            return;

        self.val, curve_x, curve_y = self._calculate_gauge_value(dist, tip, line, center, fit)
        if self.val:
            print("Gauge value = {:4.4f}".format(self.val))
        print("Time taken = {:4.6f}s".format(time.time()-start))
        self.reset()

        if visualize:
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            
            ## Plot the tick centroids along with the respective pair memebers joined by lines
            for number, (tick_centroid, bb_centroid) in pairs.items():
                plt.plot([tick_centroid[0], bb_centroid[0]], [tick_centroid[1], bb_centroid[1]], 'orange', linewidth=2)
            plt.plot([c[0] for c in ticks], [c[1] for c in ticks], 'b+', markersize=15)
           
            ## Plot the fitted curves (horizontal and vertical (if present))
            if curve_x is not None and fit == "vertical":
                x_plot = np.linspace(0,1,1000)
                try:
                    y_plot = np.clip(parabola(x_plot, *curve_x), 0, 1)
                except TypeError:
                    y_plot = np.clip(linear(x_plot, *curve_x), 0, 1)
                plt.plot(x_plot * self.norm_x, y_plot * self.norm_x, label='curve_x')
            if curve_y is not None and fit == "horizontal":
                y_plot = np.linspace(0,1,1000)
                x_plot = np.clip(parabola(y_plot, *curve_y), 0, 1)
                plt.plot(x_plot * self.norm_x, y_plot * self.norm_y, label='curve_y')

            ## Plot the line estimating the needle in the gauge
            x_plot = np.linspace(0,1,1000)
            y_line = np.clip(np.polyval(line, x_plot), 0, 1)
            plt.plot(x_plot * 800, y_line * 800, label='needle_line')
            plt.legend(loc='lower right')
            plt.show()

    @staticmethod
    def _get_needle_swing(center : tuple, dist : list) -> str:
        """
        Computes whether the swing of the needle is clockwise or anticlockwise based on
        the number locations on the calibration dial.
        If clockwise, the needle must go through quadrants in this order 3 -> 2 -> 1 -> 4
        Else if the order is 4 -> 1 -> 2 -> 3, we can infer the swing is anticlockwise
        """
        x1, y1 = dist[0].tick_centroid
        x2, y2 = dist[1].tick_centroid
        q1 = find_quadrant((int(x1), int(y1)), center) 
        q2 = find_quadrant((int(x2), int(y2)), center)

        ## These dictionaries tell us the possible quadrants of the higher number based on the 
        ## quadrant of the lower one for clockwise and anticlockwise swings for the needle
        quad_dict_clockwise = {3 : [2,1,4], 2 : [1,4], 1 : [4]}
        quad_dict_anticlockwise = {4 : [1,2,3], 1 : [2,3],  2 : [3]}

        if q1 == q2:
            if dist[0].number < dist[1].number:
                ## q1 needs to be on the left 
                if q1 == 1 or q1 == 2:
                    if x1 < x2 :
                        needle_swing = "clockwise"
                    else :
                        needle_swing = "anticlockwise"
                
                ## q1 needs to be to the right 
                elif q1 == 3 or q1 == 4 :
                    if x1 > x2 : 
                        needle_swing = "clockwise"
                    else:
                        needle_swing = "anticlockwise"

            else:
                ## q1 needs to be on the right 
                if q1 == 1 or q1 == 2:
                    if x1 > x2 :
                        needle_swing = "clockwise"
                    else :
                        needle_swing = "anticlockwise"
                
                ## q1 needs to be to the left 
                elif q1 == 3 or q1 == 4 :
                    if x1 < x2 : 
                        needle_swing = "clockwise"
                    else:
                        needle_swing = "anticlockwise"
        
        elif q1 != q2:
            if dist[0].number < dist[1].number:
                try:
                    if q2 in quad_dict_clockwise[q1]:
                        needle_swing = "clockwise"
                    else:
                        needle_swing = None
                
                except KeyError: 
                    if q2 in quad_dict_anticlockwise[q1]:
                        needle_swing = "anticlockwise"
                    else:
                        needle_swing = None
            
            else:
                try:
                    if q1 in quad_dict_clockwise[q2]:
                        needle_swing = "clockwise"
                    else:
                        needle_swing = None
                    
                except KeyError:
                    if q1 in quad_dict_anticlockwise[q2]:
                        needle_swing = "anticlockwise"
                    else:
                        needle_swing = None

        return needle_swing

    def _get_needle_line_and_pts(self, needle : str) -> tuple[np.array, tuple, tuple]:
        """
        Computes the 2 end points of the needle and the equation of the line joning these 
        2 points using 2-point formula. The end point further away from the center of the 
        image is the tip of the needle (point of interest)
        """
        if self.needle.line_white is not None and needle == "white":
            x_white = np.array([self.needle.line_white[0]/self.norm_x, self.needle.line_white[2]/self.norm_x]) 
            y_white = np.array([self.needle.line_white[1]/self.norm_y, self.needle.line_white[3]/self.norm_y]) 
            line = np.poly1d(fit_line(x_white, y_white))
            pt1 = (self.needle.line_white[0], self.needle.line_white[1])
            pt2 = (self.needle.line_white[2], self.needle.line_white[3])

        if self.needle.line_red is not None and needle == "red":
            x_red = np.array([self.needle.line_red[0]/self.norm_x, self.needle.line_red[2]/self.norm_x]) 
            y_red = np.array([self.needle.line_red[1]/self.norm_y, self.needle.line_red[3]/self.norm_y]) 
            line = np.poly1d(fit_line(x_red, y_red))
            pt1 = (self.needle.line_red[0], self.needle.line_red[1])
            pt2 = (self.needle.line_red[2], self.needle.line_red[3])

        return line, pt1, pt2     

    def _get_tip_position_wrt_ticks(self, pt_1 : tuple, pt_2 : tuple, pt_tip : tuple, pt_int : tuple,  map : dict, curve : np.poly1d, ecc : str) -> tuple[int, str, float]:
        """
        Computes the position of the tip of the needle with respect to the 2 closest tick marks
        This computed position is used for the next step to interpolate on the fitted curves 
        (horizontal or vertical depending on positions of numbers and tick marks)
        """
        x1, y1 = pt_1
        x2, y2 = pt_2
        xt, yt = pt_tip
        if ecc == "vertical":
            ## Determine whether the needle is inside the range, left or right
            if (xt - x1 > 0 and xt - x2 < 0) or (xt - x1 < 0 and xt - x2 > 0):
                ## Needle in between 2 numbers
                nearest_num = map[min(x1, x2)]
                direction = "right"
                if self.mode == "linear":
                    distance = euclidean_dist((x1,y1), pt_int)
                else:
                    distance = get_arc_length(min(x1, x2), xt, curve)

            else:
                ## The needle is outside the range of the 2 numbers
                if xt - x1 > 0 and xt - x2 > 0:
                    direction = "right"
                    nearest_num = map[max(x1, x2)]
                    if self.mode == "linear":
                        distance = euclidean_dist((x2,y2), pt_int)
                    else:
                        distance = get_arc_length(max(x1,x2), xt, curve)
                
                elif xt - x1 < 0 and xt - x2 < 0:
                    direction = "left"
                    nearest_num = map[min(x1, x2)]
                    if self.mode == "linear":
                        distance = euclidean_dist((x1,y1), pt_int)
                    else:
                        distance = get_arc_length(xt, min(x1,x2), curve)
                
                else:
                    raise DirectionError("Tip should be in range, to the left or right of the range.")
        
        elif ecc == "horizontal":

            ## Determine whether the needle is inside the range, top or bottom
            if (yt - y1 > 0 and yt - y2 < 0) or (yt - y1 < 0 and yt - y2 > 0):
                ## Needle in between 2 numbers
                nearest_num = map[min(y1, y2)]
                direction = "bottom"
                distance = get_arc_length(min(y1,y2), yt, curve)
            
            else:
                ## The needle is outside the range of the 2 numbers
                if yt - y1 > 0 and yt - y2 > 0:
                    print("sdfkjsd")
                    nearest_num = map[max(y1, y2)]
                    direction = "bottom"
                    distance = get_arc_length(max(y1,y2), yt, curve)

                elif yt - y1 < 0 and yt - y2 < 0:
                    direction = "top"
                    nearest_num = map[min(y1, y2)]
                    distance = get_arc_length(yt, min(y1, y2), curve)

        else :
            raise ValueError("Parabola has to be horizontal or vertical (h/v)")

        return nearest_num, direction, distance

    def _calculate_gauge_value(self, dist : list, tip : tuple, line : np.poly1d, center : tuple, fit : str) -> tuple[np.array, np.array]: 
        """
        Calculates the value read by the gauge using information about the tip of the needle,  
        2 closest tick marks and the values they represent. It calibrates the gauge based on 
        quadratic curve fitting, maps the calibration to the curve and interpolates along this curve
        to compute the exact value at the tip of the needle
        """
        x1, y1 = dist[0].tick_centroid
        x2, y2 = dist[1].tick_centroid
        xt, yt = tip
        needle_swing = Gauge_numeric._get_needle_swing(center, dist)

        x1 /= self.norm_x; y1 /= self.norm_y
        x2 /= self.norm_x; y2 /= self.norm_y
        xt /= self.norm_x; yt /= self.norm_y
        if needle_swing is None:
            print("Cannot determine needle swing !")
            return None, None, None
        
        calibration = abs(dist[1].number - dist[0].number)
        if self.mode == "linear":
            x_fit = [x1, x2]
            y_fit = [y1, y2]
            ref_line = np.poly1d(fit_line(np.array(x_fit), np.array(y_fit)))
            reference = euclidean_dist((x1,y1), (x2,y2))
            curve_x = ref_line
            curve_y = None

            ## Get point of intersection and calibration of the gauge
            pt_x = (ref_line - line).r
            pt_y = np.polyval(line, pt_x)
            pt = (pt_x, pt_y)
             
        else :
            x_fit = [x1 , xt, x2]
            y_fit = [y1 , yt , y2]
            curve_x, _ = curve_fit(parabola, x_fit, y_fit)
            curve_y, _ = curve_fit(parabola, y_fit, x_fit)
            reference = get_arc_length(min(x1, x2), max(x1, x2), curve_x)
            pt = None

        if fit == "vertical":
            map = {
                x1 : dist[0].number,
                x2 : dist[1].number
            }
            nearest_num, direction, distance = self._get_tip_position_wrt_ticks((x1, y1), (x2, y2), (xt, yt), pt, map, curve_x, fit)
        
        elif fit == "horizontal":
            map = {
                y1 : dist[0].number,
                y2 : dist[1].number
            }
            nearest_num, direction, distance = self._get_tip_position_wrt_ticks((x1, y1), (x2, y2), (xt, yt), pt, map, curve_y, fit)
        
        else : 
            print("Parabola has to be horizontal or vertical (h/v)")
            return None, None, None

        print("Range = ("+str(dist[0].number)+","+str(dist[1].number)+")")
        print("Distance to nearest major tick = {:2.2f}".format(distance))
        print("Direction wrt nearest major tick = ", direction)
        print("Nearest associated number = ", nearest_num)
        print("Calibration value for reference distance = ", calibration)
        print("Needle swing = ", needle_swing)

        delta_val = (calibration / reference) * distance
        if needle_swing == "clockwise":
            if direction == "left" or direction == "top":
                return nearest_num - delta_val, curve_x, curve_y
            elif direction == "right" or direction == "bottom":
                return nearest_num + delta_val, curve_x, curve_y

        else:
            if direction == "left" or direction == "top":
                return nearest_num + delta_val, curve_x, curve_y
            elif direction == "right" or direction == "bottom":
                return nearest_num - delta_val, curve_x, curve_y

def main(idx : int) -> None:
    gauge = Gauge_numeric("curve")
    
    if idx == 0: ## Works fine 
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4761.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red", "vertical")

    if idx == 1: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4762.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red", "vertical")

    if idx == 2: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4763.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red", "vertical")
    
    if idx == 3: ## Works fine --> Horizontal parabola
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4764.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red", "horizontal")
    
    if idx == 4: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4765.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red", "vertical")

    if idx == 5: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4766.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red", "vertical")

    if idx == 6: ## OCR mistake (120 getting misread as 20)
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4767.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.ocr.kblur = (3,3)
        gauge.ocr.sigblur = 3
        gauge._read_gauge(image, True, "red", "vertical")

    if idx == 7: ## Timeout
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4768.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white", "vertical")

    if idx == 8: ## Pairing mismatch due to region props error
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4769.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white", "vertical")

    if idx == 9: ## Timeout
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4770.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.props.kopen = (3,3)
        gauge.props.kclose = (3,3)
        gauge._read_gauge(image, True, "white", "vertical")

    if idx == 10: ## Works fine --> Horizontal parabola
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4771.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white", "horizontal")

    if idx == 11: ## Works fine --> Horizontal parabola
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4772.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white", "horizontal")

    if idx == 12: ## Works fine --> Horizontal parabola
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4773.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white", "horizontal")


if __name__ == "__main__":
    main(1)
