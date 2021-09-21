import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import curve_fit

from rectangle_detector import Rectangle_detector
from ocr import Ocr
from needle import Needle
from helper import *

@dataclass
class tick_distance_object:
    number : int
    tick_centroid : tuple
    number_centroid : tuple

class Gauge_numeric(object):
    def __init__(self) -> None:
        super().__init__()
        self.ocr = Ocr()
        self.props = Rectangle_detector()
        self.needle = Needle()
        self.pairs = {}
        self.reset()

    def reset(self) -> None:
        self.val = None
        self.pairs.clear()
        self.ocr.reset()
        self.props.reset()

    def __get_equation_and_pts(self, needle : str) -> tuple[np.poly1d, tuple, tuple]:
        """
        Computes the 2 end points of the needle and the equation of the line joning these 
        2 points using 2-point formula. The end point further away from the center of the 
        image is the tip of the needle (point of interest)
        """
        if self.needle.line_white is not None and needle == "white":
            x_white = np.array([self.needle.line_white[0]/self.norm_x, self.needle.line_white[2]/self.norm_x]) 
            y_white = np.array([self.needle.line_white[1]/self.norm_y, self.needle.line_white[3]/self.norm_y]) 
            line, _ = curve_fit(linear, x_white, y_white)
            pt1 = (self.needle.line_white[0], self.needle.line_white[1])
            pt2 = (self.needle.line_white[2], self.needle.line_white[3])

        if self.needle.line_red is not None and needle == "red":
            x_red = np.array([self.needle.line_red[0]/self.norm_x, self.needle.line_red[2]/self.norm_x]) 
            y_red = np.array([self.needle.line_red[1]/self.norm_y, self.needle.line_red[3]/self.norm_y]) 
            line, _ = curve_fit(linear, x_red, y_red)
            pt1 = (self.needle.line_red[0], self.needle.line_red[1])
            pt2 = (self.needle.line_red[2], self.needle.line_red[3])

        return np.poly1d(line),pt1,pt2

    def __compute_needle_pivot(self, pt1 : tuple, pt2 : tuple) -> tuple:
        """
        Computes the center of the needle which is considered to be the gauge center 
        in all further computations. The Hough circle which satisfies both conditions
        of close to the image center and collinear with the needle is considered to 
        be the pivot of the needle
        """
        p1, p2 = np.array(pt1), np.array(pt2)
        min_dist = 1e20
        min_dist_from_center = 1e20
        try:
            for i in self.needle.circles[0,:]:
                ## Center of the circle
                p3 = np.array([i[0], i[1]])
                dist = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
                if dist < min_dist:
                    dist_from_center = euclidean_dist((i[0], i[1]), (400,400))
                    if dist_from_center < min_dist_from_center:
                        center = (i[0], i[1]) 
                        min_dist = dist
                        min_dist_from_center = dist_from_center
        except TypeError:
            raise("No hough circles found for needle center estimation")

        return center

    def __get_polynomial_direction(self, p1 : tuple, p2 : tuple, p3 : tuple) -> str:
        """
        Based on the polynomial direction the equation of the curve fitting changes
        Vertical --> y = ax^2 + bx + c
        Horizontal --> x = ay^2 + by + c
        """
        delta_x = max(abs(p1[0]-p2[0]), abs(p1[0]-p3[0]), abs(p2[0]-p3[0]))
        delta_y = max(abs(p1[1]-p2[1]), abs(p1[1]-p3[1]), abs(p2[1]-p3[1]))
        if delta_x >= delta_y:
            return "vertical"
        else: 
            return "horizontal"

    def __interpolate_to_tip(self, dist : list, curve : np.array, point : tuple, center : tuple ,fit : str) -> tuple[int, str, float]:
        """
        Interpolates along the computed curve to find the exact position of the needle 
        with respect to the 2 closest tick marks. In most cases these are tick marks on
        either side of the needle so interpolation is accurate
        """
        ## All points are normalized
        x1, y1 = dist[0][1].tick_centroid
        xt, yt = point
        center = (center[0] / self.norm_x, center[1] / self.norm_y)

        nearest_num = dist[0][1].number
        if fit == "vertical":
            ## Top half of the image
            if yt < center[1] and y1 < center[1]:
                if xt > x1 :
                    direction = "right"
                else:
                    direction = "left"
            ## Bottom half of the image
            else:
                if xt > x1:
                    direction = "left"
                else:
                    direction = "right"
            distance = get_arc_length(min(x1, xt), max(x1, xt), curve)
        
        elif fit == "horizontal":
            ## Left half of the image
            if xt < center[0] and x1 < center[0]:
                if yt < y1: 
                    direction = "right"
                else:
                    direction = "left"
            
            ## Right half of the image
            else:
                if yt < y1:
                    direction = "left"
                else:
                    direction = "right"

            distance = get_arc_length(min(yt,y1), max(yt,y1), curve)

        else :
            raise ValueError("Cannot interpolate from nearest tick mark")

        return nearest_num, direction, distance   

    def __calculate_gauge_value(self, tip : tuple, line : np.poly1d, center : tuple, swing : str) -> tuple[float, np.ndarray, str]:
        """
        Runs the actual gauge value computation using the calibration information
        from the tick marks, the computed curve (horizontal or vertical parabola),
        and position of the needle tip with respect to these calibrated tick marks
        """
        if swing is None:
            print("Not able to determine needle swing")
            return None, None, None

        distance_dict = {}
        ## Construct the dictionary to identify closest 3 numbers to the needle tip
        tip = (tip[0] / self.norm_x, tip[1] / self.norm_y)
        for number, (tick_centroid, numb_centroid) in self.pairs.items():
            ## Normalize the coordinates before feeding into the distance dictionary (standardization)
            tick_centroid = (tick_centroid[0] / self.norm_x, tick_centroid[1] / self.norm_y)
            numb_centroid = (numb_centroid[0] / self.norm_x, numb_centroid[1] / self.norm_y)
            dist = euclidean_dist(tip, numb_centroid)
            distance_dict[dist] = tick_distance_object(number, tick_centroid, numb_centroid)
        distance_dict = list(sorted(distance_dict.items()))

        ## 3 points to fit the curve
        try:
            x1, y1 = distance_dict[0][1].tick_centroid
            x2, y2 = distance_dict[1][1].tick_centroid
        except IndexError:
            print("Not even 2 pairs found !!")
            print("Exitting code")
            return None, None, None

        try: 
            if distance_dict[2][0] * self.norm_x < 250:
                x3, y3 = distance_dict[2][1].tick_centroid
            else:
                print("WARNING :- Only 2 ticks identified, gauge reading can be erroneous")
                x3, y3 = tip
        except IndexError:
            print("WARNING :- Only 2 ticks identified, gauge reading can be erroneous")
            x3, y3 = tip

        calibration = abs(distance_dict[0][1].number - distance_dict[1][1].number)
        fit = self.__get_polynomial_direction((x1,y1),(x2,y2),(x3,y3))
        x_fit, y_fit = [x1,x2,x3], [y1,y2,y3]
        curve = None
        
        if fit == "vertical":
            curve, _ = curve_fit(parabola, x_fit, y_fit)
            reference = get_arc_length(min(x1, x2), max(x1, x2), curve)
            try:
                if distance_dict[2][0] * self.norm_x < 250:
                    ## Find the point of intersection
                    roots = (np.poly1d(curve) - line).r
                    for root in roots:
                        if root >= 0 and root <= 1:
                            x_intersection = root
                            y_intersection = parabola(x_intersection, *curve)
                            break
                    intersection = (x_intersection, y_intersection)
                else:
                    intersection = tip
            except IndexError:
                intersection = tip

        elif fit == "horizontal":
            curve, _ = curve_fit(parabola, y_fit, x_fit)
            reference = get_arc_length(min(y1, y2), max(y1, y2), curve)
            try:
                if distance_dict[2][0] * self.norm_x < 250:
                    roots = (np.poly1d(curve) - line).r
                    for root in roots:
                        if root >= 0 and root <= 1:
                            y_intersection = root
                            x_intersection = parabola(y_intersection, *curve)
                            break
                    intersection = (x_intersection, y_intersection)
                else:
                    intersection = tip
            except IndexError:
                intersection = tip
        else:
            print("Parabola has to be vertical or horizontal (h/v)")
            return None, None, None

        nearest_num, direction, distance = self.__interpolate_to_tip(distance_dict, curve, intersection, center, fit)
        print("Polynomial fit = ", fit)
        print("Nearest number = {}".format(nearest_num))
        print("Direction from nearest number = ",direction)
        print("Needle swing = ", swing)
        print("Calibration value = {}".format(calibration))

        delta_val = (calibration / reference) * distance
        if swing == "clockwise":
            if direction == "left" :
                return nearest_num - delta_val, curve, fit
            elif direction == "right" :
                return nearest_num + delta_val, curve, fit

        else:
            if direction == "left" :
                return nearest_num + delta_val, curve, fit
            elif direction == "right" :
                return nearest_num - delta_val, curve, fit

    def read_gauge(self, image : np.ndarray, visualize : bool = True, needle : str = "white") -> None:
        """
        A wrapper method that runs all the required methods to read a gauge and
        displays the value. Runs the OCR, needle and gauge center estimation, 
        regionprops (for tick mark extraction), number-tick pairing, and all the
        commands needed to visualize the results (if needed)
        """
        ## Common parameters
        start = time.time()
        self.norm_x, self.norm_y = image.shape[0], image.shape[1]
        curve = None

        ## OCR 
        self.ocr.run_ocr(image.copy())
        if len(self.ocr.lookup) < 2:
            plt.figure(figsize=(12,12))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
            print("At least 2 values needed")
            print("Exitting code")
            return;

        ## Needle and gauge center estimation
        self.needle.isolate_needle(image.copy(), needle)
        if self.needle.line_white is None and self.needle.line_red is None:
            print("No needle found !")
            return;
        line, pt1, pt2 = self.__get_equation_and_pts(needle)
        center = self.__compute_needle_pivot(pt1, pt2)
        tip = pt1 if euclidean_dist(pt1, center) > euclidean_dist(pt2, center) else pt2
        swing = self.ocr.filter_numbers_based_on_position(center)

        ## Regionprops and gauge value calculation
        ticks = self.props.detect_rect(image, False)
        self.pairs = self.props.pair_ticks_with_numbers(ticks, self.ocr.lookup.copy(), center)
        self.val, curve, fit = self.__calculate_gauge_value(tip, line, center, swing)
        if self.val is not None:
            print("Gauge Value = {:4.4f}".format(self.val))
        print("Time taken = {:4.4f}s".format(time.time() - start))  

        if visualize:
            # OCR bounding boxes
            for text, obj in self.ocr.lookup.items():
                [tl, tr, br, bl] = obj.box
                try:
                    cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)
                    cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                except cv2.error:
                    continue

            # Display the image
            plt.figure(figsize=(12,12))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Ticks, needle pivot, and matches (number-tick)
            plt.plot(tip[0], tip[1], 'go', markersize=15)
            plt.plot(center[0], center[1], 'g*', markersize=15)
            plt.plot([i.centroid[0] for i in ticks], [i.centroid[1] for i in ticks], 'b+', markersize=15)
            for _, (tick, bb) in self.pairs.items():
                plt.plot([tick[0], bb[0]],[tick[1], bb[1]], 'orange', linewidth=2)

            # Needle line (from the estimated equation)
            x_plot = np.linspace(0,1,1000)
            y_line = np.clip(np.polyval(line, x_plot),0,1)
            plt.plot(x_plot * 800, y_line * 800, 'yellow')

            # Plot the fit polynomial
            if curve is not None:
                if fit == "vertical":
                    x_plot = np.linspace(0,1,1000)
                    y_plot = np.clip(parabola(x_plot, *curve),0,1)
                    plt.plot(x_plot * 800, y_plot * 800, 'purple')
                elif fit == "horizontal":
                    y_plot = np.linspace(0,1,1000)
                    x_plot = np.clip(parabola(y_plot, *curve),0,1)
                    plt.plot(x_plot * 800, y_plot * 800, 'purple')
            
            plt.show()

        self.reset()
        return;

def main(idx : int) -> None:
    gauge = Gauge_numeric()
    visualize = True
    
    if idx == 0: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4761.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    if idx == 1: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4762.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 2: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4763.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")
    
    elif idx == 3: ## Works fine 
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4764.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")
    
    elif idx == 4: ## Works fine 
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4765.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 5: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4766.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 6: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4767.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 7: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4807.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 8: ## OCR too few values
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4808.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 9: ## Needle swing issue 
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4809.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 10:  ## Works Fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4804.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 11: ## Works Fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4805.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 12: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4806.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 13: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4801.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")
    
    elif idx == 14: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4802.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 15: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4803.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")
    
    else :
        print("Enter a valid idx value")
    
    return;

if __name__ == "__main__":
    main(0)