from time import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import curve_fit

from ocr import Ocr
from rectangle_detector import Rectangle_detector
from needle import Needle
from helper import *

class InterpolationError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

@dataclass
class tick_distance_object:
    number : int
    tick_centroid : tuple
    numb_centroid : tuple

class Gauge_numeric(object):
    def __init__(self) -> None:
        super().__init__()
        self.ocr = Ocr()
        self.props = Rectangle_detector()
        self.needle = Needle()
        self.pairs = {}
        self.reset()

    def reset(self):
        self.val = None
        self.pairs.clear()
        self.ocr.reset()
        self.third_point_thresh = 300
        self.distance_dict = {}

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
            line = np.poly1d(np.polyfit(x_white, y_white,1))
            pt1 = (self.needle.line_white[0], self.needle.line_white[1])
            pt2 = (self.needle.line_white[2], self.needle.line_white[3])

        if self.needle.line_red is not None and needle == "red":
            x_red = np.array([self.needle.line_red[0]/self.norm_x, self.needle.line_red[2]/self.norm_x]) 
            y_red = np.array([self.needle.line_red[1]/self.norm_y, self.needle.line_red[3]/self.norm_y]) 
            line, _ = curve_fit(linear, x_red, y_red)
            pt1 = (self.needle.line_red[0], self.needle.line_red[1])
            pt2 = (self.needle.line_red[2], self.needle.line_red[3])

        return np.poly1d(line),pt1,pt2

    def __compute_needle_pivot_points(self, pt1 : tuple, pt2 : tuple) -> tuple:
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
                    dist_from_center = euclidean_dist((i[0], i[1]), (self.norm_x//2,self.norm_y//2))
                    if dist_from_center < min_dist_from_center:
                        center = (i[0], i[1]) 
                        min_dist = dist
                        min_dist_from_center = dist_from_center
        except:
            raise TypeError("No hough circles found for center estimation")

        return center

    def __compute_needle_pivot(self, line : np.poly1d):
        """
        This function computes the pivot of the needle with the assumption that this point
        for all gauges lies on the vertical bisector of the gauge. The point of intersection
        of the needle line and vertical bisector can be identified as the needle pivot
        """
        x1, y1 = 400/self.norm_x, 0/self.norm_x
        x2, y2 = 400/self.norm_x, 800/self.norm_y
        vertical, _ = curve_fit(linear, (x1,x2), (y1,y2))
        vertical = np.poly1d(vertical)
        x_intersection = (vertical - line).r
        y_intersection = linear(x_intersection, *line)
        return (x_int*self.norm_x, y_int*self.norm_y)

    def __get_fit_direction(self, p1 : tuple, p2 : tuple, p3 : tuple) -> str:
        """
        Function computes the direction of the parabola (horizontal/vertical)
        and the direction of the tip of the needle from the nearest major tick
        mark (paired up with a number)
        """
        delta_x = max(abs(p1[0]-p2[0]), abs(p1[0]-p3[0]), abs(p2[0]-p3[0]))
        delta_y = max(abs(p1[1]-p2[1]), abs(p1[1]-p3[1]), abs(p2[1]-p3[1]))
        if delta_x >= delta_y:
            fit = "vertical"
        else: 
            fit = "horizontal"

        return fit

    def __compute_parameters(self, x_fit : list, y_fit : list, line : np.poly1d, tip : tuple, center : tuple, fit : str) -> tuple[np.poly1d, float, tuple, str]:
        """
        This function fits a curve along 3 closest tick marks to the tip of the needle, 
        computes the point of intersection between the line estimating the needle and 
        the fit curve, and estimates the direction of this point of intersection from
        the closest major tick mark and calibrates by calculating a reference distance 
        (along the same curve) between 2 known tick marks
        """ 
        x1, x2, _ = x_fit
        y1, y2, _ = y_fit

        curve = None
        if fit == "vertical":
            curve, _ = curve_fit(parabola, x_fit, y_fit)
            reference = get_arc_length(min(x1, x2), max(x1, x2), curve)
            try:
                if self.distance_dict[2][0] * self.norm_x < 265:
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
                if self.distance_dict[2][0] * self.norm_x < 265:
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
            return None, None, None, None

        ## Direction to (xt,yt) from (x1,y1)
        xt, yt = intersection
        ## Conditions for direction choosing
        ## Both are in top half of the image
        if y1 <= center[1] and yt <= center[1]:
            if xt > x1:
                direction = "right"
            else:
                direction = "left"
        ## Both are in bottom half
        elif y1 >= center[1] and yt >= center[1]:
            if xt < x1:
                direction = "right"
            else:
                direction = "left"
        ## One is in top half and other in the bottom
        else:
            ## Both are in left half
            if x1 <= center[0] and xt <= center[0]:
                if yt < y1:
                    direction = "right"
                else:
                    direction = "left"
            ## Both are in right half
            elif x1 >= center[0] and xt >= center[0]:
                if yt > y1:
                    direction = "right"
                else:
                    direction = "left"
            ## Both are diagonally opposites
            else:
                raise InterpolationError("Needle is too far from the nearest number")
        
        return curve, reference, intersection, direction

    def __interpolate_to_tip(self, curve : np.poly1d, pt_int : tuple, fit : str) -> float:
        """
        Interpolates along the computed curve to find the exact position of the needle 
        with respect to the 2 closest tick marks. In most cases these are tick marks on
        either side of the needle so interpolation is accurate
        """
        x1, y1 = self.distance_dict[0][1].tick_centroid
        xt, yt = pt_int
        if fit == "vertical":
            return get_arc_length(min(x1, xt), max(x1, xt), curve)
        elif fit == "horizontal":
            return get_arc_length(min(yt, y1), max(yt, y1), curve)
        else:
            raise InterpolationError("Cannot interpolate from nearest tick mark")

    def __calculate_gauge_value(self, tip : tuple, line : np.poly1d, center : tuple, swing : str) -> tuple[float, np.poly1d, str]:
        """
        Runs the actual gauge value computation using the calibration information
        from the tick marks, the computed curve (horizontal or vertical parabola),
        and position of the needle tip with respect to these calibrated tick marks
        """
        ## This dict sorts the tick marks in ascending order of distance from the needle tip
        tip = (tip[0] / self.norm_x, tip[1] / self.norm_y)
        center = (center[0] / self.norm_x, center[1] / self.norm_y)
        for number, (tick_centroid, numb_centroid) in self.pairs.items():
            tick_centroid = (tick_centroid[0] / self.norm_x, tick_centroid[1] / self.norm_y)
            numb_centroid = (numb_centroid[0] / self.norm_x, numb_centroid[1] / self.norm_y)
            dist = euclidean_dist(tip, numb_centroid)
            self.distance_dict[dist] = tick_distance_object(number, tick_centroid, numb_centroid)
        
        self.distance_dict = list(sorted(self.distance_dict.items()))
        try:
            x1,y1 = self.distance_dict[0][1].tick_centroid
            x2,y2 = self.distance_dict[1][1].tick_centroid
            xt,yt = tip
        except IndexError:
            print("Not enough tick marks close to the tip to fit a curve")
            return None, None, None

        try:
            if self.distance_dict[2][0] * self.norm_x < self.third_point_thresh:
                ## Need some extra logic to check if the third number is a valid point (Ex:-80,120,-60 cant be 3 numbers)
                x3,y3 = self.distance_dict[2][1].tick_centroid
            else:
                x3, y3 = tip
        except IndexError:
            x3, y3 = tip

        nearest_num = self.distance_dict[0][1].number
        calibration = abs(self.distance_dict[0][1].number - self.distance_dict[1][1].number)
        fit = self.__get_fit_direction((x1,y1), (x2,y2), (x3,y3), tip, center)
        curve, reference, intersection, direction = self.__compute_parameters([x1,x2,x3], [y1,y2,y3], line, tip, center, fit)

        if fit == "vertical":
            ## In between x1, x2
            if np.sign(xt-x1) != np.sign(xt-x2):
                distance = self.__interpolate_to_tip(curve, intersection, fit)
            else :
                if euclidean_dist(tip, (x1, y1)) < euclidean_dist((x1,y1),(x2, y2)) * 0.6:
                    distance = self.__interpolate_to_tip(curve, intersection, fit)
                else:
                    print("Linear extrapolation")
                    distance = euclidean_dist(self.distance_dict[0][1].tick_centroid, intersection)
        elif fit == "horizontal":
            if np.sign(yt-y1) != np.sign(yt-y2):
                distance = self.__interpolate_to_tip(curve, intersection, fit)
            else :
                if euclidean_dist(tip, (x1, y1)) < euclidean_dist((x1,y1),(x2, y2)) * 0.6:
                    distance = self.__interpolate_to_tip(curve, intersection, fit)
                else:
                    print("Linear extrapolation")
                    distance = euclidean_dist(self.distance_dict[0][1].tick_centroid, intersection)
                    #distance = self.__extrapolate_to_tip(curve, intersection, fit)
        else:
            print("Parabola has to be vertical or horizonatal")
            return None, None, None

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
        start = time()
        self.norm_x, self.norm_y = image.shape[0], image.shape[1]

        ## OCR 
        self.ocr.run_ocr(image.copy())
        if len(self.ocr.lookup) < 2:
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
        if swing is None:
            print("Unable to determine needle swing")
            return;

        ## Detect tick marks
        ticks = self.props.detect_rect(image, False)
        self.pairs = self.props.pair_ticks_with_numbers(ticks, self.ocr.lookup.copy(), center)
        self.val, curve, fit = self.__calculate_gauge_value(tip, line, center, swing)
        if self.val is not None:
            print("Gauge value = {:4.2f}".format(self.val))
        print("Time taken = {:4.4f}s".format(time() - start))

        if visualize:
            for text, obj in self.ocr.lookup.items():
                [tl, tr, br, bl] = obj.box
                try:
                    ## Rectangle plotting (normal and tilted)
                    cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

                    cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                
                except cv2.error:
                    continue

            plt.figure(figsize=(10,10))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            plt.plot([pt.centroid[0] for pt in ticks], [pt.centroid[1] for pt in ticks], 'b+', markersize=12)
            plt.plot(center[0], center[1], 'g*', markersize=12)
            plt.plot(tip[0], tip[1], 'go', markersize=10)
            
            for _, (tick_centroid, numb_centroid) in self.pairs.items():
                plt.plot([tick_centroid[0], numb_centroid[0]], [tick_centroid[1], numb_centroid[1]], 'r', linewidth=2)
            
            x_plot = np.linspace(0,1,1000)
            y_line = np.clip(np.polyval(line, x_plot),0,1)
            plt.plot(x_plot * 800, y_line * 800, 'yellow')

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

def main(idx : int) -> None:
    gauge = Gauge_numeric()
    visualize = True
    
    if idx == 0: ## Works fine
        image = cv2.resize(cv2.imread("number_gauge_test/IMG_4761.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 1: ## Works fine
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

    elif idx == 8: ## Works fine
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

    elif idx == 16: ## Works fine
        image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 17: ## Works fine
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 18: ## Works fine
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 19: ## Works fine
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidtemp_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")

    elif idx == 20: ## Works fine
        image = cv2.resize(cv2.imread("ptz_gauge_images/thyoda_actual_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "red")

    elif idx == 21: ## Works fine
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gasvolume_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.read_gauge(image, visualize, "white")
    
    else :
        print("Enter a valid idx value")
    
    return;

if __name__ == "__main__":
    main(1)