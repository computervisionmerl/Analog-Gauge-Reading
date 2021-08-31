import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass

from helper import *
from needle import Needle
from ocr import Ocr
from region_props import *

class DirectionError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

@dataclass
class region:
    number : int
    tick_centroid : tuple
    bb_centroid : tuple

class Gauge(object):
    def __init__(self) -> None:
        super().__init__()
        self.needle = Needle()
        self.ocr = Ocr()
        self.val = None
        self.ratio_thresh = 0.8
        self.area_thresh = 100

    def reset(self):
        self.needle.reset()
        self.ocr.reset()
        self.val = None

    def _read_gauge(self, image : np.array, visualize : bool = True, needle = "white") -> None:
        """
        Runs all components of the algorithm namely, OCR, needle estimation, curve fitting
        and final value computation based on the tick marks and needle position
        """
        start = time.time()
        ## Run OCR 
        hat = Ocr._pre_processing(image.copy())
        self.ocr._run_ocr(hat)
        if len(self.ocr.lookup) < 2:
            print("Not enough values detected by OCR !!")
            print("Need at least 2 OCR values")
            return;

        ## Run needle estimation
        center = (image.shape[0]//2, image.shape[1]//2)
        self.norm_x = image.shape[0]; self.norm_y = image.shape[1]

        self.needle._isolate_needle(image.copy(), color="white")
        self.needle._isolate_needle(image.copy(), color="red")
        if self.needle.line_white is None and self.needle.line_red is None:
            print("Needle not found !")
            return;

        line, pt1, pt2 = self._get_needle_line_and_pts(needle)
        if visualize:
            cv2.line(image, pt1, pt2, (128,128,128), 2)

        ## Get the tip of the needle
        if euclidean_dist(pt1, center) > euclidean_dist(pt2, center):
            tip = pt1
        else :
            tip = pt2
                
        ## Get regionprops and extract tick mark locations
        dist = {}
        ticks, _ = run_regionprops(image.copy(), self.area_thresh, self.ratio_thresh)
        pairs = pair_numbers_with_ticks(ticks, self.ocr.lookup, center)
        
        if visualize:
            plt.figure(figsize=(8,8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            x_plot = np.linspace(0,1,1000)
            y_plot = np.polyval(line, x_plot); y_plot[y_plot < 0] = 0; y_plot[y_plot > 1] = 1
            
            plt.plot(x_plot * self.norm_x, y_plot * self.norm_x, 'red')
            plt.plot([c[0] for c in ticks], [c[1] for c in ticks], 'b+', markersize=15)        

        for number, (tick_centroid, bb_centroid) in pairs.items():
            dist[euclidean_dist(tip, bb_centroid)] = region(number, tick_centroid, bb_centroid)
            if visualize:
                plt.plot([tick_centroid[0], bb_centroid[0]], [tick_centroid[1], bb_centroid[1]], 'orange', linewidth=2)

        dist = dict(sorted(dist.items()))
        dist = list(dist.values())[0:2]
        self.val = self._calculate_gauge_value(tip, dist, needle, "curve")

        print("Gauge Value = ", self.val)
        print("Time elapsed = {:4.6f}s".format(time.time() - start))

        if visualize:
            plt.show()
        
        return;

    def _get_needle_line_and_pts(self, needle):
        """
        Computes the 2 end points of the needle and the equation of the line joning these 
        2 points using 2-point formula
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

        return line,pt1,pt2

    def _calculate_gauge_value(self, tip : tuple, dist : list, needle_line : np.poly1d, mode : str = "curve") -> float:
        """
        Calculate the gauge value based on the x positions of the needle tip and the 2 closest
        tick marks. Determines whether the needle is in between these identified tick marks or
        not (left or right based on detected OCR numbers)
        """
        x1 = dist[0].tick_centroid[0] / self.norm_x; y1 = dist[0].tick_centroid[1] / self.norm_y
        x3 = dist[1].tick_centroid[0] / self.norm_x; y3 = dist[1].tick_centroid[1] / self.norm_y
        x2 = tip[0] / self.norm_x
        map = {
            x1 : dist[0].number,
            x3 : dist[1].number
        }

        ## Get needle swing (clockwise / anticlockwise)
        if dist[0].number < dist[1].number and x1 < x3:
            needle_swing = "clockwise"
        elif dist[0].number > dist[1].number and x1 < x3:
            needle_swing = "anticlockwise"
        elif dist[0].number < dist[1].number and x1 > x3:
            needle_swing = "anticlockwise"
        elif dist[0].number > dist[1].number and x1 > x3:
            needle_swing = "clockwise"
        else:
            needle_swing = None

        if mode == "linear":
            x_fit = [dist[0].tick_centroid[0] / self.norm_x, dist[1].tick_centroid[0]/self.norm_x]
            y_fit = [dist[0].tick_centroid[1] / self.norm_y, dist[1].tick_centroid[1]/self.norm_y]
            ref_line = np.poly1d(fit_line(np.array(x_fit), np.array(y_fit)))

            ## Get point of intersection and calibration of the gauge
            pt_x = x2 = (ref_line - needle_line).r
            pt_y = np.polyval(needle_line, pt_x)
            pt = (pt_x, pt_y)
            reference = euclidean_dist((x1,y1), (x3,y3))

        else:
            x_fit = [dist[0].tick_centroid[0] / self.norm_x, tip[0]/self.norm_x, dist[1].tick_centroid[0]/self.norm_x]
            y_fit = [dist[0].tick_centroid[1] / self.norm_y, tip[1]/self.norm_y, dist[1].tick_centroid[1]/self.norm_y]
            curve = np.poly1d(fit_curve(np.array(x_fit), np.array(y_fit)))
            
            x_plot = np.linspace(0,1,1000)
            y_plot = np.polyval(curve, x_plot); y_plot[y_plot > 1] = 1.0; y_plot[y_plot < 0] = 0.0
            plt.plot(x_plot*self.norm_x, y_plot*self.norm_y, 'blue', linewidth=2)    

            reference = get_arc_length(min(x1,x3), max(x1, x3), curve)

        calibration = abs(dist[1].number - dist[0].number)

        ## Get whether the needle is between the 2 numbers, to the left or right
        if (x2 - x1 > 0 and x2 - x3 < 0) or (x2 - x1 < 0 and x2 - x3 > 0):
            ## Needle in between 2 numbers
            nearest_num = map[min(x1, x3)]
            direction = "right"
            if mode == "linear":
                distance = euclidean_dist((x1,y1), pt)
            else:
                distance = get_arc_length(min(x1, x3), x2, curve)

        else:
            ## The needle is outside the range of the 2 numbers
            if x2 - x1 > 0 and x2 - x3 > 0:
                direction = "right"
                nearest_num = map[max(x1, x3)]
                if mode == "linear":
                    distance = euclidean_dist((x3,y3), pt)
                else:
                    distance = get_arc_length(max(x1, x3), x2, curve)
            
            elif x2 - x1 < 0 and x2 - x3 < 0:
                direction = "left"
                nearest_num = map[min(x1, x3)]
                if mode == "linear":
                    distance = euclidean_dist((x1,y1), pt)
                else:
                    distance = get_arc_length(x2, min(x1,x3), curve)
            
            else:
                raise DirectionError("Tip should be in range, to the left or right of the range.")

        print("Range = ("+str(dist[0].number)+","+str(dist[1].number)+")")
        print("Distance to nearest major tick = ", distance)
        print("Direction wrt nearest major tick = ", direction)
        print("Nearest associated number = ", nearest_num)
        print("Calibration value for reference distance = ", calibration)
        print("Needle swing = ", needle_swing)

        delta_val = (calibration / reference) * distance
        if needle_swing == "clockwise":
            if direction == "left":
                return nearest_num - delta_val
            elif direction == "right":
                return nearest_num + delta_val

        else:
            if direction == "left":
                return nearest_num + delta_val
            elif direction == "right":
                return nearest_num - delta_val

def main(idx : int) -> None:
    gauge = Gauge()

    if idx == 0: 
        image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_negative_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    if idx == 1: 
        image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge.ratio_thresh = 0.65
        gauge._read_gauge(image, True, "white")

    elif idx == 2:
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 3:
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 4: 
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gaspressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 5:
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gasvolume_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 6:
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidtemp_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 7:
        image = cv2.resize(cv2.imread("ptz_gauge_images/thyoda_actual_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red")
        
    elif idx == 8:
        image = cv2.resize(cv2.imread("substation_images/spot_ptz_temp_gauge.png"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red")
    
    elif idx == 9: 
        image = cv2.resize(cv2.imread("substation_images/spot_ptz_gasvolume_gauge.png"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red")

    elif idx == 10: 
        image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800), cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 11: 
        image = cv2.resize(cv2.imread("substation_images/qualitrol_temperature_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 12: 
        image = cv2.resize(cv2.imread("substation_images/meppi_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    else:
        print("Enter a valid idx value")
        print("Exitting the code")

    return;

if __name__ == "__main__":
    main(1)
