import numpy as np
import cv2
import matplotlib.pyplot as plt

from helper import *
from needle import Needle
from ocr import Ocr
from region_props import *

class Gauge(object):
    def __init__(self) -> None:
        super().__init__()
        self.needle = Needle()
        self.ocr = Ocr()
    
    def _read_gauge(self, image : np.array, visualize : bool = True, needle = "white") -> None:
        ## Run needle estimation
        hat_white, hat_red = Needle._pre_processing(image.copy())
        self.norm_x = hat_white.shape[0]; self.norm_y = hat_white.shape[1]

        self.needle._isolate_needle(hat_white, hat_red, color="white")
        self.needle._isolate_needle(hat_white, hat_red, color="red")
        if self.needle.line_white is not None:
            x_white = np.array([self.needle.line_white[0]/self.norm_x, self.needle.line_white[2]/self.norm_x]) 
            y_white = np.array([self.needle.line_white[1]/self.norm_y, self.needle.line_white[3]/self.norm_y]) 
        if self.needle.line_red is not None:
            x_red = np.array([self.needle.line_red[0]/self.norm_x, self.needle.line_red[2]/self.norm_x]) 
            y_red = np.array([self.needle.line_red[1]/self.norm_y, self.needle.line_red[3]/self.norm_y]) 

        ## Run OCR 
        hat = Ocr._pre_processing(image.copy())
        self.ocr._run_ocr(hat)

        if visualize:
            self._visualize(image)
        
        ## Get regionprops and extract tick mark locations
        ticks, pairs = run_regionprops(image.copy(), self.ocr.lookup, self.ocr.mask)

        plt.figure(figsize=(12,12))
        plt.imshow(image)
        plt.plot([c[0] for c in ticks], [c[1] for c in ticks], 'b+', markersize=15)

        x_fit = []; y_fit = []
        num_low, num_high = None, None
        crd_low, crd_high = None, None
        min_dist = 1e20
        for idx, (number, (tick, bb)) in enumerate(pairs.items()):
            x = [tick[0], bb[0]]; y = [tick[1], bb[1]]
            x_fit.append(tick[0]/self.norm_x); y_fit.append(tick[1]/self.norm_y)
            plt.plot(x,y,'orange',linewidth=2)

            if idx == 0:
                num_low = number
                crd_low = (tick[0]/self.norm_x, tick[1]/self.norm_y)

            if idx == 1:
                num_high = number
                crd_high = (tick[0]/self.norm_x, tick[1]/self.norm_y)

        curve = np.poly1d(fit_curve(np.array(x_fit), np.array(y_fit)))
        if needle == "red":
            line = np.poly1d(fit_line(x_red, y_red))
        else :
            line = np.poly1d(fit_line(x_white, y_white))

        (x_int, y_int) = self._get_point_of_intersection(curve, line)
        for idx, (number, (tick,_)) in enumerate(pairs.items()):
            tick = (tick[0]/self.norm_x, tick[1]/self.norm_y)
            dist = euclidean_dist((x_int, y_int), tick)
            if dist < min_dist:
                min_dist = dist
                x_nearest_tick = tick[0]
                nearest_num = number

        print(x_int, x_nearest_tick, nearest_num, crd_low, crd_high, num_low, num_high, curve)
        self._calculate_gauge_value(x_int, x_nearest_tick, nearest_num, crd_low[0], crd_high[0], num_low, num_high, curve)
    
        x_plot = np.linspace(0,1,1000)
        y_plot = np.polyval(curve, x_plot); y_plot[y_plot > 1] = 1.0; y_plot[y_plot < 0] = 0.0
        plt.plot(x_plot*self.norm_x, y_plot*self.norm_y, 'blue', linewidth=2)
        y_plot = np.polyval(line, x_plot); y_plot[y_plot > 1] = 1.0; y_plot[y_plot < 0] = 0.0
        plt.plot(x_plot*self.norm_x, y_plot*self.norm_y, 'red', linewidth=2)

        plt.show()
        
        self.needle.reset()
        self.ocr.reset()

    def _get_point_of_intersection(self, curve : np.poly1d, line : np.poly1d) -> tuple:
        roots = (curve - line).r
        cond = False
        for root in roots:
            if root >= 0 and root <= 1:
                    cond = True
                    x_int = root
                    break
        
        if not cond:
            raise ValueError("Both points outside the image")
        
        return (x_int, np.polyval(curve, x_int))
    
    def _calculate_gauge_value(self, x_int : float, x_tick : float, number : int, x1 : float, x2 : float, num_low : int, num_high : int, curve : np.poly1d):
        distance = get_arc_length(min(x_int, x_tick), max(x_int, x_tick), curve)
        reference = get_arc_length(min(x1, x2), max(x1, x2), curve)

        if min(x_int, x_tick) == x_int:
            direction = "Left"
        else:
            direction = "Right" 

        if x1 < x2:
            clockwise = True
            print("Gauge direction = Clockwise")
        else:
            clockwise = False
            print("Gauge direction = Anticlockwise")

        print("Calibration numbers ",num_low," ",num_high)
        print("Nearest Number = ", number)
        print("Direction from nearest number = ", direction)
        print("Distance from nearest number = ", distance)
        print("Distance between 2 known values in the gauge = ", reference)
        
        if direction == "Left" and clockwise:
            print("Gauge value = ", number - ((num_high-num_low) * distance / reference))
        elif direction == "Left" and not clockwise :
            print("Gauge value = ", number + ((num_high-num_low) * distance / reference))
        elif direction == "Right" and clockwise:
            print("Gauge value = ", number + ((num_high-num_low) * distance / reference))
        else:
            print("Gauge value = ", number - ((num_high-num_low) * distance / reference))

    def _visualize(self, image : np.array) -> None:
        if self.needle.line_white is not None:
            l = self.needle.line_white 
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (0,255,0), 2)
        
        if self.needle.line_red is not None:
            l = self.needle.line_red
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2)

        for text, obj in self.ocr.lookup.items():
            (tl, tr, br, bl) = obj.box
            try:
                ## Rectangle plotting (normal and tilted)
                cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

                cv2.putText(image, text, (tl[0], tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            
            except cv2.error:
                continue

        cv2.imshow("image", image)
        cv2.waitKey(0)

def main():    
    gauge = Gauge()

    if idx == 0: ## OCR giving only 1 number
        image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_negative_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    if idx == 1: ## Not sure why calculation is off
        image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 2:
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "red")

    elif idx == 3:
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 4: ## Needle polynomial isn't estimated correctly (need to find out)
        image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gaspressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
        gauge._read_gauge(image, True, "white")

    elif idx == 5: ## Distance to tick mark = 0.0
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

    else:
        print("Enter a valid idx value")

    return;

if __name__ == "__main__":
    main(2)
