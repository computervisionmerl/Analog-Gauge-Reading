import numpy as np
import cv2
import matplotlib.pyplot as plt

from poly import Poly
from helper import *

class Gauge(object):
    def __init__(self) -> None:
        super().__init__()
        self.brightness_param = 0.58
        self.contrast_param = 70
        self.norm_x = None
        self.norm_y = None

    def _pre_processing(self, image : np.array) -> np.ndarray:
        """
        For needle estimation
        1) Edge detection 
        -------------------
        For OCR
        1) Contrast enhancement (if necessary)
        2) Image denoising
        3) Morphological transformtions
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 85, 250)

        if gray.std() < self.contrast_param:
            gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5,5), 5)

        if calculate_brightness(image) > self.brightness_param:
            print("White Gauge")
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        else:
            print("Black Gauge")
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        
        self.norm_x = canny.shape[0]
        self.norm_y = canny.shape[1]
        return hat, canny

    def _read_gauge(self, hat : np.array, canny : np.array) -> None:
        """
        Wrapper calling all the necessary functions, holding necessary data
        Needs some more work and editing
        """
        poly = Poly(hat, canny.copy())
        poly._get_polynomial_coeffs()

        new_canny = self._get_only_ticks(poly.curve, canny)
        ticks = cv2.findContours(new_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        good_ticks = []
        for tick in ticks:
            if cv2.contourArea(tick) > 80:
                good_ticks.append(tick)

        # The coordinates in the dictionary are normalized
        pairs = self._pair_numbers_with_ticks(good_ticks, poly.ocr.lookup)
        # Gives points in normalized coordinates
        (x_int, y_int) = self._get_point_of_intersection(poly)
        self._calculate_gauge_value(pairs, (x_int, y_int), poly.curve)
        return pairs, good_ticks, poly.line, (x_int * self.norm_x, y_int * self.norm_y)

    def _get_only_ticks(self, curve : np.poly1d, canny : np.array) -> np.ndarray :
        """
        Extracts the ROI pixels and masks the canny to get only tick marked edges
        (Easier to extract contours after masking)
        """
        mask = np.zeros(canny.shape, dtype=np.uint8)
        for i in range(canny.shape[0]):
            for j in range(canny.shape[1]):
                y1 = np.polyval(curve, i/canny.shape[0]) * canny.shape[1]
                y2 = y1 // 1.5
                y1 = y1 // 1.1

                if j >= y2 and j <= y1:
                    mask[j,i] = 255

        return cv2.bitwise_and(canny, canny, mask=mask)

    def _pair_numbers_with_ticks(self, ticks : np.array, lookup : dict) -> dict:
        """
        Pairs up the major tick marks to the nearest number (given by the OCR result)
        The coordinates of the centroid of the identified tick marks are normalized 
        before returning
        """
        pairs = {}
        for number, (tl, _, br, _) in lookup.items():
            centroid_of_bb = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
            min_dist = 1e20
            for tick in ticks:
                centroid_of_rect, (_,_), _ = cv2.minAreaRect(tick)
                dist = euclidean_dist(centroid_of_bb, centroid_of_rect)
                if dist < min_dist:
                    min_dist = dist
                    centroid_of_rect = (centroid_of_rect[0]/self.norm_x, centroid_of_rect[1]/self.norm_y)
                    pairs[int(number)] = centroid_of_rect
        return dict(sorted(pairs.items()))

    def _get_point_of_intersection(self, poly : np.poly1d):
        """
        Finds the point of intersection between the polynomial curve and the line
        estimating the needle. One of the roots is inside the image and the other 
        is outside (2 points of intersection between line and curve)
        """
        roots = (poly.curve - poly.line).r
        print(roots)

        cond = False
        for root in roots:
            # Check which of the roots is inside the image
            if root >= 0 and root <= 1:
                cond = True
                x_int = root
                break

        if not cond:
            raise ValueError("Both the intersection points seem to be outside the image !!")

        return (x_int, np.polyval(poly.curve, x_int))   
    
    def _calculate_gauge_value(self, pairs : dict, point_int : tuple, curve : np.poly1d) -> None:
        """
        Computes the final gauge value based on the coordinates of the nearest tick mark, number
        corresponding to the tick mark and some reference from the gauge calibration
        """
        min_dist = 1e20 
        point = None
        n1, n2 = 0,0
        c1, c2 = (),()
        for idx, (num, coord) in enumerate(pairs.items()):
            ## For getting some reference along the polynomial curve
            if idx == 0:
                n1 = num
                c1 = coord
            elif idx == 1:
                n2 = num
                c2 = coord

            dist = euclidean_dist(point_int, coord)
            if dist < min_dist:
                min_dist = dist
                point = coord
                nearest_num = num

        x_int = point_int[0]; x_pt = point[0]; x1 = c1[0]; x2 = c2[0]
        length = get_arc_length(min(x_pt, x_int), max(x_pt, x_int), curve)
        reference = get_arc_length(min(x1, x2), max(x1, x2), curve)
        if min(x_pt, x_int) == x_int:
            direction = "Left"
        else:
            direction = "Right" 

        print("Calibration numbers ",n1," ",n2)
        print("Nearest Number = ", nearest_num)
        print("Direction from nearest number = ", direction)
        print("Distance from nearest number = ", length)
        print("Distance between 2 known values in the gauge = ", reference)
       
        if direction == "Left":
            print("Gauge value = ", nearest_num - ((n2-n1) * length / reference))
        else:
            print("Gauge value = ", nearest_num + ((n2-n1) * length / reference))
        return;

def main(visualize : bool = True):
    image = cv2.resize(cv2.imread("gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gaspressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_liquidsmall_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_transformer_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_winding_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/thyoda_actual_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gasvolume_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_lowhigh_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_hilo_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_liquidtemp_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_minmax_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_pump_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    gauge = Gauge()
    hat, canny = gauge._pre_processing(image)
    pairs, ticks, line, (x_int, y_int) = gauge._read_gauge(hat, canny)

    if visualize:
        ax = plt.axes()
        for tick in ticks:
            rect = cv2.minAreaRect(tick)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,(0,0,255),2)

        for _,  c_rect in pairs.items():
            c_rect = tuple(np.array(c_rect, dtype=int))
            cv2.circle(image, c_rect, 3, (0,255,0), -1)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ax.imshow(image)
        x_plot = np.linspace(0.25,0.45,1000)
        y_plot = np.polyval(line, x_plot)
        ax.plot(x_plot * gauge.norm_x, y_plot * gauge.norm_y, 'b')   
        print(x_int, y_int)
        plt.show() 

if __name__ == "__main__":
    main()
