from math import sqrt

from numpy.core.function_base import linspace
from needle import Needle
from ocr import Ocr
from helper import calulate_brightness, euclidean_dist

import cv2
import numpy as np

class Gauge_reader(object):
    def __init__(self):
        super().__init__()
        self.ocr = Ocr()
        self.needle = Needle()

        self.contrast_param = 70
        self.brightness_param = 0.58 #0.48

    def _pre_processing(self, image : np.array, visualize : bool = False) -> np.array:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        ## Contrast enhancement if needed
        if gray.std() < self.contrast_param:
            gray = cv2.equalizeHist(gray)
        ## Image denoising 
        blur = cv2.GaussianBlur(gray, (5,5), 5)

        ## Check white / dark background
        if calculate_brightness(image) > self.brightness_param:
            print("White Gauge")
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        else:
            print("Black Gauge")
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        
        ## Egde detection for needle estimation
        canny_needle = cv2.Canny(gray, 85, 255)
        canny_tick = cv2.Canny(hat, 85, 255)
        return (hat, canny_needle, canny_tick)

    def _pair_ticks_with_numbers(self, canny : np.array) -> dict:
        """
        Very primitive way of pairing the numbers with the tick marks 
        Contour closest to the number is the tick mark corresponding to the number
        """
        ticks = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
        pair_dict = {}
        for key, (tl, _, br, _, _) in self.ocr.lookup.items():
            center_bb = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
            min_dist = 1e20
            for tick in ticks:
                rect = cv2.minAreaRect(tick); (center, (_,_), _) = rect
                box = np.int0(cv2.boxPoints(rect))
                dist = euclidean_dist(center, center_bb)
                if dist < min_dist:
                    min_dist = dist
                    pair_dict[key] = box
                
        return pair_dict

    def _get_arc_length(self, x1 : float, x2 : float, bestmodel : np.ndarray, n_steps : int = 1000) -> int:
        """
        Estimates the length along the polynomial curve (bestmodel) between the ticks near the minimum and 
        maximum numbers. This applies piecewise linear approximation to the curve to estimate the length
        """
        ## x1 and x2 are normalized but the length is scaled back to pixel coordinates before returning
        x1 = (x1 - self.ocr.old_mu_x) / self.ocr.old_std_x
        x2 = (x2 - self.ocr.old_mu_x) / self.ocr.old_std_x
        length = 0.0
        dx = (x2 - x1) / n_steps
        curr_x = x1

        while curr_x <= x2:
            prev_x = curr_x
            prev_y = np.polyval(bestmodel, prev_x)
            curr_x = curr_x + dx
            curr_y = np.polyval(bestmodel, curr_x)

            x_start = prev_x*self.ocr.old_std_x + self.ocr.old_mu_x
            y_start = prev_y*self.ocr.old_std_y + self.ocr.old_mu_y
            x_finish = curr_x*self.ocr.old_std_x + self.ocr.old_mu_x
            y_finish = curr_y*self.ocr.old_std_y + self.ocr.old_mu_y

            length += euclidean_dist((x_start, y_start), (x_finish, y_finish))
        
        return length

    def _read_gauge(self, image : np.array) -> None:
        hat, canny_needle, canny_tick = self._pre_processing(image.copy())
        self.ocr._recognize_digits(hat.copy())
        
        ## Extract the needle 
        self.needle._isolate_needle(canny_needle)

        ## Extract the segment with the tick marks
        (x, z, bestmodel) = self.ocr._fit_polynomial()
        mask = self.ocr._get_roi(bestmodel)
        canny_tick[mask == 0] = 0

        pair_dict = self._pair_ticks_with_numbers(canny_tick.copy())
        for _, box in pair_dict.items():
            cv2.drawContours(image, [box], -1, (0,255,0), 2)

        import matplotlib.pyplot as plt
        ax = plt.axes()
        ax.imshow(image)
        ax.plot(x, z, 'blue')
        ax.plot(x, z-50, 'red')
        plt.show()

        cv2.imshow("image", image)
        cv2.waitKey(0)  

    def _get_gauge_value(self) -> float:
        ## At this point, we have the bounding boxes, needle and center computed along with the calibration done (min and max of gauge)
        if self.ocr.min_text == 1e20 or self.ocr.max_text == 0:
            raise ValueError("Not all computations done, OCR not finished properly")
        
        if self.needle.needle is None or self.needle.center is None:
            raise ValueError("Not all computations done, Needle estimation not finished properly")

        ## Coordinates of the minimum and maximum detected text values
        min_point, max_point = None, None
        for box, text, prob in self.ocr.result:
            if prob > self.ocr.prob:
                (tl, tr, br, bl) = box
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                bl = (int(bl[0]), int(bl[1]))
                br = (int(br[0]), int(br[1]))

                if (text == str(self.ocr.min_text) or text == "MIN") and min_point is None:
                    min_point = (int((int(tl[0]) + int(br[0]))/2), int((int(tl[1]) + int(br[1]))/2))
                if (text == str(self.ocr.max_text) or text == "MAX") and max_point is None:
                    max_point = (int((int(tl[0]) + int(br[0]))/2), int((int(tl[1]) + int(br[1]))/2))

        quad_min_number = self.needle._find_quadrant(min_point)
        quad_max_number = self.needle._find_quadrant(max_point)
        quad_needle_point = self.needle._find_quadrant((self.needle.needle[0], self.needle.needle[1]))

        ## Find angle of min, max and needle wrt downward y axis
        angle_min_number = self.needle._find_angle_based_on_quadrant(quad_min_number, min_point)
        angle_max_number = self.needle._find_angle_based_on_quadrant(quad_max_number, max_point)
        angle_needle_line = self.needle._find_angle_based_on_quadrant(quad_needle_point, (self.needle.needle[0], self.needle.needle[1]))

        ## Find angles of needle and max wrt min
        angle_line_min = angle_needle_line - angle_min_number
        angle_min_max = angle_max_number - angle_min_number
        calibration_range = self.ocr.max_text - self.ocr.min_text
        value_per_degree = calibration_range / angle_min_max
        return value_per_degree * angle_line_min + self.ocr.min_text

    def _visualize(self, image : np.array) -> None:
        if self.needle.needle is not None:
            cv2.line(image, (self.needle.needle[0],self.needle.needle[1]), (self.needle.needle[2],self.needle.needle[3]), (255,0,0), 2)

        if self.needle.center is not None:
            cv2.circle(image, (self.needle.center[0], self.needle.center[1]), 5, (255,0,0), -1, cv2.LINE_AA)

        if self.ocr.result:
            for box, text, prob in self.ocr.result:
                if prob > self.ocr.prob:
                    (tl, tr, br, bl) = box
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    br = (int(br[0]), int(br[1]))

                    ## Drawing a tilted rectangle
                    cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

                    cv2.putText(image, text, (tl[0], tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                    if text == str(self.ocr.min_text) or text == str(self.ocr.max_text) or text == "MIN" or text == "MAX":
                        point = (int((int(tl[0]) + int(br[0]))/2), int((int(tl[1]) + int(br[1]))/2))
                        cv2.line(image, self.needle.center, point, (0,255,0), 2, cv2.LINE_AA)

        length_of_needle = sqrt((self.needle.center[0] - self.needle.needle[0])**2 + 
                                (self.needle.center[1] - self.needle.needle[1])**2)
        cv2.circle(image, self.needle.center, int(length_of_needle), (0,0,0), 2, cv2.LINE_AA)
        cv2.line(image, (int(self.needle.w/2), 0), (int(self.needle.w/2), self.needle.h), (0,0,0), 2, cv2.LINE_AA)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        return;


def main():
    reader = Gauge_reader()
    # Black background gauge
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_oil_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_winding_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_transformer_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/thyoda_actual_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_small_gauge.jfif"), (800,800), interpolation=cv2.INTER_CUBIC)

    # White background gauge
    #image = cv2.resize(cv2.imread("gauge images/white_pressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    # Spot PTZ images
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_gasvolume_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_lowhigh_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_gaspressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_hilo_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_liquidsmall_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_liquidtemp_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_minmax_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_pump_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    reader._read_gauge(image)

if __name__ == "__main__":
    main()
