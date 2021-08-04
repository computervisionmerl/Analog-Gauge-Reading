from time import time
import warnings

import cv2
import numpy as np
from ocr import Ocr
from needle import Needle
from helper import *

class Gauge(object):
    def __init__(self) -> None:
        super().__init__()
        self.ocr = Ocr()
        self.needle = Needle()

        self.brightness_param = 0.58
        self.contrast_param = 70

    def _pre_processing(self, image : np.array) -> np.ndarray:
        """
        Some basic image pre_processing techniques
        The first is for needle estimation and center extraction

        The next set is for the OCR.
        First is contrast enhancement (if needed), next comes denoising of the image,
        and finally morphological transformation based on whether gauge has white or
        black background (to enhance the text from the background)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.w, self.h = gray.shape
        canny = cv2.Canny(gray, 85, 255)

        if gray.std() < self.contrast_param:
            gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5,5), 5)

        if calculate_brightness(image) > self.brightness_param:
            print("White Gauge")
            return (cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))), canny)
        else:
            print("Black Gauge")
            return (cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))), canny)

    def _get_roi_mask(self, c1 : tuple, c2 : tuple) -> np.array:
        """
        Gets a circular ROI (points in between 2 cirles), the other pixels are blacked out
        Need the argument for the circles in the format (cx, cy, r)
        c1 --> Bigger circle and c2 --> Smaller circle
        """
        mask = np.zeros((self.w, self.h), dtype=np.uint8)
        for j in range(0,self.h):
            for i in range(0,self.w):
                if point_inside_circle((c1[0],c1[1]), c1[2], (i,j)) and not point_inside_circle((c2[0],c2[1]), c2[2], (i,j)):
                    mask[j,i] = 255
        return mask

    def _extract_ticks(self, canny : np.array) -> np.ndarray:
        """
        Tries to extract the ticks based on OCR located numbers 
            (since the ticks are radially outward as seen from the center of the gauge)

        Picking points for the ROI contour for extracting the region only with ticks
        """
        points = []
        for _,v in self.ocr.lookup.items():
            (tl, tr, br, bl, _) = v
            dist_dict = {
                euclidean_dist(tl, self.needle.center) : tl,
                euclidean_dist(tr, self.needle.center) : tr,
                euclidean_dist(br, self.needle.center) : br,
                euclidean_dist(bl, self.needle.center) : bl
            }
            largest_key = max(dist_dict)
            points.append((dist_dict[largest_key][0], dist_dict[largest_key][1]))

        if len(points) < 3:
            """
            Without at least 3 points in the contour, we cannot compute the ROI mask and hence cannot  isolate
            the tick marks in the image. The reading has to come from the numbers, this is inaccurate sometimes
            """
            warnings.warn("Less than 3 OCR values, gauge reading is not accurate", RuntimeWarning, stacklevel=1)
            tl, _, br, _, _ = self.ocr.lookup[self.ocr.max_text]
            self.max_tick_center = ((tl[0] + br[0])//2, (tl[1] + br[1])//2)
            tl, _, br, _, _ = self.ocr.lookup[self.ocr.min_text]
            self.min_tick_center = ((tl[0] + br[0])//2, (tl[1] + br[1])//2)
            print(self._calculate_gauge_reading())
            return (None, None)

        ((cx, cy), r) = define_circle(points[0], points[1], points[2])
        mask = self._get_roi_mask((cx,cy,r+50),(cx,cy,r))
        masked_canny = cv2.bitwise_and(canny, canny, mask=mask)
        ticks = cv2.findContours(masked_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

        return (ticks, masked_canny)

    def _find_min_max_ticks(self) -> None:
        """
        Find the ticks closest to the minimum and maximum values detected by the OCR
        Stores their centroid values for angular position computation
        """
        tl, _, br, _, _ = self.ocr.lookup[self.ocr.max_text]
        center_max = (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2))
        tl, _, br, _, _ = self.ocr.lookup[self.ocr.min_text]
        center_min = (int((tl[0] + br[0])/2), int((tl[1] + br[1])/2))

        dist_max_number = 1e20; dist_min_number = 1e20
        for tick in self.ticks:
            x,y,w,h = cv2.boundingRect(tick)
            centroid = (x + w//2, y + h//2)

            if euclidean_dist(centroid, center_max) < dist_max_number:
                dist_max_number = euclidean_dist(centroid, center_max)
                self.max_tick_center =  centroid

            if euclidean_dist(centroid, center_min) < dist_min_number:
                dist_min_number = euclidean_dist(centroid, center_min)
                self.min_tick_center = centroid

        return;

    def _calculate_gauge_reading(self):
        """
        Computes the value read by the needle based on the locations of the ticks, needle 
        """
        needle_tip = (self.needle.needle[0], self.needle.needle[1])
        angle_max_tic = self.needle._find_angle_based_on_quadrant(self.needle._find_quadrant(self.max_tick_center), self.max_tick_center)
        angle_min_tic = self.needle._find_angle_based_on_quadrant(self.needle._find_quadrant(self.min_tick_center), self.min_tick_center)
        angle_needle = self.needle._find_angle_based_on_quadrant(self.needle._find_quadrant(needle_tip), needle_tip)

        dial_angle = angle_max_tic - angle_min_tic
        angle_needle_min_tic = angle_needle - angle_min_tic
        calibration_range = self.ocr.max_text - self.ocr.min_text
        reading_per_degree_rotation = calibration_range / dial_angle
        return angle_needle_min_tic * reading_per_degree_rotation + self.ocr.min_text

    def _read_gauge(self, image : np.array) -> None:
        start = time()
        (hat, canny) = self._pre_processing(image)
        self.ocr._recognize_digits(hat)
        self.needle._isolate_needle(canny)
        self.ocr._correct_ocr(self.needle.center)
        (self.ticks, _) = self._extract_ticks(canny.copy())
        if self.ticks is not None:
            self._find_min_max_ticks()
            print(self._calculate_gauge_reading())

        print("Time elapsed = {}s".format(time() - start, ".4f"))
        self._visualize(image)
        return;

    def _visualize(self, image : np.array) -> None:
        if self.needle.needle is not None:
            l = self.needle.needle
            cv2.line(image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 2)

        if self.needle.center is not None:
            cv2.circle(image, self.needle.center, 5, (255,0,0), -1, cv2.LINE_AA)

        for _,v in self.ocr.lookup.items():
            (tl, tr, br, bl, text) = v
            ## Drawing a tilted rectangle
            cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
            cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
            cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
            cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

            cv2.putText(image, text, (tl[0], tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        cv2.circle(image, self.min_tick_center, 3, (0,128,255), -1)
        cv2.line(image, self.needle.center, self.min_tick_center, (0,255,0), 2, cv2.LINE_AA)
        cv2.circle(image, self.max_tick_center, 3, (0,128,255), -1)
        cv2.line(image, self.needle.center, self.max_tick_center, (0,255,0), 2, cv2.LINE_AA)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        return;

def main() -> None:
    gauge = Gauge()
    # Black background gauge
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_oil_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_winding_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_transformer_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("gauge images/qualitrol_negative_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)  ## --> OCR too bad (misty glass)
    #image = cv2.resize(cv2.imread("gauge images/thyoda_actual_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    # White background gauge
    #image = cv2.resize(cv2.imread("gauge images/white_pressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    gauge._read_gauge(image)

if __name__ == "__main__":
    main()
