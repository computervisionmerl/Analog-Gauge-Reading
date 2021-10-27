import time
import cv2
import numpy as np

from ocr import Ocr
from ellipse_detection import Ellipse_dlsq, params
from helper import *
from skimage.morphology import skeletonize

import warnings
warnings.filterwarnings("ignore")

class pressure_gauge(object):
    def __init__(self) -> None:
        super().__init__()
        self.ocr = Ocr()
        self.ell = Ellipse_dlsq()

    def __preprocessing(self, image : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.std() > 70 or gray.std() < 35:
            gray = cv2.equalizeHist(gray)
        
        blur = cv2.GaussianBlur(gray, (5,5), 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
        if calculate_brightness(image) > 0.52:
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
        else:
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)
        thresh = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        thresh_float = np.array(thresh, dtype="float64") / 255.
        edge_map = np.array(skeletonize(thresh_float)*255., dtype="uint8")

        return hat, thresh, edge_map

    def read_gauge(self, image : np.ndarray) -> params:
        ## Run preprocessing, OCR, and the needle estimation
        hat, _, edge_map = self.__preprocessing(image.copy())
        self.ocr.construct_initial_lookup(hat.copy())

        ## Separate the numbers into respective scales
        ellipse = self.ell.detect_tick_ellipse(edge_map.copy())
        self.ocr.separate_scales(ellipse)
        return ellipse;

    def visualize(self, image : np.ndarray, ellipse : params) -> None:
        center = (int(ellipse.center[0]), int(ellipse.center[1]))
        axes = (int(ellipse.A), int(ellipse.B))
        angle = int(math.degrees(ellipse.theta))
        cv2.ellipse(image, center, axes, angle, 0, 360, (0,0,255), 2)

        ## All the numbers inside the ellipse
        for text, obj in self.ocr.inside.items():
            [tl, tr, br, bl] = obj.box
            try:
                cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(image, text, (tl[0], tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            except cv2.error:
                continue

        ## All the numbers outside the ellipse
        for text, obj in self.ocr.outside.items():
            [tl, tr, br, bl] = obj.box
            try:
                cv2.line(image, tl, tr, (128,128,0), 2, cv2.LINE_AA)
                cv2.line(image, tr, br, (128,128,0), 2, cv2.LINE_AA)
                cv2.line(image, br, bl, (128,128,0), 2, cv2.LINE_AA)
                cv2.line(image, bl, tl, (128,128,0), 2, cv2.LINE_AA)
                cv2.putText(image, text, (tl[0], tl[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            except cv2.error:
                continue

        cv2.imshow("image", image)
        cv2.waitKey(0)

def main(idx : int):
    if idx == 0:
        image = cv2.resize(cv2.imread("substation_images/ferguson_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    elif idx == 1:
        image = cv2.resize(cv2.imread("substation_images/proflo_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    elif idx == 2:
        image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    elif idx == 3:
        image = cv2.resize(cv2.imread("substation_images/mitsubishi_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 4:
        image = cv2.resize(cv2.imread("substation_images/trafag_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 5:
        image = cv2.resize(cv2.imread("substation_images/negative_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    else:
        print ("Invalid idx value in the main function. Must be 0-4")
        return;

    gauge = pressure_gauge()
    start = time.time()
    ellipse = gauge.read_gauge(image)
    print("Time taken = {:4.4f} sec".format(time.time() - start))
    gauge.visualize(image, ellipse)

if __name__ == "__main__":
    main(3)