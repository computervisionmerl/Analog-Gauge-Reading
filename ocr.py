import warnings
from numpy.lib import polynomial
from torch.nn.init import xavier_normal_

warnings.filterwarnings("ignore")
import numpy as np
import cv2
import random
import easyocr
import string
import matplotlib.pyplot as plt
from collections import OrderedDict
from helper include euclidean_dist

"""
Line numbers 64, 66 for MIN/MAX gauge
The values are dummy values right now, but need to be set based on gauge datasheet
"""
class Ocr:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.result = None
        self.min_text = 1e20
        self.max_text = 0
        self.prob = 0.85
        self.lookup = OrderedDict()

    @staticmethod
    def _unsharp_mask(image : np.array, kernel_size : tuple = (5,5), sigma : float = 1.0, amount : float = 1.0, threshold : int =0) -> np.array:
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype('uint8')
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def _calibrate_gauge(self) -> None:
        min_max_flag = False
        for box, text, prob in self.result:
            if prob > self.prob:
                (tl, tr, br, bl) = box
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                bl = (int(bl[0]), int(bl[1]))
                br = (int(br[0]), int(br[1]))

                try:
                    if text == "MIN":
                        self.lookup[20] = (tl, tr, br, bl, "20")
                        min_max_flag = True

                    elif text == "MAX":
                        self.lookup[50] = (tl, tr, br, bl, "50")
                        min_max_flag = True

                    else:
                        self.lookup[int(text)] = (tl, tr, br, bl, text)
                        if int(text) < self.min_text:
                            self.min_text = int(text)
                        elif int(text) > self.max_text:
                            self.max_text = int(text)

                except ValueError:
                    pass
                    
        if min_max_flag:
            self.min_text = 20
            self.max_text = 50

        self.lookup = dict(sorted(self.lookup.items()))
        return;

    def _recognize_digits(self, image : np.array) -> None:
        sharp = self._unsharp_mask(image)
        close = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
        self.w, self.h = close.shape

        self.result = self.reader.readtext(close)
        if self.result is None:
            raise ValueError("No text detected")
        self._calibrate_gauge()
        return;

    def _get_roi(self, bestmodel : np.ndarray) -> np.array:
        """
        Gets the image with only tick marks based on the fit polynomial and the heuristic to define the ROI
        """
        mask = np.zeros((self.w, self.h), dtype=np.uint8)
        for i in range(self.w):
            x = (i - self.old_mu_x) / self.old_std_x
            y1 = (np.polyval(bestmodel, x) * self.old_std_y) + self.old_mu_y; 
            y2 = y1 - 50
            for j in range(self.h):
                if j < y1 and j > y2:
                    mask[j,i] = 255

        return mask

    def _fit_polynomial(self, n=100) -> np.ndarray:
        """
        Fits a 2nd degree polynomial through fixed points in the image
        These are points on the bounding box given by OCR, and are chosen based on the distance from the center of the image
        """
        x = []; y = []
        for _, val in self.lookup.items():
            (tl, tr, br, bl, _) = val
            dist_dict = {
                euclidean_dist(tl, (self.w//2, self.h//2)) : tl,
                euclidean_dist(tr, (self.w//2, self.h//2)) : tr,
            }
            largest_key = max(dist_dict)
            x.append(dist_dict[largest_key][0]); y.append(dist_dict[largest_key][1])

        x = np.array(x); y = np.array(y)
        if len(x) > 3:
            ## Can consider circle on this case because polynomial fitting isn't accurate
            pass

        ## Normalize the points
        self.old_mu_x = x.mean(); self.old_std_x = x.std()
        self.old_mu_y = y.mean(); self.old_std_y = y.std()
        ## New normalized data (0 mean, 1 std dev)
        x = (x - x.mean()) / x.std() 
        y = (y - y.mean()) / y.std()

        ## Curve fitting
        iterations = 0
        bestmodel = None
        besterr = 1e20
        while iterations < n:
            poly = np.polyfit(x,y,len(x)-1)
            z = np.polyval(poly, x)
            err = np.sum((z - y)**2)
            if err < besterr:
                besterr = err
                bestmodel = poly

            iterations += 1

        x_plot = np.linspace(min(x), max(x), 1000)
        z = np.polyval(bestmodel, x_plot)
        ## Restore x,y values back to pixel coordinates
        x = (x_plot * self.old_std_x) + self.old_mu_x
        y = (y * self.old_std_y) + self.old_mu_y
        z = (z * self.old_std_y) + self.old_mu_y
        return (x, z, bestmodel)
    
    def _visualize(self, image : np.array) -> None:
        if self.result:
            for box, text, prob in self.result:
                if prob > self.prob:
                    (tl, tr, br, bl) = box
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    br = (int(br[0]), int(br[1]))

                    ## Rectangle plotting (normal and tilted)
                    cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                    cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

                    cv2.putText(image, text, (tl[0], tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            cv2.imshow("Image", image)
            cv2.waitKey(0)
        return;

def main():
    ocr = Ocr()
    
    # Black background gauge
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_oil_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_winding_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_transformer_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/thyoda_actual_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_small_gauge.jfif"), (800,800), interpolation=cv2.INTER_CUBIC)

    # White background gauge
    #image = cv2.resize(cv2.imread("gauge images/white_pressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    # SPOT PTZ images
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_gasvolume_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_lowhigh_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_gaspressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_hilo_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_liquidsmall_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_liquidtemp_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_minmax_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz gauge images/spot_ptz_pump_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.std() < 70:
        gray = cv2.equalizeHist(gray)
        
    blur = cv2.GaussianBlur(gray, (5,5), 5)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
    canny = cv2.Canny(tophat, 85, 255)

    ocr._recognize_digits(tophat)
    ocr._visualize(image)

    
    (x, z, bestmodel) = ocr._fit_polynomial()
    mask = ocr._get_roi(bestmodel)
    canny[mask==0] = 0
    
    ocr._visualize(image)
    ax = plt.axes()
    ax.imshow(image)
    ax.plot(x, z, 'blue')
    ax.plot(x, z-50, 'red')
    
    cv2.imshow("canny", canny)
    cv2.waitKey(0)
    plt.show()
    

if __name__ == "__main__":
    main()
