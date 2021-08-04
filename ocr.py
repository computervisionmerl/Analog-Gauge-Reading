import warnings
warnings.filterwarnings("ignore")
import numpy as np
import cv2
import easyocr
import string
from collections import OrderedDict

class Ocr:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.result = None
        self.min_text = 1e20
        self.max_text = 0
        self.prob = 0.85
        self.min_text_location = ()
        self.max_text_location = ()
        self.lookup = OrderedDict()

    @staticmethod
    def _unsharp_mask(image : np.array, kernel_size : tuple = (5,5), sigma : float = 1.0, amount : float = 1.0, threshold : int =0) -> np.array:
        """
        Sharpens the denoised image (gaussian blurred) for the OCR to recognize the numbers correctly
        """
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype('uint8')
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def _recognize_digits(self, image : np.array) -> None:
        """
        Uses the easyocr to recognize the numbers in the image and returns the bounding box of each
        recognized number along with a confidence value
        """
        sharp = self._unsharp_mask(image)
        close = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
        self.result = self.reader.readtext(close, allowlist=string.digits+"-"+"MINAX", decoder='beamsearch')
        if self.result is None:
            raise ValueError("No text detected")
        return;

    def _correct_ocr(self, center : tuple) -> None:
        """
        Correct the OCR output, check to see if the same number is detected multiple times, 
        if yes, which detected one is correct (based on the radii of the numbers from center of needle)

        Correction ==> Not yet implemented  (only calibration implemented)
        """
        min_max_flag = False
        for box, text, prob in self.result:
            if prob > self.prob:
                (tl, tr, br, bl) = box
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                bl = (int(bl[0]), int(bl[1]))
                br = (int(br[0]), int(br[1]))
                center_of_bb = ((tl[0] + br[0])//2, (tl[1] + br[1])//2)

                ## Gauge calibration to find out min and max values
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
    image = cv2.resize(cv2.imread("gauge images/qualitrol_oil_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_winding_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_transformer_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_negative_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)  ## --> OCR too bad (misty glass)
    #image = cv2.resize(cv2.imread("gauge images/thyoda_actual_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    # White background gauge
    #image = cv2.resize(cv2.imread("gauge images/white_pressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.std() < 70:
        gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 5)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))

    ocr._recognize_digits(tophat)
    ocr._correct_ocr(gray.shape)
    ocr._visualize(image)
    return;

if __name__ == "__main__":
    main()
