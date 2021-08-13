import cv2
import numpy as np
import easyocr
from helper import *

import warnings
warnings.filterwarnings("ignore")

class Ocr:
    def __init__(self) -> None:
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.min_text = None
        self.max_text = None
        self.prob = 0.80
        self.lookup = dict()

    @staticmethod
    def _unsharp_mask(image : np.array, kernel_size : tuple = (5,5), sigma : float = 1.0, amount : float = 1.0, threshold : int =0) -> np.array:
        """
        Applies unsharp masking to the morphologically transformed image
        Also applies closing to ensure there aren't any gaps in the numbers for the OCR to detect
        """
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype('uint8')
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        #return cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2,2)))
        return sharpened

    def _run_ocr(self, image : np.array) -> None:
        """
        Saves the OCR result in a dictionary and
        classifies gauge depending on the OCR values
        """
        sharp = self._unsharp_mask(image)
        self.w, self.h = sharp.shape
        self.result = self.reader.readtext(
            sharp, decoder='beamsearch' #allowlist=string.digits+"-"+"MINAX"+"PUOF"+"HLW"
        )

        if self.result:
            for box, text, prob in self.result:
                if prob > self.prob:
                    (tl, tr, br, bl) = box
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    br = (int(br[0]), int(br[1]))

                    self.lookup[text] = (tl, tr, br, bl)

            if "PUMP" in self.lookup or "ON" in self.lookup or "OFF" in self.lookup:
                self.gauge_type = "PUMP-ON-OFF"
            
            elif "HI" in self.lookup or "LO" in self.lookup:
                self.gauge_type = "HI-LO"

            elif "HIGH" in self.lookup or "LOW" in self.lookup:
                self.gauge_type = "HIGH-LOW"
                
            elif "MIN" in self.lookup or "MAX" in self.lookup:
                self.gauge_type = "MIN-MAX"
                
            else:
                self.gauge_type = "NUMERIC" 

            self._clean_lookup_dict()   

        return;

    def _clean_lookup_dict(self) -> None:
        """
        Other than the numeric gauge, we are interested only
        in a couple of positions per gauge. We can manually 
        set the OCR dict to contain only the information we want
        """
        if self.gauge_type == "PUMP-ON-OFF":
            if not ("ON" in self.lookup and "OFF" in self.lookup):
                if "ON" in self.lookup:
                    temp = self.lookup["ON"]
                    self.lookup.clear()
                    self.lookup = {
                        "ON" : temp
                    }
                
                elif "OFF" in self.lookup:
                    temp = self.lookup["OFF"]
                    self.lookup.clear()
                    self.lookup = {
                        "OFF" : temp
                    }
            
            else:
                temp1 = self.lookup["OFF"]
                temp2 = self.lookup["ON"]
                self.lookup.clear()
                self.lookup = {
                    "OFF" : temp1,
                    "ON" : temp2
                }

        elif self.gauge_type == "HI-LO":
            if not ("HI" in self.lookup and "LO" in self.lookup):
                if "HI" in self.lookup:
                    temp = self.lookup["HI"]
                    self.lookup.clear()
                    self.lookup = {
                        "HI" : temp
                    }
                
                elif "LO" in self.lookup:
                    temp = self.lookup["LO"]
                    self.lookup.clear()
                    self.lookup = {
                        "LO" : temp
                    }
            
            else:
                temp1 = self.lookup["HI"]
                temp2 = self.lookup["LO"]
                self.lookup.clear()
                self.lookup = {
                    "HI" : temp1,
                    "LO" : temp2
                }

        elif self.gauge_type == "HIGH-LOW":
            if not ("HIGH" in self.lookup and "LOW" in self.lookup):
                if "HIGH" in self.lookup:
                    temp = self.lookup["HIGH"]
                    self.lookup.clear()
                    self.lookup = {
                        "HIGH" : temp
                    }
                
                elif "LOW" in self.lookup:
                    temp = self.lookup["LOW"]
                    self.lookup.clear()
                    self.lookup = {
                        "LOW" : temp
                    }
            
            else:
                temp1 = self.lookup["HIGH"]
                temp2 = self.lookup["LOW"]
                self.lookup.clear()
                self.lookup = {
                    "HIGH" : temp1,
                    "LOW" : temp2
                }
        
        elif self.gauge_type == "MIN-MAX":
            if not ("MIN" in self.lookup and "MAX" in self.lookup):
                if "MIN" in self.lookup:
                    temp = self.lookup["MIN"]
                    self.lookup.clear()
                    self.lookup = {
                        "MIN" : temp
                    }
                
                elif "MAX" in self.lookup:
                    temp = self.lookup["MAX"]
                    self.lookup.clear()
                    self.lookup = {
                        "MAX" : temp
                    }
            
            else:
                temp1 = self.lookup["MIN"]
                temp2 = self.lookup["MAX"]
                self.lookup.clear()
                self.lookup = {
                    "MIN" : temp1,
                    "MAX" : temp2
                }
        
        else:
            remove_list = []
            for text, _ in self.lookup.items():
                if not text.isnumeric() or text == "6":
                    remove_list.append(text)
            
            for key in remove_list:
                self.lookup.pop(key)

        return;

    def _compute_poly_points(self) -> tuple:
        """
        Gets the points to fit the polynomial based on the quadrant of the number
        (Right now only gets the centroids of all the detected numbers)
        """
        x = []; y = []
        for _, box in self.lookup.items():
            (tl, _, br, _) = box
            centroid_of_bb = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
            x.append(centroid_of_bb[0])
            y.append(centroid_of_bb[1])

        return (np.array(x), np.array(y))

    def _visualize(self, image : np.ndarray) -> None:
        """
        Visualize OCR bounding boxes
        """
        for text, box in self.lookup.items():
            (tl, tr, br, bl) = box
            try:
                ## Rectangle plotting (normal and tilted)
                cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

                cv2.putText(image, text, (tl[0], tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            
            except cv2.error:
                continue
        
        cv2.imshow("image",image)
        cv2.waitKey(0)
        return;

def main() -> None:
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

    ocr = Ocr()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.std() < 70:
        gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 5)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))

    ocr._run_ocr(tophat)
    print(ocr.gauge_type)
    ocr._visualize(image)

    return;

if __name__ == "__main__":
    main()
