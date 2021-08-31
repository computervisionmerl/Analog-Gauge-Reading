import numpy as np
import cv2
import easyocr
from dataclasses import dataclass
from helper import *
import operator
from typing import Tuple

import warnings
warnings.filterwarnings("ignore")

@dataclass
class ocr_result:
    box: tuple
    conf: float

class Ocr(object):
    def __init__(self) -> None:
        super().__init__()
        self.reader = easyocr.Reader(['en'], gpu=True, verbose=False) 
        self.lookup = dict()
        self.reset()

    def reset(self) -> None:
        self.conf = 0.95
        self.lookup.clear()

    @staticmethod
    def _pre_processing(image : np.array) -> np.ndarray:
        """
        Preprocessing = Image denoising + Contrast enhancement + Morphological transforms
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.std() > 66 or gray.std() < 60: 
            gray = cv2.equalizeHist(gray)
        
        blur = cv2.GaussianBlur(gray, (5,5), 5)
        if calculate_brightness(image) > 0.52:
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        else:
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))

        return hat

    @staticmethod
    def _classify(lookup : dict) -> Tuple[dict, str]:
        """
        Classifies based on the lookup dictionary constructed using the OCR results. If the gauge is not numeric, we are
        not interested in anything other than a couple of points of interest from the OCR standpoint
        """
        gauge_type = ""
        lookup_clone = lookup.copy(); lookup.clear()
        keys = lookup_clone.keys()
        if "PUMP" in keys or "ON" in keys or "OFF" in keys:
            gauge_type = "pump-on-off"
            for key in ["PUMP", "ON", "OFF"]:
                try:
                    lookup[key] = lookup_clone[key]
                except KeyError:
                    continue

        elif "MIN" in keys or "MAX" in keys:
            gauge_type = "min-max"
            for key in ["MIN", "MAX"]:
                try:
                    lookup[key] = lookup_clone[key]
                except KeyError:
                    continue

        elif "HIGH" in keys or "LOW" in keys:
            gauge_type = "high-low"
            for key in ["HIGH", "LOW"]:
                try:
                    lookup[key] = lookup_clone[key]
                except KeyError:
                    continue

        elif "HI" in keys or "LO" in keys:
            gauge_type = "hi-lo"
            for key in ["HI", "LO"]:
                try:
                    lookup[key] = lookup_clone[key]
                except KeyError:
                    continue

        else:
            gauge_type = "numeric"
            lookup = lookup_clone.copy()
        
        return lookup, gauge_type

    def _construct_initial_lookup(self, hat : np.array) -> list:
        """
        Fills the lookup dictionary with initial predictions. These predictions have to be filtered and corrected based
        on the type of the gauge, scale of calibration, etc.
        """
        boxes = self.reader.readtext(hat)
        key_list = []
        idx = 0
        if boxes:
            for box, text, conf in boxes:
                if conf > self.conf:
                    (tl, tr, br, bl) = box
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    br = (int(br[0]), int(br[1]))

                    try:
                        ## Recognized text is found with a higher confidence in the dictionary
                        if text in self.lookup and conf > self.lookup[text].conf:
                            self.lookup[text] = ocr_result((tl, tr, br, bl), conf)

                        ## Recognized text is found with a lower confidence in the dictionary
                        elif text in self.lookup and conf < self.lookup[text].conf:
                            self.lookup[text+"_"+str(idx)] = ocr_result((tl, tr, br, bl), conf)
                            idx+=1

                        ## Recognized text not found in the dictionary
                        else:
                            self.lookup[text] = ocr_result((tl, tr, br, bl), conf)

                    except ValueError:
                        continue

                    if text.isnumeric():
                        if int(text) not in key_list:
                            key_list.append(int(text))

        else:
            raise ValueError("No numbers detected by the OCR !!")
            
        return key_list       
    
    def _filter_values(self, key_list : list) -> list:
        """
        Filters the values in a numeric type gauge based on the most common scale in the dial
        Ex: - If the OCR detects the following numbers --> [0,5,8,100,200,400,500,600]
              Most common scale = 100 ==> Retained numbers --> [0,100,200,400,500,600]
        """
        diff_dict = dict()
        for i in range(len(key_list) - 1):
            diff = abs(key_list[i] - key_list[i+1])
            if not diff in diff_dict:
                diff_dict[diff] = 1
            else:
                diff_dict[diff] += 1

        try:
            good_numbers = []
            diff = max(diff_dict.items(), key=operator.itemgetter(1))[0]
            for i in range(0, len(key_list)):
                for j in range(i+1, len(key_list)):
                    curr_diff = key_list[i] - key_list[j] 
                    ## If the scale is most common one or a multiple / factor --> Consider the number
                    if curr_diff == diff or curr_diff % diff == 0 or diff % curr_diff == 0:
                        if key_list[i] not in good_numbers:
                            good_numbers.append(key_list[i])
                        if key_list[j] not in good_numbers:
                            good_numbers.append(key_list[j])
        except ValueError:
            return []

        return good_numbers

    def _run_ocr(self, hat : np.array) -> None:
        """
        Runs OCR, classifies the gauge based on recognized text, cleans up the lookup dictionary to remove any falsely
        recognized text and stores the dictionary as a class attribute
        """
        if len(hat.shape) > 2:
            hat = Ocr._pre_processing(hat)

        key_list = self._construct_initial_lookup(hat) 
        lookup, gauge_type = Ocr._classify(self.lookup.copy())

        if gauge_type == "numeric":
            good_numbers = self._filter_values(key_list)
            lookup_clone = self.lookup.copy(); self.lookup.clear()
            for num in good_numbers:
                self.lookup[str(num)] = lookup_clone[str(num)]

        else:
            self.lookup = lookup
        
        return;  

    def _visualize(self, image : np.array) -> None:
        for text, obj in self.lookup.items():
            (tl, tr, br, bl) = obj.box
            try:
                ## Rectangle plotting (normal and tilted)
                cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

                cv2.putText(image, text, (tl[0], tl[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            except cv2.error:
                continue

        cv2.imshow("image", image)
        cv2.waitKey(0)
        return;

def main():
    image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_negative_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gaspressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gasvolume_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidtemp_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/thyoda_actual_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_temp_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_gasvolume_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("substation_images/qualitrol_temperature_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("substation_images/meppi_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/meppi_kpa_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)

    ocr = Ocr()
    ocr._run_ocr(image)
    ocr._visualize(image)
    
if __name__ == "__main__":
    main()
