from cv2 import data
import numpy as np
import cv2
import easyocr
import string
from dataclasses import dataclass
from helper import *

import warnings
warnings.filterwarnings("ignore")

@dataclass
class ocr_result:
    box: tuple
    prob: float

class Ocr(object):
    def __init__(self) -> None:
        super().__init__()
        self.reader = easyocr.Reader(['en'], True) 
        self.lookup = dict()
        self.number_list = list()
        self.prob = 0.95
        self.reset()

    def reset(self):
        self.lookup.clear()
        self.number_list.clear()
        self.min_text = None
        self.max_text = None   
        self.mask = np.ones((1,1), dtype=np.uint8) * 255

    @staticmethod
    def _pre_processing(image : np.array) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if gray.std() < 70: # or gray.std() < 35:
            gray = cv2.equalizeHist(gray)
        blur = cv2.GaussianBlur(gray, (5,5), 5)

        if calculate_brightness(image) > 0.58:
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        else:
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        return hat

    def _run_ocr(self, hat : np.array) -> None:
        if len(hat.shape) > 2:
            hat = Ocr._pre_processing(hat)

        boxes = self.reader.readtext(hat)
        self.mask = np.ones(hat.shape, dtype=np.uint8) * 255
        idx = 0
        if boxes:
            for box, text, prob in boxes:
                if prob > self.prob:
                    (tl, tr, br, bl) = box
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    br = (int(br[0]), int(br[1]))

                    try:
                        if text in self.lookup and prob > self.lookup[text].prob:
                            self.lookup[text] = ocr_result((tl, tr, br, bl), prob)

                        elif text in self.lookup and prob < self.lookup[text].prob:
                            self.lookup[text+"_"+str(idx)] = ocr_result((tl, tr, br, bl), prob)
                            idx+=1

                        else:
                            self.lookup[text] = ocr_result((tl, tr, br, bl), prob)
                            self.number_list.append(int(text))

                        self.mask[tl[1]:bl[1], tl[0]:tr[0]] = 0

                    except ValueError:
                        continue

            self.number_list.sort()
            self.min_text = self.number_list[0]; self.max_text = self.number_list[-1]

        else:
            raise ValueError("No numbers detected by the OCR !!") 

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

        cv2.imshow("image",image)
        cv2.waitKey(0)
        return;

def main():
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_negative_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gaspressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gasvolume_gauge.jpg"),(800,800),cv2.INTER_CUBIC) # 0.65 --> ratio
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidtemp_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/thyoda_actual_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_temp_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_gasvolume_gauge.png"),(800,800),cv2.INTER_CUBIC)

    ocr = Ocr()
    ocr._run_ocr(image)
    ocr._visualize(image)

if __name__ == "__main__":
    main()
