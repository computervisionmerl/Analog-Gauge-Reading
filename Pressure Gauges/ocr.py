import cv2 
import numpy as np
import easyocr

from dataclasses import dataclass
from helper import *
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

@dataclass
class box_object:
    box : list
    centroid : tuple
    conf : tuple

@dataclass
class params:
    center : tuple
    theta : float
    A : float
    B : float

class Ocr:
    def __init__(self) -> None:
        self._reader = easyocr.Reader(['en'],gpu=True,verbose=False)
        self.lookup = {}
        self.inside = {}
        self.outside = {}
        self.reset()
    
    def reset(self) -> None:
        self.lookup.clear()
        self.inside.clear()
        self.outside.clear()
        self.__conf = 0.5

    def construct_initial_lookup(self, image : np.ndarray) -> None:
        """
        Constructs an initial dictionary with recognized numbers and their corresponding bounding box
        information. The image input is preferred to be tophat / blackhat transformed image for better
        and more accurate results from the OCR
        """
        boxes = self._reader.readtext(image.copy())
        if boxes:
            for box, text, conf in boxes:
                if conf > self.__conf:
                    (tl, tr, br, bl) = box
                    tl = (int(tl[0]), int(tl[1]))
                    tr = (int(tr[0]), int(tr[1]))
                    bl = (int(bl[0]), int(bl[1]))
                    br = (int(br[0]), int(br[1]))
                    centroid = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)

                    try:
                        int(text)
                        if text in self.lookup and self.lookup[text].conf < conf:
                            self.lookup[text] = box_object([tl, tr, br, bl], centroid, conf)
                        
                        elif text not in self.lookup:
                            self.lookup[text] = box_object([tl, tr, br, bl], centroid, conf)
                        
                        else :
                            continue

                    except (KeyError, ValueError) as e:
                        print(str(e) + "in construct_initial_lookup() method")
                        continue

    def separate_scales(self, ellipse : params) -> None:
        """
        Separates the scales of the identified numbers based whether the centroid of the number is inside or
        outside the recognized ellipse
        """
        (h,k) = ellipse.center
        (a,b) = (ellipse.A, ellipse.B)

        for text, obj in self.lookup.items():
            (cx, cy) = obj.centroid
            X = (cx - h)**2 / a**2
            Y = (cy - k)**2 / b**2

            if X+Y < 1:
                self.inside[text] = obj

            elif X+Y > 1:
                self.outside[text] = obj

        return;

    def compute_gauge_scale_and_filter(self, mode : str = "in") -> None:
        """
        Computes the possible scale of the gauge using the values detected by the OCR, also
        filters the numbers based on this scale. This also returs the sorted lookup dictionary
        Ex :- If the detected numbers are [0,1,50,100,200,400,500]
        Scale = 50; Numbers Included after filtering = [0,50,100,200,400,500]. The "1" detected
        is excluded since it is neither a factor, nor a multiple of the scale
        """
        if mode == "in":
            key_list = [int(key) for key in self.inside.keys()]
        else:
            key_list = [int(key) for key in self.outside.keys()]

        key_list.sort()
        diff_list = [abs(key_list[i] - key_list[i+1]) for i in range(len(key_list) - 1)]
        try:
            scale = Counter(diff_list).most_common(1)[0][0]
        except:
            return;

        ## Filter the values based on scale
        diff = abs(key_list[-1] - key_list[-2])
        good_keys = [
            str(key_list[i]) for i in range(len(key_list)-1) if \
            (abs(key_list[i]-key_list[i+1]) == scale) or \
            (abs(key_list[i]-key_list[i+1]) % scale == 0) or \
            (scale % abs(key_list[i]-key_list[i+1]) == 0)
        ]
        if diff == scale or diff % scale == 0 or scale % diff == 0: 
            good_keys.append(str(key_list[-1]))
            
        if mode == "in":
            lookup_clone = self.inside.copy()
            self.inside.clear()
            for key in good_keys:
                try:
                    self.inside[key] = lookup_clone[key]
                except KeyError:
                    continue
        else:
            lookup_clone = self.outside.copy()
            self.outside.clear()
            for key in good_keys:
                try:
                    self.outside[key] = lookup_clone[key]
                except KeyError:
                    continue

        return;

    def visualize(self, image : np.ndarray) -> None:
        for text, obj in self.lookup.items():
            [tl, tr, br, bl] = obj.box
            try:
                ## Rectangle plotting (normal and tilted)
                cv2.line(image, tl, tr, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, tr, br, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, br, bl, (0,255,0), 2, cv2.LINE_AA)
                cv2.line(image, bl, tl, (0,255,0), 2, cv2.LINE_AA)

                cv2.putText(image, text, (tl[0] - 25, tl[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            except cv2.error:
                continue

        cv2.imshow("image", cv2.resize(image, (600,600), cv2.INTER_CUBIC))
        cv2.waitKey(0)
        return;

def main():
    image = cv2.resize(cv2.imread("substation_images/ferguson_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/proflo_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/mitsubishi_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/negative_pressure_gauge.jpg"), (800,800), cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/trafag_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)

    ocr = Ocr()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.std() > 70 or gray.std() < 35:
        gray = cv2.equalizeHist(gray)
    
    blur = cv2.GaussianBlur(gray, (5,5), 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35,35))
    if calculate_brightness(image) > 0.52:
        hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)
    else:
        hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, kernel)

    ocr.construct_initial_lookup(hat)
    ocr.visualize(image)

if __name__ == "__main__":
    main()