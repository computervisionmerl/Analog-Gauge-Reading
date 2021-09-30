import numpy as np
import cv2
import easyocr
from dataclasses import dataclass
from collections import Counter
from helper import *

import warnings
warnings.filterwarnings("ignore")

@dataclass
class box_object:
    box : list
    centroid : tuple
    conf : float

class Ocr:
    def __init__(self) -> None:
        self._reader = easyocr.Reader(['en'], gpu=True, verbose=False)
        self.lookup = dict()
        self.reset()
    
    def reset(self):
        self.__conf = 0.98
        self.lookup.clear()
        self.kmorph = (35,35)
        self.sigblur = 5
        self.kblur = (5,5)

    def _preprocessing(self, image : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Preprocessing = Image denoising + Contrast enhancement + Morphological transforms
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.std() > 70 or gray.std() < 35: 
            gray = cv2.equalizeHist(gray)
        
        blur = cv2.GaussianBlur(gray, self.kblur, self.sigblur)
        if calculate_brightness(image) > 0.575: #0.52:
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.kmorph))
        else:
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.kmorph))

        return hat

    def __construct_initial_lookup(self, image : np.ndarray) -> None:
        """
        Constructs an initial lookup dictionary. Need additional filtering based on the
        detected values, positions of numbers in the image, etc, to make the OCR output 
        more reliable
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

                    except (KeyError, ValueError):
                        continue
        return;

    def __compute_gauge_scale_and_filter(self) -> None:
        """
        Computes the possible scale of the gauge using the values detected by the OCR, also
        filters the numbers based on this scale. This also returs the sorted lookup dictionary
        Ex :- If the detected numbers are [0,1,50,100,200,400,500]
        Scale = 50; Numbers Included after filtering = [0,50,100,200,400,500]. The "1" detected
        is excluded since it is neither a factor, nor a multiple of the scale
        """
        key_list = [int(key) for key in self.lookup.keys()]
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
        if diff == scale or diff % scale == 0 or scale % diff == 0: good_keys.append(str(key_list[-1]))
        lookup_clone = self.lookup.copy()
        self.lookup.clear()
        for key in good_keys:
            try:
                self.lookup[key] = lookup_clone[key]
            except KeyError:
                continue
        return;

    def filter_numbers_based_on_position(self, center : tuple) -> str:
        """
        This simultaneously tries to determine the swing of the needle as well as the sign
        and legitimacy of these predicted numbers. This is based on angular positions of 
        these numbers with respect to the negative y axis. Modifies the lookup dictionary
        by modifying negative numbers and removing wrongly detected values
        """
        numbers = [int(i) for i in self.lookup.keys()]
        obj_list = [obj for obj in self.lookup.values()]
        
        angle_prev, angle_dict = None, {}
        clockwise_counter, anticlockwise_counter = 0, 0
        for i in range(len(numbers)):
            quad_num = find_quadrant(obj_list[i].centroid, center)
            angle_num = find_angle_based_on_quad(quad_num, obj_list[i].centroid, center)
            if angle_prev is not None:
                if angle_num > angle_prev:
                    clockwise_counter += 1
                elif angle_num < angle_prev : 
                    anticlockwise_counter += 1
                else:
                    raise ValueError("Angles of 2 numbers on the gauge cannot be the same")
            
            if i == 0:
                angle_min = angle_num

            if i == len(numbers) - 1:
                angle_max = angle_num

            angle_prev = angle_num
            angle_dict[angle_num] = numbers[i]

        ## If the needle moves clockwise
        if clockwise_counter > anticlockwise_counter:
            negative = [num for angle, num in angle_dict.items() if angle < angle_min]
            wrong = [num for angle, num in angle_dict.items() if angle > angle_max and angle_max > angle_min]
            swing = "clockwise"

        elif anticlockwise_counter > clockwise_counter:
            negative = [num for angle, num in angle_dict.items() if angle > angle_min]
            wrong = [num for angle, num in angle_dict.items() if angle < angle_max and angle_max > angle_min]
            swing = "anticlockwise"

        else :
            print("clockwise_count = {}".format(clockwise_counter))
            print("anticlockwise_count = {}".format(anticlockwise_counter))
            return None

        lookup_clone = {}
        for num, obj in self.lookup.items():
            try:
                if int(num) not in wrong:
                    if int(num) in negative:
                        lookup_clone["-"+str(num)] = obj
                    else:
                        lookup_clone[str(num)] = obj
            except (ValueError, KeyError):
                continue
        self.lookup = lookup_clone.copy()
        return swing
        
    def run_ocr(self, image : np.ndarray) -> None:
        """
        Wrapper method that runs the OCR and filters the values based on scale of 
        the gauge. The next step of filtering based on the position requires the 
        information of pivot of the needle (can be computed in the pipeline or can
        be fed from an external algorithm)
        """
        if len(image.shape) > 2:
            image = self._preprocessing(image)
        
        self.__construct_initial_lookup(image.copy())
        self.__compute_gauge_scale_and_filter()
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

                cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            
            except cv2.error:
                continue

        cv2.imshow("image", image)
        cv2.waitKey(0)
        return;

def main():
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gasvolume_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidtemp_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/thyoda_actual_gauge.jpg"),(800,800),cv2.INTER_CUBIC)

    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4761.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4762.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4763.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4764.jpg"),(800,800),cv2.INTER_CUBIC)

    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4765.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4766.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4767.jpg"),(800,800),cv2.INTER_CUBIC)

    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4807.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4808.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4809.jpg"),(800,800),cv2.INTER_CUBIC)

    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4801.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4802.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4803.jpg"),(800,800),cv2.INTER_CUBIC)

    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4804.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4805.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("number_gauge_test/IMG_4806.jpg"),(800,800),cv2.INTER_CUBIC)

    ocr = Ocr()
    ocr.run_ocr(image)
    ocr.visualize(image)
    
if __name__ == "__main__":
    main()