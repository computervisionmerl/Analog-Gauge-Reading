import cv2
from math import sqrt
import numpy as np
from skimage import measure
from dataclasses import dataclass
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from helper import euclidean_dist

@dataclass
class tick_object:
    centroid : tuple
    tick_area : float
    bbox_area : float

class Rectangle_detector:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.thresh = 200
        self.kmorph = (35,35)
        self.kernel = (5,5)
        self.rs_thresh = 2 #10
        self.rl_thresh = 0.35 # 0.5

    def __preprocessing(self, image : np.ndarray) -> np.ndarray:
        """
        Preprocessing --> Denoising + morphological transforms + binary thresholding
        """
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        if gray.std() > 70 or gray.std() < 35:
            gray = cv2.equalizeHist(gray)
    
        blur = cv2.GaussianBlur(gray, (5,5), 5)
        hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.kmorph))
        opening = cv2.morphologyEx(hat, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, self.kernel))
        thresh = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        return thresh
    
    def __point_in_box(self, box : tuple, point : tuple) -> bool:
        """
        Checks if the point is inside the bounding box of the number. This is to ensure
        that the algorithm doesn't give a false pairing because the tick mark is definitely
        outside the bounding box of the number
        """
        (tl, tr, br, bl) = box
        if point[0] > max(tl[0], bl[0]) and point[0] < min(tr[0], br[0]):
            if point [1] > max(tl[1], tr[1]) and point[1] < min(bl[1], br[1]):
                return True
        
        return False

    def detect_rect(self, image : np.ndarray, visualize : bool = False) -> list:
        """
        Detects all the rectangles within a certain area range. These rectangles are
        tick marks more often than not since they are the only rectangular regions on
        the gauge

        Algorithm inspired by the paper "Fast Method for Rectangle Detection" by Cheng Wang.
        Please refer to the paper for the steps and explanation of the algorithm
        """
        thresh = self.__preprocessing(image)
        labels = measure.label(thresh, connectivity=2)
        props = measure.regionprops(labels, cache=False)

        tick_points = []
        for prop in props:
            if prop.area > 200:
                points = prop.coords
                points = np.hstack((points[:,1].reshape(-1,1), points[:,0].reshape(-1,1)))
                centroid = prop.centroid

                point_dist_map = {euclidean_dist(point, centroid) : point for point in points}
                point_dist_map = dict(sorted(point_dist_map.items()))
                distances = list(point_dist_map.keys())
                points_new = list(point_dist_map.values())

                min_value = 1e20
                min_point = points_new[0]
                for point in points_new:
                    try:
                        slope = (point[1] - centroid[1]) / (point[0] - centroid[0])
                        factor = (point[1] - min_point[1]) / (point[0] - min_point[0])
                        value = abs(slope*factor + 1)
                        if value < min_value:
                            min_value = value
                            b_dist = euclidean_dist(point, centroid)

                    except IndexError:
                        continue

                rs = abs((prop.area - distances[0]*b_dist)/prop.area)
                rl = abs((distances[-1] - sqrt(distances[0]**2 + b_dist ** 2))/distances[-1])
                if rs > self.rs_thresh:
                    isRectangle = False
                if rl < self.rl_thresh:
                    isRectangle = True
                else:
                    isRectangle = False

                if isRectangle:
                    centroid = (prop.centroid[1], prop.centroid[0])
                    ((x,y), (w,h), _) = cv2.minAreaRect(points)
                    tick_points.append(tick_object((x,y), prop.area, w*h))

        if visualize:
            plt.figure(figsize=(8,8))
            plt.imshow(labels, cmap='inferno')

            plt.figure(figsize=(8,8))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            plt.plot([pt.centroid[0] for pt in tick_points], [pt.centroid[1] for pt in tick_points], 'b+', markersize=12)
            plt.show()

        return tick_points

    def pair_ticks_with_numbers(self, ticks : list, lookup : dict, center : tuple) -> dict:
        """
        Pairs the major ticks with the nearest numbers. There are some check conditions to 
        be passed before the match can be accepted. The center in this case is the pivot of
        the needle. Conditions for matches :-
        (i) Area of the region 
        (ii) Whether or not the region is closer to the center than the number
        (iii) Whether or not the centroid of the region is inside the number bounding box
        (iv) If 2 regions have similar distances from number, pick the larger one
        """
        pairs = {}
        distance_dict = {}
        for number, obj in lookup.items():
            distance_dict.clear()
            (tl, _, br, _) = obj.box
            number_centroid = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
            for tick in ticks:
                dist = euclidean_dist(number_centroid, tick.centroid)
                if dist < self.thresh and euclidean_dist(number_centroid, center) < euclidean_dist(tick.centroid, center):
                    if not self.__point_in_box(obj.box, tick.centroid):
                        distance_dict[dist] = tick
            
            try:
                distance_dict = dict(sorted(distance_dict.items())[0:3])
                dist = list(distance_dict.keys())
                vals = list(distance_dict.values())
                area = []

                for val in vals:
                    area.append(val.tick_area)

                if dist[1] - dist[0] < 10:
                    if dist[2] - dist[1] <= 10:
                        max_area_index = area.index(max(area))
                    else:
                        max_area_index = area.index(max(area[0:2]))
                else:
                    max_area_index = 0
                
                key = dist[max_area_index]
                pairs[int(number)] = (distance_dict[key].centroid, number_centroid)

            except (IndexError, ValueError, KeyError):
                continue

        return pairs

    def pair_ticks_with_text(self, lookup: dict, ticks : list, center : tuple) -> dict:
        """
        This is primarily for the non numerical gauge types where the text is minimal
        and not many tick marks which can be paired with text. The optimization criterion is
        only euclidean ditance in this case
        """
        pairs = {}
        for text, obj in lookup.items():
            (tl,_,br,_) = obj.box
            numb_centroid = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
            min_dist = 1e20
            for tick_centroid in ticks:
                dist = euclidean_dist(tick_centroid, numb_centroid)
                dt = euclidean_dist(tick_centroid, center)
                dn = euclidean_dist(numb_centroid, center)
                if dist < min_dist and dt > dn:
                    if not self.__point_in_box(obj.box, tick_centroid):
                        min_dist = dist
                        pairs[text] = (tick_centroid, numb_centroid)
        return pairs

def main():
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gasvolume_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidtemp_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/thyoda_actual_gauge.jpg"),(800,800),cv2.INTER_CUBIC)

    image = cv2.resize(cv2.imread("number_gauge_test/IMG_4761.jpg"),(800,800),cv2.INTER_CUBIC)
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
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4806.jpg"),(800,800),cv2.INTER_CUBIC)

    rect = Rectangle_detector()
    rect.detect_rect(image, True)

if __name__ == "__main__":
    main()