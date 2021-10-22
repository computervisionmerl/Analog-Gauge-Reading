import cv2
import numpy as np
from skimage import measure
from helper import *
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class tick_object:
    centroid : tuple
    tick_area : float
    bbox_area : float

class Regionprops:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.kmorph = (35,35)
        self.kopen = (5,5)
        self.kclose = (5,5)

    def __preprocessing(self, image : np.ndarray) -> np.ndarray:
        """
        Preprocessing --> Grayscale + Noise removal + Tophat / Blackhat + Binary thresholding
        """
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if gray.std() > 70 or gray.std() < 35:
            gray = cv2.equalizeHist(gray)

        if calculate_brightness(image) > 0.575:
            hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.kmorph))
        else:
            hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.kmorph))

        opening = cv2.morphologyEx(hat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, self.kopen))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, self.kclose))
        thresh = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        return thresh

    def get_tick_marks(self, image : np.ndarray, area_thresh : int = 200, ratio_thresh : float = 0.95) -> tuple[list, np.ndarray]:
        """
        Filter out only the rectangular regions based on ratio of pixel area and area of the min enclosing 
        rectangle. The rectangular regions which are at a definable position with respect to the numbers and
        satisfy certain conditions of area and aspect ratio are saved as tick mark objects
        """
        bw_img = self.__preprocessing(image)
        labels = measure.label(bw_img, connectivity=2)
        properties = measure.regionprops(labels, cache=False)
        tick_points = []
        for prop in properties:
            if prop.area > area_thresh:
                points = prop.coords
                contour = np.hstack((points[:,1].reshape(-1,1), points[:,0].reshape(-1,1)))

                rect = cv2.minAreaRect(contour)
                (x,y), (_,_), _ = rect
                box = cv2.boxPoints(rect)
                box = np.int0(box)  

                if prop.area / cv2.contourArea(box) > ratio_thresh:
                    tick_points.append(tick_object((x,y), prop.area, cv2.contourArea(box)))
        
        return tick_points, labels

    @staticmethod
    def pair_numbers_with_ticks(ticks : np.array, lookup : dict, center : tuple) -> dict:
        """
        Pairs the major ticks with the nearest numbers. There are some check conditions to 
        be passed before the match can be accepted. The center in this case is the pivot of
        the needle. If couple of tick marks are almost at similar distance from the number
        the one with higher area is chosen because that is most probable to be a major tick
        """
        pairs = {}
        distance_dict = {}
        for number, obj in lookup.items():
            distance_dict.clear()
            (tl, _, br, _) = obj.box
            number_centroid = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
            for tick in ticks:
                dist = euclidean_dist(number_centroid, tick.centroid)
                if dist < 200 and euclidean_dist(number_centroid, center) < euclidean_dist(tick.centroid, center):
                    distance_dict[dist] = tick
            
            distance_dict = dict(sorted(distance_dict.items())[0:3])
            dist = list(distance_dict.keys())
            vals = list(distance_dict.values())
            area = []
            for val in vals:
                area.append(val.tick_area)
            
            try:
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

    props = Regionprops()
    ticks, label = props.get_tick_marks(image, 200, 0.85)

    plt.figure(figsize=(16,12))
    plt.subplot(121); plt.imshow(label)

    plt.subplot(122)
    plt.imshow(image)
    plt.plot([pt.centroid[0] for pt in ticks], [pt.centroid[1] for pt in ticks], 'b+', markersize=15)
    plt.show()
    
if __name__ == "__main__":
    main()