from typing import Tuple
import cv2 
import numpy as np
from skimage import measure
from helper import *
import matplotlib.pyplot as plt

class Regionprops(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
    
    def reset(self):
        self.kmorph = (35,35)
        self.kopen = (5,5)
        self.kclose = (5,5)

    def _pre_processing(self, hat : np.array) -> np.array:
        """
        Preprocessing to isolate tick marks from the background. Returns a binary image which is 
        required to compute regionprops and extract out only rectangular regions (tick marks)
        """
        if len(hat.shape) > 2:
            gray = cv2.cvtColor(hat, cv2.COLOR_RGB2GRAY)
            if gray.std() > 70 or gray.std() < 35:
                gray = cv2.equalizeHist(gray)
            if calculate_brightness(hat) > 0.52:
                hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.kmorph))
            else:
                hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.kmorph))

        opening = cv2.morphologyEx(hat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, self.kopen))
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, self.kclose))
        return cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    def _get_tick_marks(self, image : np.array, area_thresh : int = 200, ratio_thresh : float = 0.8) -> Tuple[list, np.array]:
        """
        Filter out only the rectangular regions based on ratio of pixel area and area of the min enclosing 
        rectangle. The rectangular regions are the tick marks on the gauge (safe to assume that)
        """
        bw_img = self._pre_processing(image)
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
                    cv2.drawContours(image, [box], -1, (0,255,0), 2)
                    tick_points.append((x,y))
        
        return tick_points, labels

    @staticmethod
    def _pair_numbers_with_ticks(good_contours : np.array, lookup : dict, image_center : tuple) -> dict:
        """
        Pair the tick marks with the closest number based on distance between their centroids
        Some other criteria can be added to prevent wrong pairing
        """
        pairs = {}
        for number, obj in lookup.items():
            (tl, _, br, _) = obj.box
            centroid_of_bb = ((tl[0]+br[0])//2, (tl[1]+br[1])//2)
            min_dist = 1e20
            for centroid in good_contours:
                dist = euclidean_dist(centroid_of_bb, centroid)
                if dist < min_dist and euclidean_dist(centroid_of_bb, image_center) < euclidean_dist(centroid, image_center):
                    if number.isnumeric():
                        pairs[int(number)] = (tuple(centroid), centroid_of_bb)
                        min_dist = dist
                    else:
                        continue

        return dict(sorted(pairs.items()))

def main():
    image = cv2.resize(cv2.imread("number_gauge_test/IMG_4727.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4728.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4729.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4730.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4731.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4732.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4733.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4734.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4735.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4736.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("number_gauge_test/IMG_4737.jpg"),(800,800),cv2.INTER_CUBIC)

    bw_img = Regionprops._preprocessing(image)
    labels = measure.label(bw_img, connectivity=2)

    plt.figure(figsize=(16,8))
    plt.subplot(121); plt.imshow(bw_img, cmap='gray')
    plt.subplot(122); plt.imshow(labels, cmap='inferno')
    plt.show()

if __name__ == "__main__":
    main()