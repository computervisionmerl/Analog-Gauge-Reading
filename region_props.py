import cv2 
import numpy as np
from skimage import measure
from helper import *

def run_regionprops(image : np.array, lookup : dict, ocr_mask : np.array) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.std() > 73: # or gray.std() < 35:
        gray = cv2.equalizeHist(gray)

    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, (35,35)))
    opening = cv2.morphologyEx(tophat, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)))
    bw_img = cv2.threshold(closing, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
   
    labels = measure.label(bw_img, connectivity=2)
    properties = measure.regionprops(labels, cache=False)
    ellipse_points = []
    for prop in properties:
        if prop.area > 200:
            points = prop.coords
            contour = np.hstack((points[:,1].reshape(-1,1), points[:,0].reshape(-1,1)))

            rect = cv2.minAreaRect(contour)
            (x,y), (_,_), _ = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)  

            if prop.area / cv2.contourArea(box) > 0.8:
                cv2.drawContours(image, [box], -1, (0,255,0), 2)
                ellipse_points.append((x,y))
    
    pairs = pair_numbers_with_ticks(ellipse_points, lookup, (gray.shape[0]//2, gray.shape[1]//2))
    return ellipse_points, pairs


def pair_numbers_with_ticks(good_contours : np.array, lookup : dict, image_center : tuple) -> dict:
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
