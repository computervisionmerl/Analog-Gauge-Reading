import time
import cv2
import numpy as np

from helper import *
from ocr import *
from needle import Needle
from ellipse_detection import Ellipse_dlsq, params
from skimage.morphology import skeletonize

class Gauge_pressure(object):
    def __init__(self) -> None:
        super().__init__()
        self.ocr = Ocr()
        self.ell = Ellipse_dlsq()
        self.needle = Needle()
        self.lookup = dict()
        self.reset()

    def reset(self) -> None:
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (24,24))
        self.ocr.reset()
        self.lookup.clear()
        self.val = None

    def __preprocessing(self, image : np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Image preprocessing
        OCR --> Histogram equalization Gaussian blur + Tophat / Blackhat
        Needle --> Histogram equalization Gaussian blur + Tophat / Blackhat + Thresholding + Edge detection
        Ellipse --> Histogram equalization Gaussian blur + Tophat / Blackhat + Thresholding + Skeletonize
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.std() > 70 or gray.std() < 35:
            gray = cv2.equalizeHist(gray)
        
        blur = cv2.GaussianBlur(gray, (5,5), 5)
        if calculate_brightness(image) > 0.52:
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, self.kernel)
        else:
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, self.kernel)
        thresh = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        thresh_float = np.array(thresh, dtype="float64") / 255.
        edge_map = np.array(skeletonize(thresh_float)*255., dtype="uint8")
        points = np.argwhere(edge_map != 0)

        return hat, thresh, points

    @staticmethod
    def compute_intersection_with_ellipse(box_center : tuple, ellipse : params) -> tuple:
        """
        Extend the line joining center of ellipse to the center of the number (given by OCR) radially
        outward and compute the point of intersection of that line segment with the ellipse
        """
        h,k = ellipse.center
        x,y = box_center
        a,b = ellipse.A, ellipse.B
        m = (y - k)/(x - h)
        c = k - m*h

        a_squared, b_squared, m_squared, h_squared = a**2, b**2, m**2, h**2
        K = a_squared*b_squared / (a_squared*m_squared + b_squared)
        A, B, C = 1, -2*h, h_squared - K
        det = B**2 - 4*A*C

        x1 = (-B + math.sqrt(det)) / 2; y1 = m*x1 + c
        x2 = (-B - math.sqrt(det)) / 2; y2 = m*x2 + c
        return (x1,y1) if euclidean_dist((x1,y1),(x,y)) <= euclidean_dist((x2,y2),(x,y)) else (x2,y2)

    @staticmethod
    def find_ellipse_arclength(start : tuple, stop : tuple, ellipse : params, n_steps : int = 1000) -> float:
        """
        Computes the arclength along the given ellipse from the start to the stop points. This takes into 
        consideration the parametric representation of a point on the elliptical circumference.
        """
        quad_start = find_quadrant(start, ellipse.center)
        angle_start = find_angle_based_on_quad(quad_start, start, ellipse.center)
        quad_stop = find_quadrant(stop, ellipse.center)
        angle_stop = find_angle_based_on_quad(quad_stop, stop, ellipse.center)
        a,b = ellipse.A, ellipse.B

        theta = min(angle_start, angle_stop)
        del_theta = abs(angle_stop - angle_start) / n_steps
        length = 0
        while theta <= max(angle_stop, angle_start):
            pt1 = (a*math.cos(theta), b*math.sin(theta))
            pt2 = (a*math.cos(theta+del_theta), b*math.sin(theta+del_theta))
            length += euclidean_dist(pt1, pt2)
            theta += del_theta
        
        return length

    @staticmethod
    def get_direction_from_closest(closest : tuple, needle_point : tuple, center : tuple) -> str:
        """
        Computes the direction of the needle from the nearest known entity. This is the point at which a known 
        number is projected onto the ellipse based on the compute_intersection_with_ellipse() method specified above
        """
        quad_needle = find_quadrant(needle_point, center)
        quad_close = find_quadrant(closest, center)
        quad_dict = {4 : [1,2,3], 1 : [2,3], 2 : [3]}
        
        if quad_needle != quad_close:
            try:
                if quad_needle in quad_dict[quad_close]:
                    return "left"
                else:
                    return "right"
            except KeyError:
                pass
        
        if quad_needle == quad_close or quad_close == 3:
            if quad_needle in [4,1,2] and quad_close == 3:
                return "right"

            else:
                if quad_needle == 1 or quad_needle == 2:
                    if needle_point[0] < closest[0]:
                        return "left"
                    else:
                        return "right"
                
                if quad_needle == 3 or quad_needle == 4:
                    if needle_point[0] > closest[0]:
                        return "left"
                    else:
                        return "right"
        
        return None

    def read_gauge(self, image : np.ndarray, visualize : bool = True) -> None:
        """
        Wrapper function that runs the entire gauge reading right from OCR, needle estimation to ellipse detection
        and projecting all number locations and needle location to the ellipse and finally interpolating from the 
        nearest known location to compute the value read by the gauge.
        """
        start = time.time()
        ## Preprocessing, OCR and needle extraction
        hat, thresh, points = self.__preprocessing(image.copy())
        self.ocr.construct_initial_lookup(hat.copy())
        self.needle.isolate_needle(cv2.Canny(thresh, 85, 255))
        self.norm_x, self.norm_y = hat.shape

        ## Detect ellipse and separate the scales in the gauge
        self.ell.set_data(points)
        self.ell.fit_ellipse(points_per_sample=1500, n_iter=100)
        eigenvector = self.ell.cluster_eigenvecs().mean(axis=0)
        ellipse = self.ell.calculate_ellipse_params(eigenvector)
        self.ocr.separate_scales(ellipse)
        if len(self.ocr.inside) >= len(self.ocr.outside):
            self.lookup = self.ocr.inside.copy()
        else:
            self.lookup = self.ocr.outside.copy()

        ## Filter numbers by position, interpolate needle from nearest number along ellipse
        x1, y1, x2, y2 = self.needle.needle
        tip = (x1, y1) if euclidean_dist((x1,y1),ellipse.center) > euclidean_dist((x2,y2),ellipse.center) else (x2,y2)
        needle_intersection = self.compute_intersection_with_ellipse(tip, ellipse)
        ell_intersection = dict()
        for num, obj in self.lookup.items():
            point = self.compute_intersection_with_ellipse(obj.centroid, ellipse)
            ell_intersection[point] = int(num)
        
        distance_dict = dict()
        for point in ell_intersection.keys():
            dist = euclidean_dist(point, needle_intersection)
            distance_dict[dist] = point
        distance_dict = dict(sorted(distance_dict.items()))
        points = list(distance_dict.values())

        try:
            reference = self.find_ellipse_arclength(points[0], points[1], ellipse)
            distance = self.find_ellipse_arclength(needle_intersection, points[0], ellipse)
            calibrate = abs(ell_intersection[points[1]] - ell_intersection[points[0]])
            direction = self.get_direction_from_closest(points[0], needle_intersection, ellipse.center)
            delta_val = calibrate * distance / reference

            if direction == "left":
                val = ell_intersection[points[0]] - delta_val
            elif direction == "right":
                val = ell_intersection[points[0]] + delta_val
            else :
                raise ValueError("Incorrect direction, maybe NoneType")

            print("Gauge value = ", val)

        except IndexError:
            print("Index error with the points. Either fault of OCR or scale separtion")

        print("Time taken = {:4.4f} secs".format(time.time() - start))
        if visualize:
            self.visualize(image, ellipse)

    def visualize(self, image : np.ndarray, ellipse : params = None) -> None:
        if self.needle.needle is not None:
            l = self.needle.needle
            cv2.line(image, (l[0], l[1]), (l[2],l[3]), (0,255,255), 3)

        if ellipse is not None:
            center = (int(ellipse.center[0]), int(ellipse.center[1]))
            axes = (int(ellipse.A), int(ellipse.B))
            angle = int(math.degrees(ellipse.theta))
            cv2.ellipse(image, center, axes, angle, 0, 360, (0,0,255), 2)
            cv2.circle(image, center, 3, (0,0,255), -1)

        for text, obj in self.lookup.items():
            [tl, tr, br, bl] = obj.box
            try:
                ## Rectangle plotting
                cv2.line(image, tl, tr, (128,128,0), 2, cv2.LINE_AA)
                cv2.line(image, tr, br, (128,128,0), 2, cv2.LINE_AA)
                cv2.line(image, br, bl, (128,128,0), 2, cv2.LINE_AA)
                cv2.line(image, bl, tl, (128,128,0), 2, cv2.LINE_AA)
                ## Text with respect to the bounding box
                cv2.putText(image, text, (tl[0]-50, tl[1]+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            
            except cv2.error:
                continue

        cv2.imshow("image", image)
        cv2.waitKey(0)

def main(idx : int):
    if idx == 0:
        image = cv2.resize(cv2.imread("substation_images/ferguson_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    elif idx == 1:
        image = cv2.resize(cv2.imread("substation_images/proflo_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    elif idx == 2:
        image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    elif idx == 3:
        image = cv2.resize(cv2.imread("substation_images/mitsubishi_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 4:
        image = cv2.resize(cv2.imread("substation_images/trafag_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 5:
        image = cv2.resize(cv2.imread("substation_images/negative_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    
    elif idx == 6:
        image = cv2.resize(cv2.imread("mounted pressure gauges/img-2.jpeg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 7:
        image = cv2.resize(cv2.imread("mounted pressure gauges/img-8.jpeg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 8:
        image = cv2.resize(cv2.imread("mounted pressure gauges/img-11.jpeg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 9:
        image = cv2.resize(cv2.imread("mounted pressure gauges/img-16.jpeg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 10:
        image = cv2.resize(cv2.imread("mounted pressure gauges/img-23.jpeg"),(800,800),cv2.INTER_CUBIC)
    elif idx == 11:
        image = cv2.resize(cv2.imread("mounted pressure gauges/img-25.jpeg"),(800,800),cv2.INTER_CUBIC)

    else:
        print ("Invalid idx value in the main function. Must be 0-4")
        return;

    gauge = Gauge_pressure()
    gauge.read_gauge(image)

if __name__ == "__main__":
    main(1)