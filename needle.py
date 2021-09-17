import cv2
import numpy as np
from helper import *

class Needle:
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.line_white = None
        self.line_red = None
        self.needle_pivot = None
        self.canny_thresh = 85
        self.hat_kernel_size = (35,35)
        self.circles = None

    def __pre_processing(self, image : np.ndarray) -> np.ndarray:
        """
        White --> Contrast enhancement + Denoising + Morphological transforms + Edge detection
        Red --> HSV conversion + Red Color Masking + Morphological transforms
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.std() > 70 or gray.std() < 35:
            gray = cv2.equalizeHist(gray)
        
        blur = cv2.GaussianBlur(gray, (3,3), 3)
        if calculate_brightness(image) > 0.575:
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.hat_kernel_size))
        else:
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, self.hat_kernel_size))
        hat_white = cv2.Canny(hat, self.canny_thresh, 255)

        ## For red needle
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,50,50), (5,255,255))
        mask2 = cv2.inRange(hsv, (165,50,50), (180,255,255))
        red = cv2.bitwise_or(mask1, mask2)
        masked = cv2.bitwise_and(blur, blur, mask = red)
        if calculate_brightness(image) > 0.58:
            hat = cv2.morphologyEx(masked, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, self.hat_kernel_size))
        else:
            hat = cv2.morphologyEx(masked, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_CROSS, self.hat_kernel_size))
        hat_red = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

        return hat_white, hat_red

    def isolate_needle(self, image : np.ndarray, color : str = "white") -> None:
        """
        Isolates the needle depending on the parameter "color". Each of the needles demands unique
        masking for efficient detection. Based on the color, we can detect the needle using efficient
        pre-processing. The longest hough line is assumed to be the needle (true in most cases).
        """
        if len(image.shape) > 2:
            hat_white, hat_red = self.__pre_processing(image)
        
        if color == "white":
            linesP = cv2.HoughLinesP(hat_white, 1, np.pi/180, 120, None, 2, 10)
        elif color == "red":
            linesP = cv2.HoughLinesP(hat_red, 1, np.pi/180, 120, None, 2, 10)
        else:
            raise ValueError("Needle must be red or white")
        
        if linesP is not None:
            max_length = 1e-20
            for i in range(len(linesP)):
                l = linesP[i][0]
                length = np.sqrt((l[0]-l[2])**2 + (l[1]-l[3])**2)
                if length > max_length:
                    max_length = length
                    if color == "white":
                        self.line_white = l
                    elif color == "red":
                        self.line_red = l

        if color == "white":
            self.circles = cv2.HoughCircles(hat_white, cv2.HOUGH_GRADIENT, 1, 20, None, 40, 12, 0, 300)
        elif color == "red":
            self.circles = cv2.HoughCircles(hat_red, cv2.HOUGH_GRADIENT, 1, 20, None, 40, 12, 0, 300)

        if self.circles is not None:
            self.circles = np.uint16(np.around(self.circles))     
        return;

    def visualize(self, image : np.array) -> None:
        if self.line_white is not None:
            cv2.line(image, (self.line_white[0], self.line_white[1]), (self.line_white[2], self.line_white[3]), (0,255,0), 2)
        
        if self.line_red is not None:
            cv2.line(image, (self.line_red[0], self.line_red[1]), (self.line_red[2], self.line_red[3]), (255,0,0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)

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

    needle = Needle()
    needle.isolate_needle(image)
    needle.visualize(image)
    
if __name__ == "__main__":
    main()