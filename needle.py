import numpy as np
import cv2
from helper import calculate_brightness, fit_line
import string
import matplotlib.pyplot as plt

class Needle(object):
    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self):
        self.line_red = None
        self.line_white = None

    @staticmethod
    def _pre_processing(image : np.array) -> np.array:
        """
        White --> Contrast enhancement + Denoising + Morphological transforms + Edge detection
        Red --> HSV conversion + Red Color Masking + Morphological transforms
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if gray.std() > 70 or gray.std() < 35:
            gray = cv2.equalizeHist(gray)
        
        blur = cv2.GaussianBlur(gray, (3,3), 3)
        if calculate_brightness(image) > 0.52:
            hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))
        else:
            hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))

        ## For red needle
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,50,50), (5,255,255))
        mask2 = cv2.inRange(hsv, (165,50,50), (180,255,255))
        red = cv2.bitwise_or(mask1, mask2)
        masked = cv2.bitwise_and(blur, blur, mask = red)
        if calculate_brightness(image) > 0.58:
            hat_red = cv2.morphologyEx(masked, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS,(35,35)))
        else:
            hat_red = cv2.morphologyEx(masked, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_CROSS,(35,35)))

        return cv2.Canny(hat, 85, 255), cv2.threshold(hat_red, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    def _isolate_needle(self, hat_white : np.array, hat_red : np.array = np.array([]), color : string = "white") -> None:
        """
        Isolates the needle depending on the parameter "color". Each of the needles demands unique
        masking for efficient detection. Based on the color, we can detect the needle using efficient
        pre-processing. The longest hough line is assumed to be the needle (true in most cases)
        """
        if len(hat_white.shape) > 2 or not hat_red.size:
            hat_white, hat_red = self._pre_processing(hat_white)
        
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
                    
        return;
            
    def _visualize(self, image : np.array) -> None:
        if self.line_white is not None:
            cv2.line(image, (self.line_white[0], self.line_white[1]), (self.line_white[2], self.line_white[3]), (0,255,0), 2)
        
        if self.line_red is not None:
            cv2.line(image, (self.line_red[0], self.line_red[1]), (self.line_red[2], self.line_red[3]), (255,0,0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)

def main():
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_negative_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/qualitrol_transformer_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidsmall_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gaspressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_gasvolume_gauge.jpg"),(800,800),cv2.INTER_CUBIC) # 0.65 --> ratio
    #image = cv2.resize(cv2.imread("ptz_gauge_images/spot_ptz_liquidtemp_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("ptz_gauge_images/thyoda_actual_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_temp_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_gasvolume_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/qualitrol_temperature_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/meppi_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/meppi_kpa_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)

    needle = Needle()
    needle._isolate_needle(image, color="white")
    #needle._visualize(image)

    plt.figure(figsize=(8,8))
    plt.imshow(image)
    x = [needle.line_white[0] / image.shape[0], needle.line_white[2] / image.shape[0]]
    y = [needle.line_white[1] / image.shape[1], needle.line_white[3] / image.shape[1]]
    print(x,y)
    line = np.poly1d(fit_line(x,y))
    x_plot = np.linspace(0,1,1000)
    y_plot = np.polyval(line, x_plot)
    plt.plot(x_plot*image.shape[0], y_plot*image.shape[1], 'red')
    plt.show()

if __name__ == "__main__":
    main()