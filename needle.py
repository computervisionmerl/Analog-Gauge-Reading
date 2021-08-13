import cv2
import numpy as np
from helper import *

class Needle:
    def __init__(self) -> None:
        ## Hough transform parameters
        self.rho = 1
        self.theta = np.pi / 180
        self.thresh = 120
        self.srn = 2
        self.stn = 10

    def _isolate_needle(self, image : np.array) -> tuple:
        """
        Hough transformation + longest hough line estimation
        to isolate the needle. Returns the 2 end points of the needle
        """
        line = None
        self.w, self.h = image.shape
        linesP = cv2.HoughLinesP(image, self.rho, self.theta, self.thresh, None, self.srn, self.stn)
        max_length = 0
        if linesP is not None:
            for i in range(len(linesP)):
                l = linesP[i][0]
                length = euclidean_dist((l[0],l[1]), (l[2],l[3]))
                if length > max_length :
                    max_length = length
                    line = l
            x_line = np.array([line[0], line[2]]); y_line = np.array([line[1], line[3]])
            return (x_line, y_line)

        else:
            raise ValueError("No hough lines detected !!")

    def _visualize(self, image : np.array, x_line: np.array, y_line: np.array) -> None:
        if x_line is not None and y_line is not None:
            cv2.line(image, (x_line[0],y_line[0]), (x_line[1],y_line[1]), (255,0,0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)
        return;

def main() -> None:
    image = cv2.resize(cv2.imread("gauge images/spot_ptz_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gaspressure_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_liquidsmall_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_transformer_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/qualitrol_winding_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/thyoda_actual_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_gasvolume_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_lowhigh_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_hilo_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_liquidtemp_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_minmax_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("gauge images/spot_ptz_pump_gauge.jpg"), (800,800), interpolation=cv2.INTER_CUBIC)

    needle = Needle()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 85,255)
    (x_line, y_line) = needle._isolate_needle(canny)
    needle._visualize(image, x_line, y_line)

if __name__ == "__main__":
    main()
