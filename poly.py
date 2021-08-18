import cv2
import numpy as np
import matplotlib.pyplot as plt
from random import randint

from ocr import Ocr
from needle import Needle

class Poly:
    def __init__(self, hat : np.array, canny : np.array) -> None:
        """
        Instantiates a polynomial class to run polynomial fitting, and creates class
        instanes to perform OCR and needle estimation
        """
        self.ocr = Ocr()
        self.needle = Needle()

        ## Polynomials for curve (through centroids of ticks) and needle line
        self.curve = None
        self.line = None
        self.x_curve = None
        self.y_curve = None
        self.x_line = None
        self.y_line = None

        self._get_pre_requisites(hat, canny)

    def _get_pre_requisites(self, hat : np.array, canny : np.array) -> None:
        """
        Gets the points for the curve and line estimations
        """
        # Run OCR 
        self.ocr._run_ocr(hat)
        (self.x_curve, self.y_curve) = self.ocr._compute_poly_points()

        # Needle detection
        self.w, self.h = canny.shape
        (self.x_line, self.y_line) = self.needle._isolate_needle(canny)

    @staticmethod
    def _fit_curve(x : np.array, y : np.array) -> np.ndarray:
        """
        Fits a quadratic curve through a set of data points using
        least squares fitting
        """
        iterations = 0
        bestmodel = None
        besterr = 1e20
        while iterations < 50:
            x_fit = []; y_fit = []
            for _ in range(50): # 50
                random_idx = randint(0,len(x)-1)
                x_fit.append(x[random_idx]); y_fit.append(y[random_idx])
            
            poly = np.polyfit(x_fit, y_fit, 2)
            z_fit = np.polyval(poly, x_fit)
            err = np.sum((z_fit - y_fit)**2)
            if err < besterr:
                besterr = err
                bestmodel = poly

            iterations += 1
        
        return bestmodel

    @staticmethod
    def _fit_line(x : np.array, y : np.array) -> np.ndarray:
        """
        Line fitting using 2-point formula
        """
        return np.polyfit(x, y, 1)

    def _get_polynomial_coeffs(self):
        """
        Computes the coefficients of the polynomial computed and returns
        as a 1-D polynomial, i.e
        y = Ax^2 + Bx + C; y = Mx + K
        """
        # At this point, we have needle end points, polynomial curve points, first normalize points
        x_curve_norm = self.x_curve / self.w; y_curve_norm = self.y_curve / self.h
        x_line_norm = self.x_line / self.w; y_line_norm = self.y_line / self.h

        self.curve = np.poly1d(self._fit_curve(x_curve_norm, y_curve_norm))
        self.line = np.poly1d(self._fit_line(x_line_norm, y_line_norm))

    def _visualize(self, image : np.array) -> None:
        """
        Visualize needle, polynomial and the points used to compute the coefficients
        """
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.x_curve is not None and self.y_curve is not None:
            for x,y in zip(self.x_curve, self.y_curve):
                cv2.circle(image, (x,y), 5, (0,255,0), -1)

        ax = plt.axes()
        ax.imshow(image)
        x_plot = np.linspace(0.1,0.9,1000)
        if self.curve is not None:
            y_plot = np.polyval(self.curve, x_plot)
            ax.plot(x_plot * self.w, (y_plot * self.h) // 1.1, 'red', linewidth=2)
            ax.plot(x_plot * self.w, (y_plot * self.h) // 1.5, 'orange', linewidth=2)
        
        x_plot = np.linspace(0.4,0.45,1000)
        if self.line is not None:
            y_plot = np.polyval(self.line, x_plot)
            ax.plot(x_plot * self.w, y_plot * self.h, 'blue', linewidth=2)
        
        ax.plot(image.shape[0]/2, image.shape[1]/2,'red', linewidth=2)
        plt.show()
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

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 85,255)

    if gray.std() < 70:
        gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 5)
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (35,35)))

    poly = Poly(tophat, canny)
    poly._get_polynomial_coeffs()
    poly._visualize(image)

if __name__ == "__main__":
    main()
