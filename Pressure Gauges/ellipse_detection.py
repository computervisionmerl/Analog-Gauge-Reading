import time
import cv2
import numpy as np

from helper import *
from scipy.linalg import eig
from dataclasses import dataclass

@dataclass
class params:
    center : tuple
    theta : float
    A : float
    B : float

class Direct_least_sq_ellipse:
    def __init__(self, points : list) -> None:
        self.data_x = np.array(points[:,0])
        self.data_y = np.array(points[:,1])
        self.len = len(self.data_x)
        self.ellipse_list = list()

    def __del__(self):
        """
        Destructor to ensure the instance is deleted once it goes out of scope
        """
        print("Deleted")

    def fit_ellipse(self, points_per_sample : int = 6) -> list:
        """
        Constructs a generalized eigen system using scatter and 
        constraint matrices to try to estimate the 6 parameters
        of a quadratic conic section (ax^2 + bxy + cy^2 + dx + ey + f = 0)
        """
        n = 0
        cluster = list()
        while n < 100:
            x, y = list(), list()
            idx = np.random.permutation(self.len)
            for i in range(points_per_sample):
                x.append(self.data_x[idx[i]])
                y.append(self.data_y[idx[i]])
            
            x, y = np.array(x), np.array(y)
            # Design matrix
            D = np.array([x*x, x*y, y*y, x, y, np.ones((x.shape), dtype="int")], dtype="float64").T
            # Scatter matrix
            S = D.T @ D 
            # Constraint matrix
            C = np.zeros(S.shape, dtype="float64")
            C[0,2] = C[2,0] = -2; C[1,1] = 1

            eigvals, eigvecs = eig(S, C)
            eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
            # Since the system is +ve semi definite, only 1 negative eigenvalue
            idx = np.where(eigvals < 0)[0]
            try:
                cluster.append(eigvecs[:,idx])
            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                print(str(e) + " in the fit_ellipse() method")

            n += 100

        self.param_cluster = np.array(cluster).reshape(-1,6)

    def calculate_ellipse_params(self):
        """
        Recognizes the ellipse from a list of general conic section equations 
        and extracts computes the 5 required parameters namely center_x, center_y,
        semi major axis, semi minor axis and orientation
        """
        for (a,b,c,d,e,f) in self.param_cluster:
            det = 4*a*c - b*b
            if det > 0:
                try:
                    E = math.sqrt(b**2 + (a-c)**2)
                    F = b*d*e - a*(e**2) - c*(d**2)
                    q = 64 * ((f*det + F) / (det**2))
                    s = 0.25 * math.sqrt(abs(q)*E)
                    A = 0.125 * math.sqrt(2*abs(q)*E - 2*q*(a+c))
                    B = math.sqrt((A**2) - (s**2))
                
                    x0 = (b*e - 2*c*d) / det
                    y0 = (b*d - 2*a*e) / det

                    Q1 = q*a - q*c
                    Q2 = q*b
                    tanbac = math.atan2(b, a-c)
                    if Q1 == 0 and Q2 == 0:
                        theta = 0
                    elif Q1 == 0 and Q2 > 0:
                        theta = np.pi/4
                    elif Q1 == 0 and Q2 < 0:
                        theta =  3 * np.pi/4
                    elif Q1 > 0 and Q2 >= 0:
                        theta = 0.5 * tanbac
                    elif Q1 > 0 and Q2 < 0:
                        theta = 0.5 * tanbac + np.pi
                    elif Q1 < 0:
                        theta = 0.5 * (tanbac + np.pi) 

                    self.ellipse_list.append(params((x0, y0), theta, A, B))

                except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                    print(str(e) + " in calculate_ellipse_params() method")

    def visualize_ellipse(self, image : np.ndarray) -> None:
        """
        Viualize the ellipse on the image and also plot the points used to
        fit the ellipse
        """
        for point in zip(self.data_x, self.data_y):
            cv2.circle(image, point, 3, (255,0,0), -1)

        for ellipse in self.ellipse_list:
            center = (int(ellipse.center[0]), int(ellipse.center[1]))
            cv2.ellipse(image, center, (int(ellipse.A), int(ellipse.B)), int(math.degrees(ellipse.theta)), 0, 360, (0,255,0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)


def main():
    image = cv2.resize(cv2.imread("substation_images/ferguson_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/proflo_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/mitsubishi_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/trafag_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)

    points = np.loadtxt("custom_ellipses/ferguson_pressure_gauge.txt", dtype="int")
    #points = np.loadtxt("custom_ellipses/proflo_pressure_gauge.txt", dtype="int")
    #points = np.loadtxt("custom_ellipses/spot_ptz_breaker_gauge.txt", dtype="int")
    #points = np.loadtxt("custom_ellipses/mitsubishi_pressure_gauge.txt", dtype="int")

    start = time.time()
    ell = Direct_least_sq_ellipse(np.array(points))
    ell.fit_ellipse(max(len(points), 20))
    ell.calculate_ellipse_params()
    print("Time taken = {:4.4f}s".format(time.time() - start))
    ell.visualize_ellipse(image)

if __name__ == "__main__":
    main()