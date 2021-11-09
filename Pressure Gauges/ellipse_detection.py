import time
import cv2
import numpy as np
import pandas as pd

from helper import *
from scipy.linalg import eig
from dataclasses import dataclass
from sklearn.cluster import OPTICS
from skimage.morphology import skeletonize

@dataclass
class params:
    center : tuple
    theta : float
    A : float
    B : float

class Ellipse_dlsq:
    def __init__(self) -> None:
        self.param_cluster = list()

    def __del__(self) -> None:
        pass

    def generalized_eigensystem(self, x : np.ndarray, y : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Formulates the generalized eigensystem problem and solves it by decomposing the system to its eigen values 
        and corresponding eigenvectors. The formulation is done using the design matrix which is in turn constructed
        using the input coordinate arrays (x,y)
        """
        # Design Matrix
        D = np.array([x*x, x*y, y*y, x, y, np.ones((x.shape), dtype="int")], dtype="float64").T
        # Scatter Matrix
        S = D.T @ D 
        # Constraint Matrix
        C = np.zeros(S.shape, dtype="float64")
        C[0,2] = C[2,0] = -2; C[1,1] = 1
        # Generalized eigensystem
        eigvals, eigvecs = eig(S, C)
        eigvals, eigvecs = np.real(eigvals), np.real(eigvecs)
        return eigvals, eigvecs

    def set_data(self, points : np.ndarray) -> None:
        """
        Sets the points list and separates the x, y coordinates into separate respective arrays. This is to
        solve the generalized eigensystem
        """
        self.data_x = np.array(points[:,0])
        self.data_y = np.array(points[:,1])
        self.len = len(self.data_x)
        return;

    def filter_points(self, x : np.ndarray, y : np.ndarray, idx : int, eigvecs : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Takes a detected ellipse, calculates the outliers of the ellipse and removes them. This then
        gives us a cleaner set of points to estimate the ellipse better on the second attempt
        """
        ellipse = self.calculate_ellipse_params(eigvecs[:,idx].flatten())
        dist_dict = dict()
        for (xi,yi) in zip(x,y):
            h,k = ellipse.center
            K1 = (xi - h)**2 / ellipse.A ** 2
            K2 = (yi - k)**2 / ellipse.B ** 2
            dist = K1 + K2 - 1
            dist_dict[dist] = (xi,yi)

        keys = list(sorted(dist_dict))
        keys = keys[0:int(0.8*len(keys))]
        x_new, y_new = list(), list()
        for key in keys:
            (xi,yi) = dist_dict[key]
            x_new.append(xi)
            y_new.append(yi)
        
        return np.array(x_new), np.array(y_new)

    def fit_ellipse(self, points_per_sample : int = 6, n_iter : int = 50) -> list:
        """
        Fits ellipses on batches of points for a number of iterations, and refines the ellipse fitting based on 
        the distances of the chosen points from the corresponding ellipse fit. This refinement is only to avoid
        any major effects of outlier points (point/s which don't lie on the same ellipse)
        """
        n = 0
        cluster = list()
        while n < n_iter:
            x, y = list(), list()
            idx = np.random.permutation(self.len)
            for i in range(points_per_sample):
                x.append(self.data_x[idx[i]])
                y.append(self.data_y[idx[i]])
            
            try:
                x, y = np.array(x), np.array(y)
                eigvals, eigvecs = self.generalized_eigensystem(x, y)
                # Since the system is +ve semi definite, only 1 negative eigenvalue
                idx = np.where(eigvals < 0)[0]
                # Filter the points and estimate the ellipse again to get a better estimate
                x_new, y_new = self.filter_points(x, y, idx, eigvecs)
                eigvals, eigvecs = self.generalized_eigensystem(x_new, y_new)
                idx = np.where(eigvals < 0)[0]
                cluster.append(eigvecs[:,idx])

            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                print(str(e) + " in the fit_ellipse() method")

            n += 1

        self.param_cluster = np.array(cluster).reshape(-1,6)
    
    def cluster_eigenvecs(self) -> np.ndarray:
        """
        Implements the OPTICS (Ordering Points To Identify Cluster Structure) clustering
        to cluster the 6D eigen vector space. The centroid of the largest cluster is taken
        as the ellipse to segregate the numbers.
        """
        clusters = OPTICS(min_samples=10).fit_predict(self.param_cluster)
        cluster_data = pd.DataFrame(list(zip(self.param_cluster, clusters)), columns=['Params', 'Cluster_id'])
        max_cluster = cluster_data['Cluster_id'].value_counts()[:5].to_dict()
        for k,_ in max_cluster.items():
            if k != -1:
                mode = k
        
        df = cluster_data.loc[cluster_data['Cluster_id'] == mode]
        param_list = np.array(df['Params'].to_list()).reshape(-1,6)
        largest_cluster = list()
        for param in param_list:
            a, b, c, _, _, _ = param
            if b**2 - 4*a*c < 0:
                largest_cluster.append(param)

        return np.array(largest_cluster).reshape(-1,6)

    def calculate_ellipse_params(self, eigenvector : np.ndarray) -> params:
        """
        Recognizes the ellipse from a list of general conic section equations 
        and extracts computes the 5 required parameters namely center_x, center_y,
        semi major axis, semi minor axis and orientation
        """
        a,b,c,d,e,f = eigenvector[0], eigenvector[1], eigenvector[2], eigenvector[3], eigenvector[4], eigenvector[5]
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

                return params((x0, y0), theta, A, B)

            except (ValueError, IndexError, np.linalg.LinAlgError) as e:
                print(str(e) + " in calculate_ellipse_params() method") 
                return None

        else:
            print("WARNING :- Conic section is not an ellipse")
            return None       

    def visualize_ellipse(self, image : np.ndarray, ellipse : params) -> None:
        """
        For visualization purposes only
        """
        if ellipse is not None:
            center = (int(ellipse.center[0]), int(ellipse.center[1]))
            axes = (int(ellipse.A), int(ellipse.B))
            angle = int(math.degrees(ellipse.theta))
            cv2.ellipse(image, center, axes, angle, 0, 360, (0,255,0), 2)

        cv2.imshow("image", image)
        cv2.waitKey(0)

def main():
    image = cv2.resize(cv2.imread("substation_images/ferguson_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/proflo_pressure_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/spot_ptz_breaker_gauge.png"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/mitsubishi_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/negative_pressure_gauge.jpg"), (800,800), cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("substation_images/trafag_pressure_gauge.jpg"),(800,800),cv2.INTER_CUBIC)

    #image = cv2.resize(cv2.imread("mounted pressure gauges/img-2.jpeg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("mounted pressure gauges/img-8.jpeg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("mounted pressure gauges/img-11.jpeg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("mounted pressure gauges/img-16.jpeg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("mounted pressure gauges/img-23.jpeg"),(800,800),cv2.INTER_CUBIC)
    #image = cv2.resize(cv2.imread("mounted pressure gauges/img-25.jpeg"),(800,800),cv2.INTER_CUBIC)

    ell = Ellipse_dlsq()
    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.std() > 70 or gray.std() < 35:
        gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5,5), 5)
    if calculate_brightness(image) > 0.52:
        hat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (24,24)))
    else:
        hat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (24,24)))
    thresh = cv2.threshold(hat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    thresh_float = np.array(thresh, dtype="float64") / 255.
    edge_map = np.array(skeletonize(thresh_float)*255., dtype="uint8")
    pixels = np.argwhere(edge_map != 0)

    ell.set_data(pixels)
    ell.fit_ellipse(points_per_sample=1500, n_iter=100)
    eigenvector = ell.cluster_eigenvecs().mean(axis=0)
    ellipse = ell.calculate_ellipse_params(eigenvector.flatten())
    print("Time taken = {:4.4f}s".format(time.time() - start))
    ell.visualize_ellipse(image, ellipse)

if __name__ == "__main__":
    main()
