# Analog Gauge Reading

### New algorithm

1) Image acquisition and pre-processing 
   <br /> a. Ideally need some classification to recognize the type of gauge (analog/digital, tick marks present(y/n), etc.)
   <br /> b. Right now assume it is a numbered analog gauge with tick marks 
   
2) OCR and tick mark extraction
   <br /> a. Run OCR, get bounding box around numbers (and other calibration text --> Varies in every gauge) on the gauge
   <br /> b. Identify certain points relative to the bounding boxes to fit a quadratic polynomial (base of the tick mark region)
   <br /> c. Define a heuristic function to choose another polynomial, both polynomials together encompass the region of tick marks in the image, 
   <br /> d. Mask the original image to extract region with ticks

3) Needle isolation
   <br /> a. Run edge detection on the pre-processed image and run probabilistic hough line analysis
   <br /> b. It is safe to assume that the longest hough line in the image will be the needle (assuming that the gauge takes up the entire image)
 
4) Tick mark labeling and interpolation
   <br /> a. Pair up the tick marks with corresponding numbers (based on contour estimation)
   <br /> b. Fit a polynomial p(x) through the centroids of these contours (ticks and numbers are along similar curves), find the outliers using RANSAC
   <br /> c. Find the arc length along p(x) between the needle tip and nearest labeled tick mark and interpolate to find the reading 
   
### OCR analysis
https://varunharitsa.atlassian.net/wiki/spaces/OR/overview
