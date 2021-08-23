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

### Updates 

1) Image acquisition and pre-processing 
   <br /> a. Ideally need some classification to recognize the type of gauge (analog/digital, tick marks present(y/n), etc.)
   <br /> b. Right now assume it is a numbered analog gauge with tick marks
   
2) OCR
   <br /> a. Get bounding boxes around numbers (and other calibration text --> Varies in every gauge) on the gauge
   <br /> b. Filter out wrongly detected numbers and non number text and build a lookup dictionary with position and confidence of the predictions

3) Region props
   <br /> a. Extract various connected regions in the image
   <br /> b. Filter out using 2 thresholds, area thresholding to remove small and very large regions, aspect ratio to get only rectangular regions
   <br /> c. Pair the tick mark with the OCR number based on euclidean distance criterion

4) Calculate the gauge value
   <br /> a. Fit a quadratic curve along the centroids of these major tick marks (ones paired with numbers)
   <br /> b. Find the point of interesection of line estimating the needle with this curve
   <br /> c. Calculate arc_length along the curve between needle and nearest major tick mark
   <br /> d. We can compute the reading based on prior calibration of distance v/s value along this polynomial (interpolation)
