# Analog Gauge Reading

## OCR analysis
https://varunharitsa.atlassian.net/wiki/spaces/OR/overview

## Working algorithm

#### Numerical gauges with tick marks
1) OCR
   <br /> a. Pre processing (Noise removal + Contrast enhancement + morphological transform)
   <br /> b. Initial dictionary with recognized numbers and location
   <br /> c. Filter the recognized numbers based on a common scale between the numbers
   
2) Needle estimation
   <br /> a. Find equation of the line approximating the needle and the 2 end points of the line segment of the needle
   <br /> b. Compute the pivot of the needle ==> Can be considered as center of the gauge
   <br /> c. Recognize negative numbers (if any) using positions of these numbers and update the OCR lookup dictionary
   <br /> d. Find out the swing of the needle (clockwise / anticlockwise)

3) Tick mark extraction
   <br /> a. Compute the different connected regions in the image, retain only rectangular regions (based on area and aspect ratio)
   <br /> b. Pair the numbers with corresponding major tick marks again with the distance and area criteria
   <br /> c. With these matched up tick marks, fit a quadratic polynomial through the ticks in the vicinity of the needle tip

4) Final value computation
   <br /> a. Find point of intersection between the polynomial and the line estimating the needle
   <br /> b. Interpolate the distance of the point of intersection to the nearest tick mark along the curve
   <br /> c. Compute the gauge value based on the direction of interpolation
