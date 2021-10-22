# Analog Gauge Reading

## OCR analysis
https://varunharitsa.atlassian.net/wiki/spaces/OR/overview

## Working algorithm

1) Temperature gauges usually have only one scale which can be inferred from the text on the backplate

2) Pressure gauges (more often than not) have 2 scales, one with KPa and one with PSI. 
   OCR will read numbers from both scales which is why segregating these numbers becomes
   a crucial step before diving into reading the gauge

### Temperature Gauges

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

#### Non numerical gauges (4 categories = MIN-MAX, HI-LO, LOW-HIGH, PUMP-ON-OFF)
1) OCR and Needle estimation
   <br /> a. Image pre processing ==> Denoising + Morphological operations
   <br /> b. Construct OCR lookup dictionary with numbers and corresponding locations (No filtering)
   <br /> c. Compute the equation of the needle line along with the endpoints of the detected needle
   <br /> d. Classify the gauge into one of the 4 above types and filter the OCR to retain only required information

2) Region props and tick mark extraction
   <br /> a. Based on the gauge type, we know the color of the tick marks ==> Color masking 
   <br /> b. Regions which abide by the area thresold and the aspect ratio considered are rectanges
   <br /> c. Pair the text in the OCR lookup with the identified tick marks reliably

3) Interpolate to the tip of the needle based on minimum and maximum tick mark location
   <br /> a. If the signed differences of the tip with the min and max tick marks are different, the
   needle lies in between  
   <br /> b. Calculate the metric distance from min tick mark / distance from max tick mark to define 
   a heuristic to determine (normal / abnormal) conditions


### Pressure Gauges
1) OCR and Needle estimation
   <br /> a. Image pre processing ==> Denoising + Morphological operations
   <br /> b. Construct an initial OCR lookup dictionary with numbers and corresponding locations 

2) Ellipse detection and clustering --> Step in progress

3) Segregating numbers on the basis of scale (In most cases KPa v/s PSI) --> To be done

4) Read one of the scales, convert to the other --> To be done
